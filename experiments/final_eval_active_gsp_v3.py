import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.maddpg import MADDPG
from utils.make_env import make_env


METRICS = [
    "return",
    "final_coverage",
    "max_coverage",
    "nearest_landmark_distance",
    "collisions",
]
DEFAULT_EXPERIMENTS = [
    "raw_mlp",
    "action_raw_potential_residual005",
    "v3_active_gsp_residual005",
]
METHOD_LABELS = {
    "raw_mlp": "MADDPG",
    "action_raw_potential_residual005": "Raw action prior",
    "v3_active_gsp_residual005": "Active GSP v3",
}
METHOD_COLORS = {
    "raw_mlp": "#4C78A8",
    "action_raw_potential_residual005": "#F58518",
    "v3_active_gsp_residual005": "#54A24B",
}


def coverage_metrics(env, threshold=0.10):
    agents = env.world.agents
    landmarks = env.world.landmarks
    distances = np.asarray(
        [
            [
                np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
                for landmark in landmarks
            ]
            for agent in agents
        ],
        dtype=np.float32,
    )
    nearest = distances.min(axis=0)
    collisions = 0
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            dist = np.linalg.norm(agents[i].state.p_pos - agents[j].state.p_pos)
            if dist < agents[i].size + agents[j].size:
                collisions += 1
    return int((nearest < threshold).sum()), float(nearest.mean()), collisions


def read_source(path):
    path = Path(path)
    if (path / "combined_results.csv").exists():
        return pd.read_csv(path / "combined_results.csv")
    if (path / "results.csv").exists():
        return pd.read_csv(path / "results.csv")
    raise FileNotFoundError(f"missing combined_results.csv or results.csv in {path}")


def read_sources(paths):
    frames = []
    for path in paths:
        frame = read_source(path)
        frame.insert(0, "eval_source", str(path))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def stable_rows(results, experiments):
    rows = results[results["experiment"].isin(experiments)].copy()
    rows = rows.sort_values(["seed", "experiment"]).reset_index(drop=True)
    missing_models = []
    for _, row in rows.iterrows():
        model_path = Path(row["run_dir"]) / "model.pt"
        if not model_path.exists():
            missing_models.append(str(model_path))
    if missing_models:
        joined = "\n".join(missing_models)
        raise FileNotFoundError(f"missing model files:\n{joined}")
    return rows


def evaluate_one_model(row, episodes, episode_length, test_seed_base):
    model_path = Path(row["run_dir"]) / "model.pt"
    maddpg = MADDPG.init_from_save(str(model_path))
    maddpg.prep_rollouts(device="cpu")
    env = make_env(row["env_id"], discrete_action=maddpg.discrete_action)
    output_rows = []
    try:
        for eval_episode in range(episodes):
            test_seed = test_seed_base + eval_episode
            torch.manual_seed(test_seed)
            np.random.seed(test_seed)
            env.seed(test_seed)
            obs = env.reset()
            episode_return = 0.0
            max_coverage = 0
            final_coverage = 0
            final_distance = 0.0
            final_collisions = 0
            for _ in range(episode_length):
                torch_obs = [
                    torch.as_tensor(obs[i], dtype=torch.float32).view(1, -1)
                    for i in range(maddpg.nagents)
                ]
                with torch.no_grad():
                    torch_actions = maddpg.step(torch_obs, explore=False)
                actions = [
                    action.detach().cpu().numpy().flatten()
                    for action in torch_actions
                ]
                obs, rewards, dones, _ = env.step(actions)
                episode_return += float(np.mean(rewards))
                final_coverage, final_distance, final_collisions = coverage_metrics(env)
                max_coverage = max(max_coverage, final_coverage)
                if all(dones):
                    break
            output_rows.append(
                {
                    "experiment": row["experiment"],
                    "train_seed": int(row["seed"]),
                    "eval_episode": eval_episode,
                    "test_seed": test_seed,
                    "env_id": row["env_id"],
                    "actor_model": row["actor_model"],
                    "model_name": row["model_name"],
                    "run_dir": row["run_dir"],
                    "return": episode_return,
                    "final_coverage": final_coverage,
                    "max_coverage": max_coverage,
                    "nearest_landmark_distance": final_distance,
                    "collisions": final_collisions,
                }
            )
    finally:
        env.close()
    return output_rows


def summarize_by_train_seed(per_episode):
    grouped = per_episode.groupby(["experiment", "train_seed"])[METRICS]
    return grouped.agg(["count", "mean", "std"]).reset_index()


def flatten_columns(frame):
    frame = frame.copy()
    frame.columns = [
        "_".join([str(part) for part in column if part]).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in frame.columns.values
    ]
    return frame


def summarize_across_train_seeds(train_seed_summary):
    mean_cols = [f"{metric}_mean" for metric in METRICS]
    grouped = train_seed_summary.groupby("experiment")[mean_cols]
    summary = grouped.agg(["count", "mean", "std"]).reset_index()
    return flatten_columns(summary)


def bootstrap_ci(values, rng, n_bootstrap=5000, alpha=0.05):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(n_bootstrap, values.size), replace=True)
    means = samples.mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def paired_deltas(per_episode, baseline, primary, control):
    index_cols = ["train_seed", "eval_episode", "test_seed"]
    pivot = per_episode.pivot_table(
        index=index_cols,
        columns="experiment",
        values=METRICS,
        aggfunc="first",
    )
    rows = []
    comparisons = [(primary, baseline), (control, baseline), (primary, control)]
    for metric in METRICS:
        for left, right in comparisons:
            if (metric, left) not in pivot.columns or (metric, right) not in pivot.columns:
                continue
            delta = pivot[(metric, left)] - pivot[(metric, right)]
            for index, value in delta.dropna().items():
                train_seed, eval_episode, test_seed = index
                rows.append(
                    {
                        "metric": metric,
                        "left": left,
                        "right": right,
                        "train_seed": int(train_seed),
                        "eval_episode": int(eval_episode),
                        "test_seed": int(test_seed),
                        "delta": float(value),
                    }
                )
    return pd.DataFrame(rows)


def summarize_deltas(deltas, bootstrap_seed):
    if deltas.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for (metric, left, right), group in deltas.groupby(["metric", "left", "right"]):
        train_seed_means = group.groupby("train_seed")["delta"].mean().to_numpy()
        ci_low, ci_high = bootstrap_ci(train_seed_means, rng)
        rows.append(
            {
                "metric": metric,
                "left": left,
                "right": right,
                "n_train_seeds": int(train_seed_means.size),
                "mean_delta": float(train_seed_means.mean()),
                "std_across_train_seeds": float(train_seed_means.std(ddof=1))
                if train_seed_means.size > 1
                else 0.0,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def write_csv_incremental(rows, output_path):
    if not rows:
        return
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_readme(output_dir, args, per_episode, train_seed_summary, final_summary, delta_summary):
    lines = [
        "# Active GSP v3 Final Evaluation",
        "",
        f"Created: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "This evaluation uses fixed test seeds shared by all methods for paired comparison.",
        "",
        "## Configuration",
        "",
        f"- Episodes per model: `{args.eval_episodes}`",
        f"- Episode length: `{args.episode_length}`",
        f"- Test seed base: `{args.test_seed_base}`",
        f"- Baseline: `{args.baseline}`",
        f"- Primary: `{args.primary}`",
        f"- Control: `{args.control}`",
        "",
        "## Files",
        "",
        "- `per_episode_results.csv`",
        "- `train_seed_summary.csv`",
        "- `summary_across_train_seeds.csv`",
        "- `paired_deltas.csv`",
        "- `paired_delta_summary.csv`",
        "- `fig_final_eval_metrics.png` / `.pdf`",
        "- `fig_final_eval_paired_deltas.png` / `.pdf`",
        "",
        "## Summary Across Training Seeds",
        "",
    ]
    if not final_summary.empty:
        lines.append("| experiment | n seeds | return | final_cov | max_cov | distance | collisions |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, row in final_summary.iterrows():
            lines.append(
                f"| {row['experiment']} | {int(row['return_mean_count'])} | "
                f"{row['return_mean_mean']:.4f} +/- {row['return_mean_std']:.4f} | "
                f"{row['final_coverage_mean_mean']:.4f} +/- {row['final_coverage_mean_std']:.4f} | "
                f"{row['max_coverage_mean_mean']:.4f} +/- {row['max_coverage_mean_std']:.4f} | "
                f"{row['nearest_landmark_distance_mean_mean']:.4f} +/- {row['nearest_landmark_distance_mean_std']:.4f} | "
                f"{row['collisions_mean_mean']:.4f} +/- {row['collisions_mean_std']:.4f} |"
            )
    if not delta_summary.empty:
        lines.extend(["", "## Paired Delta Summary", ""])
        lines.append("| metric | left | right | mean delta | 95% bootstrap CI |")
        lines.append("| --- | --- | --- | ---: | ---: |")
        for _, row in delta_summary.iterrows():
            lines.append(
                f"| {row['metric']} | {row['left']} | {row['right']} | "
                f"{row['mean_delta']:.4f} | "
                f"[{row['bootstrap_ci_low']:.4f}, {row['bootstrap_ci_high']:.4f}] |"
            )
    lines.extend(["", "## Sources", ""])
    for source in args.sources:
        lines.append(f"- `{source}`")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def configure_matplotlib():
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def method_label(name):
    return METHOD_LABELS.get(name, name)


def method_color(name):
    return METHOD_COLORS.get(name, "#9D755D")


def save_figure(fig, output_dir, stem):
    fig.savefig(output_dir / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(axis, bars, fmt="{:.2f}", pad=2):
    for bar in bars:
        height = bar.get_height()
        if not np.isfinite(height):
            continue
        va = "bottom" if height >= 0 else "top"
        offset = pad if height >= 0 else -pad
        axis.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=7,
        )


def plot_final_metrics(output_dir, final_summary, experiments):
    if final_summary.empty:
        return
    order = [name for name in experiments if name in set(final_summary["experiment"])]
    rank = {name: idx for idx, name in enumerate(order)}
    rows = final_summary.copy()
    rows["_rank"] = rows["experiment"].map(rank)
    rows = rows.dropna(subset=["_rank"]).sort_values("_rank")
    metrics = [
        ("return_mean", "Return"),
        ("final_coverage_mean", "Final coverage"),
        ("max_coverage_mean", "Max coverage"),
        ("nearest_landmark_distance_mean", "Nearest distance"),
        ("collisions_mean", "Collisions"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    axes = axes.reshape(-1)
    x = np.arange(len(rows))
    labels = [method_label(name) for name in rows["experiment"]]
    colors = [method_color(name) for name in rows["experiment"]]
    for axis, (metric, title) in zip(axes, metrics):
        means = rows[f"{metric}_mean"].to_numpy(dtype=float)
        stds = rows[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
        bars = axis.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.4)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        if "distance" in metric:
            axis.set_ylabel("lower is better")
        if metric == "return_mean":
            axis.axhline(0, color="#666666", linewidth=0.6)
        axis.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.7)
        add_bar_labels(axis, bars, "{:.2f}")
    axes[-1].axis("off")
    fig.suptitle("Final paired evaluation", y=1.02, fontsize=11)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_final_eval_metrics")


def plot_delta_summary(output_dir, delta_summary):
    if delta_summary.empty:
        return
    keep = delta_summary[
        delta_summary["left"].eq("v3_active_gsp_residual005")
        & delta_summary["right"].isin(["raw_mlp", "action_raw_potential_residual005"])
    ].copy()
    if keep.empty:
        return
    metric_order = ["return", "final_coverage", "max_coverage", "nearest_landmark_distance", "collisions"]
    metric_labels = ["Return", "Final cov.", "Max cov.", "Distance", "Collisions"]
    comparison_order = [
        ("v3_active_gsp_residual005", "raw_mlp", "Active GSP v3 - MADDPG"),
        (
            "v3_active_gsp_residual005",
            "action_raw_potential_residual005",
            "Active GSP v3 - raw action prior",
        ),
    ]
    x = np.arange(len(metric_order))
    width = 0.34
    fig, axis = plt.subplots(figsize=(7.2, 3.2))
    for idx, (left, right, label) in enumerate(comparison_order):
        rows = keep[keep["left"].eq(left) & keep["right"].eq(right)].set_index("metric")
        means = [rows.loc[metric, "mean_delta"] if metric in rows.index else np.nan for metric in metric_order]
        low = [rows.loc[metric, "bootstrap_ci_low"] if metric in rows.index else np.nan for metric in metric_order]
        high = [rows.loc[metric, "bootstrap_ci_high"] if metric in rows.index else np.nan for metric in metric_order]
        lower_err = np.asarray(means) - np.asarray(low)
        upper_err = np.asarray(high) - np.asarray(means)
        bars = axis.bar(
            x + (idx - 0.5) * width,
            means,
            width,
            yerr=np.vstack([lower_err, upper_err]),
            capsize=3,
            label=label,
        )
        add_bar_labels(axis, bars, "{:.2f}", pad=3)
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels(metric_labels)
    axis.set_ylabel("Paired delta with 95% bootstrap CI")
    axis.set_title("Final paired deltas")
    axis.legend(loc="best", frameon=False)
    axis.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    save_figure(fig, output_dir, "fig_final_eval_paired_deltas")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS))
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--test-seed-base", type=int, default=880000)
    parser.add_argument("--baseline", default="raw_mlp")
    parser.add_argument("--primary", default="v3_active_gsp_residual005")
    parser.add_argument("--control", default="action_raw_potential_residual005")
    parser.add_argument("--bootstrap-seed", type=int, default=12345)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    configure_matplotlib()
    args = parse_args()
    experiments = [name.strip() for name in args.experiments.split(",") if name.strip()]
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("experiments") / f"active_gsp_v3_final_eval_{time.strftime('%Y%m%d_%H%M%S')}"
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"output directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = stable_rows(read_sources(args.sources), experiments)
    all_rows = []
    per_episode_path = output_dir / "per_episode_results.csv"
    for _, row in results.iterrows():
        print(
            f"Evaluating {row['experiment']} train_seed={row['seed']} "
            f"episodes={args.eval_episodes}",
            flush=True,
        )
        all_rows.extend(
            evaluate_one_model(
                row=row,
                episodes=args.eval_episodes,
                episode_length=args.episode_length,
                test_seed_base=args.test_seed_base,
            )
        )
        write_csv_incremental(all_rows, per_episode_path)

    per_episode = pd.DataFrame(all_rows)
    train_seed_summary = flatten_columns(summarize_by_train_seed(per_episode))
    final_summary = summarize_across_train_seeds(train_seed_summary)
    deltas = paired_deltas(per_episode, args.baseline, args.primary, args.control)
    delta_summary = summarize_deltas(deltas, args.bootstrap_seed)

    per_episode.to_csv(per_episode_path, index=False)
    train_seed_summary.to_csv(output_dir / "train_seed_summary.csv", index=False)
    final_summary.to_csv(output_dir / "summary_across_train_seeds.csv", index=False)
    deltas.to_csv(output_dir / "paired_deltas.csv", index=False)
    delta_summary.to_csv(output_dir / "paired_delta_summary.csv", index=False)
    write_readme(output_dir, args, per_episode, train_seed_summary, final_summary, delta_summary)
    plot_final_metrics(output_dir, final_summary, experiments)
    plot_delta_summary(output_dir, delta_summary)
    print(output_dir)


if __name__ == "__main__":
    main()
