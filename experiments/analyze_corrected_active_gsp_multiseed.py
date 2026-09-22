import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    "raw_mlp",
    "learned_raw_potential_residual",
    "learned_active_gsp_residual",
]
LABELS = {
    "raw_mlp": "Raw MADDPG",
    "learned_raw_potential_residual": "Action-aware control",
    "learned_active_gsp_residual": "Active GSP",
}
METRICS = [
    "return",
    "final_coverage",
    "max_coverage",
    "nearest_landmark_distance",
    "collisions",
    "collision_step_rate",
]
CHECKPOINTS = [20000, 40000, 60000, 80000, 100000]
T_CRITICAL_95_DF2 = 4.3026527297


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std_ci(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    half = T_CRITICAL_95_DF2 * std / math.sqrt(values.size)
    return mean, std, mean - half, mean + half


def source_map(args):
    seed1_active = Path(args.seed1_active_root)
    seed1_raw = Path(args.seed1_raw_root)
    seeds23 = Path(args.seeds23_root)
    output = {
        1: {
            "raw_mlp": seed1_raw / "raw_mlp" / "seed_1",
            "learned_raw_potential_residual": (
                seed1_active / "learned_raw_potential_residual" / "seed_1"
            ),
            "learned_active_gsp_residual": (
                seed1_active / "learned_active_gsp_residual" / "seed_1"
            ),
        }
    }
    for seed in [2, 3]:
        output[seed] = {
            method: seeds23 / method / f"seed_{seed}" for method in METHODS
        }
    return output


def load_all(sources):
    summaries = []
    curves = []
    episodes = []
    for seed, methods in sources.items():
        for method, run_dir in methods.items():
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            summary["seed"] = seed
            summary["method"] = method
            summary["source_run_dir"] = str(run_dir)
            summaries.append(summary)

            for row in read_csv(run_dir / "learning_curve_eval.csv"):
                row["seed"] = seed
                row["method"] = method
                row["env_steps"] = int(str(row["checkpoint"]).replace("model_step", ""))
                curves.append(row)

            for row in read_csv(run_dir / "per_evaluation_episode_metrics.csv"):
                row["seed"] = seed
                row["method"] = method
                episodes.append(row)
    return summaries, curves, episodes


def aggregate_summaries(summaries):
    rows = []
    for method in METHODS:
        method_rows = [row for row in summaries if row["method"] == method]
        output = {"method": method, "label": LABELS[method], "n_seeds": len(method_rows)}
        for metric in METRICS:
            mean, std, low, high = mean_std_ci([float(row[metric]) for row in method_rows])
            output[f"{metric}_mean"] = mean
            output[f"{metric}_std"] = std
            output[f"{metric}_ci95_low"] = low
            output[f"{metric}_ci95_high"] = high
        rows.append(output)
    return rows


def paired_seed_deltas(summaries):
    lookup = {
        (int(row["seed"]), row["method"]): row for row in summaries
    }
    comparisons = [
        ("active_vs_raw", "learned_active_gsp_residual", "raw_mlp"),
        (
            "active_vs_action_control",
            "learned_active_gsp_residual",
            "learned_raw_potential_residual",
        ),
        (
            "action_control_vs_raw",
            "learned_raw_potential_residual",
            "raw_mlp",
        ),
    ]
    detail = []
    summary = []
    for comparison, left, right in comparisons:
        for metric in METRICS:
            deltas = []
            for seed in [1, 2, 3]:
                delta = float(lookup[(seed, left)][metric]) - float(
                    lookup[(seed, right)][metric]
                )
                deltas.append(delta)
                detail.append(
                    {
                        "comparison": comparison,
                        "left": left,
                        "right": right,
                        "metric": metric,
                        "seed": seed,
                        "delta": delta,
                    }
                )
            mean, std, low, high = mean_std_ci(deltas)
            summary.append(
                {
                    "comparison": comparison,
                    "left": left,
                    "right": right,
                    "metric": metric,
                    "n_seeds": 3,
                    "mean_delta": mean,
                    "std_delta": std,
                    "ci95_low": low,
                    "ci95_high": high,
                    "all_seed_deltas": ";".join(f"{value:.9f}" for value in deltas),
                }
            )
    return detail, summary


def aggregate_curves(curves):
    seed_rows = []
    aggregate_rows = []
    for method in METHODS:
        for step in CHECKPOINTS:
            matching = [
                row for row in curves
                if row["method"] == method and int(row["env_steps"]) == step
            ]
            for row in matching:
                seed_output = {
                    "method": method,
                    "seed": int(row["seed"]),
                    "env_steps": step,
                }
                for metric in METRICS:
                    seed_output[metric] = float(row[metric])
                seed_rows.append(seed_output)
            output = {"method": method, "label": LABELS[method], "env_steps": step}
            for metric in METRICS:
                values = [float(row[metric]) for row in matching]
                output[f"{metric}_mean"] = float(np.mean(values))
                output[f"{metric}_std"] = float(np.std(values, ddof=1))
            aggregate_rows.append(output)
    return seed_rows, aggregate_rows


def paired_episode_rows(episodes):
    lookup = {
        (int(row["seed"]), int(row["test_seed"]), row["method"]): row
        for row in episodes
    }
    rows = []
    for seed in [1, 2, 3]:
        test_seeds = sorted(
            {
                int(row["test_seed"])
                for row in episodes
                if int(row["seed"]) == seed
            }
        )
        for test_seed in test_seeds:
            for comparison, left, right in [
                ("active_vs_raw", "learned_active_gsp_residual", "raw_mlp"),
                (
                    "active_vs_action_control",
                    "learned_active_gsp_residual",
                    "learned_raw_potential_residual",
                ),
            ]:
                output = {
                    "comparison": comparison,
                    "train_seed": seed,
                    "test_seed": test_seed,
                }
                for metric in METRICS:
                    output[f"delta_{metric}"] = float(
                        lookup[(seed, test_seed, left)][metric]
                    ) - float(lookup[(seed, test_seed, right)][metric])
                rows.append(output)
    return rows


def plot_final(aggregate, output_dir):
    metrics = ["return", "final_coverage", "max_coverage", "nearest_landmark_distance"]
    titles = ["Episode return", "Final coverage", "Maximum coverage", "Final distance"]
    colors = ["#4C566A", "#D08770", "#5E81AC"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric, title in zip(axes.flat, metrics, titles):
        means = [row[f"{metric}_mean"] for row in aggregate]
        stds = [row[f"{metric}_std"] for row in aggregate]
        xs = np.arange(len(METHODS))
        ax.bar(xs, means, yerr=stds, capsize=4, color=colors, width=0.68)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=12, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_final_metrics.png", dpi=220)
    fig.savefig(output_dir / "fig_final_metrics.pdf")
    plt.close(fig)


def plot_curves(curves, output_dir):
    metrics = ["return", "final_coverage", "max_coverage", "nearest_landmark_distance"]
    titles = ["Episode return", "Final coverage", "Maximum coverage", "Final distance"]
    colors = {"raw_mlp": "#4C566A", "learned_raw_potential_residual": "#D08770",
              "learned_active_gsp_residual": "#5E81AC"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, metric, title in zip(axes.flat, metrics, titles):
        for method in METHODS:
            rows = sorted(
                [row for row in curves if row["method"] == method],
                key=lambda row: int(row["env_steps"]),
            )
            x = np.asarray([int(row["env_steps"]) for row in rows])
            mean = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
            std = np.asarray([float(row[f"{metric}_std"]) for row in rows])
            ax.plot(x, mean, marker="o", label=LABELS[method], color=colors[method])
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=colors[method])
        ax.set_title(title)
        ax.set_xlabel("Environment steps")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_learning_curves.png", dpi=220)
    fig.savefig(output_dir / "fig_learning_curves.pdf")
    plt.close(fig)


def latex_table(aggregate, path):
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\hline",
        "Method & Return & Final cov. & Max cov. & Distance \\\\",
        "\\hline",
    ]
    for row in aggregate:
        lines.append(
            f"{row['label']} & "
            f"${row['return_mean']:.3f}\\pm{row['return_std']:.3f}$ & "
            f"${row['final_coverage_mean']:.3f}\\pm{row['final_coverage_std']:.3f}$ & "
            f"${row['max_coverage_mean']:.3f}\\pm{row['max_coverage_std']:.3f}$ & "
            f"${row['nearest_landmark_distance_mean']:.3f}\\pm"
            f"{row['nearest_landmark_distance_std']:.3f}$ \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(output_dir, aggregate, paired):
    agg = {row["method"]: row for row in aggregate}
    pair = {(row["comparison"], row["metric"]): row for row in paired}
    lines = [
        "# Corrected Active GSP Three-Seed Analysis",
        "",
        "All methods use exactly 100000 environment steps, actor-only local features,",
        "raw 18D replay observations, and the unchanged 69D centralized critic.",
        "",
        "## Final Mean +/- Training-Seed Standard Deviation",
        "",
        "| method | return | final coverage | max coverage | distance | collision-step rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = agg[method]
        lines.append(
            f"| {LABELS[method]} | {row['return_mean']:.3f} +/- {row['return_std']:.3f} | "
            f"{row['final_coverage_mean']:.3f} +/- {row['final_coverage_std']:.3f} | "
            f"{row['max_coverage_mean']:.3f} +/- {row['max_coverage_std']:.3f} | "
            f"{row['nearest_landmark_distance_mean']:.3f} +/- "
            f"{row['nearest_landmark_distance_std']:.3f} | "
            f"{row['collision_step_rate_mean']:.3f} +/- "
            f"{row['collision_step_rate_std']:.3f} |"
        )
    cov = pair[("active_vs_action_control", "final_coverage")]
    collision = pair[("active_vs_action_control", "collision_step_rate")]
    lines.extend(
        [
            "",
            "## Main Paired Result",
            "",
            f"Active GSP improves final coverage over the action-aware control by "
            f"{cov['mean_delta']:+.4f} (training-seed t interval "
            f"[{cov['ci95_low']:+.4f}, {cov['ci95_high']:+.4f}]).",
            "",
            f"The collision-step-rate tradeoff is {collision['mean_delta']:+.4f} "
            f"([{collision['ci95_low']:+.4f}, {collision['ci95_high']:+.4f}]).",
            "",
            "Interpretation: the spectral channels consistently improve strict final coverage,",
            "but they do not dominate the action-aware control on return and increase collisions.",
            "This is a coverage-specialization result, not uniform superiority.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed1-raw-root", required=True)
    parser.add_argument("--seed1-active-root", required=True)
    parser.add_argument("--seeds23-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = source_map(args)
    summaries, curves, episodes = load_all(sources)
    aggregate = aggregate_summaries(summaries)
    paired_detail, paired_summary = paired_seed_deltas(summaries)
    curve_seed, curve_aggregate = aggregate_curves(curves)
    episode_deltas = paired_episode_rows(episodes)

    write_csv(output_dir / "combined_seed_summaries.csv", summaries)
    write_csv(output_dir / "aggregate_summary.csv", aggregate)
    write_csv(output_dir / "paired_training_seed_deltas.csv", paired_detail)
    write_csv(output_dir / "paired_training_seed_summary.csv", paired_summary)
    write_csv(output_dir / "combined_learning_curve_seed.csv", curve_seed)
    write_csv(output_dir / "aggregate_learning_curve.csv", curve_aggregate)
    write_csv(output_dir / "paired_episode_deltas.csv", episode_deltas)
    (output_dir / "sources.json").write_text(
        json.dumps(
            {
                str(seed): {method: str(path) for method, path in methods.items()}
                for seed, methods in sources.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_final(aggregate, output_dir)
    plot_curves(curve_aggregate, output_dir)
    latex_table(aggregate, output_dir / "table_final_summary.tex")
    write_readme(output_dir, aggregate, paired_summary)
    print(output_dir)


if __name__ == "__main__":
    main()
