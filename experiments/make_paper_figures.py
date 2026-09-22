import argparse
import shutil
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ACTIVE_REVIEWS = {
    "1k": Path("experiments/active_gsp_v3_residual_lowlr_1k_review_20260703_062017"),
    "5k": Path("experiments/active_gsp_v3_residual_lowlr_5k_review_20260703_074629"),
    "20k": Path("experiments/active_gsp_v3_residual_lowlr_20k_review_20260703_225610"),
}
PASSIVE_REVIEW = Path("experiments/formal100k_passive_combined_review_20260703_190622")
FASTPATH_PROFILE = Path("experiments/active_gsp_v3_profile_fastpath_20260703_020538")

METHOD_LABELS = {
    "raw_mlp": "MADDPG",
    "action_score_control": "Action scorer",
    "action_raw_potential_residual005": "Raw action prior",
    "v3_active_gsp_residual005": "Active GSP v3",
    "v1_full": "Passive GSP v1",
    "v2a_spectrum": "Passive spectrum",
    "none": "Raw rollout",
    "reference": "Reference v3",
    "fast": "Fast v3",
}

METHOD_COLORS = {
    "raw_mlp": "#4C78A8",
    "action_score_control": "#B279A2",
    "action_raw_potential_residual005": "#F58518",
    "v3_active_gsp_residual005": "#54A24B",
    "v1_full": "#72B7B2",
    "v2a_spectrum": "#E45756",
    "none": "#4C78A8",
    "reference": "#E45756",
    "fast": "#54A24B",
}

METRICS = [
    ("return", "Return", True),
    ("final_coverage", "Final coverage", True),
    ("max_coverage", "Max coverage", True),
    ("nearest_landmark_distance", "Nearest distance", False),
    ("collisions", "Collisions", False),
]


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


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def latest_existing_dir(pattern):
    matches = [path for path in Path("experiments").glob(pattern) if path.is_dir()]
    if not matches:
        return None
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def save_figure(fig, output_dir, stem):
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def load_summary(review_dir):
    return pd.read_csv(review_dir / "summary.csv")


def load_combined(review_dir):
    path = review_dir / "combined_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def method_label(name):
    return METHOD_LABELS.get(name, name)


def method_color(name):
    return METHOD_COLORS.get(name, "#9D755D")


def ordered_rows(frame, order):
    rank = {name: idx for idx, name in enumerate(order)}
    rows = frame[frame["experiment"].isin(order)].copy()
    rows["_rank"] = rows["experiment"].map(rank)
    return rows.sort_values("_rank").drop(columns=["_rank"])


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


def plot_active_20k(output_dir):
    return plot_active_review_metrics(
        output_dir=output_dir,
        review_dir=ACTIVE_REVIEWS["20k"],
        title="Active GSP v3 20k screening",
        stem="fig_active_gsp_v3_20k_metrics_prelim",
    )


def plot_active_review_metrics(output_dir, review_dir, title, stem):
    summary = load_summary(ACTIVE_REVIEWS["20k"])
    if review_dir != ACTIVE_REVIEWS["20k"]:
        summary = load_summary(review_dir)
    order = [
        "raw_mlp",
        "action_raw_potential_residual005",
        "v3_active_gsp_residual005",
    ]
    rows = ordered_rows(summary, order)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    axes = axes.reshape(-1)
    for axis, (metric, title, higher_better) in zip(axes, METRICS):
        x = np.arange(len(rows))
        means = rows[f"{metric}_mean"].to_numpy(dtype=float)
        stds = rows[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
        colors = [method_color(name) for name in rows["experiment"]]
        bars = axis.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.4)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels([method_label(name) for name in rows["experiment"]], rotation=25, ha="right")
        if metric == "nearest_landmark_distance":
            axis.set_ylabel("lower is better")
        if metric == "return":
            axis.axhline(0, color="#666666", linewidth=0.6)
        add_bar_labels(axis, bars, "{:.2f}")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.7)
    axes[-1].axis("off")
    fig.suptitle(title, y=1.02, fontsize=11)
    fig.tight_layout()
    return save_figure(fig, output_dir, stem)


def paired_delta(frame, left, right, metric):
    left_rows = frame[frame["experiment"] == left].set_index("seed")
    right_rows = frame[frame["experiment"] == right].set_index("seed")
    seeds = sorted(set(left_rows.index).intersection(right_rows.index))
    return np.asarray([left_rows.loc[seed, metric] - right_rows.loc[seed, metric] for seed in seeds], dtype=float)


def plot_active_20k_paired_deltas(output_dir):
    return plot_active_review_paired_deltas(
        output_dir=output_dir,
        review_dir=ACTIVE_REVIEWS["20k"],
        title="20k paired deltas across matched seeds",
        stem="fig_active_gsp_v3_20k_paired_deltas_prelim",
    )


def plot_active_review_paired_deltas(output_dir, review_dir, title, stem):
    rows = load_combined(review_dir)
    comparisons = [
        ("v3_active_gsp_residual005", "raw_mlp", "Active GSP v3 - MADDPG"),
        (
            "v3_active_gsp_residual005",
            "action_raw_potential_residual005",
            "Active GSP v3 - raw action prior",
        ),
    ]
    delta_metrics = [
        ("return", "Return", True),
        ("final_coverage", "Final cov.", True),
        ("max_coverage", "Max cov.", True),
        ("nearest_landmark_distance", "Distance", False),
        ("collisions", "Collisions", False),
    ]
    x = np.arange(len(delta_metrics))
    width = 0.34
    fig, axis = plt.subplots(figsize=(7.2, 3.2))
    for idx, (left, right, label) in enumerate(comparisons):
        means = [paired_delta(rows, left, right, metric).mean() for metric, _, _ in delta_metrics]
        stds = [paired_delta(rows, left, right, metric).std(ddof=0) for metric, _, _ in delta_metrics]
        offsets = x + (idx - 0.5) * width
        bars = axis.bar(offsets, means, width, yerr=stds, capsize=3, label=label)
        add_bar_labels(axis, bars, "{:.2f}", pad=3)
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_xticks(x)
    axis.set_xticklabels([label for _, label, _ in delta_metrics])
    axis.set_ylabel("Paired delta")
    axis.set_title(title)
    axis.legend(loc="best", frameon=False)
    axis.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    return save_figure(fig, output_dir, stem)


def plot_learning_funnel(output_dir, formal_review=None):
    stages = ["1k", "5k", "20k"]
    review_by_stage = dict(ACTIVE_REVIEWS)
    if formal_review is not None:
        stages.append("100k")
        review_by_stage["100k"] = formal_review
    metrics = [
        ("return", "Return delta"),
        ("final_coverage", "Final coverage delta"),
        ("max_coverage", "Max coverage delta"),
        ("nearest_landmark_distance", "Distance delta"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4), sharex=True)
    for axis, (metric, title) in zip(axes, metrics):
        values = []
        errors = []
        for stage in stages:
            rows = load_combined(review_by_stage[stage])
            deltas = paired_delta(rows, "v3_active_gsp_residual005", "raw_mlp", metric)
            values.append(float(deltas.mean()))
            errors.append(float(deltas.std(ddof=0)))
        x = np.arange(len(stages))
        bars = axis.bar(x, values, yerr=errors, color="#54A24B", capsize=3, edgecolor="black", linewidth=0.4)
        axis.axhline(0, color="#333333", linewidth=0.7)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(stages)
        axis.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.7)
        add_bar_labels(axis, bars, "{:.2f}", pad=2)
    axes[0].set_ylabel("Active GSP v3 - MADDPG")
    fig.suptitle("Screening funnel paired gains", y=1.08, fontsize=11)
    fig.tight_layout()
    suffix = "final" if formal_review is not None else "prelim"
    return save_figure(fig, output_dir, f"fig_active_gsp_v3_screening_funnel_{suffix}")


def plot_passive_ablation(output_dir):
    summary = load_summary(PASSIVE_REVIEW)
    order = ["raw_mlp", "v1_full", "v2a_spectrum"]
    rows = ordered_rows(summary, order)
    metrics = [
        ("return", "Return"),
        ("final_coverage", "Final coverage"),
        ("max_coverage", "Max coverage"),
        ("nearest_landmark_distance", "Nearest distance"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.6))
    for axis, (metric, title) in zip(axes, metrics):
        x = np.arange(len(rows))
        means = rows[f"{metric}_mean"].to_numpy(dtype=float)
        stds = rows[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
        colors = [method_color(name) for name in rows["experiment"]]
        bars = axis.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="black", linewidth=0.4)
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels([method_label(name) for name in rows["experiment"]], rotation=25, ha="right")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.7)
        if metric == "return":
            axis.axhline(0, color="#666666", linewidth=0.6)
        add_bar_labels(axis, bars, "{:.2f}", pad=2)
    fig.suptitle("Passive GSP 100k ablation", y=1.08, fontsize=11)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig_passive_gsp_100k_ablation_prelim")


def plot_fastpath_profile(output_dir):
    profile = pd.read_csv(FASTPATH_PROFILE / "profile.csv")
    latency = pd.read_csv(FASTPATH_PROFILE / "standalone_latency.csv").iloc[0]
    order = ["none", "reference", "fast"]
    rows = profile.set_index("feature_mode").loc[order].reset_index()
    x = np.arange(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    colors = [method_color(name) for name in rows["feature_mode"]]
    bars = axes[0].bar(x, rows["env_steps_per_sec"], color=colors, edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([method_label(name) for name in rows["feature_mode"]], rotation=20, ha="right")
    axes[0].set_ylabel("Env steps/sec")
    axes[0].set_title("Rollout throughput")
    add_bar_labels(axes[0], bars, "{:.0f}", pad=2)
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.7)

    latency_names = ["Reference", "Fast"]
    latency_values = [latency["reference_ms_per_obs"], latency["fast_ms_per_obs"]]
    bars = axes[1].bar(
        np.arange(2),
        latency_values,
        color=["#E45756", "#54A24B"],
        edgecolor="black",
        linewidth=0.4,
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks(np.arange(2))
    axes[1].set_xticklabels(latency_names)
    axes[1].set_ylabel("ms/local obs, log scale")
    axes[1].set_title(f"Feature latency, {latency['standalone_speedup']:.0f}x speedup")
    add_bar_labels(axes[1], bars, "{:.4f}", pad=2)
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.6, alpha=0.7)
    fig.suptitle("Active GSP v3 fast path", y=1.06, fontsize=11)
    fig.tight_layout()
    return save_figure(fig, output_dir, "fig_active_gsp_v3_fastpath_profile")


def copy_tables(output_dir, formal_review=None):
    table_dir = ensure_dir(output_dir / "tables")
    sources = [
        ACTIVE_REVIEWS["20k"] / "summary.csv",
        ACTIVE_REVIEWS["20k"] / "paired_delta_means_vs_baseline.csv",
        ACTIVE_REVIEWS["20k"] / "primary_vs_control_delta_means.csv",
        PASSIVE_REVIEW / "summary.csv",
        FASTPATH_PROFILE / "profile.csv",
        FASTPATH_PROFILE / "standalone_latency.csv",
    ]
    if formal_review is not None:
        sources.extend(
            [
                formal_review / "summary.csv",
                formal_review / "paired_delta_means_vs_baseline.csv",
                formal_review / "primary_vs_control_delta_means.csv",
            ]
        )
    copied = []
    for source in sources:
        if source.exists():
            target = table_dir / source.name
            if target.exists():
                target = table_dir / f"{source.parent.name}_{source.name}"
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def write_readme(output_dir, figure_paths, copied_tables, formal_review=None):
    bundle_kind = "final" if formal_review is not None else "preliminary"
    lines = [
        "# Paper Figure Bundle",
        "",
        f"Created: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        f"This {bundle_kind} bundle uses completed active-GSP screens, passive 100k ablations, and the fast-path profiling result.",
        "",
        "## Figures",
        "",
    ]
    for path in figure_paths:
        lines.append(f"- `{path.name}`")
    if copied_tables:
        lines.extend(["", "## Copied Tables", ""])
        for path in copied_tables:
            lines.append(f"- `tables/{path.name}`")
    lines.extend(
        [
            "",
            "## Source Reviews",
            "",
            f"- active 1k: `{ACTIVE_REVIEWS['1k']}`",
            f"- active 5k: `{ACTIVE_REVIEWS['5k']}`",
            f"- active 20k: `{ACTIVE_REVIEWS['20k']}`",
            f"- passive 100k: `{PASSIVE_REVIEW}`",
            f"- fast path: `{FASTPATH_PROFILE}`",
        ]
    )
    if formal_review is not None:
        lines.append(f"- active formal 100k: `{formal_review}`")
    else:
        lines.append("")
        lines.append("Regenerate with `--active-formal-review <review_dir>` after the focused active-GSP 100k formal run finishes.")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--active-formal-review", default=None)
    parser.add_argument(
        "--auto-latest-formal",
        action="store_true",
        help="Include the newest active-GSP formal 100k review if one exists.",
    )
    return parser.parse_args()


def main():
    configure_matplotlib()
    args = parse_args()
    formal_review = None
    if args.active_formal_review:
        formal_review = Path(args.active_formal_review)
    elif args.auto_latest_formal:
        formal_review = latest_existing_dir("active_gsp_v3_residual_lowlr_100k_formal_review_*")
    if formal_review is not None and not (formal_review / "summary.csv").exists():
        raise FileNotFoundError(f"formal review summary not found: {formal_review / 'summary.csv'}")
    if args.output_dir:
        output_dir = ensure_dir(Path(args.output_dir))
    else:
        kind = "final" if formal_review is not None else "prelim"
        output_dir = ensure_dir(Path("experiments") / f"paper_figures_{kind}_{time.strftime('%Y%m%d_%H%M%S')}")
    figures = []
    figures.extend(plot_active_20k(output_dir))
    figures.extend(plot_active_20k_paired_deltas(output_dir))
    if formal_review is not None:
        figures.extend(
            plot_active_review_metrics(
                output_dir=output_dir,
                review_dir=formal_review,
                title="Active GSP v3 100k formal validation",
                stem="fig_active_gsp_v3_100k_formal_metrics",
            )
        )
        figures.extend(
            plot_active_review_paired_deltas(
                output_dir=output_dir,
                review_dir=formal_review,
                title="100k formal paired deltas across matched seeds",
                stem="fig_active_gsp_v3_100k_formal_paired_deltas",
            )
        )
    figures.extend(plot_learning_funnel(output_dir, formal_review=formal_review))
    figures.extend(plot_passive_ablation(output_dir))
    figures.extend(plot_fastpath_profile(output_dir))
    tables = copy_tables(output_dir, formal_review=formal_review)
    write_readme(output_dir, figures, tables, formal_review=formal_review)
    print(output_dir)


if __name__ == "__main__":
    main()
