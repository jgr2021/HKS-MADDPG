import argparse
import time
from pathlib import Path

import pandas as pd


METRICS = [
    "return",
    "final_coverage",
    "max_coverage",
    "nearest_landmark_distance",
    "collisions",
    "train_wall_time_sec",
]


def read_sources(source_dirs):
    frames = []
    for source_dir in source_dirs:
        source = Path(source_dir)
        csv_path = source / "results.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"missing results file: {csv_path}")
        frame = pd.read_csv(csv_path)
        frame.insert(0, "source_dir", str(source))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def summarize(results):
    summary = (
        results.groupby("experiment")[METRICS]
        .agg(["count", "mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join([part for part in column if part]).rstrip("_")
        for column in summary.columns.values
    ]
    summary = summary.rename(columns={"return_count": "n"})
    columns = ["experiment", "n"]
    for metric in METRICS:
        columns.extend([f"{metric}_mean", f"{metric}_std"])
    return summary[columns]


def paired_deltas(results, baseline):
    baseline_rows = results[results["experiment"] == baseline]
    if baseline_rows.empty:
        raise ValueError(f"baseline experiment not found: {baseline}")
    baseline_by_seed = baseline_rows.set_index("seed")

    rows = []
    for _, row in results[results["experiment"] != baseline].iterrows():
        seed = row["seed"]
        if seed not in baseline_by_seed.index:
            continue
        base = baseline_by_seed.loc[seed]
        rows.append(
            {
                "seed": int(seed),
                "experiment": row["experiment"],
                "delta_return_vs_baseline": row["return"] - base["return"],
                "delta_final_coverage_vs_baseline": row["final_coverage"]
                - base["final_coverage"],
                "delta_max_coverage_vs_baseline": row["max_coverage"]
                - base["max_coverage"],
                "delta_nearest_landmark_distance_vs_baseline": row[
                    "nearest_landmark_distance"
                ]
                - base["nearest_landmark_distance"],
                "delta_collisions_vs_baseline": row["collisions"]
                - base["collisions"],
            }
        )
    return pd.DataFrame(rows)


def pairwise_delta(results, left, right):
    left_rows = results[results["experiment"] == left].set_index("seed")
    right_rows = results[results["experiment"] == right].set_index("seed")
    seeds = sorted(set(left_rows.index).intersection(right_rows.index))
    rows = []
    for seed in seeds:
        left_row = left_rows.loc[seed]
        right_row = right_rows.loc[seed]
        rows.append(
            {
                "seed": int(seed),
                f"delta_return_{left}_vs_{right}": left_row["return"]
                - right_row["return"],
                f"delta_final_coverage_{left}_vs_{right}": left_row[
                    "final_coverage"
                ]
                - right_row["final_coverage"],
                f"delta_max_coverage_{left}_vs_{right}": left_row[
                    "max_coverage"
                ]
                - right_row["max_coverage"],
                f"delta_nearest_landmark_distance_{left}_vs_{right}": left_row[
                    "nearest_landmark_distance"
                ]
                - right_row["nearest_landmark_distance"],
                f"delta_collisions_{left}_vs_{right}": left_row["collisions"]
                - right_row["collisions"],
            }
        )
    return pd.DataFrame(rows)


def format_mean_std(summary, experiment, metric, digits=4):
    row = summary.set_index("experiment").loc[experiment]
    return f"{row[f'{metric}_mean']:.{digits}f} +/- {row[f'{metric}_std']:.{digits}f}"


def write_readme(
    output_dir,
    source_dirs,
    summary,
    deltas,
    delta_means,
    primary=None,
    control=None,
    primary_vs_control=None,
    primary_vs_control_means=None,
):
    summary_idx = summary.set_index("experiment")
    lines = [
        "# Active GSP v3 Results Review",
        "",
        "Sources:",
    ]
    for source in source_dirs:
        lines.append(f"- `{source}`")
    lines.extend(
        [
            "",
            "## Mean Results",
            "",
            "| experiment | n | return | final_cov | max_cov | distance | collisions | train_sec |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for experiment in summary["experiment"]:
        row = summary_idx.loc[experiment]
        lines.append(
            f"| {experiment} | {int(row['n'])} | "
            f"{format_mean_std(summary, experiment, 'return')} | "
            f"{format_mean_std(summary, experiment, 'final_coverage')} | "
            f"{format_mean_std(summary, experiment, 'max_coverage')} | "
            f"{format_mean_std(summary, experiment, 'nearest_landmark_distance')} | "
            f"{format_mean_std(summary, experiment, 'collisions')} | "
            f"{format_mean_std(summary, experiment, 'train_wall_time_sec', 2)} |"
        )

    if not delta_means.empty:
        lines.extend(
            [
                "",
                "## Paired Delta Vs Baseline",
                "",
                "| experiment | mean delta return | mean delta final_cov | mean delta max_cov | mean delta distance | mean delta collisions |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in delta_means.iterrows():
            lines.append(
                f"| {row['experiment']} | "
                f"{row['delta_return_vs_baseline']:.4f} | "
                f"{row['delta_final_coverage_vs_baseline']:.4f} | "
                f"{row['delta_max_coverage_vs_baseline']:.4f} | "
                f"{row['delta_nearest_landmark_distance_vs_baseline']:.4f} | "
                f"{row['delta_collisions_vs_baseline']:.4f} |"
            )

    if primary_vs_control_means is not None and not primary_vs_control_means.empty:
        lines.extend(
            [
                "",
                f"## {primary} Vs {control}",
                "",
                "| metric | mean delta |",
                "| --- | ---: |",
            ]
        )
        for _, row in primary_vs_control_means.iterrows():
            lines.append(f"| {row['metric']} | {row['mean']:.4f} |")

    best_return = summary.sort_values("return_mean", ascending=False).iloc[0]
    best_max_coverage = summary.sort_values("max_coverage_mean", ascending=False).iloc[0]
    best_distance = summary.sort_values(
        "nearest_landmark_distance_mean", ascending=True
    ).iloc[0]
    lines.extend(
        [
            "",
            "## Auto Interpretation",
            "",
            f"- Best mean return: `{best_return['experiment']}` ({best_return['return_mean']:.4f}).",
            f"- Best mean max coverage: `{best_max_coverage['experiment']}` ({best_max_coverage['max_coverage_mean']:.4f}).",
            f"- Best mean nearest-landmark distance: `{best_distance['experiment']}` ({best_distance['nearest_landmark_distance_mean']:.4f}).",
        ]
    )

    if primary and primary in summary_idx.index:
        primary_row = summary_idx.loc[primary]
        lines.append(
            f"- Primary candidate `{primary}` mean return: {primary_row['return_mean']:.4f}; "
            f"final coverage: {primary_row['final_coverage_mean']:.4f}; "
            f"max coverage: {primary_row['max_coverage_mean']:.4f}; "
            f"distance: {primary_row['nearest_landmark_distance_mean']:.4f}."
        )

    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dirs", nargs="+")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--baseline", default="raw_mlp")
    parser.add_argument("--primary", default="v3_active_gsp_residual005")
    parser.add_argument("--control", default="action_raw_potential_residual005")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    results = read_sources(args.source_dirs)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        stem = Path(args.source_dirs[0]).name
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = Path("experiments") / f"{stem}_review_{stamp}"
    if output_dir.exists() and not args.force:
        raise FileExistsError(f"output directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = summarize(results)
    deltas = paired_deltas(results, args.baseline)
    delta_means = (
        deltas.drop(columns=["seed"]).groupby("experiment").mean(numeric_only=True).reset_index()
        if not deltas.empty
        else pd.DataFrame()
    )

    primary_vs_control = None
    primary_vs_control_means = None
    if (
        args.primary in set(results["experiment"])
        and args.control in set(results["experiment"])
    ):
        primary_vs_control = pairwise_delta(results, args.primary, args.control)
        primary_vs_control_means = (
            primary_vs_control.drop(columns=["seed"])
            .mean(numeric_only=True)
            .to_frame("mean")
            .reset_index()
            .rename(columns={"index": "metric"})
        )

    results.sort_values(["source_dir", "seed", "experiment"]).to_csv(
        output_dir / "combined_results.csv", index=False
    )
    summary.to_csv(output_dir / "summary.csv", index=False)
    deltas.to_csv(output_dir / "paired_deltas_vs_baseline.csv", index=False)
    delta_means.to_csv(output_dir / "paired_delta_means_vs_baseline.csv", index=False)
    if primary_vs_control is not None:
        primary_vs_control.to_csv(
            output_dir / "primary_vs_control_paired_deltas.csv", index=False
        )
        primary_vs_control_means.to_csv(
            output_dir / "primary_vs_control_delta_means.csv", index=False
        )
    write_readme(
        output_dir=output_dir,
        source_dirs=args.source_dirs,
        summary=summary,
        deltas=deltas,
        delta_means=delta_means,
        primary=args.primary,
        control=args.control,
        primary_vs_control=primary_vs_control,
        primary_vs_control_means=primary_vs_control_means,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
