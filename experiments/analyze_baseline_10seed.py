"""Merge the locked 3-seed screen with its 7-seed baseline extension."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


METHODS = ("raw_maddpg", "action_aware_geometric_control")
METRICS = (
    "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate",
    "return", "final_coverage", "max_coverage", "minimum_agent_separation",
    "unique_landmarks_covered",
)
T_CRITICAL_95_DF9 = 2.2621571629


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-root", default="experiments/vector_signal_gsp_locked_screen/run_20260711_231009")
    parser.add_argument("--extension-root", required=True)
    parser.add_argument("--output-dir", default="experiments/baseline_10seed_analysis_20260712")
    args = parser.parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    seed_rows = read_csv(Path(args.screen_root) / "seed_summaries.csv")
    seed_rows += read_csv(Path(args.extension_root) / "seed_summaries.csv")
    seed_rows = [row for row in seed_rows if row["method"] in METHODS and 21 <= int(row["seed"]) <= 30]
    seed_rows.sort(key=lambda row: (row["method"], int(row["seed"])))
    lookup = {(row["method"], int(row["seed"])): row for row in seed_rows}
    missing = [(method, seed) for method in METHODS for seed in range(21, 31) if (method, seed) not in lookup]
    if missing:
        raise RuntimeError(f"Incomplete 10-seed baseline: {missing}")

    aggregate = []
    for method in METHODS:
        selected = [lookup[(method, seed)] for seed in range(21, 31)]
        row = {"method": method, "n_seeds": 10}
        for metric in METRICS:
            values = np.asarray([float(item[metric]) for item in selected])
            row[f"{metric}_mean"] = values.mean(); row[f"{metric}_std"] = values.std(ddof=1)
            row[f"{metric}_median"] = np.median(values); row[f"{metric}_min"] = values.min()
            row[f"{metric}_max"] = values.max()
        aggregate.append(row)

    paired = []
    for metric in METRICS:
        values = np.asarray([
            float(lookup[(METHODS[1], seed)][metric]) - float(lookup[(METHODS[0], seed)][metric])
            for seed in range(21, 31)
        ])
        mean = values.mean(); half = T_CRITICAL_95_DF9 * values.std(ddof=1) / math.sqrt(10)
        paired.append({
            "comparison": "geometry_minus_raw", "metric": metric, "mean_delta": mean,
            "ci95_low": mean - half, "ci95_high": mean + half,
            "positive_seed_count": int((values > 0).sum()), "negative_seed_count": int((values < 0).sum()),
            "seed_deltas": ";".join(f"{value:+.9f}" for value in values),
        })

    curve_rows = read_csv(Path(args.screen_root) / "learning_curve_eval.csv")
    curve_rows += read_csv(Path(args.extension_root) / "learning_curve_eval.csv")
    curve_rows = [row for row in curve_rows if row["method"] in METHODS and 21 <= int(row["seed"]) <= 30]
    curve_summary = []
    checkpoints = (20000, 40000, 60000, 80000, 100000)
    for method in METHODS:
        for step in checkpoints:
            selected = [row for row in curve_rows if row["method"] == method and row["checkpoint"] == f"model_step{step}"]
            if len(selected) != 10:
                raise RuntimeError(f"Expected 10 curve rows for {method}/{step}, got {len(selected)}")
            row = {"method": method, "checkpoint_step": step, "n_seeds": 10}
            for metric in ("return", "final_coverage", "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate"):
                values = np.asarray([float(item[metric]) for item in selected])
                row[f"{metric}_mean"] = values.mean(); row[f"{metric}_std"] = values.std(ddof=1)
            curve_summary.append(row)

    write_csv(output_dir / "combined_seed_summaries.csv", seed_rows)
    write_csv(output_dir / "aggregate_summary.csv", aggregate)
    write_csv(output_dir / "paired_seed_deltas.csv", paired)
    write_csv(output_dir / "learning_curve_summary.csv", curve_summary)
    agg = {row["method"]: row for row in aggregate}
    delta = {row["metric"]: row for row in paired}
    labels = {METHODS[0]: "Raw MADDPG", METHODS[1]: "Action-aware geometry"}
    lines = ["# Unified 10-Seed Raw/Geometry Baseline", "",
             "Seeds 21--30, exactly 100,000 transitions, 500 matched final evaluations per seed.", "",
             "| method | Hungarian | radius AUC | collision-step | return | final cov@0.10 | max cov@0.10 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in METHODS:
        row = agg[method]
        lines.append(f"| {labels[method]} | {row['hungarian_assignment_distance_mean']:.4f} +/- {row['hungarian_assignment_distance_std']:.4f} | "
                     f"{row['coverage_radius_auc_mean']:.4f} +/- {row['coverage_radius_auc_std']:.4f} | "
                     f"{row['collision_step_rate_mean']:.4f} +/- {row['collision_step_rate_std']:.4f} | "
                     f"{row['return_mean']:.3f} +/- {row['return_std']:.3f} | "
                     f"{row['final_coverage_mean']:.3f} +/- {row['final_coverage_std']:.3f} | "
                     f"{row['max_coverage_mean']:.3f} +/- {row['max_coverage_std']:.3f} |")
    lines += ["", "## Geometry minus Raw", "", "| metric | mean delta | 95% interval | positive/negative seeds |", "| --- | ---: | ---: | ---: |"]
    for metric in METRICS:
        row = delta[metric]
        lines.append(f"| {metric} | {float(row['mean_delta']):+.4f} | [{float(row['ci95_low']):+.4f},{float(row['ci95_high']):+.4f}] | {row['positive_seed_count']}/{row['negative_seed_count']} |")
    hungarian_better = delta["hungarian_assignment_distance"]["ci95_high"] < 0
    auc_better = delta["coverage_radius_auc"]["ci95_low"] > 0
    collision_worse = delta["collision_step_rate"]["ci95_low"] > 0
    lines += ["", "## Decision", "",
              f"- Resolved Hungarian improvement: **{hungarian_better}**.",
              f"- Resolved radius-AUC improvement: **{auc_better}**.",
              f"- Resolved collision penalty: **{collision_worse}**.", ""]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "analysis_metadata.json").write_text(json.dumps({
        "screen_root": str(Path(args.screen_root).resolve()),
        "extension_root": str(Path(args.extension_root).resolve()),
        "training_seed_is_statistical_unit": True,
        "paired_t_interval_df": 9,
    }, indent=2) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
