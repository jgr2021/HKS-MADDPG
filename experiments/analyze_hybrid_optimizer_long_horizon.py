"""Five-seed analysis for the 100k optimizer durability experiment."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


METRICS = (
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "return",
    "final_coverage",
    "max_coverage",
    "minimum_agent_separation",
)
FROZEN = "hybrid_frozen"
FIXED = "hybrid_fixed_teacher_kl002"
TCRIT_N5 = 2.7764451051977987
EXPECTED_CHECKPOINTS = [64, 100, 1000, 20000, 40000, 60000, 80000, 100000]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    half = TCRIT_N5 * values.std(ddof=1) / math.sqrt(len(values))
    return {
        "mean_delta": float(values.mean()),
        "ci95_low": float(values.mean() - half),
        "ci95_high": float(values.mean() + half),
        "seeds_positive": int((values > 0).sum()),
        "seeds_negative": int((values < 0).sum()),
        "per_seed_deltas": ";".join(f"{value:.9g}" for value in values),
    }


def final_audit_kl(seed_dir):
    payload = json.loads(
        (seed_dir / "tensorboard_summary.json").read_text(encoding="utf-8")
    )
    values = [
        float(series[-1][2])
        for key, series in payload.items()
        if key.endswith("initial_reference_kl") and series
    ]
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output = run_dir / "long_horizon_analysis"
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(
        (run_dir / "protocol_metadata.json").read_text(encoding="utf-8")
    )
    methods = metadata["methods"]
    seeds = [int(seed) for seed in metadata["seeds"]]
    if len(seeds) != 5:
        raise ValueError(f"Expected five seeds, received {seeds}")

    rows_by_method_seed = {}
    audit_rows = []
    counters_ok = True
    for method in methods:
        for seed in seeds:
            seed_dir = run_dir / method / f"seed_{seed}"
            rows_by_method_seed[(method, seed)] = sorted(
                read_csv(seed_dir / "per_evaluation_episode_metrics.csv"),
                key=lambda row: int(row["test_seed"]),
            )
            audit_rows.append({
                "method": method,
                "training_seed": seed,
                "final_initial_reference_kl": final_audit_kl(seed_dir),
            })
            counters = json.loads(
                (seed_dir / "training_counters.json").read_text(encoding="utf-8")
            )
            counters_ok &= (
                int(counters["global_env_steps"]) == 100_000
                and counters["saved_checkpoints"] == EXPECTED_CHECKPOINTS
            )

    seed_rows = []
    episode_rows = []
    for method in methods:
        for seed in seeds:
            frozen = rows_by_method_seed[(FROZEN, seed)]
            candidate = rows_by_method_seed[(method, seed)]
            if [row["test_seed"] for row in frozen] != [
                row["test_seed"] for row in candidate
            ]:
                raise ValueError(f"Unmatched rows for {method}, seed {seed}")
            summary = {
                "method": method,
                "training_seed": seed,
                "episodes": len(candidate),
            }
            for metric in METRICS:
                reference = np.asarray(
                    [float(row[metric]) for row in frozen], dtype=np.float64
                )
                values = np.asarray(
                    [float(row[metric]) for row in candidate], dtype=np.float64
                )
                summary[f"final_{metric}"] = float(values.mean())
                summary[f"delta_vs_frozen_{metric}"] = float(
                    (values - reference).mean()
                )
            seed_rows.append(summary)
            episode_rows.extend(
                {**row, "method": method, "training_seed": seed}
                for row in candidate
            )

    comparisons = []
    comparison_lookup = {}
    for method in methods:
        selected = [row for row in seed_rows if row["method"] == method]
        for metric in METRICS:
            result = interval(
                [row[f"delta_vs_frozen_{metric}"] for row in selected]
            )
            row = {
                "method": method,
                "comparison": "method_minus_frozen",
                "metric": metric,
                **result,
            }
            comparisons.append(row)
            comparison_lookup[(method, "method_minus_frozen", metric)] = row

    if FIXED in methods:
        fixed_by_seed = {
            row["training_seed"]: row
            for row in seed_rows
            if row["method"] == FIXED
        }
        for method in methods:
            selected = [row for row in seed_rows if row["method"] == method]
            for metric in METRICS:
                result = interval([
                    row[f"final_{metric}"]
                    - fixed_by_seed[row["training_seed"]][f"final_{metric}"]
                    for row in selected
                ])
                comparisons.append({
                    "method": method,
                    "comparison": "method_minus_fixed_teacher",
                    "metric": metric,
                    **result,
                })

    decisions = []
    for method in methods:
        h = comparison_lookup[
            (method, "method_minus_frozen", "hungarian_assignment_distance")
        ]
        auc = comparison_lookup[
            (method, "method_minus_frozen", "coverage_radius_auc")
        ]
        collision = comparison_lookup[
            (method, "method_minus_frozen", "collision_step_rate")
        ]
        durable = (
            h["ci95_low"] <= 0.0
            and auc["ci95_high"] >= 0.0
            and collision["ci95_low"] <= 0.0
        )
        positive = (
            h["ci95_high"] < 0.0
            and auc["ci95_low"] > 0.0
            and collision["ci95_low"] <= 0.0
        )
        decisions.append({
            "method": method,
            "durable": durable,
            "positive_improvement": positive,
        })

    write_csv(output / "per_episode.csv", episode_rows)
    write_csv(output / "seed_summaries.csv", seed_rows)
    write_csv(output / "training_seed_intervals.csv", comparisons)
    write_csv(output / "policy_audit_kl.csv", audit_rows)
    write_csv(output / "decisions.csv", decisions)

    lines = [
        "# Hybrid Optimizer Long-Horizon Result",
        "",
        "Intervals are paired 95% t intervals over five fresh training seeds.",
        "",
        f"Protocol integrity: **{'PASS' if counters_ok else 'FAIL'}**.",
        "",
        "| method | durable | positive improvement | H delta [95% CI] | AUC delta [95% CI] | collision delta [95% CI] |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for decision in decisions:
        method = decision["method"]
        h = comparison_lookup[
            (method, "method_minus_frozen", "hungarian_assignment_distance")
        ]
        auc = comparison_lookup[
            (method, "method_minus_frozen", "coverage_radius_auc")
        ]
        collision = comparison_lookup[
            (method, "method_minus_frozen", "collision_step_rate")
        ]
        lines.append(
            f"| {method} | {'yes' if decision['durable'] else 'no'} | "
            f"{'yes' if decision['positive_improvement'] else 'no'} | "
            f"{h['mean_delta']:+.4f} [{h['ci95_low']:+.4f}, {h['ci95_high']:+.4f}] | "
            f"{auc['mean_delta']:+.4f} [{auc['ci95_low']:+.4f}, {auc['ci95_high']:+.4f}] | "
            f"{collision['mean_delta']:+.5f} "
            f"[{collision['ci95_low']:+.5f}, {collision['ci95_high']:+.5f}] |"
        )
    (output / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps({
            "run_dir": str(run_dir),
            "methods": methods,
            "training_seeds": seeds,
            "t_critical": TCRIT_N5,
            "counters_and_checkpoints_verified": counters_ok,
        }, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
