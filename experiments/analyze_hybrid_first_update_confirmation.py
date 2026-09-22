"""Analyze the fresh five-seed first-update confirmation."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


FROZEN = "hybrid_frozen"
CANDIDATE = "hybrid_first_update_then_freeze"
METRICS = (
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "return",
    "minimum_agent_separation",
)
TCRIT_N5 = 2.7764451051977987


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output = run_dir / "first_update_analysis"
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(
        (run_dir / "protocol_metadata.json").read_text(encoding="utf-8")
    )
    seeds = [int(seed) for seed in metadata["seeds"]]
    if len(seeds) != 5:
        raise ValueError(f"Expected five fresh seeds, got {seeds}")
    if metadata["methods"] != [FROZEN, CANDIDATE]:
        raise ValueError(metadata["methods"])

    seed_rows = []
    per_episode = []
    deltas = {metric: [] for metric in METRICS}
    protocol_ok = True
    for seed in seeds:
        by_method = {}
        for method in (FROZEN, CANDIDATE):
            seed_dir = run_dir / method / f"seed_{seed}"
            rows = sorted(
                read_csv(seed_dir / "per_evaluation_episode_metrics.csv"),
                key=lambda row: int(row["test_seed"]),
            )
            by_method[method] = rows
            counters = json.loads(
                (seed_dir / "training_counters.json").read_text(
                    encoding="utf-8"
                )
            )
            protocol_ok &= (
                int(counters["global_env_steps"]) == 100
                and counters["saved_checkpoints"] == [64, 100]
            )
            if method == CANDIDATE:
                shapes = json.loads(
                    (seed_dir / "tensor_shape_report.json").read_text(
                        encoding="utf-8"
                    )
                )
                protocol_ok &= (
                    int(shapes["global_env_steps_at_first_update"]) == 100
                )

        frozen = by_method[FROZEN]
        candidate = by_method[CANDIDATE]
        if [row["test_seed"] for row in frozen] != [
            row["test_seed"] for row in candidate
        ]:
            raise ValueError(f"Unmatched evaluation seeds for training seed {seed}")

        summary = {
            "method": CANDIDATE,
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
            delta = float((values - reference).mean())
            summary[f"final_{metric}"] = float(values.mean())
            summary[f"delta_vs_frozen_{metric}"] = delta
            deltas[metric].append(delta)
        seed_rows.append(summary)
        for method, rows in by_method.items():
            per_episode.extend(
                {**row, "method": method, "training_seed": seed}
                for row in rows
            )

    interval_rows = [
        {
            "method": CANDIDATE,
            "comparison": "method_minus_frozen",
            "metric": metric,
            **interval(values),
        }
        for metric, values in deltas.items()
    ]
    lookup = {row["metric"]: row for row in interval_rows}
    write_csv(output / "per_episode.csv", per_episode)
    write_csv(output / "seed_summaries.csv", seed_rows)
    write_csv(output / "training_seed_intervals.csv", interval_rows)

    h = lookup["hungarian_assignment_distance"]
    auc = lookup["coverage_radius_auc"]
    collision = lookup["collision_step_rate"]
    positive = (
        h["ci95_high"] < 0.0
        and auc["ci95_low"] > 0.0
        and collision["ci95_low"] <= 0.0
    )
    lines = [
        "# Fresh First-Update-Then-Freeze Confirmation",
        "",
        "Intervals are paired 95% t intervals over five fresh training seeds.",
        "",
        f"Protocol integrity: **{'PASS' if protocol_ok else 'FAIL'}**.",
        "",
        f"Joint positive-improvement gate: **{'PASS' if positive else 'FAIL'}**.",
        "",
        "| H delta [95% CI] | AUC delta [95% CI] | collision delta [95% CI] |",
        "| ---: | ---: | ---: |",
        (
            f"| {h['mean_delta']:+.4f} [{h['ci95_low']:+.4f}, "
            f"{h['ci95_high']:+.4f}] | {auc['mean_delta']:+.4f} "
            f"[{auc['ci95_low']:+.4f}, {auc['ci95_high']:+.4f}] | "
            f"{collision['mean_delta']:+.5f} "
            f"[{collision['ci95_low']:+.5f}, "
            f"{collision['ci95_high']:+.5f}] |"
        ),
    ]
    (output / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "training_seeds": seeds,
                "statistical_unit": "training_seed",
                "t_critical": TCRIT_N5,
                "protocol_integrity": protocol_ok,
                "joint_positive_improvement": positive,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
