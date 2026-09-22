"""Replicate-level analysis across three independent distilled policies."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


REPLICATES = (
    "fold21_t101_s101",
    "fold22_t102_s102",
    "fold23_t103_s103",
)
METHODS = (
    "hybrid_frozen",
    "hybrid_unanchored",
    "hybrid_fixed_teacher_kl002",
)
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
TCRIT_N3 = 4.302652729696142
EXPECTED_CHECKPOINTS = [64, 100, 1000, 5000, 10000, 20000]


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
    half = TCRIT_N3 * values.std(ddof=1) / math.sqrt(len(values))
    return {
        "mean_delta": float(values.mean()),
        "ci95_low": float(values.mean() - half),
        "ci95_high": float(values.mean() + half),
        "replicates_positive": int((values > 0).sum()),
        "replicates_negative": int((values < 0).sum()),
        "per_replicate_deltas": ";".join(
            f"{value:.9g}" for value in values
        ),
    }


def only_run_dir(base):
    runs = sorted(base.glob("run_*"))
    if len(runs) != 1:
        raise ValueError(f"Expected exactly one run under {base}, got {runs}")
    return runs[0]


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
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="experiments/independent_hybrid_online_analysis_20260716",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    rows_by_replicate_method = {}
    run_dirs = {}
    protocol_ok = True
    audit_rows = []
    for replicate in REPLICATES:
        base = (
            root
            / "experiments"
            / f"independent_hybrid_online_{replicate}_20260716"
        )
        run_dir = only_run_dir(base)
        run_dirs[replicate] = run_dir
        metadata = json.loads(
            (run_dir / "protocol_metadata.json").read_text(encoding="utf-8")
        )
        protocol_ok &= metadata["methods"] == list(METHODS)
        protocol_ok &= int(metadata["total_env_steps"]) == 20_000
        seed = int(metadata["seeds"][0])
        for method in METHODS:
            seed_dir = run_dir / method / f"seed_{seed}"
            rows_by_replicate_method[(replicate, method)] = sorted(
                read_csv(seed_dir / "per_evaluation_episode_metrics.csv"),
                key=lambda row: int(row["test_seed"]),
            )
            counters = json.loads(
                (seed_dir / "training_counters.json").read_text(
                    encoding="utf-8"
                )
            )
            protocol_ok &= (
                int(counters["global_env_steps"]) == 20_000
                and counters["saved_checkpoints"] == EXPECTED_CHECKPOINTS
            )
            audit_rows.append({
                "replicate": replicate,
                "method": method,
                "training_seed": seed,
                "final_initial_reference_kl": final_audit_kl(seed_dir),
            })

    replicate_rows = []
    episode_rows = []
    for replicate in REPLICATES:
        reference = rows_by_replicate_method[(replicate, FROZEN)]
        reference_seeds = [row["test_seed"] for row in reference]
        for method in METHODS:
            candidate = rows_by_replicate_method[(replicate, method)]
            if [row["test_seed"] for row in candidate] != reference_seeds:
                raise ValueError(
                    f"Unmatched evaluation rows for {replicate}, {method}"
                )
            summary = {
                "replicate": replicate,
                "method": method,
                "episodes": len(candidate),
            }
            for metric in METRICS:
                frozen_values = np.asarray(
                    [float(row[metric]) for row in reference],
                    dtype=np.float64,
                )
                values = np.asarray(
                    [float(row[metric]) for row in candidate],
                    dtype=np.float64,
                )
                summary[f"final_{metric}"] = float(values.mean())
                summary[f"delta_vs_frozen_{metric}"] = float(
                    (values - frozen_values).mean()
                )
            replicate_rows.append(summary)
            episode_rows.extend(
                {**row, "replicate": replicate, "method": method}
                for row in candidate
            )

    comparisons = []
    lookup = {}
    for method in METHODS:
        selected = [
            row for row in replicate_rows if row["method"] == method
        ]
        for metric in METRICS:
            result = interval([
                row[f"delta_vs_frozen_{metric}"] for row in selected
            ])
            item = {
                "method": method,
                "comparison": "method_minus_frozen",
                "metric": metric,
                **result,
            }
            comparisons.append(item)
            lookup[(method, metric)] = item

    decisions = []
    for method in METHODS:
        h = lookup[(method, "hungarian_assignment_distance")]
        auc = lookup[(method, "coverage_radius_auc")]
        collision = lookup[(method, "collision_step_rate")]
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
    write_csv(output / "replicate_summaries.csv", replicate_rows)
    write_csv(output / "replicate_intervals.csv", comparisons)
    write_csv(output / "policy_audit_kl.csv", audit_rows)
    write_csv(output / "decisions.csv", decisions)

    lines = [
        "# Independent Hybrid Online Replication",
        "",
        "The independently distilled policy is the statistical unit. Intervals "
        "are paired 95% t intervals over three teacher/student replicates.",
        "",
        f"Protocol integrity: **{'PASS' if protocol_ok else 'FAIL'}**.",
        "",
        "| method | durable | positive improvement | H delta [95% CI] | "
        "AUC delta [95% CI] | collision delta [95% CI] |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for decision in decisions:
        method = decision["method"]
        h = lookup[(method, "hungarian_assignment_distance")]
        auc = lookup[(method, "coverage_radius_auc")]
        collision = lookup[(method, "collision_step_rate")]
        lines.append(
            f"| {method} | {'yes' if decision['durable'] else 'no'} | "
            f"{'yes' if decision['positive_improvement'] else 'no'} | "
            f"{h['mean_delta']:+.4f} "
            f"[{h['ci95_low']:+.4f}, {h['ci95_high']:+.4f}] | "
            f"{auc['mean_delta']:+.4f} "
            f"[{auc['ci95_low']:+.4f}, {auc['ci95_high']:+.4f}] | "
            f"{collision['mean_delta']:+.5f} "
            f"[{collision['ci95_low']:+.5f}, "
            f"{collision['ci95_high']:+.5f}] |"
        )
    (output / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (output / "metadata.json").write_text(
        json.dumps({
            "run_dirs": {
                key: str(value) for key, value in run_dirs.items()
            },
            "replicates": list(REPLICATES),
            "methods": list(METHODS),
            "t_critical": TCRIT_N3,
            "protocol_integrity": protocol_ok,
        }, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
