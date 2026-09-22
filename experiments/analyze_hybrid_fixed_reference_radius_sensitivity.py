"""Paired five-seed analysis for the fixed-reference KL-radius study."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


FROZEN = "hybrid_frozen"
METHODS = (
    "hybrid_frozen",
    "hybrid_fixed_teacher_kl001",
    "hybrid_fixed_teacher_kl002",
    "hybrid_fixed_teacher_kl005",
)
METRICS = (
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "minimum_agent_separation",
)
EXPECTED_CHECKPOINTS = [64, 100, 1000, 4996, 5000, 10000, 20000]
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
    half = TCRIT_N5 * values.std(ddof=1) / math.sqrt(values.size)
    return float(values.mean()), float(values.mean() - half), float(values.mean() + half)


def tensorboard_audit(seed_dir):
    payload = json.loads((seed_dir / "tensorboard_summary.json").read_text(encoding="utf-8"))
    kl = [float(series[-1][2]) for key, series in payload.items()
          if key.endswith("initial_reference_kl") and series]
    scales = [float(event[2]) for key, series in payload.items()
              if key.endswith("anchor_accepted_scale") for event in series]
    return {
        "heldout_kl": float(np.mean(kl)) if kl else 0.0,
        "proposals": len(scales),
        "accepted": int(np.count_nonzero(np.asarray(scales) > 0.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output = run_dir / "radius_sensitivity_analysis"
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((run_dir / "protocol_metadata.json").read_text(encoding="utf-8"))
    seeds = [int(seed) for seed in metadata["seeds"]]
    if tuple(metadata["methods"]) != METHODS or len(seeds) != 5:
        raise ValueError("Unexpected locked methods or seeds")

    by_method_seed = {}
    audit_rows = []
    counters_ok = True
    for method in METHODS:
        for seed in seeds:
            seed_dir = run_dir / method / f"seed_{seed}"
            by_method_seed[(method, seed)] = sorted(
                read_csv(seed_dir / "per_evaluation_episode_metrics.csv"),
                key=lambda row: int(row["test_seed"]),
            )
            audit_rows.append({"method": method, "training_seed": seed,
                               **tensorboard_audit(seed_dir)})
            counters = json.loads((seed_dir / "training_counters.json").read_text(encoding="utf-8"))
            counters_ok &= (
                int(counters["global_env_steps"]) == 20_000
                and counters["saved_checkpoints"] == EXPECTED_CHECKPOINTS
            )

    seed_rows = []
    for method in METHODS:
        for seed in seeds:
            reference = by_method_seed[(FROZEN, seed)]
            candidate = by_method_seed[(method, seed)]
            if [row["test_seed"] for row in reference] != [row["test_seed"] for row in candidate]:
                raise ValueError(f"Unmatched evaluation rows: {method}, seed {seed}")
            row = {"method": method, "training_seed": seed}
            for metric in METRICS:
                ref = np.asarray([float(item[metric]) for item in reference])
                val = np.asarray([float(item[metric]) for item in candidate])
                row[f"delta_{metric}"] = float((val - ref).mean())
            seed_rows.append(row)

    interval_rows = []
    lines = [
        "# Fixed-Reference KL-Radius Sensitivity",
        "",
        "Paired 95% t intervals over five fresh training seeds.",
        "",
        f"Protocol integrity: **{'PASS' if counters_ok else 'FAIL'}**.",
        "",
        "| method | H delta [95% CI] | AUC delta [95% CI] | collision delta [95% CI] | min-separation delta [95% CI] | held-out KL | accepted proposals |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        selected = [row for row in seed_rows if row["method"] == method]
        stats = {}
        for metric in METRICS:
            stats[metric] = interval([row[f"delta_{metric}"] for row in selected])
            interval_rows.append({
                "method": method,
                "metric": metric,
                "mean_delta": stats[metric][0],
                "ci95_low": stats[metric][1],
                "ci95_high": stats[metric][2],
            })
        audits = [row for row in audit_rows if row["method"] == method]
        heldout_kl = float(np.mean([row["heldout_kl"] for row in audits]))
        proposals = sum(row["proposals"] for row in audits)
        accepted = sum(row["accepted"] for row in audits)
        def cell(metric, digits):
            mean, low, high = stats[metric]
            return f"{mean:+.{digits}f} [{low:+.{digits}f}, {high:+.{digits}f}]"
        lines.append(
            f"| {method} | {cell('hungarian_assignment_distance', 4)} | "
            f"{cell('coverage_radius_auc', 4)} | {cell('collision_step_rate', 5)} | "
            f"{cell('minimum_agent_separation', 5)} | {heldout_kl:.5f} | "
            f"{accepted}/{proposals} |"
        )

    write_csv(output / "seed_summaries.csv", seed_rows)
    write_csv(output / "training_seed_intervals.csv", interval_rows)
    write_csv(output / "proposal_audit.csv", audit_rows)
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps({
        "run_dir": str(run_dir),
        "methods": list(METHODS),
        "training_seeds": seeds,
        "t_critical": TCRIT_N5,
        "counters_and_checkpoints_verified": counters_ok,
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
