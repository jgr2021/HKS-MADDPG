"""Apply the predeclared soft-anchor development selection rule."""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np


FROZEN = "hybrid_frozen"
FAMILIES = {
    "kl_loss": (
        "hybrid_soft_kl_0p1", "hybrid_soft_kl_1", "hybrid_soft_kl_10"
    ),
    "l2": (
        "hybrid_soft_l2_1", "hybrid_soft_l2_10", "hybrid_soft_l2_100"
    ),
}
COEFFICIENTS = {
    "hybrid_soft_kl_0p1": 0.1,
    "hybrid_soft_kl_1": 1.0,
    "hybrid_soft_kl_10": 10.0,
    "hybrid_soft_l2_1": 1.0,
    "hybrid_soft_l2_10": 10.0,
    "hybrid_soft_l2_100": 100.0,
}
METRICS = (
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "minimum_agent_separation",
)


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
    critical = 12.7062 if len(values) == 2 else 1.96
    half = critical * values.std(ddof=1) / math.sqrt(len(values))
    return float(values.mean()), float(values.mean() - half), float(values.mean() + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    output = run_dir / "soft_anchor_selection"
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(
        (run_dir / "protocol_metadata.json").read_text(encoding="utf-8")
    )
    expected = [FROZEN, *FAMILIES["kl_loss"], *FAMILIES["l2"]]
    protocol_pass = (
        metadata["methods"] == expected
        and metadata["seeds"] == [75, 76]
        and metadata["total_env_steps"] == 20000
        and metadata["eval_episodes_final"] == 200
    )
    if not protocol_pass:
        raise RuntimeError("Soft-anchor development protocol mismatch")
    drift_dir = run_dir / "soft_anchor_policy_drift"
    if not drift_dir.exists():
        subprocess.run([
            sys.executable,
            "experiments/analyze_hybrid_policy_margin_drift.py",
            "--run-dir", str(run_dir),
            "--episodes", "100",
            "--horizon", "25",
            "--seed-base", "42200000",
            "--output-name", "soft_anchor_policy_drift",
        ], check=True)
    drift = read_csv(drift_dir / "margin_drift_summary.csv")
    drift_lookup = {
        (row["method"], int(row["training_seed"]), int(row["checkpoint_step"])): row
        for row in drift
    }
    summaries = read_csv(run_dir / "seed_summaries.csv")
    by_seed_method = {
        (int(row["seed"]), row["method"]): row for row in summaries
    }
    per_seed = []
    for method in expected:
        for seed in (75, 76):
            row = by_seed_method[(seed, method)]
            frozen = by_seed_method[(seed, FROZEN)]
            output_row = {"method": method, "seed": seed}
            for metric in METRICS:
                output_row["delta_" + metric] = (
                    float(row[metric]) - float(frozen[metric])
                )
            audit = drift_lookup[(method, seed, 20000)]
            output_row["heldout_initial_kl"] = float(
                audit["initial_reference_kl_mean"]
            )
            output_row["initial_action_agreement"] = float(
                audit["initial_action_agreement"]
            )
            per_seed.append(output_row)
    write_csv(output / "per_seed_deltas.csv", per_seed)

    method_rows = []
    for method in expected:
        selected = [row for row in per_seed if row["method"] == method]
        result = {
            "method": method,
            "family": (
                "frozen" if method == FROZEN
                else "kl_loss" if method in FAMILIES["kl_loss"] else "l2"
            ),
            "coefficient": COEFFICIENTS.get(method, 0.0),
            "n_dev_seeds": len(selected),
        }
        for metric in METRICS:
            mean, low, high = interval([
                row["delta_" + metric] for row in selected
            ])
            result["delta_" + metric + "_mean"] = mean
            result["delta_" + metric + "_ci95_low"] = low
            result["delta_" + metric + "_ci95_high"] = high
        result["heldout_initial_kl_mean"] = float(np.mean([
            row["heldout_initial_kl"] for row in selected
        ]))
        result["initial_action_agreement_mean"] = float(np.mean([
            row["initial_action_agreement"] for row in selected
        ]))
        result["resolved_safety_penalty"] = bool(
            result["delta_collision_step_rate_ci95_low"] > 0.0
            or result["delta_minimum_agent_separation_ci95_high"] < 0.0
        )
        result["development_stable"] = bool(
            method != FROZEN
            and result["delta_hungarian_assignment_distance_mean"] <= 0.02
            and result["delta_coverage_radius_auc_mean"] >= -0.02
            and not result["resolved_safety_penalty"]
        )
        method_rows.append(result)
    write_csv(output / "method_intervals.csv", method_rows)

    selections = {}
    for family, methods in FAMILIES.items():
        candidates = [
            row for row in method_rows
            if row["method"] in methods and row["development_stable"]
        ]
        family_failed = not candidates
        if not candidates:
            candidates = [row for row in method_rows if row["method"] in methods]
        selected = sorted(
            candidates,
            key=lambda row: (
                -row["delta_coverage_radius_auc_mean"],
                row["heldout_initial_kl_mean"],
                row["coefficient"],
            ),
        )[0]
        selections[family] = {
            "method": selected["method"],
            "coefficient": selected["coefficient"],
            "development_failed": family_failed,
            "development_stable": selected["development_stable"],
            "selection_rule": (
                "stable then max AUC, lower KL, lower coefficient; "
                "if none stable, least adverse AUC"
            ),
        }
    selection_payload = {
        "protocol_pass": protocol_pass,
        "heldout_outcomes_viewed": False,
        "selected": selections,
    }
    (output / "selection.json").write_text(
        json.dumps(selection_payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Soft-Anchor Development Selection", "",
        "Held-out 100k outcomes were not used for selection.", "",
        "| family | selected method | coefficient | stable | family failed |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for family, value in selections.items():
        lines.append(
            f"| {family} | {value['method']} | {value['coefficient']:.4g} | "
            f"{value['development_stable']} | {value['development_failed']} |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
