"""Merge held-out soft anchors with immutable optimizer controls."""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LONG_RUN = Path(
    "experiments/hybrid_optimizer_long_horizon_20260716/"
    "run_20260716_223503"
)
STAGE_B_RUN = Path(
    "experiments/hybrid_optimizer_controls_stage_b_20260716/"
    "run_20260716_214709"
)
OUTPUT = Path("experiments/hybrid_soft_anchor_heldout_analysis_20260811")
METRICS = (
    "hungarian_assignment_distance", "coverage_radius_auc",
    "collision_step_rate", "minimum_agent_separation", "return",
)
METHOD_SOURCES = {
    "hybrid_frozen": (LONG_RUN, 100000),
    "hybrid_actor_lr1e4": (LONG_RUN, 100000),
    "hybrid_fixed_teacher_kl002": (LONG_RUN, 100000),
    "hybrid_unanchored": (STAGE_B_RUN, 20000),
    "hybrid_old_policy_kl002": (STAGE_B_RUN, 20000),
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def t_interval(values, one_sided=False):
    values = np.asarray(values, dtype=np.float64)
    if one_sided:
        critical = {2: 2.9200, 4: 2.1318}.get(len(values) - 1, 1.6449)
    else:
        critical = {2: 4.3027, 4: 2.7764}.get(len(values) - 1, 1.96)
    half = critical * values.std(ddof=1) / math.sqrt(len(values))
    return float(values.mean()), float(values.mean() - half), float(values.mean() + half)


def latest_run(parent):
    runs = sorted(parent.glob("run_*"))
    if len(runs) != 1:
        raise RuntimeError(f"Expected one held-out run under {parent}, got {runs}")
    return runs[0]


def load_summary(run_dir):
    return {
        (row["method"], int(row["seed"])): row
        for row in read_csv(run_dir / "seed_summaries.csv")
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout-parent", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    soft_run = latest_run(args.heldout_parent)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    protocol = json.loads(
        (soft_run / "protocol_metadata.json").read_text(encoding="utf-8")
    )
    protocol_pass = (
        protocol["methods"]
        == ["hybrid_soft_kl_selected", "hybrid_soft_l2_selected"]
        and protocol["seeds"] == [55, 56, 57, 58, 59]
        and protocol["total_env_steps"] == 100000
        and protocol["eval_episodes_final"] == 500
    )
    if not protocol_pass:
        raise RuntimeError("Held-out soft-anchor protocol mismatch")
    soft_drift = soft_run / "soft_anchor_heldout_policy_drift"
    if not soft_drift.exists():
        subprocess.run([
            sys.executable,
            "experiments/analyze_hybrid_policy_margin_drift.py",
            "--run-dir", str(soft_run),
            "--episodes", "200",
            "--horizon", "25",
            "--seed-base", "43000000",
            "--output-name", "soft_anchor_heldout_policy_drift",
            "--reference-method", "hybrid_soft_kl_selected",
        ], check=True)

    long_summary = load_summary(LONG_RUN)
    stage_b_summary = load_summary(STAGE_B_RUN)
    soft_summary = load_summary(soft_run)
    seeds = (55, 56, 57, 58, 59)
    per_seed = []
    for method, (source, horizon) in METHOD_SOURCES.items():
        source_summary = long_summary if source == LONG_RUN else stage_b_summary
        source_seeds = seeds if source == LONG_RUN else (52, 53, 54)
        for seed in source_seeds:
            row = source_summary[(method, seed)]
            frozen = source_summary[("hybrid_frozen", seed)]
            result = {"method": method, "seed": seed, "horizon": horizon}
            for metric in METRICS:
                result["final_" + metric] = float(row[metric])
                result["delta_" + metric] = float(row[metric]) - float(frozen[metric])
            per_seed.append(result)
    for method in ("hybrid_soft_kl_selected", "hybrid_soft_l2_selected"):
        for seed in seeds:
            row = soft_summary[(method, seed)]
            frozen = long_summary[("hybrid_frozen", seed)]
            result = {"method": method, "seed": seed, "horizon": 100000}
            for metric in METRICS:
                result["final_" + metric] = float(row[metric])
                result["delta_" + metric] = float(row[metric]) - float(frozen[metric])
            per_seed.append(result)
    write_csv(OUTPUT / "per_seed_effects.csv", per_seed)

    intervals = []
    for method in [*METHOD_SOURCES, "hybrid_soft_kl_selected", "hybrid_soft_l2_selected"]:
        selected = [row for row in per_seed if row["method"] == method]
        output = {
            "method": method,
            "horizon": selected[0]["horizon"],
            "n_training_seeds": len(selected),
        }
        for metric in METRICS:
            mean, low, high = t_interval([
                row["delta_" + metric] for row in selected
            ])
            output["delta_" + metric + "_mean"] = mean
            output["delta_" + metric + "_ci95_low"] = low
            output["delta_" + metric + "_ci95_high"] = high
        intervals.append(output)
    write_csv(OUTPUT / "method_intervals.csv", intervals)

    fixed_rows = [
        row for row in per_seed if row["method"] == "hybrid_fixed_teacher_kl002"
    ]
    replacement = []
    for method in ("hybrid_soft_kl_selected", "hybrid_soft_l2_selected"):
        soft_rows = [row for row in per_seed if row["method"] == method]
        by_seed = {int(row["seed"]): row for row in soft_rows}
        fixed_by_seed = {int(row["seed"]): row for row in fixed_rows}
        comparisons = {}
        for metric in METRICS:
            values = [
                by_seed[seed]["final_" + metric]
                - fixed_by_seed[seed]["final_" + metric]
                for seed in seeds
            ]
            mean, low, high = t_interval(values, one_sided=True)
            comparisons[metric] = {"mean": mean, "low": low, "high": high}
        h_pass = comparisons["hungarian_assignment_distance"]["high"] <= 0.01
        auc_pass = comparisons["coverage_radius_auc"]["low"] >= -0.01
        safety_pass = not (
            comparisons["collision_step_rate"]["low"] > 0.0
            or comparisons["minimum_agent_separation"]["high"] < 0.0
        )
        replacement.append({
            "method": method,
            "h_noninferiority_upper": comparisons[
                "hungarian_assignment_distance"
            ]["high"],
            "h_tolerance": 0.01,
            "h_pass": h_pass,
            "auc_noninferiority_lower": comparisons[
                "coverage_radius_auc"
            ]["low"],
            "auc_tolerance": -0.01,
            "auc_pass": auc_pass,
            "safety_pass": safety_pass,
            "can_replace_fixed_backtracking": bool(h_pass and auc_pass and safety_pass),
        })
    write_csv(OUTPUT / "replacement_tests.csv", replacement)

    long_drift = read_csv(
        LONG_RUN / "policy_margin_drift_certified/margin_drift_summary.csv"
    )
    stage_drift = read_csv(
        STAGE_B_RUN / "policy_margin_drift_certified/margin_drift_summary.csv"
    )
    soft_drift_rows = read_csv(soft_drift / "margin_drift_summary.csv")
    audit_rows = []
    audit_specs = [
        (long_drift, "hybrid_actor_lr1e4", 100000),
        (long_drift, "hybrid_fixed_teacher_kl002", 100000),
        (stage_drift, "hybrid_unanchored", 20000),
        (stage_drift, "hybrid_old_policy_kl002", 20000),
        (soft_drift_rows, "hybrid_soft_kl_selected", 100000),
        (soft_drift_rows, "hybrid_soft_l2_selected", 100000),
    ]
    audit_metrics = [
        "initial_reference_kl_mean", "initial_action_agreement",
        "agreement_initial_margin_q1", "agreement_initial_margin_q2",
        "agreement_initial_margin_q3", "agreement_initial_margin_q4",
    ]
    for source, method, step in audit_specs:
        selected = [
            row for row in source
            if row["method"] == method and int(row["checkpoint_step"]) == step
        ]
        output = {
            "method": method, "checkpoint_step": step,
            "n_training_seeds": len(selected),
        }
        for metric in audit_metrics:
            mean, low, high = t_interval([float(row[metric]) for row in selected])
            output[metric + "_mean"] = mean
            output[metric + "_ci95_low"] = low
            output[metric + "_ci95_high"] = high
        audit_rows.append(output)
    write_csv(OUTPUT / "policy_drift_intervals.csv", audit_rows)

    curve_sources = [
        (read_csv(LONG_RUN / "learning_curve_eval.csv"), {
            "hybrid_frozen", "hybrid_actor_lr1e4", "hybrid_fixed_teacher_kl002"
        }),
        (read_csv(soft_run / "learning_curve_eval.csv"), {
            "hybrid_soft_kl_selected", "hybrid_soft_l2_selected"
        }),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3))
    for ax, metric, label in zip(
        axes,
        ("hungarian_assignment_distance", "coverage_radius_auc"),
        ("H", "coverage AUC"),
    ):
        for source, methods in curve_sources:
            for method in methods:
                rows = [row for row in source if row["method"] == method]
                groups = {}
                for row in rows:
                    step = int(row["checkpoint"].replace("model_step", ""))
                    groups.setdefault(step, []).append(float(row[metric]))
                xs = np.asarray(sorted(groups))
                ys = np.asarray([np.mean(groups[x]) for x in xs])
                ax.plot(xs, ys, marker="o", label=method)
        ax.set_xscale("log")
        ax.set_xlabel("environment transitions")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2)
    axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(OUTPUT / "soft_anchor_long_horizon.png", dpi=220)
    fig.savefig(OUTPUT / "soft_anchor_long_horizon.pdf")
    plt.close(fig)

    lines = [
        "# Held-Out Soft-Anchor Result", "",
        f"Protocol integrity: **{'PASS' if protocol_pass else 'FAIL'}**.", "",
        "| method | horizon | dH [95% CI] | dAUC [95% CI] | replace fixed? |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    replacement_lookup = {row["method"]: row for row in replacement}
    for row in intervals:
        can_replace = replacement_lookup.get(row["method"], {}).get(
            "can_replace_fixed_backtracking", "n/a"
        )
        lines.append(
            f"| {row['method']} | {row['horizon']} | "
            f"{row['delta_hungarian_assignment_distance_mean']:+.4f} "
            f"[{row['delta_hungarian_assignment_distance_ci95_low']:+.4f}, {row['delta_hungarian_assignment_distance_ci95_high']:+.4f}] | "
            f"{row['delta_coverage_radius_auc_mean']:+.4f} "
            f"[{row['delta_coverage_radius_auc_ci95_low']:+.4f}, {row['delta_coverage_radius_auc_ci95_high']:+.4f}] | "
            f"{can_replace} |"
        )
    (OUTPUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUT / "metadata.json").write_text(json.dumps({
        "protocol_pass": protocol_pass,
        "selection": selection,
        "soft_run": str(soft_run),
        "immutable_control_runs": [str(LONG_RUN), str(STAGE_B_RUN)],
        "statistical_unit": "training seed",
        "replacement_tolerances": {"H": 0.01, "AUC": -0.01},
    }, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
