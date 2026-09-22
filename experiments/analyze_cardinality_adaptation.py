"""Aggregate actor-level cardinality adaptation effects and policy drift."""

from __future__ import annotations

import csv
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DISTILL_ROOT = Path("experiments/cardinality_distillations_20260811")
ONLINE_BATCH_ROOT = Path("experiments/cardinality_online_batch_20260811")
OUTPUT_ROOT = Path("experiments/cardinality_adaptation_analysis_20260811")
DRIFT_OUTPUT_NAME = "cardinality_policy_drift"
DRIFT_QUARTILE_MODE = "threshold"
METHODS = (
    "cardinality_frozen",
    "cardinality_first_update_then_freeze",
    "cardinality_unanchored",
    "cardinality_fixed_teacher_kl002",
)
METRICS = (
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "minimum_agent_separation",
    "return",
)
T_CRITICAL_975 = {
    1: 12.7062, 2: 4.3027, 3: 3.1824, 4: 2.7764,
    5: 2.5706, 6: 2.4469, 7: 2.3646, 8: 2.3060,
    9: 2.2622, 10: 2.2281,
}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_run_dir(n_agents, replicate):
    parent = Path(
        f"experiments/cardinality_online_n{n_agents}_actor{replicate}_20260811"
    )
    candidates = sorted(parent.glob("run_*")) if parent.exists() else []
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one immutable online run for N={n_agents}, "
            f"actor={replicate}; found {candidates}"
        )
    return candidates[0]


def ensure_drift(run_dir, n_agents, replicate):
    output = run_dir / DRIFT_OUTPUT_NAME / "policy_drift.csv"
    if output.exists():
        return output
    seed_base = 30_000_000 + n_agents * 1_000_000 + replicate * 100_000
    subprocess.run([
        sys.executable,
        "experiments/analyze_cardinality_policy_drift.py",
        "--run-dir", str(run_dir),
        "--episodes", "100",
        "--horizon", "25",
        "--seed-base", str(seed_base),
        "--output-name", DRIFT_OUTPUT_NAME,
        "--quartile-mode", DRIFT_QUARTILE_MODE,
    ], check=True)
    return output


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return float(values.mean()), float("nan"), float("nan"), float("nan")
    critical = T_CRITICAL_975.get(values.size - 1, 1.96)
    half = critical * values.std(ddof=1) / math.sqrt(values.size)
    return (
        float(values.mean()), float(values.std(ddof=1)),
        float(values.mean() - half), float(values.mean() + half),
    )


def collect():
    selection = json.loads(
        (ONLINE_BATCH_ROOT / "payload_selection.json").read_text(encoding="utf-8")
    )
    eligible = {
        (int(n_agents), int(replicate))
        for n_agents, replicates in selection.items()
        for replicate in replicates
    }
    actor_curve_rows, actor_final_rows, drift_rows, offline_rows = [], [], [], []
    integrity = []
    for n_agents, replicate in sorted(eligible):
        run_dir = find_run_dir(n_agents, replicate)
        offline_metadata = json.loads(
            (DISTILL_ROOT / f"n{n_agents}_actor{replicate}" / "metadata.json")
            .read_text(encoding="utf-8")
        )
        offline_summary = {
            row["controller"]: row for row in read_csv(
                DISTILL_ROOT / f"n{n_agents}_actor{replicate}" / "summary.csv"
            )
        }
        offline_rows.append({
            "n_agents": n_agents,
            "replicate": replicate,
            "offline_gate_eligible": bool(
                offline_metadata["offline_gate"]["eligible"]
            ),
            "validation_accuracy": offline_metadata["final"][
                "validation_accuracy"
            ],
            "teacher_h": float(offline_summary["teacher"][
                "hungarian_assignment_distance_mean"
            ]),
            "student_h": float(offline_summary["student"][
                "hungarian_assignment_distance_mean"
            ]),
            "teacher_auc": float(offline_summary["teacher"][
                "coverage_radius_auc_mean"
            ]),
            "student_auc": float(offline_summary["student"][
                "coverage_radius_auc_mean"
            ]),
            "teacher_collision": float(offline_summary["teacher"][
                "collision_step_rate_mean"
            ]),
            "student_collision": float(offline_summary["student"][
                "collision_step_rate_mean"
            ]),
        })
        protocol = json.loads(
            (run_dir / "protocol_metadata.json").read_text(encoding="utf-8")
        )
        passed = (
            protocol["methods"] == list(METHODS)
            and protocol["checkpoint_steps"]
            == [64, 100, 1000, 5000, 10000, 20000]
            and protocol["total_env_steps"] == 20000
            and protocol["eval_episodes_final"] == 300
            and protocol["eval_episodes_checkpoint"] == 100
        )
        summaries = read_csv(run_dir / "seed_summaries.csv")
        passed = passed and len(summaries) == 4 and all(
            row["status"] == "completed" for row in summaries
        )
        integrity.append({
            "n_agents": n_agents,
            "replicate": replicate,
            "run_dir": str(run_dir),
            "protocol_pass": passed,
            "offline_gate_eligible": bool(
                offline_metadata["offline_gate"]["eligible"]
            ),
            "offline_validation_accuracy": offline_metadata[
                "final"
            ]["validation_accuracy"],
        })
        if not passed:
            continue
        curves = read_csv(run_dir / "learning_curve_eval.csv")
        by_curve = {
            (row["method"], row["checkpoint"]): row for row in curves
        }
        checkpoints = [64, 100, 1000, 5000, 10000, 20000]
        for step in checkpoints:
            frozen = by_curve[("cardinality_frozen", f"model_step{step}")]
            for method in METHODS:
                row = by_curve[(method, f"model_step{step}")]
                output = {
                    "n_agents": n_agents,
                    "replicate": replicate,
                    "method": method,
                    "checkpoint_step": step,
                }
                for metric in METRICS:
                    output[metric] = float(row[metric])
                    output["delta_" + metric] = (
                        float(row[metric]) - float(frozen[metric])
                    )
                actor_curve_rows.append(output)
        by_final = {row["method"]: row for row in summaries}
        frozen = by_final["cardinality_frozen"]
        for method in METHODS:
            row = by_final[method]
            output = {
                "n_agents": n_agents,
                "replicate": replicate,
                "method": method,
            }
            for metric in METRICS:
                output[metric] = float(row[metric])
                output["delta_" + metric] = (
                    float(row[metric]) - float(frozen[metric])
                )
            actor_final_rows.append(output)
        drift_path = ensure_drift(run_dir, n_agents, replicate)
        for row in read_csv(drift_path):
            drift_rows.append({
                "n_agents": n_agents,
                "replicate": replicate,
                **row,
            })
    return (
        eligible, integrity, offline_rows,
        actor_curve_rows, actor_final_rows, drift_rows,
    )


def summarize_effects(actor_rows, has_checkpoint):
    rows = []
    groups = {}
    for row in actor_rows:
        key = [int(row["n_agents"]), row["method"]]
        if has_checkpoint:
            key.append(int(row["checkpoint_step"]))
        groups.setdefault(tuple(key), []).append(row)
    for key, selected in sorted(groups.items()):
        output = {
            "n_agents": key[0],
            "method": key[1],
            "n_actors": len(selected),
        }
        if has_checkpoint:
            output["checkpoint_step"] = key[2]
        for metric in METRICS:
            mean, sd, low, high = interval([
                float(row["delta_" + metric]) for row in selected
            ])
            output["delta_" + metric + "_mean"] = mean
            output["delta_" + metric + "_sd"] = sd
            output["delta_" + metric + "_ci95_low"] = low
            output["delta_" + metric + "_ci95_high"] = high
        rows.append(output)
    return rows


def summarize_drift(rows):
    metrics = [
        "initial_reference_kl_mean", "initial_action_agreement",
        "pinsker_argmax_certified_fraction",
        "agreement_initial_margin_q1", "agreement_initial_margin_q2",
        "agreement_initial_margin_q3", "agreement_initial_margin_q4",
    ]
    groups = {}
    for row in rows:
        key = (
            int(row["n_agents"]), row["method"],
            int(row["checkpoint_step"]),
        )
        groups.setdefault(key, []).append(row)
    output_rows = []
    for key, selected in sorted(groups.items()):
        output = {
            "n_agents": key[0], "method": key[1],
            "checkpoint_step": key[2], "n_actors": len(selected),
        }
        for metric in metrics:
            mean, sd, low, high = interval([
                float(row[metric]) for row in selected
            ])
            output[metric + "_mean"] = mean
            output[metric + "_sd"] = sd
            output[metric + "_ci95_low"] = low
            output[metric + "_ci95_high"] = high
        output_rows.append(output)
    return output_rows


def useful_windows(curve_summary):
    by_key = {
        (int(row["n_agents"]), row["method"], int(row["checkpoint_step"])): row
        for row in curve_summary
    }
    rows = []
    for n_agents in (3, 4, 5, 6):
        useful = []
        for step in (100, 1000, 5000, 10000, 20000):
            row = by_key[(n_agents, "cardinality_unanchored", step)]
            safety_adverse = (
                row["delta_collision_step_rate_ci95_low"] > 0.0
                or row["delta_minimum_agent_separation_ci95_high"] < 0.0
            )
            is_useful = (
                row["delta_hungarian_assignment_distance_mean"] < 0.0
                and row["delta_coverage_radius_auc_mean"] > 0.0
                and not safety_adverse
            )
            if is_useful:
                useful.append(step)
        rows.append({
            "n_agents": n_agents,
            "tested_actor_payloads": by_key[(
                n_agents, "cardinality_unanchored", 100
            )]["n_actors"],
            "first_event_useful": 100 in useful,
            "latest_useful_checkpoint": max(useful) if useful else "none",
            "useful_checkpoints": ",".join(str(value) for value in useful),
        })
    return rows


def plot_curves(summary):
    selected = [
        row for row in summary if row["method"] == "cardinality_unanchored"
        and int(row["checkpoint_step"]) >= 100
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharex=True)
    specs = [
        ("hungarian_assignment_distance", "Delta H (lower is better)"),
        ("coverage_radius_auc", "Delta AUC (higher is better)"),
    ]
    for ax, (metric, label) in zip(axes, specs):
        for n_agents in (3, 4, 5, 6):
            rows = sorted(
                [row for row in selected if int(row["n_agents"]) == n_agents],
                key=lambda row: int(row["checkpoint_step"]),
            )
            xs = np.asarray([int(row["checkpoint_step"]) for row in rows])
            ys = np.asarray([row[f"delta_{metric}_mean"] for row in rows])
            low = np.asarray([row[f"delta_{metric}_ci95_low"] for row in rows])
            high = np.asarray([row[f"delta_{metric}_ci95_high"] for row in rows])
            ax.plot(xs, ys, marker="o", label=f"N={n_agents}")
            ax.fill_between(xs, low, high, alpha=0.13)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("environment transitions")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.2)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "cardinality_unanchored_curves.png", dpi=220)
    fig.savefig(OUTPUT_ROOT / "cardinality_unanchored_curves.pdf")
    plt.close(fig)


def make_readme(window_rows, curve_summary, final_summary, integrity):
    by_curve = {
        (int(row["n_agents"]), row["method"], int(row["checkpoint_step"])): row
        for row in curve_summary
    }
    by_final = {
        (int(row["n_agents"]), row["method"]): row for row in final_summary
    }
    lines = [
        "# Cardinality Adaptation Result", "",
        "The independently distilled actor is the inferential unit. Intervals are paired 95% t intervals over the first three intention-to-treat actors; episode evaluations are matched measurement replicates.", "",
        f"Protocol integrity: **{'PASS' if all(row['protocol_pass'] for row in integrity) else 'FAIL'}**.", "",
        "Cardinality Amendment A1 runs the first three predeclared payloads for every N regardless of the original offline gate. The online trend is exploratory and avoids outcome-conditioned payload replacement.", "",
        "| N | actors | first-event dH [95% CI] | first-event dAUC [95% CI] | latest useful checkpoint | unanchored 20k dAUC | fixed 20k dAUC |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for window in window_rows:
        n_agents = int(window["n_agents"])
        first = by_curve[(n_agents, "cardinality_unanchored", 100)]
        unanchored = by_final[(n_agents, "cardinality_unanchored")]
        fixed = by_final[(n_agents, "cardinality_fixed_teacher_kl002")]
        lines.append(
            f"| {n_agents} | {first['n_actors']} | "
            f"{first['delta_hungarian_assignment_distance_mean']:+.4f} "
            f"[{first['delta_hungarian_assignment_distance_ci95_low']:+.4f}, {first['delta_hungarian_assignment_distance_ci95_high']:+.4f}] | "
            f"{first['delta_coverage_radius_auc_mean']:+.4f} "
            f"[{first['delta_coverage_radius_auc_ci95_low']:+.4f}, {first['delta_coverage_radius_auc_ci95_high']:+.4f}] | "
            f"{window['latest_useful_checkpoint']} | "
            f"{unanchored['delta_coverage_radius_auc_mean']:+.4f} | "
            f"{fixed['delta_coverage_radius_auc_mean']:+.4f} |"
        )
    endpoints = [
        None if row["latest_useful_checkpoint"] == "none"
        else int(row["latest_useful_checkpoint"])
        for row in window_rows
    ]
    monotonic = all(
        right is not None and left is not None and right <= left
        for left, right in zip(endpoints, endpoints[1:])
    )
    if monotonic:
        conclusion = "The tested useful-window endpoints are non-increasing with team size."
    elif any(value is not None for value in endpoints):
        conclusion = "The cardinality response is graded or non-monotonic; a monotonic shrinking-window claim is not supported."
    else:
        conclusion = "No useful unanchored checkpoint is resolved in the unified actor family; the shrinking-window hypothesis is inconclusive."
    lines += ["", "## Locked conclusion", "", conclusion, ""]
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    global OUTPUT_ROOT, DRIFT_OUTPUT_NAME, DRIFT_QUARTILE_MODE
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="experiments/cardinality_adaptation_analysis_20260811",
    )
    parser.add_argument(
        "--drift-output-name", default="cardinality_policy_drift"
    )
    parser.add_argument(
        "--drift-quartile-mode", choices=("threshold", "rank"),
        default="threshold",
    )
    args = parser.parse_args()
    OUTPUT_ROOT = Path(args.output_dir)
    DRIFT_OUTPUT_NAME = args.drift_output_name
    DRIFT_QUARTILE_MODE = args.drift_quartile_mode
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=False)
    (
        eligible, integrity, offline_rows,
        actor_curves, actor_finals, drift,
    ) = collect()
    curve_summary = summarize_effects(actor_curves, has_checkpoint=True)
    final_summary = summarize_effects(actor_finals, has_checkpoint=False)
    drift_summary = summarize_drift(drift)
    window_rows = useful_windows(curve_summary)
    write_csv(OUTPUT_ROOT / "protocol_integrity.csv", integrity)
    write_csv(OUTPUT_ROOT / "offline_payloads.csv", offline_rows)
    write_csv(OUTPUT_ROOT / "per_actor_checkpoint_deltas.csv", actor_curves)
    write_csv(OUTPUT_ROOT / "per_actor_final_deltas.csv", actor_finals)
    write_csv(OUTPUT_ROOT / "checkpoint_actor_intervals.csv", curve_summary)
    write_csv(OUTPUT_ROOT / "final_actor_intervals.csv", final_summary)
    write_csv(OUTPUT_ROOT / "per_actor_policy_drift.csv", drift)
    write_csv(OUTPUT_ROOT / "policy_drift_actor_intervals.csv", drift_summary)
    write_csv(OUTPUT_ROOT / "useful_windows.csv", window_rows)
    plot_curves(curve_summary)
    make_readme(window_rows, curve_summary, final_summary, integrity)
    (OUTPUT_ROOT / "metadata.json").write_text(json.dumps({
        "selected_intention_to_treat_payloads": sorted(
            [list(value) for value in eligible]
        ),
        "statistical_unit": "independently distilled actor",
        "interval": "paired 95% Student-t interval across actors",
        "protocol_integrity_pass": all(
            row["protocol_pass"] for row in integrity
        ),
        "confirmatory": False,
        "policy_drift_output_name": DRIFT_OUTPUT_NAME,
        "policy_drift_quartile_mode": DRIFT_QUARTILE_MODE,
        "amendment": "A1 intention-to-treat after partial offline, before formal online",
    }, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT_ROOT)


if __name__ == "__main__":
    main()
