"""Matched evaluation and locked gate analysis for optimizer-control Stage A."""

import argparse
import csv
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_vector_signal_gsp_experiment import evaluate_model


CHECKPOINTS = (64, 100, 1000, 4996, 5000)
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
DELAYED = "hybrid_delayed_actor5k"
FIRST_UPDATE_HUNGARIAN_THRESHOLD = 0.10
FIRST_UPDATE_AUC_THRESHOLD = -0.10
FROZEN_AUDIT_KL_THRESHOLD = 1e-5


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def paired_interval(reference, candidate):
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    half = 1.9647293909876649 * delta.std(ddof=1) / math.sqrt(len(delta))
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(delta.mean() - half),
        "ci95_high": float(delta.mean() + half),
        "fraction_positive": float((delta > 0).mean()),
    }


def read_last_audit_kl(seed_dir):
    path = seed_dir / "tensorboard_summary.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = []
    for key, series in payload.items():
        if key.endswith("initial_reference_kl") and series:
            values.append(float(series[-1][2]))
    if not values:
        return float("nan")
    return float(np.mean(values))


def evaluate_checkpoint(method, step, checkpoint, episodes, seed):
    import torch

    torch.set_num_threads(1)
    evaluated = evaluate_model(
        "simple_spread",
        checkpoint,
        episodes,
        25,
        seed,
    )
    evaluated = sorted(evaluated, key=lambda row: row["test_seed"])
    return method, step, evaluated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed-base", type=int, default=5_300_000)
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output = run_dir / "matched_stage_a_analysis"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((run_dir / "protocol_metadata.json").read_text(encoding="utf-8"))
    methods = metadata["methods"]

    episode_rows = []
    summaries = []
    audit_rows = []
    by_method_checkpoint = {}
    for method in methods:
        seed_dir = run_dir / method / f"seed_{args.seed}"
        audit_rows.append({
            "method": method,
            "final_initial_reference_kl": read_last_audit_kl(seed_dir),
        })

    evaluation_seed = args.seed_base + args.seed * 10_000
    futures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for method in methods:
            seed_dir = run_dir / method / f"seed_{args.seed}"
            for step in CHECKPOINTS:
                checkpoint = seed_dir / "checkpoints" / f"model_step{step}.pt"
                if not checkpoint.exists():
                    raise FileNotFoundError(checkpoint)
                future = executor.submit(
                    evaluate_checkpoint,
                    method,
                    step,
                    str(checkpoint),
                    args.episodes,
                    evaluation_seed,
                )
                futures[future] = (method, step)
        completed = 0
        for future in as_completed(futures):
            method, step, evaluated = future.result()
            by_method_checkpoint[(method, step)] = evaluated
            completed += 1
            print(
                f"completed {completed}/{len(futures)}: {method} step {step}",
                flush=True,
            )

    for method in methods:
        for step in CHECKPOINTS:
            evaluated = by_method_checkpoint[(method, step)]
            for row in evaluated:
                episode_rows.append({
                    **row,
                    "method": method,
                    "checkpoint_step": step,
                })
            summary = {
                "method": method,
                "checkpoint_step": step,
                "episodes": len(evaluated),
            }
            for metric in METRICS:
                values = np.asarray([row[metric] for row in evaluated], dtype=np.float64)
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_std"] = float(values.std(ddof=1))
            summaries.append(summary)

    comparisons = []
    for method in methods:
        comparison_specs = [
            ("first_update", (method, 64), (method, 100)),
            ("final_vs_frozen", (FROZEN, 5000), (method, 5000)),
        ]
        if method == DELAYED:
            comparison_specs.append(
                ("delayed_first_update", (method, 4996), (method, 5000))
            )
        for comparison, reference_key, candidate_key in comparison_specs:
            reference = by_method_checkpoint[reference_key]
            candidate = by_method_checkpoint[candidate_key]
            for metric in METRICS:
                result = paired_interval(
                    [row[metric] for row in reference],
                    [row[metric] for row in candidate],
                )
                comparisons.append({
                    "method": method,
                    "comparison": comparison,
                    "metric": metric,
                    **result,
                })

    write_csv(output / "per_episode.csv", episode_rows)
    write_csv(output / "checkpoint_summary.csv", summaries)
    write_csv(output / "paired_intervals.csv", comparisons)
    write_csv(output / "policy_audit_kl.csv", audit_rows)

    comparison_lookup = {
        (row["method"], row["comparison"], row["metric"]): row
        for row in comparisons
    }
    audit_lookup = {
        row["method"]: row["final_initial_reference_kl"] for row in audit_rows
    }
    decisions = []
    for method in methods:
        final_h = comparison_lookup[
            (method, "final_vs_frozen", "hungarian_assignment_distance")
        ]
        final_auc = comparison_lookup[
            (method, "final_vs_frozen", "coverage_radius_auc")
        ]
        final_collision = comparison_lookup[
            (method, "final_vs_frozen", "collision_step_rate")
        ]
        first_h = comparison_lookup[
            (method, "first_update", "hungarian_assignment_distance")
        ]
        first_auc = comparison_lookup[
            (method, "first_update", "coverage_radius_auc")
        ]
        no_resolved_hungarian_degradation = final_h["ci95_low"] <= 0.0
        no_resolved_auc_degradation = final_auc["ci95_high"] >= 0.0
        no_resolved_collision_penalty = final_collision["ci95_low"] <= 0.0
        no_catastrophic_first_update = not (
            first_h["ci95_low"] > FIRST_UPDATE_HUNGARIAN_THRESHOLD
            or first_auc["ci95_high"] < FIRST_UPDATE_AUC_THRESHOLD
        )
        nonfrozen_drift = (
            method == FROZEN
            or audit_lookup[method] > FROZEN_AUDIT_KL_THRESHOLD
        )
        eligible = (
            no_resolved_hungarian_degradation
            and no_resolved_auc_degradation
            and no_resolved_collision_penalty
            and no_catastrophic_first_update
            and nonfrozen_drift
        )
        decisions.append({
            "method": method,
            "eligible": eligible,
            "no_resolved_hungarian_degradation": no_resolved_hungarian_degradation,
            "no_resolved_auc_degradation": no_resolved_auc_degradation,
            "no_resolved_collision_penalty": no_resolved_collision_penalty,
            "no_catastrophic_first_update": no_catastrophic_first_update,
            "nonfrozen_policy_drift": nonfrozen_drift,
            "final_initial_reference_kl": audit_lookup[method],
        })
    write_csv(output / "stage_a_decisions.csv", decisions)

    lines = [
        "# Hybrid Optimizer Controls: Stage-A Matched Analysis",
        "",
        f"Training seed: `{args.seed}`. Matched evaluation episodes per checkpoint: `{args.episodes}`.",
        "",
        "| method | eligible | final audit KL | H no-degrade | AUC no-degrade | collision no-penalty | first-update safe | non-frozen drift |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in decisions:
        lines.append(
            f"| {row['method']} | {'PASS' if row['eligible'] else 'FAIL'} | "
            f"{row['final_initial_reference_kl']:.6g} | "
            f"{'yes' if row['no_resolved_hungarian_degradation'] else 'no'} | "
            f"{'yes' if row['no_resolved_auc_degradation'] else 'no'} | "
            f"{'yes' if row['no_resolved_collision_penalty'] else 'no'} | "
            f"{'yes' if row['no_catastrophic_first_update'] else 'no'} | "
            f"{'yes' if row['nonfrozen_policy_drift'] else 'no'} |"
        )
    lines += [
        "",
        "Eligibility applies the locked protocol gates mechanically. Selection for",
        "fresh-seed confirmation must additionally compare the joint effect sizes",
        "among eligible non-frozen controls; this script does not retune settings.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps({
        "run_dir": str(run_dir),
        "seed": args.seed,
        "episodes": args.episodes,
        "seed_base": args.seed_base,
        "workers": args.workers,
        "checkpoints": list(CHECKPOINTS),
        "first_update_hungarian_threshold": FIRST_UPDATE_HUNGARIAN_THRESHOLD,
        "first_update_auc_threshold": FIRST_UPDATE_AUC_THRESHOLD,
        "frozen_audit_kl_threshold": FROZEN_AUDIT_KL_THRESHOLD,
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
