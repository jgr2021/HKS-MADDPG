"""Locked speed/quality screen for the 6x6 actor's Sinkhorn iterations."""

import argparse
import csv
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.distill_scalable_6x6_actor import evaluate
from utils.networks import EquivariantMatchingSafety6x6Policy


ITERATIONS = (8, 16, 32, 64)
METRICS = (
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "return",
    "minimum_agent_separation",
    "final_coverage",
    "max_coverage",
)
HUNGARIAN_TOLERANCE = 0.01
AUC_TOLERANCE = 0.02
COLLISION_TOLERANCE = 0.01


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def policies_for(state, iterations):
    policies = []
    for agent_index in range(6):
        policy = EquivariantMatchingSafety6x6Policy(
            36, 5, agent_index=agent_index
        )
        policy.load_state_dict(state)
        policy.sinkhorn_iterations = int(iterations)
        policy.eval()
        policies.append(policy)
    return policies


def dataset_predictions(state, observations, agent_indices, iterations):
    policy = EquivariantMatchingSafety6x6Policy(36, 5, agent_index=0)
    policy.load_state_dict(state)
    policy.sinkhorn_iterations = int(iterations)
    policy.eval()
    predictions = []
    row_errors = []
    with torch.no_grad():
        for start in range(0, len(observations), 1024):
            stop = start + 1024
            logits = policy.forward_with_agent_indices(
                observations[start:stop], agent_indices[start:stop]
            )
            predictions.append(logits.argmax(1).cpu().numpy())
            row_errors.append(float(policy.last_matching_row_error))
    return np.concatenate(predictions), float(max(row_errors))


def paired_interval(reference, candidate):
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    half = 1.9647293909876649 * delta.std(ddof=1) / math.sqrt(len(delta))
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(delta.mean() - half),
        "ci95_high": float(delta.mean() + half),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--payload",
        default="",
        help="Optional single-payload override for smoke tests.",
    )
    parser.add_argument(
        "--payloads",
        default=",".join((
            "experiments/equivariant_6x6_sinkhorn64_seed1_20260713/"
            "equivariant_6x6_actor.pt",
            "experiments/equivariant_6x6_sinkhorn64_seed2_20260713/"
            "equivariant_6x6_actor.pt",
            "experiments/equivariant_6x6_sinkhorn64_seed3_20260713/"
            "equivariant_6x6_actor.pt",
        )),
    )
    parser.add_argument(
        "--dataset",
        default=(
            "experiments/equivariant_6x6_sinkhorn64_seed1_20260713/"
            "training_dataset.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/equivariant_6x6_sinkhorn_screen_20260716",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed-base", type=int, default=7_200_000)
    args = parser.parse_args()
    if args.episodes < 2:
        raise ValueError("--episodes must be at least 2 for paired intervals")

    torch.set_num_threads(6)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    payload_paths = (
        [args.payload]
        if args.payload
        else [
            item.strip()
            for item in args.payloads.split(",")
            if item.strip()
        ]
    )
    if not payload_paths:
        raise ValueError("At least one payload is required")
    with np.load(args.dataset) as loaded:
        observations = torch.from_numpy(
            loaded["observations"].astype(np.float32)
        )
        agent_indices = torch.from_numpy(
            loaded["agent_indices"].astype(np.int64)
        )

    episode_rows = []
    summaries = []
    comparisons = []
    by_payload_iteration = {}
    payload_ids = []
    for payload_index, payload_path in enumerate(payload_paths):
        match = re.search(r"sinkhorn64_seed(\d+)", payload_path)
        payload_id = (
            f"seed{match.group(1)}" if match else f"payload{payload_index + 1}"
        )
        payload_ids.append(payload_id)
        payload = torch.load(payload_path, map_location="cpu")
        state = payload["actor_state_dict"]

        predictions = {}
        row_errors = {}
        for iterations in ITERATIONS:
            predictions[iterations], row_errors[iterations] = (
                dataset_predictions(
                    state, observations, agent_indices, iterations
                )
            )
        reference_predictions = predictions[64]

        for iterations in ITERATIONS:
            policies = policies_for(state, iterations)
            start = time.perf_counter()
            rows = evaluate(
                "student",
                policies,
                args.episodes,
                25,
                args.seed_base + payload_index * 10_000,
            )
            elapsed = time.perf_counter() - start
            rows = sorted(rows, key=lambda row: row["test_seed"])
            by_payload_iteration[(payload_id, iterations)] = rows
            episode_rows.extend(
                {
                    **row,
                    "payload_id": payload_id,
                    "payload_path": payload_path,
                    "sinkhorn_iterations": iterations,
                }
                for row in rows
            )
            summary = {
                "payload_id": payload_id,
                "payload_path": payload_path,
                "sinkhorn_iterations": iterations,
                "episodes": len(rows),
                "wall_time_sec": elapsed,
                "episodes_per_sec": len(rows) / elapsed,
                "action_agreement_with_64": float(
                    (
                        predictions[iterations]
                        == reference_predictions
                    ).mean()
                ),
                "max_matching_row_error": row_errors[iterations],
            }
            for metric in METRICS:
                values = np.asarray(
                    [row[metric] for row in rows], dtype=np.float64
                )
                summary[f"{metric}_mean"] = float(values.mean())
                summary[f"{metric}_std"] = float(values.std(ddof=1))
            summaries.append(summary)

        reference = by_payload_iteration[(payload_id, 64)]
        for iterations in ITERATIONS:
            candidate = by_payload_iteration[(payload_id, iterations)]
            for metric in METRICS:
                comparisons.append({
                    "payload_id": payload_id,
                    "payload_path": payload_path,
                    "sinkhorn_iterations": iterations,
                    "comparison": "candidate_minus_64",
                    "metric": metric,
                    **paired_interval(
                        [row[metric] for row in reference],
                        [row[metric] for row in candidate],
                    ),
                })

    comparison_lookup = {
        (
            row["payload_id"],
            row["sinkhorn_iterations"],
            row["metric"],
        ): row
        for row in comparisons
    }
    summary_lookup = {
        (row["payload_id"], row["sinkhorn_iterations"]): row
        for row in summaries
    }
    decisions = []
    for iterations in ITERATIONS:
        payload_passes = []
        selected_summaries = []
        for payload_id in payload_ids:
            h = comparison_lookup[
                (
                    payload_id,
                    iterations,
                    "hungarian_assignment_distance",
                )
            ]
            auc = comparison_lookup[
                (payload_id, iterations, "coverage_radius_auc")
            ]
            collision = comparison_lookup[
                (payload_id, iterations, "collision_step_rate")
            ]
            payload_passes.append(
                h["ci95_low"] <= HUNGARIAN_TOLERANCE
                and auc["ci95_high"] >= -AUC_TOLERANCE
                and collision["ci95_low"] <= COLLISION_TOLERANCE
            )
            selected_summaries.append(
                summary_lookup[(payload_id, iterations)]
            )
        eligible = all(payload_passes)
        decisions.append({
            "sinkhorn_iterations": iterations,
            "eligible": eligible,
            "eligible_payloads": int(sum(payload_passes)),
            "total_payloads": len(payload_passes),
            "minimum_action_agreement_with_64": min(
                row["action_agreement_with_64"]
                for row in selected_summaries
            ),
            "maximum_matching_row_error": max(
                row["max_matching_row_error"]
                for row in selected_summaries
            ),
            "mean_episodes_per_sec": float(np.mean([
                row["episodes_per_sec"] for row in selected_summaries
            ])),
        })

    write_csv(output / "per_episode.csv", episode_rows)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "paired_intervals.csv", comparisons)
    write_csv(output / "decisions.csv", decisions)
    selected = min(
        row["sinkhorn_iterations"] for row in decisions if row["eligible"]
    )
    (output / "metadata.json").write_text(
        json.dumps({
            "payloads": payload_paths,
            "payload_ids": payload_ids,
            "dataset": args.dataset,
            "episodes": args.episodes,
            "seed_base": args.seed_base,
            "iterations": list(ITERATIONS),
            "selected_iterations": selected,
            "gate": {
                "hungarian_tolerance": HUNGARIAN_TOLERANCE,
                "auc_tolerance": AUC_TOLERANCE,
                "collision_tolerance": COLLISION_TOLERANCE,
            },
        }, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
