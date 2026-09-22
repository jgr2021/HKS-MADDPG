"""Evaluate the exact receding-horizon upper bound for local action reranking."""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from benchmarl.hydra_config import reload_experiment_from_file
from scipy.stats import ttest_rel

from evaluate_benchmarl_checkpoint import geometry, summarize
from run_counterfactual_short_horizon_ranker import (
    HORIZONS,
    N_CANDIDATES,
    action_dict,
    branch_outcomes,
    local_candidates,
    make_env,
    ordered_observations,
    policy_action,
    snapshot_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=69_000_000)
    parser.add_argument("--collision-tolerance", type=float, default=0.05)
    return parser.parse_args()


def evaluate(policy, episodes: int, seed_base: int, collision_tolerance: float):
    modes = [("baseline", None), ("oracle_h3", 0), ("oracle_h5", 1)]
    rows = []
    for mode, horizon_index in modes:
        for episode in range(episodes):
            seed = seed_base + episode
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            environment = make_env()
            observations, _ = environment.reset(seed=seed)
            branches = []
            if horizon_index is not None:
                branches = [make_env() for _ in range(N_CANDIDATES)]
                for branch in branches:
                    branch.reset(seed=0)

            step_metrics = []
            episode_return = 0.0
            interventions = 0
            oracle_gains = []
            for step_index in range(25):
                base_action = policy_action(policy, observations)
                action = base_action
                if horizon_index is not None and step_index < 20:
                    candidates = local_candidates(base_action)
                    outcomes = branch_outcomes(
                        branches, snapshot_environment(environment), candidates, policy
                    )
                    values = outcomes["return"][:, horizon_index]
                    collisions = outcomes["collision"][:, horizon_index]
                    safe = collisions <= collisions[0] + collision_tolerance
                    safe_values = np.where(safe, values, -np.inf)
                    best = int(safe_values.argmax())
                    action = candidates[best]
                    interventions += int(best != 0)
                    if best != 0:
                        oracle_gains.append(float(values[best] - values[0]))

                observations, rewards, terminations, truncations, _ = environment.step(
                    action_dict(environment, action)
                )
                episode_return += 2.0 * float(np.mean(list(rewards.values())))
                step_metrics.append(geometry(ordered_observations(observations)))
                if all(terminations.values()) or all(truncations.values()):
                    break

            for branch in branches:
                branch.close()
            environment.close()
            final = step_metrics[-1]
            rows.append(
                {
                    "mode": mode,
                    "episode": episode,
                    "seed": seed,
                    "episode_return_legacy_scale": episode_return,
                    "final_hungarian": final["hungarian"],
                    "final_radius_auc": final["radius_auc"],
                    "collision_step_rate": float(
                        np.mean([item["collision"] for item in step_metrics])
                    ),
                    "final_coverage_at_010": final["coverage_at_010"],
                    "intervention_rate": interventions / len(step_metrics),
                    "mean_oracle_gain": float(np.mean(oracle_gains)) if oracle_gains else 0.0,
                }
            )

    metric_names = [
        "episode_return_legacy_scale",
        "final_hungarian",
        "final_radius_auc",
        "collision_step_rate",
        "final_coverage_at_010",
        "intervention_rate",
        "mean_oracle_gain",
    ]
    grouped = {mode: [row for row in rows if row["mode"] == mode] for mode, _ in modes}
    mode_summary = {
        mode: {
            metric: summarize(np.asarray([float(row[metric]) for row in mode_rows]))
            for metric in metric_names
        }
        for mode, mode_rows in grouped.items()
    }
    directions = {
        "episode_return_legacy_scale": 1.0,
        "final_hungarian": -1.0,
        "final_radius_auc": 1.0,
        "collision_step_rate": -1.0,
        "final_coverage_at_010": 1.0,
    }
    baseline = grouped["baseline"]
    paired = {}
    for mode in ("oracle_h3", "oracle_h5"):
        paired[mode] = {}
        for metric, direction in directions.items():
            baseline_values = np.asarray([float(row[metric]) for row in baseline])
            mode_values = np.asarray([float(row[metric]) for row in grouped[mode]])
            improvement = direction * (mode_values - baseline_values)
            test = ttest_rel(mode_values, baseline_values)
            paired[mode][metric] = {
                "improvement_positive_is_better": summarize(improvement),
                "paired_ttest_two_sided_p": float(test.pvalue) if np.isfinite(test.pvalue) else None,
            }
    return rows, {"modes": mode_summary, "paired_vs_baseline": paired}


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    experiment = reload_experiment_from_file(str(checkpoint))
    try:
        policy = experiment.policy.to("cpu").eval()
        rows, evaluation = evaluate(
            policy, args.episodes, args.seed_base, args.collision_tolerance
        )
    finally:
        experiment.close()

    with (output_dir / "per_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "episodes": args.episodes,
        "seed_base": args.seed_base,
        "collision_tolerance": args.collision_tolerance,
        "note": "Exact simulator upper bound; not a deployable learned policy",
        "evaluation": evaluation,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
