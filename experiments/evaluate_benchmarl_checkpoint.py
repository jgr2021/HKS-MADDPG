"""Evaluate a BenchMARL MPE2 checkpoint with the repository's geometry metrics."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from benchmarl.hydra_config import reload_experiment_from_file
from torchrl.envs.utils import ExplorationType, set_exploration_type


RADII = np.linspace(0.05, 0.30, 26, dtype=np.float64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=65_000_000)
    return parser.parse_args()


def reconstruct_positions(agent_observations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover all positions from agent_0's full-observability vector."""
    obs0 = agent_observations[0]
    self_position = obs0[2:4]
    landmarks = self_position + obs0[4:10].reshape(3, 2)
    other_agents = self_position + obs0[10:14].reshape(2, 2)
    agents = np.concatenate([self_position[None, :], other_agents], axis=0)
    return agents.astype(np.float64), landmarks.astype(np.float64)


def geometry(agent_observations: np.ndarray) -> dict[str, float]:
    agents, landmarks = reconstruct_positions(agent_observations)
    distances = np.linalg.norm(agents[:, None, :] - landmarks[None, :, :], axis=-1)
    assignment_cost = min(
        sum(distances[agent, landmark] for agent, landmark in enumerate(permutation))
        for permutation in itertools.permutations(range(3))
    ) / 3.0
    nearest_landmark_distances = distances.min(axis=0)
    radius_coverage = np.asarray(
        [(nearest_landmark_distances < radius).mean() for radius in RADII], dtype=np.float64
    )
    radius_auc = float(np.trapz(radius_coverage, RADII) / (RADII[-1] - RADII[0]))

    pair_distances = np.asarray(
        [np.linalg.norm(agents[i] - agents[j]) for i in range(3) for j in range(i + 1, 3)],
        dtype=np.float64,
    )
    return {
        "hungarian": float(assignment_cost),
        "radius_auc": radius_auc,
        "collision": float(np.any(pair_distances < 0.30)),
        "min_pair_separation": float(pair_distances.min()),
        "coverage_at_010": float((nearest_landmark_distances < 0.10).sum()),
    }


def summarize(values: np.ndarray) -> dict[str, float]:
    count = int(values.size)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if count > 1 else 0.0
    sem = std / math.sqrt(count) if count else float("nan")
    return {
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
    }


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.episodes <= 0 or args.horizon <= 0:
        raise ValueError("--episodes and --horizon must be positive")

    experiment = reload_experiment_from_file(str(checkpoint))
    rows: list[dict[str, float | int]] = []
    try:
        policy = experiment.policy.to("cpu")
        env = experiment.test_env
        policy.eval()
        for episode in range(args.episodes):
            seed = args.seed_base + episode
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env.set_seed(seed)
            with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
                rollout = env.rollout(max_steps=args.horizon, policy=policy)

            observations = rollout.get(("next", "agent", "observation")).detach().cpu().numpy()
            rewards = rollout.get(("next", "agent", "reward")).detach().cpu().numpy()
            step_metrics = [geometry(observations[step]) for step in range(observations.shape[0])]
            final = step_metrics[-1]
            rows.append(
                {
                    "episode": episode,
                    "seed": seed,
                    "steps": observations.shape[0],
                    # local_ratio=0.5: 2 * agent-mean reward restores the legacy
                    # global-coverage + mean-local-collision reward convention.
                    "episode_return_legacy_scale": float(2.0 * rewards.mean(axis=1).sum()),
                    "final_hungarian": final["hungarian"],
                    "final_radius_auc": final["radius_auc"],
                    "collision_step_rate": float(np.mean([item["collision"] for item in step_metrics])),
                    "minimum_pair_separation": float(
                        min(item["min_pair_separation"] for item in step_metrics)
                    ),
                    "final_coverage_at_010": final["coverage_at_010"],
                    "max_coverage_at_010": float(
                        max(item["coverage_at_010"] for item in step_metrics)
                    ),
                }
            )
    finally:
        experiment.close()

    fieldnames = list(rows[0])
    with (output_dir / "per_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [name for name in fieldnames if name not in {"episode", "seed", "steps"}]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "episodes": args.episodes,
        "horizon": args.horizon,
        "seed_base": args.seed_base,
        "reward_note": "2 * mean MPE2 local_ratio=0.5 reward; legacy global + mean local scale",
        "metrics": {
            name: summarize(np.asarray([float(row[name]) for row in rows], dtype=np.float64))
            for name in metric_names
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
