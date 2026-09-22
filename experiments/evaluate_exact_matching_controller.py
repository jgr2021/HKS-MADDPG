"""Transparent decentralized exact-matching candidate-action controller."""

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.evaluate_supervised_probe_policies import geometry
from utils.make_env import make_env
from utils.vector_signal_gsp_features import candidate_geometry_from_local_obs_batch


PERMUTATIONS = tuple(itertools.permutations(range(3)))


def exact_matching_actions(obs, collision_safe=False):
    actions = []
    for focal in range(3):
        agents, landmarks = candidate_geometry_from_local_obs_batch(obs[focal:focal + 1])
        agents = agents[0]; landmarks = landmarks[0]
        distances = np.linalg.norm(agents[:, :, None, :] - landmarks[None, None, :, :], axis=-1)
        costs = [
            min(sum(distances[action, i, permutation[i]] for i in range(3)) for permutation in PERMUTATIONS) / 3.0
            for action in range(5)
        ]
        if collision_safe:
            pair_separations = np.stack([
                np.linalg.norm(agents[:, 0] - agents[:, 1], axis=1),
                np.linalg.norm(agents[:, 0] - agents[:, 2], axis=1),
                np.linalg.norm(agents[:, 1] - agents[:, 2], axis=1),
            ], axis=1)
            minimum_separation = pair_separations.min(axis=1)
            safe = np.flatnonzero(minimum_separation >= 0.10)
            index = int(
                safe[np.argmin(np.asarray(costs)[safe])]
                if len(safe) else np.argmax(minimum_separation)
            )
        else:
            index = int(np.argmin(costs))
        value = np.zeros(5, dtype=np.float32); value[index] = 1.0
        actions.append(value)
    return actions


def evaluate(episodes, horizon, seed_base, collision_safe=False):
    env = make_env("simple_spread", discrete_action=True); rows = []
    try:
        for episode in range(episodes):
            seed = seed_base + episode; torch.manual_seed(seed); np.random.seed(seed); env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32); total = 0.0; collision_steps = 0
            min_sep = float("inf"); max_coverage = 0
            for _ in range(horizon):
                obs, rewards, dones, _ = env.step(exact_matching_actions(obs, collision_safe)); obs = np.asarray(obs, dtype=np.float32)
                total += float(np.mean(rewards))
                assignment, auc, separation, collision, coverage = geometry(env)
                collision_steps += collision; min_sep = min(min_sep, separation); max_coverage = max(max_coverage, coverage)
                if all(dones): break
            rows.append({"episode": episode, "return": total,
                         "hungarian_assignment_distance": assignment, "coverage_radius_auc": auc,
                         "collision_step_rate": collision_steps / horizon, "minimum_agent_separation": min_sep,
                         "final_coverage": coverage, "max_coverage": max_coverage})
    finally: env.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/exact_matching_controller_20260712")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=2_400_000)
    parser.add_argument("--collision-safe", action="store_true")
    args = parser.parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    rows = evaluate(args.episodes, args.horizon, args.seed_base, args.collision_safe)
    with (output / "per_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = {}
    for key in rows[0]:
        if key != "episode":
            values = np.asarray([row[key] for row in rows]); summary[key + "_mean"] = values.mean(); summary[key + "_std"] = values.std(ddof=1)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
