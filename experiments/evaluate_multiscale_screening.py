"""Locked matched evaluation and promotion gate for multiscale Active GSP."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.maddpg import MADDPG
from experiments.audit_active_gsp_compression import assignment_distance, geometry, radius_auc
from utils.make_env import make_env


METHODS = (
    "learned_raw_potential_residual",
    "learned_active_gsp_residual",
    "learned_multiscale_spectral_residual",
)
CONTROL = METHODS[0]
CANDIDATE = METHODS[2]
METRICS = (
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "return",
    "final_coverage_r010",
    "maximum_coverage_r010",
    "minimum_agent_separation",
)
T_CRITICAL_95_DF2 = 4.3026527297


def final_model(run_dir):
    matches = sorted((run_dir / "checkpoints").glob("model_final_100000.pt"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one 100k final model under {run_dir}, got {matches}")
    return matches[0]


def evaluate(checkpoint, method, train_seed, episodes, episode_length, eval_seed_base):
    maddpg = MADDPG.init_from_save(str(checkpoint))
    maddpg.prep_rollouts(device="cpu")
    env = make_env("simple_spread", discrete_action=maddpg.discrete_action)
    rows = []
    try:
        for episode in range(episodes):
            test_seed = eval_seed_base + train_seed * 10000 + episode
            torch.manual_seed(test_seed)
            np.random.seed(test_seed)
            env.seed(test_seed)
            obs = env.reset()
            episode_return = 0.0
            collision_steps = 0
            min_separation = math.inf
            maximum_coverage = 0
            final_distances = None
            for _ in range(episode_length):
                torch_obs = [torch.as_tensor(item, dtype=torch.float32).view(1, -1) for item in obs]
                with torch.no_grad():
                    actions = [item.cpu().numpy().ravel() for item in maddpg.step(torch_obs, explore=False)]
                obs, rewards, dones, _ = env.step(actions)
                episode_return += float(np.mean(rewards))
                final_distances, separation, collision = geometry(env)
                nearest = final_distances.min(axis=0)
                maximum_coverage = max(maximum_coverage, int((nearest < 0.10).sum()))
                min_separation = min(min_separation, separation)
                collision_steps += collision
                if all(dones):
                    break
            nearest = final_distances.min(axis=0)
            rows.append({
                "method": method, "train_seed": train_seed, "eval_episode": episode,
                "test_seed": test_seed, "return": episode_return,
                "hungarian_assignment_distance": assignment_distance(final_distances),
                "coverage_radius_auc": radius_auc(nearest),
                "collision_step_rate": collision_steps / float(episode_length),
                "final_coverage_r010": int((nearest < 0.10).sum()),
                "maximum_coverage_r010": maximum_coverage,
                "minimum_agent_separation": min_separation,
            })
    finally:
        env.close()
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="11,12,13")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--eval-seed-base", type=int, default=4_000_000)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(seeds) != 3:
        raise ValueError("Locked screening requires exactly three fresh training seeds.")
    training_root = Path(args.training_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    episode_rows = []
    sources = []
    for method in METHODS:
        for seed in seeds:
            checkpoint = final_model(training_root / method / f"seed_{seed}")
            sources.append({"method": method, "train_seed": seed, "checkpoint": str(checkpoint.resolve())})
            episode_rows.extend(evaluate(
                checkpoint, method, seed, args.episodes, args.episode_length, args.eval_seed_base
            ))

    seed_rows = []
    for method in METHODS:
        for seed in seeds:
            selected = [row for row in episode_rows if row["method"] == method and row["train_seed"] == seed]
            summary = {"method": method, "train_seed": seed, "episodes": len(selected)}
            for metric in METRICS:
                summary[metric] = float(np.mean([row[metric] for row in selected]))
            seed_rows.append(summary)
    lookup = {(row["method"], row["train_seed"]): row for row in seed_rows}
    paired_rows = []
    for left in (METHODS[1], CANDIDATE):
        for metric in METRICS:
            deltas = np.asarray([
                lookup[(left, seed)][metric] - lookup[(CONTROL, seed)][metric]
                for seed in seeds
            ])
            mean = float(deltas.mean())
            std = float(deltas.std(ddof=1))
            half = T_CRITICAL_95_DF2 * std / math.sqrt(3.0)
            paired_rows.append({
                "comparison": f"{left}_minus_{CONTROL}", "metric": metric,
                "mean_delta": mean, "ci95_low": mean - half, "ci95_high": mean + half,
                "seed_deltas": ";".join(f"{value:+.9f}" for value in deltas),
                "directionally_consistent": bool(np.all(deltas > 0.0) or np.all(deltas < 0.0)),
            })

    pair = {(row["comparison"], row["metric"]): row for row in paired_rows}
    prefix = f"{CANDIDATE}_minus_{CONTROL}"
    auc = pair[(prefix, "coverage_radius_auc")]
    hungarian = pair[(prefix, "hungarian_assignment_distance")]
    collision = pair[(prefix, "collision_step_rate")]
    endpoint = pair[(prefix, "final_coverage_r010")]
    auc_gain = auc["mean_delta"] > 0 and all(float(v) > 0 for v in auc["seed_deltas"].split(";"))
    hungarian_gain = hungarian["mean_delta"] < 0 and all(float(v) < 0 for v in hungarian["seed_deltas"].split(";"))
    resolved_collision_penalty = collision["ci95_low"] > 0.0
    passes = bool((auc_gain or hungarian_gain) and not resolved_collision_penalty)
    endpoint_gain = endpoint["mean_delta"] > 0 and all(
        float(value) > 0 for value in endpoint["seed_deltas"].split(";")
    )
    decision = {
        "promote_to_ten_seeds_and_6x6": passes,
        "threshold_independent_auc_gain_all_seeds": auc_gain,
        "threshold_independent_hungarian_gain_all_seeds": hungarian_gain,
        "resolved_collision_penalty": resolved_collision_penalty,
        "endpoint_r010_gain_all_seeds": endpoint_gain,
        "branch_status": "promote" if passes else "stopped",
        "reason": (
            "Promotion gate passed."
            if passes
            else "Promotion gate failed; do not search additional scales or gates."
        ),
    }
    write_csv(output_dir / "episode_metrics.csv", episode_rows)
    write_csv(output_dir / "seed_metrics.csv", seed_rows)
    write_csv(output_dir / "paired_seed_deltas.csv", paired_rows)
    write_csv(output_dir / "sources.csv", sources)
    (output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
