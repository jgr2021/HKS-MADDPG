import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.maddpg import MADDPG
from experiments.run_passive_gsp_topology_pilot import coverage_metrics
from utils.make_env import make_env


MASKS = {
    "full": (1.0, 1.0, 1.0, 1.0),
    "potentials_only": (1.0, 1.0, 0.0, 0.0),
    "plus_coverage_energy": (1.0, 1.0, 1.0, 0.0),
    "plus_crowding_energy": (1.0, 1.0, 0.0, 1.0),
    "energies_only": (0.0, 0.0, 1.0, 1.0),
    "no_action_features": (0.0, 0.0, 0.0, 0.0),
}
METRICS = [
    "return",
    "final_coverage",
    "max_coverage",
    "final3",
    "max3",
    "nearest_landmark_distance",
    "collisions",
    "collision_step_rate",
]


def install_mask(maddpg, mask_values):
    mask = torch.tensor(mask_values, dtype=torch.float32).view(1, 1, -1)
    for policy in maddpg.policies:
        original = policy.action_features

        def masked_features(X, original=original, mask=mask):
            features = original(X)
            return features * mask.to(device=features.device, dtype=features.dtype)

        policy.action_features = masked_features


def evaluate(model_path, mask_values, episodes, episode_length, seed_base):
    maddpg = MADDPG.init_from_save(str(model_path))
    maddpg.prep_rollouts(device="cpu")
    install_mask(maddpg, mask_values)
    env = make_env("simple_spread", discrete_action=maddpg.discrete_action)
    rows = []
    try:
        for episode in range(episodes):
            test_seed = seed_base + episode
            torch.manual_seed(test_seed)
            np.random.seed(test_seed)
            env.seed(test_seed)
            obs = env.reset()
            episode_return = 0.0
            max_coverage = 0
            final_coverage = 0
            final_distance = 0.0
            final_collisions = 0
            collision_steps = 0
            first_full_step = np.nan
            for step_index in range(episode_length):
                torch_obs = [
                    torch.as_tensor(item, dtype=torch.float32).view(1, -1)
                    for item in obs
                ]
                with torch.no_grad():
                    torch_actions = maddpg.step(torch_obs, explore=False)
                actions = [item.cpu().numpy().ravel() for item in torch_actions]
                obs, rewards, dones, _ = env.step(actions)
                episode_return += float(np.mean(rewards))
                final_coverage, final_distance, final_collisions = coverage_metrics(env)
                max_coverage = max(max_coverage, final_coverage)
                collision_steps += int(final_collisions > 0)
                if final_coverage == 3 and np.isnan(first_full_step):
                    first_full_step = step_index + 1
                if all(dones):
                    break
            rows.append(
                {
                    "eval_episode": episode,
                    "test_seed": test_seed,
                    "return": episode_return,
                    "final_coverage": final_coverage,
                    "max_coverage": max_coverage,
                    "final3": int(final_coverage == 3),
                    "max3": int(max_coverage == 3),
                    "nearest_landmark_distance": final_distance,
                    "collisions": final_collisions,
                    "collision_step_rate": collision_steps / float(episode_length),
                    "time_to_first_full_coverage_step": first_full_step,
                }
            )
    finally:
        env.close()
    return rows


def write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--episode-length", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=1000000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    all_rows = []
    summaries = []
    for name, mask in MASKS.items():
        rows = evaluate(
            Path(args.model), mask, args.episodes, args.episode_length, args.seed_base
        )
        for row in rows:
            row["mask"] = name
        all_rows.extend(rows)
        summary = {"mask": name, "episodes": len(rows)}
        for metric in METRICS:
            summary[metric] = float(np.mean([row[metric] for row in rows]))
        summaries.append(summary)

    write_csv(output_dir / "per_episode.csv", all_rows)
    write_csv(output_dir / "summary.csv", summaries)
    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "episodes": args.episodes,
                "episode_length": args.episode_length,
                "seed_base": args.seed_base,
                "masks": MASKS,
                "interpretation": "post-training inference mask; not a retrained ablation",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir)


if __name__ == "__main__":
    main()
