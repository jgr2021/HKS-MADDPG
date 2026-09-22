import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.active_gsp_v3_features import compute_active_gsp_v3_from_local_obs_batch
from utils.make_env import make_env


FEATURE_SCALE = np.asarray([10.0, 70.0, 20.0, 110.0], dtype=np.float32)


def onehot(index, size=5):
    action = np.zeros(size, dtype=np.float32)
    action[int(index)] = 1.0
    return action


def coverage_metrics(env, threshold=0.10):
    agents = env.world.agents
    landmarks = env.world.landmarks
    distances = np.asarray(
        [
            [
                np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
                for landmark in landmarks
            ]
            for agent in agents
        ],
        dtype=np.float32,
    )
    nearest = distances.min(axis=0)
    collisions = 0
    for i in range(len(agents)):
        for j in range(i + 1, len(agents)):
            dist = np.linalg.norm(agents[i].state.p_pos - agents[j].state.p_pos)
            if dist < agents[i].size + agents[j].size:
                collisions += 1
    return int((nearest < threshold).sum()), float(nearest.mean()), collisions


def choose_actions(obs, policy_name, rng):
    if policy_name == "noop":
        return [onehot(0) for _ in range(len(obs))]
    if policy_name == "random":
        return [onehot(rng.randint(5)) for _ in range(len(obs))]

    features = compute_active_gsp_v3_from_local_obs_batch(np.asarray(obs, dtype=np.float32))
    if policy_name == "coverage_delta":
        scores = features[:, :, 0]
    elif policy_name == "raw_potential_sum":
        scores = features[:, :, 0] + features[:, :, 1]
    elif policy_name == "v3_unscaled_sum":
        scores = features.sum(axis=2)
    elif policy_name == "v3_scaled_sum":
        scores = (features * FEATURE_SCALE[None, None, :]).sum(axis=2)
    else:
        raise ValueError(f"Unknown heuristic policy: {policy_name}")
    return [onehot(index) for index in np.argmin(scores, axis=1)]


def evaluate_policy(policy_name, episodes, episode_length, seed):
    rng = np.random.RandomState(seed)
    env = make_env("simple_spread", discrete_action=True)
    env.seed(seed)
    returns = []
    final_coverages = []
    max_coverages = []
    final_distances = []
    collisions = []
    action_counts = np.zeros(5, dtype=np.int64)
    try:
        for episode in range(episodes):
            np.random.seed(seed + episode)
            obs = env.reset()
            episode_return = 0.0
            max_coverage = 0
            final_coverage = 0
            final_distance = 0.0
            final_collisions = 0
            for _ in range(episode_length):
                actions = choose_actions(obs, policy_name, rng)
                for action in actions:
                    action_counts[int(np.argmax(action))] += 1
                obs, rewards, dones, _ = env.step(actions)
                episode_return += float(np.mean(rewards))
                final_coverage, final_distance, final_collisions = coverage_metrics(env)
                max_coverage = max(max_coverage, final_coverage)
                if all(dones):
                    break
            returns.append(episode_return)
            final_coverages.append(final_coverage)
            max_coverages.append(max_coverage)
            final_distances.append(final_distance)
            collisions.append(final_collisions)
    finally:
        env.close()
    total_actions = max(int(action_counts.sum()), 1)
    return {
        "policy": policy_name,
        "return": float(np.mean(returns)),
        "final_coverage": float(np.mean(final_coverages)),
        "max_coverage": float(np.mean(max_coverages)),
        "nearest_landmark_distance": float(np.mean(final_distances)),
        "collisions": float(np.mean(collisions)),
        "action_noop_pct": float(action_counts[0] / total_actions),
        "action_left_pct": float(action_counts[1] / total_actions),
        "action_right_pct": float(action_counts[2] / total_actions),
        "action_down_pct": float(action_counts[3] / total_actions),
        "action_up_pct": float(action_counts[4] / total_actions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--episode_length", type=int, default=25)
    parser.add_argument("--seed", type=int, default=9400)
    parser.add_argument(
        "--policies",
        default="noop,random,coverage_delta,raw_potential_sum,v3_unscaled_sum,v3_scaled_sum",
    )
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = Path("experiments") / f"active_gsp_v3_heuristics_{stamp}"
    result_dir.mkdir(parents=True, exist_ok=False)

    policies = [name.strip() for name in args.policies.split(",") if name.strip()]
    rows = [
        evaluate_policy(policy, args.episodes, args.episode_length, args.seed + i * 1000)
        for i, policy in enumerate(policies)
    ]

    csv_path = result_dir / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    readme_path = result_dir / "README.md"
    lines = [
        "# Active GSP v3 Heuristic Evaluation",
        "",
        f"Episodes: `{args.episodes}`",
        f"Episode length: `{args.episode_length}`",
        f"Base seed: `{args.seed}`",
        "",
        "| policy | return | final_coverage | max_coverage | nearest_distance | collisions |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['return']:.4f} | "
            f"{row['final_coverage']:.4f} | {row['max_coverage']:.4f} | "
            f"{row['nearest_landmark_distance']:.4f} | {row['collisions']:.4f} |"
        )
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved heuristic results: {csv_path}", flush=True)
    print(f"Saved README: {readme_path}", flush=True)


if __name__ == "__main__":
    main()
