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
from utils.make_env import make_env


METHODS = [
    "raw_mlp",
    "learned_raw_potential_residual",
    "learned_active_gsp_residual",
]
LABELS = {
    "raw_mlp": "Raw MADDPG",
    "learned_raw_potential_residual": "Action-aware control",
    "learned_active_gsp_residual": "Active GSP",
}
THRESHOLDS = [("t10", 0.10), ("t20", 0.20), ("t30", 0.30)]
T_CRITICAL_95_DF2 = 4.3026527297


def source_map(args):
    seed1_active = Path(args.seed1_active_root)
    seed1_raw = Path(args.seed1_raw_root)
    seeds23 = Path(args.seeds23_root)
    sources = {
        1: {
            "raw_mlp": seed1_raw / "raw_mlp" / "seed_1",
            "learned_raw_potential_residual": (
                seed1_active / "learned_raw_potential_residual" / "seed_1"
            ),
            "learned_active_gsp_residual": (
                seed1_active / "learned_active_gsp_residual" / "seed_1"
            ),
        }
    }
    for seed in [2, 3]:
        sources[seed] = {
            method: seeds23 / method / f"seed_{seed}" for method in METHODS
        }
    return sources


def final_model(run_dir):
    checkpoints = run_dir / "checkpoints"
    matches = sorted(checkpoints.glob("model_final_*.pt"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one final model in {checkpoints}, got {matches}")
    return matches[0]


def geometry(env):
    agent_positions = np.asarray(
        [agent.state.p_pos for agent in env.world.agents], dtype=np.float32
    )
    landmark_positions = np.asarray(
        [landmark.state.p_pos for landmark in env.world.landmarks], dtype=np.float32
    )
    distances = np.linalg.norm(
        agent_positions[:, None, :] - landmark_positions[None, :, :], axis=2
    )
    nearest_landmark = distances.min(axis=0)
    pair_distances = []
    collisions = 0
    for i in range(len(env.world.agents)):
        for j in range(i + 1, len(env.world.agents)):
            distance = float(np.linalg.norm(agent_positions[i] - agent_positions[j]))
            pair_distances.append(distance)
            if distance < env.world.agents[i].size + env.world.agents[j].size:
                collisions += 1
    return distances, nearest_landmark, float(min(pair_distances)), collisions


def evaluate_model(model_path, seed, method, episodes, episode_length):
    maddpg = MADDPG.init_from_save(str(model_path))
    maddpg.prep_rollouts(device="cpu")
    env = make_env("simple_spread", discrete_action=maddpg.discrete_action)
    rows = []
    seed_base = 990000 + seed * 10000
    try:
        for episode in range(episodes):
            test_seed = seed_base + episode
            torch.manual_seed(test_seed)
            np.random.seed(test_seed)
            env.seed(test_seed)
            obs = env.reset()
            episode_return = 0.0
            episode_min_separation = float("inf")
            collision_steps = 0
            max_coverage = {name: 0 for name, _ in THRESHOLDS}
            full_steps = {name: 0 for name, _ in THRESHOLDS}
            longest_full_streak = {name: 0 for name, _ in THRESHOLDS}
            current_full_streak = {name: 0 for name, _ in THRESHOLDS}
            final_distances = None
            final_nearest = None
            final_min_separation = None
            final_collisions = None

            for _ in range(episode_length):
                torch_obs = [
                    torch.as_tensor(item, dtype=torch.float32).view(1, -1)
                    for item in obs
                ]
                with torch.no_grad():
                    torch_actions = maddpg.step(torch_obs, explore=False)
                actions = [item.cpu().numpy().ravel() for item in torch_actions]
                obs, rewards, dones, _ = env.step(actions)
                episode_return += float(np.mean(rewards))
                distances, nearest, min_separation, collisions = geometry(env)
                final_distances = distances
                final_nearest = nearest
                final_min_separation = min_separation
                final_collisions = collisions
                episode_min_separation = min(episode_min_separation, min_separation)
                collision_steps += int(collisions > 0)
                for name, threshold in THRESHOLDS:
                    coverage = int((nearest < threshold).sum())
                    max_coverage[name] = max(max_coverage[name], coverage)
                    if coverage == 3:
                        full_steps[name] += 1
                        current_full_streak[name] += 1
                        longest_full_streak[name] = max(
                            longest_full_streak[name], current_full_streak[name]
                        )
                    else:
                        current_full_streak[name] = 0
                if all(dones):
                    break

            output = {
                "method": method,
                "train_seed": seed,
                "eval_episode": episode,
                "test_seed": test_seed,
                "return": episode_return,
                "nearest_landmark_distance": float(final_nearest.mean()),
                "collisions": int(final_collisions),
                "collision_step_rate": collision_steps / float(episode_length),
                "distinct_serving_agents": int(
                    np.unique(final_distances.argmin(axis=0)).size
                ),
                "distinct_targeted_landmarks": int(
                    np.unique(final_distances.argmin(axis=1)).size
                ),
                "final_min_agent_separation": final_min_separation,
                "episode_min_agent_separation": episode_min_separation,
            }
            for name, threshold in THRESHOLDS:
                final_coverage = int((final_nearest < threshold).sum())
                output[f"final_coverage_{name}"] = final_coverage
                output[f"max_coverage_{name}"] = max_coverage[name]
                output[f"final3_{name}"] = int(final_coverage == 3)
                output[f"max3_{name}"] = int(max_coverage[name] == 3)
                output[f"full_step_rate_{name}"] = full_steps[name] / float(episode_length)
                output[f"longest_full_streak_{name}"] = longest_full_streak[name]
            rows.append(output)
    finally:
        env.close()
    return rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def numeric_metrics(rows):
    excluded = {"method", "train_seed", "eval_episode", "test_seed"}
    return sorted(key for key in rows[0] if key not in excluded)


def summarize_by_seed(rows):
    metrics = numeric_metrics(rows)
    output = []
    for method in METHODS:
        for seed in [1, 2, 3]:
            matching = [
                row for row in rows
                if row["method"] == method and int(row["train_seed"]) == seed
            ]
            summary = {"method": method, "train_seed": seed, "episodes": len(matching)}
            for metric in metrics:
                summary[metric] = float(np.mean([float(row[metric]) for row in matching]))
            output.append(summary)
    return output


def aggregate_seed_summaries(seed_rows):
    metrics = numeric_metrics(seed_rows)
    output = []
    for method in METHODS:
        matching = [row for row in seed_rows if row["method"] == method]
        summary = {"method": method, "label": LABELS[method], "n_seeds": 3}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in matching])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1))
        output.append(summary)
    return output


def paired_seed_summary(seed_rows):
    metrics = numeric_metrics(seed_rows)
    lookup = {
        (int(row["train_seed"]), row["method"]): row for row in seed_rows
    }
    output = []
    for comparison, left, right in [
        ("active_vs_raw", "learned_active_gsp_residual", "raw_mlp"),
        (
            "active_vs_action_control",
            "learned_active_gsp_residual",
            "learned_raw_potential_residual",
        ),
    ]:
        for metric in metrics:
            deltas = np.asarray(
                [
                    float(lookup[(seed, left)][metric])
                    - float(lookup[(seed, right)][metric])
                    for seed in [1, 2, 3]
                ]
            )
            mean = float(deltas.mean())
            std = float(deltas.std(ddof=1))
            half = T_CRITICAL_95_DF2 * std / math.sqrt(3)
            output.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "mean_delta": mean,
                    "std_delta": std,
                    "ci95_low": mean - half,
                    "ci95_high": mean + half,
                    "all_seed_deltas": ";".join(f"{value:.9f}" for value in deltas),
                }
            )
    return output


def write_readme(output_dir, aggregate, paired):
    agg = {row["method"]: row for row in aggregate}
    pair = {(row["comparison"], row["metric"]): row for row in paired}
    lines = [
        "# Corrected Active GSP Coverage Diagnostics",
        "",
        "Each of the nine final models was evaluated once for 500 deterministic",
        "episodes. Coverage thresholds were computed from the same rollout.",
        "",
        "| method | cov@0.10 | cov@0.20 | cov@0.30 | serving agents | targeted landmarks |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        row = agg[method]
        lines.append(
            f"| {LABELS[method]} | {row['final_coverage_t10_mean']:.3f} | "
            f"{row['final_coverage_t20_mean']:.3f} | "
            f"{row['final_coverage_t30_mean']:.3f} | "
            f"{row['distinct_serving_agents_mean']:.3f} | "
            f"{row['distinct_targeted_landmarks_mean']:.3f} |"
        )
    lines.extend(["", "## Active GSP Versus Action-Aware Control", ""])
    for metric in [
        "final_coverage_t10",
        "final_coverage_t20",
        "final_coverage_t30",
        "distinct_serving_agents",
        "distinct_targeted_landmarks",
        "final_min_agent_separation",
    ]:
        row = pair[("active_vs_action_control", metric)]
        lines.append(
            f"- `{metric}`: {row['mean_delta']:+.4f} "
            f"([{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}])"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed1-raw-root", required=True)
    parser.add_argument("--seed1-active-root", required=True)
    parser.add_argument("--seeds23-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--episode-length", type=int, default=25)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = source_map(args)
    all_rows = []
    for seed in [1, 2, 3]:
        for method in METHODS:
            model_path = final_model(sources[seed][method])
            all_rows.extend(
                evaluate_model(
                    model_path, seed, method, args.episodes, args.episode_length
                )
            )
    seed_rows = summarize_by_seed(all_rows)
    aggregate = aggregate_seed_summaries(seed_rows)
    paired = paired_seed_summary(seed_rows)
    write_csv(output_dir / "per_episode_diagnostics.csv", all_rows)
    write_csv(output_dir / "seed_summary.csv", seed_rows)
    write_csv(output_dir / "aggregate_summary.csv", aggregate)
    write_csv(output_dir / "paired_training_seed_summary.csv", paired)
    (output_dir / "sources.json").write_text(
        json.dumps(
            {
                str(seed): {
                    method: str(final_model(path)) for method, path in methods.items()
                }
                for seed, methods in sources.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_readme(output_dir, aggregate, paired)
    print(output_dir)


if __name__ == "__main__":
    main()
