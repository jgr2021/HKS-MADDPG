"""Pilot a transparent high-level assignment option on simple_spread.

The high-level controller computes a minimum-cost agent-landmark assignment.
The assignment is held for a configurable option horizon, while each low-level
agent greedily selects the primitive action that moves it toward its assigned
landmark.  This is an oracle feasibility test for temporal role abstraction,
not a learned hierarchical MARL method.
"""

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


N_AGENTS = 3
N_ACTIONS = 5
ASSIGNMENTS = np.asarray(
    list(itertools.permutations(range(N_AGENTS))), dtype=np.int64
)


def minimum_cost_assignment(env):
    agents = np.asarray([agent.state.p_pos for agent in env.world.agents])
    landmarks = np.asarray([landmark.state.p_pos for landmark in env.world.landmarks])
    distances = np.linalg.norm(
        agents[:, None, :] - landmarks[None, :, :], axis=2
    )
    costs = distances[np.arange(N_AGENTS)[None, :], ASSIGNMENTS].sum(axis=1)
    return ASSIGNMENTS[int(np.argmin(costs))].copy()


def assigned_landmark_actions(obs, assignment):
    actions = []
    for focal in range(N_AGENTS):
        candidate_agents, landmarks = candidate_geometry_from_local_obs_batch(
            obs[focal:focal + 1]
        )
        # Local candidate geometry always stores the focal actor in row zero;
        # the remaining rows follow that focal actor's local observation order.
        focal_positions = candidate_agents[0, :, 0, :]
        target = landmarks[0, int(assignment[focal]), :]
        action_index = int(np.argmin(np.linalg.norm(focal_positions - target, axis=1)))
        action = np.zeros(N_ACTIONS, dtype=np.float32)
        action[action_index] = 1.0
        actions.append(action)
    return actions


def evaluate_option_horizon(episodes, horizon, seed_base, option_horizon):
    env = make_env("simple_spread", discrete_action=True)
    rows = []
    try:
        for episode in range(episodes):
            seed = seed_base + episode
            torch.manual_seed(seed)
            np.random.seed(seed)
            env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            assignment = None
            previous_assignment = None
            role_changes = 0
            role_change_opportunities = 0
            total_return = 0.0
            collision_steps = 0
            minimum_separation = float("inf")
            max_coverage = 0
            elapsed = 0

            for step in range(horizon):
                if assignment is None or step % option_horizon == 0:
                    assignment = minimum_cost_assignment(env)
                    if previous_assignment is not None:
                        role_changes += int(np.sum(assignment != previous_assignment))
                        role_change_opportunities += N_AGENTS
                    previous_assignment = assignment.copy()

                actions = assigned_landmark_actions(obs, assignment)
                obs, rewards, dones, _ = env.step(actions)
                obs = np.asarray(obs, dtype=np.float32)
                total_return += float(np.mean(rewards))
                elapsed += 1
                assignment_distance, auc, separation, collision, coverage = geometry(env)
                collision_steps += collision
                minimum_separation = min(minimum_separation, separation)
                max_coverage = max(max_coverage, coverage)
                if all(dones):
                    break

            rows.append({
                "option_horizon": option_horizon,
                "episode": episode,
                "test_seed": seed,
                "return": total_return,
                "hungarian_assignment_distance": assignment_distance,
                "coverage_radius_auc": auc,
                "collision_step_rate": collision_steps / float(elapsed),
                "minimum_agent_separation": minimum_separation,
                "final_coverage": coverage,
                "max_coverage": max_coverage,
                "role_switch_rate": (
                    role_changes / float(role_change_opportunities)
                    if role_change_opportunities else 0.0
                ),
            })
    finally:
        env.close()
    return rows


def summarize(rows):
    metrics = [
        key for key in rows[0]
        if key not in {"option_horizon", "episode", "test_seed"}
    ]
    output = []
    for option_horizon in sorted({row["option_horizon"] for row in rows}):
        selected = [row for row in rows if row["option_horizon"] == option_horizon]
        summary = {"option_horizon": option_horizon, "episodes": len(selected)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected], dtype=np.float64)
            summary[metric + "_mean"] = float(values.mean())
            summary[metric + "_std"] = float(values.std(ddof=1))
        output.append(summary)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=63_000_000)
    parser.add_argument("--option-horizons", default="1,2,4,8,25")
    args = parser.parse_args()

    option_horizons = [int(value) for value in args.option_horizons.split(",")]
    if any(value <= 0 for value in option_horizons):
        raise ValueError("Option horizons must be positive integers")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for option_horizon in option_horizons:
        rows.extend(evaluate_option_horizon(
            args.episodes,
            args.horizon,
            args.seed_base,
            option_horizon,
        ))

    with (output / "per_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summaries = summarize(rows)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    (output / "summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
