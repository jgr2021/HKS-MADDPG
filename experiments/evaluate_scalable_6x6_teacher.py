"""Evaluate a transparent decentralized 6x6 assignment/reward-feasible teacher."""

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

from utils.make_env import make_env
from utils.scalable_active_features import (
    ACTION_TO_CONTROL, DAMPING, DT, SENSITIVITY,
)


N_AGENTS = 6
N_LANDMARKS = 6
PERMUTATIONS = np.asarray(list(itertools.permutations(range(6))), dtype=np.int64)
RADII = np.linspace(0.05, 0.30, 26)


def reconstruct_numpy(raw, self_index):
    raw = np.asarray(raw, dtype=np.float64)
    landmarks = raw[4:16].reshape(6, 2)
    other_agents = raw[16:26].reshape(5, 2)
    agents = np.zeros((6, 2), dtype=np.float64)
    other_indices = [index for index in range(6) if index != self_index]
    agents[other_indices] = other_agents
    return agents, landmarks


def assignment_for_positions(agents, landmarks):
    distance = np.linalg.norm(agents[:, None, :] - landmarks[None, :, :], axis=2)
    costs = distance[np.arange(6)[None, :], PERMUTATIONS].sum(axis=1)
    best = int(np.argmin(costs))
    return PERMUTATIONS[best], float(costs[best] / 6.0), distance


def candidate_self_positions(raw):
    velocity = np.asarray(raw[0:2], dtype=np.float64)
    controls = np.asarray(ACTION_TO_CONTROL, dtype=np.float64)
    return (
        velocity[None, :] * (1.0 - DAMPING) * DT
        + controls * SENSITIVITY * DT * DT
    )


def reward_proxy(agents, landmarks):
    distance = np.linalg.norm(agents[:, None, :] - landmarks[None, :, :], axis=2)
    coverage_cost = distance.min(axis=0).sum()
    aa = np.linalg.norm(agents[:, None, :] - agents[None, :, :], axis=2)
    collision_pairs = int(np.triu(aa < 0.30, k=1).sum())
    return -coverage_cost - (2.0 / N_AGENTS) * collision_pairs


def teacher_action_index(raw, self_index, safety_filter=False):
    agents, landmarks = reconstruct_numpy(raw, self_index)
    assignment, _, _ = assignment_for_positions(agents, landmarks)
    positions = candidate_self_positions(raw)
    assigned_landmark = landmarks[int(assignment[self_index])]
    assignment_distance = np.linalg.norm(positions - assigned_landmark[None, :], axis=1)
    proxy = np.zeros(5, dtype=np.float64)
    for action in range(5):
        candidate_agents = agents.copy()
        candidate_agents[self_index] = positions[action]
        proxy[action] = reward_proxy(candidate_agents, landmarks)
    feasible = np.flatnonzero(proxy >= proxy[0] - 1e-12)
    if safety_filter:
        other_indices = [index for index in range(6) if index != self_index]
        separation = np.linalg.norm(
            positions[:, None, :] - agents[other_indices][None, :, :], axis=2
        ).min(axis=1)
        safety_buffer = 0.30 + 2.0 * SENSITIVITY * DT * DT
        required = min(safety_buffer, float(separation[0]))
        safe = np.flatnonzero(separation >= required - 1e-12)
        safe_reward = np.intersect1d(safe, feasible, assume_unique=True)
        feasible = safe_reward if safe_reward.size else safe
    return int(feasible[np.argmin(assignment_distance[feasible])]), tuple(assignment.tolist())


def conservative_joint_plan(raw, self_index):
    """Reconstruct a common velocity-blind plan with a derived safety buffer."""
    agents, landmarks = reconstruct_numpy(raw, self_index)
    assignment, _, _ = assignment_for_positions(agents, landmarks)
    controls = np.asarray(ACTION_TO_CONTROL, dtype=np.float64) * SENSITIVITY * DT * DT
    desired = np.zeros(6, dtype=np.int64)
    remaining = np.zeros(6, dtype=np.float64)
    base_proxy = reward_proxy(agents, landmarks)
    for agent_index in range(6):
        target = landmarks[int(assignment[agent_index])]
        distances = np.linalg.norm(
            agents[agent_index][None, :] + controls - target[None, :], axis=1
        )
        proxy = np.zeros(5, dtype=np.float64)
        for action in range(5):
            candidate = agents.copy()
            candidate[agent_index] += controls[action]
            proxy[action] = reward_proxy(candidate, landmarks)
        feasible = np.flatnonzero(proxy >= base_proxy - 1e-12)
        desired[agent_index] = int(feasible[np.argmin(distances[feasible])])
        remaining[agent_index] = np.linalg.norm(agents[agent_index] - target)

    # A pair can close by at most two 0.05 action displacements in one step.
    # collision diameter 0.30 + relative action displacement 0.10 = 0.40.
    safety_buffer = 0.30 + 2.0 * SENSITIVITY * DT * DT
    current_aa = np.linalg.norm(
        agents[:, None, :] - agents[None, :, :], axis=2
    )[np.triu_indices(6, k=1)]
    # When collision is already present, issue a common no-op plan and let the
    # unchanged MPE contact force separate the pair instead of pushing through.
    if np.any(current_aa < 0.30):
        return tuple(np.zeros(6, dtype=np.int64).tolist()), tuple(assignment.tolist())
    planned_positions = agents.copy()
    accepted = np.zeros(6, dtype=np.int64)
    for agent_index in np.argsort(remaining, kind="stable"):
        candidate_position = agents[agent_index] + controls[desired[agent_index]]
        safe = True
        for other in range(6):
            if other == agent_index:
                continue
            current_distance = np.linalg.norm(agents[agent_index] - agents[other])
            candidate_distance = np.linalg.norm(candidate_position - planned_positions[other])
            required = min(safety_buffer, current_distance)
            if candidate_distance + 1e-12 < required:
                safe = False
                break
        if safe:
            accepted[agent_index] = desired[agent_index]
            planned_positions[agent_index] = candidate_position
    return tuple(accepted.tolist()), tuple(assignment.tolist())


def teacher_actions(obs, controller="independent_hybrid"):
    actions, plans = [], []
    for agent_index in range(6):
        if controller in {"independent_hybrid", "independent_safe_hybrid"}:
            action, plan = teacher_action_index(
                obs[agent_index], agent_index,
                safety_filter=controller == "independent_safe_hybrid"
            )
            action_plan = None
        elif controller == "conservative_joint":
            action_plan, plan = conservative_joint_plan(obs[agent_index], agent_index)
            action = int(action_plan[agent_index])
        else:
            raise ValueError(controller)
        actions.append(np.eye(5, dtype=np.float32)[action])
        plans.append((plan, action_plan))
    return actions, plans


def world_metrics(env):
    agents = np.asarray([item.state.p_pos for item in env.world.agents])
    landmarks = np.asarray([item.state.p_pos for item in env.world.landmarks])
    _, assignment, distance = None, None, None
    plan, assignment, distance = assignment_for_positions(agents, landmarks)
    nearest = distance.min(axis=0)
    auc = float(np.trapz(
        [(nearest < radius).mean() for radius in RADII], RADII
    ) / (RADII[-1] - RADII[0]))
    aa = np.linalg.norm(agents[:, None, :] - agents[None, :, :], axis=2)
    upper_values = aa[np.triu_indices(6, k=1)]
    collision = int(np.any(upper_values < 0.30))
    return assignment, auc, float(upper_values.min()), collision, int((nearest < 0.10).sum())


def evaluate(episodes, horizon, seed_base, controller):
    env = make_env("simple_spread_6x6", discrete_action=True)
    rows = []
    try:
        for episode in range(episodes):
            seed = seed_base + episode
            torch.manual_seed(seed); np.random.seed(seed); env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            total_return, collision_steps = 0.0, 0
            min_separation, max_coverage = float("inf"), 0
            plan_disagreements = 0
            for _ in range(horizon):
                actions, plans = teacher_actions(obs, controller)
                plan_disagreements += int(any(plan != plans[0] for plan in plans[1:]))
                obs, rewards, dones, _ = env.step(actions)
                obs = np.asarray(obs, dtype=np.float32)
                total_return += float(np.mean(rewards))
                assignment, auc, separation, collision, coverage = world_metrics(env)
                collision_steps += collision
                min_separation = min(min_separation, separation)
                max_coverage = max(max_coverage, coverage)
                if all(dones):
                    break
            rows.append({
                "episode": episode, "test_seed": seed, "return": total_return,
                "hungarian_assignment_distance": assignment,
                "coverage_radius_auc": auc,
                "collision_step_rate": collision_steps / float(horizon),
                "minimum_agent_separation": min_separation,
                "final_coverage": coverage, "max_coverage": max_coverage,
                "plan_disagreement_steps": plan_disagreements,
            })
    finally:
        env.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/scalable_6x6_teacher_20260713")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=5_100_000)
    parser.add_argument("--controller", choices=("independent_hybrid", "independent_safe_hybrid",
                                                  "conservative_joint"),
                        default="independent_hybrid")
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    rows = evaluate(args.episodes, args.horizon, args.seed_base, args.controller)
    with (output / "per_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {"episodes": len(rows)}
    for key in rows[0]:
        if key not in {"episode", "test_seed"}:
            values = np.asarray([row[key] for row in rows], dtype=np.float64)
            summary[key + "_mean"] = float(values.mean())
            summary[key + "_std"] = float(values.std(ddof=1))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "metadata.json").write_text(json.dumps({
        "environment": "separate simple_spread_6x6 cardinality config",
        "physics_reward_observation_callbacks": "unchanged simple_spread",
        "teacher": args.controller,
        "explicit_communication": False, "centralized_execution": False,
        "candidate_action_count": 5, "local_observation_dim": 36,
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
