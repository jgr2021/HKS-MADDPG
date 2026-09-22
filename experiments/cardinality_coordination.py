"""Shared NxN teacher, evaluation, and geometry utilities.

The module generalizes the locked 6x6 independent-safe matching teacher to
balanced simple_spread cardinalities without changing physics, reward, local
observations, or actions.  It deliberately uses exact permutation search only
for N<=6, which is the scope of the ICASSP cardinality study.
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from utils.make_env import make_env
from utils.scalable_active_features import (
    ACTION_TO_CONTROL,
    DAMPING,
    DT,
    SENSITIVITY,
    expected_observation_dim,
)


N_ACTIONS = 5
RADII = np.linspace(0.05, 0.30, 26)
COLLISION_DISTANCE = 0.30


def env_id_for_cardinality(n_agents: int) -> str:
    if int(n_agents) == 3:
        # The explicit alias is used so every sweep arm follows the same
        # cardinality factory, while the historical environment remains intact.
        return "simple_spread_3x3"
    return f"simple_spread_{int(n_agents)}x{int(n_agents)}"


def observation_dim(n_agents: int) -> int:
    return expected_observation_dim(int(n_agents), int(n_agents))


@lru_cache(maxsize=None)
def permutations_for(n_agents: int) -> np.ndarray:
    n_agents = int(n_agents)
    if not 2 <= n_agents <= 6:
        raise ValueError("Exact cardinality teacher is locked to 2<=N<=6")
    return np.asarray(
        list(itertools.permutations(range(n_agents))), dtype=np.int64
    )


def reconstruct_numpy(raw, self_index: int, n_agents: int):
    raw = np.asarray(raw, dtype=np.float64)
    expected = observation_dim(n_agents)
    if raw.shape != (expected,):
        raise ValueError(f"Expected a {expected}D local observation, got {raw.shape}")
    if not 0 <= int(self_index) < n_agents:
        raise ValueError("self_index is outside the configured cardinality")
    landmark_end = 4 + 2 * n_agents
    other_end = landmark_end + 2 * (n_agents - 1)
    landmarks = raw[4:landmark_end].reshape(n_agents, 2)
    other_agents = raw[landmark_end:other_end].reshape(n_agents - 1, 2)
    agents = np.zeros((n_agents, 2), dtype=np.float64)
    other_indices = [index for index in range(n_agents) if index != self_index]
    agents[other_indices] = other_agents
    return agents, landmarks


def assignment_for_positions(agents, landmarks):
    agents = np.asarray(agents, dtype=np.float64)
    landmarks = np.asarray(landmarks, dtype=np.float64)
    if agents.shape != landmarks.shape or agents.ndim != 2 or agents.shape[1] != 2:
        raise ValueError("Balanced agent/landmark position arrays are required")
    n_agents = agents.shape[0]
    distance = np.linalg.norm(
        agents[:, None, :] - landmarks[None, :, :], axis=2
    )
    permutations = permutations_for(n_agents)
    costs = distance[
        np.arange(n_agents)[None, :], permutations
    ].sum(axis=1)
    best = int(np.argmin(costs))
    return permutations[best], float(costs[best] / n_agents), distance


def candidate_self_positions(raw):
    velocity = np.asarray(raw[0:2], dtype=np.float64)
    controls = np.asarray(ACTION_TO_CONTROL, dtype=np.float64)
    return (
        velocity[None, :] * (1.0 - DAMPING) * DT
        + controls * SENSITIVITY * DT * DT
    )


def reward_proxy(agents, landmarks):
    n_agents = len(agents)
    distance = np.linalg.norm(
        agents[:, None, :] - landmarks[None, :, :], axis=2
    )
    coverage_cost = distance.min(axis=0).sum()
    aa = np.linalg.norm(
        agents[:, None, :] - agents[None, :, :], axis=2
    )
    collision_pairs = int(
        np.triu(aa < COLLISION_DISTANCE, k=1).sum()
    )
    return -coverage_cost - (2.0 / float(n_agents)) * collision_pairs


def teacher_action_index(raw, self_index: int, n_agents: int):
    """Independent-safe teacher action from one focal local observation."""
    agents, landmarks = reconstruct_numpy(raw, self_index, n_agents)
    assignment, _, _ = assignment_for_positions(agents, landmarks)
    positions = candidate_self_positions(raw)
    assigned_landmark = landmarks[int(assignment[self_index])]
    assignment_distance = np.linalg.norm(
        positions - assigned_landmark[None, :], axis=1
    )
    proxy = np.zeros(N_ACTIONS, dtype=np.float64)
    for action in range(N_ACTIONS):
        candidate_agents = agents.copy()
        candidate_agents[self_index] = positions[action]
        proxy[action] = reward_proxy(candidate_agents, landmarks)
    reward_feasible = np.flatnonzero(proxy >= proxy[0] - 1e-12)

    other_indices = [index for index in range(n_agents) if index != self_index]
    separation = np.linalg.norm(
        positions[:, None, :] - agents[other_indices][None, :, :], axis=2
    ).min(axis=1)
    safety_buffer = COLLISION_DISTANCE + 2.0 * SENSITIVITY * DT * DT
    required = min(safety_buffer, float(separation[0]))
    safe = np.flatnonzero(separation >= required - 1e-12)
    safe_reward = np.intersect1d(safe, reward_feasible, assume_unique=True)
    feasible = safe_reward if safe_reward.size else safe
    action = int(feasible[np.argmin(assignment_distance[feasible])])
    return action


def teacher_actions(obs, n_agents: int):
    obs = np.asarray(obs, dtype=np.float32)
    if obs.shape != (n_agents, observation_dim(n_agents)):
        raise ValueError(
            f"Expected observations {(n_agents, observation_dim(n_agents))}, "
            f"got {obs.shape}"
        )
    output = []
    for agent_index in range(n_agents):
        action = teacher_action_index(obs[agent_index], agent_index, n_agents)
        value = np.zeros(N_ACTIONS, dtype=np.float32)
        value[action] = 1.0
        output.append(value)
    return output


def policy_actions(policies, obs):
    actions = []
    with torch.no_grad():
        for index, policy in enumerate(policies):
            logits = policy(torch.as_tensor(
                obs[index:index + 1], dtype=torch.float32
            ))
            action = int(logits.argmax(1))
            actions.append(np.eye(N_ACTIONS, dtype=np.float32)[action])
    return actions


def world_metrics(env):
    agents = np.asarray([item.state.p_pos for item in env.world.agents])
    landmarks = np.asarray([item.state.p_pos for item in env.world.landmarks])
    _, assignment, distance = assignment_for_positions(agents, landmarks)
    nearest = distance.min(axis=0)
    auc = float(
        np.trapz(
            [(nearest < radius).mean() for radius in RADII], RADII
        ) / (RADII[-1] - RADII[0])
    )
    n_agents = len(agents)
    aa = np.linalg.norm(
        agents[:, None, :] - agents[None, :, :], axis=2
    )
    upper = aa[np.triu_indices(n_agents, k=1)]
    collision_pairs = int((upper < COLLISION_DISTANCE).sum())
    closest_landmarks = distance.argmin(axis=1)
    return {
        "hungarian_assignment_distance": assignment,
        "coverage_radius_auc": auc,
        "minimum_agent_separation": float(upper.min()),
        "collisions": collision_pairs,
        "final_coverage": int((nearest < 0.10).sum()),
        "nearest_landmark_distance": float(nearest.mean()),
        "unique_landmarks_covered": int(np.unique(closest_landmarks).size),
    }


def evaluate_controller(
    controller: str,
    policies,
    n_agents: int,
    episodes: int,
    horizon: int,
    seed_base: int,
):
    env = make_env(env_id_for_cardinality(n_agents), discrete_action=True)
    rows = []
    try:
        for episode in range(episodes):
            seed = seed_base + episode
            torch.manual_seed(seed)
            np.random.seed(seed)
            env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            total_return = 0.0
            collision_steps = 0
            minimum_separation = float("inf")
            max_coverage = 0
            elapsed = 0
            for _ in range(horizon):
                actions = (
                    teacher_actions(obs, n_agents)
                    if controller == "teacher"
                    else policy_actions(policies, obs)
                )
                obs, rewards, dones, _ = env.step(actions)
                obs = np.asarray(obs, dtype=np.float32)
                total_return += float(np.mean(rewards))
                elapsed += 1
                state = world_metrics(env)
                collision_steps += int(state["collisions"] > 0)
                minimum_separation = min(
                    minimum_separation, state["minimum_agent_separation"]
                )
                max_coverage = max(max_coverage, state["final_coverage"])
                if all(dones):
                    break
            rows.append({
                "controller": controller,
                "episode": episode,
                "test_seed": seed,
                "return": total_return,
                "hungarian_assignment_distance": state[
                    "hungarian_assignment_distance"
                ],
                "coverage_radius_auc": state["coverage_radius_auc"],
                "collision_step_rate": collision_steps / float(elapsed),
                "minimum_agent_separation": minimum_separation,
                "final_coverage": state["final_coverage"],
                "max_coverage": max_coverage,
            })
    finally:
        env.close()
    return rows


def evaluate_model(env_id, model_path, episodes, episode_length, seed):
    """Generic checkpoint evaluator compatible with the shared online runner."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MADDPG.init_from_save(str(Path(model_path)))
    model.prep_rollouts(device="cpu")
    n_agents = len(model.agents)
    env = make_env(env_id, discrete_action=model.discrete_action)
    rows = []
    try:
        for episode in range(episodes):
            test_seed = seed + episode
            torch.manual_seed(test_seed)
            np.random.seed(test_seed)
            env.seed(test_seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            total_return = 0.0
            collision_steps = 0
            minimum_separation = float("inf")
            max_coverage = 0
            elapsed = 0
            for _ in range(episode_length):
                tensors = [
                    torch.as_tensor(value, dtype=torch.float32).view(1, -1)
                    for value in obs
                ]
                with torch.no_grad():
                    actions = [
                        value.cpu().numpy().ravel()
                        for value in model.step(tensors, explore=False)
                    ]
                obs, rewards, dones, _ = env.step(actions)
                obs = np.asarray(obs, dtype=np.float32)
                total_return += float(np.mean(rewards))
                elapsed += 1
                state = world_metrics(env)
                collision_steps += int(state["collisions"] > 0)
                minimum_separation = min(
                    minimum_separation, state["minimum_agent_separation"]
                )
                max_coverage = max(max_coverage, state["final_coverage"])
                if all(dones):
                    break
            final_coverage = state["final_coverage"]
            rows.append({
                "eval_episode": episode,
                "test_seed": test_seed,
                "return": total_return,
                "hungarian_assignment_distance": state[
                    "hungarian_assignment_distance"
                ],
                "coverage_radius_auc": state["coverage_radius_auc"],
                "collision_step_rate": collision_steps / float(elapsed),
                "final_coverage": final_coverage,
                "max_coverage": max_coverage,
                "final3": int(final_coverage == n_agents),
                "max3": int(max_coverage == n_agents),
                "collisions": state["collisions"],
                "nearest_landmark_distance": state[
                    "nearest_landmark_distance"
                ],
                "minimum_agent_separation": minimum_separation,
                "unique_landmarks_covered": state[
                    "unique_landmarks_covered"
                ],
                "full_coverage_rate": int(final_coverage == n_agents),
            })
    finally:
        env.close()
    return rows


ONLINE_METRICS = [
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "return",
    "final_coverage",
    "max_coverage",
    "nearest_landmark_distance",
    "minimum_agent_separation",
    "unique_landmarks_covered",
    "full_coverage_rate",
    "final3",
    "max3",
    "collisions",
]
