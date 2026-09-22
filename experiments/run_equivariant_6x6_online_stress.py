"""Exact-Sinkhorn first-update stress test on the separate 6x6 task."""

import csv
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.maddpg import MADDPG
from experiments import run_passive_gsp_topology_pilot as protocol
from experiments.evaluate_scalable_6x6_teacher import RADII, assignment_for_positions
from utils.make_env import make_env


PAYLOAD = "experiments/equivariant_6x6_sinkhorn64_seed1_20260713/equivariant_6x6_actor.pt"
METRICS = [
    "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate",
    "return", "final_coverage", "max_coverage", "nearest_landmark_distance",
    "minimum_agent_separation", "unique_landmarks_covered", "full_coverage_rate",
    "final3", "max3", "collisions",
]
METHODS = {
    "equivariant_6x6_unanchored": {
        "env_id": "simple_spread_6x6", "actor_model": "equivariant_matching_safety_6x6",
        "short": "eq6", "actor_input_dim": 36, "raw_observation_dim": 36,
        "n_agents": 6, "action_dim": 5,
        "pretrained_policy_path": PAYLOAD, "pretrained_load_weights": True,
    },
    "equivariant_6x6_kl002": {
        "env_id": "simple_spread_6x6", "actor_model": "equivariant_matching_safety_6x6",
        "short": "eq6kl", "actor_input_dim": 36, "raw_observation_dim": 36,
        "n_agents": 6, "action_dim": 5,
        "pretrained_policy_path": PAYLOAD, "pretrained_load_weights": True,
        "actor_anchor": {"mode": "fixed_teacher_kl", "max_kl": 0.002,
                         "temperature": 1.0, "bisection_steps": 16},
    },
}


def geometry(env):
    agents = np.asarray([item.state.p_pos for item in env.world.agents])
    landmarks = np.asarray([item.state.p_pos for item in env.world.landmarks])
    _, assignment, distances = assignment_for_positions(agents, landmarks)
    nearest = distances.min(axis=0)
    auc = float(np.trapz([(nearest < radius).mean() for radius in RADII], RADII)
                / (RADII[-1] - RADII[0]))
    aa = np.linalg.norm(agents[:, None] - agents[None], axis=2)
    upper = aa[np.triu_indices(6, k=1)]
    collision_pairs = int((upper < 0.30).sum())
    closest_landmarks = distances.argmin(axis=1)
    return {
        "hungarian_assignment_distance": assignment,
        "coverage_radius_auc": auc,
        "minimum_agent_separation": float(upper.min()),
        "collisions": collision_pairs,
        "final_coverage": int((nearest < 0.10).sum()),
        "nearest_landmark_distance": float(nearest.mean()),
        "unique_landmarks_covered": int(np.unique(closest_landmarks).size),
    }


def evaluate_model(env_id, model_path, episodes, episode_length, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = MADDPG.init_from_save(str(model_path)); model.prep_rollouts(device="cpu")
    env = make_env(env_id, discrete_action=model.discrete_action); rows = []
    try:
        for episode in range(episodes):
            test_seed = seed + episode
            torch.manual_seed(test_seed); np.random.seed(test_seed); env.seed(test_seed)
            obs = env.reset(); total_return = 0.0; collision_steps = 0
            min_sep = float("inf"); max_coverage = 0; elapsed = 0
            for _ in range(episode_length):
                tensors = [torch.as_tensor(item, dtype=torch.float32).view(1, -1) for item in obs]
                with torch.no_grad():
                    actions = [value.numpy().ravel() for value in model.step(tensors, explore=False)]
                obs, rewards, dones, _ = env.step(actions)
                total_return += float(np.mean(rewards)); elapsed += 1
                state = geometry(env); collision_steps += int(state["collisions"] > 0)
                min_sep = min(min_sep, state["minimum_agent_separation"])
                max_coverage = max(max_coverage, state["final_coverage"])
                if all(dones): break
            final = state["final_coverage"]
            rows.append({"eval_episode": episode, "test_seed": test_seed,
                         "return": total_return,
                         "hungarian_assignment_distance": state["hungarian_assignment_distance"],
                         "coverage_radius_auc": state["coverage_radius_auc"],
                         "collision_step_rate": collision_steps / float(elapsed),
                         "final_coverage": final, "max_coverage": max_coverage,
                         "final3": int(final == 6), "max3": int(max_coverage == 6),
                         "collisions": state["collisions"],
                         "nearest_landmark_distance": state["nearest_landmark_distance"],
                         "minimum_agent_separation": min_sep,
                         "unique_landmarks_covered": state["unique_landmarks_covered"],
                         "full_coverage_rate": int(final == 6)})
    finally:
        env.close()
    return rows


def defaults(argv):
    values = {"--output-dir": "experiments/equivariant_6x6_online_stress",
              "--total-env-steps": "1000", "--eval-episodes": "100",
              "--curve-eval-episodes": "100", "--seeds": "71",
              "--checkpoint-steps": "4,64,100,500,1000",
              "--methods": ",".join(METHODS),
              "--eval-seed-base": "6400000", "--curve-eval-seed-base": "6500000"}
    present = {item.split("=", 1)[0] for item in argv}; output = list(argv)
    for flag, value in values.items():
        if flag not in present: output += [flag, value]
    return output


def main():
    protocol.METHODS = METHODS; protocol.METRICS = METRICS
    protocol.evaluate_model = evaluate_model
    protocol.STATUS_TITLE = "Equivariant 6x6 First-Update Stress Status"
    protocol.PLOT_TITLE_PREFIX = "Equivariant 6x6 exact-Sinkhorn stress"
    sys.argv = defaults(sys.argv); protocol.main()


if __name__ == "__main__":
    main()
