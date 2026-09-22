"""No-communication joint-action matching computed from each local observation."""

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
from utils.active_gsp_v3_features import ACTION_TO_CONTROL, DEFAULT_DT, DEFAULT_SENSITIVITY
from utils.gsp_features import reconstruct_geometry_from_raw_observation
from utils.make_env import make_env


ACTION_TUPLES = tuple(itertools.product(range(5), repeat=3))
ASSIGNMENTS = tuple(itertools.permutations(range(3)))
ACTION_TUPLE_ARRAY = np.asarray(ACTION_TUPLES, dtype=np.int64)
ASSIGNMENT_ARRAY = np.asarray(ASSIGNMENTS, dtype=np.int64)


def plan_from_local_observation(raw_obs, self_agent_index, collision_safe):
    agents, landmarks = reconstruct_geometry_from_raw_observation(raw_obs, self_agent_index)
    controls = np.asarray(ACTION_TO_CONTROL) * DEFAULT_SENSITIVITY * DEFAULT_DT ** 2
    candidate = agents[None] + controls[ACTION_TUPLE_ARRAY]
    distances = np.linalg.norm(candidate[:, :, None] - landmarks[None, None], axis=3)
    assignment_costs = np.stack([
        distances[:, np.arange(3), assignment].sum(axis=1)
        for assignment in ASSIGNMENT_ARRAY
    ], axis=1)
    costs = assignment_costs.min(axis=1) / 3.0
    minimum_separations = np.stack([
        np.linalg.norm(candidate[:, i] - candidate[:, j], axis=1)
        for i, j in ((0, 1), (0, 2), (1, 2))
    ], axis=1).min(axis=1)
    if collision_safe:
        feasible = np.flatnonzero(minimum_separations >= 0.10)
        if len(feasible):
            index = int(feasible[np.argmin(costs[feasible])])
        else:
            best_separation = minimum_separations.max()
            candidates = np.flatnonzero(np.isclose(minimum_separations, best_separation))
            index = int(candidates[np.argmin(costs[candidates])])
    else:
        index = int(np.argmin(costs))
    return ACTION_TUPLES[index]


def local_joint_actions(obs, collision_safe):
    plans = [plan_from_local_observation(obs[index], index, collision_safe) for index in range(3)]
    disagreement = int(any(plan != plans[0] for plan in plans[1:]))
    actions = []
    for agent_index, plan in enumerate(plans):
        value = np.zeros(5, dtype=np.float32); value[plan[agent_index]] = 1.0; actions.append(value)
    return actions, disagreement


def evaluate(episodes, horizon, seed_base, collision_safe):
    env = make_env("simple_spread", discrete_action=True); rows = []
    try:
        for episode in range(episodes):
            seed = seed_base + episode; torch.manual_seed(seed); np.random.seed(seed); env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32); total = 0.0; collision_steps = 0
            disagreements = 0; min_sep = float("inf"); max_coverage = 0
            for _ in range(horizon):
                actions, disagreement = local_joint_actions(obs, collision_safe); disagreements += disagreement
                obs, rewards, dones, _ = env.step(actions); obs = np.asarray(obs, dtype=np.float32); total += float(np.mean(rewards))
                assignment, auc, separation, collision, coverage = geometry(env)
                collision_steps += collision; min_sep = min(min_sep, separation); max_coverage = max(max_coverage, coverage)
                if all(dones): break
            rows.append({"episode": episode, "return": total,
                         "hungarian_assignment_distance": assignment, "coverage_radius_auc": auc,
                         "collision_step_rate": collision_steps / horizon, "plan_disagreement_rate": disagreements / horizon,
                         "minimum_agent_separation": min_sep, "final_coverage": coverage, "max_coverage": max_coverage})
    finally: env.close()
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=500); parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=2_500_000); parser.add_argument("--collision-safe", action="store_true")
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
