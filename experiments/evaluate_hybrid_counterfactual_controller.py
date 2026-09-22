"""Lexicographic reward-feasible Hungarian controller without tuned weights."""

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

from experiments.evaluate_supervised_probe_policies import geometry
from experiments.probe_action_value_representations import descriptor_tensors, representation_matrix, train_probe
from experiments.probe_and_evaluate_reward_value import candidate_data, train as train_reward
from utils.make_env import make_env


def actions_for_obs(obs, representation, hungarian, reward):
    descriptors = descriptor_tensors(obs); actions = []
    with torch.no_grad():
        for focal in range(3):
            data = candidate_data(obs, focal, descriptors)
            x = representation_matrix(data, representation)
            hx = torch.from_numpy((x - hungarian["x_mean"]) / hungarian["x_std"])
            h = hungarian["model"](hx).numpy()[:, 0] * hungarian["y_std"][0] + hungarian["y_mean"][0]
            rx = torch.from_numpy((x - reward["x_mean"]) / reward["x_std"])
            r = reward["model"](rx).numpy() * reward["y_std"] + reward["y_mean"]
            r = r - r[0]
            feasible = np.flatnonzero(r >= 0.0)
            index = int(feasible[np.argmin(h[feasible])])
            value = np.zeros(5, dtype=np.float32); value[index] = 1.0; actions.append(value)
    return actions


def evaluate(data, representation, model_seed, episodes, horizon, seed_base):
    _, hungarian = train_probe(data, representation, model_seed, 23, 50, 6, True)
    _, reward = train_reward(data, representation, model_seed)
    env = make_env("simple_spread", discrete_action=True); rows = []
    try:
        for episode in range(episodes):
            seed = seed_base + episode; torch.manual_seed(seed); np.random.seed(seed); env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32); total = 0.0; collisions = 0
            min_sep = float("inf"); max_coverage = 0
            for _ in range(horizon):
                obs, rewards, dones, _ = env.step(actions_for_obs(obs, representation, hungarian, reward))
                obs = np.asarray(obs, dtype=np.float32); total += float(np.mean(rewards))
                assignment, auc, separation, collision, coverage = geometry(env)
                collisions += collision; min_sep = min(min_sep, separation); max_coverage = max(max_coverage, coverage)
                if all(dones): break
            rows.append({"representation": representation, "episode": episode, "return": total,
                         "hungarian_assignment_distance": assignment, "coverage_radius_auc": auc,
                         "collision_step_rate": collisions / horizon, "minimum_agent_separation": min_sep,
                         "final_coverage": coverage, "max_coverage": max_coverage})
    finally: env.close()
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="experiments/hybrid_counterfactual_controller_20260712")
    parser.add_argument("--dataset", default="experiments/reward_probe_dataset_20260712/counterfactual_dataset.npz")
    parser.add_argument("--episodes", type=int, default=500); parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=2_600_000); parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument("--representations", default="raw_action_geom,raw_action_active")
    args = parser.parse_args(); output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    with np.load(args.dataset) as loaded: data = {key: loaded[key] for key in loaded.files}
    rows = []
    representations = args.representations.split(",")
    for representation in representations:
        rows += evaluate(data, representation, args.model_seed, args.episodes, args.horizon, args.seed_base)
    with (output / "per_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = []
    for representation in representations:
        selected = [row for row in rows if row["representation"] == representation]
        result = {"representation": representation, "episodes": len(selected)}
        for key in selected[0]:
            if key not in {"representation", "episode"}:
                values = np.asarray([row[key] for row in selected]); result[key + "_mean"] = values.mean(); result[key + "_std"] = values.std(ddof=1)
        summary.append(result)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (output / "metadata.json").write_text(json.dumps({
        "rule": "minimize predicted Hungarian delta among actions with predicted original-reward delta >= no-op; no tuned weight",
        "diagnostic_only": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__": main()
