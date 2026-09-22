"""Roll out frozen local probes to separate prediction from control utility."""

from __future__ import annotations

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

from algorithms.maddpg import MADDPG
from experiments.probe_action_value_representations import (
    N_ACTIONS,
    descriptor_tensors,
    representation_matrix,
    train_probe,
)
from utils.make_env import make_env


REPRESENTATIONS = (
    "raw_action", "raw_action_geom", "raw_action_active",
    "raw_action_multiscale", "raw_action_vector_geom", "raw_action_vector_gsp",
)
RADII = np.linspace(0.05, 0.30, 26)


def geometry(env):
    agents = np.asarray([item.state.p_pos for item in env.world.agents])
    landmarks = np.asarray([item.state.p_pos for item in env.world.landmarks])
    distances = np.linalg.norm(agents[:, None] - landmarks[None], axis=2)
    nearest = distances.min(axis=0)
    assignment = min(
        sum(distances[i, permutation[i]] for i in range(3))
        for permutation in itertools.permutations(range(3))
    ) / 3.0
    radius_auc = np.trapz(
        [(nearest < radius).mean() for radius in RADII], RADII
    ) / (RADII[-1] - RADII[0])
    separations = [np.linalg.norm(agents[i] - agents[j]) for i in range(3) for j in range(i + 1, 3)]
    collision = int(any(
        separations[k] < env.world.agents[i].size + env.world.agents[j].size
        for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)))
    ))
    return float(assignment), float(radius_auc), float(min(separations)), collision, int((nearest < .10).sum())


def candidate_rows(obs, focal, active, multiscale, vector_geom, vector_gsp):
    count = N_ACTIONS
    return {
        "raw": np.repeat(obs[focal:focal + 1], count, axis=0),
        "action": np.eye(N_ACTIONS, dtype=np.float32),
        "active": active[focal],
        "multiscale": multiscale[focal],
        "vector_geom": vector_geom[focal],
        "vector_gsp": vector_gsp[focal],
    }


def probe_actions(obs, representation, bundle, objective):
    active, multiscale, vector_geom, vector_gsp = descriptor_tensors(obs)
    actions = []
    model = bundle["model"]
    with torch.no_grad():
        for focal in range(3):
            data = candidate_rows(obs, focal, active, multiscale, vector_geom, vector_gsp)
            x = representation_matrix(data, representation)
            x = (x - bundle["x_mean"]) / bundle["x_std"]
            prediction = model(torch.from_numpy(x)).numpy() * bundle["y_std"] + bundle["y_mean"]
            if objective == "hungarian":
                index = int(np.argmin(prediction[:, 0]))
            elif objective == "auc":
                index = int(np.argmax(prediction[:, 1]))
            elif objective == "hungarian_safe":
                others = obs[focal, 10:14].reshape(2, 2)
                current_min_separation = min(
                    np.linalg.norm(others[0]), np.linalg.norm(others[1]),
                    np.linalg.norm(others[0] - others[1]),
                )
                if current_min_separation < 0.10:
                    index = int(np.argmax(prediction[:, 2]))
                else:
                    nondecreasing = np.flatnonzero(prediction[:, 2] >= 0.0)
                    index = int(
                        nondecreasing[np.argmin(prediction[nondecreasing, 0])]
                        if len(nondecreasing) else np.argmax(prediction[:, 2])
                    )
            else:
                raise ValueError(f"Unknown objective: {objective}")
            action = np.zeros(N_ACTIONS, dtype=np.float32); action[index] = 1.0
            actions.append(action)
    return actions


def evaluate_probe(representation, objective, bundle, episodes, horizon, seed_base):
    env = make_env("simple_spread", discrete_action=True); rows = []
    try:
        for episode in range(episodes):
            seed = seed_base + episode; torch.manual_seed(seed); np.random.seed(seed); env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32); total_return = 0.0
            collision_steps = 0; min_separation = float("inf"); max_coverage = 0
            for _ in range(horizon):
                actions = probe_actions(obs, representation, bundle, objective)
                obs, rewards, dones, _ = env.step(actions); obs = np.asarray(obs, dtype=np.float32)
                total_return += float(np.mean(rewards))
                assignment, auc, separation, collision, coverage = geometry(env)
                collision_steps += collision; min_separation = min(min_separation, separation)
                max_coverage = max(max_coverage, coverage)
                if all(dones): break
            rows.append({"policy": f"{representation}_{objective}", "episode": episode,
                         "return": total_return, "hungarian_assignment_distance": assignment,
                         "coverage_radius_auc": auc, "collision_step_rate": collision_steps / horizon,
                         "minimum_agent_separation": min_separation, "final_coverage": coverage,
                         "max_coverage": max_coverage})
    finally:
        env.close()
    return rows


def evaluate_baseline(checkpoint, episodes, horizon, seed_base):
    model = MADDPG.init_from_save(str(checkpoint)); model.prep_rollouts(device="cpu")
    env = make_env("simple_spread", discrete_action=True); rows = []
    try:
        for episode in range(episodes):
            seed = seed_base + episode; torch.manual_seed(seed); np.random.seed(seed); env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32); total_return = 0.0
            collision_steps = 0; min_separation = float("inf"); max_coverage = 0
            for _ in range(horizon):
                with torch.no_grad():
                    actions = model.step([torch.as_tensor(item).view(1, -1) for item in obs], explore=False)
                obs, rewards, dones, _ = env.step([item.numpy().ravel() for item in actions])
                obs = np.asarray(obs, dtype=np.float32); total_return += float(np.mean(rewards))
                assignment, auc, separation, collision, coverage = geometry(env)
                collision_steps += collision; min_separation = min(min_separation, separation)
                max_coverage = max(max_coverage, coverage)
                if all(dones): break
            rows.append({"policy": "trained_geometric_control", "episode": episode,
                         "return": total_return, "hungarian_assignment_distance": assignment,
                         "coverage_radius_auc": auc, "collision_step_rate": collision_steps / horizon,
                         "minimum_agent_separation": min_separation, "final_coverage": coverage,
                         "max_coverage": max_coverage})
    finally:
        env.close()
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", default="experiments/action_value_representation_probe_20260712")
    parser.add_argument("--output-dir", default="experiments/supervised_probe_policy_eval_20260712")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=1_600_000)
    parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument("--heldout-seed", type=int, default=23)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--representations", default=",".join(REPRESENTATIONS))
    parser.add_argument("--objectives", default="hungarian,auc")
    args = parser.parse_args()
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(Path(args.probe_root) / "counterfactual_dataset.npz") as loaded:
        data = {key: loaded[key] for key in loaded.files}
    rows = []; fit_rows = []
    for representation in args.representations.split(","):
        fit, bundle = train_probe(data, representation, args.model_seed, args.heldout_seed, args.epochs, args.threads, True)
        fit_rows.append(fit)
        for objective in args.objectives.split(","):
            rows += evaluate_probe(representation, objective, bundle, args.episodes, args.horizon, args.seed_base)
    baseline = Path("experiments/vector_signal_gsp_locked_screen/run_20260711_231009/action_aware_geometric_control/seed_23/checkpoints/model_final_100000.pt")
    rows += evaluate_baseline(baseline, args.episodes, args.horizon, args.seed_base)
    summary = []
    metrics = ("return", "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate",
               "minimum_agent_separation", "final_coverage", "max_coverage")
    for policy in sorted({row["policy"] for row in rows}):
        selected = [row for row in rows if row["policy"] == policy]
        output = {"policy": policy, "episodes": len(selected)}
        for metric in metrics:
            values = np.asarray([row[metric] for row in selected])
            output[f"{metric}_mean"] = values.mean(); output[f"{metric}_std"] = values.std(ddof=1)
        summary.append(output)
    write_csv(output_dir / "per_episode.csv", rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "probe_fit.csv", fit_rows)
    (output_dir / "metadata.json").write_text(json.dumps({
        "diagnostic_only": True, "decentralized_local_execution": True,
        "environment_reward_critic_replay_unchanged": True,
        "training_labels": "one-step real-environment counterfactuals from seeds 21-22",
        "evaluation_seed_base": args.seed_base,
    }, indent=2) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
