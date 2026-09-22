"""Predict and deploy the original one-step environment reward counterfactual."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.maddpg import MADDPG
from experiments.evaluate_supervised_probe_policies import geometry
from experiments.probe_action_value_representations import descriptor_tensors, representation_matrix
from utils.make_env import make_env


REPRESENTATIONS = (
    "raw_action", "raw_action_geom", "raw_action_active",
    "raw_action_multiscale", "raw_action_vector_geom", "raw_action_vector_gsp",
)


class RewardProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1),
        )

    def forward(self, value):
        return self.network(value).squeeze(1)


def candidate_data(obs, focal, descriptors):
    active, multiscale, vector_geom, vector_gsp = descriptors
    return {
        "raw": np.repeat(obs[focal:focal + 1], 5, axis=0),
        "action": np.eye(5, dtype=np.float32),
        "active": active[focal], "multiscale": multiscale[focal],
        "vector_geom": vector_geom[focal], "vector_gsp": vector_gsp[focal],
    }


def train(data, representation, model_seed, epochs=50, threads=6,
          heldout_seed=23):
    torch.manual_seed(model_seed); np.random.seed(model_seed); torch.set_num_threads(threads)
    x = representation_matrix(data, representation)
    y = data["reward_delta"].astype(np.float32)
    train_mask = data["train_seed"] != heldout_seed
    test_mask = data["train_seed"] == heldout_seed
    validation = train_mask & ((data["episode"] % 5) == 0); fit = train_mask & ~validation
    xm = x[fit].mean(0); xs = x[fit].std(0).clip(1e-6)
    ym = y[fit].mean(); ys = max(float(y[fit].std()), 1e-8)
    xt = torch.from_numpy((x - xm) / xs); yt = torch.from_numpy((y - ym) / ys)
    fit_idx = torch.from_numpy(np.flatnonzero(fit)); val_idx = torch.from_numpy(np.flatnonzero(validation))
    model = RewardProbe(x.shape[1]); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(model_seed); best = None; best_loss = math.inf; patience = 0
    for epoch in range(epochs):
        order = fit_idx[torch.randperm(len(fit_idx), generator=generator)]; model.train()
        for start in range(0, len(order), 1024):
            index = order[start:start + 1024]
            loss = (model(xt[index]) - yt[index]).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): loss = float((model(xt[val_idx]) - yt[val_idx]).square().mean())
        if loss < best_loss - 1e-5:
            best_loss = loss; best = {key: value.detach().clone() for key, value in model.state_dict().items()}; patience = 0
        else:
            patience += 1
            if patience >= 7: break
    model.load_state_dict(best); model.eval()
    with torch.no_grad(): prediction = model(xt[test_mask]).numpy() * ys + ym
    actual = y[test_mask]
    denom = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - np.sum((actual - prediction) ** 2) / max(denom, 1e-20)
    corr = np.corrcoef(actual, prediction)[0, 1]
    regrets = []
    groups = data["group"][test_mask]
    for group in np.unique(groups):
        mask = groups == group; chosen = np.argmax(prediction[mask]); values = actual[mask]
        regrets.append(values.max() - values[chosen])
    metrics = {"representation": representation, "model_seed": model_seed,
               "heldout_train_seed": heldout_seed,
               "reward_delta_r2": r2, "reward_delta_pearson": corr,
               "reward_selection_regret": np.mean(regrets), "epochs_run": epoch + 1}
    return metrics, {"model": model, "x_mean": xm, "x_std": xs, "y_mean": ym, "y_std": ys}


def rollout(representation, bundle, episodes, horizon, seed_base):
    env = make_env("simple_spread", discrete_action=True); rows = []
    try:
        for episode in range(episodes):
            seed = seed_base + episode; torch.manual_seed(seed); np.random.seed(seed); env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32); total = 0.0; collision_steps = 0
            min_sep = float("inf"); max_coverage = 0
            for _ in range(horizon):
                descriptors = descriptor_tensors(obs); actions = []
                with torch.no_grad():
                    for focal in range(3):
                        x = representation_matrix(candidate_data(obs, focal, descriptors), representation)
                        prediction = bundle["model"](torch.from_numpy((x - bundle["x_mean"]) / bundle["x_std"])).numpy()
                        index = int(np.argmax(prediction)); action = np.zeros(5, dtype=np.float32); action[index] = 1
                        actions.append(action)
                obs, rewards, dones, _ = env.step(actions); obs = np.asarray(obs, dtype=np.float32)
                total += float(np.mean(rewards))
                assignment, auc, separation, collision, coverage = geometry(env)
                collision_steps += collision; min_sep = min(min_sep, separation); max_coverage = max(max_coverage, coverage)
                if all(dones): break
            rows.append({"policy": representation + "_reward", "episode": episode, "return": total,
                         "hungarian_assignment_distance": assignment, "coverage_radius_auc": auc,
                         "collision_step_rate": collision_steps / horizon, "minimum_agent_separation": min_sep,
                         "final_coverage": coverage, "max_coverage": max_coverage})
    finally: env.close()
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="experiments/reward_probe_dataset_20260712/counterfactual_dataset.npz")
    parser.add_argument("--output-dir", default="experiments/reward_value_probe_eval_20260712")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=2_100_000)
    parser.add_argument("--representations", default=",".join(REPRESENTATIONS))
    parser.add_argument("--probe-model-seeds", default="1,2,3")
    parser.add_argument("--deployment-model-seed", type=int, default=1)
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    with np.load(args.dataset) as loaded: data = {key: loaded[key] for key in loaded.files}
    probe_rows = []; bundles = {}
    representations = args.representations.split(",")
    model_seeds = [int(value) for value in args.probe_model_seeds.split(",")]
    for representation in representations:
        for seed in model_seeds:
            metrics, bundle = train(data, representation, seed); probe_rows.append(metrics)
            if seed == args.deployment_model_seed: bundles[representation] = bundle
        if representation not in bundles:
            raise ValueError("deployment-model-seed must be included in probe-model-seeds")
    episode_rows = []
    for representation in representations:
        episode_rows += rollout(representation, bundles[representation], args.episodes, args.horizon, args.seed_base)
    summary = []
    for policy in sorted({row["policy"] for row in episode_rows}):
        selected = [row for row in episode_rows if row["policy"] == policy]
        result = {"policy": policy, "episodes": len(selected)}
        for metric in ("return", "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate",
                       "minimum_agent_separation", "final_coverage", "max_coverage"):
            values = np.asarray([row[metric] for row in selected]); result[f"{metric}_mean"] = values.mean(); result[f"{metric}_std"] = values.std(ddof=1)
        summary.append(result)
    write_csv(output / "probe_results.csv", probe_rows); write_csv(output / "per_episode.csv", episode_rows); write_csv(output / "summary.csv", summary)
    (output / "metadata.json").write_text(json.dumps({
        "target": "original environment mean-reward delta relative to focal no-op",
        "heldout_training_seed": 23, "diagnostic_only": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__": main()
