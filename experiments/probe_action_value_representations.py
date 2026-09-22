"""Counterfactual one-step probe for action-conditioned representations.

The environment and frozen policies are only queried, never trained.  At each
trajectory state the other two agents' deterministic policy actions are held
fixed while the focal agent is counterfactually assigned each of the five
actions.  Labels are real one-step changes in assignment, radius-AUC, and
minimum separation relative to the focal no-op counterfactual.
"""

from __future__ import annotations

import argparse
import csv
import itertools
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
from utils.active_gsp_v3_features import compute_active_gsp_v3_from_local_obs_batch
from utils.make_env import make_env
from utils.multiscale_spectral_features import multiscale_spectral_action_features
from utils.vector_signal_gsp_features import compute_vector_signal_features_from_local_obs_batch


RADII = np.linspace(0.05, 0.30, 26)
N_ACTIONS = 5
REPRESENTATIONS = (
    "raw_action",
    "raw_action_geom",
    "raw_action_active",
    "raw_action_multiscale",
    "raw_action_vector_geom",
    "raw_action_vector_gsp",
)


def one_hot(index):
    value = np.zeros(N_ACTIONS, dtype=np.float32)
    value[int(index)] = 1.0
    return value


def geometry_metrics(env):
    agents = np.asarray([item.state.p_pos for item in env.world.agents], dtype=np.float64)
    landmarks = np.asarray([item.state.p_pos for item in env.world.landmarks], dtype=np.float64)
    distances = np.linalg.norm(agents[:, None, :] - landmarks[None, :, :], axis=2)
    assignment = min(
        sum(distances[i, permutation[i]] for i in range(3))
        for permutation in itertools.permutations(range(3))
    ) / 3.0
    nearest = distances.min(axis=0)
    radius_curve = np.asarray([(nearest < radius).mean() for radius in RADII])
    radius_auc = float(np.trapz(radius_curve, RADII) / (RADII[-1] - RADII[0]))
    separations = [
        np.linalg.norm(agents[i] - agents[j])
        for i in range(3) for j in range(i + 1, 3)
    ]
    min_separation = float(min(separations))
    collision = int(any(
        separations[k] < env.world.agents[i].size + env.world.agents[j].size
        for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)))
    ))
    return assignment, radius_auc, min_separation, collision


def snapshot_world(env):
    entities = []
    for entity in env.world.entities:
        entities.append({
            "p_pos": np.array(entity.state.p_pos, copy=True),
            "p_vel": None if entity.state.p_vel is None else np.array(entity.state.p_vel, copy=True),
            "c": None if not hasattr(entity.state, "c") or entity.state.c is None else np.array(entity.state.c, copy=True),
        })
    return entities, np.random.get_state()


def restore_world(env, snapshot):
    entities, random_state = snapshot
    for entity, state in zip(env.world.entities, entities):
        entity.state.p_pos = np.array(state["p_pos"], copy=True)
        entity.state.p_vel = None if state["p_vel"] is None else np.array(state["p_vel"], copy=True)
        if hasattr(entity.state, "c"):
            entity.state.c = None if state["c"] is None else np.array(state["c"], copy=True)
    np.random.set_state(random_state)


def descriptor_tensors(obs):
    active = compute_active_gsp_v3_from_local_obs_batch(obs)
    vector_geom = compute_vector_signal_features_from_local_obs_batch(obs, "vector_signal_geom")
    vector_gsp = compute_vector_signal_features_from_local_obs_batch(obs, "vector_signal_gsp")
    multiscale = []
    with torch.no_grad():
        for agent_index in range(3):
            item = torch.as_tensor(obs[agent_index], dtype=torch.float32).view(1, -1)
            multiscale.append(
                multiscale_spectral_action_features(item, agent_index)[0].cpu().numpy()
            )
    return active, np.asarray(multiscale), vector_geom, vector_gsp


def collect_dataset(checkpoints, episodes, horizon, eval_seed_base, output_path):
    rows = {"raw": [], "action": [], "active": [], "multiscale": [],
            "vector_geom": [], "vector_gsp": [], "targets": [],
            "reward_delta": [], "collision": [], "group": [],
            "train_seed": [], "episode": []}
    group_id = 0
    for train_seed, checkpoint in checkpoints:
        model = MADDPG.init_from_save(str(checkpoint))
        model.prep_rollouts(device="cpu")
        env = make_env("simple_spread", discrete_action=True)
        try:
            for episode in range(episodes):
                test_seed = eval_seed_base + train_seed * 10000 + episode
                torch.manual_seed(test_seed); np.random.seed(test_seed); env.seed(test_seed)
                obs = np.asarray(env.reset(), dtype=np.float32)
                for _ in range(horizon):
                    tensors = [torch.as_tensor(item).view(1, -1) for item in obs]
                    with torch.no_grad():
                        policy_actions = model.step(tensors, explore=False)
                    policy_numpy = [item.cpu().numpy().ravel() for item in policy_actions]
                    active, multiscale, vector_geom, vector_gsp = descriptor_tensors(obs)
                    snapshot = snapshot_world(env)
                    for focal in range(3):
                        outcomes = []
                        for candidate in range(N_ACTIONS):
                            restore_world(env, snapshot)
                            actions = [np.array(item, copy=True) for item in policy_numpy]
                            actions[focal] = one_hot(candidate)
                            _, rewards, _, _ = env.step(actions)
                            outcomes.append(geometry_metrics(env) + (float(np.mean(rewards)),))
                        reference = np.asarray(outcomes[0][:3], dtype=np.float64)
                        for candidate, outcome in enumerate(outcomes):
                            rows["raw"].append(obs[focal])
                            rows["action"].append(one_hot(candidate))
                            rows["active"].append(active[focal, candidate])
                            rows["multiscale"].append(multiscale[focal, candidate])
                            rows["vector_geom"].append(vector_geom[focal, candidate])
                            rows["vector_gsp"].append(vector_gsp[focal, candidate])
                            rows["targets"].append(np.asarray(outcome[:3]) - reference)
                            rows["reward_delta"].append(outcome[4] - outcomes[0][4])
                            rows["collision"].append(outcome[3])
                            rows["group"].append(group_id)
                            rows["train_seed"].append(train_seed)
                            rows["episode"].append(episode)
                        group_id += 1
                    restore_world(env, snapshot)
                    obs, _, dones, _ = env.step(policy_numpy)
                    obs = np.asarray(obs, dtype=np.float32)
                    if all(dones):
                        break
        finally:
            env.close()
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    np.savez_compressed(output_path, **arrays)
    return arrays


def representation_matrix(data, name):
    base = [data["raw"], data["action"]]
    if name == "raw_action_geom":
        base.append(data["active"][:, :2])
    elif name == "raw_action_active":
        base.append(data["active"])
    elif name == "raw_action_multiscale":
        base.append(data["multiscale"])
    elif name == "raw_action_vector_geom":
        base.append(data["vector_geom"])
    elif name == "raw_action_vector_gsp":
        base.append(data["vector_gsp"])
    elif name != "raw_action":
        raise ValueError(name)
    return np.concatenate(base, axis=1).astype(np.float32)


class Probe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3),
        )

    def forward(self, value):
        return self.network(value)


def pearson(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def selection_regret(predictions, targets, groups, target_index, maximize):
    regrets = []
    for group in np.unique(groups):
        mask = groups == group
        predicted = predictions[mask, target_index]
        actual = targets[mask, target_index]
        selected = np.argmax(predicted) if maximize else np.argmin(predicted)
        oracle = np.max(actual) if maximize else np.min(actual)
        achieved = actual[selected]
        regrets.append((oracle - achieved) if maximize else (achieved - oracle))
    return float(np.mean(regrets))


def train_probe(data, representation, model_seed, heldout_seed, epochs, threads,
                return_bundle=False):
    torch.manual_seed(model_seed); np.random.seed(model_seed); torch.set_num_threads(threads)
    x = representation_matrix(data, representation)
    y = data["targets"].astype(np.float32)
    train_mask = data["train_seed"] != heldout_seed
    test_mask = data["train_seed"] == heldout_seed
    train_episodes = data["episode"][train_mask]
    validation_mask = train_mask.copy()
    validation_mask[train_mask] = (train_episodes % 5) == 0
    fit_mask = train_mask & ~validation_mask

    x_mean = x[fit_mask].mean(0); x_std = x[fit_mask].std(0).clip(1e-6)
    y_mean = y[fit_mask].mean(0); y_std = y[fit_mask].std(0).clip(1e-8)
    x_tensor = torch.from_numpy((x - x_mean) / x_std)
    y_tensor = torch.from_numpy((y - y_mean) / y_std)
    fit_indices = torch.from_numpy(np.flatnonzero(fit_mask))
    validation_indices = torch.from_numpy(np.flatnonzero(validation_mask))
    model = Probe(x.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state = None; best_loss = math.inf; patience = 0
    generator = torch.Generator().manual_seed(model_seed)
    for epoch in range(epochs):
        permutation = fit_indices[torch.randperm(len(fit_indices), generator=generator)]
        model.train()
        for start in range(0, len(permutation), 1024):
            index = permutation[start:start + 1024]
            loss = (model(x_tensor[index]) - y_tensor[index]).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = float((model(x_tensor[validation_indices]) - y_tensor[validation_indices]).square().mean())
        if loss < best_loss - 1e-5:
            best_loss = loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 7:
                break
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        predictions = model(x_tensor[test_mask]).numpy() * y_std + y_mean
    actual = y[test_mask]
    groups = data["group"][test_mask]
    output = {"representation": representation, "model_seed": model_seed,
              "heldout_train_seed": heldout_seed, "epochs_run": epoch + 1,
              "input_dim": x.shape[1], "test_rows": int(test_mask.sum())}
    names = ("hungarian_delta", "radius_auc_delta", "min_separation_delta")
    for index, name in enumerate(names):
        residual = actual[:, index] - predictions[:, index]
        denominator = np.sum((actual[:, index] - actual[:, index].mean()) ** 2)
        output[f"{name}_r2"] = float(1.0 - np.sum(residual ** 2) / max(denominator, 1e-20))
        output[f"{name}_pearson"] = pearson(actual[:, index], predictions[:, index])
    output["hungarian_selection_regret"] = selection_regret(predictions, actual, groups, 0, False)
    output["radius_auc_selection_regret"] = selection_regret(predictions, actual, groups, 1, True)
    if return_bundle:
        bundle = {
            "model": model,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
        }
        return output, bundle
    return output


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-root", default="experiments/vector_signal_gsp_locked_screen/run_20260711_231009")
    parser.add_argument("--output-dir", default="experiments/action_value_representation_probe")
    parser.add_argument("--train-seeds", default="21,22,23")
    parser.add_argument("--heldout-seed", type=int, default=23)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--eval-seed-base", type=int, default=1_400_000)
    parser.add_argument("--model-seeds", default="1,2,3")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = []
    for seed in [int(value) for value in args.train_seeds.split(",")]:
        path = Path(args.screen_root) / "action_aware_geometric_control" / f"seed_{seed}" / "checkpoints" / "model_final_100000.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        checkpoints.append((seed, path))
    dataset_path = output_dir / "counterfactual_dataset.npz"
    data = collect_dataset(checkpoints, args.episodes, args.horizon, args.eval_seed_base, dataset_path)
    if args.collect_only:
        (output_dir / "metadata.json").write_text(json.dumps({
            "checkpoints": [{"train_seed": seed, "path": str(path.resolve())} for seed, path in checkpoints],
            "episodes_per_checkpoint": args.episodes, "horizon": args.horizon,
            "counterfactual_boundary": "other agents fixed; focal action varied; real environment stepped once",
            "contains_reward_delta": True, "policy_parameters_changed": False,
        }, indent=2) + "\n", encoding="utf-8")
        print(output_dir)
        return
    results = []
    for representation in REPRESENTATIONS:
        for model_seed in [int(value) for value in args.model_seeds.split(",")]:
            results.append(train_probe(
                data, representation, model_seed, args.heldout_seed, args.epochs, args.threads
            ))
    write_csv(output_dir / "probe_results.csv", results)
    summary = []
    for representation in REPRESENTATIONS:
        matching = [row for row in results if row["representation"] == representation]
        row = {"representation": representation, "model_runs": len(matching)}
        for key in matching[0]:
            if key not in {"representation", "model_seed", "heldout_train_seed"}:
                values = np.asarray([item[key] for item in matching], dtype=np.float64)
                row[f"{key}_mean"] = values.mean(); row[f"{key}_std"] = values.std(ddof=1)
        summary.append(row)
    write_csv(output_dir / "probe_summary.csv", summary)
    (output_dir / "metadata.json").write_text(json.dumps({
        "checkpoints": [{"train_seed": seed, "path": str(path.resolve())} for seed, path in checkpoints],
        "episodes_per_checkpoint": args.episodes, "horizon": args.horizon,
        "heldout_training_seed": args.heldout_seed,
        "counterfactual_boundary": "other agents' deterministic actions fixed; focal action varied; real environment stepped once",
        "policy_parameters_changed": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
