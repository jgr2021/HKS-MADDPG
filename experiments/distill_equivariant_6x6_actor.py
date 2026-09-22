"""Offline gate for the permutation-equivariant 6x6 matching/safety actor."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.distill_scalable_6x6_actor import collect, evaluate
from utils.networks import EquivariantMatchingSafety6x6Policy


def train_actor(observations, labels, agent_indices, episode_ids, seed, epochs,
                initial_state=None):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(6)
    observations = torch.from_numpy(observations.astype(np.float32))
    labels_tensor = torch.from_numpy(labels.astype(np.int64))
    agent_tensor = torch.from_numpy(agent_indices.astype(np.int64))
    validation = (episode_ids % 5) == 0
    fit_indices = torch.from_numpy(np.flatnonzero(~validation))
    validation_indices = torch.from_numpy(np.flatnonzero(validation))
    policy = EquivariantMatchingSafety6x6Policy(36, 5, agent_index=0)
    if initial_state is not None:
        policy.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_loss, patience = None, math.inf, 0
    history = []
    for epoch in range(epochs):
        order = fit_indices[torch.randperm(len(fit_indices), generator=generator)]
        policy.train()
        for start in range(0, len(order), 512):
            index = order[start:start + 512]
            logits = policy.forward_with_agent_indices(
                observations[index], agent_tensor[index]
            )
            loss = F.cross_entropy(logits, labels_tensor[index])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        policy.eval()
        with torch.no_grad():
            logits = policy.forward_with_agent_indices(
                observations[validation_indices], agent_tensor[validation_indices]
            )
            value = float(F.cross_entropy(logits, labels_tensor[validation_indices]))
            accuracy = float((logits.argmax(1) == labels_tensor[validation_indices]).float().mean())
        history.append({"epoch": epoch + 1, "validation_loss": value,
                        "validation_accuracy": accuracy})
        if value < best_loss - 1e-5:
            best_loss = value
            best_state = {key: tensor.detach().clone() for key, tensor in policy.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                break
    policy.load_state_dict(best_state); policy.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(observations), 1024):
            stop = start + 1024
            predictions.append(policy.forward_with_agent_indices(
                observations[start:stop], agent_tensor[start:stop]
            ).argmax(1).numpy())
    predictions = np.concatenate(predictions)
    metrics = {
        "samples": int(len(labels)), "episodes": int(np.unique(episode_ids).size),
        "epochs_run": len(history), "validation_loss": best_loss,
        "fit_accuracy": float((predictions[~validation] == labels[~validation]).mean()),
        "validation_accuracy": float((predictions[validation] == labels[validation]).mean()),
        "per_agent_accuracy": [
            float((predictions[agent_indices == index] == labels[agent_indices == index]).mean())
            for index in range(6)
        ],
        "action_distribution": np.bincount(labels, minlength=5).tolist(),
        "matching_row_error": policy.last_matching_row_error,
        "matching_column_error": policy.last_matching_column_error,
    }
    return policy.state_dict(), metrics, history


def build_policies(state):
    policies = []
    for index in range(6):
        policy = EquivariantMatchingSafety6x6Policy(36, 5, agent_index=index)
        policy.load_state_dict(state); policy.eval(); policies.append(policy)
    return policies


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/equivariant_6x6_offline_gate_20260713")
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-seed-base", type=int, default=5_800_000)
    parser.add_argument("--eval-seed-base", type=int, default=5_900_000)
    parser.add_argument("--dataset", default="")
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    if args.dataset:
        with np.load(args.dataset) as loaded:
            observations = loaded["observations"]
            labels = loaded["labels"]
            agent_indices = loaded["agent_indices"]
            episode_ids = loaded["episode_ids"]
    else:
        observations, labels, agent_indices, episode_ids = collect(
            args.train_episodes, args.horizon, args.train_seed_base
        )
        np.savez_compressed(
            output / "training_dataset.npz", observations=observations,
            labels=labels, agent_indices=agent_indices, episode_ids=episode_ids,
        )
    state, training_metrics, history = train_actor(
        observations, labels, agent_indices, episode_ids, args.seed, args.epochs
    )
    payload = {"actor_state_dict": state, "target": "independent_safe_hybrid",
               "training_metrics": training_metrics}
    torch.save(payload, output / "equivariant_6x6_actor.pt")
    policies = build_policies(state)
    rows = evaluate("teacher", policies, args.eval_episodes, args.horizon, args.eval_seed_base)
    rows += evaluate("student", policies, args.eval_episodes, args.horizon, args.eval_seed_base)
    write_csv(output / "per_episode.csv", rows); write_csv(output / "training_history.csv", history)
    summary = []
    for controller in ("teacher", "student"):
        selected = [row for row in rows if row["controller"] == controller]
        result = {"controller": controller, "episodes": len(selected)}
        for key in selected[0]:
            if key not in {"controller", "episode", "test_seed"}:
                values = np.asarray([row[key] for row in selected], dtype=np.float64)
                result[key + "_mean"] = float(values.mean())
                result[key + "_std"] = float(values.std(ddof=1))
        summary.append(result)
    write_csv(output / "summary.csv", summary)
    teacher = [row for row in rows if row["controller"] == "teacher"]
    student = [row for row in rows if row["controller"] == "student"]
    paired = []
    for metric in ("hungarian_assignment_distance", "coverage_radius_auc",
                   "collision_step_rate", "return", "minimum_agent_separation",
                   "final_coverage", "max_coverage"):
        delta = np.asarray([b[metric] - a[metric] for a, b in zip(teacher, student)])
        half = 1.9647293909876649 * delta.std(ddof=1) / np.sqrt(len(delta))
        paired.append({"metric": metric, "mean_delta": float(delta.mean()),
                       "ci95_low": float(delta.mean() - half),
                       "ci95_high": float(delta.mean() + half)})
    write_csv(output / "paired_intervals.csv", paired)
    lookup = {row["metric"]: row for row in paired}
    offline_pass = (
        training_metrics["validation_accuracy"] >= 0.85
        and lookup["collision_step_rate"]["ci95_low"] <= 0.0
        and lookup["coverage_radius_auc"]["mean_delta"] >= -0.03
    )
    (output / "metadata.json").write_text(json.dumps({
        "architecture": "learned pair encoder + learned Sinkhorn matching + learned safety set encoder",
        "fixed_feature_to_action_signs": False, "handcrafted_action_prior": False,
        "training": training_metrics, "offline_gate_pass": offline_pass,
        "gate": "validation accuracy >= .85; no resolved collision penalty; AUC delta >= -.03",
        "online_training_authorized_by_gate": offline_pass,
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
