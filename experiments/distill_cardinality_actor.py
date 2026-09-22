"""Distill one independent equivariant actor for an NxN spread task."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.cardinality_coordination import (
    env_id_for_cardinality,
    evaluate_controller,
    observation_dim,
    policy_actions,
    teacher_actions,
)
from utils.make_env import make_env
from utils.networks import EquivariantMatchingSafetyPolicy8


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(
    n_agents,
    episodes,
    horizon,
    seed_base,
    rollout_policies=None,
):
    env = make_env(env_id_for_cardinality(n_agents), discrete_action=True)
    observations, labels, agent_indices, episode_ids = [], [], [], []
    try:
        for episode in range(episodes):
            seed = seed_base + episode
            torch.manual_seed(seed)
            np.random.seed(seed)
            env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            for _ in range(horizon):
                teacher = teacher_actions(obs, n_agents)
                observations.extend(obs)
                labels.extend(int(np.argmax(action)) for action in teacher)
                agent_indices.extend(range(n_agents))
                episode_ids.extend([episode] * n_agents)
                actions = (
                    teacher
                    if rollout_policies is None
                    else policy_actions(rollout_policies, obs)
                )
                obs, _, dones, _ = env.step(actions)
                obs = np.asarray(obs, dtype=np.float32)
                if all(dones):
                    break
    finally:
        env.close()
    return tuple(np.asarray(value) for value in (
        observations, labels, agent_indices, episode_ids
    ))


def train_actor(
    observations,
    labels,
    agent_indices,
    episode_ids,
    n_agents,
    seed,
    epochs,
    initial_state=None,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(6)
    observations_tensor = torch.from_numpy(observations.astype(np.float32))
    labels_tensor = torch.from_numpy(labels.astype(np.int64))
    agent_tensor = torch.from_numpy(agent_indices.astype(np.int64))
    validation = (episode_ids % 5) == 0
    fit_indices = torch.from_numpy(np.flatnonzero(~validation))
    validation_indices = torch.from_numpy(np.flatnonzero(validation))
    policy = EquivariantMatchingSafetyPolicy8(
        observation_dim(n_agents), 5, agent_index=0
    )
    if initial_state is not None:
        policy.load_state_dict(initial_state)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_loss, patience = None, math.inf, 0
    history = []
    for epoch in range(epochs):
        order = fit_indices[
            torch.randperm(len(fit_indices), generator=generator)
        ]
        policy.train()
        batch_losses = []
        for start in range(0, len(order), 512):
            index = order[start:start + 512]
            logits = policy.forward_with_agent_indices(
                observations_tensor[index], agent_tensor[index]
            )
            loss = F.cross_entropy(logits, labels_tensor[index])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(float(loss.detach()))
        policy.eval()
        with torch.no_grad():
            logits = policy.forward_with_agent_indices(
                observations_tensor[validation_indices],
                agent_tensor[validation_indices],
            )
            validation_loss = float(F.cross_entropy(
                logits, labels_tensor[validation_indices]
            ))
            validation_accuracy = float(
                (logits.argmax(1) == labels_tensor[validation_indices])
                .float().mean()
            )
        history.append({
            "epoch": epoch + 1,
            "fit_loss": float(np.mean(batch_losses)),
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {
                key: value.detach().clone()
                for key, value in policy.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                break
    if best_state is None:
        raise RuntimeError("Distillation did not produce a finite checkpoint")
    policy.load_state_dict(best_state)
    policy.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(observations_tensor), 1024):
            stop = start + 1024
            predictions.append(policy.forward_with_agent_indices(
                observations_tensor[start:stop], agent_tensor[start:stop]
            ).argmax(1).cpu().numpy())
    predictions = np.concatenate(predictions)
    metrics = {
        "samples": int(len(labels)),
        "episodes": int(np.unique(episode_ids).size),
        "epochs_run": len(history),
        "validation_loss": best_loss,
        "fit_accuracy": float((predictions[~validation] == labels[~validation]).mean()),
        "validation_accuracy": float((predictions[validation] == labels[validation]).mean()),
        "per_agent_accuracy": [
            float((predictions[agent_indices == index]
                   == labels[agent_indices == index]).mean())
            for index in range(n_agents)
        ],
        "action_distribution": np.bincount(labels, minlength=5).tolist(),
        "prediction_distribution": np.bincount(
            predictions, minlength=5
        ).tolist(),
        "matching_row_error": policy.last_matching_row_error,
        "matching_column_error": policy.last_matching_column_error,
    }
    return best_state, metrics, history


def build_policies(state, n_agents):
    policies = []
    for index in range(n_agents):
        policy = EquivariantMatchingSafetyPolicy8(
            observation_dim(n_agents), 5, agent_index=index
        )
        policy.load_state_dict(state)
        policy.eval()
        policies.append(policy)
    return policies


def paired_offline_intervals(rows):
    teacher = {
        int(row["test_seed"]): row
        for row in rows if row["controller"] == "teacher"
    }
    student = {
        int(row["test_seed"]): row
        for row in rows if row["controller"] == "student"
    }
    seeds = sorted(set(teacher) & set(student))
    if len(seeds) != len(teacher) or len(seeds) != len(student):
        raise RuntimeError("Offline evaluation seeds are not exactly matched")
    output = []
    for metric in (
        "hungarian_assignment_distance",
        "coverage_radius_auc",
        "collision_step_rate",
        "minimum_agent_separation",
        "return",
        "final_coverage",
        "max_coverage",
    ):
        delta = np.asarray([
            float(student[seed][metric]) - float(teacher[seed][metric])
            for seed in seeds
        ])
        half = 1.96 * delta.std(ddof=1) / math.sqrt(len(delta))
        output.append({
            "metric": metric,
            "n_episodes": len(delta),
            "mean_delta": float(delta.mean()),
            "ci95_low": float(delta.mean() - half),
            "ci95_high": float(delta.mean() + half),
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-agents", type=int, required=True, choices=(3, 4, 5, 6))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--dagger-episodes", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--student-seed", type=int, required=True)
    parser.add_argument("--train-seed-base", type=int, required=True)
    parser.add_argument("--dagger-seed-base", type=int, required=True)
    parser.add_argument("--eval-seed-base", type=int, required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    observations, labels, agent_indices, episode_ids = collect(
        args.n_agents,
        args.train_episodes,
        args.horizon,
        args.train_seed_base,
    )
    state, first_metrics, first_history = train_actor(
        observations,
        labels,
        agent_indices,
        episode_ids,
        args.n_agents,
        args.student_seed,
        args.epochs,
    )
    policies = build_policies(state, args.n_agents)
    if args.dagger_episodes > 0:
        dagger_obs, dagger_labels, dagger_agents, dagger_episodes = collect(
            args.n_agents,
            args.dagger_episodes,
            args.horizon,
            args.dagger_seed_base,
            rollout_policies=policies,
        )
        dagger_episodes = dagger_episodes + int(episode_ids.max()) + 1
        observations = np.concatenate([observations, dagger_obs])
        labels = np.concatenate([labels, dagger_labels])
        agent_indices = np.concatenate([agent_indices, dagger_agents])
        episode_ids = np.concatenate([episode_ids, dagger_episodes])
        state, final_metrics, second_history = train_actor(
            observations,
            labels,
            agent_indices,
            episode_ids,
            args.n_agents,
            args.student_seed + 1000,
            args.epochs,
            initial_state=state,
        )
        history = [dict(row, round=1) for row in first_history]
        history += [dict(row, round=2) for row in second_history]
    else:
        final_metrics = first_metrics
        history = [dict(row, round=1) for row in first_history]

    np.savez_compressed(
        output / "training_dataset.npz",
        observations=observations,
        labels=labels,
        agent_indices=agent_indices,
        episode_ids=episode_ids,
    )
    payload = {
        "actor_state_dict": state,
        "actor_model": "equivariant_matching_safety_nxn_sinkhorn8",
        "target": "independent_safe_matching_teacher",
        "n_agents": args.n_agents,
        "observation_dim": observation_dim(args.n_agents),
        "student_seed": args.student_seed,
        "first_round_metrics": first_metrics,
        "distillation_metrics": final_metrics,
    }
    torch.save(payload, output / "cardinality_actor.pt")
    policies = build_policies(state, args.n_agents)
    rows = evaluate_controller(
        "teacher", policies, args.n_agents, args.eval_episodes,
        args.horizon, args.eval_seed_base
    )
    rows += evaluate_controller(
        "student", policies, args.n_agents, args.eval_episodes,
        args.horizon, args.eval_seed_base
    )
    intervals = paired_offline_intervals(rows)
    interval_lookup = {row["metric"]: row for row in intervals}
    eligible = (
        final_metrics["validation_accuracy"] >= 0.85
        and interval_lookup["hungarian_assignment_distance"]["mean_delta"] <= 0.05
        and interval_lookup["coverage_radius_auc"]["mean_delta"] >= -0.05
        and interval_lookup["collision_step_rate"]["ci95_low"] <= 0.0
    )
    write_csv(output / "training_history.csv", history)
    write_csv(output / "per_episode.csv", rows)
    write_csv(output / "paired_intervals.csv", intervals)
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
    metadata = {
        "n_agents": args.n_agents,
        "n_landmarks": args.n_agents,
        "environment": env_id_for_cardinality(args.n_agents),
        "local_observation_dim": observation_dim(args.n_agents),
        "teacher": "exact assignment + reward-feasible independent safety filter",
        "actor": "shared equivariant pair/matching/safety scorer, Sinkhorn-8",
        "teacher_global_labels_only": True,
        "actor_global_state": False,
        "explicit_communication": False,
        "first_round": first_metrics,
        "final": final_metrics,
        "offline_gate": {
            "eligible": bool(eligible),
            "validation_accuracy_min": 0.85,
            "h_delta_max": 0.05,
            "auc_delta_min": -0.05,
            "collision_ci_low_max": 0.0,
        },
        "seeds": {
            "student": args.student_seed,
            "train_base": args.train_seed_base,
            "dagger_base": args.dagger_seed_base,
            "eval_base": args.eval_seed_base,
        },
        "wall_time_sec": time.perf_counter() - started,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
