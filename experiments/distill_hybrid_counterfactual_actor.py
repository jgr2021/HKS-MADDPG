"""Distill the low-collision counterfactual hybrid into one local actor.

The teacher is the previously fixed lexicographic rule: among actions whose
predicted original-reward delta is at least the predicted no-op value, choose
the smallest predicted Hungarian delta.  Distillation changes only actor
weights.  Teacher trajectories supply local 18D observations and action labels;
the deployed student remains a single decentralized action-conditioned scorer.
"""

from __future__ import annotations

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

from experiments.evaluate_hybrid_counterfactual_controller import actions_for_obs
from experiments.evaluate_supervised_probe_policies import geometry
from experiments.probe_action_value_representations import train_probe
from experiments.probe_and_evaluate_reward_value import train as train_reward
from utils.make_env import make_env
from utils.networks import CounterfactualActiveProbePolicy


REPRESENTATION = "raw_action_active"


def teacher_bundles(data, model_seed, heldout_source_seed):
    hungarian_metrics, hungarian = train_probe(
        data, REPRESENTATION, model_seed, heldout_source_seed, 50, 6, True
    )
    reward_metrics, reward = train_reward(
        data,
        REPRESENTATION,
        model_seed,
        heldout_seed=heldout_source_seed,
    )
    return hungarian, reward, hungarian_metrics, reward_metrics


def collect_teacher_trajectories(hungarian, reward, episodes, horizon, seed_base,
                                 rollout_policy=None):
    env = make_env("simple_spread", discrete_action=True)
    observations, labels, episode_ids = [], [], []
    try:
        for episode in range(episodes):
            seed = seed_base + episode
            torch.manual_seed(seed); np.random.seed(seed); env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            for _ in range(horizon):
                teacher_actions = actions_for_obs(
                    obs, REPRESENTATION, hungarian, reward
                )
                observations.extend(obs)
                labels.extend(int(np.argmax(action)) for action in teacher_actions)
                episode_ids.extend([episode] * len(teacher_actions))
                actions = (
                    teacher_actions if rollout_policy is None
                    else policy_actions(rollout_policy, obs)
                )
                obs, _, dones, _ = env.step(actions)
                obs = np.asarray(obs, dtype=np.float32)
                if all(dones):
                    break
    finally:
        env.close()
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(episode_ids, dtype=np.int64),
    )


def initialize_from_hungarian(policy, hungarian):
    policy.load_probe_payload({
        "x_mean": hungarian["x_mean"], "x_std": hungarian["x_std"],
        "y_mean": hungarian["y_mean"], "y_std": hungarian["y_std"],
        "probe_state_dict": hungarian["model"].state_dict(),
    })


def distill(observations, labels, episode_ids, hungarian, model_seed, epochs,
            initial_state=None):
    torch.manual_seed(model_seed); np.random.seed(model_seed); torch.set_num_threads(6)
    policy = CounterfactualActiveProbePolicy(18, 5)
    if initial_state is None:
        initialize_from_hungarian(policy, hungarian)
    else:
        policy.load_state_dict(initial_state)
    validation = (episode_ids % 5) == 0
    fit_indices = torch.from_numpy(np.flatnonzero(~validation))
    validation_obs = torch.from_numpy(observations[validation])
    validation_labels = torch.from_numpy(labels[validation])
    obs_tensor = torch.from_numpy(observations)
    label_tensor = torch.from_numpy(labels)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(model_seed)
    best_state, best_loss, patience = None, math.inf, 0
    history = []
    for epoch in range(epochs):
        order = fit_indices[torch.randperm(len(fit_indices), generator=generator)]
        policy.train()
        for start in range(0, len(order), 512):
            index = order[start:start + 512]
            loss = F.cross_entropy(policy(obs_tensor[index]), label_tensor[index])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        policy.eval()
        with torch.no_grad():
            logits = policy(validation_obs)
            validation_loss = float(F.cross_entropy(logits, validation_labels))
            validation_accuracy = float((logits.argmax(1) == validation_labels).float().mean())
        history.append({"epoch": epoch + 1, "validation_loss": validation_loss,
                        "validation_accuracy": validation_accuracy})
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                break
    policy.load_state_dict(best_state); policy.eval()
    with torch.no_grad():
        predictions = policy(obs_tensor).argmax(1).numpy()
    metrics = {
        "samples": int(len(labels)), "episodes": int(np.unique(episode_ids).size),
        "epochs_run": len(history), "validation_loss": best_loss,
        "validation_accuracy": float((predictions[validation] == labels[validation]).mean()),
        "fit_accuracy": float((predictions[~validation] == labels[~validation]).mean()),
        "action_distribution": np.bincount(labels, minlength=5).tolist(),
        "prediction_distribution": np.bincount(predictions, minlength=5).tolist(),
    }
    return policy, metrics, history


def policy_actions(policy, obs):
    with torch.no_grad():
        indices = policy(torch.from_numpy(obs)).argmax(1).cpu().numpy()
    return [np.eye(5, dtype=np.float32)[index] for index in indices]


def evaluate_controller(controller, policy, hungarian, reward, episodes, horizon, seed_base):
    env = make_env("simple_spread", discrete_action=True)
    rows = []
    try:
        for episode in range(episodes):
            seed = seed_base + episode
            torch.manual_seed(seed); np.random.seed(seed); env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            total_return, collision_steps = 0.0, 0
            minimum_separation, max_coverage = float("inf"), 0
            for _ in range(horizon):
                if controller == "hybrid_teacher":
                    actions = actions_for_obs(obs, REPRESENTATION, hungarian, reward)
                else:
                    actions = policy_actions(policy, obs)
                obs, rewards, dones, _ = env.step(actions)
                obs = np.asarray(obs, dtype=np.float32)
                total_return += float(np.mean(rewards))
                assignment, auc, separation, collision, coverage = geometry(env)
                collision_steps += collision
                minimum_separation = min(minimum_separation, separation)
                max_coverage = max(max_coverage, coverage)
                if all(dones):
                    break
            rows.append({
                "controller": controller, "episode": episode, "test_seed": seed,
                "return": total_return,
                "hungarian_assignment_distance": assignment,
                "coverage_radius_auc": auc,
                "collision_step_rate": collision_steps / float(horizon),
                "minimum_agent_separation": minimum_separation,
                "final_coverage": coverage, "max_coverage": max_coverage,
            })
    finally:
        env.close()
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="experiments/reward_probe_dataset_20260712/counterfactual_dataset.npz")
    parser.add_argument("--output-dir", default="experiments/hybrid_actor_distillation_20260713")
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--teacher-seed", type=int, default=1)
    parser.add_argument("--student-seed", type=int, default=1)
    parser.add_argument("--heldout-source-seed", type=int, default=23)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--train-seed-base", type=int, default=3_100_000)
    parser.add_argument("--eval-seed-base", type=int, default=3_200_000)
    parser.add_argument("--dagger-episodes", type=int, default=0)
    parser.add_argument("--dagger-seed-base", type=int, default=3_300_000)
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    with np.load(args.dataset) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    hungarian, reward, hungarian_metrics, reward_metrics = teacher_bundles(
        data, args.teacher_seed, args.heldout_source_seed
    )
    observations, labels, episode_ids = collect_teacher_trajectories(
        hungarian, reward, args.train_episodes, args.horizon, args.train_seed_base
    )
    policy, distillation_metrics, history = distill(
        observations, labels, episode_ids, hungarian, args.student_seed, args.epochs
    )
    first_round_metrics = distillation_metrics
    if args.dagger_episodes > 0:
        dagger_obs, dagger_labels, dagger_episode_ids = collect_teacher_trajectories(
            hungarian, reward, args.dagger_episodes, args.horizon,
            args.dagger_seed_base, rollout_policy=policy
        )
        dagger_episode_ids = dagger_episode_ids + int(episode_ids.max()) + 1
        observations = np.concatenate([observations, dagger_obs], axis=0)
        labels = np.concatenate([labels, dagger_labels], axis=0)
        episode_ids = np.concatenate([episode_ids, dagger_episode_ids], axis=0)
        first_state = {key: value.detach().clone() for key, value in policy.state_dict().items()}
        policy, distillation_metrics, second_history = distill(
            observations, labels, episode_ids, hungarian,
            args.student_seed + 1000, args.epochs, initial_state=first_state
        )
        history = [dict(row, round=1) for row in history] + [
            dict(row, round=2) for row in second_history
        ]
    else:
        history = [dict(row, round=1) for row in history]
    payload = {
        "representation": REPRESENTATION,
        "target": "lexicographic_reward_feasible_hungarian_teacher_action",
        "actor_state_dict": policy.state_dict(),
        "teacher_seed": args.teacher_seed, "student_seed": args.student_seed,
        "heldout_source_seed": args.heldout_source_seed,
        "distillation_metrics": distillation_metrics,
        "first_round_metrics": first_round_metrics,
    }
    torch.save(payload, output / "hybrid_distilled_actor.pt")
    rows = []
    for controller in ("hybrid_teacher", "hybrid_distilled_actor"):
        rows.extend(evaluate_controller(
            controller, policy, hungarian, reward, args.eval_episodes,
            args.horizon, args.eval_seed_base
        ))
    write_csv(output / "per_episode.csv", rows)
    write_csv(output / "training_history.csv", history)
    summary = []
    for controller in ("hybrid_teacher", "hybrid_distilled_actor"):
        selected = [row for row in rows if row["controller"] == controller]
        result = {"controller": controller, "episodes": len(selected)}
        for key in selected[0]:
            if key not in {"controller", "episode", "test_seed"}:
                values = np.asarray([row[key] for row in selected], dtype=np.float64)
                result[key + "_mean"] = float(values.mean())
                result[key + "_std"] = float(values.std(ddof=1))
        summary.append(result)
    write_csv(output / "summary.csv", summary)
    (output / "metadata.json").write_text(json.dumps({
        "teacher_rule": "min Hungarian among predicted reward-delta >= predicted no-op",
        "teacher_probe_metrics": {"hungarian": hungarian_metrics, "reward": reward_metrics},
        "distillation_round1": first_round_metrics,
        "distillation_final": distillation_metrics,
        "dagger_episodes": args.dagger_episodes,
        "heldout_source_seed": args.heldout_source_seed,
        "train_seed_base": args.train_seed_base, "eval_seed_base": args.eval_seed_base,
        "environment_changed": False, "reward_changed": False,
        "replay_or_critic_used": False, "actor_input_dim": 18,
        "action_space_changed": False, "explicit_communication": False,
    }, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)) + "\n",
        encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
