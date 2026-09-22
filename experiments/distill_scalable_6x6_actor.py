"""Distill the safe transparent 6x6 teacher into a shared local scorer."""

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

from experiments.evaluate_scalable_6x6_teacher import teacher_actions, world_metrics
from utils.make_env import make_env
from utils.networks import ScalableCounterfactual6x6Policy
from utils.scalable_active_features import scalable_active_action_features_tensor


CONTROLLER = "independent_safe_hybrid"


def policy_actions(policies, obs):
    actions = []
    with torch.no_grad():
        for index, policy in enumerate(policies):
            logits = policy(torch.from_numpy(obs[index:index + 1]))
            action = int(logits.argmax(1))
            actions.append(np.eye(5, dtype=np.float32)[action])
    return actions


def collect(episodes, horizon, seed_base, rollout_policies=None):
    env = make_env("simple_spread_6x6", discrete_action=True)
    observations, labels, agent_indices, episode_ids = [], [], [], []
    try:
        for episode in range(episodes):
            seed = seed_base + episode
            torch.manual_seed(seed); np.random.seed(seed); env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            for _ in range(horizon):
                teacher, _ = teacher_actions(obs, CONTROLLER)
                observations.extend(obs)
                labels.extend(int(np.argmax(action)) for action in teacher)
                agent_indices.extend(range(6)); episode_ids.extend([episode] * 6)
                actions = teacher if rollout_policies is None else policy_actions(rollout_policies, obs)
                obs, _, dones, _ = env.step(actions)
                obs = np.asarray(obs, dtype=np.float32)
                if all(dones):
                    break
    finally:
        env.close()
    return tuple(np.asarray(value) for value in (
        observations, labels, agent_indices, episode_ids
    ))


def candidate_inputs(observations, agent_indices):
    result = np.empty((len(observations), 5, 45), dtype=np.float32)
    action_eye = torch.eye(5)
    for agent_index in range(6):
        selected = np.flatnonzero(agent_indices == agent_index)
        for start in range(0, len(selected), 1024):
            indices = selected[start:start + 1024]
            raw = torch.from_numpy(observations[indices].astype(np.float32))
            with torch.no_grad():
                features = scalable_active_action_features_tensor(raw, agent_index, 6, 6)
                expanded_raw = raw[:, None, :].expand(-1, 5, -1)
                expanded_action = action_eye[None].expand(len(indices), -1, -1)
                value = torch.cat([expanded_raw, expanded_action, features], dim=2)
            result[indices] = value.numpy()
    return result


def distill(observations, labels, agent_indices, episode_ids, seed, epochs,
            initial_actor_state=None, fixed_normalization=None):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(6)
    inputs = candidate_inputs(observations, agent_indices)
    validation = (episode_ids % 5) == 0
    fit = ~validation
    flattened_fit = inputs[fit].reshape(-1, inputs.shape[2])
    if fixed_normalization is None:
        x_mean = flattened_fit.mean(0)
        x_std = flattened_fit.std(0).clip(1e-6)
    else:
        x_mean, x_std = fixed_normalization
    normalized = torch.from_numpy((inputs - x_mean[None, None]) / x_std[None, None])
    targets = torch.from_numpy(labels.astype(np.int64))
    fit_indices = torch.from_numpy(np.flatnonzero(fit))
    validation_indices = torch.from_numpy(np.flatnonzero(validation))
    policy = ScalableCounterfactual6x6Policy(36, 5, agent_index=0)
    if initial_actor_state is not None:
        policy.load_state_dict(initial_actor_state)
    optimizer = torch.optim.Adam(policy.scorer.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(seed)
    best_state, best_loss, patience = None, math.inf, 0
    history = []
    for epoch in range(epochs):
        order = fit_indices[torch.randperm(len(fit_indices), generator=generator)]
        policy.scorer.train()
        for start in range(0, len(order), 1024):
            index = order[start:start + 1024]
            logits = policy.scorer(normalized[index]).squeeze(2)
            loss = F.cross_entropy(logits, targets[index])
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.scorer.parameters(), 1.0)
            optimizer.step()
        policy.scorer.eval()
        with torch.no_grad():
            logits = policy.scorer(normalized[validation_indices]).squeeze(2)
            loss_value = float(F.cross_entropy(logits, targets[validation_indices]))
            accuracy = float((logits.argmax(1) == targets[validation_indices]).float().mean())
        history.append({"epoch": epoch + 1, "validation_loss": loss_value,
                        "validation_accuracy": accuracy})
        if loss_value < best_loss - 1e-5:
            best_loss = loss_value
            best_state = {key: value.detach().clone() for key, value in policy.scorer.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 8:
                break
    policy.scorer.load_state_dict(best_state)
    policy.x_mean.copy_(torch.from_numpy(x_mean)); policy.x_std.copy_(torch.from_numpy(x_std))
    with torch.no_grad():
        predicted = policy.scorer(normalized).squeeze(2).argmax(1).numpy()
    metrics = {
        "samples": len(labels), "episodes": int(np.unique(episode_ids).size),
        "epochs_run": len(history), "validation_loss": best_loss,
        "fit_accuracy": float((predicted[fit] == labels[fit]).mean()),
        "validation_accuracy": float((predicted[validation] == labels[validation]).mean()),
        "per_agent_accuracy": [
            float((predicted[agent_indices == index] == labels[agent_indices == index]).mean())
            for index in range(6)
        ],
        "action_distribution": np.bincount(labels, minlength=5).tolist(),
    }
    return policy.state_dict(), metrics, history, (x_mean, x_std)


def build_policies(actor_state):
    policies = []
    for index in range(6):
        policy = ScalableCounterfactual6x6Policy(36, 5, agent_index=index)
        policy.load_state_dict(actor_state); policy.eval(); policies.append(policy)
    return policies


def evaluate(controller, policies, episodes, horizon, seed_base):
    env = make_env("simple_spread_6x6", discrete_action=True)
    rows = []
    try:
        for episode in range(episodes):
            seed = seed_base + episode
            torch.manual_seed(seed); np.random.seed(seed); env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            total_return, collision_steps = 0.0, 0
            minimum_separation, max_coverage = float("inf"), 0
            for _ in range(horizon):
                actions = (teacher_actions(obs, CONTROLLER)[0] if controller == "teacher"
                           else policy_actions(policies, obs))
                obs, rewards, dones, _ = env.step(actions)
                obs = np.asarray(obs, dtype=np.float32)
                total_return += float(np.mean(rewards))
                assignment, auc, separation, collision, coverage = world_metrics(env)
                collision_steps += collision; minimum_separation = min(minimum_separation, separation)
                max_coverage = max(max_coverage, coverage)
                if all(dones): break
            rows.append({"controller": controller, "episode": episode, "test_seed": seed,
                         "return": total_return, "hungarian_assignment_distance": assignment,
                         "coverage_radius_auc": auc, "collision_step_rate": collision_steps / horizon,
                         "minimum_agent_separation": minimum_separation,
                         "final_coverage": coverage, "max_coverage": max_coverage})
    finally:
        env.close()
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="experiments/scalable_6x6_distillation_20260713")
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-seed-base", type=int, default=5_200_000)
    parser.add_argument("--eval-seed-base", type=int, default=5_300_000)
    parser.add_argument("--dagger-episodes", type=int, default=0)
    parser.add_argument("--dagger-seed-base", type=int, default=5_400_000)
    args = parser.parse_args()
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=False)
    observations, labels, agent_indices, episode_ids = collect(
        args.train_episodes, args.horizon, args.train_seed_base
    )
    actor_state, metrics, history, normalization = distill(
        observations, labels, agent_indices, episode_ids, args.seed, args.epochs
    )
    first_round_metrics = metrics
    if args.dagger_episodes > 0:
        rollout_policies = build_policies(actor_state)
        d_obs, d_labels, d_agents, d_episodes = collect(
            args.dagger_episodes, args.horizon, args.dagger_seed_base,
            rollout_policies=rollout_policies
        )
        d_episodes = d_episodes + int(episode_ids.max()) + 1
        observations = np.concatenate([observations, d_obs])
        labels = np.concatenate([labels, d_labels])
        agent_indices = np.concatenate([agent_indices, d_agents])
        episode_ids = np.concatenate([episode_ids, d_episodes])
        actor_state, metrics, second_history, _ = distill(
            observations, labels, agent_indices, episode_ids,
            args.seed + 1000, args.epochs,
            initial_actor_state=actor_state, fixed_normalization=normalization
        )
        history = [dict(row, round=1) for row in history] + [
            dict(row, round=2) for row in second_history
        ]
    else:
        history = [dict(row, round=1) for row in history]
    payload = {"actor_state_dict": actor_state, "target": CONTROLLER,
               "distillation_metrics": metrics,
               "first_round_metrics": first_round_metrics}
    torch.save(payload, output / "scalable_6x6_distilled_actor.pt")
    policies = build_policies(actor_state)
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
    (output / "metadata.json").write_text(json.dumps({
        "teacher": CONTROLLER, "distillation_round1": first_round_metrics,
        "distillation_final": metrics, "dagger_episodes": args.dagger_episodes,
        "train_seed_base": args.train_seed_base, "eval_seed_base": args.eval_seed_base,
        "shared_scorer": True, "local_observation_dim": 36,
        "graph_nodes": 12, "explicit_communication": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
