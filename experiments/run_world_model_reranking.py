"""Train a joint action-consequence model and test conservative action reranking."""

from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from benchmarl.hydra_config import reload_experiment_from_file
from scipy.stats import ttest_rel
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp

from evaluate_benchmarl_checkpoint import geometry, summarize


N_AGENTS = 3
N_ACTIONS = 5
STATE_DIM = 54


class ConsequenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(STATE_DIM + N_AGENTS * N_ACTIONS, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
        )
        self.delta_head = nn.Linear(256, STATE_DIM)
        self.reward_head = nn.Linear(256, 1)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return self.delta_head(hidden), self.reward_head(hidden).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--transitions", type=int, default=25_000)
    parser.add_argument("--random-action-prob", type=float, default=0.5)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--evaluation-episodes", type=int, default=500)
    parser.add_argument("--uncertainty-coef", type=float, default=1.0)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    parser.add_argument("--collection-seed-base", type=int, default=67_000_000)
    parser.add_argument("--evaluation-seed-base", type=int, default=65_000_000)
    parser.add_argument("--training-seed", type=int, default=41_000)
    return parser.parse_args()


def policy_action(policy: nn.Module, tensordict) -> torch.Tensor:
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        policy(tensordict)
    return tensordict.get(("agent", "action")).detach().clone().long()


def collect_dataset(
    env,
    policy: nn.Module,
    transition_count: int,
    random_action_prob: float,
    seed_base: int,
) -> dict[str, np.ndarray]:
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    rewards: list[float] = []
    episode_ids: list[int] = []

    episode = 0
    while len(states) < transition_count:
        seed = seed_base + episode
        rng = np.random.default_rng(seed)
        env.set_seed(seed)
        tensordict = env.reset()
        for _ in range(25):
            action = policy_action(policy, tensordict)
            if rng.random() < random_action_prob:
                action = torch.as_tensor(rng.integers(0, N_ACTIONS, size=N_AGENTS), dtype=torch.long)
            tensordict.set(("agent", "action"), action)
            transition = env.step(tensordict)

            states.append(
                tensordict.get(("agent", "observation"))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )
            actions.append(action.detach().cpu().numpy().astype(np.int64))
            next_states.append(
                transition.get(("next", "agent", "observation"))
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
                .astype(np.float32)
            )
            agent_rewards = transition.get(("next", "agent", "reward"))
            rewards.append(float(2.0 * agent_rewards.mean().item()))
            episode_ids.append(episode)

            tensordict = step_mdp(transition)
            if len(states) >= transition_count or bool(transition.get(("next", "done")).any()):
                break
        episode += 1

    return {
        "state": np.asarray(states, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.int64),
        "next_state": np.asarray(next_states, dtype=np.float32),
        "reward": np.asarray(rewards, dtype=np.float32),
        "episode": np.asarray(episode_ids, dtype=np.int64),
    }


def make_features(
    states: torch.Tensor,
    actions: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
) -> torch.Tensor:
    normalized_state = (states - state_mean) / state_std
    one_hot_action = F.one_hot(actions.long(), num_classes=N_ACTIONS).float().flatten(start_dim=-2)
    return torch.cat([normalized_state, one_hot_action], dim=-1)


def train_ensemble(
    dataset: dict[str, np.ndarray],
    ensemble_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    training_seed: int,
    device: torch.device,
) -> tuple[list[ConsequenceModel], dict[str, torch.Tensor], dict[str, float]]:
    val_mask_np = dataset["episode"] % 5 == 0
    train_mask_np = ~val_mask_np
    state_np = dataset["state"]
    delta_np = dataset["next_state"] - state_np
    reward_np = dataset["reward"]

    state_mean = torch.as_tensor(state_np[train_mask_np].mean(axis=0), device=device)
    state_std = torch.as_tensor(state_np[train_mask_np].std(axis=0) + 1e-6, device=device)
    delta_mean = torch.as_tensor(delta_np[train_mask_np].mean(axis=0), device=device)
    delta_std = torch.as_tensor(delta_np[train_mask_np].std(axis=0) + 1e-6, device=device)
    reward_mean = torch.tensor(float(reward_np[train_mask_np].mean()), device=device)
    reward_std = torch.tensor(float(reward_np[train_mask_np].std() + 1e-6), device=device)
    normalizer = {
        "state_mean": state_mean,
        "state_std": state_std,
        "delta_mean": delta_mean,
        "delta_std": delta_std,
        "reward_mean": reward_mean,
        "reward_std": reward_std,
    }

    states = torch.as_tensor(state_np, device=device)
    actions = torch.as_tensor(dataset["action"], device=device)
    deltas = torch.as_tensor(delta_np, device=device)
    rewards = torch.as_tensor(reward_np, device=device)
    train_indices = torch.as_tensor(np.flatnonzero(train_mask_np), device=device)
    val_indices = torch.as_tensor(np.flatnonzero(val_mask_np), device=device)
    features = make_features(states, actions, state_mean, state_std)
    delta_targets = (deltas - delta_mean) / delta_std
    reward_targets = (rewards - reward_mean) / reward_std

    models: list[ConsequenceModel] = []
    histories: list[dict[str, float | int]] = []
    for model_index in range(ensemble_size):
        seed = training_seed + model_index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = ConsequenceModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0

        generator = torch.Generator(device=device).manual_seed(seed + 10_000)
        bootstrap = train_indices[
            torch.randint(train_indices.numel(), (train_indices.numel(),), generator=generator, device=device)
        ]
        for epoch in range(epochs):
            model.train()
            order = bootstrap[torch.randperm(bootstrap.numel(), generator=generator, device=device)]
            for start in range(0, order.numel(), batch_size):
                index = order[start : start + batch_size]
                predicted_delta, predicted_reward = model(features[index])
                loss = F.mse_loss(predicted_delta, delta_targets[index]) + F.mse_loss(
                    predicted_reward, reward_targets[index]
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                predicted_delta, predicted_reward = model(features[val_indices])
                val_loss = float(
                    F.mse_loss(predicted_delta, delta_targets[val_indices]).item()
                    + F.mse_loss(predicted_reward, reward_targets[val_indices]).item()
                )
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= 10:
                break

        if best_state is None:
            raise RuntimeError("World model did not produce a valid checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        models.append(model)
        histories.append({"model": model_index, "best_val_loss": best_loss, "epochs": epoch + 1})

    with torch.no_grad():
        predictions = [model(features[val_indices]) for model in models]
        predicted_delta = torch.stack([item[0] for item in predictions]).mean(dim=0) * delta_std + delta_mean
        predicted_reward = (
            torch.stack([item[1] for item in predictions]).mean(dim=0) * reward_std + reward_mean
        )
        true_delta = deltas[val_indices]
        true_reward = rewards[val_indices]
        position_indices = torch.as_tensor(
            [agent * 18 + coordinate for agent in range(3) for coordinate in (2, 3)], device=device
        )
        metrics = {
            "train_transitions": int(train_indices.numel()),
            "validation_transitions": int(val_indices.numel()),
            "next_state_rmse": float(torch.sqrt(F.mse_loss(predicted_delta, true_delta)).item()),
            "next_position_rmse": float(
                torch.sqrt(
                    F.mse_loss(predicted_delta[:, position_indices], true_delta[:, position_indices])
                ).item()
            ),
            "reward_mae": float(F.l1_loss(predicted_reward, true_reward).item()),
            "reward_rmse": float(torch.sqrt(F.mse_loss(predicted_reward, true_reward)).item()),
            "ensemble": histories,
        }
    return models, normalizer, metrics


def candidate_actions(base_action: torch.Tensor) -> torch.Tensor:
    candidates = [base_action.detach().cpu().clone()]
    for agent in range(N_AGENTS):
        for action in range(N_ACTIONS):
            if action != int(base_action[agent]):
                candidate = base_action.detach().cpu().clone()
                candidate[agent] = action
                candidates.append(candidate)
    return torch.stack(candidates)


def rerank_action(
    state: torch.Tensor,
    base_action: torch.Tensor,
    models: list[ConsequenceModel],
    normalizer: dict[str, torch.Tensor],
    threshold: float,
    uncertainty_coef: float,
    device: torch.device,
) -> tuple[torch.Tensor, bool, float, float]:
    candidates = candidate_actions(base_action).to(device)
    states = state.to(device).unsqueeze(0).expand(candidates.shape[0], -1)
    features = make_features(states, candidates, normalizer["state_mean"], normalizer["state_std"])
    with torch.no_grad():
        normalized_rewards = torch.stack([model(features)[1] for model in models])
        rewards = normalized_rewards * normalizer["reward_std"] + normalizer["reward_mean"]
    reward_mean = rewards.mean(dim=0)
    reward_std = rewards.std(dim=0, unbiased=False)
    conservative_score = reward_mean - uncertainty_coef * reward_std
    best_index = int(conservative_score.argmax())
    predicted_gain = float(conservative_score[best_index] - reward_mean[0])
    intervene = best_index != 0 and predicted_gain > threshold
    chosen_index = best_index if intervene else 0
    return (
        candidates[chosen_index].detach().cpu(),
        intervene,
        predicted_gain,
        float(reward_std[best_index]),
    )


def evaluate_modes(
    env,
    policy: nn.Module,
    models: list[ConsequenceModel],
    normalizer: dict[str, torch.Tensor],
    thresholds: list[float],
    uncertainty_coef: float,
    episodes: int,
    seed_base: int,
    device: torch.device,
) -> tuple[list[dict[str, float | int | str]], dict]:
    modes: list[tuple[str, float | None]] = [("baseline", None)] + [
        (f"rerank_t{threshold:.2f}", threshold) for threshold in thresholds
    ]
    rows: list[dict[str, float | int | str]] = []
    for mode, threshold in modes:
        for episode in range(episodes):
            seed = seed_base + episode
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env.set_seed(seed)
            tensordict = env.reset()
            episode_return = 0.0
            step_metrics: list[dict[str, float]] = []
            interventions = 0
            predicted_gains: list[float] = []
            selected_uncertainties: list[float] = []

            for _ in range(25):
                base_action = policy_action(policy, tensordict)
                action = base_action
                if threshold is not None:
                    action, intervened, gain, uncertainty = rerank_action(
                        tensordict.get(("agent", "observation")).flatten(),
                        base_action,
                        models,
                        normalizer,
                        threshold,
                        uncertainty_coef,
                        device,
                    )
                    interventions += int(intervened)
                    if intervened:
                        predicted_gains.append(gain)
                        selected_uncertainties.append(uncertainty)
                tensordict.set(("agent", "action"), action)
                transition = env.step(tensordict)
                rewards = transition.get(("next", "agent", "reward"))
                episode_return += float(2.0 * rewards.mean().item())
                observation = (
                    transition.get(("next", "agent", "observation")).detach().cpu().numpy()
                )
                step_metrics.append(geometry(observation))
                tensordict = step_mdp(transition)
                if bool(transition.get(("next", "done")).any()):
                    break

            final = step_metrics[-1]
            rows.append(
                {
                    "mode": mode,
                    "episode": episode,
                    "seed": seed,
                    "steps": len(step_metrics),
                    "episode_return_legacy_scale": episode_return,
                    "final_hungarian": final["hungarian"],
                    "final_radius_auc": final["radius_auc"],
                    "collision_step_rate": float(
                        np.mean([item["collision"] for item in step_metrics])
                    ),
                    "minimum_pair_separation": float(
                        min(item["min_pair_separation"] for item in step_metrics)
                    ),
                    "final_coverage_at_010": final["coverage_at_010"],
                    "max_coverage_at_010": float(
                        max(item["coverage_at_010"] for item in step_metrics)
                    ),
                    "intervention_rate": interventions / len(step_metrics),
                    "mean_predicted_gain_when_intervening": float(np.mean(predicted_gains))
                    if predicted_gains
                    else 0.0,
                    "mean_selected_uncertainty": float(np.mean(selected_uncertainties))
                    if selected_uncertainties
                    else 0.0,
                }
            )

    metric_names = [
        "episode_return_legacy_scale",
        "final_hungarian",
        "final_radius_auc",
        "collision_step_rate",
        "minimum_pair_separation",
        "final_coverage_at_010",
        "max_coverage_at_010",
        "intervention_rate",
        "mean_predicted_gain_when_intervening",
        "mean_selected_uncertainty",
    ]
    grouped: dict[str, list[dict[str, float | int | str]]] = {
        mode: [row for row in rows if row["mode"] == mode] for mode, _ in modes
    }
    mode_summary = {
        mode: {
            metric: summarize(
                np.asarray([float(row[metric]) for row in mode_rows], dtype=np.float64)
            )
            for metric in metric_names
        }
        for mode, mode_rows in grouped.items()
    }

    baseline = grouped["baseline"]
    paired: dict[str, dict] = {}
    direction = {
        "episode_return_legacy_scale": 1.0,
        "final_hungarian": -1.0,
        "final_radius_auc": 1.0,
        "collision_step_rate": -1.0,
        "minimum_pair_separation": 1.0,
        "final_coverage_at_010": 1.0,
        "max_coverage_at_010": 1.0,
    }
    for mode, threshold in modes[1:]:
        mode_rows = grouped[mode]
        paired[mode] = {}
        for metric, sign in direction.items():
            base_values = np.asarray([float(row[metric]) for row in baseline])
            mode_values = np.asarray([float(row[metric]) for row in mode_rows])
            improvement = sign * (mode_values - base_values)
            test = ttest_rel(mode_values, base_values)
            paired[mode][metric] = {
                "improvement_positive_is_better": summarize(improvement),
                "paired_ttest_two_sided_p": float(test.pvalue) if np.isfinite(test.pvalue) else None,
            }
        paired[mode]["threshold"] = threshold
    return rows, {"modes": mode_summary, "paired_vs_baseline": paired}


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.random_action_prob <= 1.0:
        raise ValueError("--random-action-prob must be in [0, 1]")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    experiment = reload_experiment_from_file(str(checkpoint))
    try:
        policy = experiment.policy.to("cpu").eval()
        env = experiment.test_env
        dataset = collect_dataset(
            env,
            policy,
            args.transitions,
            args.random_action_prob,
            args.collection_seed_base,
        )
        state_variation = float(dataset["state"].std(axis=0).mean())
        transition_magnitude = float(
            np.abs(dataset["next_state"] - dataset["state"]).mean()
        )
        if state_variation < 1e-6 or transition_magnitude < 1e-8:
            raise RuntimeError(
                "Degenerate joint-observation dataset: "
                f"variation={state_variation}, transition_magnitude={transition_magnitude}"
            )
        np.savez_compressed(output_dir / "joint_transition_dataset.npz", **dataset)

        models, normalizer, validation_metrics = train_ensemble(
            dataset,
            args.ensemble_size,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.training_seed,
            device,
        )
        torch.save(
            {
                "models": [{key: value.detach().cpu() for key, value in model.state_dict().items()} for model in models],
                "normalizer": {key: value.detach().cpu() for key, value in normalizer.items()},
                "validation_metrics": validation_metrics,
                "checkpoint_anchor": str(checkpoint),
            },
            output_dir / "world_model_ensemble.pt",
        )

        rows, evaluation = evaluate_modes(
            env,
            policy,
            models,
            normalizer,
            args.thresholds,
            args.uncertainty_coef,
            args.evaluation_episodes,
            args.evaluation_seed_base,
            device,
        )
    finally:
        experiment.close()

    with (output_dir / "per_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_anchor": str(checkpoint),
        "protocol": {
            "transitions": args.transitions,
            "random_action_prob": args.random_action_prob,
            "state_source": "flattened 3x18 per-agent observations",
            "state_variation": state_variation,
            "mean_absolute_transition": transition_magnitude,
            "ensemble_size": args.ensemble_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "evaluation_episodes": args.evaluation_episodes,
            "uncertainty_coef": args.uncertainty_coef,
            "thresholds": args.thresholds,
            "collection_seed_base": args.collection_seed_base,
            "evaluation_seed_base": args.evaluation_seed_base,
            "training_seed": args.training_seed,
        },
        "validation": validation_metrics,
        "evaluation": evaluation,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
