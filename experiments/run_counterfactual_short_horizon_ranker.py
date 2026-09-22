"""Exact same-state counterfactual ranking around a frozen QMIX policy."""

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
from mpe2 import simple_spread_v3
from scipy.stats import ttest_rel
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from evaluate_benchmarl_checkpoint import geometry, summarize


N_AGENTS = 3
N_ACTIONS = 5
N_CANDIDATES = 13
OBS_DIM = 18
STATE_DIM = N_AGENTS * OBS_DIM
HORIZONS = (3, 5)


class CounterfactualRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        input_dim = STATE_DIM + 2 * N_AGENTS * N_ACTIONS
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 4),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--states", type=int, default=4_000)
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--evaluation-episodes", type=int, default=500)
    parser.add_argument("--uncertainty-coef", type=float, default=1.0)
    parser.add_argument("--collision-tolerance", type=float, default=0.05)
    parser.add_argument("--minimum-precision", type=float, default=0.75)
    parser.add_argument("--minimum-coverage", type=float, default=0.05)
    parser.add_argument(
        "--threshold-grid", type=float, nargs="+", default=[0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    )
    parser.add_argument("--collection-seed-base", type=int, default=68_000_000)
    parser.add_argument("--evaluation-seed-base", type=int, default=69_000_000)
    parser.add_argument("--training-seed", type=int, default=42_000)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def make_env():
    return simple_spread_v3.parallel_env(
        N=N_AGENTS,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
    )


def snapshot_environment(environment) -> dict:
    """Copy exact MPE runtime state without unpicklable render caches."""
    render_fields = {"screen", "game_font", "viewer"}
    return {
        key: deepcopy(value)
        for key, value in environment.unwrapped.__dict__.items()
        if key not in render_fields
    }


def restore_environment(environment, snapshot: dict) -> None:
    environment.unwrapped.__dict__.update(deepcopy(snapshot))


def ordered_observations(observations: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([observations[f"agent_{index}"] for index in range(N_AGENTS)]).astype(
        np.float32
    )


def policy_action(policy: nn.Module, observations: dict[str, np.ndarray]) -> np.ndarray:
    tensor = torch.as_tensor(ordered_observations(observations), dtype=torch.float32)
    tensordict = TensorDict({("agent", "observation"): tensor}, batch_size=[])
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        policy(tensordict)
    return tensordict.get(("agent", "action")).detach().cpu().numpy().astype(np.int64)


def batched_policy_actions(
    policy: nn.Module, observation_batch: list[dict[str, np.ndarray]]
) -> np.ndarray:
    tensor = torch.as_tensor(
        np.stack([ordered_observations(observations) for observations in observation_batch]),
        dtype=torch.float32,
    )
    tensordict = TensorDict({("agent", "observation"): tensor}, batch_size=[len(observation_batch)])
    with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
        policy(tensordict)
    return tensordict.get(("agent", "action")).detach().cpu().numpy().astype(np.int64)


def action_dict(environment, action: np.ndarray) -> dict[str, int]:
    return {agent: int(action[index]) for index, agent in enumerate(environment.agents)}


def local_candidates(base_action: np.ndarray) -> np.ndarray:
    candidates = [base_action.copy()]
    for agent in range(N_AGENTS):
        for action in range(N_ACTIONS):
            if action != int(base_action[agent]):
                candidate = base_action.copy()
                candidate[agent] = action
                candidates.append(candidate)
    result = np.asarray(candidates, dtype=np.int64)
    if result.shape != (N_CANDIDATES, N_AGENTS):
        raise RuntimeError(f"Unexpected candidate shape: {result.shape}")
    return result


def branch_outcomes(
    branches: list, snapshot: dict, first_actions: np.ndarray, policy: nn.Module
) -> dict[str, np.ndarray]:
    for branch in branches:
        restore_environment(branch, snapshot)
    returns = np.zeros((N_CANDIDATES, 2), dtype=np.float32)
    collisions = np.zeros((N_CANDIDATES, 2), dtype=np.float32)
    hungarian = np.zeros((N_CANDIDATES, 2), dtype=np.float32)
    radius_auc = np.zeros((N_CANDIDATES, 2), dtype=np.float32)
    discounted_return = np.zeros(N_CANDIDATES, dtype=np.float64)
    collision_count = np.zeros(N_CANDIDATES, dtype=np.float64)
    observation_batch = None

    for step in range(HORIZONS[-1]):
        actions = first_actions if step == 0 else batched_policy_actions(policy, observation_batch)
        next_observation_batch = []
        for candidate_index, branch in enumerate(branches):
            observations, rewards, _, _, _ = branch.step(
                action_dict(branch, actions[candidate_index])
            )
            next_observation_batch.append(observations)
            discounted_return[candidate_index] += (
                (0.9**step) * 2.0 * float(np.mean(list(rewards.values())))
            )
            current_geometry = geometry(ordered_observations(observations))
            collision_count[candidate_index] += current_geometry["collision"]
            if step + 1 in HORIZONS:
                horizon_index = HORIZONS.index(step + 1)
                returns[candidate_index, horizon_index] = discounted_return[candidate_index]
                collisions[candidate_index, horizon_index] = collision_count[candidate_index]
                hungarian[candidate_index, horizon_index] = current_geometry["hungarian"]
                radius_auc[candidate_index, horizon_index] = current_geometry["radius_auc"]
        observation_batch = next_observation_batch
    return {
        "return": returns,
        "collision": collisions,
        "hungarian": hungarian,
        "radius_auc": radius_auc,
    }


def collect_counterfactual_dataset(
    policy: nn.Module, state_count: int, seed_base: int
) -> dict[str, np.ndarray]:
    states: list[np.ndarray] = []
    base_actions: list[np.ndarray] = []
    candidate_actions: list[np.ndarray] = []
    returns: list[np.ndarray] = []
    collisions: list[np.ndarray] = []
    hungarians: list[np.ndarray] = []
    radius_aucs: list[np.ndarray] = []
    episode_ids: list[int] = []

    episode = 0
    while len(states) < state_count:
        environment = make_env()
        observations, _ = environment.reset(seed=seed_base + episode)
        branches = [make_env() for _ in range(N_CANDIDATES)]
        for branch in branches:
            branch.reset(seed=0)
        for _ in range(20):
            base_action = policy_action(policy, observations)
            candidates = local_candidates(base_action)
            snapshot = snapshot_environment(environment)
            outcomes = branch_outcomes(branches, snapshot, candidates, policy)

            states.append(ordered_observations(observations).reshape(-1))
            base_actions.append(base_action)
            candidate_actions.append(candidates)
            returns.append(outcomes["return"])
            collisions.append(outcomes["collision"])
            hungarians.append(outcomes["hungarian"])
            radius_aucs.append(outcomes["radius_auc"])
            episode_ids.append(episode)

            observations, _, terminations, truncations, _ = environment.step(
                action_dict(environment, base_action)
            )
            if len(states) >= state_count or all(terminations.values()) or all(truncations.values()):
                break
        for branch in branches:
            branch.close()
        environment.close()
        episode += 1

    return {
        "state": np.asarray(states, dtype=np.float32),
        "base_action": np.asarray(base_actions, dtype=np.int64),
        "candidate_action": np.asarray(candidate_actions, dtype=np.int64),
        "return": np.asarray(returns, dtype=np.float32),
        "collision": np.asarray(collisions, dtype=np.float32),
        "hungarian": np.asarray(hungarians, dtype=np.float32),
        "radius_auc": np.asarray(radius_aucs, dtype=np.float32),
        "episode": np.asarray(episode_ids, dtype=np.int64),
    }


def build_features(
    states: torch.Tensor,
    base_actions: torch.Tensor,
    candidates: torch.Tensor,
    state_mean: torch.Tensor,
    state_std: torch.Tensor,
) -> torch.Tensor:
    normalized_state = ((states - state_mean) / state_std).unsqueeze(1).expand(-1, N_CANDIDATES, -1)
    base_one_hot = (
        F.one_hot(base_actions.long(), N_ACTIONS)
        .float()
        .flatten(start_dim=-2)
        .unsqueeze(1)
        .expand(-1, N_CANDIDATES, -1)
    )
    candidate_one_hot = F.one_hot(candidates.long(), N_ACTIONS).float().flatten(start_dim=-2)
    return torch.cat([normalized_state, base_one_hot, candidate_one_hot], dim=-1)


def ranking_loss(
    predicted_returns: torch.Tensor, true_returns: torch.Tensor, margin: float = 0.05
) -> torch.Tensor:
    predicted_difference = predicted_returns.unsqueeze(2) - predicted_returns.unsqueeze(1)
    true_difference = true_returns.unsqueeze(2) - true_returns.unsqueeze(1)
    mask = true_difference.abs() > margin
    if not bool(mask.any()):
        return predicted_returns.sum() * 0.0
    signs = true_difference.sign()
    return F.softplus(-signs[mask] * predicted_difference[mask]).mean()


def listwise_loss(predicted_returns: torch.Tensor, true_returns: torch.Tensor) -> torch.Tensor:
    losses = []
    for horizon_index in range(len(HORIZONS)):
        optimal_candidate = true_returns[..., horizon_index].argmax(dim=1)
        losses.append(
            F.cross_entropy(predicted_returns[..., horizon_index], optimal_candidate)
        )
    return torch.stack(losses).mean()


def train_ranker_ensemble(
    dataset: dict[str, np.ndarray],
    ensemble_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    training_seed: int,
    device: torch.device,
) -> tuple[list[CounterfactualRanker], dict[str, torch.Tensor], dict]:
    if np.unique(dataset["episode"]).size >= 5:
        validation_mask_np = dataset["episode"] % 5 == 0
    else:
        validation_mask_np = np.arange(dataset["episode"].size) % 5 == 0
    training_mask_np = ~validation_mask_np
    absolute_targets_np = np.concatenate([dataset["return"], dataset["collision"]], axis=-1)
    targets_np = absolute_targets_np - absolute_targets_np[:, [0], :]

    state_mean = torch.as_tensor(dataset["state"][training_mask_np].mean(axis=0), device=device)
    state_std = torch.as_tensor(dataset["state"][training_mask_np].std(axis=0) + 1e-6, device=device)
    target_mean = torch.as_tensor(
        targets_np[training_mask_np].reshape(-1, 4).mean(axis=0), device=device
    )
    target_std = torch.as_tensor(
        targets_np[training_mask_np].reshape(-1, 4).std(axis=0) + 1e-6, device=device
    )
    normalizer = {
        "state_mean": state_mean,
        "state_std": state_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }

    states = torch.as_tensor(dataset["state"], device=device)
    base_actions = torch.as_tensor(dataset["base_action"], device=device)
    candidates = torch.as_tensor(dataset["candidate_action"], device=device)
    raw_targets = torch.as_tensor(targets_np, device=device)
    features = build_features(states, base_actions, candidates, state_mean, state_std)
    targets = (raw_targets - target_mean) / target_std
    training_indices = torch.as_tensor(np.flatnonzero(training_mask_np), device=device)
    validation_indices = torch.as_tensor(np.flatnonzero(validation_mask_np), device=device)

    models: list[CounterfactualRanker] = []
    histories: list[dict] = []
    for model_index in range(ensemble_size):
        seed = training_seed + model_index
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed + 1_000)
        model = CounterfactualRanker().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bootstrap = training_indices[
            torch.randint(
                training_indices.numel(),
                (training_indices.numel(),),
                generator=generator,
                device=device,
            )
        ]
        best_loss = float("inf")
        best_state = None
        stale = 0
        for epoch in range(epochs):
            model.train()
            order = bootstrap[torch.randperm(bootstrap.numel(), generator=generator, device=device)]
            for start in range(0, order.numel(), batch_size):
                index = order[start : start + batch_size]
                prediction = model(features[index])
                regression = F.mse_loss(prediction, targets[index])
                pairwise = ranking_loss(prediction[..., :2], raw_targets[index, ..., :2])
                listwise = listwise_loss(prediction[..., :2], raw_targets[index, ..., :2])
                loss = 0.25 * regression + 0.5 * pairwise + listwise
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                prediction = model(features[validation_indices])
                validation_loss = float(
                    (
                        0.25 * F.mse_loss(prediction, targets[validation_indices])
                        + 0.5
                        * ranking_loss(
                            prediction[..., :2], raw_targets[validation_indices, ..., :2]
                        )
                        + listwise_loss(
                            prediction[..., :2], raw_targets[validation_indices, ..., :2]
                        )
                    ).item()
                )
            if validation_loss < best_loss - 1e-5:
                best_loss = validation_loss
                best_state = deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= 12:
                break
        if best_state is None:
            raise RuntimeError("Ranker did not produce a checkpoint")
        model.load_state_dict(best_state)
        model.eval()
        models.append(model)
        histories.append({"model": model_index, "best_val_loss": best_loss, "epochs": epoch + 1})

    return models, normalizer, {
        "training_states": int(training_indices.numel()),
        "validation_states": int(validation_indices.numel()),
        "validation_indices": validation_indices.detach().cpu().numpy(),
        "ensemble": histories,
    }


def ensemble_prediction(
    models: list[CounterfactualRanker],
    features: torch.Tensor,
    normalizer: dict[str, torch.Tensor],
) -> torch.Tensor:
    with torch.no_grad():
        normalized = torch.stack([model(features) for model in models])
    return normalized * normalizer["target_std"] + normalizer["target_mean"]


def choose_candidates(
    predictions: np.ndarray,
    horizon_index: int,
    uncertainty_coef: float,
    collision_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    return_prediction = predictions[..., horizon_index]
    collision_prediction = predictions[..., 2 + horizon_index]
    return_mean = return_prediction.mean(axis=0)
    return_std = return_prediction.std(axis=0)
    collision_mean = collision_prediction.mean(axis=0)
    score = return_mean - uncertainty_coef * return_std
    safe = collision_mean <= collision_mean[:, [0]] + collision_tolerance
    safe[:, 0] = True
    score = np.where(safe, score, -np.inf)
    best = score.argmax(axis=1)
    gain = score[np.arange(score.shape[0]), best] - return_mean[:, 0]
    return best, gain


def offline_gate(
    dataset: dict[str, np.ndarray],
    models: list[CounterfactualRanker],
    normalizer: dict[str, torch.Tensor],
    training_info: dict,
    thresholds: list[float],
    uncertainty_coef: float,
    collision_tolerance: float,
    minimum_precision: float,
    minimum_coverage: float,
    device: torch.device,
) -> dict:
    index = training_info["validation_indices"]
    states = torch.as_tensor(dataset["state"][index], device=device)
    base_actions = torch.as_tensor(dataset["base_action"][index], device=device)
    candidates = torch.as_tensor(dataset["candidate_action"][index], device=device)
    features = build_features(
        states,
        base_actions,
        candidates,
        normalizer["state_mean"],
        normalizer["state_std"],
    )
    predictions = ensemble_prediction(models, features, normalizer).detach().cpu().numpy()
    true_returns = dataset["return"][index]
    result: dict[str, dict] = {}

    for horizon_index, horizon in enumerate(HORIZONS):
        best, conservative_gain = choose_candidates(
            predictions, horizon_index, uncertainty_coef, collision_tolerance
        )
        true_values = true_returns[..., horizon_index]
        true_chosen = true_values[np.arange(true_values.shape[0]), best]
        true_base = true_values[:, 0]
        true_gain = true_chosen - true_base
        regret = true_values.max(axis=1) - true_chosen
        predicted_difference = predictions[:, :, :, horizon_index].mean(axis=0)[:, 1:] - predictions[
            :, :, 0, horizon_index
        ].mean(axis=0)[:, None]
        true_difference = true_values[:, 1:] - true_base[:, None]
        sign_mask = np.abs(true_difference) > 0.05
        sign_accuracy = float(
            np.mean(np.sign(predicted_difference[sign_mask]) == np.sign(true_difference[sign_mask]))
        )

        calibration = []
        for threshold in thresholds:
            intervene = (best != 0) & (conservative_gain > threshold)
            count = int(intervene.sum())
            calibration.append(
                {
                    "threshold": threshold,
                    "coverage": float(intervene.mean()),
                    "count": count,
                    "positive_precision": float(np.mean(true_gain[intervene] > 0.0)) if count else 0.0,
                    "mean_true_gain": float(np.mean(true_gain[intervene])) if count else 0.0,
                    "mean_regret": float(np.mean(regret[intervene])) if count else 0.0,
                }
            )
        eligible = [
            item
            for item in calibration
            if item["coverage"] >= minimum_coverage
            and item["positive_precision"] >= minimum_precision
            and item["mean_true_gain"] > 0.0
        ]
        selected = max(eligible, key=lambda item: (item["mean_true_gain"], item["coverage"])) if eligible else None
        result[f"h{horizon}"] = {
            "pairwise_sign_accuracy": sign_accuracy,
            "ungated_top1_regret_mean": float(regret.mean()),
            "ungated_top1_near_optimal_rate": float(np.mean(regret < 0.02)),
            "calibration": calibration,
            "selected_gate": selected,
            "gate_passed": selected is not None,
        }
    return result


def online_action(
    observations: dict[str, np.ndarray],
    base_action: np.ndarray,
    horizon_index: int,
    threshold: float,
    models: list[CounterfactualRanker],
    normalizer: dict[str, torch.Tensor],
    uncertainty_coef: float,
    collision_tolerance: float,
    device: torch.device,
) -> tuple[np.ndarray, bool, float]:
    candidates_np = local_candidates(base_action)
    states = torch.as_tensor(
        ordered_observations(observations).reshape(1, -1), dtype=torch.float32, device=device
    )
    bases = torch.as_tensor(base_action.reshape(1, -1), device=device)
    candidates = torch.as_tensor(candidates_np.reshape(1, N_CANDIDATES, N_AGENTS), device=device)
    features = build_features(
        states, bases, candidates, normalizer["state_mean"], normalizer["state_std"]
    )
    predictions = ensemble_prediction(models, features, normalizer).detach().cpu().numpy()
    best, gain = choose_candidates(
        predictions, horizon_index, uncertainty_coef, collision_tolerance
    )
    intervene = int(best[0]) != 0 and float(gain[0]) > threshold
    return candidates_np[int(best[0])] if intervene else base_action, intervene, float(gain[0])


def evaluate_online(
    policy: nn.Module,
    models: list[CounterfactualRanker],
    normalizer: dict[str, torch.Tensor],
    gates: dict,
    episodes: int,
    seed_base: int,
    uncertainty_coef: float,
    collision_tolerance: float,
    device: torch.device,
) -> tuple[list[dict], dict]:
    modes: list[tuple[str, int | None, float | None]] = [("baseline", None, None)]
    for horizon_index, horizon in enumerate(HORIZONS):
        selected = gates[f"h{horizon}"]["selected_gate"]
        if selected is not None:
            modes.append((f"rank_h{horizon}", horizon_index, float(selected["threshold"])))

    rows: list[dict] = []
    for mode, horizon_index, threshold in modes:
        for episode in range(episodes):
            seed = seed_base + episode
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            environment = make_env()
            observations, _ = environment.reset(seed=seed)
            step_metrics = []
            episode_return = 0.0
            interventions = 0
            gains = []
            for _ in range(25):
                base_action = policy_action(policy, observations)
                action = base_action
                if horizon_index is not None and threshold is not None:
                    action, intervened, gain = online_action(
                        observations,
                        base_action,
                        horizon_index,
                        threshold,
                        models,
                        normalizer,
                        uncertainty_coef,
                        collision_tolerance,
                        device,
                    )
                    interventions += int(intervened)
                    if intervened:
                        gains.append(gain)
                observations, rewards, terminations, truncations, _ = environment.step(
                    action_dict(environment, action)
                )
                episode_return += 2.0 * float(np.mean(list(rewards.values())))
                step_metrics.append(geometry(ordered_observations(observations)))
                if all(terminations.values()) or all(truncations.values()):
                    break
            environment.close()
            final = step_metrics[-1]
            rows.append(
                {
                    "mode": mode,
                    "episode": episode,
                    "seed": seed,
                    "episode_return_legacy_scale": episode_return,
                    "final_hungarian": final["hungarian"],
                    "final_radius_auc": final["radius_auc"],
                    "collision_step_rate": float(np.mean([item["collision"] for item in step_metrics])),
                    "final_coverage_at_010": final["coverage_at_010"],
                    "intervention_rate": interventions / len(step_metrics),
                    "mean_predicted_gain": float(np.mean(gains)) if gains else 0.0,
                }
            )

    metric_names = [
        "episode_return_legacy_scale",
        "final_hungarian",
        "final_radius_auc",
        "collision_step_rate",
        "final_coverage_at_010",
        "intervention_rate",
        "mean_predicted_gain",
    ]
    grouped = {mode: [row for row in rows if row["mode"] == mode] for mode, _, _ in modes}
    summary = {
        mode: {
            metric: summarize(np.asarray([float(row[metric]) for row in mode_rows]))
            for metric in metric_names
        }
        for mode, mode_rows in grouped.items()
    }
    paired = {}
    directions = {
        "episode_return_legacy_scale": 1.0,
        "final_hungarian": -1.0,
        "final_radius_auc": 1.0,
        "collision_step_rate": -1.0,
        "final_coverage_at_010": 1.0,
    }
    baseline = grouped["baseline"]
    for mode in grouped:
        if mode == "baseline":
            continue
        paired[mode] = {}
        for metric, direction in directions.items():
            base_values = np.asarray([float(row[metric]) for row in baseline])
            mode_values = np.asarray([float(row[metric]) for row in grouped[mode]])
            improvement = direction * (mode_values - base_values)
            test = ttest_rel(mode_values, base_values)
            paired[mode][metric] = {
                "improvement_positive_is_better": summarize(improvement),
                "paired_ttest_two_sided_p": float(test.pvalue) if np.isfinite(test.pvalue) else None,
            }
    return rows, {"modes": summary, "paired_vs_baseline": paired}


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.checkpoint.resolve()
    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    experiment = reload_experiment_from_file(str(checkpoint))
    try:
        policy = experiment.policy.to("cpu").eval()
        if args.dataset is None:
            dataset = collect_counterfactual_dataset(policy, args.states, args.collection_seed_base)
            np.savez_compressed(output_dir / "counterfactual_short_horizon_dataset.npz", **dataset)
            dataset_source = str((output_dir / "counterfactual_short_horizon_dataset.npz").resolve())
        else:
            dataset_source = str(args.dataset.resolve())
            with np.load(args.dataset.resolve()) as loaded:
                dataset = {key: loaded[key] for key in loaded.files}
        models, normalizer, training_info = train_ranker_ensemble(
            dataset,
            args.ensemble_size,
            args.epochs,
            args.batch_size,
            args.learning_rate,
            args.training_seed,
            device,
        )
        gates = offline_gate(
            dataset,
            models,
            normalizer,
            training_info,
            args.threshold_grid,
            args.uncertainty_coef,
            args.collision_tolerance,
            args.minimum_precision,
            args.minimum_coverage,
            device,
        )
        rows, online = evaluate_online(
            policy,
            models,
            normalizer,
            gates,
            args.evaluation_episodes,
            args.evaluation_seed_base,
            args.uncertainty_coef,
            args.collision_tolerance,
            device,
        )
    finally:
        experiment.close()

    torch.save(
        {
            "models": [
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
                for model in models
            ],
            "normalizer": {key: value.detach().cpu() for key, value in normalizer.items()},
            "gates": gates,
            "checkpoint_anchor": str(checkpoint),
        },
        output_dir / "counterfactual_ranker_ensemble.pt",
    )
    with (output_dir / "per_episode.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    training_public = {key: value for key, value in training_info.items() if key != "validation_indices"}
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_anchor": str(checkpoint),
        "protocol": {
            "states": args.states,
            "dataset_source": dataset_source,
            "prediction_target": "candidate minus frozen-policy base outcome",
            "training_objective": "listwise best-candidate classification plus pairwise ranking and delta regression",
            "candidates_per_state": N_CANDIDATES,
            "horizons": HORIZONS,
            "gamma": 0.9,
            "ensemble_size": args.ensemble_size,
            "evaluation_episodes": args.evaluation_episodes,
            "uncertainty_coef": args.uncertainty_coef,
            "collision_tolerance": args.collision_tolerance,
            "minimum_precision": args.minimum_precision,
            "minimum_coverage": args.minimum_coverage,
            "threshold_grid": args.threshold_grid,
            "collection_seed_base": args.collection_seed_base,
            "evaluation_seed_base": args.evaluation_seed_base,
        },
        "training": training_public,
        "offline_gate": gates,
        "online_evaluation": online,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
