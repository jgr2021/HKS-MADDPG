"""Causal top-1-to-second-best interventions on frozen 3x3 states."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.maddpg import MADDPG
from experiments.distill_hybrid_counterfactual_actor import (
    REPRESENTATION,
    teacher_bundles,
)
from experiments.evaluate_supervised_probe_policies import geometry
from experiments.probe_action_value_representations import (
    descriptor_tensors,
    representation_matrix,
)
from experiments.probe_and_evaluate_reward_value import candidate_data
from utils.make_env import make_env


RUN_DIR = Path(
    "experiments/hybrid_first_update_confirmation_20260716/"
    "run_20260717_004335"
)
FROZEN_CHECKPOINT = (
    RUN_DIR / "hybrid_frozen/seed_60/checkpoints/model_step64.pt"
)
FIRST_UPDATE_CHECKPOINTS = {
    seed: RUN_DIR / (
        f"hybrid_first_update_then_freeze/seed_{seed}/checkpoints/"
        "model_step100.pt"
    )
    for seed in (60, 61, 62, 63, 64)
}
DATASET = Path(
    "experiments/reward_probe_dataset_20260712/counterfactual_dataset.npz"
)
OUTCOMES = (
    "hungarian_assignment_distance",
    "coverage_radius_auc",
    "collision_step_rate",
    "minimum_agent_separation",
    "return",
)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_probabilities(model, observations):
    output = []
    with torch.no_grad():
        for agent_index, agent in enumerate(model.agents):
            values = torch.from_numpy(
                observations[:, agent_index, :].astype(np.float32)
            )
            batches = []
            for start in range(0, len(values), 2048):
                logits = agent.policy(values[start:start + 2048])
                batches.append(torch.softmax(logits, dim=1).cpu().numpy())
            output.append(np.concatenate(batches))
    return np.stack(output, axis=1)


def snapshot_world(env):
    return {
        "agent_pos": np.asarray([
            agent.state.p_pos.copy() for agent in env.world.agents
        ]),
        "agent_vel": np.asarray([
            agent.state.p_vel.copy() for agent in env.world.agents
        ]),
        "agent_comm": np.asarray([
            agent.state.c.copy() for agent in env.world.agents
        ]),
        "landmark_pos": np.asarray([
            landmark.state.p_pos.copy() for landmark in env.world.landmarks
        ]),
        "landmark_vel": np.asarray([
            landmark.state.p_vel.copy() for landmark in env.world.landmarks
        ]),
    }


def restore_world(env, snapshot):
    for index, agent in enumerate(env.world.agents):
        agent.state.p_pos = snapshot["agent_pos"][index].copy()
        agent.state.p_vel = snapshot["agent_vel"][index].copy()
        agent.state.c = snapshot["agent_comm"][index].copy()
    for index, landmark in enumerate(env.world.landmarks):
        landmark.state.p_pos = snapshot["landmark_pos"][index].copy()
        landmark.state.p_vel = snapshot["landmark_vel"][index].copy()


def collect_frozen_states(model, episodes, horizon, seed_base, time_stride):
    env = make_env("simple_spread", discrete_action=True)
    states = []
    try:
        for episode in range(episodes):
            seed = seed_base + episode
            torch.manual_seed(seed)
            np.random.seed(seed)
            env.seed(seed)
            obs = np.asarray(env.reset(), dtype=np.float32)
            for step in range(horizon):
                if step % time_stride == 0:
                    states.append({
                        "state_id": len(states),
                        "episode": episode,
                        "episode_seed": seed,
                        "time_step": step,
                        "obs": obs.copy(),
                        **snapshot_world(env),
                    })
                tensors = [
                    torch.from_numpy(value).view(1, -1) for value in obs
                ]
                with torch.no_grad():
                    actions = [
                        action.cpu().numpy().ravel()
                        for action in model.step(tensors, explore=False)
                    ]
                obs, _, dones, _ = env.step(actions)
                obs = np.asarray(obs, dtype=np.float32)
                if all(dones):
                    break
    finally:
        env.close()
    return states


def save_states(path, states):
    keys = (
        "obs", "agent_pos", "agent_vel", "agent_comm",
        "landmark_pos", "landmark_vel",
    )
    payload = {
        key: np.stack([state[key] for state in states]) for key in keys
    }
    for key in ("state_id", "episode", "episode_seed", "time_step"):
        payload[key] = np.asarray([state[key] for state in states])
    np.savez_compressed(path, **payload)


def onehot(indices):
    return [np.eye(5, dtype=np.float32)[int(index)] for index in indices]


def rollout_branch(env, model, snapshot, first_actions, remaining_steps):
    restore_world(env, snapshot)
    obs, rewards, dones, _ = env.step(onehot(first_actions))
    obs = np.asarray(obs, dtype=np.float32)
    total_return = float(np.mean(rewards))
    assignment, auc, separation, collision, coverage = geometry(env)
    collision_steps = collision
    minimum_separation = separation
    elapsed = 1
    for _ in range(max(0, remaining_steps - 1)):
        tensors = [
            torch.from_numpy(value).view(1, -1) for value in obs
        ]
        with torch.no_grad():
            actions = [
                action.cpu().numpy().ravel()
                for action in model.step(tensors, explore=False)
            ]
        obs, rewards, dones, _ = env.step(actions)
        obs = np.asarray(obs, dtype=np.float32)
        total_return += float(np.mean(rewards))
        assignment, auc, separation, collision, coverage = geometry(env)
        collision_steps += collision
        minimum_separation = min(minimum_separation, separation)
        elapsed += 1
        if all(dones):
            break
    return {
        "hungarian_assignment_distance": assignment,
        "coverage_radius_auc": auc,
        "collision_step_rate": collision_steps / float(elapsed),
        "minimum_agent_separation": minimum_separation,
        "return": total_return,
        "final_coverage": coverage,
    }


def teacher_scores(obs, focal, hungarian, reward):
    descriptors = descriptor_tensors(obs)
    data = candidate_data(obs, focal, descriptors)
    x = representation_matrix(data, REPRESENTATION)
    with torch.no_grad():
        hx = torch.from_numpy((x - hungarian["x_mean"]) / hungarian["x_std"])
        h = hungarian["model"](hx).numpy()[:, 0]
        h = h * hungarian["y_std"][0] + hungarian["y_mean"][0]
        rx = torch.from_numpy((x - reward["x_mean"]) / reward["x_std"])
        r = reward["model"](rx).numpy()
        r = r * reward["y_std"] + reward["y_mean"]
    r = r - r[0]
    feasible = r >= 0.0
    feasible_indices = np.flatnonzero(feasible)
    teacher_action = int(feasible_indices[np.argmin(h[feasible_indices])])
    return h, r, feasible, teacher_action


def cluster_bootstrap(rows, selector, value_key, rng, samples=10000):
    episodes = sorted({int(row["episode"]) for row in rows})
    per_episode = []
    for episode in episodes:
        values = np.asarray([
            float(row[value_key]) for row in rows
            if int(row["episode"]) == episode and selector(row)
        ], dtype=np.float64)
        per_episode.append((values.sum(), values.size))
    sums = np.asarray([value[0] for value in per_episode])
    counts = np.asarray([value[1] for value in per_episode])
    if counts.sum() == 0:
        return float("nan"), float("nan"), float("nan"), 0
    observed = float(sums.sum() / counts.sum())
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.integers(0, len(episodes), len(episodes))
        count = counts[selected].sum()
        estimates[index] = (
            sums[selected].sum() / count if count else np.nan
        )
    estimates = estimates[np.isfinite(estimates)]
    low, high = np.percentile(estimates, [2.5, 97.5])
    return observed, float(low), float(high), int(counts.sum())


def cluster_difference(rows, lhs, rhs, value_key, rng, samples=10000):
    episodes = sorted({int(row["episode"]) for row in rows})
    payload = []
    for episode in episodes:
        selected = [row for row in rows if int(row["episode"]) == episode]
        lhs_values = np.asarray([
            float(row[value_key]) for row in selected if lhs(row)
        ])
        rhs_values = np.asarray([
            float(row[value_key]) for row in selected if rhs(row)
        ])
        payload.append((
            lhs_values.sum(), lhs_values.size,
            rhs_values.sum(), rhs_values.size,
        ))
    values = np.asarray(payload, dtype=np.float64)
    observed = (
        values[:, 0].sum() / values[:, 1].sum()
        - values[:, 2].sum() / values[:, 3].sum()
    )
    estimates = []
    for _ in range(samples):
        indices = rng.integers(0, len(episodes), len(episodes))
        sample = values[indices]
        if sample[:, 1].sum() and sample[:, 3].sum():
            estimates.append(
                sample[:, 0].sum() / sample[:, 1].sum()
                - sample[:, 2].sum() / sample[:, 3].sum()
            )
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(observed), float(low), float(high)


def binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    return float(
        ((positive[:, None] > negative[None, :]).mean()
         + 0.5 * (positive[:, None] == negative[None, :]).mean())
    )


def t_interval(values):
    values = np.asarray(values, dtype=np.float64)
    critical = {2: 4.3027, 4: 2.7764}.get(len(values) - 1, 1.96)
    half = critical * values.std(ddof=1) / math.sqrt(len(values))
    return float(values.mean()), float(values.mean() - half), float(values.mean() + half)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="experiments/margin_coordination_intervention_20260811",
    )
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--time-stride", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=41_000_000)
    parser.add_argument("--bootstrap-seed", type=int, default=4100811)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)

    initial_model = MADDPG.init_from_save(str(FROZEN_CHECKPOINT))
    initial_model.prep_rollouts(device="cpu")
    update_models = {}
    for seed, checkpoint in FIRST_UPDATE_CHECKPOINTS.items():
        model = MADDPG.init_from_save(str(checkpoint))
        model.prep_rollouts(device="cpu")
        update_models[seed] = model

    states = collect_frozen_states(
        initial_model, args.episodes, args.horizon,
        args.seed_base, args.time_stride
    )
    save_states(output / "frozen_states.npz", states)
    observations = np.stack([state["obs"] for state in states])
    initial_probs = model_probabilities(initial_model, observations)
    update_probs = {
        seed: model_probabilities(model, observations)
        for seed, model in update_models.items()
    }
    flat_margins = []
    for state_index in range(len(states)):
        for focal in range(3):
            values = np.sort(initial_probs[state_index, focal])
            flat_margins.append(values[-1] - values[-2])
    quartiles = np.quantile(flat_margins, [0.25, 0.50, 0.75])

    with np.load(DATASET) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    hungarian, reward, _, _ = teacher_bundles(data, 1, 23)
    branch_env = make_env("simple_spread", discrete_action=True)
    rows = []
    try:
        for state_index, state in enumerate(states):
            probabilities = initial_probs[state_index]
            top_actions = probabilities.argmax(axis=1)
            baseline = rollout_branch(
                branch_env, initial_model, state, top_actions,
                args.horizon - int(state["time_step"])
            )
            for focal in range(3):
                order = np.argsort(probabilities[focal])
                top1, second = int(order[-1]), int(order[-2])
                margin = float(
                    probabilities[focal, top1]
                    - probabilities[focal, second]
                )
                margin_quartile = int(np.searchsorted(
                    quartiles, margin, side="left"
                )) + 1
                intervention_actions = top_actions.copy()
                intervention_actions[focal] = second
                intervention = rollout_branch(
                    branch_env, initial_model, state, intervention_actions,
                    args.horizon - int(state["time_step"])
                )
                h_score, r_score, feasible, teacher_action = teacher_scores(
                    state["obs"], focal, hungarian, reward
                )
                entropy = float(-np.sum(
                    probabilities[focal]
                    * np.log(np.clip(probabilities[focal], 1e-12, None))
                ))
                row = {
                    "state_id": state["state_id"],
                    "episode": state["episode"],
                    "episode_seed": state["episode_seed"],
                    "time_step": state["time_step"],
                    "focal_agent": focal,
                    "initial_top1_action": top1,
                    "initial_second_action": second,
                    "initial_margin": margin,
                    "initial_entropy": entropy,
                    "margin_quartile": margin_quartile,
                    "teacher_action": teacher_action,
                    "top1_matches_teacher": top1 == teacher_action,
                    "second_matches_teacher": second == teacher_action,
                    "teacher_h_top1": float(h_score[top1]),
                    "teacher_h_second": float(h_score[second]),
                    "teacher_h_second_minus_top1": float(
                        h_score[second] - h_score[top1]
                    ),
                    "teacher_reward_top1": float(r_score[top1]),
                    "teacher_reward_second": float(r_score[second]),
                    "teacher_reward_second_minus_top1": float(
                        r_score[second] - r_score[top1]
                    ),
                    "top1_teacher_feasible": bool(feasible[top1]),
                    "second_teacher_feasible": bool(feasible[second]),
                    "teacher_feasible_action_count": int(feasible.sum()),
                    "first_update_changed_count": int(sum(
                        update_probs[seed][state_index, focal].argmax() != top1
                        for seed in update_probs
                    )),
                }
                for seed in update_probs:
                    row[f"first_update_changed_seed_{seed}"] = bool(
                        update_probs[seed][state_index, focal].argmax() != top1
                    )
                for outcome in OUTCOMES:
                    row["baseline_" + outcome] = baseline[outcome]
                    row["intervention_" + outcome] = intervention[outcome]
                    row["delta_" + outcome] = (
                        intervention[outcome] - baseline[outcome]
                    )
                rows.append(row)
    finally:
        branch_env.close()
    write_csv(output / "per_intervention.csv", rows)

    rng = np.random.default_rng(args.bootstrap_seed)
    stratified = []
    for quartile in (1, 2, 3, 4):
        selector = lambda row, q=quartile: int(row["margin_quartile"]) == q
        for outcome in OUTCOMES:
            mean, low, high, count = cluster_bootstrap(
                rows, selector, "delta_" + outcome, rng
            )
            stratified.append({
                "stratum": f"margin_q{quartile}",
                "outcome": outcome,
                "decisions": count,
                "mean_delta_intervention_minus_baseline": mean,
                "episode_cluster_bootstrap_ci95_low": low,
                "episode_cluster_bootstrap_ci95_high": high,
            })
    for outcome in OUTCOMES:
        mean, low, high = cluster_difference(
            rows,
            lambda row: int(row["margin_quartile"]) == 4,
            lambda row: int(row["margin_quartile"]) == 1,
            "delta_" + outcome,
            rng,
        )
        stratified.append({
            "stratum": "margin_q4_minus_q1",
            "outcome": outcome,
            "decisions": len(rows),
            "mean_delta_intervention_minus_baseline": mean,
            "episode_cluster_bootstrap_ci95_low": low,
            "episode_cluster_bootstrap_ci95_high": high,
        })
    write_csv(output / "margin_stratified_effects.csv", stratified)

    update_group_rows = []
    actor_group_means = []
    for seed in sorted(update_probs):
        for changed in (False, True):
            selector = lambda row, s=seed, c=changed: bool(
                row[f"first_update_changed_seed_{s}"]
            ) == c
            for outcome in OUTCOMES:
                mean, low, high, count = cluster_bootstrap(
                    rows, selector, "delta_" + outcome, rng
                )
                update_group_rows.append({
                    "first_update_seed": seed,
                    "group": "changed" if changed else "unchanged",
                    "outcome": outcome,
                    "decisions": count,
                    "mean_delta": mean,
                    "episode_cluster_bootstrap_ci95_low": low,
                    "episode_cluster_bootstrap_ci95_high": high,
                })
                actor_group_means.append({
                    "first_update_seed": seed,
                    "group": "changed" if changed else "unchanged",
                    "outcome": outcome,
                    "mean_delta": mean,
                })
    write_csv(output / "first_update_changed_strata.csv", update_group_rows)
    actor_intervals = []
    for group in ("changed", "unchanged"):
        for outcome in OUTCOMES:
            values = [
                row["mean_delta"] for row in actor_group_means
                if row["group"] == group and row["outcome"] == outcome
            ]
            mean, low, high = t_interval(values)
            actor_intervals.append({
                "group": group,
                "outcome": outcome,
                "n_first_update_actors": len(values),
                "actor_mean_delta": mean,
                "actor_t_ci95_low": low,
                "actor_t_ci95_high": high,
            })
    write_csv(output / "first_update_actor_intervals.csv", actor_intervals)

    ambiguity = []
    indicators = {
        "negative_margin": lambda row: -float(row["initial_margin"]),
        "entropy": lambda row: float(row["initial_entropy"]),
        "negative_abs_teacher_h_gap": lambda row: -abs(float(
            row["teacher_h_second_minus_top1"]
        )),
        "feasible_action_count": lambda row: float(
            row["teacher_feasible_action_count"]
        ),
    }
    for name, function in indicators.items():
        scores = np.asarray([function(row) for row in rows])
        for outcome in (
            "hungarian_assignment_distance", "coverage_radius_auc"
        ):
            values = np.asarray([
                float(row["delta_" + outcome]) for row in rows
            ])
            ambiguity.append({
                "indicator": name,
                "target": "intervention_delta_" + outcome,
                "value": float(np.corrcoef(scores, values)[0, 1]),
                "statistic": "pearson_correlation",
            })
        for seed in sorted(update_probs):
            labels = [
                bool(row[f"first_update_changed_seed_{seed}"]) for row in rows
            ]
            ambiguity.append({
                "indicator": name,
                "target": f"first_update_changed_seed_{seed}",
                "value": binary_auc(labels, scores),
                "statistic": "roc_auc",
            })
    write_csv(output / "ambiguity_indicators.csv", ambiguity)

    lookup = {
        (row["stratum"], row["outcome"]): row for row in stratified
    }
    h_difference = lookup[(
        "margin_q4_minus_q1", "hungarian_assignment_distance"
    )]
    auc_difference = lookup[(
        "margin_q4_minus_q1", "coverage_radius_auc"
    )]
    criticality_gate = (
        h_difference["episode_cluster_bootstrap_ci95_low"] > 0.0
        and auc_difference["episode_cluster_bootstrap_ci95_high"] < 0.0
    )
    lines = [
        "# Margin to Coordination-Criticality Intervention", "",
        f"Frozen states: {len(states)} from {args.episodes} episodes; focal interventions: {len(rows)}.",
        "Non-focal actions and the pre-action world state are held fixed. The focal initial top-1 action is replaced by its initial second-best action, then both branches continue with the frozen initial actor.", "",
        f"Predeclared high-margin criticality gate: **{'PASS' if criticality_gate else 'FAIL'}**.", "",
        "| margin stratum | delta H [cluster 95% CI] | delta AUC [cluster 95% CI] |",
        "| --- | ---: | ---: |",
    ]
    for quartile in (1, 2, 3, 4):
        h = lookup[(f"margin_q{quartile}", "hungarian_assignment_distance")]
        auc = lookup[(f"margin_q{quartile}", "coverage_radius_auc")]
        lines.append(
            f"| Q{quartile} | {h['mean_delta_intervention_minus_baseline']:+.5f} "
            f"[{h['episode_cluster_bootstrap_ci95_low']:+.5f}, {h['episode_cluster_bootstrap_ci95_high']:+.5f}] | "
            f"{auc['mean_delta_intervention_minus_baseline']:+.5f} "
            f"[{auc['episode_cluster_bootstrap_ci95_low']:+.5f}, {auc['episode_cluster_bootstrap_ci95_high']:+.5f}] |"
        )
    lines += ["", (
        "The stronger causal high-margin claim is supported under the locked gate."
        if criticality_gate else
        "The locked causal gate is not met; retain the weaker observational claim that high-margin flips accompany failure."
    ), ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")

    q_rows = [
        lookup[(f"margin_q{quartile}", outcome)]
        for outcome in ("hungarian_assignment_distance", "coverage_radius_auc")
        for quartile in (1, 2, 3, 4)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, outcome, label in zip(
        axes,
        ("hungarian_assignment_distance", "coverage_radius_auc"),
        ("Delta H", "Delta AUC"),
    ):
        selected = [
            lookup[(f"margin_q{quartile}", outcome)]
            for quartile in (1, 2, 3, 4)
        ]
        means = np.asarray([
            row["mean_delta_intervention_minus_baseline"] for row in selected
        ])
        low = np.asarray([
            row["episode_cluster_bootstrap_ci95_low"] for row in selected
        ])
        high = np.asarray([
            row["episode_cluster_bootstrap_ci95_high"] for row in selected
        ])
        ax.errorbar(
            range(1, 5), means, yerr=[means - low, high - means],
            marker="o", capsize=3
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(range(1, 5), ["Q1", "Q2", "Q3", "Q4"])
        ax.set_xlabel("initial margin quartile")
        ax.set_ylabel(label + " (second-best - top-1)")
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "margin_intervention_effects.png", dpi=220)
    fig.savefig(output / "margin_intervention_effects.pdf")
    plt.close(fig)

    (output / "metadata.json").write_text(json.dumps({
        "protocol_date": "2026-08-11",
        "frozen_checkpoint": str(FROZEN_CHECKPOINT),
        "first_update_checkpoints": {
            str(seed): str(path) for seed, path in FIRST_UPDATE_CHECKPOINTS.items()
        },
        "episodes": args.episodes,
        "horizon": args.horizon,
        "time_stride": args.time_stride,
        "frozen_states": len(states),
        "focal_interventions": len(rows),
        "margin_quartiles": quartiles.tolist(),
        "bootstrap": "episode-cluster nonparametric, 10000 samples",
        "high_margin_criticality_gate_pass": bool(criticality_gate),
    }, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
