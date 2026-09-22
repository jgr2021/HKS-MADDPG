"""Locked audit of the corrected action-aware Active-GSP checkpoints.

This script is evaluation-only.  It replays deterministic evaluation seeds,
hashes every source checkpoint, and never mutates a policy parameter.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.maddpg import MADDPG
from utils.active_gsp_v3_features import (
    ACTION_TO_CONTROL,
    DEFAULT_DAMPING,
    DEFAULT_DT,
    DEFAULT_SENSITIVITY,
)
from utils.gsp_features import (
    AGENT_AGENT_SCALE,
    SIGMA_AGENT_AGENT,
    SIGMA_AGENT_LANDMARK,
)
from utils.make_env import make_env


METHODS = ("raw_mlp", "learned_raw_potential_residual", "learned_active_gsp_residual")
LABELS = {
    "raw_mlp": "Raw MADDPG",
    "learned_raw_potential_residual": "Action-aware geometric control",
    "learned_active_gsp_residual": "Current Active GSP",
}
CHANNELS = (
    "delta_coverage_potential",
    "delta_crowding_potential",
    "delta_coverage_energy",
    "delta_crowding_energy",
)
PERMUTATIONS = tuple(itertools.permutations(range(3)))


def source_map(args):
    seed1_active = Path(args.seed1_active_root)
    seed1_raw = Path(args.seed1_raw_root)
    seeds23 = Path(args.seeds23_root)
    sources = {
        1: {
            "raw_mlp": seed1_raw / "raw_mlp" / "seed_1",
            "learned_raw_potential_residual": seed1_active / "learned_raw_potential_residual" / "seed_1",
            "learned_active_gsp_residual": seed1_active / "learned_active_gsp_residual" / "seed_1",
        }
    }
    for seed in (2, 3):
        sources[seed] = {method: seeds23 / method / f"seed_{seed}" for method in METHODS}
    return sources


def final_model(run_dir):
    matches = sorted((run_dir / "checkpoints").glob("model_final_*.pt"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one final checkpoint under {run_dir}, got {matches}")
    return matches[0]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_geometry(raw_obs, action):
    raw = np.asarray(raw_obs, dtype=np.float64)
    landmarks = raw[4:10].reshape(3, 2)
    others = raw[10:14].reshape(2, 2)
    displacement = (
        raw[0:2] * (1.0 - DEFAULT_DAMPING) * DEFAULT_DT
        + ACTION_TO_CONTROL[action] * DEFAULT_SENSITIVITY * DEFAULT_DT ** 2
    )
    agents = np.concatenate([displacement[None, :], others], axis=0)
    return agents, landmarks


def adjacency_and_signals(agents, landmarks):
    adjacency = np.zeros((6, 6), dtype=np.float64)
    aa_dist = np.linalg.norm(agents[:, None, :] - agents[None, :, :], axis=2)
    aa = AGENT_AGENT_SCALE * np.exp(-(aa_dist ** 2) / (2.0 * SIGMA_AGENT_AGENT ** 2))
    np.fill_diagonal(aa, 0.0)
    adjacency[:3, :3] = aa
    al_dist = np.linalg.norm(agents[:, None, :] - landmarks[None, :, :], axis=2)
    al = np.exp(-(al_dist ** 2) / (2.0 * SIGMA_AGENT_LANDMARK ** 2))
    adjacency[:3, 3:] = al
    adjacency[3:, :3] = al.T
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    coverage = np.concatenate([np.zeros(3), al_dist.min(axis=0)])
    masked = aa_dist + np.eye(3) * 1e6
    nearest = masked.min(axis=1)
    crowding = np.concatenate([
        AGENT_AGENT_SCALE * np.exp(-(nearest ** 2) / (2.0 * SIGMA_AGENT_AGENT ** 2)),
        np.zeros(3),
    ])
    return laplacian, coverage, crowding


def energy_decomposition(raw_obs):
    states = []
    for action in range(5):
        agents, landmarks = candidate_geometry(raw_obs, action)
        states.append(adjacency_and_signals(agents, landmarks))
    l0, c0, r0 = states[0]
    output = np.zeros((5, 2, 4), dtype=np.float64)
    for action, (la, ca, ra) in enumerate(states):
        for signal_index, (x0, xa) in enumerate(((c0, ca), (r0, ra))):
            base = x0 @ l0 @ x0
            total = xa @ la @ xa - base
            signal_only = xa @ l0 @ xa - base
            graph_only = x0 @ la @ x0 - base
            interaction = total - signal_only - graph_only
            output[action, signal_index] = total, signal_only, graph_only, interaction
    return output


def residual_with_features(policy, obs, features):
    batch_size = obs.shape[0]
    scaled = torch.tanh(features * policy.feature_scale.to(obs))
    context = policy.context_encoder(obs).unsqueeze(1).expand(-1, 5, -1)
    feature_hidden = policy.feature_encoder(scaled)
    action_hidden = policy.action_embedding(policy.action_indices).unsqueeze(0).expand(batch_size, -1, -1)
    return policy.residual_scorer(torch.cat([context, feature_hidden, action_hidden], dim=2)).squeeze(2)


def assignment_distance(distances):
    return min(sum(distances[i, perm[i]] for i in range(3)) for perm in PERMUTATIONS) / 3.0


def geometry(env):
    agents = np.asarray([agent.state.p_pos for agent in env.world.agents], dtype=np.float64)
    landmarks = np.asarray([landmark.state.p_pos for landmark in env.world.landmarks], dtype=np.float64)
    distances = np.linalg.norm(agents[:, None, :] - landmarks[None, :, :], axis=2)
    separations = [np.linalg.norm(agents[i] - agents[j]) for i in range(3) for j in range(i + 1, 3)]
    collision = any(
        separations[k] < env.world.agents[i].size + env.world.agents[j].size
        for k, (i, j) in enumerate(((0, 1), (0, 2), (1, 2)))
    )
    return distances, float(min(separations)), int(collision)


def radius_auc(nearest):
    # Normalized integral of covered-landmark fraction over r in [0.05, 0.30].
    contribution = np.clip(0.30 - np.maximum(nearest, 0.05), 0.0, 0.25) / 0.25
    return float(contribution.mean())


def evaluate(model_path, method, train_seed, episodes, episode_length, feature_store, attribution_store, decomposition_store):
    maddpg = MADDPG.init_from_save(str(model_path))
    maddpg.prep_rollouts(device="cpu")
    for parameter in itertools.chain.from_iterable(agent.policy.parameters() for agent in maddpg.agents):
        parameter.requires_grad_(False)
    env = make_env("simple_spread", discrete_action=maddpg.discrete_action)
    rows = []
    seed_base = 990000 + train_seed * 10000
    try:
        for episode in range(episodes):
            test_seed = seed_base + episode
            torch.manual_seed(test_seed)
            np.random.seed(test_seed)
            env.seed(test_seed)
            obs = env.reset()
            episode_return = 0.0
            min_separation = math.inf
            collision_steps = 0
            final_distances = None
            for _ in range(episode_length):
                torch_obs = [torch.as_tensor(item, dtype=torch.float32).view(1, -1) for item in obs]
                if method == "learned_active_gsp_residual":
                    with torch.no_grad():
                        for agent_index, (agent, item) in enumerate(zip(maddpg.agents, torch_obs)):
                            policy = agent.policy
                            features = policy.action_features(item)
                            feature_store.append(features[0, 1:, :].cpu().numpy())
                            decomposition_store.append(energy_decomposition(obs[agent_index])[1:, :, :])
                            full_residual = residual_with_features(policy, item, features)
                            full_centered = full_residual - full_residual.mean(dim=1, keepdim=True)
                            for channel, name in enumerate(CHANNELS):
                                masked = features.clone()
                                masked[:, :, channel] = 0.0
                                masked_residual = residual_with_features(policy, item, masked)
                                masked_centered = masked_residual - masked_residual.mean(dim=1, keepdim=True)
                                effect = (full_centered - masked_centered)[0].cpu().numpy()
                                attribution_store[name]["effects"].append(effect)
                                attribution_store[name]["flips"].append(
                                    int(full_residual.argmax(1).item() != masked_residual.argmax(1).item())
                                )
                with torch.no_grad():
                    torch_actions = maddpg.step(torch_obs, explore=False)
                actions = [item.cpu().numpy().ravel() for item in torch_actions]
                obs, rewards, dones, _ = env.step(actions)
                episode_return += float(np.mean(rewards))
                final_distances, separation, collision = geometry(env)
                min_separation = min(min_separation, separation)
                collision_steps += collision
                if all(dones):
                    break
            nearest = final_distances.min(axis=0)
            rows.append({
                "method": method,
                "train_seed": train_seed,
                "eval_episode": episode,
                "test_seed": test_seed,
                "return": episode_return,
                "hungarian_assignment_distance": assignment_distance(final_distances),
                "coverage_radius_auc": radius_auc(nearest),
                "collision_step_rate": collision_steps / float(episode_length),
                "minimum_agent_separation": min_separation,
                "final_coverage_r010": int((nearest < 0.10).sum()),
            })
    finally:
        env.close()
    return rows


def distribution_row(name, values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "channel": name, "count": values.size, "mean": values.mean(), "std": values.std(),
        "min": values.min(), "p01": np.quantile(values, 0.01), "p05": np.quantile(values, 0.05),
        "p50": np.quantile(values, 0.50), "p95": np.quantile(values, 0.95),
        "p99": np.quantile(values, 0.99), "max": values.max(),
        "near_zero_rate": np.mean(np.abs(values) < 1e-8),
    }


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed1-raw-root", required=True)
    parser.add_argument("--seed1-active-root", required=True)
    parser.add_argument("--seeds23-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--episode-length", type=int, default=25)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    sources = source_map(args)
    manifest = []
    feature_store = []
    attribution_store = defaultdict(lambda: defaultdict(list))
    decomposition_store = []
    episode_rows = []
    for seed in (1, 2, 3):
        for method in METHODS:
            checkpoint = final_model(sources[seed][method])
            manifest.append({
                "method": method, "train_seed": seed, "path": str(checkpoint.resolve()),
                "bytes": checkpoint.stat().st_size, "sha256": sha256(checkpoint),
            })
            episode_rows.extend(evaluate(
                checkpoint, method, seed, args.episodes, args.episode_length,
                feature_store, attribution_store, decomposition_store,
            ))

    features = np.concatenate(feature_store, axis=0)
    decompositions = np.concatenate(decomposition_store, axis=0)
    distribution_rows = [distribution_row(name, features[:, index]) for index, name in enumerate(CHANNELS)]
    correlations = [{
        "pair": "coverage_potential_vs_coverage_energy",
        "pearson_r": np.corrcoef(features[:, 0], features[:, 2])[0, 1],
        "count": features.shape[0],
    }, {
        "pair": "crowding_potential_vs_crowding_energy",
        "pearson_r": np.corrcoef(features[:, 1], features[:, 3])[0, 1],
        "count": features.shape[0],
    }]
    attribution_rows = []
    for name in CHANNELS:
        effects = np.asarray(attribution_store[name]["effects"]).reshape(-1)
        attribution_rows.append({
            "channel": name, "mean_abs_centered_logit_effect": np.abs(effects).mean(),
            "rms_centered_logit_effect": np.sqrt(np.mean(effects ** 2)),
            "p95_abs_centered_logit_effect": np.quantile(np.abs(effects), 0.95),
            "masked_argmax_flip_rate": np.mean(attribution_store[name]["flips"]),
        })
    decomposition_rows = []
    for signal_index, signal in enumerate(("coverage", "crowding")):
        for component_index, component in enumerate(("total", "signal_only", "graph_only", "interaction")):
            decomposition_rows.append(distribution_row(f"{signal}_{component}", decompositions[:, signal_index, component_index]))

    seed_summaries = []
    for method in METHODS:
        for seed in (1, 2, 3):
            selected = [row for row in episode_rows if row["method"] == method and row["train_seed"] == seed]
            summary = {"method": method, "train_seed": seed, "episodes": len(selected)}
            for metric in ("return", "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate", "minimum_agent_separation", "final_coverage_r010"):
                summary[metric] = np.mean([row[metric] for row in selected])
            seed_summaries.append(summary)

    write_csv(output_dir / "checkpoint_manifest.csv", manifest)
    write_csv(output_dir / "episode_metrics.csv", episode_rows)
    write_csv(output_dir / "seed_metrics.csv", seed_summaries)
    write_csv(output_dir / "channel_distributions.csv", distribution_rows)
    write_csv(output_dir / "channel_correlations.csv", correlations)
    write_csv(output_dir / "residual_logit_attribution.csv", attribution_rows)
    write_csv(output_dir / "energy_decomposition.csv", decomposition_rows)
    metadata = {
        "policy_parameters_changed": False,
        "evaluation_deterministic": True,
        "evaluation_seed_formula": "990000 + train_seed*10000 + episode",
        "spectral_channels": ["no-op-relative coverage Dirichlet-energy delta", "no-op-relative crowding Dirichlet-energy delta"],
        "dirichlet_laplacian": "combinatorial D-A (the current production fast path)",
        "feature_statistics_exclude_noop": True,
        "coverage_auc": "normalized integral of final covered-landmark fraction over radius [0.05,0.30]",
        "attribution": "full minus leave-one-channel-out learned residual logits, centered over actions",
    }
    (output_dir / "audit_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
