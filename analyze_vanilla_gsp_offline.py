"""
Offline graph-spectral diagnostic for the already trained Vanilla MADDPG policy.

This script DOES NOT modify training code, the environment, the reward, or the model.
It only:
  1) runs deterministic evaluation episodes with an existing checkpoint;
  2) reconstructs a fixed-order 6-node agent--landmark graph at every state;
  3) computes A_t, normalized L_t, eigenvalues, and HKS;
  4) saves figures and machine-readable traces for offline inspection.

Run from the project root, e.g.
    python analyze_vanilla_gsp_offline.py

The graph used here is an analysis graph, not yet a training input.
Node order is fixed globally for easy visualization:
    [A0, A1, A2, L0, L1, L2]

Important interpretation note:
"crowded" means that two or more agents have the same nearest landmark at a
snapshot. It is a geometric proxy for possible duplicate pursuit, not a direct
observation of the agents' latent intentions.
"""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable

from algorithms.maddpg import MADDPG
from utils.make_env import make_env


# ============================================================================
# 1. Configuration: only edit this block when analysing a different checkpoint
# ============================================================================
ENV_ID = "simple_spread"
MODEL_NAME = "spread_maddpg_original_gumbel_fast4_100k_seed1"
RUN_NUM = 1
CHECKPOINT_EPISODE: Optional[int] = None   # None -> model.pt; e.g. 95001 -> model_ep95001.pt
EVAL_SEED = 2040
N_EPISODES = 20
EPISODE_LENGTH = 25

# Analysis-graph hyperparameters. These are NOT environment or RL parameters.
# They must be fixed before any later GSP training experiment.
SIGMA_AGENT_LANDMARK = 0.60
SIGMA_AGENT_AGENT = 0.80
AGENT_AGENT_SCALE = 0.25
HKS_TIMES = (0.5, 1.0, 2.0)
COVERAGE_THRESHOLD = 0.10

OUTPUT_DIR_NAME = "gsp_offline_analysis_seed2040"
DPI = 180

AGENT_NAMES = ["A0", "A1", "A2"]
LANDMARK_NAMES = ["L0", "L1", "L2"]
NODE_NAMES = AGENT_NAMES + LANDMARK_NAMES


@dataclass
class Snapshot:
    episode: int
    step: int
    agent_pos: np.ndarray
    landmark_pos: np.ndarray
    al_dist: np.ndarray
    adjacency: np.ndarray
    laplacian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    hks_agents: np.ndarray
    coverage: int
    min_dist_sum: float
    nearest_landmark_by_agent: np.ndarray
    unique_nearest_targets: int
    assignment: np.ndarray
    assignment_cost: float
    collision_pairs: int
    actions: Optional[np.ndarray] = None
    reward: Optional[float] = None


def gaussian_affinity(distance: float, sigma: float) -> float:
    return float(np.exp(-(distance ** 2) / (2.0 * sigma ** 2)))


def build_graph_from_world(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a fixed-order 6-node graph from the current MPE world state.

    Nodes: [A0, A1, A2, L0, L1, L2].
    Edges:
      A_i--A_j: AGENT_AGENT_SCALE * exp(-||p_i-p_j||^2 / 2 sigma_AA^2)
      A_i--L_l: exp(-||p_i-g_l||^2 / 2 sigma_AL^2)
      L_l--L_m: no direct edge
    """
    agent_pos = np.vstack([a.state.p_pos.copy() for a in env.world.agents]).astype(np.float64)
    landmark_pos = np.vstack([l.state.p_pos.copy() for l in env.world.landmarks]).astype(np.float64)

    n_agents, n_landmarks = 3, 3
    adjacency = np.zeros((n_agents + n_landmarks, n_agents + n_landmarks), dtype=np.float64)

    # Agent--agent proximity / congestion edges.
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            d = float(np.linalg.norm(agent_pos[i] - agent_pos[j]))
            w = AGENT_AGENT_SCALE * gaussian_affinity(d, SIGMA_AGENT_AGENT)
            adjacency[i, j] = w
            adjacency[j, i] = w

    # Agent--landmark task-affinity edges.
    al_dist = np.linalg.norm(agent_pos[:, None, :] - landmark_pos[None, :, :], axis=-1)
    for i in range(n_agents):
        for l in range(n_landmarks):
            w = gaussian_affinity(float(al_dist[i, l]), SIGMA_AGENT_LANDMARK)
            j = n_agents + l
            adjacency[i, j] = w
            adjacency[j, i] = w

    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    norm_adjacency = inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    laplacian = np.eye(n_agents + n_landmarks) - norm_adjacency
    laplacian = 0.5 * (laplacian + laplacian.T)

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvalues = np.clip(eigenvalues, 0.0, 2.0)

    # HKS at agent nodes only. Squaring eigenvector entries removes sign ambiguity.
    squared_modes = eigenvectors[:n_agents, :] ** 2
    hks_agents = np.array(
        [
            np.sum(np.exp(-tau * eigenvalues)[None, :] * squared_modes, axis=1)
            for tau in HKS_TIMES
        ],
        dtype=np.float64,
    ).T

    return agent_pos, landmark_pos, al_dist, adjacency, laplacian, eigenvalues, eigenvectors, hks_agents


def optimal_one_to_one_assignment(al_dist: np.ndarray) -> Tuple[np.ndarray, float]:
    """Brute-force 3! assignment; avoids a SciPy dependency."""
    best_perm = None
    best_cost = float("inf")
    for perm in itertools.permutations(range(3)):
        cost = sum(float(al_dist[i, perm[i]]) for i in range(3))
        if cost < best_cost:
            best_cost = cost
            best_perm = np.array(perm, dtype=int)
    return best_perm, best_cost


def collision_pair_count(agent_pos: np.ndarray, size: float = 0.15) -> int:
    count = 0
    for i in range(3):
        for j in range(i + 1, 3):
            if np.linalg.norm(agent_pos[i] - agent_pos[j]) < 2.0 * size:
                count += 1
    return count


def capture_snapshot(env, episode: int, step: int, actions=None, reward=None) -> Snapshot:
    (
        agent_pos,
        landmark_pos,
        al_dist,
        adjacency,
        laplacian,
        eigenvalues,
        eigenvectors,
        hks_agents,
    ) = build_graph_from_world(env)

    min_dist_by_landmark = al_dist.min(axis=0)
    coverage = int(np.sum(min_dist_by_landmark < COVERAGE_THRESHOLD))
    nearest_landmark_by_agent = al_dist.argmin(axis=1)
    unique_nearest_targets = int(len(np.unique(nearest_landmark_by_agent)))
    assignment, assignment_cost = optimal_one_to_one_assignment(al_dist)

    return Snapshot(
        episode=episode,
        step=step,
        agent_pos=agent_pos,
        landmark_pos=landmark_pos,
        al_dist=al_dist,
        adjacency=adjacency,
        laplacian=laplacian,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        hks_agents=hks_agents,
        coverage=coverage,
        min_dist_sum=float(min_dist_by_landmark.sum()),
        nearest_landmark_by_agent=nearest_landmark_by_agent,
        unique_nearest_targets=unique_nearest_targets,
        assignment=assignment,
        assignment_cost=float(assignment_cost),
        collision_pairs=collision_pair_count(agent_pos),
        actions=None if actions is None else np.asarray(actions, dtype=float),
        reward=None if reward is None else float(reward),
    )


def model_path(project_root: Path) -> Path:
    base = project_root / "models" / ENV_ID / MODEL_NAME / f"run{RUN_NUM}"
    if CHECKPOINT_EPISODE is None:
        return base / "model.pt"
    return base / "incremental" / f"model_ep{CHECKPOINT_EPISODE}.pt"


def collect_traces(project_root: Path) -> List[Snapshot]:
    path = model_path(project_root)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{path}\n\n"
            "Check MODEL_NAME, RUN_NUM, and CHECKPOINT_EPISODE at the top of this script."
        )

    print(f"Loading policy: {path}")
    torch.manual_seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)

    maddpg = MADDPG.init_from_save(path)
    env = make_env(ENV_ID, discrete_action=maddpg.discrete_action)
    env.seed(EVAL_SEED)
    maddpg.prep_rollouts(device="cpu")

    all_records: List[Snapshot] = []

    try:
        for episode in range(N_EPISODES):
            obs = env.reset()
            all_records.append(capture_snapshot(env, episode, step=0))

            for step in range(1, EPISODE_LENGTH + 1):
                torch_obs = [
                    Variable(torch.tensor(obs[i], dtype=torch.float32).view(1, -1), requires_grad=False)
                    for i in range(maddpg.nagents)
                ]
                torch_actions = maddpg.step(torch_obs, explore=False)
                actions = [a.detach().cpu().numpy().flatten() for a in torch_actions]
                obs, rewards, _, _ = env.step(actions)
                action_ids = np.array([int(np.argmax(a)) for a in actions], dtype=int)
                all_records.append(
                    capture_snapshot(
                        env,
                        episode,
                        step=step,
                        actions=action_ids,
                        reward=float(rewards[0]),
                    )
                )

            final = all_records[-1]
            print(
                f"Episode {episode + 1:02d}/{N_EPISODES}: "
                f"final coverage={final.coverage}/3 | "
                f"final dist={final.min_dist_sum:.3f} | "
                f"unique nearest targets={final.unique_nearest_targets}/3"
            )
    finally:
        env.close()

    return all_records


def choose_crowded_snapshot(records: List[Snapshot]) -> Snapshot:
    """Geometric duplicate-target proxy, preferring non-initial policy states."""
    candidates = [r for r in records if r.step >= 3]
    if not candidates:
        candidates = records

    # Minimise number of unique nearest targets, then coverage; choose largest
    # remaining distance to make the crowded geometry visually evident.
    return sorted(
        candidates,
        key=lambda r: (r.unique_nearest_targets, r.coverage, -r.min_dist_sum, r.episode, r.step),
    )[0]


def choose_balanced_snapshot(records: List[Snapshot]) -> Snapshot:
    """One-to-one nearest-landmark geometry, then prefer genuine coverage."""
    candidates = [r for r in records if r.step >= 3 and r.unique_nearest_targets == 3]
    if not candidates:
        # Fallback: closest global matching structure available.
        candidates = [r for r in records if r.step >= 3] or records
        return sorted(candidates, key=lambda r: (r.assignment_cost, r.min_dist_sum, r.episode, r.step))[0]

    return sorted(
        candidates,
        key=lambda r: (-r.coverage, r.min_dist_sum, r.assignment_cost, r.episode, r.step),
    )[0]


def hks_pairwise_distances(hks_agents: np.ndarray) -> Dict[str, float]:
    values = {}
    for i in range(3):
        for j in range(i + 1, 3):
            values[f"A{i}-A{j}"] = float(np.linalg.norm(hks_agents[i] - hks_agents[j]))
    values["mean"] = float(np.mean(list(values.values())))
    return values


def snapshot_title(kind: str, s: Snapshot) -> str:
    nearest = ", ".join(f"A{i}→L{int(l)}" for i, l in enumerate(s.nearest_landmark_by_agent))
    assignment = ", ".join(f"A{i}→L{int(l)}" for i, l in enumerate(s.assignment))
    return (
        f"{kind}: episode {s.episode + 1}, step {s.step}\n"
        f"nearest-target proxy [{nearest}] | unique={s.unique_nearest_targets}/3 | "
        f"coverage={s.coverage}/3 | min-dist sum={s.min_dist_sum:.3f}\n"
        f"minimum-cost one-to-one assignment [{assignment}] | cost={s.assignment_cost:.3f}"
    )


def plot_spatial_graph(ax, s: Snapshot) -> None:
    # Edges first.
    for i in range(3):
        for j in range(i + 1, 3):
            w = s.adjacency[i, j]
            ax.plot(
                [s.agent_pos[i, 0], s.agent_pos[j, 0]],
                [s.agent_pos[i, 1], s.agent_pos[j, 1]],
                linestyle="--",
                linewidth=0.5 + 4.0 * w,
                alpha=0.75,
                color="gray",
                zorder=1,
            )
    for i in range(3):
        for l in range(3):
            w = s.adjacency[i, 3 + l]
            ax.plot(
                [s.agent_pos[i, 0], s.landmark_pos[l, 0]],
                [s.agent_pos[i, 1], s.landmark_pos[l, 1]],
                linewidth=0.35 + 3.8 * w,
                alpha=0.60,
                color="tab:blue",
                zorder=1,
            )

    agent_colors = ["tab:red", "tab:green", "tab:purple"]
    for i, pos in enumerate(s.agent_pos):
        ax.scatter(pos[0], pos[1], s=150, color=agent_colors[i], edgecolors="black", zorder=3)
        ax.annotate(f"A{i}", (pos[0], pos[1]), xytext=(6, 6), textcoords="offset points", fontsize=9)
    for l, pos in enumerate(s.landmark_pos):
        ax.scatter(pos[0], pos[1], s=200, marker="*", color="gold", edgecolors="black", zorder=3)
        ax.annotate(f"L{l}", (pos[0], pos[1]), xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Spatial task graph\nsolid blue: agent–landmark; dashed gray: agent–agent", fontsize=10)
    ax.grid(alpha=0.25)


def matrix_heatmap(ax, matrix: np.ndarray, labels: List[str], title: str, vmin=None, vmax=None, cmap="viridis") -> None:
    image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=10)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if matrix[i, j] > (matrix.max() + matrix.min()) / 2 else "black")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_spectrum(ax, s: Snapshot) -> None:
    x = np.arange(2, 7)
    ax.bar(x, s.eigenvalues[1:])
    ax.set_ylim(0, 2.05)
    ax.set_xticks(x)
    ax.set_xlabel("k")
    ax.set_ylabel(r"$\lambda_k$")
    ax.set_title(r"Nontrivial spectrum $\lambda_2,\ldots,\lambda_6$", fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    for k, value in zip(x, s.eigenvalues[1:]):
        ax.text(k, value + 0.04, f"{value:.2f}", ha="center", va="bottom", fontsize=8)


def plot_hks(ax, s: Snapshot) -> None:
    image = ax.imshow(s.hks_agents, aspect="auto", cmap="magma")
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(AGENT_NAMES)
    ax.set_xticks(np.arange(len(HKS_TIMES)))
    ax.set_xticklabels([f"τ={tau:g}" for tau in HKS_TIMES])
    ax.set_title("Agent-local HKS\n(sign-invariant spectral role)", fontsize=10)
    for i in range(3):
        for j in range(len(HKS_TIMES)):
            ax.text(j, i, f"{s.hks_agents[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if s.hks_agents[i, j] > s.hks_agents.mean() else "black")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def plot_snapshot_comparison(crowded: Snapshot, balanced: Snapshot, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(30, 10), constrained_layout=True)
    rows = [("Crowded / duplicate-target proxy", crowded), ("Balanced / one-to-one proxy", balanced)]

    for row, (kind, s) in enumerate(rows):
        plot_spatial_graph(axes[row, 0], s)
        matrix_heatmap(axes[row, 1], s.al_dist, LANDMARK_NAMES, "Raw agent–landmark distances", vmin=0, vmax=max(1.5, s.al_dist.max()), cmap="YlOrRd")
        axes[row, 1].set_yticks(np.arange(3))
        axes[row, 1].set_yticklabels(AGENT_NAMES)
        matrix_heatmap(axes[row, 2], s.adjacency, NODE_NAMES, r"Adjacency $A_t$", vmin=0, vmax=1.0)
        matrix_heatmap(axes[row, 3], s.laplacian, NODE_NAMES, r"Normalized Laplacian $L_t$", vmin=-1.0, vmax=1.0, cmap="coolwarm")
        plot_spectrum(axes[row, 4], s)
        plot_hks(axes[row, 5], s)
        axes[row, 0].text(
            0.02,
            -0.24,
            snapshot_title(kind, s),
            transform=axes[row, 0].transAxes,
            fontsize=9,
            va="top",
        )

    fig.suptitle(
        "Offline dynamic graph-spectral diagnostic for Vanilla MADDPG\n"
        f"Graph: 3 agents + 3 landmarks | σ_AL={SIGMA_AGENT_LANDMARK}, "
        f"σ_AA={SIGMA_AGENT_AGENT}, α_AA={AGENT_AGENT_SCALE}",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_episode_timeseries(records: List[Snapshot], episode: int, output_path: Path) -> None:
    episode_records = sorted([r for r in records if r.episode == episode], key=lambda r: r.step)
    t = np.array([r.step for r in episode_records])
    coverage = np.array([r.coverage for r in episode_records])
    min_dist = np.array([r.min_dist_sum for r in episode_records])
    eigs = np.vstack([r.eigenvalues[1:] for r in episode_records])
    hks = np.stack([r.hks_agents for r in episode_records], axis=0)

    pair_sep = []
    hks_step_change = [np.nan]
    for idx, r in enumerate(episode_records):
        pair_sep.append(hks_pairwise_distances(r.hks_agents)["mean"])
        if idx > 0:
            hks_step_change.append(float(np.mean(np.linalg.norm(hks[idx] - hks[idx - 1], axis=1))))

    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(t, coverage, marker="o", label="coverage count")
    ax.set_ylim(-0.1, 3.1)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel("Covered landmarks")
    ax.grid(alpha=0.25)
    twin = ax.twinx()
    twin.plot(t, min_dist, marker="s", linestyle="--", label="min-distance sum")
    twin.set_ylabel("Sum of closest distances")
    ax.set_title(f"Episode {episode + 1}: task quality and graph spectrum over time")

    ax = axes[1]
    for k in range(5):
        ax.plot(t, eigs[:, k], marker="o", markersize=3, label=rf"$\lambda_{k + 2}$")
    ax.set_ylim(-0.05, 2.05)
    ax.set_ylabel("Eigenvalue")
    ax.set_title(r"Evolution of nontrivial normalized-Laplacian eigenvalues")
    ax.grid(alpha=0.25)
    ax.legend(ncol=5, loc="upper center")

    ax = axes[2]
    ax.plot(t, pair_sep, marker="o", label="mean pairwise HKS separation")
    ax.plot(t, hks_step_change, marker="s", linestyle="--", label="mean one-step HKS change")
    ax.set_xlabel("Environment step")
    ax.set_ylabel("HKS distance")
    ax.set_title("Local spectral roles: distinctness versus step-to-step change")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def write_trace_csv(records: List[Snapshot], output_path: Path) -> None:
    fields = [
        "episode", "step", "coverage", "min_dist_sum", "unique_nearest_targets",
        "assignment_cost", "collision_pairs", "reward",
        "nearest_A0", "nearest_A1", "nearest_A2",
        "assign_A0", "assign_A1", "assign_A2",
    ]
    fields += [f"lambda_{k}" for k in range(1, 7)]
    fields += [f"HKS_A{i}_tau{str(tau).replace('.', '_')}" for i in range(3) for tau in HKS_TIMES]

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            row = {
                "episode": r.episode + 1,
                "step": r.step,
                "coverage": r.coverage,
                "min_dist_sum": r.min_dist_sum,
                "unique_nearest_targets": r.unique_nearest_targets,
                "assignment_cost": r.assignment_cost,
                "collision_pairs": r.collision_pairs,
                "reward": "" if r.reward is None else r.reward,
                "nearest_A0": int(r.nearest_landmark_by_agent[0]),
                "nearest_A1": int(r.nearest_landmark_by_agent[1]),
                "nearest_A2": int(r.nearest_landmark_by_agent[2]),
                "assign_A0": int(r.assignment[0]),
                "assign_A1": int(r.assignment[1]),
                "assign_A2": int(r.assignment[2]),
            }
            row.update({f"lambda_{k}": float(r.eigenvalues[k - 1]) for k in range(1, 7)})
            for i in range(3):
                for j, tau in enumerate(HKS_TIMES):
                    row[f"HKS_A{i}_tau{str(tau).replace('.', '_')}"] = float(r.hks_agents[i, j])
            writer.writerow(row)


def save_snapshot_npz(s: Snapshot, output_path: Path) -> None:
    np.savez_compressed(
        output_path,
        episode=np.array(s.episode + 1),
        step=np.array(s.step),
        agent_pos=s.agent_pos,
        landmark_pos=s.landmark_pos,
        al_dist=s.al_dist,
        adjacency=s.adjacency,
        laplacian=s.laplacian,
        eigenvalues=s.eigenvalues,
        eigenvectors=s.eigenvectors,
        hks_agents=s.hks_agents,
        coverage=np.array(s.coverage),
        min_dist_sum=np.array(s.min_dist_sum),
        nearest_landmark_by_agent=s.nearest_landmark_by_agent,
        assignment=s.assignment,
        assignment_cost=np.array(s.assignment_cost),
    )


def sign_invariance_test(s: Snapshot) -> float:
    """Confirm HKS is invariant to arbitrary eigenvector sign flips."""
    rng = np.random.default_rng(2026)
    signs = rng.choice(np.array([-1.0, 1.0]), size=s.eigenvectors.shape[1])
    flipped = s.eigenvectors * signs[None, :]
    original = np.array([
        np.sum(np.exp(-tau * s.eigenvalues)[None, :] * (s.eigenvectors[:3, :] ** 2), axis=1)
        for tau in HKS_TIMES
    ]).T
    recomputed = np.array([
        np.sum(np.exp(-tau * s.eigenvalues)[None, :] * (flipped[:3, :] ** 2), axis=1)
        for tau in HKS_TIMES
    ]).T
    return float(np.max(np.abs(original - recomputed)))


def summarize(records: List[Snapshot], crowded: Snapshot, balanced: Snapshot) -> Dict[str, object]:
    spectral_diff = float(np.linalg.norm(crowded.eigenvalues[1:] - balanced.eigenvalues[1:]))
    adjacency_diff = float(np.linalg.norm(crowded.adjacency - balanced.adjacency, ord="fro"))
    laplacian_diff = float(np.linalg.norm(crowded.laplacian - balanced.laplacian, ord="fro"))

    # HKS temporal stability descriptor across all within-episode transitions.
    step_changes = []
    for episode in range(N_EPISODES):
        e = sorted([r for r in records if r.episode == episode], key=lambda r: r.step)
        for prev, curr in zip(e[:-1], e[1:]):
            step_changes.append(float(np.mean(np.linalg.norm(curr.hks_agents - prev.hks_agents, axis=1))))

    return {
        "configuration": {
            "env_id": ENV_ID,
            "model_name": MODEL_NAME,
            "run_num": RUN_NUM,
            "checkpoint_episode": CHECKPOINT_EPISODE,
            "eval_seed": EVAL_SEED,
            "n_episodes": N_EPISODES,
            "episode_length": EPISODE_LENGTH,
            "sigma_agent_landmark": SIGMA_AGENT_LANDMARK,
            "sigma_agent_agent": SIGMA_AGENT_AGENT,
            "agent_agent_scale": AGENT_AGENT_SCALE,
            "hks_times": HKS_TIMES,
        },
        "selected_crowded_snapshot": {
            "episode": crowded.episode + 1,
            "step": crowded.step,
            "coverage": crowded.coverage,
            "min_dist_sum": crowded.min_dist_sum,
            "unique_nearest_targets": crowded.unique_nearest_targets,
            "nearest_landmark_by_agent": crowded.nearest_landmark_by_agent.tolist(),
            "assignment": crowded.assignment.tolist(),
            "assignment_cost": crowded.assignment_cost,
            "eigenvalues": crowded.eigenvalues.tolist(),
            "hks_agents": crowded.hks_agents.tolist(),
            "mean_pairwise_hks_separation": hks_pairwise_distances(crowded.hks_agents)["mean"],
        },
        "selected_balanced_snapshot": {
            "episode": balanced.episode + 1,
            "step": balanced.step,
            "coverage": balanced.coverage,
            "min_dist_sum": balanced.min_dist_sum,
            "unique_nearest_targets": balanced.unique_nearest_targets,
            "nearest_landmark_by_agent": balanced.nearest_landmark_by_agent.tolist(),
            "assignment": balanced.assignment.tolist(),
            "assignment_cost": balanced.assignment_cost,
            "eigenvalues": balanced.eigenvalues.tolist(),
            "hks_agents": balanced.hks_agents.tolist(),
            "mean_pairwise_hks_separation": hks_pairwise_distances(balanced.hks_agents)["mean"],
        },
        "contrast": {
            "frobenius_A_crowded_vs_balanced": adjacency_diff,
            "frobenius_L_crowded_vs_balanced": laplacian_diff,
            "l2_lambda2_to_lambda6_difference": spectral_diff,
            "hks_sign_flip_max_abs_error": sign_invariance_test(balanced),
            "mean_one_step_hks_change_all_episodes": float(np.mean(step_changes)),
            "median_one_step_hks_change_all_episodes": float(np.median(step_changes)),
        },
    }


def write_human_readable_summary(summary: Dict[str, object], output_path: Path) -> None:
    c = summary["selected_crowded_snapshot"]
    b = summary["selected_balanced_snapshot"]
    contrast = summary["contrast"]
    cfg = summary["configuration"]

    lines = [
        "Offline GSP diagnostic summary",
        "=" * 72,
        f"Model: {cfg['model_name']} | run{cfg['run_num']} | seed={cfg['eval_seed']}",
        f"Graph parameters: sigma_AL={cfg['sigma_agent_landmark']}, sigma_AA={cfg['sigma_agent_agent']}, alpha_AA={cfg['agent_agent_scale']}",
        "",
        "1) Crowded / duplicate-target proxy snapshot",
        f"   episode={c['episode']}, step={c['step']}",
        f"   nearest targets A0,A1,A2 = {c['nearest_landmark_by_agent']} | unique={c['unique_nearest_targets']}/3",
        f"   coverage={c['coverage']}/3 | min-distance sum={c['min_dist_sum']:.4f}",
        f"   optimal one-to-one assignment={c['assignment']} | cost={c['assignment_cost']:.4f}",
        f"   lambda2..lambda6 = {[round(x, 5) for x in c['eigenvalues'][1:]]}",
        f"   mean pairwise HKS separation = {c['mean_pairwise_hks_separation']:.6f}",
        "",
        "2) Balanced / one-to-one proxy snapshot",
        f"   episode={b['episode']}, step={b['step']}",
        f"   nearest targets A0,A1,A2 = {b['nearest_landmark_by_agent']} | unique={b['unique_nearest_targets']}/3",
        f"   coverage={b['coverage']}/3 | min-distance sum={b['min_dist_sum']:.4f}",
        f"   optimal one-to-one assignment={b['assignment']} | cost={b['assignment_cost']:.4f}",
        f"   lambda2..lambda6 = {[round(x, 5) for x in b['eigenvalues'][1:]]}",
        f"   mean pairwise HKS separation = {b['mean_pairwise_hks_separation']:.6f}",
        "",
        "3) Contrast checks",
        f"   ||A_crowded - A_balanced||_F = {contrast['frobenius_A_crowded_vs_balanced']:.6f}",
        f"   ||L_crowded - L_balanced||_F = {contrast['frobenius_L_crowded_vs_balanced']:.6f}",
        f"   ||lambda_crowded - lambda_balanced||_2 (lambda2..6) = {contrast['l2_lambda2_to_lambda6_difference']:.6f}",
        f"   HKS sign-flip max absolute error = {contrast['hks_sign_flip_max_abs_error']:.3e}",
        f"   mean one-step HKS change = {contrast['mean_one_step_hks_change_all_episodes']:.6f}",
        f"   median one-step HKS change = {contrast['median_one_step_hks_change_all_episodes']:.6f}",
        "",
        "Interpretation guardrails:",
        "- The spectrum is a compact structural descriptor, not a unique identifier of a graph.",
        "- A nonzero spectral difference only shows that this chosen graph encoding changes across geometries.",
        "- HKS is sign-invariant by construction; temporal robustness still needs to be judged from the saved trajectory plots.",
        "- This diagnostic does not change training, reward, action space, or observation inputs.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    records = collect_traces(project_root)
    crowded = choose_crowded_snapshot(records)
    balanced = choose_balanced_snapshot(records)

    print("\nSelected crowded proxy:")
    print(snapshot_title("Crowded", crowded))
    print("\nSelected balanced proxy:")
    print(snapshot_title("Balanced", balanced))

    trace_csv = output_dir / "gsp_trace_all_states.csv"
    comparison_png = output_dir / "01_crowded_vs_balanced_graph_spectrum.png"
    timeseries_png = output_dir / f"02_episode_{balanced.episode + 1:02d}_spectral_timeseries.png"
    crowded_npz = output_dir / "selected_crowded_snapshot.npz"
    balanced_npz = output_dir / "selected_balanced_snapshot.npz"
    summary_json = output_dir / "gsp_offline_summary.json"
    summary_txt = output_dir / "gsp_offline_summary.txt"

    write_trace_csv(records, trace_csv)
    plot_snapshot_comparison(crowded, balanced, comparison_png)
    plot_episode_timeseries(records, balanced.episode, timeseries_png)
    save_snapshot_npz(crowded, crowded_npz)
    save_snapshot_npz(balanced, balanced_npz)

    summary = summarize(records, crowded, balanced)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_human_readable_summary(summary, summary_txt)

    print("\n" + "=" * 72)
    print("Offline graph-spectral analysis complete.")
    print("=" * 72)
    print(f"Output directory: {output_dir}")
    print(f"1. Comparison figure: {comparison_png.name}")
    print(f"2. Episode time series: {timeseries_png.name}")
    print(f"3. Full trace CSV:     {trace_csv.name}")
    print(f"4. Human summary:      {summary_txt.name}")
    print(f"5. JSON summary:       {summary_json.name}")


if __name__ == "__main__":
    main()
