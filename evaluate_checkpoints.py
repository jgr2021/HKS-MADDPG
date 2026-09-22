# evaluate_checkpoints.py
# Batch greedy evaluation for selected MADDPG checkpoints.
# Run this file directly in PyCharm.

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from utils.make_env import make_env


# ============================================================
# 1. Experiment configuration
# ============================================================

ENV_NAME = "simple_spread_gsp_v1"
DISCRETE_ACTION = True

# This must match the episode length used during training.
# Standard MPE simple_spread usually uses 25.
MAX_STEPS = 25

# Fixed held-out evaluation seed:
# every checkpoint sees exactly the same 20 initial situations.
EVAL_SEED = 2040
N_EVAL_EPISODES = 20

COVERAGE_THRESHOLD = 0.10

CHECKPOINT_DIR = Path(
    "models/simple_spread_gsp_v1/"
    "spread_gsp_v1_baseline_fast4_seed1/"
    "run1/incremental"
)

CHECKPOINT_EPISODES = [10001, 25001, 50001, 75001, 97501]


# ============================================================
# 2. Environment / seed helpers
# ============================================================

def set_seed(seed: int) -> None:
    """Set all relevant random seeds for fair checkpoint comparison."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_env():
    """Build one non-vectorized evaluation environment."""
    return make_env(
        ENV_NAME,
        discrete_action=DISCRETE_ACTION,
    )


def reset_env_with_seed(env, seed: int):
    """
    Supports both old Gym/MPE reset() and Gymnasium-style reset(seed=...).
    The np.random seed is especially important for legacy MPE scenarios.
    """
    set_seed(seed)

    try:
        reset_output = env.reset(seed=seed)
    except TypeError:
        reset_output = env.reset()

    # Gymnasium reset returns (obs, info); old Gym returns obs.
    if isinstance(reset_output, tuple):
        return reset_output[0]

    return reset_output


# ============================================================
# 3. Metrics
# ============================================================

def get_coverage_metrics(env, threshold: float = COVERAGE_THRESHOLD) -> dict:
    """
    Count how many landmarks are covered by at least one agent.

    final_covered:
        Number of landmarks with nearest-agent distance < threshold.

    mean_nearest_landmark_distance:
        For each landmark, find its nearest agent; then average over landmarks.
    """
    agents = env.world.agents
    landmarks = env.world.landmarks

    distances = np.asarray(
        [
            [
                np.linalg.norm(agent.state.p_pos - landmark.state.p_pos)
                for landmark in landmarks
            ]
            for agent in agents
        ],
        dtype=np.float32,
    )

    # For each landmark, take the nearest agent distance.
    nearest_agent_distances = distances.min(axis=0)

    covered = nearest_agent_distances < threshold

    return {
        "final_covered": int(covered.sum()),
        "mean_nearest_landmark_distance": float(nearest_agent_distances.mean()),
        "distances": distances,
    }


# ============================================================
# 4. Deterministic MADDPG rollout
# ============================================================

def get_greedy_actions(maddpg: MADDPG, observations):
    """
    Run deterministic / greedy actions:
    explore=False means no exploration noise and no random action sampling.
    """
    torch_obs = [
        torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        for obs in observations
    ]

    with torch.no_grad():
        torch_actions = maddpg.step(torch_obs, explore=False)

    actions = [
        action.detach().cpu().numpy().squeeze(0)
        for action in torch_actions
    ]

    return actions


def evaluate_checkpoint(checkpoint_path: Path) -> dict:
    """Evaluate one checkpoint over the fixed validation episodes."""
    print("\n" + "=" * 72)
    print(f"Evaluating: {checkpoint_path.name}")
    print("=" * 72)

    env = build_env()

    maddpg = MADDPG.init_from_save(str(checkpoint_path))
    maddpg.prep_rollouts(device="cpu")

    final_coverages = []
    max_coverages = []
    final_distances = []
    episode_returns = []

    n_landmarks = len(env.world.landmarks)

    try:
        for episode_idx in range(N_EVAL_EPISODES):
            # Same episode seed for each checkpoint.
            episode_seed = EVAL_SEED + episode_idx
            observations = reset_env_with_seed(env, episode_seed)

            max_covered_this_episode = 0
            episode_return = 0.0

            for _ in range(MAX_STEPS):
                actions = get_greedy_actions(maddpg, observations)

                step_output = env.step(actions)

                # Compatible with both Gym and Gymnasium step APIs.
                if len(step_output) == 5:
                    next_observations, rewards, terminated, truncated, _ = step_output
                    done = np.logical_or(terminated, truncated)
                else:
                    next_observations, rewards, done, _ = step_output

                # In simple_spread, rewards are usually shared across agents.
                # Taking the mean avoids tripling the same global reward.
                episode_return += float(np.mean(rewards))

                current_metrics = get_coverage_metrics(env)
                max_covered_this_episode = max(
                    max_covered_this_episode,
                    current_metrics["final_covered"],
                )

                observations = next_observations

                if np.all(done):
                    break

            final_metrics = get_coverage_metrics(env)

            final_coverages.append(final_metrics["final_covered"])
            max_coverages.append(max_covered_this_episode)
            final_distances.append(
                final_metrics["mean_nearest_landmark_distance"]
            )
            episode_returns.append(episode_return)

            print(
                f"Episode {episode_idx + 1:02d}/{N_EVAL_EPISODES} | "
                f"final={final_metrics['final_covered']}/{n_landmarks} | "
                f"max={max_covered_this_episode}/{n_landmarks} | "
                f"dist={final_metrics['mean_nearest_landmark_distance']:.3f} | "
                f"return={episode_return:.2f}"
            )

    finally:
        env.close()

    final_coverages = np.asarray(final_coverages, dtype=np.float32)
    max_coverages = np.asarray(max_coverages, dtype=np.float32)
    final_distances = np.asarray(final_distances, dtype=np.float32)
    episode_returns = np.asarray(episode_returns, dtype=np.float32)

    result = {
        "checkpoint": checkpoint_path.name,
        "final3": int(np.sum(final_coverages == n_landmarks)),
        "mean_final": float(final_coverages.mean()),
        "max3": int(np.sum(max_coverages == n_landmarks)),
        "mean_max": float(max_coverages.mean()),
        "dist": float(final_distances.mean()),
        "return": float(episode_returns.mean()),
        "n_landmarks": n_landmarks,
    }

    print("\nSummary")
    print(
        f"{checkpoint_path.name} | "
        f"final3={result['final3']}/{N_EVAL_EPISODES} | "
        f"mean_final={result['mean_final']:.2f}/{n_landmarks} | "
        f"max3={result['max3']}/{N_EVAL_EPISODES} | "
        f"mean_max={result['mean_max']:.2f}/{n_landmarks} | "
        f"dist={result['dist']:.3f} | "
        f"return={result['return']:.2f}"
    )

    return result


# ============================================================
# 5. Batch evaluation and ranking
# ============================================================

def save_csv(results: list[dict], output_path: Path) -> None:
    fieldnames = [
        "checkpoint",
        "final3",
        "mean_final",
        "max3",
        "mean_max",
        "dist",
        "return",
        "n_landmarks",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    print("=" * 72)
    print(
        f"Batch greedy evaluation | "
        f"validation seed={EVAL_SEED} | "
        f"episodes={N_EVAL_EPISODES}"
    )
    print("=" * 72)

    missing = []
    checkpoint_paths = []

    # 1. Evaluate selected incremental checkpoints.
    for episode in CHECKPOINT_EPISODES:
        checkpoint_path = CHECKPOINT_DIR / f"model_ep{episode}.pt"

        if checkpoint_path.exists():
            checkpoint_paths.append(checkpoint_path)
        else:
            missing.append(checkpoint_path.name)

    # 2. Also evaluate the final model saved at the end of training.
    # CHECKPOINT_DIR is:
    # .../run1/incremental
    # so its parent is:
    # .../run1
    final_model_path = CHECKPOINT_DIR.parent / "model.pt"

    if final_model_path.exists():
        checkpoint_paths.append(final_model_path)
    else:
        missing.append(str(final_model_path))

    if missing:
        raise FileNotFoundError(
            "These checkpoints were not found:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )

    results = [
        evaluate_checkpoint(checkpoint_path)
        for checkpoint_path in checkpoint_paths
    ]

    # Priority:
    # 1) ending with 3/3 coverage;
    # 2) higher average final coverage;
    # 3) ever reaching 3/3 during an episode;
    # 4) higher mean maximum coverage;
    # 5) lower landmark distance;
    # 6) higher return.
    ranked_results = sorted(
        results,
        key=lambda x: (
            x["final3"],
            x["mean_final"],
            x["max3"],
            x["mean_max"],
            -x["dist"],
            x["return"],
        ),
        reverse=True,
    )

    n_landmarks = ranked_results[0]["n_landmarks"]

    print("\n" + "=" * 72)
    print(f"Checkpoint ranking on validation seed {EVAL_SEED}")
    print("=" * 72)

    for rank, result in enumerate(ranked_results, start=1):
        print(
            f"#{rank:02d} | "
            f"{result['checkpoint']:<16} | "
            f"final3={result['final3']:2d}/{N_EVAL_EPISODES} | "
            f"mean_final={result['mean_final']:.2f}/{n_landmarks} | "
            f"max3={result['max3']:2d}/{N_EVAL_EPISODES} | "
            f"mean_max={result['mean_max']:.2f}/{n_landmarks} | "
            f"dist={result['dist']:.3f} | "
            f"return={result['return']:.2f}"
        )

    output_csv = CHECKPOINT_DIR.parent / f"checkpoint_eval_seed{EVAL_SEED}.csv"
    save_csv(ranked_results, output_csv)

    print("\n" + "=" * 72)
    print(f"CSV saved to: {output_csv}")
    print("Evaluation finished.")
    print("=" * 72)


if __name__ == "__main__":
    main()