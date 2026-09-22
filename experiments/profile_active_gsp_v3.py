import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.active_gsp_v3_features import (
    active_gsp_action_features,
    compute_active_gsp_v3_from_local_obs_batch,
)
from utils.make_env import make_env


def random_one_hot(rng, n_actions=5):
    action = np.zeros(n_actions, dtype=np.float32)
    action[int(rng.randint(n_actions))] = 1.0
    return action


AGENT_INDICES = np.asarray([0, 1, 2], dtype=np.int64)


def run_rollout_profile(episodes, episode_length, seed, feature_mode):
    rng = np.random.RandomState(seed)
    env = make_env("simple_spread", discrete_action=True)
    env.seed(seed)

    env_steps = 0
    feature_calls = 0
    started = time.perf_counter()
    try:
        for episode in range(episodes):
            np.random.seed(seed + episode)
            obs = env.reset()
            for _ in range(episode_length):
                if feature_mode == "reference":
                    for agent_index in range(len(obs)):
                        active_gsp_action_features(obs[agent_index], agent_index)
                        feature_calls += 1
                elif feature_mode == "fast":
                    compute_active_gsp_v3_from_local_obs_batch(
                        np.asarray(obs, dtype=np.float32),
                    )
                    feature_calls += len(obs)
                actions = [random_one_hot(rng) for _ in range(len(obs))]
                obs, _, _, _ = env.step(actions)
                env_steps += 1
    finally:
        env.close()

    elapsed = time.perf_counter() - started
    return {
        "episodes": episodes,
        "episode_length": episode_length,
        "seed": seed,
        "feature_mode": feature_mode,
        "env_steps": env_steps,
        "feature_calls": feature_calls,
        "elapsed_sec": elapsed,
        "env_steps_per_sec": env_steps / elapsed,
        "feature_calls_per_sec": feature_calls / elapsed if feature_calls else 0.0,
    }


def collect_latency_batch(count, seed):
    rng = np.random.RandomState(seed)
    env = make_env("simple_spread", discrete_action=True)
    env.seed(seed)
    observations = []
    indices = []
    try:
        obs = env.reset()
        while len(observations) < count:
            for agent_index, local_obs in enumerate(obs):
                observations.append(np.asarray(local_obs, dtype=np.float32))
                indices.append(agent_index)
                if len(observations) >= count:
                    break
            actions = [random_one_hot(rng) for _ in range(3)]
            obs, _, _, _ = env.step(actions)
    finally:
        env.close()
    return np.asarray(observations, dtype=np.float32), np.asarray(indices, dtype=np.int64)


def standalone_latency(count, seed):
    observations, indices = collect_latency_batch(count, seed)

    started = time.perf_counter()
    for local_obs, agent_index in zip(observations, indices):
        active_gsp_action_features(local_obs, int(agent_index))
    reference_elapsed = time.perf_counter() - started

    started = time.perf_counter()
    compute_active_gsp_v3_from_local_obs_batch(observations)
    fast_elapsed = time.perf_counter() - started

    return {
        "latency_observations": int(count),
        "reference_total_sec": reference_elapsed,
        "fast_total_sec": fast_elapsed,
        "reference_ms_per_obs": reference_elapsed * 1000.0 / count,
        "fast_ms_per_obs": fast_elapsed * 1000.0 / count,
        "standalone_speedup": reference_elapsed / fast_elapsed if fast_elapsed else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--episode_length", type=int, default=25)
    parser.add_argument("--seed", type=int, default=9100)
    parser.add_argument("--latency_count", type=int, default=4096)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("experiments") / f"active_gsp_v3_profile_fastpath_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    rows = [
        run_rollout_profile(args.episodes, args.episode_length, args.seed, "none"),
        run_rollout_profile(args.episodes, args.episode_length, args.seed, "reference"),
        run_rollout_profile(args.episodes, args.episode_length, args.seed, "fast"),
    ]
    raw_rate = rows[0]["env_steps_per_sec"]
    reference_rate = rows[1]["env_steps_per_sec"]
    fast_rate = rows[2]["env_steps_per_sec"]
    reference_ratio = reference_rate / raw_rate if raw_rate else 0.0
    fast_ratio = fast_rate / raw_rate if raw_rate else 0.0
    latency = standalone_latency(args.latency_count, args.seed + 17)

    csv_path = output_dir / "profile.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    latency_path = output_dir / "standalone_latency.csv"
    with latency_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(latency.keys()))
        writer.writeheader()
        writer.writerow(latency)

    readme = output_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Active GSP v3 Rollout Profiling",
                "",
                f"Episodes: `{args.episodes}`",
                f"Episode length: `{args.episode_length}`",
                f"Seed: `{args.seed}`",
                "",
                f"Raw env steps/sec: `{raw_rate:.4f}`",
                f"Raw + reference v3 steps/sec: `{reference_rate:.4f}`",
                f"Raw + fast-path v3 steps/sec: `{fast_rate:.4f}`",
                f"Reference/raw ratio: `{reference_ratio:.4f}`",
                f"Fast-path/raw ratio: `{fast_ratio:.4f}`",
                "",
                f"Reference standalone latency: `{latency['reference_ms_per_obs']:.6f}` ms/local obs",
                f"Fast-path standalone latency: `{latency['fast_ms_per_obs']:.6f}` ms/local obs",
                f"Standalone feature speedup: `{latency['standalone_speedup']:.4f}x`",
                "",
                "Gate: fast path should be at least 80% of raw ideally, 70% minimally.",
                f"Minimum gate passed: `{fast_ratio >= 0.70}`",
                f"Ideal gate passed: `{fast_ratio >= 0.80}`",
                "",
                "This profile does not train and does not save model checkpoints.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved {csv_path}")
    print(f"Saved {latency_path}")
    print(f"Saved {readme}")
    print(f"raw_steps_per_sec={raw_rate:.4f}")
    print(f"reference_steps_per_sec={reference_rate:.4f}")
    print(f"fast_steps_per_sec={fast_rate:.4f}")
    print(f"reference_ratio={reference_ratio:.4f}")
    print(f"fast_ratio={fast_ratio:.4f}")
    print(f"reference_ms_per_obs={latency['reference_ms_per_obs']:.6f}")
    print(f"fast_ms_per_obs={latency['fast_ms_per_obs']:.6f}")
    print(f"minimum_gate_passed={fast_ratio >= 0.70}")
    print(f"ideal_gate_passed={fast_ratio >= 0.80}")


if __name__ == "__main__":
    main()
