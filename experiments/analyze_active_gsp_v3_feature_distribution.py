import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.active_gsp_v3_features import compute_active_gsp_v3_from_local_obs_batch
from utils.make_env import make_env


FEATURE_NAMES = [
    "delta_coverage_potential",
    "delta_crowding_potential",
    "delta_coverage_energy",
    "delta_crowding_energy",
]


def random_onehot(action_space, rng):
    action_index = int(rng.randint(action_space.n))
    action = np.zeros(action_space.n, dtype=np.float32)
    action[action_index] = 1.0
    return action


def collect_features(samples, episode_length, seed):
    rng = np.random.RandomState(seed)
    env = make_env("simple_spread", discrete_action=True)
    env.seed(seed)
    batches = []
    collected = 0
    try:
        while collected < samples:
            obs = env.reset()
            for _ in range(episode_length):
                raw = np.asarray(obs, dtype=np.float32)
                features = compute_active_gsp_v3_from_local_obs_batch(raw)
                batches.append(features)
                collected += raw.shape[0]
                if collected >= samples:
                    break
                actions = [
                    random_onehot(env.action_space[agent_index], rng)
                    for agent_index in range(env.n)
                ]
                obs, _, dones, _ = env.step(actions)
                if all(dones):
                    break
    finally:
        env.close()
    return np.concatenate(batches, axis=0)[:samples]


def summarize(features):
    # Drop no-op rows because they are exactly zero by construction.
    non_noop = features[:, 1:, :].reshape(-1, features.shape[2])
    rows = []
    for index, name in enumerate(FEATURE_NAMES):
        values = non_noop[:, index]
        abs_values = np.abs(values)
        p95_abs = float(np.percentile(abs_values, 95))
        suggested_scale = 1.0 / max(p95_abs, 1e-6)
        rows.append(
            {
                "feature": name,
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "p01": float(np.percentile(values, 1)),
                "p05": float(np.percentile(values, 5)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "max_abs": float(np.max(abs_values)),
                "p95_abs": p95_abs,
                "suggested_scale_for_p95_abs_1": suggested_scale,
            }
        )
    return rows


def write_outputs(result_dir, args, rows):
    csv_path = result_dir / "feature_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    readme_path = result_dir / "README.md"
    lines = [
        "# Active GSP v3 Feature Distribution",
        "",
        f"Samples: `{args.samples}` local observations",
        f"Episode length: `{args.episode_length}`",
        f"Seed: `{args.seed}`",
        "",
        "Statistics are computed over non-noop action deltas only.",
        "",
        "| feature | mean | std | p95_abs | suggested_scale | max_abs |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['feature']} | {row['mean']:.6f} | {row['std']:.6f} | "
            f"{row['p95_abs']:.6f} | "
            f"{row['suggested_scale_for_p95_abs_1']:.4f} | "
            f"{row['max_abs']:.6f} |"
        )
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, readme_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--episode_length", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7310)
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    result_dir = Path("experiments") / f"active_gsp_v3_feature_stats_{stamp}"
    result_dir.mkdir(parents=True, exist_ok=False)

    features = collect_features(args.samples, args.episode_length, args.seed)
    rows = summarize(features)
    csv_path, readme_path = write_outputs(result_dir, args, rows)
    print(f"Saved feature stats: {csv_path}", flush=True)
    print(f"Saved README: {readme_path}", flush=True)


if __name__ == "__main__":
    main()
