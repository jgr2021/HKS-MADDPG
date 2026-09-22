import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.gsp_features import (
    HKS_DIM,
    RAW_OBS_DIM,
    TOPOLOGY_3NODE_AA_HKS,
    TOPOLOGY_6NODE_AAL_HKS,
    TOPOLOGY_6NODE_AL_HKS,
    build_topology_hks_graph_from_positions,
    descriptor_for_mode,
    reconstruct_geometry_from_raw_observation,
)
from utils.make_env import make_env


MODES = [
    TOPOLOGY_3NODE_AA_HKS,
    TOPOLOGY_6NODE_AL_HKS,
    TOPOLOGY_6NODE_AAL_HKS,
]


def random_discrete_action(rng, n_actions=5):
    action = np.zeros(n_actions, dtype=np.float32)
    action[int(rng.integers(0, n_actions))] = 1.0
    return action


def assert_graph_constraints(raw_obs, agent_index, mode):
    agent_positions, landmark_positions = reconstruct_geometry_from_raw_observation(
        raw_obs,
        agent_index,
    )
    adjacency, laplacian, eigenvalues, _ = build_topology_hks_graph_from_positions(
        agent_positions,
        landmark_positions,
        mode=mode,
    )
    if not np.allclose(adjacency, adjacency.T, rtol=0.0, atol=1e-10):
        raise RuntimeError(f"{mode}: edge weights are not symmetric.")
    if not np.allclose(np.diag(adjacency), 0.0, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"{mode}: edge-weight diagonal is not zero.")
    if not np.allclose(laplacian, laplacian.T, rtol=0.0, atol=1e-10):
        raise RuntimeError(f"{mode}: Laplacian is not symmetric.")
    if not np.all(np.isfinite(eigenvalues)):
        raise RuntimeError(f"{mode}: eigenvalues are not finite.")
    if np.min(eigenvalues) < -1e-8:
        raise RuntimeError(f"{mode}: eigenvalues are negative: {eigenvalues}.")

    if mode == TOPOLOGY_3NODE_AA_HKS and adjacency.shape != (3, 3):
        raise RuntimeError(f"{mode}: expected 3-node agent-agent graph.")
    if mode == TOPOLOGY_6NODE_AL_HKS:
        if not np.allclose(adjacency[:3, :3], 0.0):
            raise RuntimeError(f"{mode}: contains agent-agent edges.")
        if not np.allclose(adjacency[3:, 3:], 0.0):
            raise RuntimeError(f"{mode}: contains landmark-landmark edges.")
    if mode == TOPOLOGY_6NODE_AAL_HKS:
        if np.abs(adjacency[:3, :3]).sum() <= 0.0:
            raise RuntimeError(f"{mode}: missing agent-agent edges.")
        if not np.allclose(adjacency[3:, 3:], 0.0):
            raise RuntimeError(f"{mode}: contains landmark-landmark edges.")


def summarize_features(features):
    percentiles = np.percentile(features, [1, 5, 25, 50, 75, 95, 99], axis=0)
    rows = []
    for index in range(features.shape[1]):
        rows.append(
            {
                "feature": f"hks_tau_{index}",
                "mean": float(features[:, index].mean()),
                "std": float(features[:, index].std(ddof=0)),
                "min": float(features[:, index].min()),
                "p01": float(percentiles[0, index]),
                "p05": float(percentiles[1, index]),
                "p25": float(percentiles[2, index]),
                "p50": float(percentiles[3, index]),
                "p75": float(percentiles[4, index]),
                "p95": float(percentiles[5, index]),
                "p99": float(percentiles[6, index]),
                "max": float(features[:, index].max()),
            }
        )
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--output-dir", default="experiments/passive_gsp_topology_pilot/diagnostics")
    parser.add_argument("--near-constant-std", type=float, default=1e-6)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_mode_features = {mode: [] for mode in MODES}
    env = make_env("simple_spread", discrete_action=True)
    try:
        env.seed(args.seed)
        observations = env.reset()
        for step in range(args.steps):
            for agent_index, raw_obs in enumerate(observations):
                raw = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
                if raw.shape != (RAW_OBS_DIM,):
                    raise RuntimeError(f"Expected raw 18D observation, got {raw.shape}.")
                for mode in MODES:
                    assert_graph_constraints(raw, agent_index, mode)
                    descriptor = descriptor_for_mode(raw, agent_index, mode=mode)
                    if descriptor.shape != (HKS_DIM,):
                        raise RuntimeError(f"{mode}: expected 3 HKS values, got {descriptor.shape}.")
                    if not np.all(np.isfinite(descriptor)):
                        raise RuntimeError(f"{mode}: feature vector has NaN/Inf.")
                    all_mode_features[mode].append(descriptor)

            actions = [random_discrete_action(rng) for _ in range(3)]
            observations, _, dones, _ = env.step(actions)
            if all(dones):
                observations = env.reset()
    finally:
        env.close()

    manifest = {
        "steps": args.steps,
        "seed": args.seed,
        "raw_obs_dim": RAW_OBS_DIM,
        "descriptor_dim": HKS_DIM,
        "modes": MODES,
        "sampled_feature_vectors_per_mode": args.steps * 3,
    }
    (output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    for mode, values in all_mode_features.items():
        features = np.asarray(values, dtype=np.float64)
        if features.shape != (args.steps * 3, HKS_DIM):
            raise RuntimeError(f"{mode}: unexpected feature matrix shape {features.shape}.")
        if not np.all(np.isfinite(features)):
            raise RuntimeError(f"{mode}: feature matrix has NaN/Inf.")
        std = features.std(axis=0)
        if np.any(std < args.near_constant_std):
            raise RuntimeError(f"{mode}: near-constant feature detected, std={std}.")

        write_csv(output_dir / f"{mode}_stats.csv", summarize_features(features))
        corr = np.corrcoef(features, rowvar=False)
        np.savetxt(
            output_dir / f"{mode}_correlation.csv",
            corr,
            delimiter=",",
            header="hks_tau_0,hks_tau_1,hks_tau_2",
            comments="",
        )
        np.save(output_dir / f"{mode}_features.npy", features.astype(np.float32))

    print(f"Diagnostics passed: {output_dir}")


if __name__ == "__main__":
    main()
