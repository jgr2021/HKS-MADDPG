"""Frozen deployment-mask transfer on the existing seeds 11--13 cohort."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.maddpg import MADDPG
from experiments.audit_active_gsp_compression import assignment_distance, geometry, radius_auc, sha256
from experiments.diagnose_active_gsp_counterfactual_masks import policy_digest, policy_sources
from experiments.diagnose_learned_active_gsp_masks import MASKS, install_mask, write_csv
from experiments.evaluate_multiscale_screening import METRICS
from utils.make_env import make_env

ARCHIVE = REPO_ROOT / "experiments/multiscale_spectral_screening_eval_20260711"
CONTROL = "learned_raw_potential_residual"
FULL = "learned_active_gsp_residual"
DEPLOYMENT_MASKS = ("plus_crowding_energy", "potentials_only")


def evaluate(checkpoint, mask_name, seed, episodes):
    model = MADDPG.init_from_save(str(checkpoint))
    model.prep_rollouts(device="cpu")
    before = policy_digest(model.policies)
    install_mask(model, MASKS[mask_name])
    env = make_env("simple_spread", discrete_action=True)
    rows = []
    try:
        for episode in range(episodes):
            test_seed = 4_000_000 + seed * 10000 + episode
            torch.manual_seed(test_seed)
            np.random.seed(test_seed)
            env.seed(test_seed)
            obs = env.reset()
            reward_sum, collision_steps, maximum_coverage = 0.0, 0, 0
            minimum_separation = math.inf
            for _ in range(25):
                local_inputs = [torch.as_tensor(item, dtype=torch.float32).view(1, 18) for item in obs]
                with torch.no_grad():
                    actions = [action.cpu().numpy().ravel() for action in model.step(local_inputs, explore=False)]
                obs, rewards, dones, _ = env.step(actions)
                reward_sum += float(np.mean(rewards))
                distances, separation, collision = geometry(env)
                nearest = distances.min(axis=0)
                maximum_coverage = max(maximum_coverage, int((nearest < 0.10).sum()))
                collision_steps += collision
                minimum_separation = min(minimum_separation, separation)
                if all(dones):
                    break
            rows.append({"method": mask_name, "train_seed": seed, "eval_episode": episode,
                         "test_seed": test_seed, "return": reward_sum,
                         "hungarian_assignment_distance": assignment_distance(distances),
                         "coverage_radius_auc": radius_auc(nearest),
                         "collision_step_rate": collision_steps / 25.0,
                         "final_coverage_r010": int((nearest < 0.10).sum()),
                         "maximum_coverage_r010": maximum_coverage,
                         "minimum_agent_separation": minimum_separation})
            if (episode + 1) % 100 == 0:
                print(f"seed={seed} mask={mask_name} episodes={episode + 1}/{episodes}", flush=True)
    finally:
        env.close()
    if before != policy_digest(model.policies):
        raise RuntimeError("Frozen actor weights or buffers changed")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    sources = [source for source in policy_sources() if source["seed"] in (11, 12, 13)]
    for source in sources:
        source["sha256_before"] = sha256(Path(source["checkpoint"]))
    archive_path = ARCHIVE / "episode_metrics.csv"
    archive_hash = sha256(archive_path)
    with archive_path.open(newline="", encoding="utf-8") as handle:
        archive = list(csv.DictReader(handle))
    reused = []
    for row in archive:
        if row["method"] not in (CONTROL, FULL):
            continue
        reused.append({key: row[key] if key == "method" else (
            int(row[key]) if key in ("train_seed", "eval_episode", "test_seed") else float(row[key]))
                       for key in row})
    for method in (CONTROL, FULL):
        for seed in (11, 12, 13):
            subset = [row for row in reused if row["method"] == method and row["train_seed"] == seed]
            if len(subset) != 500 or sorted(row["test_seed"] for row in subset) != list(
                    range(4_000_000 + seed * 10000, 4_000_000 + seed * 10000 + 500)):
                raise ValueError("Archived reference lacks the exact paired 500-episode schedule")
    protocol = {"status": "started", "sources": sources, "archive": str(archive_path),
                "archive_sha256_before": archive_hash, "masks": DEPLOYMENT_MASKS,
                "reused_methods": [CONTROL, FULL], "new_training_steps": 0,
                "episodes": 500, "horizon": 25, "eval_seed_base": 4_000_000,
                "seed_formula": "4000000 + train_seed*10000 + episode",
                "selection": "fixed masks from the historical seeds 1-3 study; no threshold search",
                "interpretation": "exploratory transfer, following same-state results; no retrained ablation",
                "decision_rule": "Do not promote unless return, Hungarian and AUC improve in all three seeds versus the trained control, with no mean collision increase or mean separation decrease. Passing would remain exploratory.",
                "command": subprocess.list2cmdline([sys.executable, *sys.argv])}
    protocol_path = args.output_dir / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    replays = []
    for source in sources:
        seed = source["seed"]
        replay = evaluate(source["checkpoint"], "full", seed, 1)[0]
        expected = next(row for row in reused if row["method"] == FULL
                        and row["train_seed"] == seed and row["eval_episode"] == 0)
        for metric in METRICS:
            np.testing.assert_allclose(replay[metric], expected[metric], rtol=1e-6, atol=1e-7)
        replays.append(replay)
    write_csv(args.output_dir / "baseline_replay_validation.csv", replays)

    rows = list(reused)
    for source in sources:
        for mask in DEPLOYMENT_MASKS:
            new_rows = evaluate(source["checkpoint"], mask, source["seed"], 500)
            write_csv(args.output_dir / f"seed_{source['seed']}_{mask}_episodes.csv", new_rows)
            rows.extend(new_rows)
        source["sha256_after"] = sha256(Path(source["checkpoint"]))
        if source["sha256_after"] != source["sha256_before"]:
            raise RuntimeError("Source checkpoint changed")
    write_csv(args.output_dir / "combined_episodes.csv", rows)
    summaries = []
    for method in (CONTROL, FULL, *DEPLOYMENT_MASKS):
        for seed in (11, 12, 13):
            selected = [row for row in rows if row["method"] == method and row["train_seed"] == seed]
            summaries.append({"method": method, "train_seed": seed, "episodes": len(selected),
                              **{metric: float(np.mean([row[metric] for row in selected])) for metric in METRICS}})
    write_csv(args.output_dir / "seed_metrics.csv", summaries)
    lookup = {(row["method"], row["train_seed"]): row for row in summaries}
    differences = []
    for mask in DEPLOYMENT_MASKS:
        for reference in (FULL, CONTROL):
            for metric in METRICS:
                values = np.asarray([lookup[(mask, seed)][metric] - lookup[(reference, seed)][metric]
                                     for seed in (11, 12, 13)])
                half = 4.3026527297 * values.std(ddof=1) / math.sqrt(3)
                differences.append({"mask": mask, "reference": reference, "metric": metric,
                                    "mean_delta": float(values.mean()), "ci95_low": float(values.mean() - half),
                                    "ci95_high": float(values.mean() + half),
                                    "per_seed_deltas": ";".join(f"{value:.9g}" for value in values)})
    write_csv(args.output_dir / "paired_seed_deltas.csv", differences)
    protocol.update({"status": "completed", "sources": sources,
                     "archive_sha256_after": sha256(archive_path),
                     "baseline_replay_validation": "three exact matching first episodes",
                     "new_evaluation_episodes": 3003, "reused_evaluation_episodes": len(reused),
                     "elapsed_seconds": time.perf_counter() - start})
    if protocol["archive_sha256_after"] != archive_hash:
        raise RuntimeError("Archived reference changed")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(args.output_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
