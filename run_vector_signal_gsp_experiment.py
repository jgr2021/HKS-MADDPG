"""Locked five-method screen for the three-agent vector-signal branch.

This delegates the unchanged training loop, replay, critic, exploration, and
checkpoint machinery to the current actor-only protocol runner.  Only method
registration and deterministic evaluation metrics are extended here.
"""

from __future__ import annotations

import itertools
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

from experiments import run_passive_gsp_topology_pilot as protocol
from algorithms.maddpg import MADDPG
from utils.make_env import make_env


METHODS = {
    "raw_maddpg": {"env_id": "simple_spread", "actor_model": "mlp", "short": "raw", "actor_input_dim": 18},
    "action_aware_geometric_control": {"env_id": "simple_spread", "actor_model": "learned_raw_potential_residual", "short": "geom", "actor_input_dim": 18},
    "active_gsp_6node": {"env_id": "simple_spread", "actor_model": "learned_active_gsp_residual", "short": "gsp6", "actor_input_dim": 18},
    "vector_signal_geometric_control": {"env_id": "simple_spread", "actor_model": "vector_signal_geom", "short": "vgeom", "actor_input_dim": 18},
    "vector_signal_gsp_3agent": {"env_id": "simple_spread", "actor_model": "vector_signal_gsp", "short": "vgsp", "actor_input_dim": 18},
}

METRICS = [
    "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate",
    "return", "final_coverage", "max_coverage", "nearest_landmark_distance",
    "minimum_agent_separation", "unique_landmarks_covered", "full_coverage_rate",
    # Retained for compatibility with the existing protocol status writer.
    "final3", "max3", "collisions",
]
RADII = np.linspace(0.05, 0.30, 26)


def geometry(env):
    agents = np.asarray([item.state.p_pos for item in env.world.agents])
    landmarks = np.asarray([item.state.p_pos for item in env.world.landmarks])
    distances = np.linalg.norm(agents[:, None] - landmarks[None], axis=2)
    separations = [np.linalg.norm(agents[i] - agents[j])
                   for i in range(3) for j in range(i + 1, 3)]
    collision_pairs = sum(
        separations[k] < env.world.agents[i].size + env.world.agents[j].size
        for k, (i, j) in enumerate(itertools.combinations(range(3), 2))
    )
    assignment = min(sum(distances[i, p[i]] for i in range(3))
                     for p in itertools.permutations(range(3))) / 3.0
    return distances, float(min(separations)), int(collision_pairs), float(assignment)


def evaluate_model(env_id, model_path, episodes, episode_length, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    maddpg = MADDPG.init_from_save(str(model_path)); maddpg.prep_rollouts(device="cpu")
    env = make_env(env_id, discrete_action=maddpg.discrete_action)
    rows = []
    try:
        for episode in range(episodes):
            test_seed = seed + episode
            torch.manual_seed(test_seed); np.random.seed(test_seed); env.seed(test_seed)
            obs = env.reset(); episode_return = 0.0; collision_steps = 0
            min_sep = float("inf"); max_coverage = 0; elapsed = 0
            for _ in range(episode_length):
                tensors = [torch.as_tensor(item, dtype=torch.float32).view(1, -1) for item in obs]
                with torch.no_grad():
                    actions = [a.cpu().numpy().ravel() for a in maddpg.step(tensors, explore=False)]
                obs, rewards, dones, _ = env.step(actions)
                episode_return += float(np.mean(rewards)); elapsed += 1
                distances, separation, collisions, assignment = geometry(env)
                nearest = distances.min(axis=0)
                min_sep = min(min_sep, separation); collision_steps += int(collisions > 0)
                max_coverage = max(max_coverage, int((nearest < 0.10).sum()))
                if all(dones): break
            radius_curve = np.asarray([(nearest < radius).mean() for radius in RADII])
            radius_auc = float(np.trapz(radius_curve, RADII) / (RADII[-1] - RADII[0]))
            closest_landmarks = distances.argmin(axis=1)
            final_coverage = int((nearest < 0.10).sum())
            rows.append({
                "eval_episode": episode, "test_seed": test_seed,
                "return": episode_return,
                "hungarian_assignment_distance": assignment,
                "coverage_radius_auc": radius_auc,
                "collision_step_rate": collision_steps / float(elapsed),
                "final_coverage": final_coverage, "max_coverage": max_coverage,
                "final3": int(final_coverage == 3), "max3": int(max_coverage == 3),
                "collisions": collisions,
                "nearest_landmark_distance": float(nearest.mean()),
                "minimum_agent_separation": min_sep,
                "unique_landmarks_covered": int(np.unique(closest_landmarks).size),
                "full_coverage_rate": int(final_coverage == 3),
            })
    finally:
        env.close()
    return rows


def add_locked_defaults(argv):
    defaults = {
        "--output-dir": "experiments/vector_signal_gsp_locked_screen",
        "--total-env-steps": "100000", "--eval-episodes": "500",
        "--curve-eval-episodes": "100", "--seeds": "21,22,23",
        "--checkpoint-steps": "20000,40000,60000,80000,100000",
    }
    present = {arg.split("=", 1)[0] for arg in argv}
    output = list(argv)
    for flag, value in defaults.items():
        if flag not in present: output.extend([flag, value])
    return output


def _write_locked_results(root):
    with (root / "seed_summaries.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    successful = [row for row in rows if row.get("status", "complete") != "failed"]
    lookup = {(row["method"], int(row["seed"])): row for row in successful}
    seeds = sorted({int(row["seed"]) for row in successful})
    paired = []
    target = "vector_signal_gsp_3agent"
    for baseline in ("action_aware_geometric_control", "active_gsp_6node"):
        if not all((target, seed) in lookup and (baseline, seed) in lookup for seed in seeds):
            continue
        for metric in METRICS:
            deltas = np.asarray([float(lookup[(target, seed)][metric])
                                 - float(lookup[(baseline, seed)][metric]) for seed in seeds])
            half = 4.3026527297 * deltas.std(ddof=1) / math.sqrt(len(deltas))
            paired.append({"comparison": f"{target}_vs_{baseline}", "metric": metric,
                           "mean_delta": deltas.mean(), "ci95_low": deltas.mean() - half,
                           "ci95_high": deltas.mean() + half,
                           "per_seed_deltas": ";".join(f"{x:.9g}" for x in deltas)})
    if paired:
        with (root / "paired_intervals.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired[0])); writer.writeheader(); writer.writerows(paired)

    labels = {key: value["short"] for key, value in METHODS.items()}
    lines = ["# Vector-Signal GSP Locked Screen Results", "",
             "Training seed is the statistical unit; intervals are paired 95% t intervals (n=3).", "",
             "| method | Hungarian (lower) | radius AUC (higher) | collision rate (lower) | return (higher) | coverage@0.10 (higher) |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for method in METHODS:
        matching = [row for row in successful if row["method"] == method]
        if not matching: continue
        mean = lambda key: np.mean([float(row[key]) for row in matching])
        lines.append(f"| {labels[method]} | {mean('hungarian_assignment_distance'):.4f} | "
                     f"{mean('coverage_radius_auc'):.4f} | {mean('collision_step_rate'):.4f} | "
                     f"{mean('return'):.3f} | {mean('final_coverage'):.3f} |")

    def deltas(baseline, metric):
        return np.asarray([float(lookup[(target, seed)][metric]) - float(lookup[(baseline, seed)][metric])
                           for seed in seeds])
    primary_pass = False
    full_screen = all((method, seed) in lookup for method in METHODS for seed in seeds)
    if len(seeds) == 3 and full_screen:
        for metric, sign in (("hungarian_assignment_distance", -1), ("coverage_radius_auc", 1)):
            values = deltas("action_aware_geometric_control", metric) * sign
            primary_pass |= bool(np.all(values > 0))
        collision_delta = deltas("action_aware_geometric_control", "collision_step_rate")
        collision_penalty = bool(collision_delta.mean() > 0 and np.all(collision_delta > 0))
        promote = primary_pass and not collision_penalty
        lines.extend(["", "## Promotion decision", "",
                      ("**PASS:** expand to 10 seeds." if promote else
                       "**STOP:** the locked promotion criteria were not met; do not tune or extend this branch."), "",
                      "The decision uses assignment distance/radius AUC consistency and collision-step rate; "
                      "coverage@0.10 alone cannot promote the branch."])
    (root / "FINAL_RESULT_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    protocol.METHODS = METHODS
    protocol.METRICS = METRICS
    protocol.evaluate_model = evaluate_model
    protocol.STATUS_TITLE = "Vector-Signal GSP Locked Screen Status"
    protocol.PLOT_TITLE_PREFIX = "Vector-signal GSP locked screen"
    sys.argv = add_locked_defaults(sys.argv)
    output_flag = sys.argv.index("--output-dir")
    output_root = Path(sys.argv[output_flag + 1])
    before = set(output_root.glob("run_*")) if output_root.exists() else set()
    protocol.main()
    created = set(output_root.glob("run_*")) - before
    if len(created) == 1:
        _write_locked_results(created.pop())


if __name__ == "__main__":
    main()
