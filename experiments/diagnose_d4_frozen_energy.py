"""One fixed energy-removal intervention; no training, fitting or mask search."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from algorithms.maddpg import MADDPG
from experiments.analyze_d4_actor_gpu_screen import read_csv, validate_episodes, write_csv, write_json
from experiments.diagnose_active_gsp_counterfactual_masks import policy_digest
from experiments.diagnose_learned_active_gsp_masks import install_mask
from experiments.run_d4_actor_gpu_screen import sha256
import run_vector_signal_gsp_experiment as evaluator


ENERGY_REMOVAL = (1.0, 1.0, 0.0, 0.0)


def without_energy_logits(policy, observations):
    original = policy.action_features
    def masked(value):
        features = original(value)
        return features * features.new_tensor(ENERGY_REMOVAL)
    policy.action_features = masked
    try:
        return policy(observations)
    finally:
        policy.action_features = original


def evaluate(checkpoint, remove_energy=False, attribution=False):
    captured = {}
    statistics = [{"agent": index, "decisions": 0, "changed_decisions": 0,
                   "full_action_ties": 0, "ablated_action_ties": 0,
                   "centered_raw_sum_squares": 0.0, "centered_residual_sum_squares": 0.0}
                  for index in range(3)]

    class FrozenMADDPG(MADDPG):
        @classmethod
        def init_from_save(cls, filename):
            model = super().init_from_save(filename)
            captured.update(model=model, before=policy_digest(model.policies))
            if remove_energy:
                install_mask(model, ENERGY_REMOVAL)
            return model

        def step(self, observations, explore=False):
            if attribution:
                for index, (policy, obs) in enumerate(zip(self.policies, observations)):
                    full = policy(obs)
                    ablated = without_energy_logits(policy, obs)
                    raw = policy.mlp(obs)
                    residual = full - raw
                    raw_centered = raw - raw.mean(1, keepdim=True)
                    residual_centered = residual - residual.mean(1, keepdim=True)
                    stats = statistics[index]
                    full_actions = full == full.max(1, keepdim=True).values
                    ablated_actions = ablated == ablated.max(1, keepdim=True).values
                    stats["decisions"] += len(obs)
                    stats["changed_decisions"] += int((full_actions != ablated_actions).any(1).sum())
                    stats["full_action_ties"] += int((full_actions.sum(1) > 1).sum())
                    stats["ablated_action_ties"] += int((ablated_actions.sum(1) > 1).sum())
                    stats["centered_raw_sum_squares"] += float(raw_centered.square().sum())
                    stats["centered_residual_sum_squares"] += float(residual_centered.square().sum())
            return super().step(observations, explore=explore)

    previous = evaluator.MADDPG
    try:
        evaluator.MADDPG = FrozenMADDPG
        rows = evaluator.evaluate_model("simple_spread", checkpoint, 500, 25, 1000000)
    finally:
        evaluator.MADDPG = previous
    assert policy_digest(captured["model"].policies) == captured["before"], "Frozen policy state changed"
    validate_episodes(rows)
    for stats in statistics:
        if stats["decisions"]:
            stats["changed_fraction"] = stats["changed_decisions"] / stats["decisions"]
            for branch in ("raw", "residual"):
                stats[f"centered_{branch}_logit_rms"] = (stats[f"centered_{branch}_sum_squares"] / (5 * stats["decisions"])) ** 0.5
    return rows, statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    root, output = options.run_dir.resolve(), options.output_dir.resolve()
    assert json.loads((root / "status.json").read_text())["status"] == "completed"
    checkpoint = root / "d4_dirichlet_gsp/seed_1/checkpoints/model_final_100000.pt"
    before = sha256(checkpoint)
    archived_path = root / "d4_dirichlet_gsp/seed_1/per_evaluation_episode_metrics.csv"
    archived_hash = sha256(archived_path)
    archived = read_csv(archived_path)
    validate_episodes(archived)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {"status": "running", "new_training_steps": 0, "checkpoint": str(checkpoint),
                "checkpoint_sha256": before, "mask": ENERGY_REMOVAL,
                "device": "cpu (evaluation only; chosen by the recorded D4 inference benchmark)",
                "n_training_seeds": 1, "episodes_per_condition": 500, "horizon": 25,
                "selection": "single fixed removal of both energies, no mask or threshold search",
                "interpretation": "post-training input intervention, not a retrained ablation or promotion test"}
    write_json(output / "protocol.json", protocol)
    torch.set_num_threads(1)
    start = time.perf_counter()
    try:
        print("Replaying all 500 full-policy episodes and auditing same-state action changes", flush=True)
        full, stats = evaluate(checkpoint, attribution=True)
        for actual, expected in zip(full, archived):
            for metric in evaluator.METRICS:
                np.testing.assert_allclose(actual[metric], float(expected[metric]), rtol=0, atol=1e-10)
        write_csv(output / "full_replay_episodes.csv", full)
        write_json(output / "same_state_attribution.json", stats)
        print("Full replay matched; evaluating 500 episodes with both energies removed", flush=True)
        masked, _ = evaluate(checkpoint, remove_energy=True)
        write_csv(output / "energy_removed_episodes.csv", masked)
        summary = []
        for metric in evaluator.METRICS:
            left = np.asarray([row[metric] for row in full], dtype=float)
            right = np.asarray([row[metric] for row in masked], dtype=float)
            summary.append({"metric": metric, "full_mean": float(left.mean()),
                            "energy_removed_mean": float(right.mean()),
                            "removed_minus_full": float((right - left).mean())})
        write_csv(output / "metrics.csv", summary)
        assert sha256(checkpoint) == before and sha256(archived_path) == archived_hash
        protocol.update(status="completed", elapsed_seconds=time.perf_counter() - start,
                        full_replay_matches_all_500=True, weights_and_archived_results_unchanged=True)
    except BaseException as error:
        protocol.update(status="failed", error=repr(error))
        raise
    finally:
        write_json(output / "protocol.json", protocol)
    print(json.dumps({"protocol": protocol, "attribution": stats, "metrics": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
