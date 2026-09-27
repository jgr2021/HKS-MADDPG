"""Regression checks for the public entry points and retained model behavior."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from _common import load_runtime, summarize
from evaluate import select_checkpoint
from summarize import aggregate
import train

import torch
from algorithms.maddpg import MADDPG
from utils.catalog76_policies import graph, features, operators
from main import make_parallel_env


class PublicWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.protocol, cls.evaluation = load_runtime()

    def test_hks_matches_matrix_exponential_on_bipartite_graph(self):
        torch.manual_seed(41)
        raw = torch.randn(16, 18) * 0.4
        weights = graph(raw, {})
        self.assertEqual(torch.count_nonzero(weights[:, :3, :3]).item(), 0)
        self.assertEqual(torch.count_nonzero(weights[:, 3:, 3:]).item(), 0)
        torch.testing.assert_close(weights, weights.transpose(1, 2))
        _, laplacian, _ = operators(weights)
        expected = torch.stack([torch.matrix_exp(-t * laplacian)[:, 0, 0]
                                for t in (0.5, 1.0, 2.0)], dim=-1)
        torch.testing.assert_close(features(raw, {}), expected, atol=2e-6, rtol=2e-5)

    def test_descriptor_is_invariant_to_landmark_order(self):
        torch.manual_seed(42)
        raw = torch.randn(8, 18) * 0.4
        permuted = raw.clone()
        permuted[:, 4:10] = raw[:, 4:10].reshape(-1, 3, 2)[:, [2, 0, 1]].reshape(-1, 6)
        torch.testing.assert_close(features(raw, {}), features(permuted, {}), atol=2e-6, rtol=2e-5)

    def test_raw_and_hks_checkpoint_roundtrip_and_dimensions(self):
        for actor, width in (("mlp", 18), ("confirm_fixed_hks", 21)):
            with self.subTest(actor=actor), tempfile.TemporaryDirectory() as tmp:
                env = make_parallel_env("simple_spread", 1, 41, True)
                try:
                    model = MADDPG.init_from_env(env, actor_model=actor)
                    model.prep_rollouts()
                    observations = [torch.randn(4, 18) for _ in range(3)]
                    expected = model.step(observations, explore=False)
                    for agent in model.agents:
                        policy = getattr(agent.policy, "mlp", agent.policy)
                        self.assertEqual(policy.fc1.in_features, width)
                        self.assertEqual(agent.critic.fc1.in_features, 69)
                    checkpoint = Path(tmp) / "model.pt"
                    model.save(checkpoint)
                    loaded = MADDPG.init_from_save(checkpoint)
                    loaded.prep_rollouts()
                    for left, right in zip(expected, loaded.step(observations, explore=False)):
                        torch.testing.assert_close(left, right)
                finally:
                    env.close()

    def test_environment_evaluation_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = make_parallel_env("simple_spread", 1, 41, True)
            try:
                model = MADDPG.init_from_env(env, actor_model="confirm_fixed_hks")
                checkpoint = Path(tmp) / "model.pt"
                model.save(checkpoint)
            finally:
                env.close()
            rows = self.evaluation.evaluate_model("simple_spread", checkpoint, 2, 25, 92000000)
            again = self.evaluation.evaluate_model("simple_spread", checkpoint, 2, 25, 92000000)
            self.assertEqual(rows, again)
            report = summarize(rows)
            self.assertEqual(report["coverage_fraction"], report["final_coverage"] / 3)

    def test_checkpoint_selection_uses_return_and_portable_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "checkpoints").mkdir()
            (root / "checkpoints/model_step40000.pt").touch()
            (root / "protocol.json").write_text(json.dumps(dict(
                curve_eval_seed_base=91000000-410000, curve_eval_episodes=100)))
            rows = [dict(checkpoint="model_step20000", seed=41, method="hks", **{"return": -30}),
                    dict(checkpoint="model_step40000", seed=41, method="hks", **{"return": -20}),
                    dict(checkpoint="model_step60000", seed=41, method="hks", **{"return": -40})]
            with (root / "learning_curve_eval.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            selected, _, bank = select_checkpoint(root)
            self.assertEqual(selected.name, "model_step40000.pt")
            self.assertEqual(bank, (91000000, 100))

    def test_summary_requires_five_seeds_and_uses_sample_std(self):
        rows = [dict(method="hks", seed=seed, selection="highest_mean_curve_return",
                     seed_start=92000000, eval_episodes=500, episode_length=25,
                     device="cpu", curve_seed_start=91000000, curve_episodes=100,
                     coverage_fraction=0.5, full_coverage_rate=0.2, collision_step_rate=0.1,
                     **{"return": float(seed-40)}) for seed in range(41, 46)]
        result = aggregate(rows)
        return_row = next(r for r in result if r["metric"] == "return")
        self.assertEqual(return_row["mean"], 3)
        self.assertAlmostEqual(return_row["sample_std"], 2.5 ** 0.5)
        with self.assertRaises(ValueError):
            aggregate(rows[:-1])
        rows[-1]["seed_start"] += 1
        with self.assertRaises(ValueError):
            aggregate(rows)

    def test_training_entry_uses_common_banks_and_refuses_existing_run(self):
        # Exercise public dispatch without starting a training job.
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["train.py", "--method", "hks", "--seed", "41", "--output", tmp]
            previous = (self.protocol.MADDPG, self.protocol.USE_CUDA, self.protocol.METHODS)
            try:
                with patch.object(sys, "argv", argv), patch(
                    "utils.cuda_protocol_audit.require_cuda", return_value={"test": True}
                ), patch.object(self.protocol, "train_one") as dispatch:
                    train.main()
                    method, seed, args, output, _ = dispatch.call_args.args
                    self.assertEqual((method, seed), ("hks", 41))
                    self.assertEqual(args.n_rollout_threads, 4)
                    self.assertEqual(args.total_env_steps, 2000000)
                    self.assertEqual(args.checkpoint_steps[-1], 2000000)
                    self.assertEqual(args.curve_eval_seed_base + seed * 10000, 91000000)
                    self.assertEqual(args.eval_seed_base + seed * 10000, 92000000)
                    self.assertEqual(self.protocol.METHODS["hks"]["actor_model"], "confirm_fixed_hks")
                    self.assertTrue((output / "hks/seed_41/protocol.json").exists())
                    with self.assertRaises(SystemExit):
                        train.main()
                    self.assertEqual(dispatch.call_count, 1)
            finally:
                self.protocol.MADDPG, self.protocol.USE_CUDA, self.protocol.METHODS = previous


if __name__ == "__main__":
    unittest.main()
