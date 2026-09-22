import json
from pathlib import Path
import tempfile
import unittest

from experiments.run_gsp_long_campaign import expected_updates, methods, refresh, training_args
from experiments.run_gsp_exploration_campaign import write_json


class LongCampaignTests(unittest.TestCase):
    def test_budget_and_checkpoint_units(self):
        args = training_args("formal")
        self.assertEqual(args.total_env_steps, 100000 * 25)
        self.assertEqual(args.total_env_steps // args.episode_length, 100000)
        self.assertEqual(args.budget_episodes, 100000)
        self.assertEqual(args.batch_size, 1024)
        self.assertEqual(args.checkpoint_steps, list(range(20000, 2500001, 20000)))
        self.assertEqual(len(args.checkpoint_steps), 125)
        self.assertEqual(args.eval_episodes, 500)
        self.assertEqual(args.curve_eval_episodes, 100)

    def test_warmup_aware_update_count(self):
        smoke = training_args("smoke")
        self.assertEqual(smoke.total_env_steps, 2000)
        self.assertEqual(smoke.batch_size, 1024)
        self.assertEqual(expected_updates(smoke), 120)
        self.assertEqual(expected_updates(training_args("formal")), 299880)
        steps, updates = 0, 0
        args = training_args("formal")
        while steps < args.total_env_steps:
            steps += args.n_rollout_threads
            if steps >= args.batch_size and steps % args.steps_per_update < args.n_rollout_threads:
                updates += args.n_rollout_threads * 3
        self.assertEqual(steps, 2500000)
        self.assertEqual(updates, expected_updates(args))

    def test_all_methods_fresh_and_controls_precede_candidates(self):
        specs = methods()
        self.assertEqual(len(specs), 23)
        self.assertEqual(list(specs)[:2], ["raw_mlp", "explore_geometric_stats3"])
        seen = set()
        for name, spec in specs.items():
            self.assertNotIn("reuse_from", spec)
            self.assertTrue(spec["fresh"])
            if spec["control"]:
                self.assertIn(spec["control"], seen)
            for control in spec.get("additional_controls", []):
                self.assertIn(control, seen)
            seen.add(name)
        self.assertEqual(len({spec["short"] for spec in specs.values()}), 23)

    def test_short_names_avoid_legacy_windows_log_path(self):
        root = Path("C:/Users/10431/gsp_runs/long_20260906a")
        for spec in methods().values():
            model = "pgt_" + spec["short"] + "_s1_0906_060000s"
            path = root / "models/simple_spread" / model / "run1/logs/agent0/losses/initial_reference_kl/events.out.tfevents.1788644433.LAPTOP-D9RVKJDT"
            self.assertLess(len(str(path)), 240)

    def test_pending_results_not_confused_with_short_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "manifest.json", {"methods": methods()})
            refresh(root)
            rows = json.loads((root / "long_master.json").read_text())
            self.assertEqual(len(rows), 23)
            self.assertTrue(all(row["budget_env_steps"] == 2500000 for row in rows))
            self.assertTrue(all(row["budget_episodes"] == 100000 for row in rows))
            self.assertTrue(all(row["status"] == "queued" and "return" not in row for row in rows))


if __name__ == "__main__":
    unittest.main()
