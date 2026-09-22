import json
from pathlib import Path
import tempfile
import unittest

from experiments.run_gsp_exploration_campaign import (
    METHODS, code_hashes, gate, read_json, refresh_table, verify_snapshot, write_json,
    run_directory,
)
from experiments.run_d4_actor_gpu_screen import training_args


class CampaignTests(unittest.TestCase):
    def test_locked_budgets(self):
        formal = training_args("formal")
        self.assertEqual(formal.total_env_steps, 100000)
        self.assertEqual(formal.checkpoint_steps, [20000, 40000, 60000, 80000, 100000])
        self.assertEqual(formal.eval_episodes, 500)
        self.assertEqual(formal.curve_eval_episodes, 100)
        self.assertEqual(training_args("smoke").total_env_steps, 1000)

    def test_collision_tradeoff_is_not_success(self):
        baseline = dict(zip(("return", "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate"), [-38, .6, .1, .06]))
        candidate = dict(zip(baseline, [-34, .5, .2, .061]))
        result = gate(candidate, baseline)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_criteria"], ["collision rate"])
        candidate["collision_step_rate"] = .05
        self.assertTrue(gate(candidate, baseline)["passed"])

    def test_source_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root / "code"
            code.mkdir()
            file = code / "example.py"
            file.write_text("value = 1\n")
            write_json(root / "manifest.json", {"source_hashes": code_hashes(code)})
            verify_snapshot(root)
            file.write_text("value = 2\n")
            with self.assertRaises(RuntimeError):
                verify_snapshot(root)

    def test_pending_table_does_not_invent_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "manifest.json", {"methods": METHODS})
            refresh_table(root)
            rows = read_json(root / "exploration_master.json")
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["status"] == "queued" for row in rows))
            self.assertTrue(all("return" not in row for row in rows))

    def test_reuse_is_explicit_and_only_affects_formal_path(self):
        root = Path("campaign")
        source = Path("archived") / "seed_1"
        spec = {"reuse_from": str(source)}
        self.assertEqual(run_directory(root, "formal", "raw_mlp", spec), source)
        self.assertEqual(run_directory(root, "smoke", "raw_mlp", spec), root / "smoke/raw_mlp/seed_1")

    def test_reused_comparison_does_not_write_to_archived_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "archive"
            old.mkdir()
            metrics = {"return": -38., "hungarian_assignment_distance": .6,
                       "coverage_radius_auc": .1, "collision_step_rate": .05,
                       "final_coverage": .1, "env_steps_per_sec": 100., "global_env_steps": 100000}
            write_json(old / "summary.json", metrics)
            write_json(old / "screening_gate.json", {"preserve": "archive"})
            original = (old / "screening_gate.json").read_bytes()
            methods = {"raw_mlp": {"reuse_from": str(old), "factor": "raw", "control": None},
                       "geometry": {"reuse_from": str(old), "factor": "geometry", "control": "raw_mlp"}}
            write_json(root / "manifest.json", {"methods": methods})
            for method in methods:
                write_json(root / f"formal_{method}.json", {"status": "completed"})
            refresh_table(root)
            self.assertEqual(original, (old / "screening_gate.json").read_bytes())
            self.assertTrue((root / "comparisons/geometry/screening_gate.json").exists())

    def test_operator_group_is_bounded_and_reuses_controls(self):
        from utils.exploration_operator_policies import experiment_methods
        methods = experiment_methods(Path("repo"))
        reused = [name for name, spec in methods.items() if spec.get("reuse_from")]
        new = [spec for spec in methods.values() if not spec.get("reuse_from")]
        self.assertEqual(reused, ["raw_mlp", "explore_geometric_stats3", "explore_hks_6al", "explore_constant_star3"])
        self.assertEqual(len(new), 4)
        self.assertTrue(all(spec["actor_input_dim"] == 21 for spec in new))
        self.assertTrue(all(spec["control"] == "explore_geometric_stats3" for spec in new))
        self.assertTrue(all(spec["additional_controls"] == ["explore_hks_6al", "explore_constant_star3"] for spec in new))

    def test_descriptor_group_is_bounded_with_landmark_matched_controls(self):
        from utils.exploration_descriptor_policies import experiment_methods
        methods = experiment_methods(Path("repo"))
        new = [name for name, spec in methods.items() if not spec.get("reuse_from")]
        self.assertEqual(len(methods), 8)
        self.assertEqual(new, ["explore_landmark_affinity3", "explore_wks3",
                               "explore_landmark_heat3", "explore_regularized_resistance3"])
        for name in new:
            self.assertEqual(methods[name]["actor_input_dim"], 21)
            self.assertEqual(methods[name]["control"], "explore_geometric_stats3")
        for name in new[2:]:
            self.assertEqual(methods[name]["required_additional_controls"], [new[0]])
            self.assertIn(new[0], methods[name]["additional_controls"])

    def test_required_landmark_control_cannot_be_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            methods = {
                "raw_mlp": {"factor": "raw", "control": None},
                "geometry": {"factor": "geometry", "control": "raw_mlp"},
                "affinity": {"factor": "landmark control", "control": "geometry"},
                "heat": {"factor": "heat", "control": "geometry",
                         "additional_controls": ["affinity"],
                         "required_additional_controls": ["affinity"]},
            }
            write_json(root / "manifest.json", {"methods": methods})
            for name, values in zip(methods, [(-38, .6, .1, .05), (-38, .6, .1, .05),
                                              (-36, .4, .3, .03), (-37, .5, .2, .04)]):
                metrics = dict(zip(("return", "hungarian_assignment_distance",
                                    "coverage_radius_auc", "collision_step_rate"), values))
                metrics.update(final_coverage=.1, env_steps_per_sec=100., global_env_steps=100000)
                write_json(root / "formal" / name / "seed_1/summary.json", metrics)
                write_json(root / f"formal_{name}.json", {"status": "completed"})
            refresh_table(root)
            candidate = read_json(root / "exploration_master.json")[-1]
            self.assertFalse(candidate["screen_pass"])
            decision = read_json(root / "formal/heat/seed_1/screening_gate.json")
            self.assertTrue(decision["raw"]["passed"])
            self.assertTrue(decision["matched_control"]["passed"])
            self.assertFalse(decision["additional_controls"]["affinity"]["passed"])
            self.assertEqual(decision["required_additional_controls"], ["affinity"])
            write_json(root / "formal_affinity.json", {"status": "queued"})
            with self.assertRaisesRegex(RuntimeError, "Required matched control"):
                refresh_table(root)


if __name__ == "__main__":
    unittest.main()
