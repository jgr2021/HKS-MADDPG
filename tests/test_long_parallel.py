import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from experiments.run_long_parallel import select_next, required_controls, refresh, write, read, continuation_inventory


class ParallelQueueTests(unittest.TestCase):
    def test_selection_never_duplicates_active_or_completed(self):
        names = ["raw", "geometry", "energy", "hks"]
        self.assertEqual(select_next(names, {"raw"}, {"geometry": {}}), "energy")
        self.assertIsNone(select_next(names, set(names), {}))

    def test_next_smoke_to_formal_keeps_order(self):
        self.assertEqual(select_next(["geometry", "energy", "hks"], set(), {"energy": {}}), "geometry")

    def test_skipped_failure_is_not_completed_or_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            methods = {n: {} for n in ("raw", "energy", "hks")}
            for phase in ("smoke", "formal"):
                write(root / (phase + "_raw.json"), dict(status="completed", validation={"passed": True}))
            write(root / "formal_energy.json", dict(status="failed", error="MemoryError"))
            c = SimpleNamespace(validate_run=Mock(return_value={"passed": True}), training_args=lambda phase: phase)
            done, smoke = continuation_inventory(root, c, {"methods": methods}, {"energy"})
            self.assertEqual(done, {"raw"})
            self.assertEqual(smoke, {"raw"})
            self.assertEqual(select_next(methods, done | {"energy"}, {}), "hks")
            self.assertEqual(read(root / "formal_energy.json")["status"], "failed")
            self.assertEqual(c.validate_run.call_count, 2)

    def test_continuation_rejects_unacknowledged_or_changed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            methods = {n: {} for n in ("raw", "energy")}
            write(root / "formal_energy.json", dict(status="failed"))
            write(root / "smoke_raw.json", dict(status="running"))
            c = SimpleNamespace(validate_run=Mock(return_value={"passed": False}), training_args=lambda phase: phase)
            with self.assertRaises(RuntimeError):
                continuation_inventory(root, c, {"methods": methods}, {"energy"})
            write(root / "smoke_raw.json", dict(status="completed", validation={"passed": True}))
            with self.assertRaises(RuntimeError):
                continuation_inventory(root, c, {"methods": methods}, {"energy"})
            with self.assertRaises(ValueError):
                continuation_inventory(root, c, {"methods": methods}, {"unknown"})

    def test_skip_decision_keeps_failed_master_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root / "manifest.json", {"methods": {"energy": dict(group="active", control="raw_mlp")}})
            write(root / "formal_energy.json", dict(status="failed", error="MemoryError"))
            write(root / "queue_skip_decisions.json", {"energy": {"reason": "User declined rerun"}})
            refresh(root, SimpleNamespace(gate=Mock()))
            row = read(root / "long_master.json")[0]
            self.assertEqual(row["status"], "failed")
            self.assertEqual(row["queue_disposition"], "skipped_by_user_no_retry")
            self.assertEqual(row["observed_failure"], "MemoryError")
            self.assertNotIn("return", row)

    def test_required_controls_deduplicated(self):
        self.assertEqual(required_controls(dict(control="raw_mlp", required_additional_controls=["geometry"])), ["raw_mlp", "geometry"])

    def test_out_of_order_completion_defers_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            methods = {"raw_mlp": dict(group="x", control=None), "geometry": dict(group="x", control="raw_mlp"),
                       "energy": dict(group="x", control="geometry")}
            write(root / "manifest.json", dict(methods=methods))
            summary = {k: 1 for k in ["return", "hungarian_assignment_distance", "coverage_radius_auc", "collision_step_rate",
                                      "final_coverage", "final3", "global_env_steps", "env_steps_per_sec"]}
            for name in ["raw_mlp", "energy"]:
                d = root / "formal" / name / "seed_1"
                d.mkdir(parents=True)
                write(d / "summary.json", summary)
                write(root / ("formal_" + name + ".json"), dict(status="completed"))
            gate = Mock()
            refresh(root, SimpleNamespace(gate=gate))
            gate.assert_not_called()
            row = read(root / "long_master.json")[-1]
            self.assertEqual(row["comparison_status"], "waiting for controls: geometry")
            self.assertNotIn("screen_pass", row)


if __name__ == "__main__":
    unittest.main()
