import json
import tempfile
import unittest
from pathlib import Path

from experiments.analyze_hybrid_optimizer_controls import read_last_audit_kl


class OptimizerControlAnalysisTests(unittest.TestCase):
    def test_reads_windows_and_posix_tensorboard_key_suffixes(self):
        with tempfile.TemporaryDirectory() as directory:
            seed_dir = Path(directory)
            (seed_dir / "tensorboard_summary.json").write_text(
                json.dumps({
                    r"models\run\logs\agent0\losses\initial_reference_kl": [
                        [1.0, 0, 0.1],
                        [2.0, 1, 0.3],
                    ],
                    "models/run/logs/agent1/losses/initial_reference_kl": [
                        [1.0, 0, 0.2],
                        [2.0, 1, 0.5],
                    ],
                }),
                encoding="utf-8",
            )
            self.assertAlmostEqual(read_last_audit_kl(seed_dir), 0.4)


if __name__ == "__main__":
    unittest.main()
