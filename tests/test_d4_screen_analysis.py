import copy
import unittest

from experiments.analyze_d4_actor_gpu_screen import METRICS, validate_episodes


class D4ScreenAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.rows = [{**{metric: 0.0 for metric in METRICS},
                      "test_seed": 1000000 + index, "eval_episode": index} for index in range(500)]

    def test_matching_complete_evaluation_is_accepted(self):
        validate_episodes(self.rows)

    def test_partial_evaluation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "episode count"):
            validate_episodes(self.rows[:-1])

    def test_duplicate_or_reordered_initial_states_are_rejected(self):
        self.rows[1]["test_seed"] = self.rows[0]["test_seed"]
        with self.assertRaisesRegex(ValueError, "states"):
            validate_episodes(self.rows)

    def test_nan_metric_is_rejected(self):
        altered = copy.deepcopy(self.rows)
        altered[100]["hungarian_assignment_distance"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            validate_episodes(altered)


if __name__ == "__main__":
    unittest.main()
