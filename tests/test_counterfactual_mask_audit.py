import unittest

import numpy as np
import torch

from experiments.diagnose_active_gsp_counterfactual_masks import (
    MASK_NAMES, MASKS, group_candidates, policy_digest, score_masks, selected_outcomes,
)
from utils.networks import LearnedActiveGSPResidualPolicy


def example_dataset():
    positions = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    landmarks = np.asarray([[0.2, 0.4], [1.0, 1.0], [-0.4, 0.5]], dtype=np.float32)
    raw = np.zeros((3, 18), dtype=np.float32)
    for focal in range(3):
        raw[focal, 2:4] = positions[focal]
        raw[focal, 4:10] = (landmarks - positions[focal]).ravel()
        raw[focal, 10:14] = (positions[np.arange(3) != focal] - positions[focal]).ravel()
    return {"raw": np.repeat(raw, 5, axis=0), "action": np.tile(np.eye(5), (3, 1)),
            "active": np.zeros((15, 4)), "targets": np.zeros((15, 3)),
            "reward_delta": np.tile(np.arange(5), 3), "collision": np.zeros(15),
            "group": np.repeat(np.arange(3), 5), "train_seed": np.full(15, 21),
            "episode": np.zeros(15, dtype=np.int64)}


class CounterfactualMaskAuditTest(unittest.TestCase):
    def test_grouping_survives_row_permutation_and_selects_correct_label(self):
        data = example_dataset()
        order = np.random.RandomState(12).permutation(15)
        grouped = group_candidates({key: value[order] for key, value in data.items()})
        np.testing.assert_array_equal(grouped["focal"], [0, 1, 2])
        outcomes = selected_outcomes(grouped["outcomes"], np.asarray([4, 2, 0]))
        np.testing.assert_array_equal(outcomes[:, 3], [4, 2, 0])

    def test_rejects_duplicate_candidate(self):
        data = example_dataset()
        data["action"][1] = data["action"][0]
        with self.assertRaises(AssertionError):
            group_candidates(data)

    def test_rejects_mixed_state_and_wrong_focal_mapping(self):
        data = example_dataset()
        data["raw"][1, 4] += 1
        with self.assertRaises(AssertionError):
            group_candidates(data)
        data = example_dataset()
        data["raw"][:5, 10] += 1
        with self.assertRaises(AssertionError):
            group_candidates(data)

    def test_rejects_nonzero_noop_and_invalid_action(self):
        data = example_dataset()
        data["reward_delta"][0] = 1
        with self.assertRaises(AssertionError):
            group_candidates(data)
        with self.assertRaises(ValueError):
            selected_outcomes(np.zeros((1, 5, 5)), np.asarray([5]))

    def test_mask_logits_match_actual_forward_without_changing_state(self):
        torch.manual_seed(124)
        policy = LearnedActiveGSPResidualPolicy(18, 5).eval()
        obs = torch.randn(12, 18)
        before = policy_digest([policy])
        logits, _, _ = score_masks(policy, obs)
        original = policy.action_features
        try:
            for index, name in enumerate(MASK_NAMES):
                policy.action_features = lambda x, name=name: original(x) * x.new_tensor(MASKS[name])
                with torch.no_grad():
                    torch.testing.assert_close(logits[:, index], policy(obs))
        finally:
            policy.action_features = original
        self.assertEqual(before, policy_digest([policy]))

    def test_mask_scorer_is_local_only(self):
        torch.manual_seed(125)
        policy = LearnedActiveGSPResidualPolicy(18, 5).eval()
        obs = torch.randn(12, 18)
        expected, _, _ = score_masks(policy, obs[:1])
        obs[1:] += 1000
        actual, _, _ = score_masks(policy, obs)
        torch.testing.assert_close(actual[:1], expected, rtol=1e-5, atol=1e-6)
        with self.assertRaises(ValueError):
            score_masks(policy, torch.zeros(12, 54))


if __name__ == "__main__":
    unittest.main()
