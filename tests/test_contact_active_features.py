import unittest

import numpy as np
import torch

from utils.active_gsp_v3_features import ACTION_TO_CONTROL_FLOAT32, _state_quantities
from utils.contact_active_gsp_features import compute_contact_active_features
from utils.d4_graph_residual import square_symmetries, transform_local_geometry
from utils.local_contact_motion import predict_local_candidate_positions


class ContactActiveFeaturesTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.actions = torch.tensor(ACTION_TO_CONTROL_FLOAT32)
        self.raw = np.random.default_rng(2973).normal(0, .6, (64, 18)).astype(np.float32)

    def test_reference_equivalence_and_batch_shapes(self):
        for batch in (1, 12, 64):
            raw = self.raw[:batch]
            positions = predict_local_candidate_positions(raw, "known_contact_graph")
            expected = np.asarray([[ _state_quantities(candidate, obs[4:10].reshape(3, 2))
                                     for candidate in candidates] for obs, candidates in zip(raw, positions)])
            expected -= expected[:, :1]
            actual = compute_contact_active_features(torch.tensor(raw), self.actions)
            self.assertEqual(tuple(actual.shape), (batch, 5, 4))
            np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-3, atol=3e-6)
            torch.testing.assert_close(actual[:, 0], torch.zeros_like(actual[:, 0]), rtol=0, atol=0)

    def test_row_isolation_and_absolute_translation_invariance(self):
        raw = torch.tensor(self.raw)
        expected = compute_contact_active_features(raw, self.actions)
        changed = raw.clone()
        changed[:, 2:4] += 300
        torch.testing.assert_close(compute_contact_active_features(changed, self.actions), expected, rtol=0, atol=0)
        changed[1:] += 20
        torch.testing.assert_close(compute_contact_active_features(changed, self.actions)[:1], expected[:1], rtol=0, atol=0)
        torch.testing.assert_close(compute_contact_active_features(raw[:1], self.actions), expected[:1], rtol=0, atol=0)

    def test_all_d4_action_permutations(self):
        raw = torch.tensor(self.raw)
        matrices = square_symmetries()
        expected = compute_contact_active_features(raw, self.actions)
        orbit = transform_local_geometry(raw, matrices)
        for index, matrix in enumerate(matrices):
            transformed = self.actions @ matrix.T
            permutation = (transformed[:, None] - self.actions[None]).square().sum(-1).argmin(-1)
            actual = compute_contact_active_features(orbit[:, index], self.actions)[:, permutation]
            torch.testing.assert_close(actual, expected, rtol=2e-3, atol=3e-6)

    def test_overlap_extreme_distance_and_input_gradients_are_finite(self):
        for distance in (0, 1e-8, 1e4):
            raw = torch.zeros(12, 18)
            raw[:, 4:14] = distance
            raw.requires_grad_(True)
            features = compute_contact_active_features(raw, self.actions)
            self.assertTrue(torch.isfinite(features).all())
            features.sum().backward()
            self.assertTrue(torch.isfinite(raw.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware required")
    def test_actual_cuda_outputs_and_input_backward_match_cpu(self):
        raw = torch.tensor(self.raw)
        expected = compute_contact_active_features(raw, self.actions)
        gpu = raw.cuda().requires_grad_(True)
        actual = compute_contact_active_features(gpu, self.actions.cuda())
        self.assertEqual(str(actual.device), "cuda:0")
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-3, atol=3e-6)
        actual.square().sum().backward()
        self.assertEqual(str(gpu.grad.device), "cuda:0")
        self.assertTrue(torch.isfinite(gpu.grad).all())

    def test_wrong_dtype_and_missing_actions_are_rejected(self):
        with self.assertRaises(ValueError):
            compute_contact_active_features(torch.zeros(1, 18, dtype=torch.float64), self.actions)
        with self.assertRaises(ValueError):
            compute_contact_active_features(torch.zeros(1, 18), self.actions[:4])


if __name__ == "__main__":
    unittest.main()
