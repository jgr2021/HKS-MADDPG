import unittest

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from main import make_parallel_env
from utils.multiscale_spectral_features import (
    MULTISCALE_FEATURE_DIM,
    multiscale_rayleigh_exact,
    multiscale_spectral_action_features,
    multiscale_spectral_action_features_reference,
    normalized_laplacian_and_task_signals,
)


def random_observations(batch=12, seed=17, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    obs = torch.randn(batch, 18, generator=generator, dtype=dtype) * 0.4
    return obs


def rotate_vectors(obs):
    output = obs.clone()
    for start, stop in ((0, 2), (2, 4), (4, 10), (10, 14), (14, 18)):
        vectors = obs[:, start:stop].reshape(obs.shape[0], -1, 2)
        rotated = torch.stack([-vectors[..., 1], vectors[..., 0]], dim=-1)
        output[:, start:stop] = rotated.reshape(obs.shape[0], -1)
    return output


class TestMultiscaleSpectralFeatures(unittest.TestCase):
    def test_exact_batched_reference_equivalence(self):
        obs = random_observations(batch=7, dtype=torch.float64)
        exact = multiscale_spectral_action_features(obs, self_agent_index=1)
        reference = multiscale_spectral_action_features_reference(obs, self_agent_index=1)
        torch.testing.assert_close(exact, reference, rtol=2e-8, atol=2e-9)

    def test_noop_zero_deltas_and_batch_shape(self):
        features = multiscale_spectral_action_features(random_observations(19), 2)
        self.assertEqual(tuple(features.shape), (19, 5, MULTISCALE_FEATURE_DIM))
        torch.testing.assert_close(features[:, 0], torch.zeros_like(features[:, 0]), rtol=0.0, atol=0.0)

    def test_translation_invariance(self):
        obs = random_observations(9)
        translated = obs.clone()
        translated[:, 2:4] += torch.tensor([123.0, -77.0])
        torch.testing.assert_close(
            multiscale_spectral_action_features(obs, 0),
            multiscale_spectral_action_features(translated, 0),
            rtol=0.0,
            atol=0.0,
        )

    def test_rotation_consistency(self):
        obs = random_observations(11)
        original = multiscale_spectral_action_features(obs, 0)
        rotated = multiscale_spectral_action_features(rotate_vectors(obs), 0)
        action_after_ccw_rotation = [0, 3, 4, 2, 1]
        torch.testing.assert_close(
            original,
            rotated[:, action_after_ccw_rotation],
            rtol=2e-4,
            atol=2e-5,
        )

    def test_node_permutation_consistency(self):
        agents = torch.randn(8, 3, 2, dtype=torch.float64)
        landmarks = torch.randn(8, 3, 2, dtype=torch.float64)
        base = normalized_laplacian_and_task_signals(agents, landmarks)
        agent_perm = torch.tensor([2, 0, 1])
        landmark_perm = torch.tensor([1, 2, 0])
        permuted = normalized_laplacian_and_task_signals(
            agents[:, agent_perm], landmarks[:, landmark_perm]
        )
        node_perm = torch.cat([agent_perm, 3 + landmark_perm])
        expected_laplacian = base[0][:, node_perm][:, :, node_perm]
        torch.testing.assert_close(permuted[0], expected_laplacian, rtol=1e-12, atol=1e-12)
        expected_signals = torch.stack([base[1][:, node_perm], base[2][:, node_perm]], dim=1)
        actual_signals = torch.stack([permuted[1], permuted[2]], dim=1)
        base_r = multiscale_rayleigh_exact(base[0], torch.stack([base[1], base[2]], dim=1))
        permuted_r = multiscale_rayleigh_exact(permuted[0], actual_signals)
        torch.testing.assert_close(actual_signals, expected_signals, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(permuted_r, base_r, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(permuted[3], base[3])
        torch.testing.assert_close(permuted[4], base[4])

    def test_finite_extreme_inputs(self):
        for magnitude in (0.0, 1e4, 1e20):
            obs = torch.full((16, 18), magnitude)
            features = multiscale_spectral_action_features(obs, 0)
            self.assertTrue(torch.isfinite(features).all())
        obs = random_observations(4)
        obs[0, 4] = float("nan")
        obs[1, 5] = float("inf")
        self.assertTrue(torch.isfinite(multiscale_spectral_action_features(obs, 1)).all())

    def test_actor_only_local_input_and_raw_critic_boundary(self):
        env = make_parallel_env("simple_spread", 1, 4242, True)
        try:
            maddpg = MADDPG.init_from_env(env, actor_model="learned_multiscale_spectral_residual")
        finally:
            env.close()
        policy = maddpg.agents[0].policy
        obs = random_observations(6)
        changed = obs.clone()
        changed[1:] += 1000.0
        policy.eval()
        with torch.no_grad():
            expected = policy(obs[:1])
            actual = policy(changed)[:1]
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(policy.last_raw_input_shape, (6, 18))
        self.assertEqual(policy.last_action_feature_shape, (6, 5, MULTISCALE_FEATURE_DIM))
        self.assertEqual(maddpg.agents[0].critic.fc1.in_features, 69)
        self.assertEqual(maddpg.agents[0].target_critic.fc1.in_features, 69)


if __name__ == "__main__":
    unittest.main()
