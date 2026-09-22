import unittest

import numpy as np

from utils.active_gsp_v3_features import (
    ACTIVE_GSP_FEATURES_PER_ACTION,
    ACTION_TO_CONTROL,
    DEFAULT_SENSITIVITY,
    N_ACTIONS,
    action_raw_potential_features,
    action_score_control_features,
    active_gsp_action_features,
    active_gsp_flat_features,
    compute_active_gsp_v3_from_local_obs_batch,
)
from utils.make_env import make_env


class ActiveGSPV3FeaturesTest(unittest.TestCase):
    def _sample_observations(self, seed=2027):
        env = make_env("simple_spread", discrete_action=True)
        try:
            env.seed(seed)
            return [np.asarray(obs, dtype=np.float32) for obs in env.reset()]
        finally:
            env.close()

    def test_action_conditioned_features_are_finite_and_action_sensitive(self):
        observations = self._sample_observations()
        for agent_index, raw_obs in enumerate(observations):
            features = active_gsp_action_features(raw_obs, agent_index)
            self.assertEqual(features.shape, (N_ACTIONS, ACTIVE_GSP_FEATURES_PER_ACTION))
            self.assertTrue(np.all(np.isfinite(features)))

            flat = active_gsp_flat_features(raw_obs, agent_index)
            self.assertEqual(flat.shape, (N_ACTIONS * ACTIVE_GSP_FEATURES_PER_ACTION,))
            self.assertTrue(np.all(np.isfinite(flat)))

            raw_potential = action_raw_potential_features(raw_obs, agent_index)
            self.assertEqual(raw_potential.shape, features.shape)
            self.assertTrue(np.all(np.isfinite(raw_potential)))

            control = action_score_control_features()
            self.assertEqual(control.shape, features.shape)
            self.assertTrue(np.allclose(control, 0.0))

            # At least one candidate action should differ from no-op in a
            # random non-degenerate reset.
            self.assertGreater(np.max(np.abs(features[1:] - features[0])), 1e-8)

    def test_fast_path_matches_reference(self):
        observations = self._sample_observations(seed=2028)
        batch = np.asarray(observations, dtype=np.float32)
        indices = np.asarray([0, 1, 2], dtype=np.int64)
        fast = compute_active_gsp_v3_from_local_obs_batch(
            batch,
            config={"self_agent_indices": indices},
        )
        reference = np.stack(
            [
                active_gsp_action_features(observations[i], i)
                for i in range(3)
            ],
            axis=0,
        )
        np.testing.assert_allclose(fast, reference, rtol=2e-5, atol=2e-5)

    def test_fast_path_is_local_only_and_noop_is_zero(self):
        observations = self._sample_observations(seed=2029)
        for agent_index, obs in enumerate(observations):
            features = compute_active_gsp_v3_from_local_obs_batch(
                obs[None, :],
                config={"self_agent_index": agent_index},
            )
            self.assertEqual(features.shape, (1, N_ACTIONS, ACTIVE_GSP_FEATURES_PER_ACTION))
            self.assertTrue(np.all(np.isfinite(features)))
            np.testing.assert_allclose(features[:, 0, :], 0.0, rtol=0.0, atol=1e-7)

    def test_translation_invariance(self):
        observations = self._sample_observations(seed=2030)
        shift = np.asarray([7.5, -3.25], dtype=np.float32)
        for agent_index, obs in enumerate(observations):
            shifted = obs.copy()
            shifted[2:4] += shift
            base = compute_active_gsp_v3_from_local_obs_batch(
                obs[None, :],
                config={"self_agent_index": agent_index},
            )
            moved = compute_active_gsp_v3_from_local_obs_batch(
                shifted[None, :],
                config={"self_agent_index": agent_index},
            )
            np.testing.assert_allclose(base, moved, rtol=0.0, atol=1e-7)

    def test_finite_value_stress(self):
        stress_cases = []
        overlap = np.zeros(18, dtype=np.float32)
        overlap[10:14] = 0.0
        overlap[4:10] = 0.0
        stress_cases.append(overlap)

        far = np.zeros(18, dtype=np.float32)
        far[4:10] = np.asarray([1e3, 0.0, 0.0, -1e3, 1e3, 1e3], dtype=np.float32)
        far[10:14] = np.asarray([-1e3, 1e3, 1e3, -1e3], dtype=np.float32)
        stress_cases.append(far)

        mixed = np.zeros(18, dtype=np.float32)
        mixed[0:2] = np.asarray([10.0, -10.0], dtype=np.float32)
        mixed[4:10] = np.asarray([1e-6, -1e-6, 100.0, 100.0, -100.0, 50.0], dtype=np.float32)
        mixed[10:14] = np.asarray([1e-6, 1e-6, -100.0, -100.0], dtype=np.float32)
        stress_cases.append(mixed)

        batch = np.stack(stress_cases, axis=0)
        features = compute_active_gsp_v3_from_local_obs_batch(
            batch,
            config={"self_agent_indices": np.asarray([0, 0, 0], dtype=np.int64)},
        )
        self.assertTrue(np.all(np.isfinite(features)))

    def test_batch_shapes(self):
        observations = self._sample_observations(seed=2031)
        for batch_size in (1, 12, 64):
            batch = np.asarray(
                [observations[i % 3] for i in range(batch_size)],
                dtype=np.float32,
            )
            indices = np.asarray([i % 3 for i in range(batch_size)], dtype=np.int64)
            features = compute_active_gsp_v3_from_local_obs_batch(
                batch,
                config={"self_agent_indices": indices},
            )
            self.assertEqual(
                features.shape,
                (batch_size, N_ACTIONS, ACTIVE_GSP_FEATURES_PER_ACTION),
            )
            self.assertTrue(np.all(np.isfinite(features)))

    def test_action_control_mapping_matches_mpe_onehot_parser(self):
        env = make_env("simple_spread", discrete_action=True)
        try:
            env.seed(2032)
            env.reset()
            agent = env.agents[0]
            for action_index in range(N_ACTIONS):
                onehot = np.zeros(N_ACTIONS, dtype=np.float32)
                onehot[action_index] = 1.0
                env._set_action(onehot.copy(), agent, env.action_space[0])
                expected = ACTION_TO_CONTROL[action_index] * DEFAULT_SENSITIVITY
                np.testing.assert_allclose(
                    agent.action.u,
                    expected,
                    rtol=0.0,
                    atol=1e-7,
                )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
