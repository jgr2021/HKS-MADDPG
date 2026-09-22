import unittest

import numpy as np

from utils.gsp_features import (
    GSP_DESCRIPTOR_DIM,
    RAW_OBS_DIM,
    SPECTRAL_DIM,
    descriptor_for_mode,
    reconstruct_geometry_from_raw_observation,
)
from utils.make_env import make_env


class GSPV1DecentralizedTest(unittest.TestCase):
    def test_wrapper_descriptor_matches_obs_only_reconstruction(self):
        env = make_env("simple_spread_gsp_v1", discrete_action=True)
        try:
            env.seed(1234)
            observations = env.reset()
            self.assertEqual(len(observations), 3)
            self.assertIsNotNone(env.last_raw_observations)

            spectra = []
            geometries = []
            for agent_index, observation in enumerate(observations):
                raw = env.last_raw_observations[agent_index]
                direct_descriptor = descriptor_for_mode(raw, agent_index, mode="full")
                wrapped_descriptor = np.asarray(observation[RAW_OBS_DIM:], dtype=np.float32)

                self.assertEqual(raw.shape, (RAW_OBS_DIM,))
                self.assertEqual(wrapped_descriptor.shape, (GSP_DESCRIPTOR_DIM,))
                np.testing.assert_allclose(
                    direct_descriptor,
                    wrapped_descriptor,
                    rtol=0.0,
                    atol=1e-7,
                )
                agent_pos, landmark_pos = reconstruct_geometry_from_raw_observation(
                    raw,
                    agent_index,
                )
                geometries.append((agent_pos, landmark_pos))
                spectra.append(direct_descriptor[:SPECTRAL_DIM])

            for agent_index in range(1, 3):
                np.testing.assert_allclose(
                    geometries[agent_index][0],
                    geometries[0][0],
                    rtol=0.0,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    geometries[agent_index][1],
                    geometries[0][1],
                    rtol=0.0,
                    atol=1e-6,
                )
                np.testing.assert_allclose(
                    spectra[agent_index],
                    spectra[0],
                    rtol=0.0,
                    atol=1e-6,
                )
        finally:
            env.close()

    def test_step_descriptor_matches_obs_only_reconstruction(self):
        env = make_env("simple_spread_gsp_v1", discrete_action=True)
        try:
            env.seed(4321)
            env.reset()
            noop = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            observations, _, _, _ = env.step([noop.copy(), noop.copy(), noop.copy()])

            for agent_index, observation in enumerate(observations):
                raw = env.last_raw_observations[agent_index]
                direct_descriptor = descriptor_for_mode(raw, agent_index, mode="full")
                wrapped_descriptor = np.asarray(observation[RAW_OBS_DIM:], dtype=np.float32)
                np.testing.assert_allclose(
                    direct_descriptor,
                    wrapped_descriptor,
                    rtol=0.0,
                    atol=1e-7,
                )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
