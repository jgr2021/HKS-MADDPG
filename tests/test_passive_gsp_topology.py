import unittest

import numpy as np

from utils.gsp_features import (
    HKS_DIM,
    RAW_OBS_DIM,
    TOPOLOGY_3NODE_AA_HKS,
    TOPOLOGY_6NODE_AAL_HKS,
    TOPOLOGY_6NODE_AL_HKS,
    build_topology_hks_graph_from_positions,
    descriptor_for_mode,
    reconstruct_geometry_from_raw_observation,
)
from utils.make_env import make_env


TOPOLOGY_MODES = (
    TOPOLOGY_3NODE_AA_HKS,
    TOPOLOGY_6NODE_AL_HKS,
    TOPOLOGY_6NODE_AAL_HKS,
)


class PassiveGSPTopologyTest(unittest.TestCase):
    def test_topology_aliases_are_obs_only_and_21d(self):
        aliases = {
            "simple_spread_gsp_topology_3node_aa_hks": TOPOLOGY_3NODE_AA_HKS,
            "simple_spread_gsp_topology_6node_al_hks": TOPOLOGY_6NODE_AL_HKS,
            "simple_spread_gsp_topology_6node_aal_hks": TOPOLOGY_6NODE_AAL_HKS,
        }
        for env_id, mode in aliases.items():
            env = make_env(env_id, discrete_action=True)
            try:
                env.seed(2026)
                observations = env.reset()
                for agent_index, observation in enumerate(observations):
                    raw = env.last_raw_observations[agent_index]
                    direct = descriptor_for_mode(raw, agent_index, mode=mode)
                    wrapped = np.asarray(observation[RAW_OBS_DIM:], dtype=np.float32)
                    self.assertEqual(observation.shape, (RAW_OBS_DIM + HKS_DIM,))
                    np.testing.assert_allclose(direct, wrapped, rtol=0.0, atol=1e-7)
            finally:
                env.close()

    def test_adjacency_constraints_and_topology_masks(self):
        env = make_env("simple_spread", discrete_action=True)
        try:
            env.seed(7)
            observations = env.reset()
            raw = observations[0]
            agent_pos, landmark_pos = reconstruct_geometry_from_raw_observation(raw, 0)
            for mode in TOPOLOGY_MODES:
                adjacency, laplacian, eigenvalues, _ = build_topology_hks_graph_from_positions(
                    agent_pos,
                    landmark_pos,
                    mode=mode,
                )
                np.testing.assert_allclose(adjacency, adjacency.T, rtol=0.0, atol=1e-10)
                np.testing.assert_allclose(np.diag(adjacency), 0.0, rtol=0.0, atol=1e-12)
                np.testing.assert_allclose(laplacian, laplacian.T, rtol=0.0, atol=1e-10)
                self.assertTrue(np.all(np.isfinite(eigenvalues)))
                self.assertGreaterEqual(float(eigenvalues.min()), -1e-8)

            aa_adjacency, _, _, _ = build_topology_hks_graph_from_positions(
                agent_pos,
                landmark_pos,
                mode=TOPOLOGY_3NODE_AA_HKS,
            )
            self.assertEqual(aa_adjacency.shape, (3, 3))

            al_adjacency, _, _, _ = build_topology_hks_graph_from_positions(
                agent_pos,
                landmark_pos,
                mode=TOPOLOGY_6NODE_AL_HKS,
            )
            self.assertTrue(np.allclose(al_adjacency[:3, :3], 0.0))
            self.assertTrue(np.allclose(al_adjacency[3:, 3:], 0.0))

            aal_adjacency, _, _, _ = build_topology_hks_graph_from_positions(
                agent_pos,
                landmark_pos,
                mode=TOPOLOGY_6NODE_AAL_HKS,
            )
            self.assertGreater(float(np.abs(aal_adjacency[:3, :3]).sum()), 0.0)
            self.assertTrue(np.allclose(aal_adjacency[3:, 3:], 0.0))
        finally:
            env.close()

    def test_focal_agent_hks_is_not_fixed_to_agent0(self):
        env = make_env("simple_spread_gsp_topology_6node_aal_hks", discrete_action=True)
        try:
            env.seed(99)
            observations = env.reset()
            descriptors = np.asarray(
                [observation[RAW_OBS_DIM:] for observation in observations],
                dtype=np.float32,
            )
            self.assertEqual(descriptors.shape, (3, HKS_DIM))
            self.assertGreater(float(descriptors.std(axis=0).max()), 1e-6)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
