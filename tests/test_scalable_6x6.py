import unittest

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from main import make_parallel_env
from utils.make_env import make_env
from utils.networks import ScalableCounterfactual6x6Policy
from utils.networks import _active_gsp_v3_features_tensor
from utils.scalable_active_features import (
    expected_observation_dim,
    scalable_active_action_features_tensor,
)


class ScalableSixBySixTests(unittest.TestCase):
    def test_generalized_features_match_locked_3x3_reference(self):
        torch.manual_seed(606)
        raw = torch.randn(13, 18)
        reference = _active_gsp_v3_features_tensor(raw)
        for agent_index in range(3):
            generalized = scalable_active_action_features_tensor(
                raw, agent_index, 3, 3
            )
            torch.testing.assert_close(generalized, reference, rtol=2e-5, atol=2e-6)

    def test_original_environment_is_unchanged(self):
        env = make_env("simple_spread", discrete_action=True)
        try:
            self.assertEqual(len(env.world.agents), 3)
            self.assertEqual(len(env.world.landmarks), 3)
            self.assertEqual(env.observation_space[0].shape, (18,))
        finally:
            env.close()

    def test_separate_6x6_cardinality_and_raw_boundary(self):
        env = make_env("simple_spread_6x6", discrete_action=True)
        try:
            self.assertEqual(len(env.world.agents), 6)
            self.assertEqual(len(env.world.landmarks), 6)
            self.assertEqual(expected_observation_dim(6, 6), 36)
            self.assertTrue(all(space.shape == (36,) for space in env.observation_space))
            self.assertTrue(all(space.n == 5 for space in env.action_space))
        finally:
            env.close()

    def test_features_noop_shapes_and_finite_extremes(self):
        raw = torch.randn(7, 36) * 1e4
        for agent_index in range(6):
            features = scalable_active_action_features_tensor(raw, agent_index, 6, 6)
            self.assertEqual(features.shape, (7, 5, 4))
            torch.testing.assert_close(features[:, 0], torch.zeros(7, 4))
            self.assertTrue(torch.isfinite(features).all())

    def test_translation_independence_of_relative_features(self):
        raw = torch.randn(5, 36)
        shifted = raw.clone()
        shifted[:, 2:4] += torch.tensor([123.0, -77.0])
        first = scalable_active_action_features_tensor(raw, 2, 6, 6)
        second = scalable_active_action_features_tensor(shifted, 2, 6, 6)
        torch.testing.assert_close(first, second)

    def test_rotation_consistency_with_action_permutation(self):
        torch.manual_seed(607)
        raw = torch.randn(9, 36)
        rotated = raw.clone()
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        rotated[:, 0:2] = raw[:, 0:2] @ rotation.T
        rotated[:, 2:4] = raw[:, 2:4] @ rotation.T
        rotated[:, 4:16] = (
            raw[:, 4:16].reshape(-1, 6, 2) @ rotation.T
        ).reshape(-1, 12)
        rotated[:, 16:26] = (
            raw[:, 16:26].reshape(-1, 5, 2) @ rotation.T
        ).reshape(-1, 10)
        original_features = scalable_active_action_features_tensor(raw, 3, 6, 6)
        rotated_features = scalable_active_action_features_tensor(rotated, 3, 6, 6)
        # +x -> +y, -x -> -y, +y -> -x, -y -> +x.
        mapping = [0, 3, 4, 2, 1]
        torch.testing.assert_close(
            original_features, rotated_features[:, mapping], rtol=2e-5, atol=2e-6
        )

    def test_landmark_and_other_agent_permutation_consistency(self):
        torch.manual_seed(608)
        raw = torch.randn(8, 36)
        permuted = raw.clone()
        landmark_order = torch.tensor([4, 1, 5, 0, 3, 2])
        other_order = torch.tensor([2, 4, 0, 3, 1])
        permuted[:, 4:16] = raw[:, 4:16].reshape(-1, 6, 2)[:, landmark_order].reshape(-1, 12)
        permuted[:, 16:26] = raw[:, 16:26].reshape(-1, 5, 2)[:, other_order].reshape(-1, 10)
        permuted[:, 26:36] = raw[:, 26:36].reshape(-1, 5, 2)[:, other_order].reshape(-1, 10)
        expected = scalable_active_action_features_tensor(raw, 0, 6, 6)
        actual = scalable_active_action_features_tensor(permuted, 0, 6, 6)
        torch.testing.assert_close(expected, actual, rtol=2e-5, atol=2e-6)

    def test_actor_and_raw_critic_boundaries(self):
        env = make_parallel_env("simple_spread_6x6", 1, 6060, True)
        try:
            model = MADDPG.init_from_env(env, actor_model="scalable_counterfactual_6x6")
        finally:
            env.close()
        self.assertEqual(model.nagents, 6)
        self.assertEqual(model.agents[0].critic.fc1.in_features, 246)
        policy = model.agents[4].policy
        output = policy(torch.randn(3, 36))
        self.assertEqual(output.shape, (3, 5))
        self.assertEqual(policy.last_action_feature_shape, (3, 5, 4))

    def test_distilled_payload_round_trip(self):
        torch.manual_seed(66)
        source = ScalableCounterfactual6x6Policy(36, 5, agent_index=3)
        raw = torch.randn(4, 36)
        expected = source(raw).detach()
        target = ScalableCounterfactual6x6Policy(36, 5, agent_index=3)
        target.load_probe_payload({"actor_state_dict": source.state_dict()})
        torch.testing.assert_close(expected, target(raw).detach())


if __name__ == "__main__":
    unittest.main()
