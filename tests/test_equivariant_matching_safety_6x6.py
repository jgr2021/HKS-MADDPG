import tempfile
import unittest

import torch

from algorithms.maddpg import MADDPG
from main import make_parallel_env
from utils.networks import (
    EquivariantMatchingSafety6x6Policy,
    EquivariantMatchingSafety6x6Policy8,
    EquivariantMatchingSafety6x6Policy16,
    EquivariantMatchingSafety6x6Policy32,
)


class EquivariantMatchingSafetyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(6601)
        self.policy = EquivariantMatchingSafety6x6Policy(36, 5, agent_index=2)

    def test_shapes_finite_and_doubly_stochastic_matching(self):
        logits = self.policy(torch.randn(11, 36) * 100.0)
        self.assertEqual(logits.shape, (11, 5))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(self.policy.last_matching_shape, (11, 5, 6, 6))
        self.assertLess(self.policy.last_matching_row_error, 2e-4)
        self.assertLess(self.policy.last_matching_column_error, 2e-4)

    def test_landmark_and_other_agent_permutation_invariance(self):
        raw = torch.randn(7, 36)
        permuted = raw.clone()
        landmark_order = torch.tensor([3, 5, 1, 0, 4, 2])
        other_order = torch.tensor([4, 1, 3, 0, 2])
        permuted[:, 4:16] = raw[:, 4:16].reshape(-1, 6, 2)[:, landmark_order].reshape(-1, 12)
        permuted[:, 16:26] = raw[:, 16:26].reshape(-1, 5, 2)[:, other_order].reshape(-1, 10)
        permuted[:, 26:36] = raw[:, 26:36].reshape(-1, 5, 2)[:, other_order].reshape(-1, 10)
        torch.testing.assert_close(self.policy(raw), self.policy(permuted), rtol=2e-5, atol=2e-6)

    def test_rotation_consistency_with_action_permutation(self):
        raw = torch.randn(8, 36)
        rotated = raw.clone()
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        for start, count in ((0, 1), (2, 1), (4, 6), (16, 5)):
            stop = start + 2 * count
            rotated[:, start:stop] = (
                raw[:, start:stop].reshape(-1, count, 2) @ rotation.T
            ).reshape(-1, 2 * count)
        original = self.policy(raw)
        rotated_logits = self.policy(rotated)
        mapping = [0, 3, 4, 2, 1]
        torch.testing.assert_close(original, rotated_logits[:, mapping], rtol=3e-5, atol=3e-6)

    def test_mixed_agent_indices_and_payload_round_trip(self):
        raw = torch.randn(12, 36)
        indices = torch.arange(12) % 6
        expected = self.policy.forward_with_agent_indices(raw, indices).detach()
        target = EquivariantMatchingSafety6x6Policy(36, 5, agent_index=5)
        target.load_probe_payload({"actor_state_dict": self.policy.state_dict()})
        actual = target.forward_with_agent_indices(raw, indices).detach()
        torch.testing.assert_close(expected, actual)

    def test_actor_only_raw_critic_boundary(self):
        env = make_parallel_env("simple_spread_6x6", 1, 6611, True)
        try:
            model = MADDPG.init_from_env(
                env, actor_model="equivariant_matching_safety_6x6"
            )
        finally:
            env.close()
        self.assertEqual(model.nagents, 6)
        self.assertEqual(model.agents[0].critic.fc1.in_features, 246)
        self.assertEqual(model.agents[0].policy(torch.randn(3, 36)).shape, (3, 5))

    def test_registered_iteration_variants_survive_checkpoint_reload(self):
        variants = (
            (
                "equivariant_matching_safety_6x6_sinkhorn8",
                EquivariantMatchingSafety6x6Policy8,
                8,
            ),
            (
                "equivariant_matching_safety_6x6_sinkhorn16",
                EquivariantMatchingSafety6x6Policy16,
                16,
            ),
            (
                "equivariant_matching_safety_6x6_sinkhorn32",
                EquivariantMatchingSafety6x6Policy32,
                32,
            ),
        )
        env = make_parallel_env("simple_spread_6x6", 1, 6612, True)
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for actor_model, policy_type, iterations in variants:
                    model = MADDPG.init_from_env(
                        env, actor_model=actor_model
                    )
                    self.assertIsInstance(model.agents[0].policy, policy_type)
                    self.assertEqual(
                        model.agents[0].policy.sinkhorn_iterations, iterations
                    )
                    checkpoint = f"{temp_dir}/{iterations}.pt"
                    model.save(checkpoint)
                    restored = MADDPG.init_from_save(checkpoint)
                    self.assertEqual(restored.actor_model, actor_model)
                    self.assertIsInstance(
                        restored.agents[0].policy, policy_type
                    )
                    self.assertEqual(
                        restored.agents[0].policy.sinkhorn_iterations,
                        iterations,
                    )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
