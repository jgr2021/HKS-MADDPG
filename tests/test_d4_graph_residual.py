import unittest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from main import make_parallel_env
from utils.cuda_protocol_audit import CudaAuditedMADDPG, require_cuda
from utils.d4_graph_residual import D4DirichletResidualPolicy, D4PotentialResidualPolicy, transform_local_geometry
from utils.make_env import make_env


class D4GraphResidualTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(842)
        self.obs = torch.randn(12, 18)

    def test_residual_equivariance_for_all_rotations_and_reflections(self):
        policy = D4DirichletResidualPolicy(18, 5).eval()
        expected = policy.graph_residual(self.obs)
        orbit = transform_local_geometry(self.obs, policy.d4_matrices)
        for index in range(8):
            actual = policy.graph_residual(orbit[:, index])[:, policy.d4_actions[index]]
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=2e-7)

    def test_only_residual_discards_absolute_position_and_is_local(self):
        policy = D4DirichletResidualPolicy(18, 5).eval()
        expected = policy.graph_residual(self.obs)
        changed = self.obs.clone()
        changed[:, 2:4] += 50
        torch.testing.assert_close(policy.graph_residual(changed), expected, rtol=0, atol=0)
        changed = self.obs.clone()
        changed[1:] += 1000
        torch.testing.assert_close(policy(changed)[:1], policy(self.obs[:1]), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(policy(self.obs) - expected, policy.mlp(self.obs))

    def test_matched_control_has_same_parameters_and_only_masks_energies(self):
        full = D4DirichletResidualPolicy(18, 5)
        control = D4PotentialResidualPolicy(18, 5)
        self.assertEqual({key: value.shape for key, value in full.state_dict().items()},
                         {key: value.shape for key, value in control.state_dict().items()})
        features = full.action_features(self.obs)
        masked = control.action_features(self.obs)
        torch.testing.assert_close(features[:, :, :2], masked[:, :, :2])
        torch.testing.assert_close(masked[:, :, 2:], torch.zeros_like(masked[:, :, 2:]))
        torch.testing.assert_close(features[:, 0], torch.zeros_like(features[:, 0]))

    def test_finite_batches_and_backward(self):
        policy = D4DirichletResidualPolicy(18, 5).eval()
        for size in (1, 12, 64):
            obs = torch.zeros(size, 18)
            for distance in (0.0, 1e4):
                obs[:, 4:14] = distance
                logits = policy(obs)
                self.assertEqual(tuple(logits.shape), (size, 5))
                self.assertTrue(torch.isfinite(logits).all())
        loss = policy(self.obs).square().mean()
        loss.backward()
        self.assertGreater(policy.feature_encoder[0].weight.grad.abs().sum().item(), 0)

    def test_missing_cuda_is_an_error(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CPU fallback is disabled"):
                require_cuda()

    def test_action_permutations_match_the_actual_environment_controls(self):
        env = make_env("simple_spread", discrete_action=True)
        policy = D4DirichletResidualPolicy(18, 5)
        try:
            controls = []
            for index in range(5):
                env._set_action(np.eye(5)[index], env.agents[0], env.action_space[0])
                controls.append(env.agents[0].action.u.copy())
            controls = torch.tensor(np.asarray(controls), dtype=torch.float32)
            for matrix, permutation in zip(policy.d4_matrices, policy.d4_actions):
                torch.testing.assert_close(controls @ matrix.T, controls[permutation], rtol=0, atol=0)
        finally:
            env.close()

    def test_saved_checkpoint_restores_both_actor_variants_and_raw_critics(self):
        env = make_parallel_env("simple_spread", 1, 1, True)
        try:
            for name in ("d4_potential_residual", "d4_dirichlet_residual"):
                model = MADDPG.init_from_env(env, actor_model=name)
                model.prep_rollouts(device="cpu")
                with torch.no_grad():
                    expected = [agent.policy(self.obs) for agent in model.agents]
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "model.pt"
                    model.save(path)
                    restored = MADDPG.init_from_save(path)
                restored.prep_rollouts(device="cpu")
                for index, agent in enumerate(restored.agents):
                    torch.testing.assert_close(agent.policy(self.obs), expected[index], rtol=0, atol=0)
                    self.assertEqual(agent.critic.fc1.in_features, 69)
                    self.assertEqual(agent.target_critic.fc1.in_features, 69)
        finally:
            env.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA hardware required")
    def test_cuda_updates_match_uninstrumented_maddpg_and_keep_critic_raw(self):
        env = make_parallel_env("simple_spread", 1, 1, True)
        try:
            torch.manual_seed(38)
            original = MADDPG.init_from_env(env, actor_model="d4_dirichlet_residual")
            torch.manual_seed(38)
            audited = CudaAuditedMADDPG.init_from_env(env, actor_model="d4_dirichlet_residual")
        finally:
            env.close()
        original.prep_training(device="gpu")
        audited.prep_training(device="gpu")
        audit_obs = [torch.randn(7, 18, device="cuda") for _ in range(3)]
        for model in (original, audited):
            model.capture_policy_audit_references()
            model.set_policy_audit_observations(audit_obs)
        batch = 16
        sample = ([torch.randn(batch, 18, device="cuda") for _ in range(3)],
                  [torch.eye(5, device="cuda")[torch.arange(batch, device="cuda") % 5] for _ in range(3)],
                  [torch.randn(batch, device="cuda") for _ in range(3)],
                  [torch.randn(batch, 18, device="cuda") for _ in range(3)],
                  [torch.zeros(batch, device="cuda") for _ in range(3)])
        for agent_i in range(3):
            torch.manual_seed(501 + agent_i)
            original.update(sample, agent_i, logger=Mock())
            torch.manual_seed(501 + agent_i)
            audited.update(sample, agent_i, logger=Mock())
        for left, right in zip(original.agents, audited.agents):
            for name in ("policy", "critic", "target_policy", "target_critic"):
                for key, value in getattr(left, name).state_dict().items():
                    torch.testing.assert_close(value, getattr(right, name).state_dict()[key], rtol=0, atol=0)
        self.assertEqual(len(audited._cuda_agent_audits), 3)
        self.assertEqual(audited._cuda_update_count, 3)
        for record in audited._cuda_agent_audits.values():
            self.assertEqual(record["actor_gradient_devices"], ["cuda:0"])
            self.assertEqual(record["critic_gradient_devices"], ["cuda:0"])
            self.assertEqual(record["forwards"]["critic"][0]["shape"], [16, 69])
            self.assertEqual(record["forwards"]["actor"][-1]["shape"], [7, 18])
            self.assertEqual(record["forwards"]["actor"][-1]["purpose"], "existing_read_only_policy_audit")


if __name__ == "__main__":
    unittest.main()
