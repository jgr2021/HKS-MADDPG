import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import torch

from algorithms.maddpg import MADDPG
from main import make_parallel_env
from utils.cuda_protocol_audit import CudaAuditedMADDPG
from utils.d4_graph_residual import D4DirichletResidualPolicy, transform_local_geometry
from utils.exploration_contact_policies import (
    ContactD4PotentialPolicy, ContactD4ResidualPolicy, register_policies,
)


class ContactActorTest(unittest.TestCase):
    def setUp(self):
        register_policies()
        torch.set_num_threads(1)
        torch.manual_seed(942)
        self.obs = torch.randn(12, 18)

    def test_same_initialization_capacity_and_only_feature_changes(self):
        policies = []
        for cls in (D4DirichletResidualPolicy, ContactD4PotentialPolicy, ContactD4ResidualPolicy):
            torch.manual_seed(71)
            policies.append(cls(18, 5).eval())
        for other in policies[1:]:
            for key, value in policies[0].state_dict().items():
                torch.testing.assert_close(value, other.state_dict()[key], rtol=0, atol=0)
        geometry, spectral = policies[1:]
        left, right = geometry.action_features(self.obs), spectral.action_features(self.obs)
        torch.testing.assert_close(left[:, :, :2], right[:, :, :2])
        self.assertEqual(torch.count_nonzero(left[:, :, 2:]).item(), 0)
        self.assertEqual(torch.count_nonzero(right[:, 0]).item(), 0)

    def test_locality_translation_and_trained_residual_symmetry(self):
        for cls in (ContactD4PotentialPolicy, ContactD4ResidualPolicy):
            model = cls(18, 5).eval()
            with torch.no_grad():
                model.residual_scorer[-1].weight.mul_(500)
            expected = model.graph_residual(self.obs)
            changed = self.obs.clone()
            changed[:, 2:4] += 200
            torch.testing.assert_close(model.graph_residual(changed), expected, rtol=0, atol=0)
            changed[1:] += 1000
            torch.testing.assert_close(model(changed)[:1], model(changed[:1]), rtol=1e-5, atol=1e-6)
            orbit = transform_local_geometry(self.obs, model.d4_matrices)
            for index in range(8):
                actual = model.graph_residual(orbit[:, index])[:, model.d4_actions[index]]
                torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)

    def test_save_load_target_parity_and_finite_backward(self):
        env = make_parallel_env("simple_spread", 1, 1, True)
        try:
            for name in ("explore_contact_d4_geometry", "explore_contact_d4_energy"):
                model = MADDPG.init_from_env(env, actor_model=name)
                model.prep_rollouts(device="cpu")
                model.agents[0].target_policy.eval()
                before = model.agents[0].policy(self.obs)
                torch.testing.assert_close(before, model.agents[0].target_policy(self.obs))
                before.square().mean().backward()
                grad = model.agents[0].policy.feature_encoder[0].weight.grad
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "checkpoint.pt"
                    model.save(path)
                    restored = MADDPG.init_from_save(path)
                restored.prep_rollouts(device="cpu")
                torch.testing.assert_close(restored.agents[0].policy(self.obs), before, rtol=0, atol=0)
                for agent in restored.agents:
                    self.assertEqual(agent.critic.fc1.in_features, 69)
                    self.assertEqual(agent.target_critic.fc1.in_features, 69)
        finally:
            env.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_actual_cuda_updates_equal_uninstrumented_raw_maddpg_update(self):
        env = make_parallel_env("simple_spread", 1, 1, True)
        try:
            for name in ("explore_contact_d4_geometry", "explore_contact_d4_energy"):
                torch.manual_seed(63)
                original = MADDPG.init_from_env(env, actor_model=name)
                torch.manual_seed(63)
                audited = CudaAuditedMADDPG.init_from_env(env, actor_model=name)
                for model in (original, audited):
                    model.prep_training(device="gpu")
                    model.capture_policy_audit_references()
                sample = ([torch.randn(16, 18, device="cuda") for _ in range(3)],
                          [torch.eye(5, device="cuda")[torch.arange(16, device="cuda") % 5] for _ in range(3)],
                          [torch.randn(16, device="cuda") for _ in range(3)],
                          [torch.randn(16, 18, device="cuda") for _ in range(3)],
                          [torch.zeros(16, device="cuda") for _ in range(3)])
                for model in (original, audited):
                    model.set_policy_audit_observations([x[:7] for x in sample[0]])
                for agent_i in range(3):
                    torch.manual_seed(88 + agent_i)
                    original.update(sample, agent_i, logger=Mock())
                    torch.manual_seed(88 + agent_i)
                    audited.update(sample, agent_i, logger=Mock())
                for left, right in zip(original.agents, audited.agents):
                    for module in ("policy", "critic", "target_policy", "target_critic"):
                        for key, value in getattr(left, module).state_dict().items():
                            torch.testing.assert_close(value, getattr(right, module).state_dict()[key], rtol=0, atol=0)
                self.assertEqual(audited._cuda_update_count, 3)
        finally:
            env.close()
