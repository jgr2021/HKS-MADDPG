import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from main import make_parallel_env
from utils.cuda_protocol_audit import CudaAuditedMADDPG
from utils.exploration_topology_policies import Topology4EgoPolicy, Topology6ALPolicy
from utils.exploration_weight_policies import POLICIES, ConstantStarHKSPolicy, weight_adjacency, register_policies


def reference(raw, model):
    points = np.concatenate((np.zeros((1, 2)), raw[10:14].reshape(2, 2), raw[4:10].reshape(3, 2)))
    sigma = .6 * model.sigma_multiplier
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    w = np.zeros((6, 6))
    directed = np.zeros_like(w)
    for p in range(6):
        candidates = list(range(3, 6)) if p < 3 else list(range(3))
        nearest = sorted(candidates, key=lambda q: (distances[p, q], q))[:2]
        directed[p, nearest] = 1
        for q in candidates:
            d = distances[p, q]
            value = sigma / (sigma + d) if model.kernel == "inverse" else np.exp(-d ** 2 / (2 * sigma ** 2))
            if model.support == "soft_radius":
                z = np.clip((1 - d) / .15, -700, 700)
                value /= 1 + np.exp(-z)
            w[p, q] = value
    if model.support == "knn2_union":
        w *= np.maximum(directed, directed.T)
    inv = np.maximum(w.sum(1), 1e-12) ** -.5
    values, vectors = np.linalg.eigh(np.eye(6) - inv[:, None] * w * inv[None, :])
    return np.array([np.sum(np.exp(-t * values.clip(0, 2)) * vectors[0] ** 2) for t in (.5, 1, 2)]), w


class WeightPoliciesTest(unittest.TestCase):
    def setUp(self):
        register_policies()
        torch.manual_seed(9752)
        torch.set_num_threads(1)
        self.obs = torch.randn(64, 18)

    def test_independent_reference_hks_and_edges(self):
        for cls in POLICIES.values():
            model = cls(18, 5).eval()
            if isinstance(model, ConstantStarHKSPolicy):
                continue
            hks = model._hks_features(self.obs).numpy()
            w = weight_adjacency(self.obs, model.edge_mask, model.edge_sigma, model.kernel, model.support).numpy()
            for index, row in enumerate(self.obs.numpy()):
                expected, adjacency = reference(row.astype(np.float64), model)
                np.testing.assert_allclose(hks[index], expected, rtol=2e-5, atol=2e-6)
                np.testing.assert_allclose(w[index], adjacency, rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(w, w.transpose(0, 2, 1), rtol=0, atol=0)
            self.assertTrue(np.all(w[:, :3, :3] == 0) and np.all(w[:, 3:, 3:] == 0))

    def test_star_constant_identity_and_binary_constant(self):
        model = ConstantStarHKSPolicy(18, 5)
        actual = model._hks_features(self.obs)
        expected = (.5 * (1 + torch.exp(-2 * model.hks_times))).expand_as(actual)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        star = Topology4EgoPolicy(18, 5)
        torch.testing.assert_close(star._hks_features(self.obs * .25), expected, rtol=2e-5, atol=2e-6)
        al = Topology6ALPolicy(18, 5)
        first = weight_adjacency(self.obs, al.edge_mask, al.edge_sigma, "binary")
        second = weight_adjacency(self.obs * 1e6, al.edge_mask, al.edge_sigma, "binary")
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_same_capacity_initialization_and_locality(self):
        expected_state = None
        for cls in (Topology6ALPolicy, *POLICIES.values()):
            torch.manual_seed(137)
            model = cls(18, 5).eval()
            state = model.state_dict()
            if expected_state:
                for key, value in state.items():
                    torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)
            expected_state = state
            baseline = model._hks_features(self.obs)
            changed = self.obs.clone()
            changed[:, :4] += 99
            changed[:, 14:] += 10
            torch.testing.assert_close(model._hks_features(changed), baseline, rtol=0, atol=0)
            changed[1:] = 0
            torch.testing.assert_close(model(changed)[:1], model(changed[:1]), rtol=1e-5, atol=1e-6)
            rotated = self.obs.clone()
            pairs = self.obs[:, 4:14].reshape(-1, 5, 2)
            rotated[:, 4:14] = torch.stack((-pairs[:, :, 1], pairs[:, :, 0]), -1).flatten(1)
            torch.testing.assert_close(model._hks_features(rotated), baseline, rtol=2e-5, atol=2e-6)
            model(self.obs).square().mean().backward()
            self.assertTrue(torch.isfinite(model.mlp.fc1.weight.grad).all())

    def test_stress_shape_and_target_save_load(self):
        env = make_parallel_env("simple_spread", 1, 1, True)
        try:
            for name, cls in POLICIES.items():
                model = cls(18, 5).eval()
                for batch in (1, 12, 64):
                    for scale in (0, 1, 1e6):
                        features = model._hks_features(self.obs[:batch] * scale)
                        self.assertEqual(features.shape, (batch, 3))
                        self.assertTrue(torch.isfinite(features).all())
                agent_model = MADDPG.init_from_env(env, actor_model=name)
                agent_model.prep_rollouts(device="cpu")
                agent_model.agents[0].target_policy.eval()
                expected = agent_model.agents[0].policy(self.obs)
                torch.testing.assert_close(expected, agent_model.agents[0].target_policy(self.obs), rtol=0, atol=0)
                with tempfile.TemporaryDirectory() as directory:
                    file = Path(directory) / "weights.pt"
                    agent_model.save(file)
                    restored = MADDPG.init_from_save(file)
                restored.prep_rollouts(device="cpu")
                torch.testing.assert_close(restored.agents[0].policy(self.obs), expected, rtol=0, atol=0)
                for agent in restored.agents:
                    self.assertEqual(agent.policy.mlp.fc1.in_features, 21)
                    self.assertEqual(agent.critic.fc1.in_features, 69)
                    self.assertEqual(agent.target_critic.fc1.in_features, 69)
        finally:
            env.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_reference_and_unchanged_updates(self):
        env = make_parallel_env("simple_spread", 1, 1, True)
        try:
            for name, cls in POLICIES.items():
                cpu = cls(18, 5).eval()
                gpu = cls(18, 5).cuda().eval()
                gpu.load_state_dict(cpu.state_dict())
                torch.testing.assert_close(cpu._hks_features(self.obs), gpu._hks_features(self.obs.cuda()).cpu(), rtol=2e-5, atol=2e-6)
                torch.manual_seed(46)
                original = MADDPG.init_from_env(env, actor_model=name)
                torch.manual_seed(46)
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
                    torch.manual_seed(85 + agent_i)
                    original.update(sample, agent_i, logger=Mock())
                    torch.manual_seed(85 + agent_i)
                    audited.update(sample, agent_i, logger=Mock())
                for left, right in zip(original.agents, audited.agents):
                    for module in ("policy", "critic", "target_policy", "target_critic"):
                        for key, value in getattr(left, module).state_dict().items():
                            torch.testing.assert_close(value, getattr(right, module).state_dict()[key], rtol=0, atol=0)
                self.assertEqual(audited._cuda_update_count, 3)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
