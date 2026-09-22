import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from main import make_parallel_env
from utils.cuda_protocol_audit import CudaAuditedMADDPG
from utils.exploration_descriptor_policies import POLICIES, descriptor_features, register_policies
from utils.exploration_topology_policies import Topology6ALPolicy, adjacency_from_local_obs


def reference(raw, descriptor):
    points = np.concatenate((np.zeros((1, 2)), raw[10:14].reshape(2, 2), raw[4:10].reshape(3, 2)))
    w = np.zeros((6, 6))
    for a in range(3):
        for landmark in range(3, 6):
            w[a, landmark] = w[landmark, a] = np.exp(-np.sum((points[a] - points[landmark]) ** 2) / (2 * .6 ** 2))
    if descriptor == "landmark_affinity":
        return w[0, 3:6]
    inv = np.maximum(w.sum(1), 1e-12) ** -.5
    laplacian = np.eye(6) - inv[:, None] * w * inv[None, :]
    if descriptor == "regularized_resistance":
        kernel = np.linalg.inv(laplacian + .1 * np.eye(6))
        return np.array([.05 * (kernel[0, 0] + kernel[j, j] - kernel[0, j] - kernel[j, 0]) for j in (3, 4, 5)])
    values, vectors = np.linalg.eigh(laplacian)
    values = values.clip(0, 2)
    if descriptor == "landmark_heat":
        return (vectors @ np.diag(np.exp(-values)) @ vectors.T)[0, 3:6]
    out = []
    for center in (.25, .75, 1.5):
        weights = np.array([np.exp(-(np.log(value) - np.log(center)) ** 2 / (2 * .5 ** 2)) if value > 1e-6 else 0 for value in values])
        out.append(np.dot(vectors[0] ** 2, weights) / max(weights.sum(), 1e-12))
    return np.array(out)


class DescriptorPoliciesTest(unittest.TestCase):
    def setUp(self):
        register_policies()
        torch.set_num_threads(1)
        torch.manual_seed(7482)
        self.obs = torch.randn(64, 18)

    def test_reference_batch_stress_and_nonconstant(self):
        for cls in POLICIES.values():
            model = cls(18, 5).eval()
            for count in (1, 12, 64):
                for scale in (0, 1, 1e6):
                    raw = self.obs[:count] * scale
                    actual = model._hks_features(raw)
                    expected = np.stack([reference(row.astype(np.float64), model.descriptor) for row in raw.numpy()])
                    self.assertEqual(actual.shape, (count, 3))
                    self.assertEqual(actual.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(actual).all())
                    np.testing.assert_allclose(actual.numpy(), expected, rtol=5e-5, atol=5e-6)
            self.assertTrue((model._hks_features(self.obs).std(0) > 1e-5).all())

    def test_matched_initialization_locality_rotation_and_channel_order(self):
        torch.manual_seed(73)
        initial = Topology6ALPolicy(18, 5).state_dict()
        for cls in POLICIES.values():
            torch.manual_seed(73)
            model = cls(18, 5).eval()
            self.assertEqual(set(model.state_dict()), set(initial))
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
            baseline = model._hks_features(self.obs)
            changed = self.obs.clone()
            changed[:, :4] += 99
            changed[:, 14:] -= 13
            torch.testing.assert_close(model._hks_features(changed), baseline, rtol=0, atol=0)
            changed[1:] = 0
            torch.testing.assert_close(model._hks_features(changed[:1]), baseline[:1], rtol=5e-5, atol=5e-6)
            torch.testing.assert_close(model(changed)[:1], model(changed[:1]), rtol=1e-5, atol=1e-6)
            rotated = self.obs.clone()
            pairs = self.obs[:, 4:14].reshape(-1, 5, 2)
            rotated[:, 4:14] = torch.stack((-pairs[:, :, 1], pairs[:, :, 0]), -1).flatten(1)
            torch.testing.assert_close(model._hks_features(rotated), baseline, rtol=5e-5, atol=5e-6)
            permuted = self.obs.clone()
            permuted[:, 4:10] = self.obs[:, 4:10].reshape(-1, 3, 2)[:, [2, 0, 1]].flatten(1)
            permuted[:, 10:14] = self.obs[:, 10:14].reshape(-1, 2, 2)[:, [1, 0]].flatten(1)
            expected = baseline if model.descriptor == "wks" else baseline[:, [2, 0, 1]]
            torch.testing.assert_close(model._hks_features(permuted), expected, rtol=5e-5, atol=5e-6)
            model(self.obs).square().mean().backward()
            self.assertTrue(torch.isfinite(model.mlp.fc1.weight.grad).all())

    def test_zero_edges_repeated_spectrum_and_resolvent_identity(self):
        base = Topology6ALPolicy(18, 5)
        centers = torch.tensor([.25, .75, 1.5]).log()
        zero = torch.zeros(1, 6, 6)
        for name, expected in (("landmark_affinity", 0.), ("landmark_heat", 0.),
                                ("wks", 1 / 6), ("regularized_resistance", 1 / 11)):
            actual = descriptor_features(zero, name, centers)
            torch.testing.assert_close(actual, torch.full((1, 3), expected), rtol=1e-5, atol=1e-6)
        w = adjacency_from_local_obs(self.obs, "6al", base.edge_mask, base.edge_sigma).double()
        inv = w.sum(-1).clamp_min(1e-12).rsqrt()
        lap = torch.eye(6) - inv[:, :, None] * w * inv[:, None, :]
        values, vectors = torch.linalg.eigh(lap)
        difference = vectors[:, :1, :] - vectors[:, 3:6, :]
        spectral = .05 * (difference.square() / (values[:, None, :] + .1)).sum(-1)
        actual = descriptor_features(w.float(), "regularized_resistance", centers)
        torch.testing.assert_close(actual.double(), spectral, rtol=5e-5, atol=5e-6)

    def test_geometry_control_ignores_other_agent_positions(self):
        model = POLICIES["explore_landmark_affinity3"](18, 5)
        expected = model._hks_features(self.obs)
        changed = self.obs.clone()
        changed[:, 10:14] += 100
        torch.testing.assert_close(expected, model._hks_features(changed), rtol=0, atol=0)

    def test_target_critic_save_restore(self):
        env = make_parallel_env("simple_spread", 1, 1, True)
        try:
            for name in POLICIES:
                model = MADDPG.init_from_env(env, actor_model=name)
                model.prep_rollouts(device="cpu")
                model.agents[0].target_policy.eval()
                expected = model.agents[0].policy(self.obs)
                torch.testing.assert_close(expected, model.agents[0].target_policy(self.obs), rtol=0, atol=0)
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "descriptor.pt"
                    model.save(checkpoint)
                    restored = MADDPG.init_from_save(checkpoint)
                restored.prep_rollouts(device="cpu")
                torch.testing.assert_close(restored.agents[0].policy(self.obs), expected, rtol=0, atol=0)
                for agent in restored.agents:
                    self.assertEqual(agent.policy.mlp.fc1.in_features, 21)
                    self.assertEqual(agent.critic.fc1.in_features, 69)
                    self.assertEqual(agent.target_critic.fc1.in_features, 69)
        finally:
            env.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_reference_and_original_update(self):
        env = make_parallel_env("simple_spread", 1, 1, True)
        try:
            for name, cls in POLICIES.items():
                cpu, gpu = cls(18, 5).eval(), cls(18, 5).cuda().eval()
                torch.testing.assert_close(cpu._hks_features(self.obs), gpu._hks_features(self.obs.cuda()).cpu(), rtol=5e-5, atol=5e-6)
                torch.manual_seed(31)
                original = MADDPG.init_from_env(env, actor_model=name)
                torch.manual_seed(31)
                audited = CudaAuditedMADDPG.init_from_env(env, actor_model=name)
                sample = ([torch.randn(16, 18, device="cuda") for _ in range(3)],
                          [torch.eye(5, device="cuda")[torch.arange(16, device="cuda") % 5] for _ in range(3)],
                          [torch.randn(16, device="cuda") for _ in range(3)],
                          [torch.randn(16, 18, device="cuda") for _ in range(3)],
                          [torch.zeros(16, device="cuda") for _ in range(3)])
                for model in (original, audited):
                    model.prep_training(device="gpu")
                    model.capture_policy_audit_references()
                    model.set_policy_audit_observations([row[:7] for row in sample[0]])
                for agent_i in range(3):
                    torch.manual_seed(98 + agent_i)
                    original.update(sample, agent_i, logger=Mock())
                    torch.manual_seed(98 + agent_i)
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
