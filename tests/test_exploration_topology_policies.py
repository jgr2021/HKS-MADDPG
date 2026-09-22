import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from main import make_parallel_env
from utils.cuda_protocol_audit import CudaAuditedMADDPG
from utils.gsp_features import topology_hks_descriptor
from utils.exploration_topology_policies import (
    POLICIES, BatchedTopologyPolicy, GeometricStatisticsPolicy,
    adjacency_from_local_obs, register_policies,
)


def reference_hks(raw, topology):
    agents = np.concatenate((np.zeros((1, 2)), raw[10:14].reshape(2, 2)))
    landmarks = raw[4:10].reshape(3, 2)
    if topology == "3aa":
        nodes, agents_n = agents, 3
    elif topology == "4ego":
        nodes, agents_n = np.concatenate((agents[:1], landmarks)), 1
    else:
        nodes, agents_n = np.concatenate((agents, landmarks)), 3
    w = np.zeros((len(nodes), len(nodes)), dtype=np.float64)
    for p in range(len(nodes)):
        for q in range(p + 1, len(nodes)):
            aa = p < agents_n and q < agents_n
            al = p < agents_n <= q
            ll = p >= agents_n
            if (aa and topology in ("3aa", "6aal", "6all")) or al or (ll and topology == "6all"):
                sigma = .8 if aa else .6
                w[p, q] = w[q, p] = np.exp(-np.linalg.norm(nodes[p] - nodes[q]) ** 2 / (2 * sigma ** 2))
    d = np.maximum(w.sum(1), 1e-12) ** -.5
    laplacian = np.eye(len(nodes)) - d[:, None] * w * d[None, :]
    values, vectors = np.linalg.eigh(laplacian)
    return np.array([np.sum(np.exp(-t * values.clip(0, 2)) * vectors[0] ** 2) for t in (.5, 1, 2)])


class TopologyPoliciesTest(unittest.TestCase):
    def setUp(self):
        register_policies()
        torch.set_num_threads(1)
        torch.manual_seed(913)
        self.obs = torch.randn(64, 18)

    def test_numpy_reference_and_legacy_for_each_focal_identity(self):
        for cls in POLICIES.values():
            model = cls(18, 5).eval()
            if isinstance(model, GeometricStatisticsPolicy):
                continue
            actual = model._hks_features(self.obs).numpy()
            expected = np.array([reference_hks(row.astype(np.float64), model.topology) for row in self.obs.numpy()])
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
            old = {"3aa": "topology_3node_aa_hks", "6al": "topology_6node_al_hks", "6aal": "topology_6node_aal_hks"}
            if model.topology in old:
                for agent in range(3):
                    legacy = np.array([topology_hks_descriptor(row, agent, old[model.topology]) for row in self.obs.numpy()])
                    np.testing.assert_allclose(actual, legacy, rtol=2e-5, atol=2e-6)

    def test_same_capacity_initialization_and_forward_backward(self):
        expected = None
        for cls in POLICIES.values():
            torch.manual_seed(74)
            model = cls(18, 5).eval()
            state = model.state_dict()
            if expected is not None:
                for key, value in expected.items():
                    torch.testing.assert_close(value, state[key], rtol=0, atol=0)
            expected = state
            self.assertEqual(model.mlp.fc1.in_features, 21)
            self.assertEqual(model.augment_observation(self.obs).shape, (64, 21))
            model(self.obs).square().mean().backward()
            grad = model.mlp.fc1.weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad[:, 18:].abs().sum().item(), 0)

    def test_locality_translation_rotation_and_permutation(self):
        for cls in POLICIES.values():
            model = cls(18, 5).eval()
            expected = model._hks_features(self.obs)
            changed = self.obs.clone()
            changed[:, :4] += 1000
            changed[:, 14:] += 2000
            torch.testing.assert_close(model._hks_features(changed), expected, rtol=0, atol=0)
            changed[1:] = 0
            torch.testing.assert_close(model(changed)[:1], model(changed[:1]), rtol=1e-5, atol=1e-6)
            swapped = self.obs.clone()
            swapped[:, 10:14] = self.obs[:, 10:14].reshape(-1, 2, 2).flip(1).flatten(1)
            swapped[:, 4:10] = self.obs[:, 4:10].reshape(-1, 3, 2).flip(1).flatten(1)
            torch.testing.assert_close(model._hks_features(swapped), expected, rtol=2e-5, atol=2e-6)
            rotated = self.obs.clone()
            xy = rotated[:, 4:14].reshape(-1, 5, 2).clone()
            xy = torch.stack((-xy[:, :, 1], xy[:, :, 0]), dim=-1)
            rotated[:, 4:14] = xy.flatten(1)
            torch.testing.assert_close(model._hks_features(rotated), expected, rtol=2e-5, atol=2e-6)

    def test_batch_stress_and_isolated_node_convention(self):
        for cls in POLICIES.values():
            model = cls(18, 5).eval()
            for batch in (1, 12, 64):
                for magnitude in (0., 1., 1e6):
                    obs = self.obs[:batch] * magnitude
                    features = model._hks_features(obs)
                    self.assertEqual(tuple(features.shape), (batch, 3))
                    self.assertTrue(torch.isfinite(features).all())
                    if not isinstance(model, GeometricStatisticsPolicy) and magnitude == 1e6:
                        torch.testing.assert_close(features, torch.exp(-model.hks_times).expand(batch, -1), rtol=2e-5, atol=2e-6)
            with self.assertRaises(ValueError):
                model._hks_features(torch.zeros(2, 21))
            with self.assertRaises(ValueError):
                model._hks_features(self.obs.double())

    def test_graph_edges_and_weight_definitions(self):
        expected_edges = {"3aa": 3, "4ego": 3, "6al": 9, "6aal": 12, "6all": 15}
        for cls in POLICIES.values():
            model = cls(18, 5)
            w = adjacency_from_local_obs(torch.zeros(1, 18), model.topology, model.edge_mask, model.edge_sigma)[0]
            torch.testing.assert_close(w, w.T, rtol=0, atol=0)
            self.assertEqual(int(w.sum()), 2 * expected_edges[model.topology])
            self.assertEqual(w.diagonal().abs().sum(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_reference_and_unmodified_training_update(self):
        env = make_parallel_env("simple_spread", 1, 1, True)
        try:
            for name, cls in POLICIES.items():
                cpu = cls(18, 5).eval()
                gpu = cls(18, 5).cuda().eval()
                gpu.load_state_dict(cpu.state_dict())
                torch.testing.assert_close(cpu._hks_features(self.obs), gpu._hks_features(self.obs.cuda()).cpu(), rtol=2e-5, atol=2e-6)
                torch.manual_seed(133)
                original = MADDPG.init_from_env(env, actor_model=name)
                torch.manual_seed(133)
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
                    torch.manual_seed(99 + agent_i)
                    original.update(sample, agent_i, logger=Mock())
                    torch.manual_seed(99 + agent_i)
                    audited.update(sample, agent_i, logger=Mock())
                for left, right in zip(original.agents, audited.agents):
                    for module in ("policy", "critic", "target_policy", "target_critic"):
                        for key, value in getattr(left, module).state_dict().items():
                            torch.testing.assert_close(value, getattr(right, module).state_dict()[key], rtol=0, atol=0)
                self.assertEqual(audited._cuda_update_count, 3)
        finally:
            env.close()

    def test_save_load_target_and_raw_critic(self):
        env = make_parallel_env("simple_spread", 1, 1, True)
        try:
            for name in POLICIES:
                model = MADDPG.init_from_env(env, actor_model=name)
                model.prep_rollouts(device="cpu")
                model.agents[0].target_policy.eval()
                expected = model.agents[0].policy(self.obs)
                torch.testing.assert_close(expected, model.agents[0].target_policy(self.obs), rtol=0, atol=0)
                with tempfile.TemporaryDirectory() as directory:
                    checkpoint = Path(directory) / "model.pt"
                    model.save(checkpoint)
                    restored = MADDPG.init_from_save(checkpoint)
                restored.prep_rollouts(device="cpu")
                torch.testing.assert_close(restored.agents[0].policy(self.obs), expected, rtol=0, atol=0)
                for agent in restored.agents:
                    self.assertEqual(agent.critic.fc1.in_features, 69)
                    self.assertEqual(agent.target_critic.fc1.in_features, 69)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
