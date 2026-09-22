import unittest
import torch
from utils.adaptive_hks_controls import register_policies, METHODS
from utils.catalog76_policies import features
from utils.agents import POLICY_TYPES


class SupplementTests(unittest.TestCase):
    def setUp(self):
        register_policies()
        torch.manual_seed(456)
        self.raw = torch.randn(24, 18)

    def test_matched_capacity_and_original_descriptor(self):
        counts = []
        for name, (actor, width) in METHODS.items():
            if name == 'raw':
                continue
            policy = POLICY_TYPES[actor](18, 5, hidden_dim=64).eval()
            counts.append(sum(p.numel() for p in policy.parameters()))
            augmented = policy.augment_observation(self.raw)
            self.assertEqual(augmented.shape, (24, width))
            torch.testing.assert_close(augmented[:, :18], self.raw)
            self.assertTrue(torch.isfinite(policy(self.raw)).all())
            if name == 'adaptive_hks':
                torch.testing.assert_close(augmented[:, 18:], features(self.raw, {'kernel': 'adaptive'}))
        self.assertEqual(len(set(counts)), 1)

    def test_scale_and_landmark_permutation(self):
        scaled = self.raw.clone()
        scaled[:, 4:14] *= 2
        permuted = self.raw.clone()
        permuted[:, 4:10] = self.raw[:, 4:10].reshape(-1, 3, 2)[:, [2, 0, 1]].reshape(-1, 6)
        for name in ('adaptive_hks', 'fixed_hks', 'adaptive_adjacency', 'fixed_adjacency'):
            policy = POLICY_TYPES[METHODS[name][0]](18, 5).eval()
            feature = policy.augment_observation(self.raw)[:, 18:]
            torch.testing.assert_close(feature, policy.augment_observation(permuted)[:, 18:], atol=2e-6, rtol=2e-5)
            other = policy.augment_observation(scaled)[:, 18:]
            if name.startswith('adaptive'):
                torch.testing.assert_close(feature, other, atol=2e-6, rtol=2e-5)
            else:
                self.assertGreater((feature-other).abs().max().item(), 1e-3)


if __name__ == '__main__':
    unittest.main()
