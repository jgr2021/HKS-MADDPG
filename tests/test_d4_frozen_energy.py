import unittest

import torch

from experiments.diagnose_d4_frozen_energy import without_energy_logits
from utils.d4_graph_residual import D4DirichletResidualPolicy, D4PotentialResidualPolicy


class D4FrozenEnergyTest(unittest.TestCase):
    def test_removal_matches_identical_weight_geometric_policy_and_restores_full_actor(self):
        torch.set_num_threads(1)
        torch.manual_seed(831)
        full = D4DirichletResidualPolicy(18, 5).eval()
        control = D4PotentialResidualPolicy(18, 5).eval()
        control.load_state_dict(full.state_dict())
        inputs = torch.randn(64, 18)
        expected_full = full(inputs).detach().clone()
        expected_state = {key: value.clone() for key, value in full.state_dict().items()}
        with torch.no_grad():
            actual = without_energy_logits(full, inputs)
        torch.testing.assert_close(actual, control(inputs), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(full(inputs), expected_full, rtol=0, atol=0)
        for key, value in full.state_dict().items():
            torch.testing.assert_close(value, expected_state[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
