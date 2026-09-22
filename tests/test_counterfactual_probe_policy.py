import unittest

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from experiments.probe_action_value_representations import Probe, representation_matrix
from main import make_parallel_env
from utils.networks import CounterfactualActiveProbePolicy


PAYLOAD = "experiments/action_value_representation_probe_20260712/active_hungarian_probe_seed1.pt"
DATASET = "experiments/action_value_representation_probe_20260712/counterfactual_dataset.npz"


class CounterfactualProbePolicyTest(unittest.TestCase):
    def test_exported_probe_equivalence(self):
        payload = torch.load(PAYLOAD, map_location="cpu")
        with np.load(DATASET) as data:
            indices = np.flatnonzero(data["train_seed"] == 23)[:5]
            sample = {key: data[key][indices] for key in data.files}
            raw = data["raw"][indices[:1]]
        x = representation_matrix(sample, "raw_action_active")
        probe = Probe(x.shape[1]); probe.load_state_dict(payload["probe_state_dict"]); probe.eval()
        policy = CounterfactualActiveProbePolicy(18, 5); policy.load_probe_payload(payload); policy.eval()
        with torch.no_grad():
            expected = probe(torch.from_numpy((x - payload["x_mean"]) / payload["x_std"]))[:, 0]
            expected = expected * payload["y_std"][0] + payload["y_mean"][0]
            logits = policy(torch.from_numpy(raw))[0]
        torch.testing.assert_close(logits, -expected, rtol=1e-5, atol=1e-7)
        self.assertEqual(int(logits.argmax()), int(expected.argmin()))

    def test_actor_and_raw_critic_boundaries(self):
        env = make_parallel_env("simple_spread", 1, 7171, True)
        try:
            maddpg = MADDPG.init_from_env(env, actor_model="counterfactual_active_probe")
        finally:
            env.close()
        policy = maddpg.agents[0].policy
        output = policy(torch.randn(7, 18))
        self.assertEqual(tuple(output.shape), (7, 5))
        self.assertEqual(policy.last_raw_input_shape, (7, 18))
        self.assertEqual(policy.last_action_feature_shape, (7, 5, 4))
        self.assertEqual(maddpg.agents[0].critic.fc1.in_features, 69)
        self.assertEqual(maddpg.agents[0].target_critic.fc1.in_features, 69)

    def test_distilled_actor_payload_round_trip(self):
        torch.manual_seed(919)
        source = CounterfactualActiveProbePolicy(18, 5)
        observations = torch.randn(11, 18)
        expected = source(observations).detach()
        restored = CounterfactualActiveProbePolicy(18, 5)
        restored.load_probe_payload({"actor_state_dict": source.state_dict()})
        torch.testing.assert_close(expected, restored(observations).detach())


if __name__ == "__main__":
    unittest.main()
