import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from utils.active_gsp_v3_features import compute_active_gsp_v3_from_local_obs_batch
from utils.env_wrappers import DummyVecEnv
from utils.make_env import make_env
from utils.networks import (
    ActionRawPotentialPolicy,
    ActionRawPotentialFixedPolicy,
    ActionRawPotentialPriorPolicy,
    ActionRawPotentialResidualPolicy,
    ActionScoreControlPolicy,
    ActiveGSPV3FixedPolicy,
    ActiveGSPV3Policy,
    ActiveGSPV3PriorPolicy,
    ActiveGSPV3ResidualPolicy,
    ScaledActionRawPotentialPolicy,
    ScaledActiveGSPV3Policy,
    _active_gsp_v3_features_tensor,
)


class ActiveGSPV3PolicyTest(unittest.TestCase):
    def _sample_batch(self, seed=3100):
        env = make_env("simple_spread", discrete_action=True)
        try:
            env.seed(seed)
            return np.asarray(env.reset(), dtype=np.float32)
        finally:
            env.close()

    def test_torch_feature_path_matches_numpy_fast_path(self):
        batch = self._sample_batch()
        torch_features = _active_gsp_v3_features_tensor(
            torch.as_tensor(batch, dtype=torch.float32)
        ).detach().numpy()
        numpy_features = compute_active_gsp_v3_from_local_obs_batch(batch)
        np.testing.assert_allclose(torch_features, numpy_features, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(torch_features[:, 0, :], 0.0, rtol=0.0, atol=1e-7)

    def test_action_scoring_policy_variants_emit_finite_logits(self):
        batch = torch.as_tensor(self._sample_batch(), dtype=torch.float32)
        for policy_cls in (
            ActionScoreControlPolicy,
            ActionRawPotentialPolicy,
            ActiveGSPV3Policy,
            ScaledActionRawPotentialPolicy,
            ScaledActiveGSPV3Policy,
            ActionRawPotentialPriorPolicy,
            ActiveGSPV3PriorPolicy,
            ActionRawPotentialResidualPolicy,
            ActiveGSPV3ResidualPolicy,
            ActionRawPotentialFixedPolicy,
            ActiveGSPV3FixedPolicy,
        ):
            policy = policy_cls(18, 5, hidden_dim=32, discrete_action=True)
            logits = policy(batch)
            self.assertEqual(tuple(logits.shape), (3, 5))
            self.assertTrue(torch.isfinite(logits).all().item())

            action_features = policy._action_features(batch)
            self.assertEqual(tuple(action_features.shape), (3, 5, 4))
            self.assertTrue(torch.isfinite(action_features).all().item())
            if policy_cls is ActionScoreControlPolicy:
                self.assertTrue(torch.allclose(action_features, torch.zeros_like(action_features)))
            if policy_cls is ActionRawPotentialPolicy:
                self.assertTrue(torch.allclose(action_features[:, :, 2:], torch.zeros_like(action_features[:, :, 2:])))

    def test_prior_policy_initial_logits_match_feature_heuristic(self):
        batch = torch.as_tensor(self._sample_batch(seed=3101), dtype=torch.float32)

        raw_prior = ActionRawPotentialPriorPolicy(18, 5, hidden_dim=32, discrete_action=True)
        raw_features = raw_prior._action_features(batch)
        raw_expected = torch.argmin(raw_features[:, :, 0] + raw_features[:, :, 1], dim=1)
        raw_actual = torch.argmax(raw_prior(batch), dim=1)
        self.assertTrue(torch.equal(raw_actual, raw_expected))

        v3_prior = ActiveGSPV3PriorPolicy(18, 5, hidden_dim=32, discrete_action=True)
        v3_features = v3_prior._action_features(batch)
        v3_expected = torch.argmin(v3_features.sum(dim=2), dim=1)
        v3_actual = torch.argmax(v3_prior(batch), dim=1)
        self.assertTrue(torch.equal(v3_actual, v3_expected))

        fixed = ActiveGSPV3FixedPolicy(18, 5, hidden_dim=32, discrete_action=True)
        fixed_actual = torch.argmax(fixed(batch), dim=1)
        self.assertTrue(torch.equal(fixed_actual, v3_expected))

    def test_maddpg_active_actor_initializes_saves_and_loads(self):
        env = DummyVecEnv([lambda: make_env("simple_spread", discrete_action=True)])
        try:
            for actor_model in (
                "active_gsp_v3",
                "active_gsp_v3_scaled",
                "active_gsp_v3_prior",
                "active_gsp_v3_residual005",
                "active_gsp_v3_fixed",
            ):
                maddpg = MADDPG.init_from_env(env, actor_model=actor_model)
                obs = env.reset()
                torch_obs = [
                    torch.as_tensor(obs[:, agent_index], dtype=torch.float32)
                    for agent_index in range(maddpg.nagents)
                ]
                with torch.no_grad():
                    actions = maddpg.step(torch_obs, explore=False)
                self.assertEqual(len(actions), 3)
                for action in actions:
                    self.assertEqual(tuple(action.shape), (1, 5))
                    self.assertTrue(torch.isfinite(action).all().item())

                with tempfile.TemporaryDirectory() as tmp_dir:
                    model_path = Path(tmp_dir) / "model.pt"
                    maddpg.save(model_path)
                    loaded = MADDPG.init_from_save(str(model_path))
                    loaded.prep_rollouts(device="cpu")
                    with torch.no_grad():
                        loaded_actions = loaded.step(torch_obs, explore=False)
                    self.assertEqual(len(loaded_actions), 3)
                    for action in loaded_actions:
                        self.assertEqual(tuple(action.shape), (1, 5))
                        self.assertTrue(torch.isfinite(action).all().item())
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
