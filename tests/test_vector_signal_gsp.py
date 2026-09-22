import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from algorithms.maddpg import MADDPG
from utils.buffer import ReplayBuffer
from utils.env_wrappers import DummyVecEnv
from utils.make_env import make_env
from utils.vector_signal_gsp_features import (
    UPPER_TRIANGLE,
    candidate_geometry_from_local_obs_batch,
    compute_vector_signal_features_from_local_obs_batch,
    vector_signal_features_tensor,
)


def reference_features(obs, variant):
    agents, landmarks = candidate_geometry_from_local_obs_batch(obs)
    output = []
    for candidate_agents in agents[0]:
        x = np.linalg.norm(candidate_agents[:, None] - landmarks[0][None], axis=2)
        mu = x.mean(axis=0)
        c = x - mu
        m0 = c.T @ c
        d = candidate_agents[:, None] - candidate_agents[None]
        w = np.exp(-np.sum(d * d, axis=2) / (2.0 * 0.8 ** 2))
        np.fill_diagonal(w, 0.0)
        degree = w.sum(axis=1)
        lap = np.eye(3) - w / np.sqrt(degree[:, None] * degree[None, :])
        m1 = c.T @ lap @ c
        values = [mu, m0[UPPER_TRIANGLE]]
        if variant == "vector_signal_gsp":
            values.append(m1[UPPER_TRIANGLE])
        output.append(np.concatenate(values))
    output = np.asarray(output)
    return output - output[0]


class VectorSignalFeatureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        env = make_env("simple_spread", discrete_action=True)
        try:
            env.seed(9123)
            cls.obs = np.asarray(env.reset(), dtype=np.float32)
        finally:
            env.close()

    def test_shapes_noop_reference_and_torch(self):
        batch = np.stack([self.obs, self.obs + 0.01], axis=0)
        for variant, dim in (("vector_signal_geom", 9), ("vector_signal_gsp", 15)):
            features = compute_vector_signal_features_from_local_obs_batch(batch, variant)
            self.assertEqual(features.shape, (2, 3, 5, dim))
            np.testing.assert_array_equal(features[:, :, 0], 0.0)
            for agent in range(3):
                expected = reference_features(self.obs[agent], variant)
                np.testing.assert_allclose(features[0, agent], expected, rtol=2e-5, atol=2e-6)
            actual_torch = vector_signal_features_tensor(
                torch.as_tensor(self.obs), variant
            ).detach().numpy()
            np.testing.assert_allclose(actual_torch, features[0], rtol=3e-5, atol=3e-6)

    def test_translation_rotation_and_extreme_finite(self):
        base = self.obs[0].copy()
        translated = base.copy()
        translated[2:4] += [123.0, -51.0]  # relative fields remain unchanged
        np.testing.assert_allclose(
            compute_vector_signal_features_from_local_obs_batch(base),
            compute_vector_signal_features_from_local_obs_batch(translated), atol=1e-7,
        )
        theta = 0.71
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]], dtype=np.float32)
        rotated = base.copy()
        for start, stop in ((0, 2), (2, 4), (4, 10), (10, 14), (14, 18)):
            rotated[start:stop] = (base[start:stop].reshape(-1, 2) @ rotation.T).reshape(-1)
        np.testing.assert_allclose(
            compute_vector_signal_features_from_local_obs_batch(base),
            compute_vector_signal_features_from_local_obs_batch(
                rotated, action_deltas=np.asarray([[0, 0], rotation[:, 0], -rotation[:, 0],
                                                   rotation[:, 1], -rotation[:, 1]])
            ), rtol=2e-5, atol=2e-6,
        )
        extreme = np.tile(base, (4, 1))
        extreme[:, :14] *= np.asarray([1.0, 10.0, 1e2, 1e3])[:, None]
        self.assertTrue(np.isfinite(compute_vector_signal_features_from_local_obs_batch(extreme)).all())

    def test_landmark_and_nonfocal_permutation_consistency(self):
        base = self.obs[0].copy()
        perm = np.array([2, 0, 1])
        changed = base.copy()
        changed[4:10] = base[4:10].reshape(3, 2)[perm].reshape(-1)
        f0 = compute_vector_signal_features_from_local_obs_batch(base)[0]
        fp = compute_vector_signal_features_from_local_obs_batch(changed)[0]
        # Rebuild full symmetric matrices to express vech equivariance clearly.
        np.testing.assert_allclose(fp[:, :3], f0[:, :3][:, perm], atol=2e-6)
        for offset in (3, 9):
            for action in range(5):
                matrix = np.zeros((3, 3)); matrix[UPPER_TRIANGLE] = f0[action, offset:offset+6]
                matrix = matrix + np.triu(matrix, 1).T
                expected = matrix[np.ix_(perm, perm)][UPPER_TRIANGLE]
                np.testing.assert_allclose(fp[action, offset:offset+6], expected, atol=3e-6)
        swapped = base.copy()
        swapped[10:14] = base[10:14].reshape(2, 2)[::-1].reshape(-1)
        np.testing.assert_allclose(
            compute_vector_signal_features_from_local_obs_batch(base),
            compute_vector_signal_features_from_local_obs_batch(swapped), atol=3e-6,
        )

    def test_only_focal_candidate_moves(self):
        agents, landmarks = candidate_geometry_from_local_obs_batch(self.obs[0])
        np.testing.assert_array_equal(agents[0, :, 1:], np.broadcast_to(agents[0, 0, 1:], (5, 2, 2)))
        np.testing.assert_array_equal(landmarks[0], self.obs[0, 4:10].reshape(3, 2))


class VectorSignalBoundaryTest(unittest.TestCase):
    def test_actor_only_raw_replay_and_critic(self):
        env = DummyVecEnv([lambda: make_env("simple_spread", discrete_action=True)])
        try:
            obs = env.reset()
            for actor_model, dim in (("vector_signal_geom", 9), ("vector_signal_gsp", 15)):
                maddpg = MADDPG.init_from_env(env, actor_aug=actor_model)
                self.assertEqual(maddpg.agents[0].critic.fc1.in_features, 69)
                self.assertEqual(maddpg.agents[0].target_critic.fc1.in_features, 69)
                policy = maddpg.agents[0].policy
                policy.eval()
                self.assertEqual(policy.feature_dim, dim)
                self.assertEqual(maddpg.actor_aug, actor_model)
                logits = policy(torch.as_tensor(obs[:, 0], dtype=torch.float32))
                self.assertEqual(tuple(logits.shape), (1, 5))
                self.assertEqual(policy.last_raw_input_shape, (1, 18))
                self.assertEqual(policy.last_action_feature_shape, (1, 5, dim))
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "model.pt"
                    maddpg.save(path)
                    loaded = MADDPG.init_from_save(str(path))
                    self.assertEqual(loaded.actor_model, actor_model)
            buffer = ReplayBuffer(8, 3, [18, 18, 18], [5, 5, 5])
            actions = [np.eye(5, dtype=np.float32)[[0]] for _ in range(3)]
            buffer.push(obs, actions, np.zeros((1, 3)), obs, np.zeros((1, 3)))
            sample = buffer.sample(1, to_gpu=False)
            self.assertTrue(all(tuple(item.shape) == (1, 18) for item in sample[0]))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
