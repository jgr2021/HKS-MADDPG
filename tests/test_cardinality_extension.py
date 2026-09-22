import unittest

import numpy as np
import torch

from experiments.cardinality_coordination import (
    env_id_for_cardinality,
    observation_dim,
    teacher_actions,
)
from utils.make_env import make_env
from utils.networks import EquivariantMatchingSafetyPolicy8


class CardinalityExtensionTests(unittest.TestCase):
    def test_nxn_environment_boundaries(self):
        for n_agents in (3, 4, 5, 6):
            with self.subTest(n_agents=n_agents):
                env = make_env(
                    env_id_for_cardinality(n_agents), discrete_action=True
                )
                try:
                    obs = np.asarray(env.reset(), dtype=np.float32)
                    self.assertEqual(
                        obs.shape, (n_agents, observation_dim(n_agents))
                    )
                    self.assertEqual(len(env.action_space), n_agents)
                    self.assertTrue(all(space.n == 5 for space in env.action_space))
                finally:
                    env.close()

    def test_generic_actor_shapes_and_matching(self):
        for n_agents in (3, 4, 5, 6):
            with self.subTest(n_agents=n_agents):
                policy = EquivariantMatchingSafetyPolicy8(
                    observation_dim(n_agents), 5, agent_index=n_agents - 1
                )
                values = torch.randn(7, observation_dim(n_agents))
                logits = policy(values)
                self.assertEqual(tuple(logits.shape), (7, 5))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertEqual(
                    policy.last_matching_shape, (7, 5, n_agents, n_agents)
                )
                self.assertLess(policy.last_matching_row_error, 0.2)
                self.assertLess(policy.last_matching_column_error, 1e-5)

    def test_teacher_actions_are_local_and_valid(self):
        for n_agents in (3, 4, 5, 6):
            with self.subTest(n_agents=n_agents):
                env = make_env(
                    env_id_for_cardinality(n_agents), discrete_action=True
                )
                try:
                    env.seed(9100 + n_agents)
                    obs = np.asarray(env.reset(), dtype=np.float32)
                    actions = teacher_actions(obs, n_agents)
                    self.assertEqual(len(actions), n_agents)
                    for action in actions:
                        self.assertEqual(action.shape, (5,))
                        self.assertAlmostEqual(float(action.sum()), 1.0)
                        self.assertTrue(np.all((action == 0.0) | (action == 1.0)))
                    next_obs, _, _, _ = env.step(actions)
                    self.assertEqual(
                        np.asarray(next_obs).shape,
                        (n_agents, observation_dim(n_agents)),
                    )
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
