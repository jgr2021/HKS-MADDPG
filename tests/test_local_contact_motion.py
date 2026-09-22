import unittest

import numpy as np

from experiments.diagnose_local_contact_motion import assert_environment_supported, branch_state
from experiments.probe_action_value_representations import snapshot_world, restore_world
from utils.active_gsp_v3_features import ACTION_TO_CONTROL_FLOAT32, predict_self_position_after_action
from utils.local_contact_motion import MODES, observable_contact_displacements, predict_local_candidate_positions
from utils.make_env import make_env


class LocalContactMotionTest(unittest.TestCase):
    def setUp(self):
        self.env = make_env("simple_spread", discrete_action=True)
        assert_environment_supported(self.env)

    def tearDown(self):
        self.env.close()

    def observations(self):
        return [self.env._get_obs(agent) for agent in self.env.world.agents]

    def test_batch_shapes_float32_and_finite_overlap_convention(self):
        for batch in (1, 12, 64):
            raw = np.zeros((batch, 18), dtype=np.float32)
            for distance in (0, 1e-8, 1e4):
                raw[:, 10:14] = distance
                for mode in MODES:
                    positions = predict_local_candidate_positions(raw, mode)
                    self.assertEqual(positions.shape, (batch, 5, 3, 2))
                    self.assertEqual(positions.dtype, np.float32)
                    self.assertTrue(np.isfinite(positions).all())

    def test_no_absolute_or_landmark_or_other_observation_dependency(self):
        raw = np.asarray(self.env.reset(), dtype=np.float32)
        original = predict_local_candidate_positions(raw, "known_contact_graph")
        changed = raw.copy()
        changed[:, 2:10] += 30
        changed[:, 14:18] += 50
        np.testing.assert_array_equal(original, predict_local_candidate_positions(changed, "known_contact_graph"))
        changed = raw.copy()
        changed[1:] += 100
        np.testing.assert_array_equal(original[:1], predict_local_candidate_positions(changed, "known_contact_graph")[:1])
        np.testing.assert_array_equal(original[:1], predict_local_candidate_positions(raw[:1], "known_contact_graph"))

    def test_legacy_matches_original_reference_and_noop_relative_actions(self):
        raw = np.asarray(self.env.reset(), dtype=np.float32)
        raw[:, :2] = [[.2, -.4], [.1, .3], [-.5, .2]]
        legacy = predict_local_candidate_positions(raw)
        for agent in range(3):
            for action in range(5):
                expected = predict_self_position_after_action(raw[agent], action) - raw[agent, 2:4]
                np.testing.assert_allclose(legacy[agent, action, 0], expected, rtol=1e-6, atol=1e-7)
        for mode in MODES:
            predicted = predict_local_candidate_positions(raw, mode)
            delta = predicted[:, :, 0] - predicted[:, :1, 0]
            np.testing.assert_allclose(delta, np.broadcast_to(.05 * ACTION_TO_CONTROL_FLOAT32, delta.shape), rtol=0, atol=1e-7)

    def test_contact_predictions_match_actual_focal_motion_for_all_actions(self):
        for distance in (.001, .1, .29, .3, .301, .6):
            self.env.reset()
            for agent, position, velocity in zip(self.env.world.agents,
                    ([0, 0], [distance, 0], [-1, 1]), ([.3, -.2], [-.8, .4], [.2, .6])):
                agent.state.p_pos = np.asarray(position, dtype=float)
                agent.state.p_vel = np.asarray(velocity, dtype=float)
            cases, expected = branch_state(self.env, self.observations(), [np.eye(5)[value] for value in (1, 3, 4)])
            for case in cases:
                np.testing.assert_allclose(case["predicted"][1:, :, 0],
                                           np.broadcast_to(case["actual"][:, 0], (2, 5, 2)), rtol=2e-5, atol=2e-6)
            self.env.step([np.eye(5)[value] for value in (1, 3, 4)])
            np.testing.assert_allclose([agent.state.p_pos for agent in self.env.world.agents], expected, rtol=0, atol=1e-12)

    def test_hidden_velocity_cannot_be_inferred_but_does_not_affect_focal_motion(self):
        self.env.reset()
        for agent, position in zip(self.env.world.agents, ([0, 0], [.2, 0], [-1, 1])):
            agent.state.p_pos = np.asarray(position, dtype=float)
            agent.state.p_vel = np.zeros(2)
        initial = snapshot_world(self.env)
        raw_a = self.observations()[0].copy()
        self.env.step([np.eye(5)[value] for value in (1, 0, 0)])
        next_a = np.asarray([agent.state.p_pos for agent in self.env.world.agents]).copy()
        restore_world(self.env, initial)
        self.env.world.agents[1].state.p_vel = np.asarray([.7, -.3])
        raw_b = self.observations()[0].copy()
        self.env.step([np.eye(5)[value] for value in (1, 4, 0)])
        next_b = np.asarray([agent.state.p_pos for agent in self.env.world.agents]).copy()
        np.testing.assert_array_equal(raw_a, raw_b)
        np.testing.assert_array_equal(next_a[0], next_b[0])
        self.assertGreater(np.linalg.norm(next_a[1] - next_b[1]), .01)
        np.testing.assert_array_equal(predict_local_candidate_positions(raw_a[None], "known_contact_graph"),
                                      predict_local_candidate_positions(raw_b[None], "known_contact_graph"))

    def test_rotation_and_reflection_equivariance(self):
        raw = np.asarray(self.env.reset(), dtype=np.float32)
        raw[:, :2] = [.2, -.4]
        original = predict_local_candidate_positions(raw, "known_contact_graph")
        for matrix in (np.asarray([[0, -1], [1, 0]], dtype=np.float32),
                       np.asarray([[-1, 0], [0, 1]], dtype=np.float32)):
            changed = raw.copy()
            changed[:, :14] = (changed[:, :14].reshape(-1, 7, 2) @ matrix.T).reshape(-1, 14)
            transformed_controls = ACTION_TO_CONTROL_FLOAT32 @ matrix.T
            permutation = ((transformed_controls[:, None] - ACTION_TO_CONTROL_FLOAT32[None]) ** 2).sum(-1).argmin(-1)
            transformed = predict_local_candidate_positions(changed, "known_contact_graph")[:, permutation]
            np.testing.assert_allclose(transformed, original @ matrix.T, rtol=1e-5, atol=2e-7)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            predict_local_candidate_positions(np.zeros(18))
        with self.assertRaises(ValueError):
            observable_contact_displacements(np.full((1, 18), np.nan))


if __name__ == "__main__":
    unittest.main()
