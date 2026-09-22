import unittest
import numpy as np
import torch
from algorithms.ahks_mappo import MAPPO, generalized_advantage, ppo_update
from utils.ahks_environments import (SpreadEnv, ActiveSensingEnv, SensingConfig, kalman_update,
                                   assignment, navigation_controller, sensing_controller)


class EnvironmentTests(unittest.TestCase):
    def test_legacy_trajectory_and_rng_isolation(self):
        from utils.make_env import make_env
        from run_vector_signal_gsp_experiment import geometry, RADII
        direct = make_env('simple_spread', discrete_action=True)
        wrapped = SpreadEnv()
        try:
            np.random.seed(824)
            expected = np.asarray(direct.reset())
            state = np.random.get_state()
            observed = wrapped.reset(824)
            np.testing.assert_allclose(observed, expected, atol=1e-7)
            after = np.random.get_state()
            np.testing.assert_array_equal(after[1], state[1])
            self.assertEqual(after[2:], state[2:])
            total = 0.
            for i in range(25):
                actions = np.array([i % 5, (i+1) % 5, (i+2) % 5])
                expected, rewards, _, _ = direct.step(np.eye(5)[actions].copy())
                observed, reward, done, info = wrapped.step(actions)
                np.testing.assert_allclose(observed, expected, atol=1e-7)
                self.assertAlmostEqual(reward, np.mean(rewards))
                total += reward
                self.assertEqual(done, i == 24)
            d, sep, collisions, cost = geometry(direct)
            self.assertAlmostEqual(info['metrics']['return'], total)
            self.assertAlmostEqual(info['metrics']['hungarian_assignment_distance'], cost)
            self.assertAlmostEqual(info['metrics']['coverage_radius_auc'],
                np.trapz([(d.min(0) < r).mean() for r in RADII], RADII)/.25)
            self.assertTrue(info['truncated'])
            self.assertFalse(info['terminated'])
        finally:
            direct.close()
            wrapped.close()

    def test_action_directions(self):
        env = SpreadEnv()
        try:
            for action, axis, sign in ((1, 0, 1), (2, 0, -1), (3, 1, 1), (4, 1, -1)):
                env.reset(9)
                for agent, position in zip(env.env.world.agents, ([0, 0], [10, 10], [-10, -10])):
                    agent.state.p_pos = np.array(position, dtype=float)
                    agent.state.p_vel = np.zeros(2)
                obs, _, _, _ = env.step(np.array([action, 0, 0]))
                self.assertGreater(sign*obs[0, 2+axis], 0)
        finally:
            env.close()

    def test_filter_against_information_form(self):
        p = np.diag([.2, .2, .02, .02])
        mean = np.zeros(4)
        noise = .01*np.eye(2)
        for _ in range(2):
            mean, p = kalman_update(mean, p, np.array([1., -1.]), noise)
        np.testing.assert_allclose(p.diagonal()[:2], 1/(1/.2+2/.01))
        np.testing.assert_allclose(mean[:2], np.array([1, -1])*200/205)
        self.assertTrue(np.linalg.eigvalsh(p).min() > 0)

    def test_belief_interface_and_paired_noise(self):
        first, second = ActiveSensingEnv(), ActiveSensingEnv()
        a, b = first.reset(34), second.reset(34)
        np.testing.assert_array_equal(a, b)
        truth = first.truth.copy()
        first.truth += 10
        np.testing.assert_array_equal(first.observation(), a)
        first.truth = truth
        for _ in range(100):
            a, _, done, info = first.step(np.array([0, 1, 2]))
            b, _, _, _ = second.step(np.array([4, 3, 1]))
            np.testing.assert_array_equal(first.truth, second.truth)
            self.assertTrue(np.isfinite(a).all())
            self.assertTrue(np.linalg.eigvalsh(first.covariance).min() > 0)
        self.assertTrue(done and info['terminated'] and not info['truncated'])
        self.assertEqual(a.shape, (3, 79))
        self.assertEqual(a[0, -1], 0)
        with self.assertRaises(RuntimeError):
            first.step(np.array([0, 0, 0]))

    def test_assignment_and_controllers(self):
        cost = np.array([[10, 1, 5], [1, 5, 10], [5, 10, 1]])
        np.testing.assert_array_equal(assignment(cost), [1, 0, 2])
        env = ActiveSensingEnv()
        obs = env.reset(1)
        for controller in ('nearest', 'uncertainty', 'information'):
            actions = sensing_controller(obs, controller)
            self.assertEqual(actions.shape, (3,))
            self.assertTrue(((actions >= 0) & (actions < 5)).all())
        for controller in ('nearest', 'hungarian'):
            self.assertEqual(navigation_controller(obs[:, :18], controller).shape, (3,))


class LearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(21)

    def test_gae_terminal_truncation_and_rollout_cut(self):
        reward = np.array([[1., 1.], [100., 100.]], dtype=np.float32)
        value = np.array([[2., 2.], [3., 3.]], dtype=np.float32)
        next_value = np.array([[4., 4.], [5., 5.]], dtype=np.float32)
        terminated = np.array([[1., 0.], [0., 0.]], dtype=np.float32)
        ended = np.array([[1., 1.], [0., 0.]], dtype=np.float32)
        advantage, returns = generalized_advantage(reward, value, next_value, terminated, ended, .9, .8)
        np.testing.assert_allclose(advantage[0], [-1, 2.6], atol=1e-6)
        np.testing.assert_allclose(returns[1], [104.5, 104.5])

    def test_actor_only_descriptor_and_critic_dimensions(self):
        raw = torch.randn(8, 3, 79)
        counts = []
        for kind in ('raw', 'fixed_hks', 'adaptive_hks'):
            model = MAPPO(79, kind)
            self.assertEqual(model.critic[0].in_features, 237)
            self.assertEqual(model.actors[0].input_dim, 79 if kind == 'raw' else 82)
            changed = raw.clone()
            changed[:, 1] += 1
            torch.testing.assert_close(model.distributions(raw).logits[:, 0], model.distributions(changed).logits[:, 0])
            torch.testing.assert_close(model.actors[0].augment(raw[:, 0])[:, :79], raw[:, 0])
            counts.append(sum(p.numel() for p in model.actors.parameters()))
        self.assertEqual(counts[1], counts[2])
        self.assertEqual(counts[1]-counts[0], 3*3*64)

    def test_optimizer_updates_all_actors_and_critic(self):
        model = MAPPO(18, 'adaptive_hks')
        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        obs = torch.randn(24, 3, 18)
        action, logp, value = model.act(obs)
        before = [next(m.parameters()).detach().clone() for m in [*model.actors, model.critic]]
        advantage = torch.randn(24)
        stats = ppo_update(model, optimizer, [obs, action, logp, value, advantage, value+advantage], epochs=2, minibatch=12)
        self.assertTrue(all(np.isfinite(list(stats.values()))))
        for previous, module in zip(before, [*model.actors, model.critic]):
            self.assertFalse(torch.equal(previous, next(module.parameters())))


if __name__ == '__main__':
    unittest.main()
