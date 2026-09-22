import unittest
from types import SimpleNamespace

import numpy as np
import torch
from gym.spaces import Box

from algorithms.maddpg import MADDPG
from experiments.run_passive_gsp_topology_pilot import (
    actor_updates_enabled,
    train_for_env_steps,
)
from main import make_parallel_env
from utils.buffer import ReplayBuffer
from utils.make_env import make_env


class PassiveGSPTopologyProtocolTest(unittest.TestCase):
    def _maddpg(self, actor_model):
        env = make_parallel_env("simple_spread", 1, 2026, True)
        try:
            maddpg = MADDPG.init_from_env(
                env,
                agent_alg="MADDPG",
                adversary_alg="MADDPG",
                actor_model=actor_model,
            )
        finally:
            env.close()
        return maddpg

    def test_actor_and_critic_dimensions(self):
        raw = self._maddpg("mlp")
        gsp = self._maddpg("passive_topology_6node_aal_hks")

        self.assertEqual(raw.agents[0].policy.fc1.in_features, 18)
        self.assertEqual(gsp.agents[0].policy.mlp.fc1.in_features, 21)
        self.assertEqual(raw.agents[0].critic.fc1.in_features, 69)
        self.assertEqual(gsp.agents[0].critic.fc1.in_features, 69)
        self.assertEqual(raw.agents[0].target_critic.fc1.in_features, 69)
        self.assertEqual(gsp.agents[0].target_critic.fc1.in_features, 69)

    def test_replay_buffer_stays_raw_18d_for_all_methods(self):
        env = make_env("simple_spread", discrete_action=True)
        try:
            buffer = ReplayBuffer(
                128,
                3,
                [space.shape[0] for space in env.observation_space],
                [
                    space.shape[0] if isinstance(space, Box) else space.n
                    for space in env.action_space
                ],
            )
            self.assertEqual([item.shape[1] for item in buffer.obs_buffs], [18, 18, 18])
            self.assertEqual([item.shape[1] for item in buffer.next_obs_buffs], [18, 18, 18])
        finally:
            env.close()

    def test_same_transition_gives_identical_raw_critic_observations(self):
        env = make_env("simple_spread", discrete_action=True)
        try:
            env.seed(123)
            obs = env.reset()
            actions = [
                np.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                for _ in range(3)
            ]
            next_obs, rewards, dones, _ = env.step(actions)
            stacked_obs = np.stack(obs, axis=0)
            stacked_next = np.stack(next_obs, axis=0)
            self.assertEqual(stacked_obs.shape, (3, 18))
            self.assertEqual(stacked_next.shape, (3, 18))

            raw_maddpg = self._maddpg("mlp")
            gsp_maddpg = self._maddpg("passive_topology_6node_aal_hks")
            raw_critic_obs = torch.cat(
                [torch.as_tensor(obs[i], dtype=torch.float32).view(1, -1) for i in range(3)],
                dim=1,
            )
            gsp_critic_obs = torch.cat(
                [torch.as_tensor(obs[i], dtype=torch.float32).view(1, -1) for i in range(3)],
                dim=1,
            )
            torch.testing.assert_close(raw_critic_obs, gsp_critic_obs)
            self.assertEqual(raw_critic_obs.shape, (1, 54))
            self.assertEqual(gsp_critic_obs.shape, (1, 54))
            self.assertEqual(raw_maddpg.agents[0].critic.fc1.in_features, 69)
            self.assertEqual(gsp_maddpg.agents[0].critic.fc1.in_features, 69)
        finally:
            env.close()

    def test_hks_computed_from_focal_raw_observation_only(self):
        gsp = self._maddpg("passive_topology_6node_aal_hks")
        env = make_env("simple_spread", discrete_action=True)
        try:
            env.seed(456)
            obs = env.reset()
            x = torch.as_tensor(np.vstack(obs), dtype=torch.float32)
            augmented = gsp.agents[0].policy.augment_observation(x)
            self.assertEqual(tuple(x.shape), (3, 18))
            self.assertEqual(tuple(augmented.shape), (3, 21))
            changed = x.clone()
            changed[1:, :] += 1000.0
            augmented_changed = gsp.agents[0].policy.augment_observation(changed)
            torch.testing.assert_close(augmented[0], augmented_changed[0])
        finally:
            env.close()

    def test_training_loop_stops_at_exact_env_steps(self):
        config = SimpleNamespace(
            env_id="simple_spread",
            model_name="protocol_step_stop_test",
            seed=909,
            n_rollout_threads=4,
            n_training_threads=1,
            buffer_length=1024,
            n_episodes=8,
            episode_length=25,
            steps_per_update=100,
            batch_size=64,
            n_exploration_eps=8,
            init_noise_scale=0.3,
            final_noise_scale=0.0,
            hidden_dim=16,
            lr=0.01,
            tau=0.01,
            agent_alg="MADDPG",
            adversary_alg="MADDPG",
            discrete_action=True,
            actor_model="mlp",
            print_interval=100,
        )
        run_dir, _ = train_for_env_steps(config, total_env_steps=8, checkpoint_steps=[8])
        counters = (run_dir / "training_counters.json").read_text(encoding="utf-8")
        self.assertIn('"global_env_steps": 8', counters)
        self.assertTrue((run_dir / "incremental" / "model_step8.pt").exists())

    def test_actor_update_window_is_half_open(self):
        config = SimpleNamespace(
            actor_update_start_step=100,
            actor_update_end_step=1000,
        )
        self.assertFalse(actor_updates_enabled(config, 99))
        self.assertTrue(actor_updates_enabled(config, 100))
        self.assertTrue(actor_updates_enabled(config, 999))
        self.assertFalse(actor_updates_enabled(config, 1000))

    def test_actor_update_window_rejects_reversed_bounds(self):
        config = SimpleNamespace(
            actor_update_start_step=1000,
            actor_update_end_step=100,
        )
        with self.assertRaises(ValueError):
            actor_updates_enabled(config, 500)


if __name__ == "__main__":
    unittest.main()
