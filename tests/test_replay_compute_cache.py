import unittest
from unittest.mock import Mock

import numpy as np
import torch

from utils.buffer import ReplayBuffer
from utils.replay_compute_cache import CachedRewardStatsReplayBuffer


class ReplayCacheTests(unittest.TestCase):
    def compare(self, original, cached, gpu=False):
        for normalized in (False, True, True):
            for seed in (1, 37):
                np.random.seed(seed)
                expected = original.sample(min(8, len(original)), to_gpu=gpu, norm_rews=normalized)
                before = np.random.get_state()
                np.random.seed(seed)
                actual = cached.sample(min(8, len(cached)), to_gpu=gpu, norm_rews=normalized)
                after = np.random.get_state()
                self.assertEqual(before[0], after[0])
                np.testing.assert_array_equal(before[1], after[1])
                self.assertEqual(before[2:], after[2:])
                for family_a, family_b in zip(expected, actual):
                    for a, b in zip(family_a, family_b):
                        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)

    def test_exact_samples_rng_and_invalidation_across_wrap(self):
        original = ReplayBuffer(17, 3, [18]*3, [5]*3)
        cached = CachedRewardStatsReplayBuffer(17, 3, [18]*3, [5]*3)
        rng = np.random.RandomState(19)
        for _ in range(12):
            values = (rng.randn(4, 3, 18), [rng.randn(4, 5) for _ in range(3)],
                      rng.randn(4, 3), rng.randn(4, 3, 18), rng.randint(0, 2, (4, 3)))
            original.push(*values)
            cached.push(*values)
            self.assertIsNone(cached._reward_stats)
            self.compare(original, cached)

    def test_unscaled_sampling_does_not_compute_stats(self):
        cached = CachedRewardStatsReplayBuffer(8, 3, [18]*3, [5]*3)
        cached.filled_i = 8
        cached.sample(4, norm_rews=False)
        self.assertIsNone(cached._reward_stats)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_cast_exact(self):
        original = ReplayBuffer(64, 3, [18]*3, [5]*3)
        cached = CachedRewardStatsReplayBuffer(64, 3, [18]*3, [5]*3)
        for buffer in (original, cached):
            buffer.filled_i = 64
            for i in range(3):
                buffer.rew_buffs[i][:] = np.arange(64) + i
        self.compare(original, cached, gpu=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_paired_cuda_updates_and_optimizer_states(self):
        from algorithms.maddpg import MADDPG
        from main import make_parallel_env
        from utils.exploration_topology_policies import register_policies

        register_policies()
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        env = make_parallel_env("simple_spread", 1, 1, True)
        try:
            for name in ("mlp", "explore_hks_6al"):
                buffers = (ReplayBuffer(4096, 3, [18]*3, [5]*3),
                           CachedRewardStatsReplayBuffer(4096, 3, [18]*3, [5]*3))
                rng = np.random.RandomState(19)
                models = []
                for _ in range(2):
                    torch.manual_seed(31)
                    model = MADDPG.init_from_env(env, actor_model=name)
                    model.prep_training(device="gpu")
                    models.append(model)
                for cycle in range(4):
                    count = 2048 if cycle == 0 else 4
                    values = (rng.randn(count, 3, 18),
                              [np.eye(5)[rng.randint(5, size=count)] for _ in range(3)],
                              rng.randn(count, 3), rng.randn(count, 3, 18),
                              rng.randint(0, 2, (count, 3)))
                    for buffer in buffers:
                        buffer.push(*values)
                    for agent_i in range(3):
                        for model, buffer in zip(models, buffers):
                            np.random.seed(100 + 3 * cycle + agent_i)
                            sample = buffer.sample(1024, to_gpu=True)
                            torch.manual_seed(200 + 3 * cycle + agent_i)
                            model.update(sample, agent_i, logger=Mock())
                    for model in models:
                        model.update_all_targets()
                    for left, right in zip(models[0].agents, models[1].agents):
                        for component in ("policy", "critic", "target_policy", "target_critic",
                                          "policy_optimizer", "critic_optimizer"):
                            torch.testing.assert_close(
                                getattr(left, component).state_dict(),
                                getattr(right, component).state_dict(), rtol=0, atol=0)
        finally:
            env.close()
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
