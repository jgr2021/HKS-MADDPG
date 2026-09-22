"""Compute-equivalent reward statistics cache; original sampling RNG is retained."""

import numpy as np
from torch import Tensor
from torch.autograd import Variable

from utils.buffer import ReplayBuffer


class CachedRewardStatsReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reward_stats = None

    def push(self, *args, **kwargs):
        self._reward_stats = None
        return super().push(*args, **kwargs)

    def sample(self, N, to_gpu=False, norm_rews=True):
        # Keep the exact legacy choice call and RNG consumption for comparability.
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            if self._reward_stats is None:
                self._reward_stats = [(array[:self.filled_i].mean(), array[:self.filled_i].std())
                                      for array in self.rew_buffs]
            ret_rews = [cast((self.rew_buffs[i][inds] - self._reward_stats[i][0]) / self._reward_stats[i][1])
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(array[inds]) for array in self.obs_buffs],
                [cast(array[inds]) for array in self.ac_buffs], ret_rews,
                [cast(array[inds]) for array in self.next_obs_buffs],
                [cast(array[inds]) for array in self.done_buffs])
