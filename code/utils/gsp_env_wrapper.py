"""Observation-only environment wrapper for GSP-MADDPG v1.

The wrapper leaves the underlying MPE task untouched. It changes only what
``reset`` and ``step`` return to the learning code:

    original 18D observation -> [original 18D, 8D graph-spectral descriptor]

Reward, actions, dynamics, landmarks, collision behavior, episode horizon, and
all environment callbacks remain exactly those of ``simple_spread``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from gym.spaces import Box

from utils.gsp_features import (
    RAW_OBS_DIM,
    augment_joint_observations,
    descriptor_dim_for_mode,
)


class GSPObservationWrapper:
    """Wrap a three-agent MPE simple_spread environment with 26D observations."""

    def __init__(self, env, mode="full"):
        self.env = env
        self.mode = mode
        self.raw_observation_space = env.observation_space
        self.action_space = env.action_space
        self.n = env.n
        augmented_obs_dim = RAW_OBS_DIM + descriptor_dim_for_mode(mode)

        if len(self.raw_observation_space) != 3:
            raise ValueError(
                "GSP-MADDPG v1 is intentionally fixed to the current "
                "3-agent simple_spread setup."
            )
        for index, space in enumerate(self.raw_observation_space):
            if not isinstance(space, Box) or space.shape != (RAW_OBS_DIM,):
                raise ValueError(
                    "Expected an 18D Box observation for every agent; "
                    f"agent {index} has {space}."
                )

        # The raw MPE spaces are unbounded Box spaces. Keep that convention.
        self.observation_space = [
            Box(
                low=-np.inf,
                high=np.inf,
                shape=(augmented_obs_dim,),
                dtype=np.float32,
            )
            for _ in range(self.n)
        ]

        # Kept only for transparent debugging / sanity tests; the learning code
        # uses the returned augmented observations, never these fields directly.
        self.last_raw_observations = None
        self.last_augmented_observations = None

    def _augment(self, raw_observations: Sequence[Sequence[float]]):
        raw = [np.asarray(obs, dtype=np.float32).reshape(-1).copy() for obs in raw_observations]
        augmented = augment_joint_observations(raw, mode=self.mode)
        self.last_raw_observations = raw
        self.last_augmented_observations = augmented
        return augmented

    def reset(self):
        return self._augment(self.env.reset())

    def step(self, actions):
        raw_observations, rewards, dones, infos = self.env.step(actions)
        return self._augment(raw_observations), rewards, dones, infos

    def seed(self, seed=None):
        return self.env.seed(seed)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        """Delegate world, agents, and other normal MPE attributes to env."""
        return getattr(self.env, name)
