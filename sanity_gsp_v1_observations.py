"""Stage-1 sanity check for the separate GSP-MADDPG v1 project.

This script does NOT train, save, or overwrite any model. It checks only that:
  1) ``simple_spread_gsp_v1`` keeps the original discrete action space;
  2) raw 18D observations are preserved as the first 18 entries;
  3) GSP-v1 appends finite 8D descriptors;
  4) global eigenvalue features agree across the three decentralized views;
  5) MADDPG initializes with actor input 26 and centralized critic input 93;
  6) rollout actions remain legal five-way hard one-hot vectors.
"""

import numpy as np
import torch
from torch.autograd import Variable

from algorithms.maddpg import MADDPG
from utils.env_wrappers import DummyVecEnv
from utils.gsp_features import (
    AUGMENTED_OBS_DIM,
    GSP_DESCRIPTOR_DIM,
    HKS_DIM,
    RAW_OBS_DIM,
    SPECTRAL_DIM,
)
from utils.make_env import make_env


SEED = 2040


def make_single_env():
    return make_env("simple_spread_gsp_v1", discrete_action=True)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Direct wrapper inspection.
    single_env = make_single_env()
    single_env.seed(SEED)
    observations = single_env.reset()

    print("=" * 72)
    print("GSP-MADDPG v1 Stage-1 sanity check")
    print("=" * 72)
    print("Action spaces:", single_env.action_space)
    print("Augmented observation spaces:", single_env.observation_space)

    assert len(observations) == 3
    assert single_env.last_raw_observations is not None

    spectra = []
    hks = []
    for agent_index, observation in enumerate(observations):
        raw = single_env.last_raw_observations[agent_index]
        observation = np.asarray(observation, dtype=np.float32)
        descriptor = observation[RAW_OBS_DIM:]
        eig = descriptor[:SPECTRAL_DIM]
        local_hks = descriptor[SPECTRAL_DIM:]

        assert raw.shape == (RAW_OBS_DIM,)
        assert observation.shape == (AUGMENTED_OBS_DIM,)
        assert descriptor.shape == (GSP_DESCRIPTOR_DIM,)
        assert np.allclose(observation[:RAW_OBS_DIM], raw, atol=1e-7)
        assert np.all(np.isfinite(descriptor))
        assert np.all(eig >= -1e-6) and np.all(eig <= 2.0 + 1e-6)
        assert local_hks.shape == (HKS_DIM,)

        spectra.append(eig)
        hks.append(local_hks)
        print(f"\nAgent A{agent_index}")
        print(f"  raw observation dimension: {raw.shape[0]}")
        print(f"  augmented dimension:       {observation.shape[0]}")
        print("  lambda_2..lambda_6:       ", np.round(eig, 6))
        print("  HKS(tau=0.5,1,2):         ", np.round(local_hks, 6))

    spectra = np.vstack(spectra)
    hks = np.vstack(hks)
    assert np.allclose(spectra, spectra[0], atol=1e-5), (
        "The global spectrum reconstructed from each actor's raw observation "
        "should agree up to numerical tolerance."
    )

    print("\nPASS 1: raw 18D observations are preserved exactly as the first 18 entries.")
    print("PASS 2: each actor receives a finite 8D GSP descriptor.")
    print("PASS 3: lambda_2..lambda_6 agree across decentralized reconstructions.")
    print("Pairwise HKS distances:")
    for i in range(3):
        for j in range(i + 1, 3):
            print(f"  ||HKS(A{i}) - HKS(A{j})||_2 = {np.linalg.norm(hks[i] - hks[j]):.6f}")

    # Same wrapper through the normal vectorized-training initialization path.
    vec_env = DummyVecEnv([make_single_env])
    maddpg = MADDPG.init_from_env(vec_env)
    maddpg.prep_rollouts(device="cpu")

    actor_input_dim = maddpg.agents[0].policy.fc1.in_features
    critic_input_dim = maddpg.agents[0].critic.fc1.in_features
    assert actor_input_dim == AUGMENTED_OBS_DIM
    assert critic_input_dim == 3 * AUGMENTED_OBS_DIM + 3 * 5

    vec_obs = vec_env.reset()
    torch_obs = [
        Variable(torch.tensor(vec_obs[:, agent_index], dtype=torch.float32), requires_grad=False)
        for agent_index in range(maddpg.nagents)
    ]
    actions = maddpg.step(torch_obs, explore=True)

    for agent_index, action in enumerate(actions):
        action_np = action.detach().cpu().numpy()
        assert action_np.shape == (1, 5)
        assert np.all(np.isin(action_np, [0.0, 1.0]))
        assert np.allclose(action_np.sum(axis=1), 1.0)
        print(f"Sample rollout action A{agent_index}: {action_np}")

    print("\nPASS 4: MADDPG actor input = 26D and centralized critic input = 93D.")
    print("PASS 5: rollout actions remain valid 5-way hard one-hot vectors.")
    print("\nALL STAGE-1 GSP SANITY CHECKS PASSED.")

    single_env.close()
    vec_env.close()


if __name__ == "__main__":
    main()
