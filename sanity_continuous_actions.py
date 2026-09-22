import numpy as np
import torch
from torch.autograd import Variable

from utils.make_env import make_env
from utils.env_wrappers import DummyVecEnv
from algorithms.maddpg import MADDPG


def make_single_env():
    # 关键：不传 discrete_action=True
    return make_env("simple_spread", discrete_action=False)


env = DummyVecEnv([make_single_env])

print("Agent types:", env.agent_types)
print("Action spaces:", env.action_space)
print("Observation spaces:", env.observation_space)

maddpg = MADDPG.init_from_env(env)
maddpg.prep_rollouts(device="cpu")

assert maddpg.nagents == 3
assert maddpg.discrete_action is False, (
    "Expected continuous actions, but MADDPG reports discrete_action=True."
)

for action_space in env.action_space:
    assert action_space.shape == (2,), (
        f"Expected Box(2), got {action_space}"
    )

obs = env.reset()

torch_obs = [
    Variable(
        torch.tensor(obs[:, i], dtype=torch.float32),
        requires_grad=False
    )
    for i in range(maddpg.nagents)
]

actions = maddpg.step(torch_obs, explore=True)

print("\nSampled continuous rollout actions:")

for i, action in enumerate(actions):
    action_np = action.detach().cpu().numpy()

    print(f"Agent {i}: {action_np}")

    assert action_np.shape == (1, 2), (
        f"Agent {i}: expected shape (1, 2), got {action_np.shape}"
    )
    assert np.all(action_np >= -1.0) and np.all(action_np <= 1.0), (
        f"Agent {i}: action is outside [-1, 1]: {action_np}"
    )

print("\nPASS: continuous Box(2) actions are correctly configured.")