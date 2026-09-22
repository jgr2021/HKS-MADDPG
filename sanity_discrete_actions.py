import inspect
import numpy as np
import torch
from torch.autograd import Variable

from utils.make_env import make_env
from utils.env_wrappers import DummyVecEnv
from utils.agents import DDPGAgent
from algorithms.maddpg import MADDPG


def make_single_env():
    # 与 main.py 中 --discrete_action 的环境保持一致
    return make_env("simple_spread", discrete_action=True)


# 1. 用 DummyVecEnv 包装，和正式训练的单环境路径一致
env = DummyVecEnv([make_single_env])

print("Agent types:", env.agent_types)
print("Action spaces:", env.action_space)
print("Observation spaces:", env.observation_space)

# 2. 静态检查：确认 agents.py 已恢复 Gumbel-Softmax rollout
step_source = inspect.getsource(DDPGAgent.step)

assert "gumbel_softmax(action, hard=True)" in step_source, (
    "\nFAIL: DDPGAgent.step() 里没有找到 "
    "`gumbel_softmax(action, hard=True)`。\n"
    "请检查 utils/agents.py 是否已按要求修改。"
)

assert "onehot_from_logits(\n                    action,\n                    eps=self.exploration" not in step_source, (
    "\nFAIL: DDPGAgent.step() 仍然包含 ε-greedy rollout。\n"
    "请把离散动作 explore=True 分支改回 Gumbel-Softmax。"
)

print("\nPASS 1: agents.py 已检测到 Gumbel-Softmax rollout。")

# 3. 初始化 MADDPG，确认环境接口与正式训练一致
maddpg = MADDPG.init_from_env(env)
maddpg.prep_rollouts(device='cpu')
assert maddpg.nagents == 3, f"Expected 3 agents, got {maddpg.nagents}"
assert maddpg.discrete_action is True, "Expected discrete_action=True"

# 4. reset 后的 obs 形状是 [n_env=1, n_agent=3, ...]
obs = env.reset()

torch_obs = [
    Variable(
        torch.tensor(obs[:, i], dtype=torch.float32),
        requires_grad=False
    )
    for i in range(maddpg.nagents)
]

# explore=True：应走 Gumbel-Softmax hard one-hot
actions = maddpg.step(torch_obs, explore=True)

print("\nSampled rollout actions:")

for i, action in enumerate(actions):
    action_np = action.detach().cpu().numpy()

    print(f"Agent {i}: {action_np}")

    # 一个 parallel env，所以 shape 是 (1, 5)
    assert action_np.shape == (1, 5), (
        f"Agent {i}: expected shape (1, 5), got {action_np.shape}"
    )

    # 每行必须恰好是一个 one-hot 向量
    assert np.all(np.isin(action_np, [0.0, 1.0])), (
        f"Agent {i}: action is not binary one-hot: {action_np}"
    )

    assert np.allclose(action_np.sum(axis=1), 1.0), (
        f"Agent {i}: action does not sum to 1: {action_np}"
    )

print("\nPASS 2: rollout actions are 5-way hard one-hot vectors.")
print("PASS 3: environment/wrapper/MADDPG initialization matches training.")
print("\nALL SANITY CHECKS PASSED.")