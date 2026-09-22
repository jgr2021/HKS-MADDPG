"""Feed-forward MAPPO: independent actors and a centralized team-value critic.

Local implementation of clipped PPO/GAE, not a claim to reproduce the official
MAPPO implementation's complete hyperparameter/normalization configuration.
Reference: https://arxiv.org/abs/2103.01955
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from utils.catalog76_policies import features


def mlp(inputs, outputs, hidden=64, output_gain=1.):
    layers = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                           nn.Tanh(), nn.Linear(hidden, outputs))
    for layer in layers:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layers[-1].weight, gain=output_gain)
    return layers


class LocalActor(nn.Module):
    def __init__(self, obs_dim, descriptor, hidden=64):
        super().__init__()
        if descriptor not in ('raw', 'fixed_hks', 'adaptive_hks'):
            raise ValueError(descriptor)
        self.descriptor = descriptor
        self.obs_dim = obs_dim
        self.input_dim = obs_dim + (0 if descriptor == 'raw' else 3)
        self.network = mlp(self.input_dim, 5, hidden, .01)

    def augment(self, raw):
        if raw.shape[-1] != self.obs_dim:
            raise ValueError('Actor requires raw observation')
        if self.descriptor == 'raw':
            return raw
        with torch.no_grad():
            hks = features(raw[:, :18], {'kernel': 'adaptive'} if self.descriptor == 'adaptive_hks' else {})
        return torch.cat((raw, hks), -1)

    def forward(self, raw):
        return self.network(self.augment(raw))


class MAPPO(nn.Module):
    def __init__(self, obs_dim, descriptor='raw', hidden=64):
        super().__init__()
        self.obs_dim, self.descriptor, self.hidden = obs_dim, descriptor, hidden
        self.actors = nn.ModuleList([LocalActor(obs_dim, descriptor, hidden) for _ in range(3)])
        self.critic = mlp(3*obs_dim, 1, hidden)

    def distributions(self, obs):
        return Categorical(logits=torch.stack([actor(obs[:, i]) for i, actor in enumerate(self.actors)], 1))

    def value(self, obs):
        return self.critic(obs.flatten(1)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        distribution = self.distributions(obs)
        actions = distribution.logits.argmax(-1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), self.value(obs)


def generalized_advantage(rewards, values, next_values, terminated, episode_end, gamma=.95, lam=.95):
    """Bootstrap truncations, stop traces at all resets, bootstrap rollout cuts.

    Arrays are [time, env]. next_values are evaluated on pre-reset observations.
    terminated and episode_end deliberately differ for MPE time-limit resets.
    """
    advantages = np.zeros_like(rewards, dtype=np.float32)
    trace = np.zeros(rewards.shape[1], dtype=np.float32)
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma*(1-terminated[i])*next_values[i] - values[i]
        trace = delta + gamma*lam*(1-episode_end[i])*trace
        advantages[i] = trace
    return advantages, advantages + values


def ppo_update(model, optimizer, batch, *, epochs=10, minibatch=250, clip=.2,
               entropy_coef=.01, value_coef=.5, max_grad_norm=.5):
    obs, actions, old_logp, old_value, advantages, returns = batch
    advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    records = []
    for _ in range(epochs):
        permutation = torch.randperm(len(obs), device=obs.device)
        for index in permutation.split(minibatch):
            dist = model.distributions(obs[index])
            log_ratio = dist.log_prob(actions[index])-old_logp[index]
            ratio = log_ratio.exp()
            local_adv = advantages[index, None]
            policy_loss = -torch.minimum(ratio*local_adv, ratio.clamp(1-clip, 1+clip)*local_adv).mean()
            value = model.value(obs[index])
            clipped_value = old_value[index] + (value-old_value[index]).clamp(-clip, clip)
            value_loss = .5*torch.maximum((value-returns[index]).square(), (clipped_value-returns[index]).square()).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + value_coef*value_loss - entropy_coef*entropy
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite PPO loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
            optimizer.step()
            records.append([float(policy_loss.detach()), float(value_loss.detach()), float(entropy.detach()),
                            float(((ratio-1)-log_ratio).mean().detach()), float(norm)])
    return dict(zip(('policy_loss', 'value_loss', 'entropy', 'approx_kl', 'gradient_norm'),
                    np.mean(records, axis=0).tolist()))
