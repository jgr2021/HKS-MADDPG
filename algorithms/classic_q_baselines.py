import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F

from utils.misc import hard_update
from utils.networks import MLPNetwork


MSELoss = nn.MSELoss()


def _index_from_actions(action_tensor, num_actions):
    """Return integer action indices for all supported action representations."""
    if action_tensor.dim() == 1:
        return action_tensor.long()
    if action_tensor.shape[1] != num_actions:
        raise ValueError("Action tensor must be shape [batch] or one-hot [batch, num_actions]")
    return action_tensor.max(dim=1)[1].long()


class IndependentDQNAgent(object):
    """Classic independent DQN agent for one environment role."""

    def __init__(
        self,
        num_in_obs,
        num_out_acs,
        gamma=0.95,
        lr=0.001,
        hidden_dim=64,
        epsilon=1.0,
        device=None,
        use_double_q=False,
    ):
        self.device = torch.device(device or "cpu")
        self.n_actions = int(num_out_acs)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.use_double_q = bool(use_double_q)
        self.q_net = MLPNetwork(
            num_in_obs,
            self.n_actions,
            hidden_dim=hidden_dim,
            constrain_out=False,
            discrete_action=True,
        ).to(self.device)
        self.target_q_net = MLPNetwork(
            num_in_obs,
            self.n_actions,
            hidden_dim=hidden_dim,
            constrain_out=False,
            discrete_action=True,
        ).to(self.device)
        hard_update(self.target_q_net, self.q_net)
        self.optimizer = Adam(self.q_net.parameters(), lr=float(lr))

    def prep_rollouts(self):
        self.q_net.eval()
        self.target_q_net.eval()

    def prep_training(self):
        self.q_net.train()
        self.target_q_net.train()

    def set_exploration(self, epsilon):
        self.epsilon = float(epsilon)

    def step(self, obs_batch, explore=True):
        """Return a one-hot action tensor for a batch of observations."""
        with torch.no_grad():
            q_logits = self.q_net(obs_batch)
            if self.training_for_exploration(explore):
                random_draw = torch.rand(obs_batch.shape[0], device=self.device)
                random_actions = torch.randint(self.n_actions, size=(obs_batch.shape[0],), device=self.device)
                greedy_actions = q_logits.max(1)[1]
                chosen_actions = torch.where(random_draw < self.epsilon, random_actions, greedy_actions)
                return F.one_hot(chosen_actions, num_classes=self.n_actions).float()
            return F.one_hot(q_logits.max(1)[1], num_classes=self.n_actions).float()

    def training_for_exploration(self, explore):
        return explore and self.epsilon > 0.0

    def sync_target(self):
        hard_update(self.target_q_net, self.q_net)

    def update(self, sample, agent_i, logger=None, niter=None):
        obs, acs, rews, next_obs, dones = sample
        obs_i = obs[agent_i]
        acs_i = acs[agent_i]
        rews_i = rews[agent_i].view(-1, 1)
        next_obs_i = next_obs[agent_i]
        done_i = dones[agent_i].view(-1, 1)

        action_indices = _index_from_actions(acs_i, self.n_actions).view(-1, 1)
        q_sa = self.q_net(obs_i).gather(1, action_indices)

        with torch.no_grad():
            if self.use_double_q:
                next_action_indices = self.q_net(next_obs_i).max(1, keepdim=True)[1]
                next_q = self.target_q_net(next_obs_i).gather(1, next_action_indices)
            else:
                next_q = self.target_q_net(next_obs_i).max(1, keepdim=True)[0]
            target_q = rews_i + self.gamma * (1.0 - done_i) * next_q

        loss = MSELoss(q_sa, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.optimizer.step()

        if logger is not None:
            if niter is not None:
                logger.add_scalar("agent%i/q_loss" % agent_i, loss.item(), niter)
                logger.add_scalar("agent%i/q_target_mean" % agent_i, target_q.mean().item(), niter)
        return {"q_loss": loss.item(), "q_sa_mean": q_sa.mean().item()}

    def state_dict(self):
        return {
            "q_net": self.q_net.state_dict(),
            "target_q_net": self.target_q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "n_actions": self.n_actions,
            "use_double_q": self.use_double_q,
        }
