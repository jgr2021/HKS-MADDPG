import copy

import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent

MSELoss = torch.nn.MSELoss()

ACTOR_AUG_MODELS = {
    'raw': 'mlp',
    'action_aware_control': 'learned_raw_potential_residual',
    'active_gsp': 'learned_active_gsp_residual',
    'vector_signal_geom': 'vector_signal_geom',
    'vector_signal_gsp': 'vector_signal_gsp',
}

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False, actor_model='mlp', actor_aug=None,
                 actor_anchor=None, actor_lr=None, critic_lr=None):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.actor_lr = float(lr if actor_lr is None else actor_lr)
        self.critic_lr = float(lr if critic_lr is None else critic_lr)
        self.agents = [DDPGAgent(
                                 lr=lr, actor_lr=self.actor_lr,
                                 critic_lr=self.critic_lr,
                                 discrete_action=discrete_action,
                                 hidden_dim=hidden_dim,
                                 **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.actor_model = actor_model
        self.actor_aug = actor_aug
        self.actor_anchor = self._validate_actor_anchor(actor_anchor)
        self.policy_anchors = [None for _ in self.agents]
        self.policy_audit_references = [None for _ in self.agents]
        self.policy_audit_observations = [None for _ in self.agents]
        self.last_anchor_update = None
        self.last_policy_audit = None
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @staticmethod
    def _validate_actor_anchor(config):
        if config is None:
            return None
        config = dict(config)
        mode = config.get('mode')
        valid_modes = {
            'fixed_teacher_kl',
            'fixed_teacher_kl_penalty',
            'fixed_parameter_l2_penalty',
            'old_policy_kl',
        }
        if mode not in valid_modes:
            raise ValueError(
                f"actor_anchor mode must be one of {sorted(valid_modes)}, got {mode!r}"
            )
        config['temperature'] = float(config.get('temperature', 1.0))
        config['bisection_steps'] = int(config.get('bisection_steps', 12))
        if config['temperature'] <= 0.0 or config['bisection_steps'] < 1:
            raise ValueError('Invalid actor_anchor temperature or bisection_steps')
        if mode in {'fixed_teacher_kl', 'old_policy_kl'}:
            max_kl = float(config.get('max_kl', 0.002))
            if not 0.0 < max_kl < 1.0:
                raise ValueError('actor_anchor max_kl must lie in (0, 1)')
            config['max_kl'] = max_kl
        if mode in {'fixed_teacher_kl_penalty', 'fixed_parameter_l2_penalty'}:
            coefficient = float(config.get('coefficient', 1.0))
            if coefficient <= 0.0:
                raise ValueError('soft actor-anchor coefficient must be positive')
            config['coefficient'] = coefficient
        return config

    @property
    def requires_fixed_teacher(self):
        return (
            self.actor_anchor is not None
            and self.actor_anchor['mode'] in {
                'fixed_teacher_kl',
                'fixed_teacher_kl_penalty',
                'fixed_parameter_l2_penalty',
            }
        )

    def capture_policy_anchors(self):
        """Freeze the current actors as fixed teachers for policy-space projection."""
        if not self.requires_fixed_teacher:
            raise RuntimeError('The selected actor constraint does not use a fixed teacher')
        self.policy_anchors = []
        for policy in self.policies:
            anchor = copy.deepcopy(policy)
            anchor.eval()
            for parameter in anchor.parameters():
                parameter.requires_grad_(False)
            self.policy_anchors.append(anchor)

    def capture_policy_audit_references(self):
        """Freeze initial policies used only for fixed-state policy-drift auditing."""
        self.policy_audit_references = []
        for policy in self.policies:
            reference = copy.deepcopy(policy)
            reference.eval()
            for parameter in reference.parameters():
                parameter.requires_grad_(False)
            self.policy_audit_references.append(reference)

    def set_policy_audit_observations(self, observations):
        if len(observations) != self.nagents:
            raise ValueError('Policy audit observations must contain one tensor per agent')
        self.policy_audit_observations = [
            observation.detach().clone() for observation in observations
        ]

    def _reference_probabilities(self, reference, observations):
        temperature = self.actor_anchor['temperature'] if self.actor_anchor else 1.0
        with torch.no_grad():
            return F.softmax(reference(observations) / temperature, dim=1)

    def _categorical_kl(self, agent_i, observations, reference_probabilities,
                        track_student_grad=False):
        temperature = self.actor_anchor['temperature'] if self.actor_anchor else 1.0
        if track_student_grad:
            student_log_prob = F.log_softmax(
                self.agents[agent_i].policy(observations) / temperature, dim=1
            )
            return F.kl_div(
                student_log_prob, reference_probabilities, reduction='batchmean'
            )
        with torch.no_grad():
            student_log_prob = F.log_softmax(
                self.agents[agent_i].policy(observations) / temperature, dim=1
            )
            return F.kl_div(
                student_log_prob, reference_probabilities, reduction='batchmean'
            )

    def _fixed_teacher_kl(self, agent_i, observations, track_student_grad=False):
        anchor = self.policy_anchors[agent_i]
        if anchor is None:
            raise RuntimeError('Actor anchoring is enabled but anchors were not captured')
        teacher_prob = self._reference_probabilities(anchor, observations)
        return self._categorical_kl(
            agent_i,
            observations,
            teacher_prob,
            track_student_grad=track_student_grad,
        )

    def _fixed_parameter_l2(self, agent_i):
        """Mean squared displacement from the frozen initial actor parameters."""
        anchor = self.policy_anchors[agent_i]
        if anchor is None:
            raise RuntimeError('Parameter anchoring is enabled but anchors were not captured')
        squared_sum = None
        count = 0
        for parameter, reference in zip(
            self.agents[agent_i].policy.parameters(), anchor.parameters()
        ):
            value = (parameter - reference.detach()).square().sum()
            squared_sum = value if squared_sum is None else squared_sum + value
            count += parameter.numel()
        if squared_sum is None or count == 0:
            raise RuntimeError('Actor has no trainable parameters for L2 anchoring')
        return squared_sum / float(count)

    def _anchor_kl(self, agent_i, observations):
        """Backward-compatible fixed-teacher KL accessor used by prior audits."""
        return self._fixed_teacher_kl(agent_i, observations, track_student_grad=False)

    def _policy_audit_kl(self, agent_i):
        reference = self.policy_audit_references[agent_i]
        observations = self.policy_audit_observations[agent_i]
        if reference is None or observations is None:
            return None
        probabilities = self._reference_probabilities(reference, observations)
        value = float(
            self._categorical_kl(
                agent_i, observations, probabilities, track_student_grad=False
            ).item()
        )
        self.last_policy_audit = {
            'agent_i': int(agent_i),
            'initial_reference_kl': value,
            'audit_batch_size': int(observations.shape[0]),
        }
        return self.last_policy_audit

    @staticmethod
    def _assign_interpolated(parameters, old_values, proposed_values, scale):
        with torch.no_grad():
            for parameter, old, proposed in zip(parameters, old_values, proposed_values):
                parameter.copy_(old + float(scale) * (proposed - old))

    def _project_actor_update_against_probabilities(
            self, agent_i, observations, old_values, proposed_values,
            reference_probabilities, mode):
        """Project an actor proposal onto an empirical categorical KL region.

        The projection is performed after the optimizer proposal, so it constrains
        the very first update (unlike a zero-at-initialization KL penalty).  When a
        new minibatch reveals that the old actor is already outside the empirical
        region, only a KL-improving proposal is accepted.
        """
        parameters = list(self.agents[agent_i].policy.parameters())
        max_kl = self.actor_anchor['max_kl']
        proposed_kl = float(
            self._categorical_kl(
                agent_i, observations, reference_probabilities
            ).item()
        )
        self._assign_interpolated(parameters, old_values, proposed_values, 0.0)
        old_kl = float(
            self._categorical_kl(
                agent_i, observations, reference_probabilities
            ).item()
        )

        accepted_scale = 1.0
        if old_kl > max_kl:
            if proposed_kl >= old_kl:
                accepted_scale = 0.0
            else:
                self._assign_interpolated(parameters, old_values, proposed_values, 1.0)
        elif proposed_kl > max_kl:
            low, high = 0.0, 1.0
            for _ in range(self.actor_anchor['bisection_steps']):
                midpoint = 0.5 * (low + high)
                self._assign_interpolated(
                    parameters, old_values, proposed_values, midpoint
                )
                midpoint_kl = float(
                    self._categorical_kl(
                        agent_i, observations, reference_probabilities
                    ).item()
                )
                if midpoint_kl <= max_kl:
                    low = midpoint
                else:
                    high = midpoint
            accepted_scale = low
            self._assign_interpolated(parameters, old_values, proposed_values, low)
        else:
            self._assign_interpolated(parameters, old_values, proposed_values, 1.0)

        accepted_kl = float(
            self._categorical_kl(
                agent_i, observations, reference_probabilities
            ).item()
        )
        self.last_anchor_update = {
            'agent_i': int(agent_i),
            'mode': mode,
            'old_kl': old_kl,
            'proposed_kl': proposed_kl,
            'accepted_kl': accepted_kl,
            'accepted_scale': float(accepted_scale),
            'max_kl': max_kl,
        }
        return self.last_anchor_update

    def _project_actor_update(self, agent_i, observations, old_values, proposed_values):
        """Project an actor proposal against the frozen initial teacher."""
        anchor = self.policy_anchors[agent_i]
        if anchor is None:
            raise RuntimeError('Fixed teacher was not captured')
        reference_probabilities = self._reference_probabilities(anchor, observations)
        return self._project_actor_update_against_probabilities(
            agent_i,
            observations,
            old_values,
            proposed_values,
            reference_probabilities,
            mode='fixed_teacher_kl',
        )

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                 observations)]

    def update(self, sample, agent_i, parallel=False, logger=None,
               update_actor=True):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]

        curr_agent.critic_optimizer.zero_grad()
        if self.alg_types[agent_i] == 'MADDPG':
            if self.discrete_action: # one-hot encode action
                all_trgt_acs = [onehot_from_logits(pi(nobs)) for pi, nobs in
                                zip(self.target_policies, next_obs)]
            else:
                all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                             next_obs)]
            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        else:  # DDPG
            if self.discrete_action:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        onehot_from_logits(
                                            curr_agent.target_policy(
                                                next_obs[agent_i]))),
                                       dim=1)
            else:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        curr_agent.target_policy(next_obs[agent_i])),
                                       dim=1)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        curr_agent.target_critic(trgt_vf_in) *
                        (1 - dones[agent_i].view(-1, 1)))

        if self.alg_types[agent_i] == 'MADDPG':
            vf_in = torch.cat((*obs, *acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
        actual_value = curr_agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm_(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        if not update_actor:
            if logger is not None:
                values = {'vf_loss': vf_loss}
                audit_report = self._policy_audit_kl(agent_i)
                if audit_report is not None:
                    values['initial_reference_kl'] = audit_report['initial_reference_kl']
                logger.add_scalars('agent%i/losses' % agent_i, values, self.niter)
            return

        curr_agent.policy_optimizer.zero_grad()

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        if self.alg_types[agent_i] == 'MADDPG':
            all_pol_acs = []
            for i, pi, ob in zip(range(self.nagents), self.policies, obs):
                if i == agent_i:
                    all_pol_acs.append(curr_pol_vf_in)
                elif self.discrete_action:
                    all_pol_acs.append(onehot_from_logits(pi(ob)))
                else:
                    all_pol_acs.append(pi(ob))
            vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                              dim=1)
        pol_loss = -curr_agent.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        penalty_kl = None
        penalty_l2 = None
        if (
            self.actor_anchor is not None
            and self.actor_anchor['mode'] == 'fixed_teacher_kl_penalty'
        ):
            penalty_kl = self._fixed_teacher_kl(
                agent_i, obs[agent_i], track_student_grad=True
            )
            pol_loss = (
                pol_loss
                + self.actor_anchor['coefficient'] * penalty_kl
            )
        elif (
            self.actor_anchor is not None
            and self.actor_anchor['mode'] == 'fixed_parameter_l2_penalty'
        ):
            penalty_l2 = self._fixed_parameter_l2(agent_i)
            pol_loss = (
                pol_loss
                + self.actor_anchor['coefficient'] * penalty_l2
            )
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm_(curr_agent.policy.parameters(), 0.5)
        anchor_report = None
        if (
            self.actor_anchor is None
            or self.actor_anchor['mode'] in {
                'fixed_teacher_kl_penalty',
                'fixed_parameter_l2_penalty',
            }
        ):
            curr_agent.policy_optimizer.step()
            if self.actor_anchor is not None:
                if self.actor_anchor['mode'] == 'fixed_teacher_kl_penalty':
                    accepted_kl = float(self._fixed_teacher_kl(
                        agent_i, obs[agent_i], track_student_grad=False
                    ).item())
                    parameter_l2 = float('nan')
                    old_value = float(penalty_kl.detach().item())
                else:
                    accepted_kl = float('nan')
                    parameter_l2 = float(
                        self._fixed_parameter_l2(agent_i).detach().item()
                    )
                    old_value = float(penalty_l2.detach().item())
                anchor_report = {
                    'agent_i': int(agent_i),
                    'mode': self.actor_anchor['mode'],
                    'old_kl': old_value,
                    'proposed_kl': accepted_kl,
                    'accepted_kl': accepted_kl,
                    'accepted_scale': 1.0,
                    'max_kl': float('nan'),
                    'parameter_l2': parameter_l2,
                }
                self.last_anchor_update = anchor_report
        else:
            old_policy_values = [
                parameter.detach().clone() for parameter in curr_agent.policy.parameters()
            ]
            old_policy_probabilities = None
            if self.actor_anchor['mode'] == 'old_policy_kl':
                temperature = self.actor_anchor['temperature']
                old_policy_probabilities = F.softmax(
                    curr_pol_out.detach() / temperature, dim=1
                )
            curr_agent.policy_optimizer.step()
            proposed_policy_values = [
                parameter.detach().clone() for parameter in curr_agent.policy.parameters()
            ]
            if self.actor_anchor['mode'] == 'fixed_teacher_kl':
                anchor_report = self._project_actor_update(
                    agent_i, obs[agent_i], old_policy_values, proposed_policy_values
                )
            elif self.actor_anchor['mode'] == 'old_policy_kl':
                anchor_report = self._project_actor_update_against_probabilities(
                    agent_i,
                    obs[agent_i],
                    old_policy_values,
                    proposed_policy_values,
                    old_policy_probabilities,
                    mode='old_policy_kl',
                )
            else:
                raise RuntimeError(
                    f"Unhandled actor constraint mode {self.actor_anchor['mode']}"
                )
        if logger is not None:
            values = {'vf_loss': vf_loss, 'pol_loss': pol_loss}
            if anchor_report is not None:
                if anchor_report['mode'] == 'fixed_parameter_l2_penalty':
                    values['anchor_parameter_l2'] = anchor_report['parameter_l2']
                else:
                    values.update({
                        'anchor_old_kl': anchor_report['old_kl'],
                        'anchor_proposed_kl': anchor_report['proposed_kl'],
                        'anchor_accepted_kl': anchor_report['accepted_kl'],
                        'anchor_accepted_scale': anchor_report['accepted_scale'],
                    })
            audit_report = self._policy_audit_kl(agent_i)
            if audit_report is not None:
                values['initial_reference_kl'] = audit_report['initial_reference_kl']
            logger.add_scalars('agent%i/losses' % agent_i, values, self.niter)

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device
        if self.actor_anchor is not None:
            for index, anchor in enumerate(self.policy_anchors):
                if anchor is not None:
                    self.policy_anchors[index] = fn(anchor)
        for index, reference in enumerate(self.policy_audit_references):
            if reference is not None:
                self.policy_audit_references[index] = fn(reference)
        self.policy_audit_observations = [
            None if observation is None else fn(observation)
            for observation in self.policy_audit_observations
        ]

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents],
                     'policy_anchor_states': [
                         None if anchor is None else anchor.state_dict()
                         for anchor in self.policy_anchors
                     ],
                     'policy_audit_reference_states': [
                         None if reference is None else reference.state_dict()
                         for reference in self.policy_audit_references
                     ]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                      actor_model='mlp', actor_aug=None, actor_anchor=None,
                      actor_lr=None, critic_lr=None):
        """
        Instantiate instance of this class from multi-agent environment
        """
        if actor_aug is not None:
            if actor_aug not in ACTOR_AUG_MODELS:
                raise ValueError(f"Unknown actor_aug: {actor_aug}")
            resolved_model = ACTOR_AUG_MODELS[actor_aug]
            if actor_model != 'mlp' and actor_model != resolved_model:
                raise ValueError("actor_model and actor_aug select different actor policies")
            actor_model = resolved_model
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        for agent_index, (acsp, obsp, algtype) in enumerate(
                zip(env.action_space, env.observation_space, alg_types)):
            num_in_pol = obsp.shape[0]
            if isinstance(acsp, Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic,
                                      'actor_model': actor_model,
                                      'agent_index': agent_index})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'actor_lr': actor_lr, 'critic_lr': critic_lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'actor_model': actor_model,
                     'actor_aug': actor_aug,
                     'actor_anchor': actor_anchor}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        anchor_states = save_dict.get('policy_anchor_states')
        if anchor_states is not None and any(state is not None for state in anchor_states):
            instance.capture_policy_anchors()
            for anchor, state in zip(instance.policy_anchors, anchor_states):
                if state is not None:
                    anchor.load_state_dict(state)
        audit_states = save_dict.get('policy_audit_reference_states')
        if audit_states is not None and any(state is not None for state in audit_states):
            instance.capture_policy_audit_references()
            for reference, state in zip(instance.policy_audit_references, audit_states):
                if state is not None:
                    reference.load_state_dict(state)
        return instance
