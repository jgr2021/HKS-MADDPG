from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import (
    ActionRawPotentialPolicy,
    ActionRawPotentialFixedPolicy,
    ActionRawPotentialPriorPolicy,
    ActionRawPotentialResidualPolicy,
    ActionScoreControlPolicy,
    ActiveGSPV3FixedPolicy,
    ActiveGSPV3Policy,
    ActiveGSPV3PriorPolicy,
    ActiveGSPV3ResidualPolicy,
    CounterfactualActiveProbePolicy,
    ScalableCounterfactual6x6Policy,
    EquivariantMatchingSafetyPolicy,
    EquivariantMatchingSafetyPolicy8,
    EquivariantMatchingSafety6x6Policy,
    EquivariantMatchingSafety6x6Policy8,
    EquivariantMatchingSafety6x6Policy16,
    EquivariantMatchingSafety6x6Policy32,
    GatedGraphPolicy,
    LearnedActiveGSPResidualPolicy,
    LearnedActiveRWSEResidualPolicy,
    LearnedActiveSCFResidualPolicy,
    LearnedCrowdingGSPResidualPolicy,
    LearnedGatedDirichletControlPolicy,
    LearnedGatedDirichletGSPPolicy,
    LearnedGatedDirichletGSP025Policy,
    LearnedMatchingControlPolicy,
    LearnedMultiscaleSpectralResidualPolicy,
    LearnedRWSEPotentialControlPolicy,
    LearnedSCFPotentialControlPolicy,
    LearnedSpectralMatchingResidualPolicy,
    LinearMatchingControlPolicy,
    LinearSpectralMatchingPolicy,
    MatchingOnlyControlPolicy,
    SpectralMatchingOnlyPolicy,
    LearnedRawPotentialResidualPolicy,
    LocalGeometryResidualPolicy,
    LocalSpectralALResidualPolicy,
    MLPNetwork,
    MessageGraphPolicy,
    ScaledActionRawPotentialPolicy,
    ScaledActiveGSPV3Policy,
    VectorSignalGeomPolicy,
    VectorSignalGSPPolicy,
    PassiveTopology3NodeAAHKSPolicy,
    PassiveTopology6NodeAALHKSPolicy,
    PassiveTopology6NodeALHKSPolicy,
)
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise
from .d4_graph_residual import D4DirichletResidualPolicy, D4PotentialResidualPolicy


POLICY_TYPES = {
    'd4_potential_residual': D4PotentialResidualPolicy,
    'd4_dirichlet_residual': D4DirichletResidualPolicy,
    'mlp': MLPNetwork,
    'gated_graph': GatedGraphPolicy,
    'message_graph': MessageGraphPolicy,
    'learned_raw_potential_residual': LearnedRawPotentialResidualPolicy,
    'learned_active_gsp_residual': LearnedActiveGSPResidualPolicy,
    'counterfactual_active_probe': CounterfactualActiveProbePolicy,
    'scalable_counterfactual_6x6': ScalableCounterfactual6x6Policy,
    'equivariant_matching_safety_nxn': EquivariantMatchingSafetyPolicy,
    'equivariant_matching_safety_nxn_sinkhorn8':
        EquivariantMatchingSafetyPolicy8,
    'equivariant_matching_safety_6x6': EquivariantMatchingSafety6x6Policy,
    'equivariant_matching_safety_6x6_sinkhorn8':
        EquivariantMatchingSafety6x6Policy8,
    'equivariant_matching_safety_6x6_sinkhorn16':
        EquivariantMatchingSafety6x6Policy16,
    'equivariant_matching_safety_6x6_sinkhorn32':
        EquivariantMatchingSafety6x6Policy32,
    'learned_multiscale_spectral_residual': LearnedMultiscaleSpectralResidualPolicy,
    'learned_crowding_gsp_residual': LearnedCrowdingGSPResidualPolicy,
    'learned_rwse_potential_control': LearnedRWSEPotentialControlPolicy,
    'learned_active_rwse_residual': LearnedActiveRWSEResidualPolicy,
    'learned_scf_potential_control': LearnedSCFPotentialControlPolicy,
    'learned_active_scf_residual': LearnedActiveSCFResidualPolicy,
    'learned_gated_dirichlet_control': LearnedGatedDirichletControlPolicy,
    'learned_gated_dirichlet_gsp': LearnedGatedDirichletGSPPolicy,
    'learned_gated_dirichlet_gsp025': LearnedGatedDirichletGSP025Policy,
    'learned_matching_control': LearnedMatchingControlPolicy,
    'learned_spectral_matching_residual': LearnedSpectralMatchingResidualPolicy,
    'linear_matching_control': LinearMatchingControlPolicy,
    'linear_spectral_matching': LinearSpectralMatchingPolicy,
    'matching_only_control': MatchingOnlyControlPolicy,
    'spectral_matching_only': SpectralMatchingOnlyPolicy,
    'local_geometry_residual': LocalGeometryResidualPolicy,
    'local_spectral_al_residual': LocalSpectralALResidualPolicy,
    'action_score_control': ActionScoreControlPolicy,
    'action_raw_potential': ActionRawPotentialPolicy,
    'active_gsp_v3': ActiveGSPV3Policy,
    'action_raw_potential_scaled': ScaledActionRawPotentialPolicy,
    'active_gsp_v3_scaled': ScaledActiveGSPV3Policy,
    'action_raw_potential_prior': ActionRawPotentialPriorPolicy,
    'active_gsp_v3_prior': ActiveGSPV3PriorPolicy,
    'action_raw_potential_residual005': ActionRawPotentialResidualPolicy,
    'active_gsp_v3_residual005': ActiveGSPV3ResidualPolicy,
    'action_raw_potential_fixed': ActionRawPotentialFixedPolicy,
    'active_gsp_v3_fixed': ActiveGSPV3FixedPolicy,
    'vector_signal_geom': VectorSignalGeomPolicy,
    'vector_signal_gsp': VectorSignalGSPPolicy,
    'passive_topology_3node_aa_hks': PassiveTopology3NodeAAHKSPolicy,
    'passive_topology_6node_al_hks': PassiveTopology6NodeALHKSPolicy,
    'passive_topology_6node_aal_hks': PassiveTopology6NodeAALHKSPolicy,
}

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, actor_lr=None, critic_lr=None,
                 discrete_action=True, actor_model='mlp', agent_index=0):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        if actor_model not in POLICY_TYPES:
            raise ValueError(f"Unknown actor_model: {actor_model}")
        policy_cls = POLICY_TYPES[actor_model]
        self.policy = policy_cls(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action,
                                 agent_index=agent_index)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False)
        self.target_policy = policy_cls(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action,
                                        agent_index=agent_index)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        actor_lr = float(lr if actor_lr is None else actor_lr)
        critic_lr = float(lr if critic_lr is None else critic_lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                action += Variable(Tensor(self.exploration.noise()),
                                   requires_grad=False)
            action = action.clamp(-1, 1)
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
