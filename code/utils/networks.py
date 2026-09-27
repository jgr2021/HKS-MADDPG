import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from utils.gsp_features import descriptor_for_mode
from utils.multiscale_spectral_features import (
    MULTISCALE_FEATURE_DIM,
    multiscale_spectral_action_features,
)
from utils.vector_signal_gsp_features import (
    VECTOR_SIGNAL_GEOM_DIM,
    VECTOR_SIGNAL_GSP_DIM,
    vector_signal_features_tensor,
)
from utils.scalable_active_features import scalable_active_action_features_tensor


N_AGENTS = 3
N_LANDMARKS = 3
RAW_OBS_DIM = 18
N_ACTIONS = 5
ACTIVE_GSP_FEATURES_PER_ACTION = 4
AGENT_AGENT_SCALE = 0.25
SIGMA_AGENT_AGENT = 0.80
SIGMA_AGENT_LANDMARK = 0.60
ACTION_TO_CONTROL = (
    (0.0, 0.0),
    (1.0, 0.0),
    (-1.0, 0.0),
    (0.0, 1.0),
    (0.0, -1.0),
)
DEFAULT_DT = 0.1
DEFAULT_DAMPING = 0.25
DEFAULT_SENSITIVITY = 5.0
SOFT_MATCHING_TEMPERATURE = 2.0

class MLPNetwork(nn.Module):
    """
    MLP network (can be used as value or policy)
    """
    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True,
                 agent_index=0):
        """
        Inputs:
            input_dim (int): Number of dimensions in input
            out_dim (int): Number of dimensions in output
            hidden_dim (int): Number of hidden dimensions
            nonlin (PyTorch function): Nonlinearity to apply to hidden layers
        """
        super(MLPNetwork, self).__init__()

        if norm_in:  # normalize inputs
            self.in_fn = nn.BatchNorm1d(input_dim)
            self.in_fn.weight.data.fill_(1)
            self.in_fn.bias.data.fill_(0)
        else:
            self.in_fn = lambda x: x
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        if constrain_out and not discrete_action:
            # initialize small to prevent saturation
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:  # logits for discrete action (will softmax later)
            self.out_fn = lambda x: x

    def forward(self, X):
        """
        Inputs:
            X (PyTorch Matrix): Batch of observations
        Outputs:
            out (PyTorch Matrix): Output of network (actions, values, etc)
        """
        h1 = self.nonlin(self.fc1(self.in_fn(X)))
        h2 = self.nonlin(self.fc2(h1))
        out = self.out_fn(self.fc3(h2))
        return out


class PassiveTopologyHKSPolicy(nn.Module):
    """Actor-only passive topology-HKS augmentation over raw 18D observations."""

    topology_mode = "topology_6node_aal_hks"

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True,
                 agent_index=0):
        super(PassiveTopologyHKSPolicy, self).__init__()
        if input_dim != RAW_OBS_DIM:
            raise ValueError(
                f"PassiveTopologyHKSPolicy expects raw {RAW_OBS_DIM}D observations, "
                f"got {input_dim}."
            )
        self.agent_index = int(agent_index)
        self.actor_input_dim = RAW_OBS_DIM + 3
        self.last_raw_input_shape = None
        self.last_augmented_input_shape = None
        self.mlp = MLPNetwork(
            self.actor_input_dim,
            out_dim,
            hidden_dim=hidden_dim,
            nonlin=nonlin,
            constrain_out=constrain_out,
            norm_in=norm_in,
            discrete_action=discrete_action,
            agent_index=agent_index,
        )

    def _hks_features(self, X):
        raw = X[:, :RAW_OBS_DIM].detach().cpu().numpy()
        features = np.asarray(
            [
                descriptor_for_mode(row, self.agent_index, mode=self.topology_mode)
                for row in raw
            ],
            dtype=np.float32,
        )
        return torch.as_tensor(features, dtype=X.dtype, device=X.device)

    def augment_observation(self, X):
        if X.shape[1] != RAW_OBS_DIM:
            raise ValueError(
                f"Topology-HKS actor expects raw {RAW_OBS_DIM}D input, got {X.shape}."
            )
        hks = self._hks_features(X)
        augmented = torch.cat([X, hks], dim=1)
        self.last_raw_input_shape = tuple(X.shape)
        self.last_augmented_input_shape = tuple(augmented.shape)
        return augmented

    def forward(self, X):
        return self.mlp(self.augment_observation(X))


class PassiveTopology3NodeAAHKSPolicy(PassiveTopologyHKSPolicy):
    topology_mode = "topology_3node_aa_hks"


class PassiveTopology6NodeALHKSPolicy(PassiveTopologyHKSPolicy):
    topology_mode = "topology_6node_al_hks"


class PassiveTopology6NodeAALHKSPolicy(PassiveTopologyHKSPolicy):
    topology_mode = "topology_6node_aal_hks"


def _reconstruct_relative_task_nodes(raw_obs, self_agent_index):
    """Build local task nodes using only one actor's relative observation."""
    if raw_obs.ndim != 2 or raw_obs.shape[1] != RAW_OBS_DIM:
        raise ValueError(
            f"Local graph actor expects [batch, {RAW_OBS_DIM}], got {tuple(raw_obs.shape)}."
        )

    batch_size = raw_obs.shape[0]
    device = raw_obs.device
    dtype = raw_obs.dtype
    landmark_rel = raw_obs[:, 4:10].reshape(batch_size, N_LANDMARKS, 2)
    other_rel = raw_obs[:, 10:14].reshape(batch_size, N_AGENTS - 1, 2)

    agent_rel = torch.zeros(batch_size, N_AGENTS, 2, device=device, dtype=dtype)
    other_indices = [index for index in range(N_AGENTS) if index != self_agent_index]
    for local_index, global_index in enumerate(other_indices):
        agent_rel[:, global_index, :] = other_rel[:, local_index, :]
    positions = torch.cat([agent_rel, landmark_rel], dim=1)

    velocities = torch.zeros_like(positions)
    velocities[:, self_agent_index, :] = raw_obs[:, 0:2]
    type_bits = torch.zeros(batch_size, N_AGENTS + N_LANDMARKS, 2,
                            device=device, dtype=dtype)
    type_bits[:, :N_AGENTS, 0] = 1.0
    type_bits[:, N_AGENTS:, 1] = 1.0
    self_bit = torch.zeros(batch_size, N_AGENTS + N_LANDMARKS, 1,
                           device=device, dtype=dtype)
    self_bit[:, self_agent_index, 0] = 1.0
    node_features = torch.cat([positions, velocities, type_bits, self_bit], dim=2)
    return positions, node_features


def _normalized_agent_landmark_adjacency(positions):
    """Return symmetric D^-1/2 A D^-1/2 for the local AL task graph."""
    agent_positions = positions[:, :N_AGENTS, :]
    landmark_positions = positions[:, N_AGENTS:, :]
    distance_sq = (
        agent_positions[:, :, None, :] - landmark_positions[:, None, :, :]
    ).square().sum(dim=-1)
    weights = torch.exp(-distance_sq / (2.0 * SIGMA_AGENT_LANDMARK ** 2))

    batch_size = positions.shape[0]
    adjacency = positions.new_zeros(
        batch_size, N_AGENTS + N_LANDMARKS, N_AGENTS + N_LANDMARKS
    )
    adjacency[:, :N_AGENTS, N_AGENTS:] = weights
    adjacency[:, N_AGENTS:, :N_AGENTS] = weights.transpose(1, 2)
    degree = adjacency.sum(dim=2).clamp_min(1e-6)
    degree_inv_sqrt = degree.rsqrt()
    return adjacency * degree_inv_sqrt[:, :, None] * degree_inv_sqrt[:, None, :]


class LocalGraphResidualPolicy(nn.Module):
    """Raw actor plus a small local-geometry graph residual.

    The graph branch uses only the focal actor's raw observation.  The
    no-diffusion subclass is an architecture-matched geometry control; the
    spectral subclass applies fixed polynomial filters of the normalized
    agent-landmark graph shift S = I - L.
    """

    use_spectral_diffusion = False

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True,
                 agent_index=0):
        super(LocalGraphResidualPolicy, self).__init__()
        if input_dim != RAW_OBS_DIM:
            raise ValueError(
                f"LocalGraphResidualPolicy expects raw {RAW_OBS_DIM}D observations, "
                f"got {input_dim}."
            )
        self.agent_index = int(agent_index)
        self.nonlin = nonlin
        self.actor_input_dim = RAW_OBS_DIM
        self.mlp = MLPNetwork(
            input_dim,
            out_dim,
            hidden_dim=hidden_dim,
            nonlin=nonlin,
            constrain_out=constrain_out,
            norm_in=norm_in,
            discrete_action=discrete_action,
            agent_index=agent_index,
        )

        graph_dim = max(16, hidden_dim // 2)
        self.node_encoder = nn.Linear(7, graph_dim)
        self.filter_mixer = nn.Linear(graph_dim * 4, graph_dim)
        self.graph_fc = nn.Linear(graph_dim * 4, hidden_dim)
        self.graph_out = nn.Linear(hidden_dim, out_dim)
        nn.init.normal_(self.graph_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.graph_out.bias)

        self.last_raw_input_shape = None
        self.last_graph_node_shape = None
        self.last_graph_embedding_shape = None

    def _filter_bank(self, node_hidden, positions):
        if not self.use_spectral_diffusion:
            return torch.cat([node_hidden] * 4, dim=2)

        shift = _normalized_agent_landmark_adjacency(positions)
        order_1 = torch.bmm(shift, node_hidden)
        order_2 = torch.bmm(shift, order_1)
        order_3 = torch.bmm(shift, order_2)
        order_4 = torch.bmm(shift, order_3)
        return torch.cat([node_hidden, order_1, order_2, order_4], dim=2)

    def graph_embedding(self, X):
        positions, node_features = _reconstruct_relative_task_nodes(
            X, self.agent_index
        )
        node_hidden = self.nonlin(self.node_encoder(node_features))
        filtered = self.nonlin(self.filter_mixer(self._filter_bank(node_hidden, positions)))

        focal = filtered[:, self.agent_index, :]
        other_indices = [index for index in range(N_AGENTS) if index != self.agent_index]
        other_agents = filtered[:, other_indices, :].mean(dim=1)
        landmarks = filtered[:, N_AGENTS:, :]
        landmark_mean = landmarks.mean(dim=1)
        landmark_max = landmarks.max(dim=1).values
        embedding = torch.cat([focal, other_agents, landmark_mean, landmark_max], dim=1)
        self.last_graph_node_shape = tuple(node_features.shape)
        self.last_graph_embedding_shape = tuple(embedding.shape)
        return embedding

    def forward(self, X):
        if X.ndim != 2 or X.shape[1] != RAW_OBS_DIM:
            raise ValueError(
                f"Local graph actor expects [batch, {RAW_OBS_DIM}], got {tuple(X.shape)}."
            )
        self.last_raw_input_shape = tuple(X.shape)
        graph_hidden = self.nonlin(self.graph_fc(self.graph_embedding(X)))
        return self.mlp(X) + self.graph_out(graph_hidden)


class LocalGeometryResidualPolicy(LocalGraphResidualPolicy):
    """Local node-geometry residual without graph diffusion (control)."""


class LocalSpectralALResidualPolicy(LocalGraphResidualPolicy):
    """Local agent-landmark spectral polynomial residual actor."""

    use_spectral_diffusion = True


def _reconstruct_nodes(raw_obs, self_agent_index):
    """Build six task nodes [A0,A1,A2,L0,L1,L2] from batched raw observations."""
    raw = raw_obs[:, :RAW_OBS_DIM]
    batch_size = raw.shape[0]
    device = raw.device
    dtype = raw.dtype

    self_pos = raw[:, 2:4]
    self_vel = raw[:, 0:2]
    landmark_pos = self_pos[:, None, :] + raw[:, 4:10].view(batch_size, N_LANDMARKS, 2)
    other_rel = raw[:, 10:14].view(batch_size, N_AGENTS - 1, 2)

    agent_pos = torch.zeros(batch_size, N_AGENTS, 2, device=device, dtype=dtype)
    agent_vel = torch.zeros(batch_size, N_AGENTS, 2, device=device, dtype=dtype)
    agent_pos[:, self_agent_index, :] = self_pos
    agent_vel[:, self_agent_index, :] = self_vel

    other_indices = [idx for idx in range(N_AGENTS) if idx != self_agent_index]
    for local_index, global_index in enumerate(other_indices):
        agent_pos[:, global_index, :] = self_pos + other_rel[:, local_index, :]

    positions = torch.cat([agent_pos, landmark_pos], dim=1)
    velocities = torch.zeros(batch_size, N_AGENTS + N_LANDMARKS, 2, device=device, dtype=dtype)
    velocities[:, :N_AGENTS, :] = agent_vel

    type_bits = torch.zeros(batch_size, N_AGENTS + N_LANDMARKS, 2, device=device, dtype=dtype)
    type_bits[:, :N_AGENTS, 0] = 1.0
    type_bits[:, N_AGENTS:, 1] = 1.0

    self_bit = torch.zeros(batch_size, N_AGENTS + N_LANDMARKS, 1, device=device, dtype=dtype)
    self_bit[:, self_agent_index, 0] = 1.0

    node_features = torch.cat([positions, velocities, type_bits, self_bit], dim=2)
    return positions, node_features


def _task_edge_prior(positions):
    """Fixed geometric prior with the same constants used by the GSP descriptors."""
    batch_size = positions.shape[0]
    device = positions.device
    dtype = positions.dtype
    adjacency = torch.zeros(batch_size, N_AGENTS + N_LANDMARKS, N_AGENTS + N_LANDMARKS,
                            device=device, dtype=dtype)

    agent_pos = positions[:, :N_AGENTS, :]
    landmark_pos = positions[:, N_AGENTS:, :]

    for i in range(N_AGENTS):
        for j in range(i + 1, N_AGENTS):
            dist_sq = ((agent_pos[:, i, :] - agent_pos[:, j, :]) ** 2).sum(dim=1)
            weight = 0.25 * torch.exp(-dist_sq / (2.0 * 0.80 ** 2))
            adjacency[:, i, j] = weight
            adjacency[:, j, i] = weight

    for i in range(N_AGENTS):
        for j in range(N_LANDMARKS):
            dist_sq = ((agent_pos[:, i, :] - landmark_pos[:, j, :]) ** 2).sum(dim=1)
            weight = torch.exp(-dist_sq / (2.0 * 0.60 ** 2))
            node_j = N_AGENTS + j
            adjacency[:, i, node_j] = weight
            adjacency[:, node_j, i] = weight

    return adjacency


def _active_gsp_v3_features_tensor(raw_obs):
    """Torch fast path for local-observation-only Active GSP v3 features.

    Returns shape [batch, 5, 4] in the order:
    [delta coverage potential, delta crowding potential,
     delta coverage energy, delta crowding energy].
    """
    raw = raw_obs[:, :RAW_OBS_DIM]
    batch_size = raw.shape[0]
    device = raw.device
    dtype = raw.dtype
    controls = raw.new_tensor(ACTION_TO_CONTROL)

    landmark_positions = raw[:, 4:10].view(batch_size, N_LANDMARKS, 2)
    other_positions = raw[:, 10:14].view(batch_size, N_AGENTS - 1, 2)
    self_velocity = raw[:, 0:2]

    dt = raw.new_tensor(DEFAULT_DT)
    damping = raw.new_tensor(DEFAULT_DAMPING)
    sensitivity = raw.new_tensor(DEFAULT_SENSITIVITY)
    self_displacements = (
        self_velocity[:, None, :] * (1.0 - damping) * dt
        + controls[None, :, :] * sensitivity * dt * dt
    )

    self_x = self_displacements[:, :, 0]
    self_y = self_displacements[:, :, 1]
    other_x = other_positions[:, :, 0]
    other_y = other_positions[:, :, 1]
    landmark_x = landmark_positions[:, :, 0]
    landmark_y = landmark_positions[:, :, 1]

    self_other_dx = self_x[:, :, None] - other_x[:, None, :]
    self_other_dy = self_y[:, :, None] - other_y[:, None, :]
    self_other_dist_sq = self_other_dx.square() + self_other_dy.square()
    self_other_weights = AGENT_AGENT_SCALE * torch.exp(
        -self_other_dist_sq / (2.0 * SIGMA_AGENT_AGENT ** 2)
    )

    other_other_dx = other_x[:, 0] - other_x[:, 1]
    other_other_dy = other_y[:, 0] - other_y[:, 1]
    other_other_dist_sq = other_other_dx.square() + other_other_dy.square()
    other_other_weight = AGENT_AGENT_SCALE * torch.exp(
        -other_other_dist_sq / (2.0 * SIGMA_AGENT_AGENT ** 2)
    )

    self_landmark_dx = self_x[:, :, None] - landmark_x[:, None, :]
    self_landmark_dy = self_y[:, :, None] - landmark_y[:, None, :]
    self_landmark_dist_sq = self_landmark_dx.square() + self_landmark_dy.square()
    self_landmark_dist = torch.sqrt(torch.clamp(self_landmark_dist_sq, min=0.0))
    self_landmark_weights = torch.exp(
        -self_landmark_dist_sq / (2.0 * SIGMA_AGENT_LANDMARK ** 2)
    )

    other_landmark_dx = other_x[:, :, None] - landmark_x[:, None, :]
    other_landmark_dy = other_y[:, :, None] - landmark_y[:, None, :]
    other_landmark_dist_sq = other_landmark_dx.square() + other_landmark_dy.square()
    other_landmark_dist = torch.sqrt(torch.clamp(other_landmark_dist_sq, min=0.0))
    other_landmark_weights = torch.exp(
        -other_landmark_dist_sq / (2.0 * SIGMA_AGENT_LANDMARK ** 2)
    )

    nearest_other_landmark_dist = torch.minimum(
        other_landmark_dist[:, 0:1, :],
        other_landmark_dist[:, 1:2, :],
    )
    coverage_residual = torch.minimum(
        self_landmark_dist,
        nearest_other_landmark_dist,
    )
    coverage_potential = coverage_residual.sum(dim=2)

    crowd_self = torch.maximum(self_other_weights[:, :, 0], self_other_weights[:, :, 1])
    crowd_other1 = torch.maximum(self_other_weights[:, :, 0], other_other_weight[:, None])
    crowd_other2 = torch.maximum(self_other_weights[:, :, 1], other_other_weight[:, None])

    crowding_potential = self_other_weights.sum(dim=2) + other_other_weight[:, None]

    coverage_residual_sq = coverage_residual.square()
    other_landmark_weight_sum = other_landmark_weights.sum(dim=1)
    coverage_energy = (
        (self_landmark_weights * coverage_residual_sq).sum(dim=2)
        + (other_landmark_weight_sum[:, None, :] * coverage_residual_sq).sum(dim=2)
    )

    self_landmark_weight_sum = self_landmark_weights.sum(dim=2)
    other1_landmark_weight_sum = other_landmark_weights[:, 0, :].sum(dim=1)[:, None]
    other2_landmark_weight_sum = other_landmark_weights[:, 1, :].sum(dim=1)[:, None]
    crowding_al_energy = (
        self_landmark_weight_sum * crowd_self.square()
        + other1_landmark_weight_sum * crowd_other1.square()
        + other2_landmark_weight_sum * crowd_other2.square()
    )
    crowding_aa_energy = (
        self_other_weights[:, :, 0] * (crowd_self - crowd_other1).square()
        + self_other_weights[:, :, 1] * (crowd_self - crowd_other2).square()
        + other_other_weight[:, None] * (crowd_other1 - crowd_other2).square()
    )
    crowding_energy = crowding_al_energy + crowding_aa_energy

    quantities = torch.stack(
        [coverage_potential, crowding_potential, coverage_energy, crowding_energy],
        dim=2,
    )
    return quantities - quantities[:, 0:1, :]


def _candidate_agent_landmark_graph_tensors(raw_obs, self_agent_index):
    """Build batched candidate AL graphs from one focal local observation."""
    if raw_obs.ndim != 2 or raw_obs.shape[1] != RAW_OBS_DIM:
        raise ValueError(
            f"Candidate AL graph expects [batch, {RAW_OBS_DIM}], "
            f"got {tuple(raw_obs.shape)}."
        )

    raw = raw_obs[:, :RAW_OBS_DIM]
    batch_size = raw.shape[0]
    controls = raw.new_tensor(ACTION_TO_CONTROL)
    self_displacements = (
        raw[:, 0:2, None].transpose(1, 2) * (1.0 - DEFAULT_DAMPING) * DEFAULT_DT
        + controls[None, :, :] * DEFAULT_SENSITIVITY * DEFAULT_DT ** 2
    )

    agent_positions = raw.new_zeros(batch_size, N_ACTIONS, N_AGENTS, 2)
    agent_positions[:, :, self_agent_index, :] = self_displacements
    other_positions = raw[:, 10:14].reshape(batch_size, N_AGENTS - 1, 2)
    other_indices = [index for index in range(N_AGENTS) if index != self_agent_index]
    for local_index, global_index in enumerate(other_indices):
        agent_positions[:, :, global_index, :] = other_positions[:, None, local_index, :]

    landmark_positions = raw[:, 4:10].reshape(batch_size, N_LANDMARKS, 2)
    distance_sq = (
        agent_positions[:, :, :, None, :] - landmark_positions[:, None, None, :, :]
    ).square().sum(dim=-1)
    weights = torch.exp(-distance_sq / (2.0 * SIGMA_AGENT_LANDMARK ** 2))

    adjacency = raw.new_zeros(
        batch_size,
        N_ACTIONS,
        N_AGENTS + N_LANDMARKS,
        N_AGENTS + N_LANDMARKS,
    )
    adjacency[:, :, :N_AGENTS, N_AGENTS:] = weights
    adjacency[:, :, N_AGENTS:, :N_AGENTS] = weights.transpose(2, 3)
    degree = adjacency.sum(dim=3).clamp_min(1e-8)
    transition = adjacency / degree[:, :, :, None]
    degree_inv_sqrt = degree.rsqrt()
    symmetric_shift = (
        adjacency
        * degree_inv_sqrt[:, :, :, None]
        * degree_inv_sqrt[:, :, None, :]
    )

    landmark_candidates = landmark_positions[:, None, :, :].expand(
        -1, N_ACTIONS, -1, -1
    )
    node_positions = torch.cat([agent_positions, landmark_candidates], dim=2)
    return raw, node_positions, transition, symmetric_shift


def _action_conditioned_rwse_features_tensor(raw_obs, self_agent_index):
    """Return potentials and focal RWSE-2/4/6 deltas for every action."""
    raw, _, transition, _ = _candidate_agent_landmark_graph_tensors(
        raw_obs, self_agent_index
    )

    transition_2 = torch.matmul(transition, transition)
    transition_4 = torch.matmul(transition_2, transition_2)
    transition_6 = torch.matmul(transition_4, transition_2)
    rwse = torch.stack(
        [
            transition_2[:, :, self_agent_index, self_agent_index],
            transition_4[:, :, self_agent_index, self_agent_index],
            transition_6[:, :, self_agent_index, self_agent_index],
        ],
        dim=2,
    )
    rwse_delta = rwse - rwse[:, 0:1, :]
    potentials = _active_gsp_v3_features_tensor(raw)[:, :, :2]
    return torch.cat([potentials, rwse_delta], dim=2)


def _action_conditioned_spectral_coordinate_features_tensor(raw_obs, self_agent_index):
    """Return directional local graph-filter responses for every action.

    The four spectral channels are the focal x/y responses after one and two
    powers of the symmetric normalized graph shift S = I - L_sym.
    """
    raw, node_positions, _, shift = _candidate_agent_landmark_graph_tensors(
        raw_obs, self_agent_index
    )
    order_1 = torch.matmul(shift, node_positions)
    order_2 = torch.matmul(shift, order_1)
    focal_responses = torch.cat(
        [
            order_1[:, :, self_agent_index, :],
            order_2[:, :, self_agent_index, :],
        ],
        dim=2,
    )
    response_delta = focal_responses - focal_responses[:, 0:1, :]
    potentials = _active_gsp_v3_features_tensor(raw)[:, :, :2]
    return torch.cat([potentials, response_delta], dim=2)


def _candidate_soft_matching_tensor(raw_obs, self_agent_index, iterations=10):
    """Build candidate-action soft one-to-one agent-landmark assignments."""
    raw, node_positions, _, _ = _candidate_agent_landmark_graph_tensors(
        raw_obs, self_agent_index
    )
    agent_positions = node_positions[:, :, :N_AGENTS, :]
    landmark_positions = node_positions[:, :, N_AGENTS:, :]
    distance_sq = (
        agent_positions[:, :, :, None, :] - landmark_positions[:, :, None, :, :]
    ).square().sum(dim=-1)
    distances = torch.sqrt(distance_sq.clamp_min(0.0))
    logits = -distance_sq / SOFT_MATCHING_TEMPERATURE
    assignment = torch.exp(logits - logits.amax(dim=3, keepdim=True)).clamp_min(1e-12)
    for _ in range(iterations):
        assignment = assignment / assignment.sum(dim=3, keepdim=True).clamp_min(1e-12)
        assignment = assignment / assignment.sum(dim=2, keepdim=True).clamp_min(1e-12)
    return raw, distances, assignment


def _action_conditioned_spectral_matching_features_tensor(raw_obs, self_agent_index):
    """Return assignment-cost and bipartite matching-spectrum action deltas."""
    raw, distances, assignment = _candidate_soft_matching_tensor(
        raw_obs, self_agent_index
    )
    matching_cost = (assignment * distances).sum(dim=(2, 3)) / float(N_AGENTS)
    focal_cost = (
        assignment[:, :, self_agent_index, :]
        * distances[:, :, self_agent_index, :]
    ).sum(dim=2)
    crowding = _active_gsp_v3_features_tensor(raw)[:, :, 1]

    singular_values = torch.linalg.svdvals(assignment)
    spectral_power = singular_values.square()
    spectral_probability = spectral_power / spectral_power.sum(
        dim=2, keepdim=True
    ).clamp_min(1e-8)
    spectral_entropy = -(
        spectral_probability
        * torch.log(spectral_probability.clamp_min(1e-8))
    ).sum(dim=2)
    quantities = torch.stack(
        [
            matching_cost,
            focal_cost,
            crowding,
            spectral_entropy,
            singular_values[:, :, 1],
            singular_values[:, :, 2],
        ],
        dim=2,
    )
    return quantities - quantities[:, 0:1, :]


class ActionScoringPolicy(nn.Module):
    """Shared action scorer over raw obs, candidate-action features, and action id."""

    feature_mode = "active_gsp_v3"
    feature_scale_values = (1.0, 1.0, 1.0, 1.0)
    prior_weights_values = (0.0, 0.0, 0.0, 0.0)
    prior_logit_scale = 0.0
    learned_logit_scale = 1.0
    zero_init_scorer = False

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True,
                 agent_index=0):
        super(ActionScoringPolicy, self).__init__()
        if out_dim != N_ACTIONS:
            raise ValueError(
                f"ActionScoringPolicy expects {N_ACTIONS} discrete actions, got {out_dim}."
            )
        self.nonlin = nonlin
        self.out_dim = out_dim
        self.raw_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.feature_encoder = nn.Sequential(
            nn.Linear(ACTIVE_GSP_FEATURES_PER_ACTION, hidden_dim),
            nn.ReLU(),
        )
        self.action_embedding = nn.Embedding(out_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.register_buffer(
            "action_indices",
            torch.arange(out_dim, dtype=torch.long),
        )
        self.register_buffer(
            "feature_scale",
            torch.tensor(self.feature_scale_values, dtype=torch.float32).view(1, 1, -1),
            persistent=False,
        )
        self.register_buffer(
            "prior_weights",
            torch.tensor(self.prior_weights_values, dtype=torch.float32).view(1, 1, -1),
            persistent=False,
        )
        if self.zero_init_scorer:
            self.scorer[-1].weight.data.zero_()
            self.scorer[-1].bias.data.zero_()
        if constrain_out and not discrete_action:
            raise ValueError("ActionScoringPolicy is only intended for discrete actions.")
        self.out_fn = lambda x: x

    def _action_features(self, X):
        features = _active_gsp_v3_features_tensor(X)
        if self.feature_mode == "action_score_control":
            return torch.zeros_like(features)
        if self.feature_mode == "action_raw_potential":
            masked = torch.zeros_like(features)
            masked[:, :, 0:2] = features[:, :, 0:2]
            return masked
        if self.feature_mode == "active_gsp_v3":
            return features
        raise ValueError(f"Unknown action feature mode: {self.feature_mode}")

    def forward(self, X):
        batch_size = X.shape[0]
        raw_action_features = self._action_features(X)
        action_features = raw_action_features * self.feature_scale.to(
            device=X.device,
            dtype=X.dtype,
        )
        raw_hidden = self.raw_encoder(X).unsqueeze(1).expand(-1, self.out_dim, -1)
        feature_hidden = self.feature_encoder(action_features)
        action_hidden = self.action_embedding(self.action_indices)
        action_hidden = action_hidden.unsqueeze(0).expand(batch_size, -1, -1)
        scorer_input = torch.cat([raw_hidden, feature_hidden, action_hidden], dim=2)
        logits = float(self.learned_logit_scale) * self.scorer(scorer_input).squeeze(2)
        if self.prior_logit_scale != 0.0:
            prior_weights = self.prior_weights.to(device=X.device, dtype=X.dtype)
            prior_score = (raw_action_features * prior_weights).sum(dim=2)
            logits = logits - float(self.prior_logit_scale) * prior_score
        return self.out_fn(logits)


class ActiveGSPV3Policy(ActionScoringPolicy):
    feature_mode = "active_gsp_v3"


class ActionRawPotentialPolicy(ActionScoringPolicy):
    feature_mode = "action_raw_potential"


class ActionScoreControlPolicy(ActionScoringPolicy):
    feature_mode = "action_score_control"


class ScaledActiveGSPV3Policy(ActiveGSPV3Policy):
    # Based on 12k local-observation feature stats:
    # p95_abs ~= [0.090, 0.014, 0.053, 0.009].
    feature_scale_values = (10.0, 70.0, 20.0, 110.0)


class ScaledActionRawPotentialPolicy(ActionRawPotentialPolicy):
    feature_scale_values = (10.0, 70.0, 20.0, 110.0)


class LearnedActionFeatureResidualPolicy(nn.Module):
    """Raw MLP actor with a learned, action-conditioned local-graph residual."""

    feature_mode = "active_gsp_v3"
    feature_scale_values = (10.0, 70.0, 20.0, 110.0)
    feature_dim = ACTIVE_GSP_FEATURES_PER_ACTION

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True,
                 agent_index=0):
        super(LearnedActionFeatureResidualPolicy, self).__init__()
        if input_dim != RAW_OBS_DIM or out_dim != N_ACTIONS:
            raise ValueError(
                f"Learned action residual expects {RAW_OBS_DIM}D input and "
                f"{N_ACTIONS} actions, got {input_dim}D and {out_dim}."
            )
        if constrain_out and not discrete_action:
            raise ValueError("Learned action residual is only intended for discrete actions.")

        self.agent_index = int(agent_index)
        self.mlp = MLPNetwork(
            input_dim,
            out_dim,
            hidden_dim=hidden_dim,
            nonlin=nonlin,
            constrain_out=constrain_out,
            norm_in=norm_in,
            discrete_action=discrete_action,
            agent_index=agent_index,
        )
        residual_dim = max(16, hidden_dim // 2)
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, residual_dim),
            nn.ReLU(),
        )
        self.feature_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, residual_dim),
            nn.ReLU(),
        )
        self.action_embedding = nn.Embedding(out_dim, residual_dim)
        self.residual_scorer = nn.Sequential(
            nn.Linear(residual_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.residual_scorer[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.residual_scorer[-1].bias)
        self.register_buffer(
            "action_indices", torch.arange(out_dim, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "feature_scale",
            torch.tensor(self.feature_scale_values, dtype=torch.float32).view(1, 1, -1),
            persistent=False,
        )
        self.last_raw_input_shape = None
        self.last_action_feature_shape = None

    def action_features(self, X):
        if self.feature_mode in {"vector_signal_geom", "vector_signal_gsp"}:
            return vector_signal_features_tensor(X, self.feature_mode)
        if self.feature_mode == "active_multiscale_spectral":
            return multiscale_spectral_action_features(X, self.agent_index)
        if self.feature_mode in {"action_matching_control", "active_spectral_matching"}:
            features = _action_conditioned_spectral_matching_features_tensor(
                X, self.agent_index
            )
            if self.feature_mode == "action_matching_control":
                masked = features.clone()
                masked[:, :, 3:] = 0.0
                return masked
            return features
        if self.feature_mode in {"action_scf_control", "active_scf"}:
            features = _action_conditioned_spectral_coordinate_features_tensor(
                X, self.agent_index
            )
            if self.feature_mode == "action_scf_control":
                masked = features.clone()
                masked[:, :, 2:] = 0.0
                return masked
            return features
        if self.feature_mode in {"action_rwse_control", "active_rwse"}:
            features = _action_conditioned_rwse_features_tensor(X, self.agent_index)
            if self.feature_mode == "action_rwse_control":
                masked = features.clone()
                masked[:, :, 2:] = 0.0
                return masked
            return features
        features = _active_gsp_v3_features_tensor(X)
        if self.feature_mode == "action_raw_potential":
            masked = torch.zeros_like(features)
            masked[:, :, :2] = features[:, :, :2]
            return masked
        if self.feature_mode == "active_gsp_crowding_energy":
            masked = features.clone()
            masked[:, :, 2] = 0.0
            return masked
        if self.feature_mode == "active_gsp_v3":
            return features
        raise ValueError(f"Unknown learned action residual mode: {self.feature_mode}")

    def forward(self, X):
        if X.ndim != 2 or X.shape[1] != RAW_OBS_DIM:
            raise ValueError(
                f"Learned action residual expects [batch, {RAW_OBS_DIM}], "
                f"got {tuple(X.shape)}."
            )
        batch_size = X.shape[0]
        features = self.action_features(X)
        scaled_features = torch.tanh(
            features * self.feature_scale.to(device=X.device, dtype=X.dtype)
        )
        context = self.context_encoder(X).unsqueeze(1).expand(-1, N_ACTIONS, -1)
        feature_hidden = self.feature_encoder(scaled_features)
        action_hidden = self.action_embedding(self.action_indices)
        action_hidden = action_hidden.unsqueeze(0).expand(batch_size, -1, -1)
        residual_input = torch.cat([context, feature_hidden, action_hidden], dim=2)
        residual = self.residual_scorer(residual_input).squeeze(2)
        self.last_raw_input_shape = tuple(X.shape)
        self.last_action_feature_shape = tuple(features.shape)
        return self.mlp(X) + residual


class LearnedRawPotentialResidualPolicy(LearnedActionFeatureResidualPolicy):
    """Non-spectral action-potential control with the same learned residual."""

    feature_mode = "action_raw_potential"


class LearnedActiveGSPResidualPolicy(LearnedActionFeatureResidualPolicy):
    """Purely learned Active GSP action residual without a handcrafted prior."""

    feature_mode = "active_gsp_v3"


class CounterfactualActiveProbePolicy(nn.Module):
    """Shared candidate scorer initialized from a counterfactual task-value probe.

    This remains a decentralized 18D-local-observation actor.  The exported
    probe predicts one-step Hungarian-distance delta; lower predicted delta is
    converted to a higher action logit.  All scorer weights remain trainable.
    """

    feature_dim = 4

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True,
                 agent_index=0):
        super().__init__()
        if input_dim != RAW_OBS_DIM or out_dim != N_ACTIONS:
            raise ValueError("Counterfactual probe actor requires 18D input and 5 actions.")
        self.agent_index = int(agent_index)
        scorer_input_dim = RAW_OBS_DIM + N_ACTIONS + self.feature_dim
        self.scorer = nn.Sequential(
            nn.Linear(scorer_input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.register_buffer("action_eye", torch.eye(N_ACTIONS), persistent=False)
        self.register_buffer("x_mean", torch.zeros(scorer_input_dim))
        self.register_buffer("x_std", torch.ones(scorer_input_dim))
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))
        self.last_raw_input_shape = None
        self.last_action_feature_shape = None

    def load_probe_payload(self, payload, load_weights=True):
        if load_weights and "actor_state_dict" in payload:
            self.load_state_dict(payload["actor_state_dict"])
            return
        self.x_mean.copy_(torch.as_tensor(payload["x_mean"], dtype=self.x_mean.dtype))
        self.x_std.copy_(torch.as_tensor(payload["x_std"], dtype=self.x_std.dtype))
        self.y_mean.copy_(torch.as_tensor(payload["y_mean"][0:1], dtype=self.y_mean.dtype))
        self.y_std.copy_(torch.as_tensor(payload["y_std"][0:1], dtype=self.y_std.dtype))
        if load_weights:
            source = payload["probe_state_dict"]
            self.scorer[0].weight.data.copy_(source["network.0.weight"])
            self.scorer[0].bias.data.copy_(source["network.0.bias"])
            self.scorer[2].weight.data.copy_(source["network.2.weight"])
            self.scorer[2].bias.data.copy_(source["network.2.bias"])
            self.scorer[4].weight.data.copy_(source["network.4.weight"][0:1])
            self.scorer[4].bias.data.copy_(source["network.4.bias"][0:1])

    def forward(self, X):
        if X.ndim != 2 or X.shape[1] != RAW_OBS_DIM:
            raise ValueError(f"Expected [batch,{RAW_OBS_DIM}], got {tuple(X.shape)}")
        batch = X.shape[0]
        features = _active_gsp_v3_features_tensor(X)
        raw = X[:, None, :].expand(-1, N_ACTIONS, -1)
        actions = self.action_eye.to(X).unsqueeze(0).expand(batch, -1, -1)
        scorer_input = torch.cat([raw, actions, features], dim=2)
        normalized = (scorer_input - self.x_mean.to(X)) / self.x_std.to(X).clamp_min(1e-6)
        prediction = self.scorer(normalized).squeeze(2) * self.y_std.to(X) + self.y_mean.to(X)
        self.last_raw_input_shape = tuple(X.shape)
        self.last_action_feature_shape = tuple(features.shape)
        return -prediction


class ScalableCounterfactual6x6Policy(nn.Module):
    """Twelve-node, 36D-local-observation version of the learned scorer."""

    feature_dim = 4
    input_dim = 36

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True,
                 agent_index=0):
        super().__init__()
        if input_dim != self.input_dim or out_dim != N_ACTIONS:
            raise ValueError("Scalable 6x6 actor requires 36D input and 5 actions")
        self.agent_index = int(agent_index)
        scorer_input_dim = input_dim + N_ACTIONS + self.feature_dim
        self.scorer = nn.Sequential(
            nn.Linear(scorer_input_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1),
        )
        self.register_buffer("action_eye", torch.eye(N_ACTIONS), persistent=False)
        self.register_buffer("x_mean", torch.zeros(scorer_input_dim))
        self.register_buffer("x_std", torch.ones(scorer_input_dim))
        self.last_raw_input_shape = None
        self.last_action_feature_shape = None

    def load_probe_payload(self, payload, load_weights=True):
        if "x_mean" in payload:
            self.x_mean.copy_(torch.as_tensor(payload["x_mean"], dtype=self.x_mean.dtype))
            self.x_std.copy_(torch.as_tensor(payload["x_std"], dtype=self.x_std.dtype))
        if load_weights and "actor_state_dict" in payload:
            self.load_state_dict(payload["actor_state_dict"])

    def forward(self, X):
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"Expected [batch,{self.input_dim}], got {tuple(X.shape)}")
        batch = X.shape[0]
        features = scalable_active_action_features_tensor(
            X, self.agent_index, n_agents=6, n_landmarks=6
        )
        raw = X[:, None, :].expand(-1, N_ACTIONS, -1)
        actions = self.action_eye.to(X).unsqueeze(0).expand(batch, -1, -1)
        scorer_input = torch.cat([raw, actions, features], dim=2)
        normalized = (scorer_input - self.x_mean.to(X)) / self.x_std.to(X).clamp_min(1e-6)
        logits = self.scorer(normalized).squeeze(2)
        self.last_raw_input_shape = tuple(X.shape)
        self.last_action_feature_shape = tuple(features.shape)
        return logits


class EquivariantMatchingSafetyPolicy(nn.Module):
    """Cardinality-parameterized equivariant matching/safety actor.

    The one-to-one structure is imposed only through Sinkhorn normalization.
    Pair scores, pair embeddings, safety embeddings, and final action logits
    are learned; no feature is assigned a fixed beneficial action sign.

    For the unchanged MPE ``simple_spread`` observation callback with two
    communication channels, a balanced N-agent/N-landmark local observation
    has dimension 6N. Cardinality is therefore inferred from ``input_dim``.
    """

    pair_hidden_dim = 16
    sinkhorn_iterations = 64

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True,
                 agent_index=0):
        super().__init__()
        if input_dim % 6 != 0 or input_dim < 12 or out_dim != N_ACTIONS:
            raise ValueError(
                "Equivariant NxN actor requires a 6N-dimensional input "
                "for N>=2 and five actions"
            )
        self.input_dim = int(input_dim)
        self.n_agents = int(input_dim // 6)
        self.n_landmarks = self.n_agents
        self.agent_index = int(agent_index)
        if not 0 <= self.agent_index < self.n_agents:
            raise ValueError("agent_index is outside the inferred cardinality")
        self.pair_encoder = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, self.pair_hidden_dim), nn.ReLU()
        )
        self.pair_scorer = nn.Sequential(
            nn.Linear(self.pair_hidden_dim, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        self.safety_encoder = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, self.pair_hidden_dim), nn.ReLU()
        )
        summary_dim = self.pair_hidden_dim * 4 + 4
        self.action_scorer = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.last_matching_shape = None
        self.last_matching_row_error = None
        self.last_matching_column_error = None
        self.last_raw_input_shape = None

    def _reconstruct_batch(self, raw, self_indices):
        batch = raw.shape[0]
        landmark_end = 4 + 2 * self.n_landmarks
        other_end = landmark_end + 2 * (self.n_agents - 1)
        landmarks = raw[:, 4:landmark_end].reshape(
            batch, self.n_landmarks, 2
        )
        other_agents = raw[:, landmark_end:other_end].reshape(
            batch, self.n_agents - 1, 2
        )
        agents = raw.new_zeros(batch, self.n_agents, 2)
        for self_index in range(self.n_agents):
            mask = self_indices == self_index
            if not torch.any(mask):
                continue
            global_indices = [
                index for index in range(self.n_agents)
                if index != self_index
            ]
            for local_index, global_index in enumerate(global_indices):
                agents[mask, global_index, :] = other_agents[mask, local_index, :]
        return agents, landmarks

    @staticmethod
    def _sinkhorn(log_scores, iterations):
        value = log_scores
        for _ in range(iterations):
            value = value - torch.logsumexp(value, dim=3, keepdim=True)
            value = value - torch.logsumexp(value, dim=2, keepdim=True)
        return torch.exp(value)

    def load_probe_payload(self, payload, load_weights=True):
        if load_weights and "actor_state_dict" in payload:
            self.load_state_dict(payload["actor_state_dict"])

    def forward_with_agent_indices(self, X, self_indices):
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"Expected [batch,{self.input_dim}], got {tuple(X.shape)}")
        self_indices = torch.as_tensor(self_indices, device=X.device, dtype=torch.long).reshape(-1)
        if (
            self_indices.numel() != X.shape[0]
            or torch.any(self_indices < 0)
            or torch.any(self_indices >= self.n_agents)
        ):
            raise ValueError("One valid self index is required for every observation")
        batch = X.shape[0]
        agents, landmarks = self._reconstruct_batch(X, self_indices)
        candidates = agents[:, None].expand(-1, N_ACTIONS, -1, -1).clone()
        controls = X.new_tensor(ACTION_TO_CONTROL)
        displacement = (
            X[:, None, 0:2] * (1.0 - DEFAULT_DAMPING) * DEFAULT_DT
            + controls[None] * DEFAULT_SENSITIVITY * DEFAULT_DT ** 2
        )
        batch_indices = torch.arange(batch, device=X.device)
        candidates[batch_indices[:, None], torch.arange(N_ACTIONS, device=X.device)[None],
                   self_indices[:, None], :] = displacement

        al_delta = candidates[:, :, :, None, :] - landmarks[:, None, None, :, :]
        al_distance = torch.linalg.vector_norm(al_delta, dim=4)
        focal_agent = (
            torch.arange(self.n_agents, device=X.device)[None, :]
            == self_indices[:, None]
        ).to(X.dtype)
        focal_pair = focal_agent[:, None, :, None].expand(
            -1, N_ACTIONS, -1, self.n_landmarks
        )
        pair_input = torch.stack(
            [torch.tanh(al_distance), torch.tanh(al_distance.square()), focal_pair], dim=4
        )
        pair_hidden = self.pair_encoder(pair_input)
        matching = self._sinkhorn(
            self.pair_scorer(pair_hidden).squeeze(4), self.sinkhorn_iterations
        )
        global_matching = (
            (matching[..., None] * pair_hidden).sum(dim=(2, 3))
            / float(self.n_agents)
        )
        focal_hidden = pair_hidden[batch_indices, :, self_indices, :, :]
        focal_matching = matching[batch_indices, :, self_indices, :]
        focal_summary = (focal_matching[..., None] * focal_hidden).sum(dim=2)
        focal_distance = al_distance[batch_indices, :, self_indices, :]
        focal_cost = (focal_matching * focal_distance).sum(dim=2)
        global_cost = (
            (matching * al_distance).sum(dim=(2, 3))
            / float(self.n_agents)
        )

        aa_delta = candidates[:, :, :, None, :] - candidates[:, :, None, :, :]
        aa_distance = torch.linalg.vector_norm(aa_delta, dim=4)
        index = torch.arange(self.n_agents, device=X.device)
        upper = (index[:, None] < index[None, :])[None, None]
        focal_aa = (
            (index[None, :] == self_indices[:, None])[:, :, None]
            | (index[None, :] == self_indices[:, None])[:, None, :]
        )[:, None].expand(-1, N_ACTIONS, -1, -1)
        safety_input = torch.stack(
            [torch.tanh(aa_distance), torch.tanh(aa_distance.square()),
             focal_aa.to(X.dtype)], dim=4
        )
        safety_hidden = self.safety_encoder(safety_input)
        upper_float = upper.to(X.dtype)
        global_safety = (
            safety_hidden * upper_float[..., None]
        ).sum(dim=(2, 3)) / float(self.n_agents * (self.n_agents - 1) // 2)
        focal_upper = upper & focal_aa
        focal_safety = (
            safety_hidden * focal_upper.to(X.dtype)[..., None]
        ).sum(dim=(2, 3)) / float(self.n_agents - 1)
        masked_global = aa_distance.masked_fill(~upper, float("inf"))
        masked_focal = aa_distance.masked_fill(~focal_upper, float("inf"))
        global_min = masked_global.amin(dim=(2, 3))
        focal_min = masked_focal.amin(dim=(2, 3))
        coverage_cost = (
            al_distance.amin(dim=2).sum(dim=2) / float(self.n_landmarks)
        )

        summary = torch.cat([
            global_matching, focal_summary, global_safety, focal_safety,
            focal_cost[..., None], global_cost[..., None],
            focal_min[..., None], coverage_cost[..., None],
        ], dim=2)
        logits = self.action_scorer(summary).squeeze(2)
        self.last_matching_shape = tuple(matching.shape)
        self.last_matching_row_error = float(
            (matching.sum(dim=3) - 1.0).abs().max().detach().cpu()
        )
        self.last_matching_column_error = float(
            (matching.sum(dim=2) - 1.0).abs().max().detach().cpu()
        )
        self.last_raw_input_shape = tuple(X.shape)
        return logits

    def forward(self, X):
        indices = torch.full(
            (X.shape[0],), self.agent_index, device=X.device, dtype=torch.long
        )
        return self.forward_with_agent_indices(X, indices)


class EquivariantMatchingSafetyPolicy8(EquivariantMatchingSafetyPolicy):
    """Eight-iteration cardinality-parameterized actor used in the N sweep."""

    sinkhorn_iterations = 8


class EquivariantMatchingSafety6x6Policy(EquivariantMatchingSafetyPolicy):
    """Backward-compatible 6x6 class for historical checkpoints."""

    def __init__(self, input_dim, out_dim, *args, **kwargs):
        if input_dim != 36:
            raise ValueError("Equivariant 6x6 actor requires 36D input")
        super().__init__(input_dim, out_dim, *args, **kwargs)


class EquivariantMatchingSafety6x6Policy8(
        EquivariantMatchingSafety6x6Policy):
    """Registered eight-iteration variant for reproducible checkpoints."""

    sinkhorn_iterations = 8


class EquivariantMatchingSafety6x6Policy16(
        EquivariantMatchingSafety6x6Policy):
    """Registered 16-iteration variant for reproducible checkpoints."""

    sinkhorn_iterations = 16


class EquivariantMatchingSafety6x6Policy32(
        EquivariantMatchingSafety6x6Policy):
    """Registered 32-iteration variant for reproducible checkpoints."""

    sinkhorn_iterations = 32


class VectorSignalGeomPolicy(LearnedActionFeatureResidualPolicy):
    """Nine-channel non-spectral vector-signal geometric control."""

    feature_mode = "vector_signal_geom"
    feature_dim = VECTOR_SIGNAL_GEOM_DIM
    feature_scale_values = (1.0,) * VECTOR_SIGNAL_GEOM_DIM


class VectorSignalGSPPolicy(LearnedActionFeatureResidualPolicy):
    """Fifteen-channel three-agent vector-signal GSP actor."""

    feature_mode = "vector_signal_gsp"
    feature_dim = VECTOR_SIGNAL_GSP_DIM
    feature_scale_values = (1.0,) * VECTOR_SIGNAL_GSP_DIM


class LearnedMultiscaleSpectralResidualPolicy(LearnedActionFeatureResidualPolicy):
    """Locked normalized-Laplacian heat-Rayleigh residual actor."""

    feature_mode = "active_multiscale_spectral"
    feature_dim = MULTISCALE_FEATURE_DIM
    # Preserve the geometric-control scaling; spectral channels have bounded
    # normalized-Rayleigh scale and receive no fixed sign or action prior.
    feature_scale_values = (10.0, 70.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


class LearnedCrowdingGSPResidualPolicy(LearnedActionFeatureResidualPolicy):
    """Learned action residual with potentials and crowding Dirichlet energy."""

    feature_mode = "active_gsp_crowding_energy"


class LearnedRWSEPotentialControlPolicy(LearnedActionFeatureResidualPolicy):
    """Five-channel architecture control with RWSE channels masked to zero."""

    feature_mode = "action_rwse_control"
    feature_dim = 5
    feature_scale_values = (10.0, 70.0, 20.0, 20.0, 20.0)


class LearnedActiveRWSEResidualPolicy(LearnedActionFeatureResidualPolicy):
    """Purely learned action-conditioned focal RWSE residual actor."""

    feature_mode = "active_rwse"
    feature_dim = 5
    feature_scale_values = (10.0, 70.0, 20.0, 20.0, 20.0)


class LearnedSCFPotentialControlPolicy(LearnedActionFeatureResidualPolicy):
    """Six-channel architecture control with spectral-coordinate channels masked."""

    feature_mode = "action_scf_control"
    feature_dim = 6
    feature_scale_values = (10.0, 70.0, 10.0, 10.0, 10.0, 10.0)


class LearnedActiveSCFResidualPolicy(LearnedActionFeatureResidualPolicy):
    """Learned directional spectral-coordinate graph-filter residual actor."""

    feature_mode = "active_scf"
    feature_dim = 6
    feature_scale_values = (10.0, 70.0, 10.0, 10.0, 10.0, 10.0)


class GatedDirichletResidualPolicy(nn.Module):
    """Raw actor with separated potential and adaptive spectral-energy paths."""

    use_dirichlet_energy = True
    potential_scale_values = (10.0, 70.0)
    energy_scale_values = (20.0, 110.0)
    energy_residual_scale = 0.1

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True,
                 agent_index=0):
        super(GatedDirichletResidualPolicy, self).__init__()
        if input_dim != RAW_OBS_DIM or out_dim != N_ACTIONS:
            raise ValueError(
                f"Gated Dirichlet actor expects {RAW_OBS_DIM}D input and "
                f"{N_ACTIONS} actions, got {input_dim}D and {out_dim}."
            )
        if constrain_out and not discrete_action:
            raise ValueError("Gated Dirichlet actor is only intended for discrete actions.")

        self.agent_index = int(agent_index)
        self.mlp = MLPNetwork(
            input_dim,
            out_dim,
            hidden_dim=hidden_dim,
            nonlin=nonlin,
            constrain_out=constrain_out,
            norm_in=norm_in,
            discrete_action=discrete_action,
            agent_index=agent_index,
        )
        residual_dim = max(16, hidden_dim // 2)
        self.context_encoder = nn.Sequential(
            nn.Linear(input_dim, residual_dim),
            nn.ReLU(),
        )
        self.potential_encoder = nn.Sequential(
            nn.Linear(2, residual_dim),
            nn.ReLU(),
        )
        self.action_embedding = nn.Embedding(out_dim, residual_dim)
        self.potential_scorer = nn.Sequential(
            nn.Linear(residual_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.normal_(self.potential_scorer[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.potential_scorer[-1].bias)

        self.energy_gate = nn.Sequential(
            nn.Linear(residual_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        nn.init.zeros_(self.energy_gate[-1].weight)
        nn.init.zeros_(self.energy_gate[-1].bias)
        self.register_buffer(
            "action_indices", torch.arange(out_dim, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "potential_scale",
            torch.tensor(self.potential_scale_values, dtype=torch.float32).view(1, 1, 2),
            persistent=False,
        )
        self.register_buffer(
            "energy_scale",
            torch.tensor(self.energy_scale_values, dtype=torch.float32).view(1, 1, 2),
            persistent=False,
        )
        self.last_raw_input_shape = None
        self.last_action_feature_shape = None
        self.last_energy_gate_shape = None
        self.last_energy_gate_mean = None

    def action_features(self, X):
        features = _active_gsp_v3_features_tensor(X)
        if not self.use_dirichlet_energy:
            features = features.clone()
            features[:, :, 2:] = 0.0
        return features

    def forward(self, X):
        if X.ndim != 2 or X.shape[1] != RAW_OBS_DIM:
            raise ValueError(
                f"Gated Dirichlet actor expects [batch, {RAW_OBS_DIM}], "
                f"got {tuple(X.shape)}."
            )
        batch_size = X.shape[0]
        features = self.action_features(X)
        potentials = torch.tanh(
            features[:, :, :2]
            * self.potential_scale.to(device=X.device, dtype=X.dtype)
        )
        energies = torch.tanh(
            features[:, :, 2:]
            * self.energy_scale.to(device=X.device, dtype=X.dtype)
        )
        context = self.context_encoder(X).unsqueeze(1).expand(-1, N_ACTIONS, -1)
        potential_hidden = self.potential_encoder(potentials)
        action_hidden = self.action_embedding(self.action_indices)
        action_hidden = action_hidden.unsqueeze(0).expand(batch_size, -1, -1)
        shared = torch.cat([context, potential_hidden, action_hidden], dim=2)
        potential_residual = self.potential_scorer(shared).squeeze(2)
        gates = torch.tanh(self.energy_gate(shared))
        energy_residual = (gates * energies).sum(dim=2)

        self.last_raw_input_shape = tuple(X.shape)
        self.last_action_feature_shape = tuple(features.shape)
        self.last_energy_gate_shape = tuple(gates.shape)
        self.last_energy_gate_mean = gates.detach().mean(dim=(0, 1)).cpu()
        return (
            self.mlp(X)
            + potential_residual
            + float(self.energy_residual_scale) * energy_residual
        )


class LearnedGatedDirichletControlPolicy(GatedDirichletResidualPolicy):
    """Exact architecture control with both Dirichlet channels masked."""

    use_dirichlet_energy = False


class LearnedGatedDirichletGSPPolicy(GatedDirichletResidualPolicy):
    """Adaptive two-channel Dirichlet-energy actor without fixed priors."""

    use_dirichlet_energy = True


class LearnedGatedDirichletGSP025Policy(LearnedGatedDirichletGSPPolicy):
    """Intermediate bounded Dirichlet correction with maximum magnitude 0.5."""

    energy_residual_scale = 0.25


class LearnedMatchingControlPolicy(LearnedActionFeatureResidualPolicy):
    """Matched assignment-cost actor with spectral channels masked."""

    feature_mode = "action_matching_control"
    feature_dim = 6
    feature_scale_values = (60.0, 25.0, 70.0, 70.0, 90.0, 120.0)


class LearnedSpectralMatchingResidualPolicy(LearnedActionFeatureResidualPolicy):
    """Action-conditioned soft-matching graph spectrum residual actor."""

    feature_mode = "active_spectral_matching"
    feature_dim = 6
    feature_scale_values = (60.0, 25.0, 70.0, 70.0, 90.0, 120.0)


class LinearMatchingResidualPolicy(nn.Module):
    """Raw MLP plus a zero-initialized learned linear matching-feature readout."""

    use_matching_spectrum = True
    feature_scale_values = (60.0, 25.0, 70.0, 70.0, 90.0, 120.0)

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True,
                 agent_index=0):
        super(LinearMatchingResidualPolicy, self).__init__()
        if input_dim != RAW_OBS_DIM or out_dim != N_ACTIONS:
            raise ValueError(
                f"Linear matching actor expects {RAW_OBS_DIM}D input and "
                f"{N_ACTIONS} actions, got {input_dim}D and {out_dim}."
            )
        if constrain_out and not discrete_action:
            raise ValueError("Linear matching actor is only intended for discrete actions.")
        self.agent_index = int(agent_index)
        self.mlp = MLPNetwork(
            input_dim,
            out_dim,
            hidden_dim=hidden_dim,
            nonlin=nonlin,
            constrain_out=constrain_out,
            norm_in=norm_in,
            discrete_action=discrete_action,
            agent_index=agent_index,
        )
        self.feature_weights = nn.Parameter(torch.zeros(6, dtype=torch.float32))
        self.register_buffer(
            "feature_scale",
            torch.tensor(self.feature_scale_values, dtype=torch.float32).view(1, 1, 6),
            persistent=False,
        )
        self.last_raw_input_shape = None
        self.last_action_feature_shape = None

    def action_features(self, X):
        features = _action_conditioned_spectral_matching_features_tensor(
            X, self.agent_index
        )
        if not self.use_matching_spectrum:
            features = features.clone()
            features[:, :, 3:] = 0.0
        return features

    def forward(self, X):
        if X.ndim != 2 or X.shape[1] != RAW_OBS_DIM:
            raise ValueError(
                f"Linear matching actor expects [batch, {RAW_OBS_DIM}], "
                f"got {tuple(X.shape)}."
            )
        features = self.action_features(X)
        scaled = torch.tanh(
            features * self.feature_scale.to(device=X.device, dtype=X.dtype)
        )
        residual = (
            scaled
            * self.feature_weights.to(device=X.device, dtype=X.dtype).view(1, 1, 6)
        ).sum(dim=2)
        self.last_raw_input_shape = tuple(X.shape)
        self.last_action_feature_shape = tuple(features.shape)
        return self.mlp(X) + residual


class LinearMatchingControlPolicy(LinearMatchingResidualPolicy):
    """Linear assignment-cost control with all spectral inputs masked."""

    use_matching_spectrum = False


class LinearSpectralMatchingPolicy(LinearMatchingResidualPolicy):
    """Learned linear readout of assignment costs and matching spectrum."""

    use_matching_spectrum = True


class MatchingOnlyPolicy(nn.Module):
    """Minimal actor whose logits are a learned matching-feature graph filter."""

    use_matching_spectrum = True
    feature_scale_values = (60.0, 25.0, 70.0, 70.0, 90.0, 120.0)

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True,
                 agent_index=0):
        super(MatchingOnlyPolicy, self).__init__()
        if input_dim != RAW_OBS_DIM or out_dim != N_ACTIONS:
            raise ValueError(
                f"Matching-only actor expects {RAW_OBS_DIM}D input and "
                f"{N_ACTIONS} actions, got {input_dim}D and {out_dim}."
            )
        if constrain_out and not discrete_action:
            raise ValueError("Matching-only actor is only intended for discrete actions.")
        self.agent_index = int(agent_index)
        self.actor_input_dim = input_dim
        self.feature_weights = nn.Parameter(torch.zeros(6, dtype=torch.float32))
        self.register_buffer(
            "feature_scale",
            torch.tensor(self.feature_scale_values, dtype=torch.float32).view(1, 1, 6),
            persistent=False,
        )
        self.last_raw_input_shape = None
        self.last_action_feature_shape = None

    def action_features(self, X):
        features = _action_conditioned_spectral_matching_features_tensor(
            X, self.agent_index
        )
        if not self.use_matching_spectrum:
            features = features.clone()
            features[:, :, 3:] = 0.0
        return features

    def forward(self, X):
        if X.ndim != 2 or X.shape[1] != RAW_OBS_DIM:
            raise ValueError(
                f"Matching-only actor expects [batch, {RAW_OBS_DIM}], "
                f"got {tuple(X.shape)}."
            )
        features = self.action_features(X)
        scaled = torch.tanh(
            features * self.feature_scale.to(device=X.device, dtype=X.dtype)
        )
        logits = (
            scaled
            * self.feature_weights.to(device=X.device, dtype=X.dtype).view(1, 1, 6)
        ).sum(dim=2)
        self.last_raw_input_shape = tuple(X.shape)
        self.last_action_feature_shape = tuple(features.shape)
        return logits


class MatchingOnlyControlPolicy(MatchingOnlyPolicy):
    use_matching_spectrum = False


class SpectralMatchingOnlyPolicy(MatchingOnlyPolicy):
    use_matching_spectrum = True


class ActionRawPotentialPriorPolicy(ActionRawPotentialPolicy):
    prior_weights_values = (1.0, 1.0, 0.0, 0.0)
    prior_logit_scale = 20.0
    zero_init_scorer = True


class ActiveGSPV3PriorPolicy(ActiveGSPV3Policy):
    prior_weights_values = (1.0, 1.0, 1.0, 1.0)
    prior_logit_scale = 20.0
    zero_init_scorer = True


class ActionRawPotentialResidualPolicy(ActionRawPotentialPriorPolicy):
    learned_logit_scale = 0.05


class ActiveGSPV3ResidualPolicy(ActiveGSPV3PriorPolicy):
    learned_logit_scale = 0.05


class FixedActionHeuristicPolicy(nn.Module):
    """Non-learning action-feature heuristic for integrated pipeline checks."""

    feature_mode = "active_gsp_v3"
    prior_weights_values = (1.0, 1.0, 1.0, 1.0)
    prior_logit_scale = 20.0

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True,
                 agent_index=0):
        super(FixedActionHeuristicPolicy, self).__init__()
        if out_dim != N_ACTIONS:
            raise ValueError(
                f"FixedActionHeuristicPolicy expects {N_ACTIONS} actions, got {out_dim}."
            )
        if constrain_out and not discrete_action:
            raise ValueError("FixedActionHeuristicPolicy is only intended for discrete actions.")
        self.dummy = nn.Parameter(torch.zeros(1))
        self.register_buffer(
            "prior_weights",
            torch.tensor(self.prior_weights_values, dtype=torch.float32).view(1, 1, -1),
            persistent=False,
        )

    def _action_features(self, X):
        features = _active_gsp_v3_features_tensor(X)
        if self.feature_mode == "action_raw_potential":
            masked = torch.zeros_like(features)
            masked[:, :, 0:2] = features[:, :, 0:2]
            return masked
        if self.feature_mode == "active_gsp_v3":
            return features
        raise ValueError(f"Unknown fixed heuristic mode: {self.feature_mode}")

    def forward(self, X):
        features = self._action_features(X)
        prior_weights = self.prior_weights.to(device=X.device, dtype=X.dtype)
        score = (features * prior_weights).sum(dim=2)
        return -float(self.prior_logit_scale) * score + self.dummy * 0.0


class ActionRawPotentialFixedPolicy(FixedActionHeuristicPolicy):
    feature_mode = "action_raw_potential"
    prior_weights_values = (1.0, 1.0, 0.0, 0.0)


class ActiveGSPV3FixedPolicy(FixedActionHeuristicPolicy):
    feature_mode = "active_gsp_v3"
    prior_weights_values = (1.0, 1.0, 1.0, 1.0)


class GatedGraphPolicy(nn.Module):
    """Actor with learnable edge gates over the fixed task-graph candidate edges."""

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True,
                 agent_index=0):
        super(GatedGraphPolicy, self).__init__()
        self.agent_index = agent_index
        self.nonlin = nonlin
        self.node_encoder = nn.Linear(7, hidden_dim)
        self.edge_gate = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.raw_encoder = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        if constrain_out and not discrete_action:
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:
            self.out_fn = lambda x: x

    def forward(self, X):
        positions, node_features = _reconstruct_nodes(X, self.agent_index)
        prior = _task_edge_prior(positions)
        batch_size, n_nodes, _ = positions.shape
        gates = torch.zeros_like(prior)

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if prior[:, i, j].abs().sum() == 0:
                    continue
                delta = positions[:, i, :] - positions[:, j, :]
                dist = torch.norm(delta, dim=1, keepdim=True)
                edge_feat = torch.cat([delta, dist, node_features[:, i, 4:5],
                                       node_features[:, j, 4:5]], dim=1)
                gate = torch.sigmoid(self.edge_gate(edge_feat)).squeeze(1)
                gates[:, i, j] = gate
                gates[:, j, i] = gate

        adjacency = prior * gates
        degree = adjacency.sum(dim=2, keepdim=True).clamp(min=1e-6)
        node_hidden = self.nonlin(self.node_encoder(node_features))
        aggregated = torch.bmm(adjacency / degree, node_hidden)
        self_embedding = aggregated[:, self.agent_index, :]
        raw_embedding = self.nonlin(self.raw_encoder(X))
        h = self.nonlin(self.fc1(torch.cat([raw_embedding, self_embedding], dim=1)))
        h = self.nonlin(self.fc2(h))
        return self.out_fn(self.fc3(h))


class MessageGraphPolicy(nn.Module):
    """Actor with one learned active-attention message passing layer."""

    def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.relu,
                 constrain_out=False, norm_in=True, discrete_action=True,
                 agent_index=0):
        super(MessageGraphPolicy, self).__init__()
        self.agent_index = agent_index
        self.nonlin = nonlin
        self.node_encoder = nn.Linear(7, hidden_dim)
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.message = nn.Linear(hidden_dim, hidden_dim)
        self.raw_encoder = nn.Linear(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        if constrain_out and not discrete_action:
            self.fc3.weight.data.uniform_(-3e-3, 3e-3)
            self.out_fn = F.tanh
        else:
            self.out_fn = lambda x: x

    def forward(self, X):
        positions, node_features = _reconstruct_nodes(X, self.agent_index)
        prior = _task_edge_prior(positions)
        node_hidden = self.nonlin(self.node_encoder(node_features))
        q = self.query(node_hidden)
        k = self.key(node_hidden)
        scores = torch.bmm(q, k.transpose(1, 2)) / (q.shape[-1] ** 0.5)
        mask = prior > 0
        eye = torch.eye(prior.shape[1], device=prior.device, dtype=torch.bool).unsqueeze(0)
        mask = mask | eye
        scores = scores.masked_fill(~mask, -1e9)
        attention = F.softmax(scores, dim=2)
        messages = torch.bmm(attention, self.message(node_hidden))
        self_embedding = self.nonlin(node_hidden[:, self.agent_index, :] +
                                     messages[:, self.agent_index, :])
        raw_embedding = self.nonlin(self.raw_encoder(X))
        h = self.nonlin(self.fc1(torch.cat([raw_embedding, self_embedding], dim=1)))
        h = self.nonlin(self.fc2(h))
        return self.out_fn(self.fc3(h))
