"""Cardinality-explicit action-conditioned task features for simple_spread."""

import torch


N_ACTIONS = 5
ACTION_TO_CONTROL = (
    (0.0, 0.0), (1.0, 0.0), (-1.0, 0.0),
    (0.0, 1.0), (0.0, -1.0),
)
DT = 0.1
DAMPING = 0.25
SENSITIVITY = 5.0
AA_SCALE = 0.25
SIGMA_AA = 0.80
SIGMA_AL = 0.60


def expected_observation_dim(n_agents, n_landmarks, communication_dim=2):
    return 4 + 2 * n_landmarks + (2 + communication_dim) * (n_agents - 1)


def reconstruct_relative_geometry(raw, self_agent_index, n_agents, n_landmarks):
    expected = expected_observation_dim(n_agents, n_landmarks)
    if raw.ndim != 2 or raw.shape[1] != expected:
        raise ValueError(f"Expected [batch,{expected}], got {tuple(raw.shape)}")
    if not 0 <= int(self_agent_index) < n_agents:
        raise ValueError("self_agent_index is outside the configured cardinality")
    batch = raw.shape[0]
    landmark_end = 4 + 2 * n_landmarks
    other_end = landmark_end + 2 * (n_agents - 1)
    landmarks = raw[:, 4:landmark_end].reshape(batch, n_landmarks, 2)
    other_agents = raw[:, landmark_end:other_end].reshape(batch, n_agents - 1, 2)
    agents = torch.zeros(batch, n_agents, 2, dtype=raw.dtype, device=raw.device)
    other_indices = [index for index in range(n_agents) if index != self_agent_index]
    for local_index, global_index in enumerate(other_indices):
        agents[:, global_index, :] = other_agents[:, local_index, :]
    return agents, landmarks


def _gaussian(distance, sigma, scale=1.0):
    return scale * torch.exp(-(distance.square()) / (2.0 * sigma * sigma))


def _state_quantities(agent_positions, landmark_positions):
    # Shapes: [batch, action, agent/landmark, xy].
    al_distance = torch.linalg.vector_norm(
        agent_positions[:, :, :, None, :] - landmark_positions[:, None, None, :, :],
        dim=-1,
    )
    aa_distance = torch.linalg.vector_norm(
        agent_positions[:, :, :, None, :] - agent_positions[:, :, None, :, :],
        dim=-1,
    )
    n_agents = agent_positions.shape[2]
    eye = torch.eye(n_agents, dtype=torch.bool, device=agent_positions.device)
    masked_aa = aa_distance.masked_fill(eye[None, None], float("inf"))

    coverage_residual = al_distance.min(dim=2).values
    coverage_potential = coverage_residual.sum(dim=2)
    upper = torch.triu(torch.ones_like(aa_distance, dtype=torch.bool), diagonal=1)
    crowding_potential = _gaussian(aa_distance, SIGMA_AA, AA_SCALE).masked_fill(
        ~upper, 0.0
    ).sum(dim=(2, 3))

    al_weight = _gaussian(al_distance, SIGMA_AL)
    aa_weight = _gaussian(aa_distance, SIGMA_AA, AA_SCALE)
    coverage_energy = (
        al_weight * coverage_residual[:, :, None, :].square()
    ).sum(dim=(2, 3))
    crowd_signal = _gaussian(masked_aa.min(dim=3).values, SIGMA_AA, AA_SCALE)
    crowd_al_energy = (
        al_weight * crowd_signal[:, :, :, None].square()
    ).sum(dim=(2, 3))
    signal_difference = crowd_signal[:, :, :, None] - crowd_signal[:, :, None, :]
    crowd_aa_energy = (
        aa_weight * signal_difference.square()
    ).masked_fill(~upper, 0.0).sum(dim=(2, 3))
    return torch.stack(
        [coverage_potential, crowding_potential,
         coverage_energy, crowd_al_energy + crowd_aa_energy],
        dim=2,
    )


def scalable_active_action_features_tensor(raw, self_agent_index,
                                            n_agents, n_landmarks):
    agents, landmarks = reconstruct_relative_geometry(
        raw, self_agent_index, n_agents, n_landmarks
    )
    batch = raw.shape[0]
    candidates = agents[:, None, :, :].expand(-1, N_ACTIONS, -1, -1).clone()
    controls = torch.as_tensor(ACTION_TO_CONTROL, dtype=raw.dtype, device=raw.device)
    no_op_displacement = raw[:, 0:2] * (1.0 - DAMPING) * DT
    displacement = no_op_displacement[:, None, :] + controls[None] * SENSITIVITY * DT * DT
    candidates[:, :, self_agent_index, :] = displacement
    quantities = _state_quantities(candidates, landmarks)
    features = quantities - quantities[:, 0:1, :]
    if not torch.isfinite(features).all():
        raise RuntimeError("Scalable Active features contain NaN or infinity")
    return features
