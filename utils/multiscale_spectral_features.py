"""Exact batched multiscale action-conditioned spectral descriptors.

The branch is decentralized and actor-only: it consumes one raw 18D local
observation, simulates only the focal agent's five candidate actions, and
returns no-op-relative action features.  No world object is accessed.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from utils.active_gsp_v3_features import (
    ACTION_TO_CONTROL,
    DEFAULT_DAMPING,
    DEFAULT_DT,
    DEFAULT_SENSITIVITY,
)
from utils.gsp_features import (
    AGENT_AGENT_SCALE,
    N_AGENTS,
    N_LANDMARKS,
    SIGMA_AGENT_AGENT,
    SIGMA_AGENT_LANDMARK,
)


RAW_OBS_DIM = 18
N_ACTIONS = 5
N_NODES = N_AGENTS + N_LANDMARKS
SPECTRAL_TIMES = (0.0, 0.5, 2.0)
MULTISCALE_FEATURE_DIM = 2 + 2 * len(SPECTRAL_TIMES)
DEFAULT_EPS = 1e-8


def _finite_coordinates(value: torch.Tensor) -> torch.Tensor:
    # This only affects nonphysical overflow inputs; normal MPE coordinates are
    # several orders of magnitude smaller.
    return torch.nan_to_num(value, nan=0.0, posinf=1e10, neginf=-1e10).clamp(-1e10, 1e10)


def normalized_laplacian_and_task_signals(
    agent_positions: torch.Tensor,
    landmark_positions: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the locked AA/AL graph and both task signals for arbitrary batches.

    Leading dimensions must match.  Returned tensors are ``L_sym``, coverage
    signal, crowding signal, coverage potential, and crowding potential.
    """
    if agent_positions.shape[-2:] != (N_AGENTS, 2):
        raise ValueError(f"Expected agent positions [...,{N_AGENTS},2], got {tuple(agent_positions.shape)}")
    if landmark_positions.shape[-2:] != (N_LANDMARKS, 2):
        raise ValueError(f"Expected landmark positions [...,{N_LANDMARKS},2], got {tuple(landmark_positions.shape)}")
    if agent_positions.shape[:-2] != landmark_positions.shape[:-2]:
        raise ValueError("Agent and landmark leading batch dimensions must match.")

    agents = _finite_coordinates(agent_positions)
    landmarks = _finite_coordinates(landmark_positions)
    aa_dist_sq = (agents.unsqueeze(-2) - agents.unsqueeze(-3)).square().sum(dim=-1)
    al_dist_sq = (agents.unsqueeze(-2) - landmarks.unsqueeze(-3)).square().sum(dim=-1)
    aa_weights = AGENT_AGENT_SCALE * torch.exp(
        -aa_dist_sq / (2.0 * SIGMA_AGENT_AGENT ** 2)
    )
    eye_agents = torch.eye(N_AGENTS, device=agents.device, dtype=agents.dtype)
    aa_weights = aa_weights * (1.0 - eye_agents)
    al_weights = torch.exp(-al_dist_sq / (2.0 * SIGMA_AGENT_LANDMARK ** 2))

    leading = agents.shape[:-2]
    adjacency = agents.new_zeros(*leading, N_NODES, N_NODES)
    adjacency[..., :N_AGENTS, :N_AGENTS] = aa_weights
    adjacency[..., :N_AGENTS, N_AGENTS:] = al_weights
    adjacency[..., N_AGENTS:, :N_AGENTS] = al_weights.transpose(-2, -1)
    degree = adjacency.sum(dim=-1)
    inv_sqrt = torch.where(degree > 0.0, degree.clamp_min(1e-30).rsqrt(), torch.zeros_like(degree))
    normalized_adjacency = adjacency * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)
    laplacian = torch.eye(N_NODES, device=agents.device, dtype=agents.dtype) - normalized_adjacency
    laplacian = 0.5 * (laplacian + laplacian.transpose(-2, -1))

    al_dist = torch.sqrt(al_dist_sq.clamp_min(0.0))
    coverage_landmarks = al_dist.min(dim=-2).values
    coverage_signal = torch.cat([agents.new_zeros(*leading, N_AGENTS), coverage_landmarks], dim=-1)

    large = torch.finfo(agents.dtype).max ** 0.25
    aa_dist = torch.sqrt(aa_dist_sq.clamp_min(0.0)) + eye_agents * large
    nearest_agent = aa_dist.min(dim=-1).values
    crowding_agents = AGENT_AGENT_SCALE * torch.exp(
        -nearest_agent.square() / (2.0 * SIGMA_AGENT_AGENT ** 2)
    )
    crowding_signal = torch.cat([crowding_agents, agents.new_zeros(*leading, N_LANDMARKS)], dim=-1)
    coverage_potential = coverage_landmarks.sum(dim=-1)
    crowding_potential = torch.triu(aa_weights, diagonal=1).sum(dim=(-2, -1))
    return laplacian, coverage_signal, crowding_signal, coverage_potential, crowding_potential


def multiscale_rayleigh_exact(
    laplacian: torch.Tensor,
    signals: torch.Tensor,
    times: Sequence[float] = SPECTRAL_TIMES,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Compute exact heat-filtered Rayleigh quotients with batched ``eigh``.

    ``laplacian`` has shape ``[...,N,N]`` and ``signals`` has shape
    ``[...,S,N]``.  The result has shape ``[...,S,len(times)]``.
    """
    if laplacian.shape[-2:] != (N_NODES, N_NODES):
        raise ValueError(f"Expected Laplacian [...,{N_NODES},{N_NODES}], got {tuple(laplacian.shape)}")
    if signals.shape[-1] != N_NODES or signals.shape[:-2] != laplacian.shape[:-2]:
        raise ValueError("Signal and Laplacian batch shapes are inconsistent.")
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)
    coefficients = torch.matmul(signals, eigenvectors)
    time_tensor = laplacian.new_tensor(tuple(times))
    heat = torch.exp(-eigenvalues.unsqueeze(-1) * time_tensor)
    filtered_coefficients = coefficients.unsqueeze(-1) * heat.unsqueeze(-3)
    squared = filtered_coefficients.square()
    numerator = (squared * eigenvalues.unsqueeze(-1).unsqueeze(-3)).sum(dim=-2)
    denominator = squared.sum(dim=-2) + float(eps)
    return numerator / denominator


def multiscale_rayleigh_matrix_exp_reference(
    laplacian: torch.Tensor,
    signals: torch.Tensor,
    times: Sequence[float] = SPECTRAL_TIMES,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Independent exact reference using ``torch.matrix_exp``."""
    outputs = []
    x = signals.unsqueeze(-1)
    for time in times:
        heat = torch.matrix_exp(-float(time) * laplacian)
        y = torch.matmul(heat.unsqueeze(-3), x).squeeze(-1)
        ly = torch.matmul(laplacian.unsqueeze(-3), y.unsqueeze(-1)).squeeze(-1)
        numerator = (y * ly).sum(dim=-1)
        denominator = y.square().sum(dim=-1) + float(eps)
        outputs.append(numerator / denominator)
    return torch.stack(outputs, dim=-1)


def candidate_graph_tensors(raw_obs: torch.Tensor, self_agent_index: int):
    if raw_obs.ndim != 2 or raw_obs.shape[1] != RAW_OBS_DIM:
        raise ValueError(f"Expected raw local observations [batch,{RAW_OBS_DIM}], got {tuple(raw_obs.shape)}")
    if not 0 <= int(self_agent_index) < N_AGENTS:
        raise ValueError("self_agent_index must be in [0,2].")
    raw = _finite_coordinates(raw_obs)
    batch_size = raw.shape[0]
    controls = raw.new_tensor(ACTION_TO_CONTROL)
    self_positions = (
        raw[:, 0:2, None].transpose(1, 2) * (1.0 - DEFAULT_DAMPING) * DEFAULT_DT
        + controls.unsqueeze(0) * DEFAULT_SENSITIVITY * DEFAULT_DT ** 2
    )
    agents = raw.new_zeros(batch_size, N_ACTIONS, N_AGENTS, 2)
    agents[:, :, int(self_agent_index), :] = self_positions
    others = raw[:, 10:14].reshape(batch_size, N_AGENTS - 1, 2)
    other_indices = [index for index in range(N_AGENTS) if index != int(self_agent_index)]
    for local_index, global_index in enumerate(other_indices):
        agents[:, :, global_index, :] = others[:, None, local_index, :]
    landmarks = raw[:, 4:10].reshape(batch_size, N_LANDMARKS, 2)
    landmarks = landmarks[:, None, :, :].expand(-1, N_ACTIONS, -1, -1)
    return normalized_laplacian_and_task_signals(agents, landmarks)


def multiscale_spectral_action_features(raw_obs: torch.Tensor, self_agent_index: int) -> torch.Tensor:
    """Return ``[batch,5,8]`` potential and multiscale spectral deltas."""
    laplacian, coverage, crowding, coverage_potential, crowding_potential = candidate_graph_tensors(
        raw_obs, self_agent_index
    )
    signals = torch.stack([coverage, crowding], dim=-2)
    rayleigh = multiscale_rayleigh_exact(laplacian, signals)
    quantities = torch.cat(
        [coverage_potential.unsqueeze(-1), crowding_potential.unsqueeze(-1), rayleigh.flatten(start_dim=-2)],
        dim=-1,
    )
    features = quantities - quantities[:, 0:1, :]
    if not torch.isfinite(features).all():
        raise RuntimeError("Multiscale spectral action features contain NaN or infinity.")
    return features


def multiscale_spectral_action_features_reference(raw_obs: torch.Tensor, self_agent_index: int) -> torch.Tensor:
    laplacian, coverage, crowding, coverage_potential, crowding_potential = candidate_graph_tensors(
        raw_obs, self_agent_index
    )
    signals = torch.stack([coverage, crowding], dim=-2)
    rayleigh = multiscale_rayleigh_matrix_exp_reference(laplacian, signals)
    quantities = torch.cat(
        [coverage_potential.unsqueeze(-1), crowding_potential.unsqueeze(-1), rayleigh.flatten(start_dim=-2)],
        dim=-1,
    )
    return quantities - quantities[:, 0:1, :]
