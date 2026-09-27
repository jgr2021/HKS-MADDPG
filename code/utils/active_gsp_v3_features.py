"""Action-conditioned graph features for Active GSP v3.

This module is intentionally side-effect free. It reconstructs the local task
geometry from one agent's 18D simple_spread observation and computes myopic
candidate-action impact features. It does not read the MPE world object, future
state, other agents' actions, centralized critic inputs, or hidden states.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from utils.gsp_features import (
    AGENT_AGENT_SCALE,
    N_AGENTS,
    N_LANDMARKS,
    SIGMA_AGENT_AGENT,
    SIGMA_AGENT_LANDMARK,
    reconstruct_geometry_from_raw_observation,
)


N_ACTIONS = 5
ACTIVE_GSP_FEATURES_PER_ACTION = 4
ACTIVE_GSP_FEATURE_DIM = N_ACTIONS * ACTIVE_GSP_FEATURES_PER_ACTION

ACTION_TO_CONTROL = np.asarray(
    [
        [0.0, 0.0],   # no-op
        [1.0, 0.0],   # +x, MPE one-hot index 1
        [-1.0, 0.0],  # -x, MPE one-hot index 2
        [0.0, 1.0],   # +y, MPE one-hot index 3
        [0.0, -1.0],  # -y, MPE one-hot index 4
    ],
    dtype=np.float64,
)
ACTION_TO_CONTROL_FLOAT32 = ACTION_TO_CONTROL.astype(np.float32)

DEFAULT_DT = 0.1
DEFAULT_DAMPING = 0.25
DEFAULT_SENSITIVITY = 5.0
DEFAULT_AGENT_MASS = 1.0
ONE_FLOAT32 = np.float32(1.0)
AA_KERNEL_DENOM_FLOAT32 = np.float32(2.0 * SIGMA_AGENT_AGENT ** 2)
AL_KERNEL_DENOM_FLOAT32 = np.float32(2.0 * SIGMA_AGENT_LANDMARK ** 2)


def _gaussian(distance: np.ndarray, sigma: float, scale: float = 1.0) -> np.ndarray:
    return scale * np.exp(-(distance ** 2) / (2.0 * sigma ** 2))


def _normalized_laplacian_from_positions(
    agent_positions: np.ndarray, landmark_positions: np.ndarray
) -> np.ndarray:
    n_nodes = N_AGENTS + N_LANDMARKS
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    for i in range(N_AGENTS):
        for j in range(i + 1, N_AGENTS):
            distance = np.linalg.norm(agent_positions[i] - agent_positions[j])
            weight = float(_gaussian(distance, SIGMA_AGENT_AGENT, scale=AGENT_AGENT_SCALE))
            adjacency[i, j] = weight
            adjacency[j, i] = weight

    for i in range(N_AGENTS):
        for landmark_index in range(N_LANDMARKS):
            distance = np.linalg.norm(agent_positions[i] - landmark_positions[landmark_index])
            weight = float(_gaussian(distance, SIGMA_AGENT_LANDMARK))
            node_index = N_AGENTS + landmark_index
            adjacency[i, node_index] = weight
            adjacency[node_index, i] = weight

    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalized_adjacency = (
        inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    )
    laplacian = np.eye(n_nodes, dtype=np.float64) - normalized_adjacency
    laplacian = 0.5 * (laplacian + laplacian.T)
    return laplacian


def _coverage_signal(agent_positions: np.ndarray, landmark_positions: np.ndarray) -> np.ndarray:
    """Six-node signal: agent entries 0, landmark entries nearest-agent distance."""
    distances = np.linalg.norm(
        agent_positions[:, None, :] - landmark_positions[None, :, :],
        axis=2,
    )
    landmark_residual = distances.min(axis=0)
    return np.concatenate(
        [np.zeros(N_AGENTS, dtype=np.float64), landmark_residual],
        axis=0,
    )


def _crowding_signal(agent_positions: np.ndarray) -> np.ndarray:
    """Six-node signal: agent entries nearest-agent proximity, landmark entries 0."""
    distances = np.linalg.norm(
        agent_positions[:, None, :] - agent_positions[None, :, :],
        axis=2,
    )
    distances = distances + np.eye(N_AGENTS, dtype=np.float64) * 1e6
    nearest = distances.min(axis=1)
    crowding = _gaussian(nearest, SIGMA_AGENT_AGENT, scale=AGENT_AGENT_SCALE)
    return np.concatenate(
        [crowding, np.zeros(N_LANDMARKS, dtype=np.float64)],
        axis=0,
    )


def _dirichlet_energy(laplacian: np.ndarray, signal: np.ndarray) -> float:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    return float(signal @ laplacian @ signal)


def _coverage_potential(agent_positions: np.ndarray, landmark_positions: np.ndarray) -> float:
    distances = np.linalg.norm(
        agent_positions[:, None, :] - landmark_positions[None, :, :],
        axis=2,
    )
    return float(distances.min(axis=0).sum())


def _crowding_potential(agent_positions: np.ndarray) -> float:
    distances = np.linalg.norm(
        agent_positions[:, None, :] - agent_positions[None, :, :],
        axis=2,
    )
    mask = ~np.eye(N_AGENTS, dtype=bool)
    proximities = _gaussian(distances[mask], SIGMA_AGENT_AGENT, scale=AGENT_AGENT_SCALE)
    return float(proximities.sum() / 2.0)


def _edge_weights_from_positions(
    agent_positions: np.ndarray, landmark_positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    aa_distances = np.linalg.norm(
        agent_positions[:, None, :] - agent_positions[None, :, :],
        axis=2,
    )
    al_distances = np.linalg.norm(
        agent_positions[:, None, :] - landmark_positions[None, :, :],
        axis=2,
    )
    aa_weights = _gaussian(aa_distances, SIGMA_AGENT_AGENT, scale=AGENT_AGENT_SCALE)
    np.fill_diagonal(aa_weights, 0.0)
    al_weights = _gaussian(al_distances, SIGMA_AGENT_LANDMARK)
    return aa_weights, al_weights


def _direct_dirichlet_energies(
    agent_positions: np.ndarray, landmark_positions: np.ndarray
) -> Tuple[float, float]:
    aa_weights, al_weights = _edge_weights_from_positions(agent_positions, landmark_positions)
    cover_signal = _coverage_signal(agent_positions, landmark_positions)
    crowd_signal = _crowding_signal(agent_positions)
    cover_landmarks = cover_signal[N_AGENTS:]
    crowd_agents = crowd_signal[:N_AGENTS]

    cover_energy = float(np.sum(al_weights * (cover_landmarks[None, :] ** 2)))
    crowd_al_energy = float(np.sum(al_weights * (crowd_agents[:, None] ** 2)))
    upper = np.triu_indices(N_AGENTS, k=1)
    crowd_aa_energy = float(
        np.sum(
            aa_weights[upper]
            * ((crowd_agents[upper[0]] - crowd_agents[upper[1]]) ** 2)
        )
    )
    return cover_energy, crowd_al_energy + crowd_aa_energy


def _state_quantities(
    agent_positions: np.ndarray, landmark_positions: np.ndarray
) -> Tuple[float, float, float, float]:
    cover_energy, crowd_energy = _direct_dirichlet_energies(
        agent_positions,
        landmark_positions,
    )
    return (
        _coverage_potential(agent_positions, landmark_positions),
        _crowding_potential(agent_positions),
        cover_energy,
        crowd_energy,
    )


def predict_self_position_after_action(
    raw_observation: Sequence[float],
    action_index: int,
    dt: float = DEFAULT_DT,
    damping: float = DEFAULT_DAMPING,
    sensitivity: float = DEFAULT_SENSITIVITY,
    mass: float = DEFAULT_AGENT_MASS,
) -> np.ndarray:
    """Approximate the controlled agent's next position for a discrete action."""
    raw = np.asarray(raw_observation, dtype=np.float64).reshape(-1)
    if not 0 <= int(action_index) < N_ACTIONS:
        raise ValueError(f"action_index must be in [0, {N_ACTIONS - 1}], got {action_index}.")
    velocity = raw[0:2]
    position = raw[2:4]
    control = ACTION_TO_CONTROL[int(action_index)] * sensitivity
    next_velocity = velocity * (1.0 - damping) + (control / mass) * dt
    return position + next_velocity * dt


def active_gsp_action_features(
    raw_observation: Sequence[float],
    self_agent_index: int,
    dt: float = DEFAULT_DT,
    damping: float = DEFAULT_DAMPING,
    sensitivity: float = DEFAULT_SENSITIVITY,
) -> np.ndarray:
    """Reference return of shape (5, 4) for one decentralized actor.

    Per action:
        [delta coverage potential,
         delta crowding potential,
         delta coverage Dirichlet energy,
         delta crowding Dirichlet energy]

    Negative coverage potential deltas are good in simple_spread because the
    reward penalizes landmark-to-nearest-agent distance. Negative crowding
    potential deltas indicate reduced agent-agent redundancy/proximity.
    Deltas are measured relative to the no-op candidate, so action 0 is zero.
    """
    agent_positions, landmark_positions = reconstruct_geometry_from_raw_observation(
        raw_observation, self_agent_index
    )
    quantities = np.zeros((N_ACTIONS, ACTIVE_GSP_FEATURES_PER_ACTION), dtype=np.float64)

    for action_index in range(N_ACTIONS):
        candidate_agents = agent_positions.copy()
        candidate_agents[int(self_agent_index)] = predict_self_position_after_action(
            raw_observation,
            action_index,
            dt=dt,
            damping=damping,
            sensitivity=sensitivity,
        )
        candidate = np.asarray(
            _state_quantities(candidate_agents, landmark_positions),
            dtype=np.float64,
        )
        quantities[action_index] = candidate

    features = quantities - quantities[0:1]

    if not np.all(np.isfinite(features)):
        raise RuntimeError("Active GSP v3 features contain NaN or infinity.")
    return features.astype(np.float32)


def active_gsp_flat_features(
    raw_observation: Sequence[float], self_agent_index: int
) -> np.ndarray:
    features = active_gsp_action_features(raw_observation, self_agent_index)
    return features.reshape(-1).astype(np.float32)


def action_raw_potential_features(
    raw_observation: Sequence[float], self_agent_index: int
) -> np.ndarray:
    """A non-spectral action-aware geometry ablation with the same 5x4 shape."""
    agent_positions, landmark_positions = reconstruct_geometry_from_raw_observation(
        raw_observation, self_agent_index
    )
    potentials = np.zeros((N_ACTIONS, 2), dtype=np.float64)

    for action_index in range(N_ACTIONS):
        candidate_agents = agent_positions.copy()
        candidate_agents[int(self_agent_index)] = predict_self_position_after_action(
            raw_observation,
            action_index,
        )
        potentials[action_index] = [
            _coverage_potential(candidate_agents, landmark_positions),
            _crowding_potential(candidate_agents),
        ]

    deltas = potentials - potentials[0:1]
    features = np.zeros((N_ACTIONS, ACTIVE_GSP_FEATURES_PER_ACTION), dtype=np.float64)
    features[:, 0:2] = deltas

    return features.astype(np.float32)


def action_score_control_features() -> np.ndarray:
    """Zero action-feature control for the action-scoring actor architecture."""
    return np.zeros((N_ACTIONS, ACTIVE_GSP_FEATURES_PER_ACTION), dtype=np.float32)


def _config_value(config: Optional[Mapping[str, object]], key: str, default):
    if config is None:
        return default
    return config.get(key, default)


def _resolve_self_agent_indices(local_obs_batch: np.ndarray, config) -> np.ndarray:
    batch_size = local_obs_batch.shape[0]
    indices = _config_value(config, "self_agent_indices", None)
    if indices is None:
        scalar = _config_value(config, "self_agent_index", None)
        if scalar is not None:
            indices = np.full(batch_size, int(scalar), dtype=np.int64)
        elif batch_size % N_AGENTS == 0:
            indices = np.tile(np.arange(N_AGENTS, dtype=np.int64), batch_size // N_AGENTS)
        else:
            indices = np.zeros(batch_size, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if indices.size != batch_size:
        raise ValueError(
            f"Expected {batch_size} self-agent indices, got {indices.size}."
        )
    if np.any(indices < 0) or np.any(indices >= N_AGENTS):
        raise ValueError("self-agent indices must be in [0, 2].")
    return indices


def _reconstruct_relative_agent_positions_batch(
    local_obs_batch: np.ndarray, self_agent_indices: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    batch_size = local_obs_batch.shape[0]
    agent_positions = np.zeros((batch_size, N_AGENTS, 2), dtype=np.float32)
    landmark_positions = local_obs_batch[:, 4:10].reshape(batch_size, N_LANDMARKS, 2)
    other_positions = local_obs_batch[:, 10:14].reshape(batch_size, N_AGENTS - 1, 2)

    for self_index, other_indices in (
        (0, (1, 2)),
        (1, (0, 2)),
        (2, (0, 1)),
    ):
        mask = self_agent_indices == self_index
        if not np.any(mask):
            continue
        agent_positions[mask, other_indices[0], :] = other_positions[mask, 0, :]
        agent_positions[mask, other_indices[1], :] = other_positions[mask, 1, :]

    return agent_positions, landmark_positions.astype(np.float32, copy=False)


def compute_active_gsp_v3_from_local_obs_batch(
    local_obs_batch: Sequence[Sequence[float]],
    action_deltas: Optional[Sequence[Sequence[float]]] = None,
    config: Optional[Mapping[str, object]] = None,
) -> np.ndarray:
    """Vectorized local-observation-only Active GSP v3 features.

    Parameters
    ----------
    local_obs_batch:
        Array-like with shape ``[M, 18]``.
    action_deltas:
        Candidate action controls with shape ``[num_actions, 2]``. Defaults to
        the MPE five-way discrete action controls.
    config:
        Optional mapping. Supported keys are ``self_agent_indices`` or
        ``self_agent_index``, ``dt``, ``damping``, and ``sensitivity``.

    Returns
    -------
    np.ndarray, shape ``[M, num_actions, 4]``
        Per action:
        ``[Delta coverage potential, Delta crowding potential,
        Delta coverage energy, Delta crowding energy]`` relative to no-op.
    """
    local_obs = np.asarray(local_obs_batch, dtype=np.float32)
    if local_obs.ndim == 1:
        local_obs = local_obs.reshape(1, -1)
    if local_obs.ndim != 2 or local_obs.shape[1] < 14:
        raise ValueError(f"Expected local observations with shape [M, >=14], got {local_obs.shape}.")

    controls = ACTION_TO_CONTROL_FLOAT32 if action_deltas is None else np.asarray(action_deltas, dtype=np.float32)
    if controls.ndim != 2 or controls.shape[1] != 2:
        raise ValueError(f"Expected action_deltas shape [num_actions, 2], got {controls.shape}.")

    dt = np.float32(_config_value(config, "dt", DEFAULT_DT))
    damping = np.float32(_config_value(config, "damping", DEFAULT_DAMPING))
    sensitivity = np.float32(_config_value(config, "sensitivity", DEFAULT_SENSITIVITY))
    batch_size = local_obs.shape[0]
    num_actions = controls.shape[0]
    landmark_positions = local_obs[:, 4:10].reshape(batch_size, N_LANDMARKS, 2)
    other_positions = local_obs[:, 10:14].reshape(batch_size, N_AGENTS - 1, 2)

    self_velocity = local_obs[:, 0:2]
    self_displacements = (
        self_velocity[:, None, :] * (ONE_FLOAT32 - damping) * dt
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
    self_other_dist_sq = self_other_dx * self_other_dx + self_other_dy * self_other_dy
    self_other_weights = AGENT_AGENT_SCALE * np.exp(
        -self_other_dist_sq / AA_KERNEL_DENOM_FLOAT32
    ).astype(np.float32, copy=False)

    other_other_dx = other_x[:, 0] - other_x[:, 1]
    other_other_dy = other_y[:, 0] - other_y[:, 1]
    other_other_dist_sq = other_other_dx * other_other_dx + other_other_dy * other_other_dy
    other_other_weight = (
        AGENT_AGENT_SCALE
        * np.exp(-other_other_dist_sq / AA_KERNEL_DENOM_FLOAT32)
    ).astype(np.float32, copy=False)

    self_landmark_dx = self_x[:, :, None] - landmark_x[:, None, :]
    self_landmark_dy = self_y[:, :, None] - landmark_y[:, None, :]
    self_landmark_dist_sq = (
        self_landmark_dx * self_landmark_dx
        + self_landmark_dy * self_landmark_dy
    )
    self_landmark_dist = np.sqrt(self_landmark_dist_sq).astype(np.float32, copy=False)
    self_landmark_weights = np.exp(
        -self_landmark_dist_sq / AL_KERNEL_DENOM_FLOAT32
    ).astype(np.float32, copy=False)

    other_landmark_dx = other_x[:, :, None] - landmark_x[:, None, :]
    other_landmark_dy = other_y[:, :, None] - landmark_y[:, None, :]
    other_landmark_dist_sq = (
        other_landmark_dx * other_landmark_dx
        + other_landmark_dy * other_landmark_dy
    )
    other_landmark_dist = np.sqrt(other_landmark_dist_sq).astype(np.float32, copy=False)
    other_landmark_weights = np.exp(
        -other_landmark_dist_sq / AL_KERNEL_DENOM_FLOAT32
    ).astype(np.float32, copy=False)

    nearest_other_landmark_dist = np.minimum(
        other_landmark_dist[:, 0:1, :],
        other_landmark_dist[:, 1:2, :],
    )
    coverage_residual = np.minimum(
        self_landmark_dist,
        nearest_other_landmark_dist,
    )
    coverage_potential = np.sum(coverage_residual, axis=2)

    crowd_self = np.maximum(self_other_weights[:, :, 0], self_other_weights[:, :, 1])
    crowd_other1 = np.maximum(self_other_weights[:, :, 0], other_other_weight[:, None])
    crowd_other2 = np.maximum(self_other_weights[:, :, 1], other_other_weight[:, None])

    crowding_potential = (
        np.sum(self_other_weights, axis=2) + other_other_weight[:, None]
    )

    coverage_residual_sq = coverage_residual * coverage_residual
    other_landmark_weight_sum = np.sum(other_landmark_weights, axis=1)
    coverage_energy = np.sum(
        self_landmark_weights * coverage_residual_sq,
        axis=2,
    ) + np.sum(
        other_landmark_weight_sum[:, None, :] * coverage_residual_sq,
        axis=2,
    )
    self_landmark_weight_sum = np.sum(self_landmark_weights, axis=2)
    other1_landmark_weight_sum = np.sum(other_landmark_weights[:, 0, :], axis=1)[:, None]
    other2_landmark_weight_sum = np.sum(other_landmark_weights[:, 1, :], axis=1)[:, None]
    crowding_al_energy = (
        self_landmark_weight_sum * (crowd_self ** 2)
        + other1_landmark_weight_sum * (crowd_other1 ** 2)
        + other2_landmark_weight_sum * (crowd_other2 ** 2)
    )
    crowding_aa_energy = (
        self_other_weights[:, :, 0] * ((crowd_self - crowd_other1) ** 2)
        + self_other_weights[:, :, 1] * ((crowd_self - crowd_other2) ** 2)
        + other_other_weight[:, None] * ((crowd_other1 - crowd_other2) ** 2)
    )
    crowding_energy = crowding_al_energy + crowding_aa_energy

    features = np.empty(
        (batch_size, num_actions, ACTIVE_GSP_FEATURES_PER_ACTION),
        dtype=np.float32,
    )
    features[:, :, 0] = coverage_potential - coverage_potential[:, 0:1]
    features[:, :, 1] = crowding_potential - crowding_potential[:, 0:1]
    features[:, :, 2] = coverage_energy - coverage_energy[:, 0:1]
    features[:, :, 3] = crowding_energy - crowding_energy[:, 0:1]
    if not np.all(np.isfinite(features)):
        raise RuntimeError("Fast Active GSP v3 features contain NaN or infinity.")
    return features.astype(np.float32, copy=False)
