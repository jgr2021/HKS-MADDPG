"""Three-agent graph features with vector-valued landmark-distance signals.

The implementation is actor-local and side-effect free.  It consumes only a
raw 18D ``simple_spread`` observation, advances only the focal agent under the
existing one-step action model, and never accesses the world or another
policy.  Landmarks are signal channels, not graph nodes.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import torch

from utils.active_gsp_v3_features import (
    ACTION_TO_CONTROL,
    DEFAULT_DAMPING,
    DEFAULT_DT,
    DEFAULT_SENSITIVITY,
)
from utils.gsp_features import N_AGENTS, N_LANDMARKS, RAW_OBS_DIM, SIGMA_AGENT_AGENT


N_ACTIONS = 5
VECH_DIM = N_LANDMARKS * (N_LANDMARKS + 1) // 2
VECTOR_SIGNAL_GEOM_DIM = N_LANDMARKS + VECH_DIM
VECTOR_SIGNAL_GSP_DIM = N_LANDMARKS + 2 * VECH_DIM
UPPER_TRIANGLE = np.triu_indices(N_LANDMARKS)


def _validate_obs(local_obs_batch: Sequence[Sequence[float]]) -> np.ndarray:
    obs = np.asarray(local_obs_batch, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs[None, :]
    if obs.ndim not in (2, 3) or obs.shape[-1] != RAW_OBS_DIM:
        raise ValueError(f"Expected [..., {RAW_OBS_DIM}] raw observations, got {obs.shape}.")
    if not np.isfinite(obs).all():
        raise ValueError("Raw observations contain NaN or infinity.")
    return obs


def candidate_geometry_from_local_obs_batch(
    local_obs_batch: Sequence[Sequence[float]],
    action_deltas: Optional[Sequence[Sequence[float]]] = None,
    config: Optional[Mapping[str, float]] = None,
):
    """Return candidate agent positions and fixed landmark positions.

    Shapes are ``[..., actions, 3, 2]`` and ``[..., 3, 2]``.  Agent row zero
    is the focal actor; rows one and two retain their local observation order.
    Only row zero varies along the candidate-action axis.
    """
    obs = _validate_obs(local_obs_batch)
    controls = np.asarray(ACTION_TO_CONTROL if action_deltas is None else action_deltas,
                          dtype=np.float64)
    if controls.ndim != 2 or controls.shape[1] != 2:
        raise ValueError(f"Expected action controls [actions, 2], got {controls.shape}.")
    cfg = {} if config is None else config
    dt = float(cfg.get("dt", DEFAULT_DT))
    damping = float(cfg.get("damping", DEFAULT_DAMPING))
    sensitivity = float(cfg.get("sensitivity", DEFAULT_SENSITIVITY))
    velocity = obs[..., 0:2]
    focal = (velocity[..., None, :] * (1.0 - damping) * dt
             + controls * sensitivity * dt * dt)
    leading = obs.shape[:-1]
    agents = np.zeros(leading + (controls.shape[0], N_AGENTS, 2), dtype=np.float64)
    agents[..., 0, :] = focal
    others = obs[..., 10:14].reshape(leading + (N_AGENTS - 1, 2))
    agents[..., 1:, :] = others[..., None, :, :]
    landmarks = obs[..., 4:10].reshape(leading + (N_LANDMARKS, 2))
    return agents, landmarks


def _descriptors_numpy(agents: np.ndarray, landmarks: np.ndarray):
    distances = np.linalg.norm(
        agents[..., :, :, None, :] - landmarks[..., None, None, :, :], axis=-1
    )
    mu = distances.mean(axis=-2)
    centered = distances - mu[..., None, :]
    m0 = np.einsum("...ajm,...ajn->...amn", centered, centered)

    differences = agents[..., :, :, None, :] - agents[..., :, None, :, :]
    distance_sq = np.sum(differences * differences, axis=-1)
    weights = np.exp(-distance_sq / (2.0 * SIGMA_AGENT_AGENT ** 2))
    diagonal = np.arange(N_AGENTS)
    weights[..., diagonal, diagonal] = 0.0
    degree = weights.sum(axis=-1)
    inv_sqrt = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalized = weights * inv_sqrt[..., :, None] * inv_sqrt[..., None, :]
    laplacian = -normalized
    laplacian[..., diagonal, diagonal] += 1.0
    m1 = np.einsum("...ajm,...ajk,...akn->...amn", centered, laplacian, centered)
    return mu, m0, m1


def compute_vector_signal_features_from_local_obs_batch(
    local_obs_batch: Sequence[Sequence[float]],
    variant: str = "vector_signal_gsp",
    action_deltas: Optional[Sequence[Sequence[float]]] = None,
    config: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    """Compute action deltas with shape ``[..., actions, 9 or 15]``."""
    agents, landmarks = candidate_geometry_from_local_obs_batch(
        local_obs_batch, action_deltas=action_deltas, config=config
    )
    mu, m0, m1 = _descriptors_numpy(agents, landmarks)
    pieces = [mu - mu[..., 0:1, :],
              (m0 - m0[..., 0:1, :, :])[..., UPPER_TRIANGLE[0], UPPER_TRIANGLE[1]]]
    if variant == "vector_signal_gsp":
        pieces.append((m1 - m1[..., 0:1, :, :])
                      [..., UPPER_TRIANGLE[0], UPPER_TRIANGLE[1]])
    elif variant != "vector_signal_geom":
        raise ValueError(f"Unknown vector-signal feature variant: {variant}.")
    features = np.concatenate(pieces, axis=-1)
    if not np.isfinite(features).all():
        raise RuntimeError("Vector-signal features contain NaN or infinity.")
    return features.astype(np.float32)


def vector_signal_features_tensor(raw_obs: torch.Tensor, variant="vector_signal_gsp"):
    """Differentiable actor fast path returning ``[batch, 5, 9 or 15]``."""
    if raw_obs.ndim != 2 or raw_obs.shape[1] != RAW_OBS_DIM:
        raise ValueError(f"Expected [batch, {RAW_OBS_DIM}], got {tuple(raw_obs.shape)}.")
    controls = raw_obs.new_tensor(ACTION_TO_CONTROL)
    focal = (raw_obs[:, None, 0:2] * (1.0 - DEFAULT_DAMPING) * DEFAULT_DT
             + controls[None] * DEFAULT_SENSITIVITY * DEFAULT_DT ** 2)
    batch = raw_obs.shape[0]
    agents = raw_obs.new_zeros(batch, N_ACTIONS, N_AGENTS, 2)
    agents[:, :, 0, :] = focal
    agents[:, :, 1:, :] = raw_obs[:, None, 10:14].reshape(batch, 1, 2, 2)
    landmarks = raw_obs[:, 4:10].reshape(batch, N_LANDMARKS, 2)
    distances = torch.linalg.vector_norm(
        agents[:, :, :, None, :] - landmarks[:, None, None, :, :], dim=-1
    )
    mu = distances.mean(dim=2)
    centered = distances - mu[:, :, None, :]
    m0 = centered.transpose(2, 3) @ centered
    pair_sq = (agents[:, :, :, None, :] - agents[:, :, None, :, :]).square().sum(-1)
    weights = torch.exp(-pair_sq / (2.0 * SIGMA_AGENT_AGENT ** 2))
    eye = torch.eye(N_AGENTS, device=raw_obs.device, dtype=raw_obs.dtype)
    weights = weights * (1.0 - eye)
    inv_sqrt = weights.sum(dim=-1).clamp_min(1e-12).rsqrt()
    laplacian = eye - weights * inv_sqrt[:, :, :, None] * inv_sqrt[:, :, None, :]
    m1 = centered.transpose(2, 3) @ laplacian @ centered
    upper = torch.triu_indices(N_LANDMARKS, N_LANDMARKS, device=raw_obs.device)
    parts = [mu - mu[:, 0:1],
             (m0 - m0[:, 0:1])[:, :, upper[0], upper[1]]]
    if variant == "vector_signal_gsp":
        parts.append((m1 - m1[:, 0:1])[:, :, upper[0], upper[1]])
    elif variant != "vector_signal_geom":
        raise ValueError(f"Unknown vector-signal feature variant: {variant}.")
    return torch.cat(parts, dim=-1)
