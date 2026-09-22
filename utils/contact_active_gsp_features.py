"""Batch float32 graph features with observable MPE contact drift.

This isolated feature kernel is not registered in an actor or training runner.
Non-focal velocity/action drift remains unknown and is not supplied as input.
"""

import torch

from utils.gsp_features import AGENT_AGENT_SCALE, SIGMA_AGENT_AGENT, SIGMA_AGENT_LANDMARK


def compute_contact_active_features(local_obs_batch, action_controls):
    """Return [M,5,4] deltas using one local raw observation per row.

    action_controls is the original fixed [5,2] unit-control tensor, already
    on the observation device. All model inputs/outputs stay on that device.
    Physical constants are for the validated legacy equal-mass simple_spread.
    """
    raw = local_obs_batch
    if raw.ndim != 2 or raw.shape[1] != 18 or raw.dtype != torch.float32:
        raise ValueError("Expected float32 [M,18] local observations")
    if action_controls.shape != (5, 2) or action_controls.dtype != raw.dtype or action_controls.device != raw.device:
        raise ValueError("Expected float32 [5,2] original action controls on the same device")
    positions = torch.cat([torch.zeros_like(raw[:, :2]).unsqueeze(1), raw[:, 10:14].reshape(-1, 2, 2)], dim=1)
    difference = positions[:, :, None] - positions[:, None]
    distance = difference.square().sum(-1).clamp_min(1e-24).sqrt()
    penetration = torch.logaddexp(torch.zeros_like(distance), (0.30 - distance) / 0.001) * 0.001
    # Contact force 100 and dt^2=.01 cancel for these unit-mass agents.
    contact_drift = (penetration[..., None] * difference / distance[..., None]).sum(2)
    candidates = (positions + contact_drift)[:, None].expand(-1, 5, -1, -1).clone()
    candidates[:, :, 0] += raw[:, None, :2] * 0.075 + action_controls[None] * 0.05

    landmarks = raw[:, 4:10].reshape(-1, 3, 2)
    aa_squared = (candidates[:, :, :, None] - candidates[:, :, None]).square().sum(-1)
    al_squared = (candidates[:, :, :, None] - landmarks[:, None, None]).square().sum(-1)
    identity = torch.eye(3, dtype=raw.dtype, device=raw.device)
    aa_weights = AGENT_AGENT_SCALE * torch.exp(-aa_squared / (2 * SIGMA_AGENT_AGENT ** 2)) * (1 - identity)
    al_weights = torch.exp(-al_squared / (2 * SIGMA_AGENT_LANDMARK ** 2))
    coverage_squared = al_squared.amin(dim=2)
    crowding = AGENT_AGENT_SCALE * torch.exp(-(aa_squared + identity * 1e12).amin(-1) / (2 * SIGMA_AGENT_AGENT ** 2))
    coverage_potential = coverage_squared.clamp_min(1e-24).sqrt().sum(-1)
    crowding_potential = aa_weights.sum((-2, -1)) * 0.5
    coverage_energy = (al_weights * coverage_squared[:, :, None]).sum((-2, -1))
    crowding_energy = (
        (al_weights * crowding.square()[:, :, :, None]).sum((-2, -1))
        + (aa_weights * (crowding[:, :, :, None] - crowding[:, :, None]).square()).sum((-2, -1)) * 0.5
    )
    quantities = torch.stack([coverage_potential, crowding_potential, coverage_energy, crowding_energy], dim=-1)
    return quantities - quantities[:, :1]
