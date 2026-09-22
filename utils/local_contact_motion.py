"""Observable one-step motion for the fixed legacy three-agent simple_spread.

No world, hidden velocity, other-agent action or future-state input is accepted.
The environment assumptions are checked separately by the diagnostic harness.
This module is not connected to any actor, critic, replay or training runner.
"""

import numpy as np

from utils.active_gsp_v3_features import ACTION_TO_CONTROL_FLOAT32


DT = np.float32(0.1)
DAMPING = np.float32(0.25)
ACTION_ACCELERATION = np.float32(5.0)
AGENT_RADIUS = np.float32(0.15)
CONTACT_FORCE = np.float32(100.0)
CONTACT_MARGIN = np.float32(0.001)
MODES = ("legacy", "focal_contact", "known_contact_graph")


def _raw_batch(local_obs_batch):
    values = np.asarray(local_obs_batch, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 18:
        raise ValueError("Expected [batch,18] focal raw observations")
    if not np.isfinite(values).all():
        raise ValueError("Local observations must be finite")
    return values


def local_agent_positions(local_obs_batch):
    raw = _raw_batch(local_obs_batch)
    positions = np.zeros((len(raw), 3, 2), dtype=np.float32)
    positions[:, 1:] = raw[:, 10:14].reshape(-1, 2, 2)
    return positions


def observable_contact_displacements(local_obs_batch):
    """Contact-only displacement for all three observed equal-mass agents.

    Exact coincidence has no direction and is singular in the installed MPE.
    Zero direction is a finite convention here, not a simulator-equivalence
    claim at that singularity.
    """
    positions = local_agent_positions(local_obs_batch)
    differences = positions[:, :, None] - positions[:, None, :]
    distances = np.linalg.norm(differences, axis=-1)
    penetration = np.logaddexp(np.float32(0),
                              (2 * AGENT_RADIUS - distances) / CONTACT_MARGIN) * CONTACT_MARGIN
    direction = differences / np.maximum(distances[..., None], np.float32(1e-12))
    forces = CONTACT_FORCE * penetration[..., None] * direction
    # Diagonal directions are zero; all physical agent masses are one.
    return forces.sum(axis=2) * DT * DT


def predict_local_candidate_positions(local_obs_batch, mode="legacy"):
    """Return [batch,5,3,2] with focal node first and the other nodes in obs order.

    Focal velocity and chosen focal action are known. Non-focal velocity and
    action drift are not available; even known_contact_graph omits those terms.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown motion mode: {mode}")
    raw = _raw_batch(local_obs_batch)
    positions = local_agent_positions(raw)
    candidates = np.broadcast_to(positions[:, None], (len(raw), 5, 3, 2)).copy()
    candidates[:, :, 0] += (
        raw[:, None, :2] * (np.float32(1) - DAMPING) * DT
        + ACTION_TO_CONTROL_FLOAT32[None] * ACTION_ACCELERATION * DT * DT
    )
    if mode != "legacy":
        contact = observable_contact_displacements(raw)
        if mode == "focal_contact":
            candidates[:, :, 0] += contact[:, None, 0]
        else:
            candidates += contact[:, None]
    return candidates
