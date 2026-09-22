"""Graph-spectral feature construction for GSP-MADDPG v1.

This module is deliberately independent of the MPE ``world`` object.
Every descriptor is reconstructed only from the original 18-dimensional
``simple_spread`` observation available to the corresponding actor.

Original MPE observation layout for agent i (3 agents, 3 landmarks):
    [ own_velocity(2), own_position(2),
      landmark_relative_positions(3 x 2),
      other_agent_relative_positions(2 x 2),
      communication_states(2 x 2) ]

The final four communication entries are zero in the current no-communication
scenario and are not used for graph construction.

Fixed graph node order:
    [A0, A1, A2, L0, L1, L2]

GSP-v1 descriptor for actor i:
    z_i = [lambda_2, ..., lambda_6, HKS_i(0.5), HKS_i(1.0), HKS_i(2.0)]

Hence the environment wrapper changes observations only from 18D to 26D.
It does not modify reward, actions, dynamics, or the MADDPG update rule.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Fixed task and graph definition. These values are frozen from the offline
# diagnostic that distinguished duplicate pursuit from one-to-one coverage.
# They are graph-encoding constants, NOT environment or RL hyperparameters.
# ---------------------------------------------------------------------------
N_AGENTS = 3
N_LANDMARKS = 3
RAW_OBS_DIM = 18
SPECTRAL_DIM = 5
HKS_TIMES: Tuple[float, float, float] = (0.5, 1.0, 2.0)
HKS_DIM = len(HKS_TIMES)
GSP_DESCRIPTOR_DIM = SPECTRAL_DIM + HKS_DIM
AUGMENTED_OBS_DIM = RAW_OBS_DIM + GSP_DESCRIPTOR_DIM
EDGE_FEATURE_DIM = 9 + 3
SPECTRAL_ROLE_DIM = SPECTRAL_DIM + SPECTRAL_DIM + HKS_DIM + N_LANDMARKS
ALL_NODE_ROLE_DIM = SPECTRAL_DIM + (N_AGENTS + N_LANDMARKS) * SPECTRAL_DIM + HKS_DIM
TOPOLOGY_HKS_DIM = HKS_DIM

SIGMA_AGENT_LANDMARK = 0.60
SIGMA_AGENT_AGENT = 0.80
AGENT_AGENT_SCALE = 0.25
TOPOLOGY_3NODE_AA_HKS = "topology_3node_aa_hks"
TOPOLOGY_6NODE_AL_HKS = "topology_6node_al_hks"
TOPOLOGY_6NODE_AAL_HKS = "topology_6node_aal_hks"
TOPOLOGY_HKS_MODES = {
    TOPOLOGY_3NODE_AA_HKS,
    TOPOLOGY_6NODE_AL_HKS,
    TOPOLOGY_6NODE_AAL_HKS,
}


def _as_raw_observation(raw_observation: Sequence[float]) -> np.ndarray:
    """Validate and return a flat float64 raw MPE observation."""
    raw = np.asarray(raw_observation, dtype=np.float64).reshape(-1)
    if raw.size != RAW_OBS_DIM:
        raise ValueError(
            "GSP-v1 expects the original 18D MPE simple_spread observation; "
            f"received shape {raw.shape} with {raw.size} values."
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("Raw observation contains NaN or infinity.")
    return raw


def reconstruct_geometry_from_raw_observation(
    raw_observation: Sequence[float], self_agent_index: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct fixed-order agent and landmark positions from one actor's raw obs.

    No ``world`` state is accessed. The reconstruction is possible because the
    original observation contains the actor's absolute position and all other
    positions relative to that actor.

    Returns
    -------
    agent_positions : ndarray, shape (3, 2)
        Canonical order [A0, A1, A2].
    landmark_positions : ndarray, shape (3, 2)
        Canonical order [L0, L1, L2].
    """
    if not 0 <= int(self_agent_index) < N_AGENTS:
        raise ValueError(
            f"self_agent_index must be in [0, {N_AGENTS - 1}], got {self_agent_index}."
        )

    raw = _as_raw_observation(raw_observation)
    self_position = raw[2:4]
    landmark_relative_positions = raw[4:10].reshape(N_LANDMARKS, 2)
    other_agent_relative_positions = raw[10:14].reshape(N_AGENTS - 1, 2)

    agent_positions = np.zeros((N_AGENTS, 2), dtype=np.float64)
    agent_positions[int(self_agent_index)] = self_position

    other_agent_indices = [idx for idx in range(N_AGENTS) if idx != int(self_agent_index)]
    for local_index, global_index in enumerate(other_agent_indices):
        agent_positions[global_index] = (
            self_position + other_agent_relative_positions[local_index]
        )

    landmark_positions = self_position[None, :] + landmark_relative_positions
    return agent_positions, landmark_positions


def _gaussian_affinity(distance: float, sigma: float) -> float:
    return float(np.exp(-(float(distance) ** 2) / (2.0 * float(sigma) ** 2)))


def build_task_graph_from_positions(
    agent_positions: np.ndarray, landmark_positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build A, normalized L, eigenvalues, and eigenvectors for the 6-node task graph."""
    agent_positions = np.asarray(agent_positions, dtype=np.float64)
    landmark_positions = np.asarray(landmark_positions, dtype=np.float64)
    if agent_positions.shape != (N_AGENTS, 2):
        raise ValueError(f"Expected agent positions {(N_AGENTS, 2)}, got {agent_positions.shape}.")
    if landmark_positions.shape != (N_LANDMARKS, 2):
        raise ValueError(
            f"Expected landmark positions {(N_LANDMARKS, 2)}, got {landmark_positions.shape}."
        )

    n_nodes = N_AGENTS + N_LANDMARKS
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    # Agent--agent proximity / congestion edges.
    for i in range(N_AGENTS):
        for j in range(i + 1, N_AGENTS):
            distance = np.linalg.norm(agent_positions[i] - agent_positions[j])
            weight = AGENT_AGENT_SCALE * _gaussian_affinity(
                distance, SIGMA_AGENT_AGENT
            )
            adjacency[i, j] = weight
            adjacency[j, i] = weight

    # Agent--landmark task-affinity edges.
    for i in range(N_AGENTS):
        for landmark_index in range(N_LANDMARKS):
            distance = np.linalg.norm(
                agent_positions[i] - landmark_positions[landmark_index]
            )
            weight = _gaussian_affinity(distance, SIGMA_AGENT_LANDMARK)
            node_index = N_AGENTS + landmark_index
            adjacency[i, node_index] = weight
            adjacency[node_index, i] = weight

    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalized_adjacency = (
        inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    )
    laplacian = np.eye(n_nodes, dtype=np.float64) - normalized_adjacency
    # Suppress tiny numerical asymmetry before eigh.
    laplacian = 0.5 * (laplacian + laplacian.T)

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    eigenvalues = np.clip(eigenvalues, 0.0, 2.0)
    return adjacency, laplacian, eigenvalues, eigenvectors


def compute_hks(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    times: Iterable[float] = HKS_TIMES,
) -> np.ndarray:
    """Compute sign-invariant heat-kernel signatures for all six graph nodes.

    HKS(node, tau) = sum_k exp(-tau lambda_k) * U[node, k]^2.
    Squaring U removes the eigenvector sign ambiguity.
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64).reshape(-1)
    eigenvectors = np.asarray(eigenvectors, dtype=np.float64)
    if eigenvectors.shape != (eigenvalues.size, eigenvalues.size):
        raise ValueError("Eigenvector matrix shape is incompatible with eigenvalues.")

    time_values = np.asarray(tuple(times), dtype=np.float64)
    squared_modes = eigenvectors ** 2
    heat_weights = np.exp(-time_values[:, None] * eigenvalues[None, :])
    # (time, mode) @ (mode, node) -> (time, node), then transpose to (node, time)
    return (heat_weights @ squared_modes.T).T


def _normalized_laplacian_from_adjacency(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency, dtype=np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Expected square adjacency, got {adjacency.shape}.")
    if not np.allclose(adjacency, adjacency.T, rtol=0.0, atol=1e-10):
        raise ValueError("Adjacency must be symmetric.")
    if not np.allclose(np.diag(adjacency), 0.0, rtol=0.0, atol=1e-12):
        raise ValueError("Adjacency diagonal must be zero.")

    degree = adjacency.sum(axis=1)
    inv_sqrt_degree = 1.0 / np.sqrt(np.maximum(degree, 1e-12))
    normalized_adjacency = (
        inv_sqrt_degree[:, None] * adjacency * inv_sqrt_degree[None, :]
    )
    laplacian = np.eye(adjacency.shape[0], dtype=np.float64) - normalized_adjacency
    return 0.5 * (laplacian + laplacian.T)


def build_topology_hks_graph_from_positions(
    agent_positions: np.ndarray,
    landmark_positions: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the topology-ablation graph from reconstructed local-observation geometry.

    Node order is fixed:
        topology_3node_aa_hks: [A0, A1, A2]
        topology_6node_*: [A0, A1, A2, L0, L1, L2]
    """
    agent_positions = np.asarray(agent_positions, dtype=np.float64)
    landmark_positions = np.asarray(landmark_positions, dtype=np.float64)
    if agent_positions.shape != (N_AGENTS, 2):
        raise ValueError(f"Expected agent positions {(N_AGENTS, 2)}, got {agent_positions.shape}.")
    if landmark_positions.shape != (N_LANDMARKS, 2):
        raise ValueError(
            f"Expected landmark positions {(N_LANDMARKS, 2)}, got {landmark_positions.shape}."
        )
    if mode == TOPOLOGY_3NODE_AA_HKS:
        n_nodes = N_AGENTS
    elif mode in {TOPOLOGY_6NODE_AL_HKS, TOPOLOGY_6NODE_AAL_HKS}:
        n_nodes = N_AGENTS + N_LANDMARKS
    else:
        raise ValueError(f"Unknown topology-HKS mode: {mode}")

    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    if mode in {TOPOLOGY_3NODE_AA_HKS, TOPOLOGY_6NODE_AAL_HKS}:
        for i in range(N_AGENTS):
            for j in range(i + 1, N_AGENTS):
                distance = np.linalg.norm(agent_positions[i] - agent_positions[j])
                weight = _gaussian_affinity(distance, SIGMA_AGENT_AGENT)
                adjacency[i, j] = weight
                adjacency[j, i] = weight

    if mode in {TOPOLOGY_6NODE_AL_HKS, TOPOLOGY_6NODE_AAL_HKS}:
        for i in range(N_AGENTS):
            for landmark_index in range(N_LANDMARKS):
                distance = np.linalg.norm(
                    agent_positions[i] - landmark_positions[landmark_index]
                )
                weight = _gaussian_affinity(distance, SIGMA_AGENT_LANDMARK)
                node_index = N_AGENTS + landmark_index
                adjacency[i, node_index] = weight
                adjacency[node_index, i] = weight

    laplacian = _normalized_laplacian_from_adjacency(adjacency)
    if not np.allclose(laplacian, laplacian.T, rtol=0.0, atol=1e-10):
        raise RuntimeError("Topology-HKS Laplacian is not symmetric.")
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    if not np.all(np.isfinite(eigenvalues)):
        raise RuntimeError("Topology-HKS eigenvalues are not finite.")
    if np.min(eigenvalues) < -1e-8:
        raise RuntimeError(f"Topology-HKS eigenvalues are negative: {eigenvalues}.")
    eigenvalues = np.clip(eigenvalues, 0.0, 2.0)
    return adjacency, laplacian, eigenvalues, eigenvectors


def topology_hks_descriptor(
    raw_observation: Sequence[float], self_agent_index: int, mode: str
) -> np.ndarray:
    """Return only the focal actor's three HKS values for a topology ablation."""
    agent_positions, landmark_positions = reconstruct_geometry_from_raw_observation(
        raw_observation, self_agent_index
    )
    adjacency, laplacian, eigenvalues, eigenvectors = build_topology_hks_graph_from_positions(
        agent_positions,
        landmark_positions,
        mode=mode,
    )
    if not np.allclose(adjacency, adjacency.T, rtol=0.0, atol=1e-10):
        raise RuntimeError("Topology-HKS adjacency is not symmetric.")
    if not np.allclose(np.diag(adjacency), 0.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("Topology-HKS adjacency diagonal is not zero.")
    if not np.allclose(laplacian, laplacian.T, rtol=0.0, atol=1e-10):
        raise RuntimeError("Topology-HKS Laplacian is not symmetric.")

    hks_all_nodes = compute_hks(eigenvalues, eigenvectors)
    descriptor = hks_all_nodes[int(self_agent_index)].astype(np.float32)
    if descriptor.shape != (TOPOLOGY_HKS_DIM,):
        raise RuntimeError(
            f"Expected {TOPOLOGY_HKS_DIM} topology-HKS values, got {descriptor.shape}."
        )
    if not np.all(np.isfinite(descriptor)):
        raise RuntimeError("Topology-HKS descriptor contains NaN or infinity.")
    return descriptor


def graph_spectral_descriptor(
    raw_observation: Sequence[float], self_agent_index: int
) -> np.ndarray:
    """Return the 8D GSP-v1 descriptor available to one decentralized actor."""
    agent_positions, landmark_positions = reconstruct_geometry_from_raw_observation(
        raw_observation, self_agent_index
    )
    _, _, eigenvalues, eigenvectors = build_task_graph_from_positions(
        agent_positions, landmark_positions
    )
    hks_all_nodes = compute_hks(eigenvalues, eigenvectors)

    nontrivial_spectrum = eigenvalues[1:]  # lambda_2 ... lambda_6
    self_hks = hks_all_nodes[int(self_agent_index)]
    descriptor = np.concatenate([nontrivial_spectrum, self_hks], axis=0)

    if descriptor.shape != (GSP_DESCRIPTOR_DIM,):
        raise RuntimeError(
            f"Expected {GSP_DESCRIPTOR_DIM} GSP values, got {descriptor.shape}."
        )
    if not np.all(np.isfinite(descriptor)):
        raise RuntimeError("GSP descriptor contains NaN or infinity.")
    return descriptor.astype(np.float32)


def graph_edge_descriptor(
    raw_observation: Sequence[float], self_agent_index: int
) -> np.ndarray:
    """Return local task-edge features available from one actor observation.

    Layout:
        agent-landmark Gaussian affinities, flattened as A0-L0 ... A2-L2;
        agent-agent Gaussian affinities [A0-A1, A0-A2, A1-A2].
    """
    agent_positions, landmark_positions = reconstruct_geometry_from_raw_observation(
        raw_observation, self_agent_index
    )

    agent_landmark = []
    for agent_index in range(N_AGENTS):
        for landmark_index in range(N_LANDMARKS):
            distance = np.linalg.norm(
                agent_positions[agent_index] - landmark_positions[landmark_index]
            )
            agent_landmark.append(_gaussian_affinity(distance, SIGMA_AGENT_LANDMARK))

    agent_agent = []
    for first in range(N_AGENTS):
        for second in range(first + 1, N_AGENTS):
            distance = np.linalg.norm(agent_positions[first] - agent_positions[second])
            agent_agent.append(
                AGENT_AGENT_SCALE * _gaussian_affinity(distance, SIGMA_AGENT_AGENT)
            )

    descriptor = np.asarray(agent_landmark + agent_agent, dtype=np.float32)
    if descriptor.shape != (EDGE_FEATURE_DIM,):
        raise RuntimeError(
            f"Expected {EDGE_FEATURE_DIM} edge values, got {descriptor.shape}."
        )
    if not np.all(np.isfinite(descriptor)):
        raise RuntimeError("Edge descriptor contains NaN or infinity.")
    return descriptor


def descriptor_for_mode(
    raw_observation: Sequence[float], self_agent_index: int, mode: str = "full"
) -> np.ndarray:
    """Return a GSP-v2 descriptor variant for ablation experiments."""
    if mode in TOPOLOGY_HKS_MODES:
        return topology_hks_descriptor(raw_observation, self_agent_index, mode=mode)
    spectral_hks = graph_spectral_descriptor(raw_observation, self_agent_index)
    if mode == "full":
        return spectral_hks
    if mode == "spectrum":
        return spectral_hks[:SPECTRAL_DIM]
    if mode == "hks":
        return spectral_hks[SPECTRAL_DIM:]
    if mode == "edges":
        return graph_edge_descriptor(raw_observation, self_agent_index)
    if mode == "full_edges":
        return np.concatenate(
            [spectral_hks, graph_edge_descriptor(raw_observation, self_agent_index)],
            axis=0,
        ).astype(np.float32)
    if mode == "spectral_role":
        return graph_spectral_role_descriptor(raw_observation, self_agent_index)
    if mode == "all_node_roles":
        return graph_all_node_role_descriptor(raw_observation, self_agent_index)
    raise ValueError(f"Unknown GSP descriptor mode: {mode}")


def descriptor_dim_for_mode(mode: str = "full") -> int:
    if mode in TOPOLOGY_HKS_MODES:
        return TOPOLOGY_HKS_DIM
    if mode == "full":
        return GSP_DESCRIPTOR_DIM
    if mode == "spectrum":
        return SPECTRAL_DIM
    if mode == "hks":
        return HKS_DIM
    if mode == "edges":
        return EDGE_FEATURE_DIM
    if mode == "full_edges":
        return GSP_DESCRIPTOR_DIM + EDGE_FEATURE_DIM
    if mode == "spectral_role":
        return SPECTRAL_ROLE_DIM
    if mode == "all_node_roles":
        return ALL_NODE_ROLE_DIM
    raise ValueError(f"Unknown GSP descriptor mode: {mode}")


def graph_spectral_role_descriptor(
    raw_observation: Sequence[float], self_agent_index: int
) -> np.ndarray:
    """Compact eigenvector-aware descriptor for one actor.

    Layout:
        lambda_2..lambda_6;
        squared nontrivial eigenvector energy of the self agent node;
        self-node HKS at the configured time scales;
        nontrivial spectral distances from self agent to each landmark.
    """
    agent_positions, landmark_positions = reconstruct_geometry_from_raw_observation(
        raw_observation, self_agent_index
    )
    _, _, eigenvalues, eigenvectors = build_task_graph_from_positions(
        agent_positions, landmark_positions
    )
    hks_all_nodes = compute_hks(eigenvalues, eigenvectors)

    nontrivial_values = eigenvalues[1:]
    nontrivial_vectors = eigenvectors[:, 1:]
    self_index = int(self_agent_index)
    self_energy = nontrivial_vectors[self_index] ** 2
    self_hks = hks_all_nodes[self_index]

    spectral_distances = []
    for landmark_index in range(N_LANDMARKS):
        node_index = N_AGENTS + landmark_index
        diff = nontrivial_vectors[self_index] - nontrivial_vectors[node_index]
        spectral_distances.append(float(np.sqrt(np.sum(diff ** 2))))

    descriptor = np.concatenate(
        [
            nontrivial_values,
            self_energy,
            self_hks,
            np.asarray(spectral_distances, dtype=np.float64),
        ],
        axis=0,
    ).astype(np.float32)
    if descriptor.shape != (SPECTRAL_ROLE_DIM,):
        raise RuntimeError(
            f"Expected {SPECTRAL_ROLE_DIM} spectral-role values, got {descriptor.shape}."
        )
    if not np.all(np.isfinite(descriptor)):
        raise RuntimeError("Spectral-role descriptor contains NaN or infinity.")
    return descriptor


def graph_all_node_role_descriptor(
    raw_observation: Sequence[float], self_agent_index: int
) -> np.ndarray:
    """Eigenvector-aware descriptor exposing all six nodes' spectral roles."""
    agent_positions, landmark_positions = reconstruct_geometry_from_raw_observation(
        raw_observation, self_agent_index
    )
    _, _, eigenvalues, eigenvectors = build_task_graph_from_positions(
        agent_positions, landmark_positions
    )
    hks_all_nodes = compute_hks(eigenvalues, eigenvectors)
    nontrivial_values = eigenvalues[1:]
    all_role_energy = (eigenvectors[:, 1:] ** 2).reshape(-1)
    self_hks = hks_all_nodes[int(self_agent_index)]
    descriptor = np.concatenate(
        [nontrivial_values, all_role_energy, self_hks],
        axis=0,
    ).astype(np.float32)
    if descriptor.shape != (ALL_NODE_ROLE_DIM,):
        raise RuntimeError(
            f"Expected {ALL_NODE_ROLE_DIM} all-node role values, got {descriptor.shape}."
        )
    if not np.all(np.isfinite(descriptor)):
        raise RuntimeError("All-node role descriptor contains NaN or infinity.")
    return descriptor


def augment_observation(
    raw_observation: Sequence[float], self_agent_index: int, mode: str = "full"
) -> np.ndarray:
    """Append the selected graph descriptor to an original 18D observation."""
    raw = _as_raw_observation(raw_observation).astype(np.float32)
    descriptor = descriptor_for_mode(raw, self_agent_index, mode=mode)
    augmented = np.concatenate([raw, descriptor], axis=0).astype(np.float32)
    expected_dim = RAW_OBS_DIM + descriptor_dim_for_mode(mode)
    if augmented.shape != (expected_dim,):
        raise RuntimeError(
            f"Expected {expected_dim}D augmented observation, got {augmented.shape}."
        )
    return augmented


def augment_joint_observations(
    raw_observations: Sequence[Sequence[float]], mode: str = "full"
) -> List[np.ndarray]:
    """Augment all three agents' observations without reading privileged world state."""
    if len(raw_observations) != N_AGENTS:
        raise ValueError(
            f"Expected {N_AGENTS} per-agent observations, got {len(raw_observations)}."
        )
    return [
        augment_observation(raw_observations[agent_index], agent_index, mode=mode)
        for agent_index in range(N_AGENTS)
    ]
