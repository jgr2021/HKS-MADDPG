"""Batch focal HKS topology screen; actor-local and raw-critic compatible."""

import torch

from utils.networks import PassiveTopologyHKSPolicy


TOPOLOGIES = ("3aa", "4ego", "6al", "6aal", "6all")


def local_positions(raw, topology):
    if raw.ndim != 2 or raw.shape[1] != 18 or raw.dtype != torch.float32:
        raise ValueError("Expected float32 raw local observations [M,18]")
    self_position = torch.zeros_like(raw[:, :2]).unsqueeze(1)
    agents = torch.cat((self_position, raw[:, 10:14].reshape(-1, 2, 2)), dim=1)
    landmarks = raw[:, 4:10].reshape(-1, 3, 2)
    if topology == "3aa":
        return agents
    if topology == "4ego":
        return torch.cat((self_position, landmarks), dim=1)
    if topology in ("6al", "6aal", "6all"):
        return torch.cat((agents, landmarks), dim=1)
    raise ValueError(f"Unknown topology {topology}")


def topology_constants(topology):
    n_agents = 1 if topology == "4ego" else 3
    n_nodes = 3 if topology == "3aa" else 4 if topology == "4ego" else 6
    mask = torch.zeros(n_nodes, n_nodes, dtype=torch.float32)
    sigma = torch.full_like(mask, .6)
    sigma[:n_agents, :n_agents] = .8
    if topology in ("3aa", "6aal", "6all"):
        mask[:n_agents, :n_agents] = 1
    if topology != "3aa":
        mask[:n_agents, n_agents:] = 1
        mask[n_agents:, :n_agents] = 1
    if topology == "6all":
        mask[n_agents:, n_agents:] = 1
    mask.fill_diagonal_(0)
    return mask, sigma


def adjacency_from_local_obs(raw, topology, mask, sigma):
    points = local_positions(raw, topology)
    squared = (points[:, :, None] - points[:, None, :]).square().sum(-1)
    return torch.exp(-squared / (2 * sigma.square())) * mask


def focal_hks(adjacency, times):
    inverse = adjacency.sum(-1).clamp_min(1e-12).rsqrt()
    normalized = inverse[:, :, None] * adjacency * inverse[:, None, :]
    laplacian = torch.eye(adjacency.shape[-1], device=adjacency.device) - normalized
    laplacian = (laplacian + laplacian.transpose(-1, -2)) * .5
    values, vectors = torch.linalg.eigh(laplacian)
    heat_weights = torch.exp(-values.clamp(0, 2).unsqueeze(-1) * times)
    return (vectors[:, 0, :].square().unsqueeze(-1) * heat_weights).sum(1)


def geometric_statistics(raw):
    points = local_positions(raw, "6al")
    agents, landmarks = points[:, :3], points[:, 3:]
    distances = (agents[:, :, None] - landmarks[:, None, :]).square().sum(-1).sqrt()
    aa_squared = (agents[:, :, None] - agents[:, None, :]).square().sum(-1)
    weights = torch.exp(-aa_squared / (2 * .8 ** 2))
    coverage = distances.min(1).values.sum(1)
    crowding = weights[:, 0, 1] + weights[:, 0, 2] + weights[:, 1, 2]
    focal_distance = distances[:, 0].mean(1)
    return torch.stack((coverage, crowding, focal_distance), dim=1)


class BatchedTopologyPolicy(PassiveTopologyHKSPolicy):
    topology = "6al"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mask, sigma = topology_constants(self.topology)
        self.register_buffer("edge_mask", mask, persistent=False)
        self.register_buffer("edge_sigma", sigma, persistent=False)
        self.register_buffer("hks_times", torch.tensor([.5, 1., 2.]), persistent=False)

    @torch.no_grad()
    def _hks_features(self, raw):
        adjacency = adjacency_from_local_obs(raw, self.topology, self.edge_mask, self.edge_sigma)
        return focal_hks(adjacency, self.hks_times)


class GeometricStatisticsPolicy(BatchedTopologyPolicy):
    @torch.no_grad()
    def _hks_features(self, raw):
        return geometric_statistics(raw)


class Topology3AAPolicy(BatchedTopologyPolicy):
    topology = "3aa"


class Topology4EgoPolicy(BatchedTopologyPolicy):
    topology = "4ego"


class Topology6ALPolicy(BatchedTopologyPolicy):
    topology = "6al"


class Topology6AALPolicy(BatchedTopologyPolicy):
    topology = "6aal"


class Topology6AllPolicy(BatchedTopologyPolicy):
    topology = "6all"


POLICIES = {
    "explore_geometric_stats3": GeometricStatisticsPolicy,
    "explore_hks_3aa": Topology3AAPolicy,
    "explore_hks_4ego": Topology4EgoPolicy,
    "explore_hks_6al": Topology6ALPolicy,
    "explore_hks_6aal": Topology6AALPolicy,
    "explore_hks_6all": Topology6AllPolicy,
}


def register_policies():
    from utils.agents import POLICY_TYPES
    POLICY_TYPES.update(POLICIES)


def experiment_methods(raw_source):
    methods = {"raw_mlp": {"actor_model": "mlp", "short": "raw", "actor_input_dim": 18,
                           "control": None, "factor": "reused unchanged phase1 CUDA raw baseline",
                           "reuse_from": str(raw_source)}}
    for name in POLICIES:
        methods[name] = {"actor_model": name, "short": name.replace("explore_", ""),
                         "actor_input_dim": 21,
                         "control": "raw_mlp" if name == "explore_geometric_stats3" else "explore_geometric_stats3",
                         "factor": "three non-spectral geometric statistics" if name == "explore_geometric_stats3" else name.replace("explore_", "focal ")}
    return methods
