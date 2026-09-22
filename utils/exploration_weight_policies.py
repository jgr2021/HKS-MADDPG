"""Fixed 6AL edge ablations and a constant-descriptor information control."""

import torch

from utils.exploration_topology_policies import BatchedTopologyPolicy, focal_hks, local_positions


def weight_adjacency(raw, mask, sigma, kernel="gaussian", support="full"):
    points = local_positions(raw, "6al")
    squared = (points[:, :, None] - points[:, None, :]).square().sum(-1)
    if kernel == "inverse":
        weights = sigma / (sigma + squared.sqrt())
    elif kernel == "binary":
        weights = torch.ones_like(squared)
    elif kernel == "gaussian":
        weights = torch.exp(-squared / (2 * sigma.square()))
    else:
        raise ValueError(f"Unknown kernel {kernel}")
    if support == "soft_radius":
        weights = weights * torch.sigmoid((1.0 - squared.sqrt()) / .15)
    elif support == "knn2_union":
        allowed_distances = squared.masked_fill(mask == 0, float("inf"))
        nearest = torch.topk(allowed_distances, k=2, dim=-1, largest=False).indices
        directed = torch.zeros_like(squared).scatter_(-1, nearest, 1.)
        weights = weights * torch.maximum(directed, directed.transpose(-1, -2))
    elif support != "full":
        raise ValueError(f"Unknown support {support}")
    return weights * mask


class Weight6ALPolicy(BatchedTopologyPolicy):
    sigma_multiplier = 1.
    kernel = "gaussian"
    support = "full"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_sigma.mul_(self.sigma_multiplier)

    @torch.no_grad()
    def _hks_features(self, raw):
        return focal_hks(weight_adjacency(raw, self.edge_mask, self.edge_sigma,
                                         self.kernel, self.support), self.hks_times)


class ConstantStarHKSPolicy(BatchedTopologyPolicy):
    @torch.no_grad()
    def _hks_features(self, raw):
        if raw.ndim != 2 or raw.shape[1] != 18 or raw.dtype != torch.float32:
            raise ValueError("Expected float32 raw local observations [M,18]")
        return (.5 * (1 + torch.exp(-2 * self.hks_times))).expand(raw.shape[0], -1)


class NarrowGaussianPolicy(Weight6ALPolicy):
    sigma_multiplier = .75


class WideGaussianPolicy(Weight6ALPolicy):
    sigma_multiplier = 1.25


class BoundedInversePolicy(Weight6ALPolicy):
    kernel = "inverse"


class SoftRadiusPolicy(Weight6ALPolicy):
    support = "soft_radius"


class KNN2UnionPolicy(Weight6ALPolicy):
    support = "knn2_union"


POLICIES = {
    "explore_constant_star3": ConstantStarHKSPolicy,
    "explore_hks_sigma075": NarrowGaussianPolicy,
    "explore_hks_sigma125": WideGaussianPolicy,
    "explore_hks_inverse": BoundedInversePolicy,
    "explore_hks_soft_radius": SoftRadiusPolicy,
    "explore_hks_knn2_union": KNN2UnionPolicy,
}


def register_policies():
    from utils.agents import POLICY_TYPES
    POLICY_TYPES.update(POLICIES)


def experiment_methods(repo):
    from utils.exploration_topology_policies import experiment_methods as topology_methods
    topology = repo / "experiments/gsp_exploration_20260906_phase2a"
    previous = topology_methods(repo / "experiments/gsp_exploration_20260906_phase1/formal/raw_mlp/seed_1")
    methods = {}
    for name in ("raw_mlp", "explore_geometric_stats3", "explore_hks_6al"):
        spec = dict(previous[name])
        if name != "raw_mlp":
            spec["reuse_from"] = str(topology / "formal" / name / "seed_1")
        methods[name] = spec
    for name in POLICIES:
        methods[name] = {"actor_model": name, "actor_input_dim": 21,
                         "short": name.replace("explore_", ""), "control": "explore_geometric_stats3",
                         "factor": "constant star HKS control" if name == "explore_constant_star3" else name.replace("explore_", "6AL "),
                         "additional_controls": [] if name == "explore_constant_star3" else ["explore_hks_6al", "explore_constant_star3"]}
        methods[name]["hypothesis"] = "Unproven: edge saturation, normalization/degree-floor effects, or weak directional information in focal HKS. The constant control has no graph information."
    return methods
