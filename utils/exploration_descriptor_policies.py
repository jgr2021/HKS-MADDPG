"""Three-channel descriptors on the fixed actor-local normalized 6AL graph."""

import torch

from utils.exploration_topology_policies import BatchedTopologyPolicy, adjacency_from_local_obs


def normalized_laplacian(adjacency):
    inverse = adjacency.sum(-1).clamp_min(1e-12).rsqrt()
    laplacian = torch.eye(6, dtype=adjacency.dtype, device=adjacency.device) - inverse[:, :, None] * adjacency * inverse[:, None, :]
    return (laplacian + laplacian.transpose(-1, -2)) * .5


def descriptor_features(adjacency, descriptor, log_centers):
    if descriptor == "landmark_affinity":
        return adjacency[:, 0, 3:6]
    laplacian = normalized_laplacian(adjacency)
    if descriptor == "regularized_resistance":
        identity = torch.eye(6, dtype=adjacency.dtype, device=adjacency.device).expand_as(laplacian)
        kernel = torch.linalg.solve(laplacian + .1 * identity, identity)
        diagonal = kernel.diagonal(dim1=-2, dim2=-1)
        contrast = diagonal[:, :1] + diagonal[:, 3:6] - kernel[:, 0, 3:6] - kernel[:, 3:6, 0]
        return .05 * contrast.clamp_min(0)
    values, vectors = torch.linalg.eigh(laplacian)
    values = values.clamp(0, 2)
    if descriptor == "landmark_heat":
        heat = torch.exp(-values)
        return (vectors[:, :1, :] * vectors[:, 3:6, :] * heat[:, None, :]).sum(-1)
    if descriptor == "wks":
        log_energy = values.clamp_min(1e-6).log().unsqueeze(-1)
        weights = torch.exp(-.5 * ((log_energy - log_centers) / .5).square())
        weights = weights * (values > 1e-6).unsqueeze(-1)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
        return (vectors[:, 0, :].square().unsqueeze(-1) * weights).sum(1)
    raise ValueError("Unknown descriptor: " + descriptor)


class Descriptor6ALPolicy(BatchedTopologyPolicy):
    descriptor = "landmark_affinity"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("wks_log_centers", torch.tensor([.25, .75, 1.5]).log(), persistent=False)

    @torch.no_grad()
    def _hks_features(self, raw):
        adjacency = adjacency_from_local_obs(raw, "6al", self.edge_mask, self.edge_sigma)
        return descriptor_features(adjacency, self.descriptor, self.wks_log_centers)


class WKSPolicy(Descriptor6ALPolicy):
    descriptor = "wks"


class LandmarkHeatPolicy(Descriptor6ALPolicy):
    descriptor = "landmark_heat"


class RegularizedResistancePolicy(Descriptor6ALPolicy):
    descriptor = "regularized_resistance"


POLICIES = {
    "explore_landmark_affinity3": Descriptor6ALPolicy,
    "explore_wks3": WKSPolicy,
    "explore_landmark_heat3": LandmarkHeatPolicy,
    "explore_regularized_resistance3": RegularizedResistancePolicy,
}


def register_policies():
    from utils.agents import POLICY_TYPES
    POLICY_TYPES.update(POLICIES)


def experiment_methods(repo):
    from utils.exploration_operator_policies import experiment_methods as operator_methods
    previous = operator_methods(repo)
    methods = {name: dict(spec) for name, spec in previous.items() if spec.get("reuse_from")}
    for name, cls in POLICIES.items():
        methods[name] = {"actor_model": name, "actor_input_dim": 21, "short": name.replace("explore_", ""),
                         "control": "explore_geometric_stats3", "factor": "6AL " + cls.descriptor,
                         "additional_controls": ["explore_hks_6al", "explore_constant_star3"],
                         "hypothesis": "Unproven: spectral bandwidth, landmark-channel information, normalized contrast scaling, or decision sensitivity; seed1 screen only."}
        if cls.descriptor in ("landmark_heat", "regularized_resistance"):
            methods[name]["additional_controls"] += ["explore_landmark_affinity3"]
            methods[name]["required_additional_controls"] = ["explore_landmark_affinity3"]
        elif cls.descriptor == "wks":
            methods[name]["additional_controls"] += ["explore_landmark_affinity3"]
    return methods
