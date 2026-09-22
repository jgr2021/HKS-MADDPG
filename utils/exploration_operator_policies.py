"""Fixed 6AL operator ablations with three actor-local descriptor values."""

import torch

from utils.exploration_topology_policies import (
    BatchedTopologyPolicy, adjacency_from_local_obs, focal_hks)


def operator_features(adjacency, times, operator):
    identity = torch.eye(6, dtype=adjacency.dtype, device=adjacency.device)
    if operator == "self_loop_hks":
        return focal_hks(adjacency + identity, times)
    if operator == "combinatorial_hks":
        laplacian = torch.diag_embed(adjacency.sum(-1)) - adjacency
        values, vectors = torch.linalg.eigh((laplacian + laplacian.transpose(-1, -2)) * .5)
        heat = torch.exp(-values.clamp_min(0).unsqueeze(-1) * times)
        return (vectors[:, 0, :].square().unsqueeze(-1) * heat).sum(1)
    inverse = adjacency.sum(-1).clamp_min(1e-12).rsqrt()
    transition = inverse[:, :, None] * adjacency * inverse[:, None, :]
    if operator == "lazy_rwse":
        transition = .5 * (identity + transition)
    elif operator != "rwse":
        raise ValueError("Unknown graph operator: " + operator)
    # Diagonal powers of S equal those of P; use one symmetric implementation.
    power2 = transition @ transition
    power4 = power2 @ power2
    power8 = power4 @ power4
    return torch.stack((power2[:, 0, 0], power4[:, 0, 0], power8[:, 0, 0]), dim=1)


class Operator6ALPolicy(BatchedTopologyPolicy):
    operator = "combinatorial_hks"

    @torch.no_grad()
    def _hks_features(self, raw):
        adjacency = adjacency_from_local_obs(raw, "6al", self.edge_mask, self.edge_sigma)
        return operator_features(adjacency, self.hks_times, self.operator)


class SelfLoopHKSPolicy(Operator6ALPolicy):
    operator = "self_loop_hks"


class RWSEPolicy(Operator6ALPolicy):
    operator = "rwse"


class LazyRWSEPolicy(Operator6ALPolicy):
    operator = "lazy_rwse"


POLICIES = {
    "explore_hks_combinatorial": Operator6ALPolicy,
    "explore_hks_self_loop": SelfLoopHKSPolicy,
    "explore_rwse_248": RWSEPolicy,
    "explore_lazy_rwse_248": LazyRWSEPolicy,
}


def register_policies():
    from utils.agents import POLICY_TYPES
    POLICY_TYPES.update(POLICIES)


def experiment_methods(repo):
    from utils.exploration_weight_policies import experiment_methods as weight_methods
    previous = weight_methods(repo)
    methods = {name: dict(previous[name]) for name in (
        "raw_mlp", "explore_geometric_stats3", "explore_hks_6al", "explore_constant_star3")}
    methods["explore_constant_star3"]["reuse_from"] = str(
        repo / "experiments/gsp_exploration_20260906_phase2b/formal/explore_constant_star3/seed_1")
    for name, cls in POLICIES.items():
        methods[name] = {"actor_model": name, "actor_input_dim": 21, "short": name.replace("explore_", ""),
                         "control": "explore_geometric_stats3", "factor": "6AL " + cls.operator,
                         "additional_controls": ["explore_hks_6al", "explore_constant_star3"],
                         "hypothesis": "Unproven: operator scale, return-probability readout compression, or coverage/collision tradeoff; seed1 only."}
    return methods
