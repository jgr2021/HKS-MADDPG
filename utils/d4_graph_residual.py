"""D4-equivariant graph residuals; the original raw actor branch is unchanged."""

import torch

from .networks import ACTION_TO_CONTROL, LearnedActionFeatureResidualPolicy


def square_symmetries():
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    reflection = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
    rotations = [torch.eye(2)]
    for _ in range(3):
        rotations.append(rotation @ rotations[-1])
    return torch.stack(rotations + [item @ reflection for item in rotations])


def transform_local_geometry(observations, matrices):
    """Transform observed velocity/relative positions, discarding absolute position."""
    if observations.ndim != 2 or observations.shape[1] != 18:
        raise ValueError("D4 graph residual expects [batch,18] raw local observations")
    vectors = observations[:, :14].reshape(-1, 7, 2).clone()
    vectors[:, 1] = 0.0
    transformed = torch.einsum("gij,bnj->bgni", matrices.to(observations), vectors)
    communications = observations[:, None, 14:18].expand(-1, len(matrices), -1)
    return torch.cat([transformed.flatten(2), communications], dim=2)


class D4GraphResidualPolicy(LearnedActionFeatureResidualPolicy):
    """Average graph-residual scores over the eight square symmetries.

    Only the residual is equivariant and translation invariant. The raw MLP
    remains unchanged and still receives the original raw observation once.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        matrices = square_symmetries()
        controls = torch.tensor(ACTION_TO_CONTROL)
        transformed_controls = torch.einsum("gij,aj->gai", matrices, controls)
        permutation = (transformed_controls[:, :, None] - controls[None, None]).square().sum(-1).argmin(-1)
        self.register_buffer("d4_matrices", matrices, persistent=False)
        self.register_buffer("d4_actions", permutation, persistent=False)
        self.last_orbit_input_shape = None

    def graph_residual(self, observations):
        orbit = transform_local_geometry(observations, self.d4_matrices)
        batch_size = len(observations)
        local_rows = orbit.reshape(-1, 18)
        features = self.action_features(local_rows)
        context = self.context_encoder(local_rows).unsqueeze(1).expand(-1, 5, -1)
        encoded = self.feature_encoder(torch.tanh(features * self.feature_scale.to(local_rows)))
        actions = self.action_embedding(self.action_indices).unsqueeze(0).expand(len(local_rows), -1, -1)
        scores = self.residual_scorer(torch.cat([context, encoded, actions], dim=2)).squeeze(2)
        scores = scores.reshape(batch_size, 8, 5)
        aligned = scores.gather(2, self.d4_actions.unsqueeze(0).expand(batch_size, -1, -1))
        self.last_raw_input_shape = tuple(observations.shape)
        self.last_action_feature_shape = tuple(features.shape)
        self.last_orbit_input_shape = tuple(orbit.shape)
        return aligned.mean(dim=1)

    def forward(self, observations):
        residual = self.graph_residual(observations)
        return self.mlp(observations) + residual


class D4PotentialResidualPolicy(D4GraphResidualPolicy):
    feature_mode = "action_raw_potential"


class D4DirichletResidualPolicy(D4GraphResidualPolicy):
    feature_mode = "active_gsp_v3"
