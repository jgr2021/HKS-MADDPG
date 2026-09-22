"""Actor-only contact correction; registration is explicit for frozen studies."""

import torch

from utils.contact_active_gsp_features import compute_contact_active_features
from utils.d4_graph_residual import D4GraphResidualPolicy
from utils.networks import ACTION_TO_CONTROL


class ContactD4ResidualPolicy(D4GraphResidualPolicy):
    use_energies = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("contact_controls", torch.tensor(ACTION_TO_CONTROL), persistent=False)

    def action_features(self, observations):
        features = compute_contact_active_features(observations, self.contact_controls)
        if not self.use_energies:
            features = features * features.new_tensor([1, 1, 0, 0])
        return features


class ContactD4PotentialPolicy(ContactD4ResidualPolicy):
    use_energies = False


def register_policies():
    from utils.agents import POLICY_TYPES
    POLICY_TYPES.update({
        "explore_contact_d4_geometry": ContactD4PotentialPolicy,
        "explore_contact_d4_energy": ContactD4ResidualPolicy,
    })
