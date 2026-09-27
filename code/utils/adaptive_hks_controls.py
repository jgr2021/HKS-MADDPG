"""Matched actor controls for the confirmatory Adaptive HKS experiment."""
import torch
from utils.catalog76_policies import FeaturePolicy, FEATURE_RECIPES, graph


class AdjacencyStatisticsPolicy(FeaturePolicy):
    recipe = 'adaptive_hks'
    adaptive = False

    def augment_observation(self, raw):
        with torch.no_grad():
            w = graph(raw, {'kernel': 'adaptive' if self.adaptive else 'gaussian'})
            # Three permutation-invariant, non-diffusive focal affinity statistics.
            focal = w[:, 0, 3:6]
            feature = torch.stack((focal.mean(-1), focal.amax(-1),
                                   focal.std(-1, unbiased=False)), -1)
        self.last_raw_input_shape = tuple(raw.shape)
        augmented = torch.cat((raw, feature), -1)
        self.last_augmented_input_shape = tuple(augmented.shape)
        return augmented


class AdaptiveAdjacencyStatisticsPolicy(AdjacencyStatisticsPolicy):
    adaptive = True


class FixedHKSPolicy(FeaturePolicy):
    recipe = 'confirm_fixed_hks'


def register_policies():
    from utils.catalog76_policies import register_policies as catalog
    from utils.agents import POLICY_TYPES
    FEATURE_RECIPES['confirm_fixed_hks'] = {}
    catalog()
    POLICY_TYPES.update({
        'confirm_fixed_hks': FixedHKSPolicy,
        'confirm_fixed_adjacency': AdjacencyStatisticsPolicy,
        'confirm_adaptive_adjacency': AdaptiveAdjacencyStatisticsPolicy,
    })


METHODS = {
    'raw': ('mlp', 18),
    'geometry': ('catalog_adaptive_hks_control', 21),
    'fixed_adjacency': ('confirm_fixed_adjacency', 21),
    'adaptive_adjacency': ('confirm_adaptive_adjacency', 21),
    'fixed_hks': ('confirm_fixed_hks', 21),
    'adaptive_hks': ('catalog_adaptive_hks', 21),
}
