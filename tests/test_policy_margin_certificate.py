import unittest

import numpy as np

from experiments.analyze_hybrid_policy_margin_drift import (
    pinsker_argmax_certificate,
)
from experiments.analyze_cardinality_policy_drift import (
    initial_margin_quartile_masks,
)


class PolicyMarginCertificateTests(unittest.TestCase):
    def test_certificate_matches_pinsker_threshold(self):
        margins = np.asarray([0.10, 0.20, 0.30, 0.40])
        kl = np.asarray([0.0049, 0.0200, 0.0449, 0.0800])
        expected = margins > np.sqrt(2.0 * kl)
        np.testing.assert_array_equal(
            pinsker_argmax_certificate(margins, kl), expected
        )

    def test_certified_distributions_retain_argmax(self):
        reference = np.asarray([
            [0.80, 0.10, 0.10],
            [0.55, 0.40, 0.05],
            [0.45, 0.35, 0.20],
        ])
        candidate = np.asarray([
            [0.76, 0.14, 0.10],
            [0.50, 0.45, 0.05],
            [0.36, 0.44, 0.20],
        ])
        margins = np.sort(reference, axis=1)[:, -1] - np.sort(
            reference, axis=1
        )[:, -2]
        kl = np.sum(
            reference * (np.log(reference) - np.log(candidate)), axis=1
        )
        certified = pinsker_argmax_certificate(margins, kl)
        agreement = reference.argmax(axis=1) == candidate.argmax(axis=1)
        self.assertTrue(np.all(agreement[certified]))
        self.assertFalse(agreement[-1])

    def test_rank_quartiles_are_equal_sized_with_boundary_ties(self):
        margins = np.asarray([0.1] * 3 + [1.0] * 9)
        masks, thresholds = initial_margin_quartile_masks(margins, "rank")
        self.assertEqual([int(mask.sum()) for mask in masks], [3, 3, 3, 3])
        self.assertEqual(thresholds, [])
        np.testing.assert_array_equal(
            np.sum(np.asarray(masks), axis=0), np.ones(len(margins))
        )

    def test_threshold_quartiles_retain_original_tied_behavior(self):
        margins = np.asarray([0.1] * 3 + [1.0] * 9)
        masks, thresholds = initial_margin_quartile_masks(
            margins, "threshold"
        )
        self.assertEqual([int(mask.sum()) for mask in masks], [3, 9, 0, 0])
        self.assertEqual(len(thresholds), 3)


if __name__ == "__main__":
    unittest.main()
