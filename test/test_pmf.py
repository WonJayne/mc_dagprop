import unittest

import numpy as np
from mc_dagprop.discrete.pmf import DiscretePMF


class TestDiscretePMF(unittest.TestCase):
    def test_truncate_right_below_first_value(self) -> None:
        pmf = DiscretePMF(np.array([1.0, 2.0]), np.array([0.5, 0.5]))
        truncated = pmf.truncate_right(0.5)
        self.assertTrue(np.allclose(truncated.values, [0.5]))
        self.assertTrue(np.allclose(truncated.probs, [1.0]))

    def test_maximum_aligned(self) -> None:
        pmf_a = DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        pmf_b = DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        result = pmf_a.maximum(pmf_b)
        self.assertTrue(np.allclose(result.values, [0.0, 1.0]))
        self.assertTrue(np.allclose(result.probs, [0.25, 0.75]))

    def test_minimum_aligned_offset(self) -> None:
        pmf_a = DiscretePMF(np.array([1.0, 2.0]), np.array([0.5, 0.5]))
        pmf_b = DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5]))
        result = pmf_a.minimum(pmf_b)
        self.assertTrue(np.allclose(result.values, [0.0, 1.0, 2.0]))
        self.assertTrue(np.allclose(result.probs, [0.5, 0.5, 0.0]))


if __name__ == "__main__":
    unittest.main()
