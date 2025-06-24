import unittest

import numpy as np
from mc_dagprop.analytic.pmf import DiscretePMF


class TestDiscretePMF(unittest.TestCase):

    def test_maximum_aligned(self) -> None:
        pmf_a = DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5]), step=1.0)
        pmf_b = DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5]), step=1.0)
        result = pmf_a.maximum(pmf_b)
        self.assertTrue(np.allclose(result.values, [0.0, 1.0]))
        self.assertTrue(np.allclose(result.probabilities, [0.25, 0.75]))



if __name__ == "__main__":
    unittest.main()
