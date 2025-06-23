import unittest

import numpy as np
from mc_dagprop.discrete.pmf import DiscretePMF


class TestDiscretePMF(unittest.TestCase):
    def test_truncate_right_below_first_value(self) -> None:
        pmf = DiscretePMF(np.array([1.0, 2.0]), np.array([0.5, 0.5]))
        truncated = pmf.truncate_right(0.5)
        self.assertTrue(np.allclose(truncated.values, [0.5]))
        self.assertTrue(np.allclose(truncated.probs, [1.0]))


if __name__ == "__main__":
    unittest.main()
