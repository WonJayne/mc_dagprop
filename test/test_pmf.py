import unittest

import numpy as np

from mc_dagprop.analytic._pmf import DiscretePMF


class TestDiscretePMF(unittest.TestCase):

    def test_maximum_aligned(self) -> None:
        pmf_a = DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5]), step=1)
        pmf_b = DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5]), step=1)
        result = pmf_a.maximum(pmf_b)
        self.assertTrue(np.allclose(result.values, [0.0, 1.0]))
        self.assertTrue(np.allclose(result.probabilities, [0.25, 0.75]))

    def test_maximum_tiny_tail_mass_stays_non_negative(self) -> None:
        tiny_mass = 5e-324
        pmf_a = DiscretePMF(np.array([0.0, 1.0]), np.array([1.0 - tiny_mass, tiny_mass]), step=1)
        pmf_b = DiscretePMF(np.array([0.0, 1.0]), np.array([1.0 - tiny_mass, tiny_mass]), step=1)

        result = pmf_a.maximum(pmf_b)

        self.assertTrue(np.all(result.probabilities >= 0.0))
        self.assertTrue(np.isclose(float(result.total_mass), 1.0, rtol=1e-12, atol=1e-15))

    def test_convolve_uses_stable_dtype_for_small_probabilities(self) -> None:
        probabilities = np.full(256, 1.0 / 256.0)
        pmf_a = DiscretePMF(np.arange(256, dtype=float), probabilities, step=1)
        pmf_b = DiscretePMF(np.arange(256, dtype=float), probabilities, step=1)

        result = pmf_a.convolve(pmf_b)

        self.assertTrue(np.isclose(float(result.total_mass), 1.0, rtol=1e-12, atol=1e-15))
        self.assertTrue(np.all(result.probabilities >= 0.0))


if __name__ == "__main__":
    unittest.main()
