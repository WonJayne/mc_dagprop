import unittest

import numpy as np

from mc_dagprop.analytic import (
    constant_pmf,
    empirical_pmf,
    exponential_pmf,
    gamma_pmf,
)


class TestDistributionHelpers(unittest.TestCase):
    def test_constant_pmf(self) -> None:
        pmf = constant_pmf(5.0, step=1.0)
        pmf.validate()
        pmf.validate_alignment(1.0)
        self.assertTrue(np.allclose(pmf.values, [5.0]))
        self.assertTrue(np.allclose(pmf.probabilities, [1.0]))

    def test_empirical_pmf(self) -> None:
        pmf = empirical_pmf([0.0, 1.0, 2.0], [1, 1, 2], step=1.0)
        pmf.validate()
        pmf.validate_alignment(1.0)
        self.assertAlmostEqual(pmf.probabilities.sum(), 1.0, places=6)

    def test_exponential_example(self) -> None:
        pmf = exponential_pmf(scale=10.0, step=1.0, start=0.0, stop=300.0)
        pmf.validate()
        pmf.validate_alignment(1.0)
        self.assertAlmostEqual(pmf.probabilities.sum(), 1.0, places=6)
        self.assertEqual(pmf.values[0], 0.0)
        self.assertEqual(pmf.step, 1.0)

    def test_gamma_pmf(self) -> None:
        pmf = gamma_pmf(shape=2.0, scale=2.0, step=1.0, start=0.0, stop=20.0)
        pmf.validate()
        pmf.validate_alignment(1.0)
        self.assertAlmostEqual(pmf.probabilities.sum(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
