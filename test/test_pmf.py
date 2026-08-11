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

    def test_maximum_corrects_mass_before_validating_operation_result(self) -> None:
        accepted_roundoff = np.array([0.5, 0.5000000000009])
        pmf = DiscretePMF(np.array([0.0, 1.0]), accepted_roundoff, step=1, allow_subprobability=True)

        result = pmf.maximum(pmf)

        self.assertEqual(float(result.total_mass), 1.0)
        np.testing.assert_allclose(result.probabilities, [0.25, 0.75], rtol=0.0, atol=1.0e-12)


    def test_operation_results_are_immutable(self) -> None:
        pmf = DiscretePMF(np.array([0.0, 1.0]), np.array([0.4, 0.6]), step=1)

        results = (pmf.shift(1.0), pmf.convolve(pmf), pmf.maximum(pmf))

        for result in results:
            with self.subTest(values=result.values):
                self.assertFalse(result.values.flags.writeable)
                self.assertFalse(result.probabilities.flags.writeable)
                with self.assertRaises(ValueError):
                    result.values[0] = 10.0
                with self.assertRaises(ValueError):
                    result.probabilities[0] = 0.0

    def test_trusted_operation_results_match_public_construction(self) -> None:
        pmf_a = DiscretePMF(np.array([0.0, 2.0]), np.array([0.25, 0.75]), step=1)
        pmf_b = DiscretePMF(np.array([0.0, 1.0]), np.array([0.6, 0.4]), step=1)

        for result in (pmf_a.shift(2.0), pmf_a.convolve(pmf_b), pmf_a.maximum(pmf_b)):
            with self.subTest(values=result.values):
                public_result = DiscretePMF(
                    result.values,
                    result.probabilities,
                    step=result.step,
                    allow_subprobability=result.allow_subprobability,
                )
                np.testing.assert_array_equal(result.values, public_result.values)
                np.testing.assert_allclose(result.probabilities, public_result.probabilities, rtol=0.0, atol=1e-15)

    def test_trusted_operation_preserves_subprobability_mass(self) -> None:
        pmf = DiscretePMF(
            np.array([0.0, 1.0]),
            np.array([0.2, 0.3]),
            step=1,
            allow_subprobability=True,
        )

        shifted = pmf.shift(2.0)
        convolved = pmf.convolve(DiscretePMF.delta(0.0, step=1))

        self.assertTrue(shifted.allow_subprobability)
        self.assertTrue(convolved.allow_subprobability)
        self.assertEqual(float(shifted.total_mass), 0.5)
        self.assertEqual(float(convolved.total_mass), 0.5)

    def test_public_constructor_still_canonicalizes_duplicates(self) -> None:
        pmf = DiscretePMF(
            np.array([1.0, 0.0, 1.0]),
            np.array([0.2, 0.3, 0.5]),
            step=1,
        )

        np.testing.assert_array_equal(pmf.values, [0.0, 1.0])
        np.testing.assert_allclose(pmf.probabilities, [0.3, 0.7], rtol=0.0, atol=1e-15)


if __name__ == "__main__":
    unittest.main()
