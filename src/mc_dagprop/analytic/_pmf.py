from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mc_dagprop.types import ProbabilityMass, Second


@dataclass(frozen=True, slots=True)
class DiscretePMF:
    """Simple probability mass function on an equidistant grid."""

    values: np.ndarray
    probabilities: np.ndarray
    # Step size that determines the grid spacing for ``values``.
    step: int

    def __post_init__(self) -> None:
        """Basic sanity checks for the distribution."""
        self.validate()
        if len(self.values) != len(self.probabilities):
            raise ValueError("values and probs must have same length")
        if self.step < 0.0:
            raise ValueError("step size must be non-negative")
        if not isinstance(self.step, int):
            raise OverflowError(
                f"step must be an integer number of seconds, got: {self.step} (type: {type(self.step)})"
                "we limit to ints to avoid floating point precision issues"
            )

    def validate(self) -> None:
        """Validate the PMF properties."""
        if len(self.values) == 0:
            raise ValueError("PMF values cannot be empty")
        if len(self.values) != len(self.probabilities):
            raise ValueError("values and probs must have same length")
        if len(self.values) > 1 and not np.all(self.values[1:] >= self.values[:-1]):
            raise ValueError("values must be sorted in non-decreasing order")
        if not (1.0 >= self.probabilities.sum() or np.isclose(self.probabilities.sum(), 1.0)):
            raise ValueError("Probabilities must sum to <= 1.0")

    def validate_alignment(self, step: Second) -> None:
        """Ensure that ``values`` align with ``step`` spacing."""
        if not np.isclose(self.step, step):
            raise ValueError(f"PMF step {self.step} does not match expected {step}")
        if step <= 0.0:
            raise ValueError("step must be positive")

        if len(self.values) == 0:
            raise ValueError("PMF values cannot be empty")

        if len(self.values) > 1:
            diffs = np.diff(self.values)
            if not np.allclose(diffs, step):
                raise ValueError("PMF grid spacing does not match step")

        if self.values.size > 0 and not np.isclose(self.values[0] % step, 0.0):
            raise ValueError("PMF values are not aligned to step grid")

    @staticmethod
    def delta(v: Second, step: Second) -> "DiscretePMF":
        """Return a unit mass at ``v`` using ``step`` spacing."""
        return DiscretePMF(np.array([v], dtype=float), np.array([1.0], dtype=float), step=step)

    @property
    def total_mass(self) -> ProbabilityMass:
        """Return the total mass of the PMF."""
        return ProbabilityMass(self.probabilities.sum())

    def shift(self, delta: Second) -> "DiscretePMF":
        """Shift the PMF by ``delta`` seconds."""
        return DiscretePMF(self.values + delta, self.probabilities.copy(), step=self.step)

    def _rescale(self, expected: float) -> "DiscretePMF":
        """Return a copy with probabilities scaled to expected mass."""
        probs = self.probabilities.copy()
        total = probs.sum()
        if total > 0 and not np.isclose(total, expected, rtol=1e-12, atol=1e-15):
            probs *= expected / total
        return DiscretePMF(self.values.copy(), probs, step=self.step)

    @staticmethod
    def _expected_mass(m1: float, m2: float) -> float:
        """Expected result mass for binary ops with drift correction."""
        if np.isclose(m1, 1.0, rtol=1e-12, atol=1e-15) and np.isclose(m2, 1.0, rtol=1e-12, atol=1e-15):
            return 1.0
        return m1 * m2

    def convolve(self, other: "DiscretePMF") -> "DiscretePMF":
        if len(self.values) == 1:
            a, p = self.values[0], self.probabilities[0]
            pmf = DiscretePMF(other.values + a, other.probabilities * p, step=self.step)
        elif len(other.values) == 1:
            b, q = other.values[0], other.probabilities[0]
            pmf = DiscretePMF(self.values + b, self.probabilities * q, step=self.step)
        else:
            start = self.values[0] + other.values[0]
            probs = np.convolve(self.probabilities, other.probabilities)
            values = start + self.step * np.arange(len(probs))
            pmf = DiscretePMF(values, probs, step=self.step)

        expected = self._expected_mass(float(self.total_mass), float(other.total_mass))
        return pmf._rescale(expected)

    def maximum(self, other: "DiscretePMF") -> "DiscretePMF":
        min_start = np.minimum(self.values[0], other.values[0])
        max_end = np.maximum(self.values[-1], other.values[-1])
        grid = np.arange(min_start, max_end + self.step, self.step)

        offset_self = int(round((self.values[0] - min_start) / self.step))
        offset_other = int(round((other.values[0] - min_start) / self.step))

        pmf_self = np.zeros(len(grid))
        pmf_other = np.zeros(len(grid))
        pmf_self[offset_self : offset_self + len(self.probabilities)] = self.probabilities
        pmf_other[offset_other : offset_other + len(other.probabilities)] = other.probabilities

        cdf_self = np.cumsum(pmf_self, dtype=np.longdouble)
        cdf_other = np.cumsum(pmf_other, dtype=np.longdouble)
        cdf_max = cdf_self * cdf_other
        probs = np.diff(np.concatenate(((0.0,), cdf_max)))

        pmf = DiscretePMF(grid, probs.astype(float), step=self.step)
        expected = self._expected_mass(float(self.total_mass), float(other.total_mass))
        return pmf._rescale(expected)
