from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mc_dagprop.types import ProbabilityMass, Second


@dataclass(frozen=True, slots=True)
class DiscretePMF:
    """Probability mass function on an equidistant integer grid.

    By default PMFs are normalized distributions with total mass one. Duplicate
    support values are aggregated deterministically during construction. Internal
    analytic bound handling may use ``allow_subprobability=True`` for explicit
    REMOVE-policy sub-distributions.
    """

    values: np.ndarray
    probabilities: np.ndarray
    step: int
    allow_subprobability: bool = False

    def __post_init__(self) -> None:
        """Validate and canonicalize the distribution."""
        values = np.asarray(self.values, dtype=float)
        probabilities = np.asarray(self.probabilities, dtype=float)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "probabilities", probabilities)
        self.validate()
        canonical_values, canonical_probabilities = self._canonical_arrays(values, probabilities)
        object.__setattr__(self, "values", canonical_values)
        object.__setattr__(self, "probabilities", canonical_probabilities)
        self.validate()

    @staticmethod
    def _canonical_arrays(values: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_probabilities = probabilities[order]
        unique_values: list[float] = []
        unique_probabilities: list[float] = []
        for value, probability in zip(sorted_values, sorted_probabilities):
            if unique_values and np.isclose(value, unique_values[-1], rtol=0.0, atol=1e-12):
                unique_probabilities[-1] += float(probability)
            else:
                unique_values.append(float(value))
                unique_probabilities.append(float(probability))
        return np.array(unique_values, dtype=float), np.array(unique_probabilities, dtype=float)

    def validate(self) -> None:
        """Validate support, grid alignment, and probability mass."""
        if not isinstance(self.step, int):
            raise TypeError(f"step must be an integer number of seconds, got {self.step!r}")
        if self.step <= 0:
            raise ValueError("step must be positive")
        if self.values.ndim != 1 or self.probabilities.ndim != 1:
            raise ValueError("values and probabilities must be one-dimensional")
        if len(self.values) == 0:
            raise ValueError("PMF values cannot be empty")
        if len(self.values) != len(self.probabilities):
            raise ValueError("values and probabilities must have same length")
        if not np.all(np.isfinite(self.values)):
            raise ValueError("PMF values must be finite")
        if not np.all(np.isfinite(self.probabilities)):
            raise ValueError("PMF probabilities must be finite")
        if np.any(self.probabilities < 0.0):
            raise ValueError("PMF probabilities must be non-negative")
        if not np.allclose(np.mod(self.values, self.step), 0.0, rtol=0.0, atol=1e-9):
            raise ValueError("PMF values are not aligned to step grid")
        total = float(self.probabilities.sum())
        if np.isclose(total, 0.0, rtol=0.0, atol=1e-15):
            if self.allow_subprobability:
                return
            raise ValueError("PMF total probability mass must be positive")
        if self.allow_subprobability:
            if total > 1.0 and not np.isclose(total, 1.0, rtol=1e-12, atol=1e-15):
                raise ValueError("Sub-probability PMF mass must not exceed 1")
        elif not np.isclose(total, 1.0, rtol=1e-12, atol=1e-15):
            raise ValueError(f"PMF probabilities must sum to 1, got {total}")

    def validate_alignment(self, step: Second) -> None:
        """Ensure that ``values`` align with ``step`` spacing."""
        if step <= 0.0:
            raise ValueError("step must be positive")
        if not np.isclose(self.step, step):
            raise ValueError(f"PMF step {self.step} does not match expected {step}")
        if not np.allclose(np.mod(self.values, step), 0.0, rtol=0.0, atol=1e-9):
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
        return DiscretePMF(self.values + delta, self.probabilities.copy(), step=self.step, allow_subprobability=self.allow_subprobability)

    def _rescale(self, expected: float) -> "DiscretePMF":
        probs = self.probabilities.copy()
        total = probs.sum()
        if total > 0 and not np.isclose(total, expected, rtol=1e-12, atol=1e-15):
            probs *= expected / total
        return DiscretePMF(self.values.copy(), probs, step=self.step, allow_subprobability=expected < 1.0)

    @staticmethod
    def _expected_mass(m1: float, m2: float) -> float:
        if np.isclose(m1, 1.0, rtol=1e-12, atol=1e-15) and np.isclose(m2, 1.0, rtol=1e-12, atol=1e-15):
            return 1.0
        return m1 * m2

    def convolve(self, other: "DiscretePMF") -> "DiscretePMF":
        """Convolve two PMFs using stable arithmetic and mass correction."""
        if len(self.values) == 1:
            pmf = DiscretePMF(other.values + self.values[0], other.probabilities * self.probabilities[0], step=self.step, allow_subprobability=True)
        elif len(other.values) == 1:
            pmf = DiscretePMF(self.values + other.values[0], self.probabilities * other.probabilities[0], step=self.step, allow_subprobability=True)
        else:
            start = self.values[0] + other.values[0]
            probs = np.convolve(self.probabilities.astype(np.longdouble), other.probabilities.astype(np.longdouble)).astype(float)
            values = start + self.step * np.arange(len(probs))
            pmf = DiscretePMF(values, probs, step=self.step, allow_subprobability=True)
        return pmf._rescale(self._expected_mass(float(self.total_mass), float(other.total_mass)))

    def maximum(self, other: "DiscretePMF") -> "DiscretePMF":
        """Return PMF of ``max(X, Y)`` with stable cumulative arithmetic."""
        min_start = np.minimum(self.values[0], other.values[0])
        max_end = np.maximum(self.values[-1], other.values[-1])
        grid = np.arange(min_start, max_end + self.step, self.step)
        pmf_self = np.zeros(len(grid), dtype=np.longdouble)
        pmf_other = np.zeros(len(grid), dtype=np.longdouble)
        off_self = int(round((self.values[0] - min_start) / self.step))
        off_other = int(round((other.values[0] - min_start) / self.step))
        pmf_self[off_self : off_self + len(self.probabilities)] = self.probabilities
        pmf_other[off_other : off_other + len(other.probabilities)] = other.probabilities
        cdf_self = np.cumsum(pmf_self, dtype=np.longdouble)
        cdf_other = np.cumsum(pmf_other, dtype=np.longdouble)
        cdf_self_prev = np.concatenate((np.array([0.0], dtype=np.longdouble), cdf_self[:-1]))
        probs = pmf_self * cdf_other + pmf_other * cdf_self_prev
        pmf = DiscretePMF(grid, probs.astype(float), step=self.step, allow_subprobability=True)
        return pmf._rescale(self._expected_mass(float(self.total_mass), float(other.total_mass)))
