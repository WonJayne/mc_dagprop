from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mc_dagprop.discrete import Second, ProbabilityMass


@dataclass(frozen=True, slots=True)
class DiscretePMF:
    """Simple probability mass function on an equidistant grid."""

    values: np.ndarray
    probabilities: np.ndarray
    # FIXME: This should be an input, passed at the beginning, not calculated on init.
    step: Second

    def __post_init__(self) -> None:
        # FIXME See above: step_size should be an input, not calculated here.
        if len(self.values) != len(self.probabilities):
            raise ValueError("values and probs must have same length")
        if len(self.values) < 2:
            step = 0.0
        else:
            diffs = np.diff(self.values)
            # FIXME: General question why would you set a step size to 0.0?
            step = float(diffs[0]) if np.allclose(diffs, diffs[0]) else 0.0
        # SEE fixme above: step_size should be an input, not calculated here.
        object.__setattr__(self, "step_size", Second(step))
        # Validation should not be done all the time, maybe we actually have a case where it is less than 1 (total mass < 1)
        assert self.step >= 0.0, "step size must be non-negative"
        self.validate()

    # FIXME: This should be a validation method, outside of the __post_init__ method, and ideally a function that can be called separately.
    def validate(self) -> None:
        """Validate the PMF properties."""
        if len(self.values) != len(self.probabilities):
            raise ValueError("values and probs must have same length")
        if len(self.values) > 1 and not np.all(self.values[1:] >= self.values[:-1]):
            raise ValueError("values must be sorted in non-decreasing order")
        if not np.isclose(self.probabilities.sum(), 1.0):
            raise ValueError("probabilities must sum to 1.0")

    # Move this to a validation method, outside of the class.
    def validate_alignment(self, step: Second) -> None:
        """Ensure that ``values`` align with ``step`` spacing."""
        if step <= 0.0:
            raise ValueError("step must be positive")

        if len(self.values) > 1:
            diffs = np.diff(self.values)
            if not np.allclose(diffs, step):
                raise ValueError("PMF grid spacing does not match step")

        if self.values.size > 0 and not np.isclose(self.values[0] % step, 0.0):
            raise ValueError("PMF values are not aligned to step grid")

    @staticmethod
    def delta(v: Second) -> "DiscretePMF":
        """Create a PMF that is a delta function at value `v`, so 100% probability at `v`."""
        return DiscretePMF(np.array([v], dtype=float), np.array([1.0], dtype=float))

    @property
    def total_mass(self) -> ProbabilityMass:
        """Return the total mass of the PMF."""
        return ProbabilityMass(self.probabilities.sum())

    def shift(self, delta: Second) -> "DiscretePMF":
        """Shift the PMF by `delta` seconds."""
        return DiscretePMF(self.values + delta, self.probabilities.copy())

    def convolve(self, other: "DiscretePMF") -> "DiscretePMF":
        """Return the distribution of ``X + Y`` for two independent PMFs."""
        is_delta = len(self.values) == 1
        if is_delta:
            other_is_delta = len(other.values) == 1
            if other_is_delta:
                # Both are delta functions, return a delta function at the sum of the values.
                return DiscretePMF.delta(self.values[0] + other.values[0])

        step = self.step
        assert self.step == other.step, f"PMFs must share a positive step size, got {self.step} and {other.step}"
        if not step or not np.isclose(step, other.step) or not np.isclose(step, self.step):
            # FIXME: This should be a validation method, outside of the convolve method.
            raise ValueError("PMFs must share a positive step size")

        start = self.values[0] + other.values[0]
        probs = np.convolve(self.probabilities, other.probabilities)
        values = start + step * np.arange(len(probs))
        return DiscretePMF(values, probs)

    def maximum(self, other: "DiscretePMF") -> "DiscretePMF":
        """Return ``max(X, Y)`` for two independent PMFs.

        This operation is used by :class:`DiscreteSimulator` to combine delay
        distributions when an event has multiple predecessors.
        """
        if len(self.values) == 1 and len(other.values) == 1:
            return DiscretePMF.delta(max(self.values[0], other.values[0]))

        step = float(self.step)
        # FIXME: This should be a validation method, outside of the maximum method.
        #  Should be validated before starting the propagation.
        if step <= 0.0 or not np.isclose(step, other.step):
            raise ValueError("PMFs must share a positive step size")
        if not np.isclose((self.values[0] - other.values[0]) % step, 0.0):
            raise ValueError("PMF grids are not aligned")

        min_start = min(self.values[0], other.values[0])
        max_end = max(self.values[-1], other.values[-1])
        grid = np.arange(min_start, max_end + step, step)

        offset_self = int(round((self.values[0] - min_start) / step))
        offset_other = int(round((other.values[0] - min_start) / step))

        pmf_self = np.zeros(len(grid))
        pmf_other = np.zeros(len(grid))
        pmf_self[offset_self : offset_self + len(self.probabilities)] = self.probabilities
        pmf_other[offset_other : offset_other + len(other.probabilities)] = other.probabilities

        cdf_self = np.cumsum(pmf_self)
        cdf_other = np.cumsum(pmf_other)
        cdf_max = cdf_self * cdf_other
        probs = np.diff(np.concatenate(([0.0], cdf_max)))
        return DiscretePMF(grid, probs)

