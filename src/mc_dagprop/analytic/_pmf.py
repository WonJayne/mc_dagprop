from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt

from mc_dagprop.types import ProbabilityMass, Second

type FloatArray = npt.NDArray[np.float64]


def _positive_step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"step must be an integer number of seconds, got {value!r}")
    if value <= 0:
        raise ValueError("step must be positive")
    return value


@dataclass(frozen=True, slots=True)
class DiscretePMF:
    """Probability mass function on an equidistant integer grid.

    By default PMFs are normalized distributions with total mass one. Duplicate
    support values are aggregated deterministically during construction. Internal
    analytic bound handling may use ``allow_subprobability=True`` for explicit
    REMOVE-policy sub-distributions.
    """

    values: FloatArray
    probabilities: FloatArray
    step: int
    allow_subprobability: bool = False

    def __post_init__(self) -> None:
        """Validate and canonicalize the distribution."""
        values = np.asarray(self.values, dtype=np.float64)
        probabilities = np.asarray(self.probabilities, dtype=np.float64)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "probabilities", probabilities)
        self.validate()
        canonical_values, canonical_probabilities = self._canonical_arrays(values, probabilities)
        object.__setattr__(self, "values", canonical_values)
        object.__setattr__(self, "probabilities", canonical_probabilities)
        self.validate()
        frozen_values = np.frombuffer(canonical_values.tobytes(), dtype=canonical_values.dtype)
        frozen_probabilities = np.frombuffer(canonical_probabilities.tobytes(), dtype=canonical_probabilities.dtype)
        frozen_values.setflags(write=False)
        frozen_probabilities.setflags(write=False)
        object.__setattr__(self, "values", frozen_values)
        object.__setattr__(self, "probabilities", frozen_probabilities)

    @staticmethod
    def _canonical_arrays(values: FloatArray, probabilities: FloatArray) -> tuple[FloatArray, FloatArray]:
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        sorted_probabilities = probabilities[order]
        unique_values: list[float] = []
        unique_probabilities: list[float] = []
        value_items = cast(Iterable[np.float64], sorted_values)
        probability_items = cast(Iterable[np.float64], sorted_probabilities)
        for value, probability in zip(value_items, probability_items, strict=True):
            if unique_values and np.isclose(value, unique_values[-1], rtol=0.0, atol=1e-12):
                unique_probabilities[-1] += float(probability)
            else:
                unique_values.append(float(value))
                unique_probabilities.append(float(probability))
        return np.array(unique_values, dtype=np.float64), np.array(unique_probabilities, dtype=np.float64)

    def validate(self) -> None:
        """Validate support, grid alignment, and probability mass."""
        step = _positive_step(self.step)
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
        if not np.allclose(np.mod(self.values, step), 0.0, rtol=0.0, atol=1e-9):
            raise ValueError("PMF values are not aligned to step grid")
        total = float(cast(np.float64, self.probabilities.sum()))
        if np.isclose(total, 0.0, rtol=0.0, atol=1e-15):
            if self.allow_subprobability:
                return
            raise ValueError("PMF total probability mass must be positive")
        if self.allow_subprobability:
            if total > 1.0 and not np.isclose(total, 1.0, rtol=1e-12, atol=1e-15):
                raise ValueError("Sub-probability PMF mass must not exceed 1")
        elif not np.isclose(total, 1.0, rtol=1e-12, atol=1e-15):
            raise ValueError(f"PMF probabilities must sum to 1, got {total}")

    def validate_alignment(self, step: int) -> None:
        """Ensure that ``values`` align with ``step`` spacing."""
        validated_step = _positive_step(step)
        if self.step != validated_step:
            raise ValueError(f"PMF step {self.step} does not match expected {validated_step}")
        if not np.allclose(np.mod(self.values, validated_step), 0.0, rtol=0.0, atol=1e-9):
            raise ValueError("PMF values are not aligned to step grid")

    @staticmethod
    def delta(v: Second, step: int) -> DiscretePMF:
        """Return a unit mass at ``v`` using ``step`` spacing."""
        if isinstance(v, bool):
            raise TypeError("PMF value must be a real number, not bool")
        return DiscretePMF(np.array([v], dtype=np.float64), np.array([1.0], dtype=np.float64), step=step)

    @property
    def total_mass(self) -> ProbabilityMass:
        """Return the total mass of the PMF."""
        return float(cast(np.float64, self.probabilities.sum()))

    def shift(self, delta: Second) -> DiscretePMF:
        """Shift the PMF by ``delta`` seconds."""
        if isinstance(delta, bool):
            raise TypeError("PMF shift must be a real number, not bool")
        if not math.isfinite(delta):
            raise ValueError("PMF shift must be finite")
        self._require_finite_support_sum(float(delta), "shift")
        return DiscretePMF(
            self.values + delta,
            self.probabilities.copy(),
            step=self.step,
            allow_subprobability=self.allow_subprobability,
        )

    def _rescale(self, expected: float) -> DiscretePMF:
        probs = self.probabilities.copy()
        total = float(cast(np.float64, probs.sum()))
        if total > 0 and not np.isclose(total, expected, rtol=1e-12, atol=1e-15):
            probs *= expected / total
        return DiscretePMF(self.values.copy(), probs, step=self.step, allow_subprobability=expected < 1.0)

    @staticmethod
    def _expected_mass(m1: float, m2: float) -> float:
        if np.isclose(m1, 1.0, rtol=1e-12, atol=1e-15) and np.isclose(m2, 1.0, rtol=1e-12, atol=1e-15):
            return 1.0
        return m1 * m2

    def _require_compatible_grid(self, other: DiscretePMF, operation: str) -> None:
        """Require another PMF to use this PMF's grid."""
        if self.step != other.step:
            raise ValueError(f"cannot {operation} PMFs with different grid steps: " f"{self.step} and {other.step}")

    def _require_finite_support_sum(self, other_value: float, operation: str) -> None:
        """Reject support arithmetic that exceeds finite floating-point time."""
        lower = float(cast(np.float64, self.values[0])) + other_value
        upper = float(cast(np.float64, self.values[-1])) + other_value
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise OverflowError(f"PMF support overflow during {operation}")

    def convolve(self, other: DiscretePMF) -> DiscretePMF:
        """Convolve two PMFs using stable arithmetic and exact sparse support."""
        self._require_compatible_grid(other, "convolve")
        self._require_finite_support_sum(float(cast(np.float64, other.values[0])), "convolution")
        self._require_finite_support_sum(float(cast(np.float64, other.values[-1])), "convolution")
        self_contiguous = (
            np.allclose(np.diff(self.values), self.step, rtol=0.0, atol=1e-9) if len(self.values) > 1 else True
        )
        other_contiguous = (
            np.allclose(np.diff(other.values), other.step, rtol=0.0, atol=1e-9) if len(other.values) > 1 else True
        )
        if self_contiguous and other_contiguous:
            if len(self.values) == 1:
                pmf = DiscretePMF(
                    other.values + cast(np.float64, self.values[0]),
                    other.probabilities * cast(np.float64, self.probabilities[0]),
                    step=self.step,
                    allow_subprobability=True,
                )
            elif len(other.values) == 1:
                pmf = DiscretePMF(
                    self.values + cast(np.float64, other.values[0]),
                    self.probabilities * cast(np.float64, other.probabilities[0]),
                    step=self.step,
                    allow_subprobability=True,
                )
            else:
                start_value = float(cast(np.float64, self.values[0])) + float(cast(np.float64, other.values[0]))
                probabilities = np.convolve(
                    self.probabilities.astype(np.longdouble), other.probabilities.astype(np.longdouble)
                ).astype(np.float64)
                values = start_value + self.step * np.arange(len(probabilities), dtype=np.float64)
                pmf = DiscretePMF(values, probabilities, step=self.step, allow_subprobability=True)
            return pmf._rescale(self._expected_mass(float(self.total_mass), float(other.total_mass)))

        masses: dict[float, np.longdouble] = {}
        self_values = cast(Iterable[np.float64], self.values)
        self_probabilities = cast(Iterable[np.float64], self.probabilities)
        other_values = cast(Iterable[np.float64], other.values)
        other_probabilities = cast(Iterable[np.float64], other.probabilities)
        for self_value, self_probability in zip(self_values, self_probabilities, strict=True):
            for other_value, other_probability in zip(other_values, other_probabilities, strict=True):
                summed_value = float(self_value + other_value)
                masses[summed_value] = masses.get(summed_value, np.longdouble(0.0)) + (
                    np.longdouble(self_probability) * np.longdouble(other_probability)
                )
        values = np.array(sorted(masses), dtype=np.float64)
        value_items = cast(Iterable[np.float64], values)
        probabilities = np.array([masses[float(value)] for value in value_items], dtype=np.float64)
        pmf = DiscretePMF(values, probabilities, step=self.step, allow_subprobability=True)
        return pmf._rescale(self._expected_mass(float(self.total_mass), float(other.total_mass)))

    def maximum(self, other: DiscretePMF) -> DiscretePMF:
        """Return PMF of ``max(X, Y)`` with stable cumulative arithmetic."""
        self._require_compatible_grid(other, "take the maximum of")
        support = np.union1d(self.values, other.values)

        def cdf_on_support(pmf: DiscretePMF) -> npt.NDArray[np.longdouble]:
            cumulative = np.cumsum(pmf.probabilities.astype(np.longdouble), dtype=np.longdouble)
            indices = np.searchsorted(pmf.values, support, side="right") - 1
            result = np.zeros(len(support), dtype=np.longdouble)
            present = indices >= 0
            result[present] = cumulative[indices[present]]
            return result

        maximum_cdf = cdf_on_support(self) * cdf_on_support(other)
        probabilities = np.diff(np.concatenate((np.array([0.0], dtype=np.longdouble), maximum_cdf)))
        probabilities = np.maximum(probabilities, np.longdouble(0.0))
        pmf = DiscretePMF(support, probabilities.astype(float), step=self.step, allow_subprobability=True)
        return pmf._rescale(self._expected_mass(float(self.total_mass), float(other.total_mass)))

    def conditional_convolve_sum_le(self, other: DiscretePMF, threshold: Second) -> DiscretePMF:
        """Return ``X + Y`` conditioned on ``X + Y <= threshold``.

        This is a low-level primitive for documenting Büker-style conditional
        convolution semantics. It is intentionally not wired into the main
        marginal propagator. The returned PMF is normalized over the accepted
        joint support.
        """
        self._require_compatible_grid(other, "conditionally convolve")
        if isinstance(threshold, bool):
            raise TypeError("conditional convolution threshold must be a real number, not bool")
        if not math.isfinite(threshold):
            raise ValueError("conditional convolution threshold must be finite")
        self._require_finite_support_sum(float(cast(np.float64, other.values[0])), "conditional convolution")
        self._require_finite_support_sum(float(cast(np.float64, other.values[-1])), "conditional convolution")
        values: list[float] = []
        probabilities: list[float] = []
        self_values = cast(Iterable[np.float64], self.values)
        self_probabilities = cast(Iterable[np.float64], self.probabilities)
        other_values = cast(Iterable[np.float64], other.values)
        other_probabilities = cast(Iterable[np.float64], other.probabilities)
        for self_value, self_probability in zip(self_values, self_probabilities, strict=True):
            for other_value, other_probability in zip(other_values, other_probabilities, strict=True):
                summed_value = float(self_value + other_value)
                if summed_value <= threshold:
                    values.append(summed_value)
                    probabilities.append(float(self_probability * other_probability))
        if not values:
            raise ValueError("conditional convolution has empty support")
        probability_array = np.array(probabilities, dtype=float)
        total = float(cast(np.float64, probability_array.sum()))
        if total <= 0.0:
            raise ValueError("conditional convolution has zero accepted probability mass")
        return DiscretePMF(np.array(values, dtype=float), probability_array / total, step=self.step)
