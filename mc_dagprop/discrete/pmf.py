from __future__ import annotations

from dataclasses import dataclass, field
from typing import NewType

import numpy as np

# Some comments on the code, things to improve or change:

# Define a custom type for values expressed in seconds.  ``NewType`` keeps the
# runtime representation as ``float`` but allows mypy or other type checkers to
# distinguish it from plain floats.
Second = NewType("Second", float)

# Likewise for probability values.
Probability = NewType("Probability", float)

# Please use from __future__ import annotations to ensure that the type hints are better readable


@dataclass(frozen=True, slots=True)
class DiscretePMF:
    """Simple probability mass function on an equidistant grid."""

    values: np.ndarray
    probs: np.ndarray
    step_size: Second = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.values) != len(self.probs):
            raise ValueError("values and probs must have same length")
        if len(self.values) < 2:
            step = 0.0
        else:
            diffs = np.diff(self.values)
            step = float(diffs[0]) if np.allclose(diffs, diffs[0]) else 0.0
        object.__setattr__(self, "step_size", Second(step))
        self.validate()

    def validate(self) -> None:
        if len(self.values) != len(self.probs):
            raise ValueError("values and probs must have same length")
        if len(self.values) > 1 and not np.all(self.values[1:] >= self.values[:-1]):
            raise ValueError("values must be sorted in non-decreasing order")
        if not np.isclose(self.probs.sum(), 1.0):
            raise ValueError("probabilities must sum to 1.0")

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

    # Step should be a static property, defined on init. Again, something that we can validate in a separate method.
    @property
    def step(self) -> Second:
        return self.step_size

    @staticmethod
    def delta(v: Second) -> "DiscretePMF":
        return DiscretePMF(np.array([v], dtype=float), np.array([1.0], dtype=float))

    def shift(self, delta: Second) -> "DiscretePMF":
        return DiscretePMF(self.values + delta, self.probs.copy())

    def convolve(self, other: "DiscretePMF") -> "DiscretePMF":
        if len(self.values) == 1 and len(other.values) == 1:
            return DiscretePMF.delta(self.values[0] + other.values[0])

        step = self.step or other.step
        if step and np.isclose(step, other.step) and np.isclose(step, self.step):
            start = self.values[0] + other.values[0]
            probs = np.convolve(self.probs, other.probs)
            values = start + step * np.arange(len(probs))
            return DiscretePMF(values, probs)

        # fallback: pairwise addition
        vals = self.values[:, None] + other.values[None, :]
        probs = self.probs[:, None] * other.probs[None, :]
        return DiscretePMF(vals.ravel(), probs.ravel())

    def cdf(self) -> np.ndarray:
        return np.cumsum(self.probs)

    def pmf_from_cdf(self, cdf: np.ndarray) -> "DiscretePMF":
        probs = np.diff(np.concatenate([[0.0], cdf]))
        return DiscretePMF(self.values.copy(), probs)

    def maximum(self, other: "DiscretePMF") -> "DiscretePMF":
        # compute via CDFs: F_max(x) = F1(x) * F2(x)
        all_vals = np.union1d(self.values, other.values)
        cdf1 = np.interp(all_vals, self.values, self.cdf(), left=0.0, right=1.0)
        cdf2 = np.interp(all_vals, other.values, other.cdf(), left=0.0, right=1.0)
        cdf_max = cdf1 * cdf2
        return DiscretePMF(all_vals, np.diff(np.concatenate([[0.0], cdf_max])))

    def minimum(self, other: "DiscretePMF") -> "DiscretePMF":
        # F_min(x) = 1 - (1-F1(x))*(1-F2(x))
        all_vals = np.union1d(self.values, other.values)
        cdf1 = np.interp(all_vals, self.values, self.cdf(), left=0.0, right=1.0)
        cdf2 = np.interp(all_vals, other.values, other.cdf(), left=0.0, right=1.0)
        cdf_min = 1.0 - (1.0 - cdf1) * (1.0 - cdf2)
        return DiscretePMF(all_vals, np.diff(np.concatenate([[0.0], cdf_min])))

    def truncate_right(self, max_value: Second) -> "DiscretePMF":
        if max_value >= self.values[-1]:
            return self
        if max_value <= self.values[0]:
            pmf = DiscretePMF(np.array([max_value], dtype=float), np.array([1.0], dtype=float))
            pmf.validate()
            return pmf
        idx = np.searchsorted(self.values, max_value, side="right") - 1
        new_vals = self.values[: idx + 1]
        new_probs = self.probs[: idx + 1]
        overflow = self.probs[idx + 1 :].sum()
        new_probs[-1] += overflow
        new_probs = new_probs / new_probs.sum()
        pmf = DiscretePMF(new_vals, new_probs)
        pmf.validate()
        return pmf

    def truncate(self, min_value: Second, max_value: Second) -> tuple["DiscretePMF", Probability, Probability]:
        """Truncate the PMF to ``[min_value, max_value]`` and return under/overflow mass."""
        under = Probability(self.probs[self.values < min_value].sum())
        over = Probability(self.probs[self.values > max_value].sum())
        mask = (self.values >= min_value) & (self.values <= max_value)
        new_vals = self.values[mask]
        new_probs = self.probs[mask]
        if new_vals.size == 0:
            new_vals = np.array([min_value], dtype=float)
            new_probs = np.array([0.0], dtype=float)
        new_probs = new_probs / new_probs.sum() if new_probs.sum() > 0.0 else new_probs
        pmf = DiscretePMF(new_vals, new_probs)
        pmf.validate()
        return pmf, under, over
