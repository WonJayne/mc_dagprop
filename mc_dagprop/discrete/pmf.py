from __future__ import annotations

from dataclasses import dataclass

import numpy as np
# Some comments on the code, things to improve or change:
# TODO: Define a custum type, Second, that is a float, but has a unit of seconds, which we can use in the future.
# TODO: Define a custom type, Probability, that is a float, but has a unit of probability, which we can use in the future.

# Please use from __future__ import annotations to ensure that the type hints are better readable

@dataclass(frozen=True, slots=True,)
class DiscretePMF:
    """Simple probability mass function on an equidistant grid."""

    values: np.ndarray
    probs: np.ndarray

    def __post_init__(self) -> None:
        if len(self.values) != len(self.probs):
            raise ValueError("values and probs must have same length")
        if not np.isclose(self.probs.sum(), 1.0):
            # FIXME: This is not something we should be doing here, as this is causing behavioral changes.
            # We can raise an error instead
            self.probs = self.probs / self.probs.sum()

        # FIXME Wouldn't it be better to just expect sorted values? I would instead have a validate method,
        #  that checks if the values are sorted. and if the probs are normalized.
        order = np.argsort(self.values)
        self.values = self.values[order]
        self.probs = self.probs[order]
        # merge identical values

        # TODO: SAme as above, this is not something we should be doing here.
        uniq, indices = np.unique(self.values, return_inverse=True)
        if len(uniq) != len(self.values):
            agg = np.zeros_like(uniq, dtype=float)
            np.add.at(agg, indices, self.probs)
            self.values = uniq
            self.probs = agg

    # Step should be a static property, defined on init. Again, someting that we can validate in a separate method.
    @property
    def step(self) -> float:
        if len(self.values) < 2:
            return 0.0
        diffs = np.diff(self.values)
        return float(diffs[0]) if np.allclose(diffs, diffs[0]) else 0.0

    @staticmethod
    def delta(v: float) -> "DiscretePMF":
        return DiscretePMF(np.array([v], dtype=float), np.array([1.0], dtype=float))

    def shift(self, delta: float) -> "DiscretePMF":
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

    def truncate_right(self, max_value: float) -> "DiscretePMF":
        if max_value >= self.values[-1]:
            return self
        if max_value <= self.values[0]:
            return DiscretePMF(np.array([max_value], dtype=float), np.array([1.0], dtype=float))
        idx = np.searchsorted(self.values, max_value, side="right") - 1
        new_vals = self.values[: idx + 1]
        new_probs = self.probs[: idx + 1]
        overflow = self.probs[idx + 1 :].sum()
        new_probs[-1] += overflow
        return DiscretePMF(new_vals, new_probs)

    def truncate(self, min_value: float, max_value: float) -> tuple["DiscretePMF", float, float]:
        """Truncate the PMF to ``[min_value, max_value]`` and return under/overflow mass."""
        under = float(self.probs[self.values < min_value].sum())
        over = float(self.probs[self.values > max_value].sum())
        mask = (self.values >= min_value) & (self.values <= max_value)
        new_vals = self.values[mask]
        new_probs = self.probs[mask]
        if new_vals.size == 0:
            new_vals = np.array([min_value], dtype=float)
            new_probs = np.array([0.0], dtype=float)
        # Here we might need to normalize the probabilities again?
        # FIXME: also, here we should then call assert object.validate() to ensure the PMF is valid,
        #  but we can use -OO to skip this check in production.
        return DiscretePMF(new_vals, new_probs), under, over
