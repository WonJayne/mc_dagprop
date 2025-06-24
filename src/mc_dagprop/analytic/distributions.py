from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from mc_dagprop.types import Second

from ._pmf import DiscretePMF


def constant_pmf(value: Second, step: Second) -> DiscretePMF:
    """Return a deterministic distribution with all mass at ``value``."""
    pmf = DiscretePMF.delta(float(value), float(step))
    pmf.validate()
    pmf.validate_alignment(step)
    return pmf


def _regularized_gamma(shape: float, x: float) -> float:
    """Return the regularized lower incomplete gamma function P(shape, x)."""
    if shape <= 0.0 or x < 0.0:
        raise ValueError("shape must be > 0 and x >= 0")

    eps = 1e-12
    max_iter = 200

    gln = math.lgamma(shape)

    if x == 0.0:
        return 0.0

    if x < shape + 1.0:
        # Series expansion
        ap = shape
        summand = 1.0 / shape
        result = summand
        for _ in range(max_iter):
            ap += 1.0
            summand *= x / ap
            result += summand
            if abs(summand) < abs(result) * eps:
                break
        return result * math.exp(-x + shape * math.log(x) - gln)

    # Continued fraction
    b = x + 1.0 - shape
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, max_iter + 1):
        an = -i * (i - shape)
        b += 2.0
        d = an * d + b
        if abs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return 1.0 - math.exp(-x + shape * math.log(x) - gln) * h


def _build_probs_from_cdf(cdf_values: np.ndarray, start: Second, step: Second, stop: Second) -> np.ndarray:
    """Return normalised probability masses from CDF samples."""
    diffs = np.diff(cdf_values)
    total = diffs.sum()
    if total == 0.0:
        raise ValueError("zero probability mass in range")
    return diffs / total


def exponential_pmf(scale: Second, step: Second, start: Second, stop: Second) -> DiscretePMF:
    """Return a discretised exponential distribution.

    Parameters
    ----------
    scale:
        Mean of the exponential distribution.
    step:
        Grid spacing for the resulting PMF.
    start, stop:
        Range over which to keep probability mass.
    """
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if step <= 0.0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be greater or equal to start")

    edges = np.arange(float(start), float(stop) + float(step), float(step))
    cdf = 1.0 - np.exp(-edges / float(scale))
    probs = _build_probs_from_cdf(cdf, start, step, stop)
    values = edges[:-1]
    pmf = DiscretePMF(values, probs, step=float(step))
    pmf.validate()
    pmf.validate_alignment(step)
    return pmf


def gamma_pmf(shape: float, scale: Second, step: Second, start: Second, stop: Second) -> DiscretePMF:
    """Return a discretised gamma distribution."""
    if shape <= 0.0 or scale <= 0.0:
        raise ValueError("shape and scale must be positive")
    if step <= 0.0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be greater or equal to start")

    edges = np.arange(float(start), float(stop) + float(step), float(step))
    cdf_vals = np.array([_regularized_gamma(shape, edge / float(scale)) for edge in edges])
    probs = _build_probs_from_cdf(cdf_vals, start, step, stop)
    values = edges[:-1]
    pmf = DiscretePMF(values, probs, step=float(step))
    pmf.validate()
    pmf.validate_alignment(step)
    return pmf


def empirical_pmf(values: Iterable[Second], weights: Iterable[float], step: Second) -> DiscretePMF:
    """Return a PMF defined by ``values`` and ``weights``."""
    arr_values = np.array(list(values), dtype=float)
    arr_weights = np.array(list(weights), dtype=float)
    if arr_values.size != arr_weights.size:
        raise ValueError("values and weights must have same length")
    if arr_weights.sum() <= 0.0:
        raise ValueError("weights must sum to a positive number")
    probs = arr_weights / arr_weights.sum()
    pmf = DiscretePMF(arr_values, probs, step=float(step))
    pmf.validate()
    pmf.validate_alignment(step)
    return pmf
