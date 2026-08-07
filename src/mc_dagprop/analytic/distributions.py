from __future__ import annotations

import math
import sys
from collections.abc import Iterable
from typing import cast

import numpy as np
import numpy.typing as npt

from mc_dagprop.types import Second

from ._pmf import DiscretePMF

type FloatArray = npt.NDArray[np.float64]


def _positive_step(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("step must be an integer number of seconds")
    if value <= 0:
        raise ValueError("step must be positive")
    return value


_GAMMA_ASYMPTOTIC_SHAPE = 10_000_000.0
_GAMMA_EPSILON = 8.0 * sys.float_info.epsilon
_GAMMA_MAX_ITERATIONS = 100_000
_GAMMA_MIN_DENOMINATOR = sys.float_info.min / _GAMMA_EPSILON
_SQRT_TWO = math.sqrt(2.0)


def constant_pmf(value: Second, step: int) -> DiscretePMF:
    """Return a deterministic distribution with all mass at ``value``."""
    if isinstance(value, bool):
        raise TypeError("constant value must be a real number, not bool")
    pmf = DiscretePMF.delta(value, step)
    pmf.validate()
    pmf.validate_alignment(step)
    return pmf


def _clamp_probability(value: float) -> float:
    """Clamp numerical round-off to the probability interval."""
    return min(1.0, max(0.0, value))


def _asymptotic_regularized_gamma_pair(shape: float, x: float) -> tuple[float, float]:
    """Approximate ``P(shape, x)`` and ``Q(shape, x)`` for very large shape."""
    if x == 0.0:
        return 0.0, 1.0
    ratio = x / shape
    if ratio == 0.0:
        centered_cube_root = -1.0
    elif math.isfinite(ratio):
        relative_difference = (x - shape) / shape
        centered_cube_root = -1.0 if relative_difference <= -1.0 else math.expm1(math.log1p(relative_difference) / 3.0)
    else:
        centered_cube_root = math.inf
    z_score = (centered_cube_root + (1.0 / shape) / 9.0) * (3.0 * math.sqrt(shape))
    lower = 0.5 * math.erfc(-z_score / _SQRT_TWO)
    upper = 0.5 * math.erfc(z_score / _SQRT_TWO)
    return _clamp_probability(lower), _clamp_probability(upper)


def _regularized_gamma_pair(shape: float, x: float) -> tuple[float, float]:
    """Return regularized lower and upper incomplete gamma ratios.

    The series and continued-fraction branches retain their stable tail, while
    a Wilson--Hilferty large-shape approximation avoids iteration counts that
    grow with ``sqrt(shape)``. Both tails are returned so callers need not
    subtract two values rounded to one.
    """
    if not math.isfinite(shape) or shape <= 0.0:
        raise ValueError("shape must be finite and positive")
    if math.isnan(x) or x < 0.0:
        raise ValueError("x must be non-negative")
    if x == 0.0:
        return 0.0, 1.0
    if math.isinf(x):
        return 1.0, 0.0
    if shape >= _GAMMA_ASYMPTOTIC_SHAPE:
        return _asymptotic_regularized_gamma_pair(shape, x)

    log_prefactor = -x + shape * math.log(x) - math.lgamma(shape)
    prefactor = math.exp(log_prefactor) if log_prefactor > math.log(sys.float_info.min) else 0.0

    if x < shape + 1.0:
        if shape < 1.0 / sys.float_info.max:
            return 1.0, 0.0
        denominator = shape
        term = 1.0 / shape
        series = term
        for _ in range(_GAMMA_MAX_ITERATIONS):
            denominator += 1.0
            term *= x / denominator
            series += term
            if abs(term) <= abs(series) * _GAMMA_EPSILON:
                lower = _clamp_probability(series * prefactor)
                return lower, _clamp_probability(1.0 - lower)
        raise ArithmeticError("regularized gamma series did not converge")

    denominator = x + 1.0 - shape
    fraction_c = 1.0 / _GAMMA_MIN_DENOMINATOR
    fraction_d = 1.0 / denominator
    fraction = fraction_d
    for iteration in range(1, _GAMMA_MAX_ITERATIONS + 1):
        numerator = -iteration * (iteration - shape)
        denominator += 2.0
        fraction_d = numerator * fraction_d + denominator
        if abs(fraction_d) < _GAMMA_MIN_DENOMINATOR:
            fraction_d = _GAMMA_MIN_DENOMINATOR
        fraction_c = denominator + numerator / fraction_c
        if abs(fraction_c) < _GAMMA_MIN_DENOMINATOR:
            fraction_c = _GAMMA_MIN_DENOMINATOR
        fraction_d = 1.0 / fraction_d
        delta = fraction_d * fraction_c
        fraction *= delta
        if abs(delta - 1.0) <= _GAMMA_EPSILON:
            upper = _clamp_probability(prefactor * fraction)
            return _clamp_probability(1.0 - upper), upper
    raise ArithmeticError("regularized gamma continued fraction did not converge")


def _validate_continuous_grid(step: int, start: Second, stop: Second) -> tuple[float, float, float]:
    """Validate a finite continuous truncation interval and quantization step."""
    validated_step = _positive_step(step)
    try:
        step_value = float(validated_step)
    except OverflowError as exc:
        raise OverflowError("step is too large for finite floating-point time") from exc
    if isinstance(start, bool) or isinstance(stop, bool):
        raise TypeError("distribution range must contain real numbers, not bool")
    start_value = float(start)
    stop_value = float(stop)
    if not math.isfinite(start_value) or not math.isfinite(stop_value):
        raise ValueError("distribution range must be finite")
    if start_value < 0.0:
        raise ValueError("distribution range must be non-negative")
    if stop_value <= start_value:
        raise ValueError("stop must be greater than start")
    return step_value, start_value, stop_value


def _quantized_intervals(step: int, start: Second, stop: Second) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return floor-to-grid values and their clipped continuous intervals."""
    step_value, start_value, stop_value = _validate_continuous_grid(step, start, stop)
    first_index = math.floor(start_value / step_value)
    final_interior = math.nextafter(stop_value, start_value)
    last_index = math.floor(final_interior / step_value)
    bin_count = last_index - first_index + 1
    if bin_count <= 0:
        raise ValueError("distribution range contains no quantization bin")
    if bin_count > np.iinfo(np.intp).max:
        raise OverflowError("distribution grid is too large")
    offsets = np.arange(bin_count, dtype=np.float64)
    values = (float(first_index) + offsets) * step_value
    if not np.all(np.isfinite(values)):
        raise OverflowError("distribution grid exceeds finite floating-point time")
    lower_edges = np.maximum(values, start_value)
    with np.errstate(over="ignore"):
        upper_edges = np.minimum(values + step_value, stop_value)
    if np.any(upper_edges <= lower_edges):
        raise RuntimeError("invalid quantization interval construction")
    return values, lower_edges, upper_edges


def _normalise_masses(masses: Iterable[float]) -> FloatArray:
    """Return finite, non-negative masses normalized with stable summation."""
    mass_array = np.asarray(tuple(masses), dtype=np.float64)
    if mass_array.ndim != 1 or mass_array.size == 0:
        raise ValueError("probability masses cannot be empty")
    if not np.all(np.isfinite(mass_array)):
        raise OverflowError("distribution probability calculation produced a non-finite value")
    if np.any(mass_array < 0.0):
        raise ArithmeticError("distribution probability calculation produced negative mass")
    maximum_mass = float(np.max(mass_array))
    if maximum_mass <= 0.0:
        raise ValueError("requested range has no representable probability mass")
    scaled_masses = mass_array / maximum_mass
    scaled_items = cast(Iterable[np.float64], scaled_masses)
    total = math.fsum(float(mass) for mass in scaled_items)
    probabilities = scaled_masses / total
    probability_items = cast(Iterable[np.float64], probabilities)
    probabilities /= math.fsum(float(probability) for probability in probability_items)
    return probabilities


def exponential_pmf(scale: Second, step: int, start: Second, stop: Second) -> DiscretePMF:
    """Return a floor-quantized, truncated exponential distribution.

    The result represents ``floor(X / step) * step`` conditional on
    ``start <= X <= stop``. A non-grid-aligned ``stop`` therefore contributes a
    partial final bin rather than being rounded outward.
    """
    if isinstance(scale, bool):
        raise TypeError("scale must be a real number, not bool")
    scale_value = float(scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError("scale must be finite and positive")
    values, lower_edges, upper_edges = _quantized_intervals(step, start, stop)
    start_value = float(start)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        relative_lower = (lower_edges - start_value) / scale_value
        widths = (upper_edges - lower_edges) / scale_value
        masses = np.exp(-relative_lower) * -np.expm1(-widths)
    if not np.any(masses > 0.0):
        masses = upper_edges - lower_edges
    probabilities = _normalise_masses(masses)
    return DiscretePMF(values, probabilities, step=step)


def _gamma_interval_probability(shape: float, lower: float, upper: float) -> float:
    """Return gamma probability on ``[lower, upper]`` without tail cancellation."""
    lower_p, lower_q = _regularized_gamma_pair(shape, lower)
    upper_p, upper_q = _regularized_gamma_pair(shape, upper)
    if upper <= shape:
        return max(0.0, upper_p - lower_p)
    if lower >= shape:
        return max(0.0, lower_q - upper_q)
    return max(0.0, upper_p - lower_p)


def _log_regularized_gamma_p_series(shape: float, x: float) -> float:
    """Return ``log(P(shape, x))`` without underflow in the lower tail."""
    if x == 0.0:
        return -math.inf
    log_prefactor = -x + shape * math.log(x) - math.lgamma(shape)
    denominator = shape
    term = 1.0 / shape
    series = term
    for _ in range(_GAMMA_MAX_ITERATIONS):
        denominator += 1.0
        term *= x / denominator
        series += term
        if abs(term) <= abs(series) * _GAMMA_EPSILON:
            result = log_prefactor + math.log(series)
            if not math.isfinite(result):
                raise ArithmeticError("log regularized gamma series produced a non-finite value")
            return min(0.0, result)
    raise ArithmeticError("log regularized gamma series did not converge")


def _log_lower_incomplete_gamma_series_factor(shape: float, x: float) -> float:
    """Return the log series factor in the lower incomplete gamma function."""
    denominator = shape
    term = 1.0
    series = term
    for _ in range(_GAMMA_MAX_ITERATIONS):
        denominator += 1.0
        term *= x / denominator
        series += term
        if abs(term) <= abs(series) * _GAMMA_EPSILON:
            return math.log(series)
    raise ArithmeticError("lower incomplete gamma series factor did not converge")


def _deep_lower_tail_log_masses(shape: float, lower_edges: FloatArray, upper_edges: FloatArray) -> tuple[float, ...]:
    """Return interval log masses relative to one common lower-tail CDF."""
    lower_items = cast(Iterable[np.float64], lower_edges)
    upper_items = cast(tuple[np.float64, ...], tuple(upper_edges))
    reference = float(upper_items[-1])
    reference_log_factor = _log_lower_incomplete_gamma_series_factor(shape, reference)

    def relative_log_cdf(x: float) -> float:
        if x == 0.0:
            return -math.inf
        relative_difference = (x - reference) / reference
        if relative_difference <= -1.0:
            return -math.inf
        log_power_ratio = shape * math.log1p(relative_difference)
        log_factor = _log_lower_incomplete_gamma_series_factor(shape, x)
        result = log_power_ratio - (x - reference) + log_factor - reference_log_factor
        if math.isnan(result) or result == math.inf:
            raise ArithmeticError("relative lower incomplete gamma calculation produced a non-finite value")
        return min(0.0, result)

    return tuple(
        _log_probability_difference(relative_log_cdf(float(upper)), relative_log_cdf(float(lower)))
        for lower, upper in zip(lower_items, upper_items, strict=True)
    )


def _log_regularized_gamma_q_continued_fraction(shape: float, x: float) -> float:
    """Return ``log(Q(shape, x))`` without underflow in the upper tail."""
    log_prefactor = -x + shape * math.log(x) - math.lgamma(shape)
    denominator = x + 1.0 - shape
    fraction_c = 1.0 / _GAMMA_MIN_DENOMINATOR
    fraction_d = 1.0 / denominator
    fraction = fraction_d
    for iteration in range(1, _GAMMA_MAX_ITERATIONS + 1):
        numerator = -iteration * (iteration - shape)
        denominator += 2.0
        fraction_d = numerator * fraction_d + denominator
        if abs(fraction_d) < _GAMMA_MIN_DENOMINATOR:
            fraction_d = _GAMMA_MIN_DENOMINATOR
        fraction_c = denominator + numerator / fraction_c
        if abs(fraction_c) < _GAMMA_MIN_DENOMINATOR:
            fraction_c = _GAMMA_MIN_DENOMINATOR
        fraction_d = 1.0 / fraction_d
        delta = fraction_d * fraction_c
        fraction *= delta
        if abs(delta - 1.0) <= _GAMMA_EPSILON:
            if fraction <= 0.0:
                raise ArithmeticError("log regularized gamma continued fraction became non-positive")
            result = log_prefactor + math.log(fraction)
            if not math.isfinite(result):
                raise ArithmeticError("log regularized gamma continued fraction produced a non-finite value")
            return min(0.0, result)
    raise ArithmeticError("log regularized gamma continued fraction did not converge")


def _log_probability_difference(log_larger: float, log_smaller: float) -> float:
    """Return ``log(exp(log_larger) - exp(log_smaller))`` stably."""
    if log_smaller == -math.inf:
        return log_larger
    log_ratio = log_smaller - log_larger
    if log_ratio >= 0.0:
        if log_ratio <= 8.0 * sys.float_info.epsilon:
            return -math.inf
        raise ArithmeticError("gamma interval endpoints are not monotonic")
    return log_larger + math.log(-math.expm1(log_ratio))


def _gamma_interval_log_probability(shape: float, lower: float, upper: float) -> float:
    """Return the log probability on ``[lower, upper]`` in either tail."""
    probability = _gamma_interval_probability(shape, lower, upper)
    if probability > 0.0:
        return math.log(probability)
    if upper < shape + 1.0:
        return _log_probability_difference(
            _log_regularized_gamma_p_series(shape, upper), _log_regularized_gamma_p_series(shape, lower)
        )
    if lower >= shape + 1.0:
        return _log_probability_difference(
            _log_regularized_gamma_q_continued_fraction(shape, lower),
            _log_regularized_gamma_q_continued_fraction(shape, upper),
        )
    return -math.inf


def _normalise_log_masses(log_masses: Iterable[float]) -> FloatArray:
    """Normalize log-domain masses after removing their common exponent."""
    log_mass_array = np.asarray(tuple(log_masses), dtype=np.float64)
    log_mass_items = cast(Iterable[np.float64], log_mass_array)
    if any(math.isnan(float(mass)) or mass == math.inf for mass in log_mass_items):
        raise ArithmeticError("distribution log probability calculation produced a non-finite value")
    maximum = float(np.max(log_mass_array))
    if maximum == -math.inf:
        raise ValueError("requested range has no representable probability mass")
    return _normalise_masses(np.exp(log_mass_array - maximum))


def gamma_pmf(shape: float, scale: Second, step: int, start: Second, stop: Second) -> DiscretePMF:
    """Return a floor-quantized, truncated gamma distribution.

    The result represents ``floor(X / step) * step`` conditional on
    ``start <= X <= stop`` and retains a partial final quantization bin.
    """
    if isinstance(shape, bool) or isinstance(scale, bool):
        raise TypeError("gamma shape and scale must be real numbers, not bool")
    shape_value = float(shape)
    scale_value = float(scale)
    if not math.isfinite(shape_value) or shape_value <= 0.0:
        raise ValueError("shape must be finite and positive")
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError("scale must be finite and positive")
    values, lower_edges, upper_edges = _quantized_intervals(step, start, stop)
    with np.errstate(over="ignore", invalid="ignore"):
        scaled_lower_edges = lower_edges / scale_value
        scaled_upper_edges = upper_edges / scale_value
    final_scaled_upper = float(cast(np.float64, scaled_upper_edges[-1]))
    if final_scaled_upper == 0.0:
        probabilities = np.zeros_like(values)
        probabilities[-1] = 1.0
        return DiscretePMF(values, probabilities, step=step)
    if final_scaled_upper <= shape_value / 2.0:
        log_masses: Iterable[float] = _deep_lower_tail_log_masses(shape_value, scaled_lower_edges, scaled_upper_edges)
    else:
        lower_items = cast(Iterable[np.float64], scaled_lower_edges)
        upper_items = cast(Iterable[np.float64], scaled_upper_edges)
        log_masses = (
            _gamma_interval_log_probability(shape_value, float(lower), float(upper))
            for lower, upper in zip(lower_items, upper_items, strict=True)
        )
    probabilities = _normalise_log_masses(log_masses)
    return DiscretePMF(values, probabilities, step=step)


def empirical_pmf(values: Iterable[Second], weights: Iterable[float], step: int) -> DiscretePMF:
    """Return a PMF defined by ``values`` and ``weights``."""
    values_tuple = tuple(values)
    weights_tuple = tuple(weights)
    if any(isinstance(value, bool) for value in values_tuple):
        raise TypeError("empirical values must be real numbers, not bool")
    if any(isinstance(weight, bool) for weight in weights_tuple):
        raise TypeError("empirical weights must be real numbers, not bool")
    arr_values = np.array(values_tuple, dtype=float)
    arr_weights = np.array(weights_tuple, dtype=float)
    if arr_values.size != arr_weights.size:
        raise ValueError("values and weights must have same length")
    if np.any(arr_values < 0.0):
        raise ValueError("empirical delays must be non-negative")
    probs = _normalise_masses(arr_weights)
    pmf = DiscretePMF(arr_values, probs, step=step)
    pmf.validate()
    pmf.validate_alignment(step)
    return pmf
