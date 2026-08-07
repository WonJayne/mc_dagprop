from __future__ import annotations

import math
import sys
from collections.abc import Callable

import numpy as np
import pytest

from mc_dagprop import Event, EventTimestamp
from mc_dagprop.analytic import (
    AnalyticActivity,
    AnalyticContext,
    DiscretePMF,
    OverflowRule,
    UnderflowRule,
    constant_pmf,
    create_analytic_propagator,
    empirical_pmf,
    exponential_pmf,
    gamma_pmf,
)
from mc_dagprop.analytic.distributions import _regularized_gamma_pair


@pytest.mark.parametrize(
    ("operation", "expected_verb"),
    [
        (lambda left, right: left.convolve(right), "convolve"),
        (lambda left, right: left.maximum(right), "maximum"),
        (lambda left, right: left.conditional_convolve_sum_le(right, 10.0), "conditionally convolve"),
    ],
)
def test_binary_pmf_operations_reject_different_grid_steps(
    operation: Callable[[DiscretePMF, DiscretePMF], DiscretePMF], expected_verb: str
) -> None:
    step_two = DiscretePMF(np.array([0.0, 2.0]), np.array([0.5, 0.5]), step=2)
    step_three = DiscretePMF(np.array([0.0, 3.0]), np.array([0.5, 0.5]), step=3)

    with pytest.raises(ValueError, match=expected_verb):
        operation(step_two, step_three)


def test_pmf_support_arithmetic_rejects_finite_input_overflow() -> None:
    largest_float = sys.float_info.max
    pmf = DiscretePMF.delta(largest_float, step=1)

    with pytest.raises(OverflowError, match="support overflow"):
        pmf.shift(largest_float)
    with pytest.raises(OverflowError, match="support overflow"):
        pmf.convolve(pmf)


@pytest.mark.parametrize(
    ("left_probabilities", "right_probabilities", "expected_probabilities"),
    [([0.5, 0.5], [0.5, 0.5], [0.25, 0.75]), ([0.25, 0.25], [0.2, 0.2], [0.05, 0.15])],
)
def test_sparse_maximum_scales_with_support_size_and_preserves_mass(
    left_probabilities: list[float], right_probabilities: list[float], expected_probabilities: list[float]
) -> None:
    support = np.array([0.0, 1_000_000.0])
    left = DiscretePMF(
        support, np.array(left_probabilities), step=1, allow_subprobability=sum(left_probabilities) < 1.0
    )
    right = DiscretePMF(
        support, np.array(right_probabilities), step=1, allow_subprobability=sum(right_probabilities) < 1.0
    )

    maximum = left.maximum(right)

    np.testing.assert_array_equal(maximum.values, support)
    np.testing.assert_allclose(maximum.probabilities, expected_probabilities, rtol=0.0, atol=1.0e-15)


def test_truncation_at_large_timestamp_inserts_the_exact_bound_bin() -> None:
    lower_bound = 1_000_000_000.0
    activity_pmf = DiscretePMF(np.array([lower_bound - 1.0, lower_bound + 1.0]), np.array([0.5, 0.5]), step=1)
    context = AnalyticContext(
        events=(
            Event("root", EventTimestamp(0.0, 0.0, 0.0)),
            Event("target", EventTimestamp(lower_bound, lower_bound + 10.0, lower_bound)),
        ),
        activities={(0, 1): (0, AnalyticActivity(0, activity_pmf))},
        precedence_list=((1, ((0, 0),)),),
        step=1,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.TRUNCATE,
    )

    result = create_analytic_propagator(context).run()[1]

    np.testing.assert_array_equal(result.pmf.values, [lower_bound, lower_bound + 1.0])
    np.testing.assert_array_equal(result.pmf.probabilities, [0.5, 0.5])


def test_exponential_pmf_avoids_large_scale_cancellation_and_retains_partial_bin() -> None:
    pmf = exponential_pmf(scale=1.0e300, step=1, start=0.0, stop=10.25)

    np.testing.assert_array_equal(pmf.values, np.arange(11, dtype=float))
    expected_widths = np.array([1.0] * 10 + [0.25])
    np.testing.assert_allclose(pmf.probabilities, expected_widths / 10.25, rtol=1.0e-14, atol=0.0)
    assert float(pmf.total_mass) == pytest.approx(1.0)


def test_empirical_pmf_normalizes_large_finite_relative_weights_without_overflow() -> None:
    pmf = empirical_pmf([0.0, 1.0], [1.0e308, 1.0e308], step=1)

    np.testing.assert_array_equal(pmf.values, [0.0, 1.0])
    np.testing.assert_array_equal(pmf.probabilities, [0.5, 0.5])


@pytest.mark.parametrize(
    ("shape", "expected"),
    [(10_000.0, 0.5013298083399552), (1_000_000.0, 0.5001329807608725), (100_000_000.0, 0.5000132980760141)],
)
def test_regularized_gamma_is_stable_at_large_shape_mean(shape: float, expected: float) -> None:
    result = _regularized_gamma_pair(shape, shape)[0]

    assert math.isfinite(result)
    assert result == pytest.approx(expected, abs=2.0e-9)


def test_gamma_pmf_uses_upper_tail_and_partial_final_bin_without_cancellation() -> None:
    pmf = gamma_pmf(shape=2.0, scale=1.0, step=1, start=40.0, stop=42.5)

    def survival(value: float) -> float:
        return math.exp(-value) * (1.0 + value)

    raw_masses = np.array(
        [survival(40.0) - survival(41.0), survival(41.0) - survival(42.0), survival(42.0) - survival(42.5)]
    )
    np.testing.assert_array_equal(pmf.values, [40.0, 41.0, 42.0])
    np.testing.assert_allclose(pmf.probabilities, raw_masses / raw_masses.sum(), rtol=2.0e-13, atol=0.0)
    assert np.all(pmf.probabilities >= 0.0)


def test_gamma_pmf_normalizes_in_a_deep_lower_tail_without_underflow() -> None:
    pmf = gamma_pmf(shape=500.0, scale=1.0, step=1, start=0.0, stop=10.0)

    np.testing.assert_array_equal(pmf.values, np.arange(10, dtype=float))
    assert np.all(np.isfinite(pmf.probabilities))
    assert np.all(pmf.probabilities >= 0.0)
    assert float(pmf.total_mass) == pytest.approx(1.0)
    assert pmf.probabilities[-1] > 1.0 - 1.0e-12


def test_gamma_pmf_with_largest_finite_shape_uses_the_last_interior_cutoff_bin() -> None:
    pmf = gamma_pmf(shape=sys.float_info.max, scale=1.0, step=1, start=0.0, stop=8.0)

    np.testing.assert_array_equal(pmf.values, np.arange(8, dtype=float))
    np.testing.assert_array_equal(pmf.probabilities[:-1], np.zeros(7))
    assert pmf.probabilities[-1] == 1.0
    assert float(pmf.total_mass) == 1.0


def test_gamma_pmf_with_underflowed_scaled_cutoff_retains_its_only_grid_bin() -> None:
    pmf = gamma_pmf(shape=2.0, scale=sys.float_info.max, step=1, start=0.0, stop=sys.float_info.min)

    np.testing.assert_array_equal(pmf.values, [0.0])
    np.testing.assert_array_equal(pmf.probabilities, [1.0])


def test_gamma_pmf_deep_tail_handles_a_lower_edge_negligible_relative_to_its_cutoff() -> None:
    pmf = gamma_pmf(shape=100.0, scale=1.0, step=1, start=sys.float_info.min, stop=0.5)

    np.testing.assert_array_equal(pmf.values, [0.0])
    np.testing.assert_array_equal(pmf.probabilities, [1.0])


def test_gamma_pmf_normalizes_in_a_deep_upper_tail_without_cancellation() -> None:
    pmf = gamma_pmf(shape=2.0, scale=1.0, step=1, start=1_000.0, stop=1_003.0)

    np.testing.assert_array_equal(pmf.values, [1_000.0, 1_001.0, 1_002.0])
    assert np.all(np.isfinite(pmf.probabilities))
    assert np.all(pmf.probabilities > 0.0)
    assert float(pmf.total_mass) == pytest.approx(1.0)


def test_large_shape_gamma_narrow_bins_do_not_oscillate_from_cancellation() -> None:
    shape = 1.0e12
    pmf = gamma_pmf(shape=shape, scale=1.0, step=1, start=shape, stop=shape + 10.0)

    assert np.ptp(pmf.probabilities) < 1.0e-8
    np.testing.assert_allclose(pmf.probabilities, np.full(10, 0.1), rtol=0.0, atol=1.0e-8)


def test_large_shape_gamma_deep_upper_tail_uses_log_survival_probabilities() -> None:
    shape = 1.0e8
    pmf = gamma_pmf(shape=shape, scale=1.0, step=1, start=2.0 * shape, stop=2.0 * shape + 3.0)

    assert np.all(np.isfinite(pmf.probabilities))
    assert np.all(pmf.probabilities > 0.0)
    assert float(pmf.total_mass) == pytest.approx(1.0)
    np.testing.assert_allclose(pmf.probabilities[1:] / pmf.probabilities[:-1], math.exp(-0.5), rtol=2.0e-6, atol=0.0)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [
        (exponential_pmf, {"scale": math.inf, "step": 1, "start": 0.0, "stop": 2.0}),
        (exponential_pmf, {"scale": 1.0, "step": 1, "start": 0.0, "stop": math.inf}),
        (gamma_pmf, {"shape": math.inf, "scale": 1.0, "step": 1, "start": 0.0, "stop": 2.0}),
        (gamma_pmf, {"shape": 2.0, "scale": math.nan, "step": 1, "start": 0.0, "stop": 2.0}),
    ],
)
def test_continuous_pmf_helpers_reject_non_finite_inputs(
    factory: Callable[..., DiscretePMF], kwargs: dict[str, float | int]
) -> None:
    with pytest.raises(ValueError, match="finite"):
        factory(**kwargs)


def test_empirical_delay_pmf_rejects_negative_support_and_boolean_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        empirical_pmf([-1.0, 0.0], [0.5, 0.5], step=1)
    with pytest.raises(TypeError, match="bool"):
        empirical_pmf([False], [1.0], step=1)
    with pytest.raises(TypeError, match="bool"):
        empirical_pmf([0.0], [True], step=1)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: constant_pmf(True, step=1),
        lambda: exponential_pmf(True, step=1, start=0.0, stop=1.0),
        lambda: exponential_pmf(1.0, step=1, start=False, stop=1.0),
        lambda: gamma_pmf(True, 1.0, step=1, start=0.0, stop=1.0),
        lambda: gamma_pmf(1.0, False, step=1, start=0.0, stop=1.0),
        lambda: DiscretePMF.delta(True, step=1),
        lambda: DiscretePMF.delta(0.0, step=1).shift(False),
        lambda: DiscretePMF.delta(0.0, step=1).conditional_convolve_sum_le(DiscretePMF.delta(0.0, step=1), True),
    ],
)
def test_low_level_analytic_numeric_inputs_reject_booleans(factory: Callable[[], object]) -> None:
    with pytest.raises(TypeError, match="bool"):
        factory()


def test_analytic_activity_rejects_a_non_unit_subprobability_pmf() -> None:
    events = (Event("source", EventTimestamp(0.0, 10.0, 0.0)), Event("target", EventTimestamp(0.0, 10.0, 0.0)))
    activity = AnalyticActivity(
        0, DiscretePMF(np.array([1.0]), np.array([0.999999]), step=1, allow_subprobability=True)
    )
    context = AnalyticContext(
        events, {(0, 1): (0, activity)}, ((1, ((0, 0),)),), 1, UnderflowRule.TRUNCATE, OverflowRule.TRUNCATE
    )

    with pytest.raises(ValueError, match="does not sum to 1"):
        create_analytic_propagator(context)
