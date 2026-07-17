from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from mc_dagprop import (
    Activity,
    AnalyticContext,
    AnalyticPropagator,
    DelayFamilyRegistry,
    Event,
    EventTimestamp,
    MonteCarloPropagator,
    OverflowRule,
    PropagationContext,
    UnderflowRule,
    create_analytic_propagator,
)
from mc_dagprop.analytic import AnalyticActivity
from mc_dagprop.analytic._pmf import DiscretePMF


def _context(activity_type: int = 1, latest: float = 1_000.0) -> PropagationContext:
    return PropagationContext(
        events=(
            Event("a", EventTimestamp(0.0, latest, 0.0)),
            Event("b", EventTimestamp(0.0, latest, 0.0)),
        ),
        activities={(0, 1): Activity(0, 10.0, activity_type)},
        precedence_list=((1, ((0, 0),)),),
    )


def _clip_result(pmf: DiscretePMF, underflow_rule: UnderflowRule, overflow_rule: OverflowRule):
    ctx = AnalyticContext(
        events=(Event("a", EventTimestamp(0, 100, 0)), Event("b", EventTimestamp(0, 10, 0))),
        activities={(0, 1): (0, AnalyticActivity(0, pmf))},
        precedence_list=((1, ((0, 0),)),),
        step=1,
        underflow_rule=underflow_rule,
        overflow_rule=overflow_rule,
    )
    return create_analytic_propagator(ctx).run()[1]


@pytest.mark.parametrize(
    "first,first_args,second,second_args",
    [
        ("add_constant", {"factor": 0.0}, "add_empirical", {"values": [0], "weights": [1]}),
        ("add_empirical", {"values": [0], "weights": [1]}, "add_gamma", {"shape": 2.0, "scale": 1.0}),
        ("add_gamma", {"shape": 2.0, "scale": 1.0}, "add_exponential", {"scale": 1.0, "max_scale": 2.0}),
        ("add_exponential", {"lambda_": 1.0, "max_scale": 2.0}, "add_constant", {"factor": 0.0}),
    ],
)
def test_duplicate_registration_across_family_types_and_aliases(first: str, first_args: dict[str, object], second: str, second_args: dict[str, object]) -> None:
    registry = DelayFamilyRegistry()
    if "lambda_" in first_args:
        with pytest.warns(DeprecationWarning, match="lambda_"):
            getattr(registry, first)(activity_type=5, **first_args)
    else:
        getattr(registry, first)(activity_type=5, **first_args)
    with pytest.raises(ValueError, match="activity type 5"):
        getattr(registry, second)(activity_type=5, **second_args)


def test_activity_type_minus_one_is_reserved_for_user_registration() -> None:
    registry = DelayFamilyRegistry()
    with pytest.raises(ValueError, match="activity type -1"):
        registry.add_constant(activity_type=-1, factor=0.0)


def test_unregistered_minus_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="activity type -1"):
        _context(activity_type=-1)


def test_frontend_exponential_lambda_alias_warns() -> None:
    registry = DelayFamilyRegistry()
    with pytest.warns(DeprecationWarning, match="lambda_"):
        registry.add_exponential(activity_type=1, lambda_=1.0, max_scale=2.0)


def test_package_root_imports_work_in_subprocess() -> None:
    code = "from mc_dagprop import PropagationContext, DelayFamilyRegistry, MonteCarloPropagator, AnalyticPropagator; print('ok')"
    completed = subprocess.run([sys.executable, "-c", code], check=True, text=True, capture_output=True)
    assert completed.stdout.strip() == "ok"


def test_pmf_shift_supports_positive_and_negative_offsets() -> None:
    pmf = DiscretePMF(np.array([0.0, 2.0]), np.array([0.25, 0.75]), step=1)
    np.testing.assert_allclose(pmf.shift(10).values, [10.0, 12.0])
    np.testing.assert_allclose(pmf.shift(-1).values, [-1.0, 1.0])


def test_pmf_convolution_exact_dirac_shift_and_mass() -> None:
    result = DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5]), step=1).convolve(
        DiscretePMF.delta(10.0, step=1)
    )
    np.testing.assert_allclose(result.values, [10.0, 11.0])
    np.testing.assert_allclose(result.probabilities, [0.5, 0.5])
    assert result.total_mass == pytest.approx(1.0)


def test_pmf_maximum_and_dirac_lower_bound_excess() -> None:
    result = DiscretePMF(np.array([0.0, 2.0]), np.array([0.5, 0.5]), step=1).maximum(
        DiscretePMF(np.array([1.0, 2.0]), np.array([0.5, 0.5]), step=1)
    )
    np.testing.assert_allclose(result.values, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(result.probabilities, [0.0, 0.25, 0.75])

    bounded = DiscretePMF(np.array([0.0, 3.0]), np.array([0.25, 0.75]), step=1).maximum(DiscretePMF.delta(2.0, step=1))
    np.testing.assert_allclose(bounded.values, [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(bounded.probabilities, [0.0, 0.0, 0.25, 0.75])


def test_conditional_convolution_sum_le_matches_direct_enumeration() -> None:
    left = DiscretePMF(np.array([0.0, 1.0]), np.array([0.25, 0.75]), step=1)
    right = DiscretePMF(np.array([0.0, 2.0]), np.array([0.5, 0.5]), step=1)
    result = left.conditional_convolve_sum_le(right, threshold=2.0)
    np.testing.assert_allclose(result.values, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(result.probabilities, [0.125 / 0.625, 0.375 / 0.625, 0.125 / 0.625])


def test_truncate_remove_and_redistribute_clipping_mass_semantics() -> None:
    pmf = DiscretePMF(np.array([-1.0, 5.0, 12.0]), np.array([0.2, 0.5, 0.3]), step=1)

    truncated = _clip_result(pmf, UnderflowRule.TRUNCATE, OverflowRule.TRUNCATE)
    np.testing.assert_allclose(truncated.pmf.values, [0.0, 5.0, 10.0])
    np.testing.assert_allclose(truncated.pmf.probabilities, [0.2, 0.5, 0.3])
    assert truncated.pmf.total_mass == pytest.approx(1.0)
    assert truncated.underflow == pytest.approx(0.0)
    assert truncated.overflow == pytest.approx(0.0)

    removed = _clip_result(pmf, UnderflowRule.REMOVE, OverflowRule.REMOVE)
    np.testing.assert_allclose(removed.pmf.values, [5.0])
    np.testing.assert_allclose(removed.pmf.probabilities, [0.5])
    assert removed.pmf.total_mass == pytest.approx(0.5)
    assert removed.underflow == pytest.approx(0.2)
    assert removed.overflow == pytest.approx(0.3)

    redistributed = _clip_result(pmf, UnderflowRule.REDISTRIBUTE, OverflowRule.REDISTRIBUTE)
    np.testing.assert_allclose(redistributed.pmf.values, [5.0])
    np.testing.assert_allclose(redistributed.pmf.probabilities, [1.0])
    assert redistributed.underflow == pytest.approx(0.0)
    assert redistributed.overflow == pytest.approx(0.0)


def test_empty_support_after_remove_raises_clear_error() -> None:
    pmf = DiscretePMF(np.array([-2.0, 12.0]), np.array([0.5, 0.5]), step=1)
    with pytest.raises(ValueError, match="PMF must not be empty after clipping"):
        _clip_result(pmf, UnderflowRule.REMOVE, OverflowRule.REMOVE)


def test_normal_frontend_propagation_is_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    registry = DelayFamilyRegistry()
    context = _context(activity_type=99)
    MonteCarloPropagator.from_context(context, registry).run(seed=1)
    AnalyticPropagator.from_context(
        context,
        registry,
        step=1,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.TRUNCATE,
    ).run()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_seeded_monte_carlo_same_seed_reproducible_and_different_seed_differs() -> None:
    registry = DelayFamilyRegistry()
    registry.add_empirical(activity_type=1, values=[0.0, 10.0], weights=[0.5, 0.5])
    simulator = MonteCarloPropagator.from_context(_context(), registry)
    assert simulator.run(seed=12).durations[0] == simulator.run(seed=12).durations[0]
    assert {simulator.run(seed=i).durations[0] for i in range(20)} == {10.0, 20.0}
