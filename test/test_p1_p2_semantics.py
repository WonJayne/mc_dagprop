from __future__ import annotations

import numpy as np
import pytest

from mc_dagprop import (
    Activity,
    AnalyticPropagator,
    DelayFamilyRegistry,
    Event,
    EventTimestamp,
    MonteCarloPropagator,
    OverflowRule,
    PropagationContext,
    UnderflowRule,
)
from mc_dagprop.analytic import AnalyticActivity, AnalyticContext, create_analytic_propagator
from mc_dagprop.analytic._pmf import DiscretePMF


def _context(activity_type: int = 1, duration: float = 60.0) -> PropagationContext:
    return PropagationContext(
        events=(Event("a", EventTimestamp(0.0, 1000.0, 0.0)), Event("b", EventTimestamp(0.0, 1000.0, 0.0))),
        activities={(0, 1): Activity(0, duration, activity_type)},
        precedence_list=((1, ((0, 0),)),),
    )


def test_frontend_shifts_extra_delay_to_analytic_increment_pmf() -> None:
    registry = DelayFamilyRegistry()
    registry.add_empirical(activity_type=1, values=[0, 10], weights=[0.5, 0.5])
    propagator = AnalyticPropagator.from_context(
        _context(), registry, step=1, underflow_rule=UnderflowRule.TRUNCATE, overflow_rule=OverflowRule.TRUNCATE
    )
    increment = propagator.context.activities[(0, 1)][1].pmf
    np.testing.assert_allclose(increment.values, [60.0, 70.0])
    np.testing.assert_allclose(increment.probabilities, [0.5, 0.5])


def test_frontend_monte_carlo_uses_base_duration_plus_extra_delay() -> None:
    registry = DelayFamilyRegistry()
    registry.add_empirical(activity_type=1, values=[0, 10], weights=[0.5, 0.5])
    simulator = MonteCarloPropagator.from_context(_context(), registry)
    samples = np.array([simulator.run(seed).durations[0] for seed in range(200)])
    assert set(np.unique(samples)).issubset({60.0, 70.0})
    assert 62.0 < samples.mean() < 68.0


def test_unregistered_activity_type_is_deterministic_in_both_frontends() -> None:
    registry = DelayFamilyRegistry()
    context = _context(activity_type=99, duration=7.0)
    analytic = AnalyticPropagator.from_context(
        context, registry, step=1, underflow_rule=UnderflowRule.TRUNCATE, overflow_rule=OverflowRule.TRUNCATE
    )
    np.testing.assert_allclose(analytic.context.activities[(0, 1)][1].pmf.values, [7.0])
    assert MonteCarloPropagator.from_context(context, registry).run(1).durations[0] == 7.0


@pytest.mark.parametrize(
    "method,args",
    [
        ("add_empirical", {"values": [0], "weights": [1]}),
        ("add_constant", {"factor": 0.1}),
        ("add_exponential", {"scale": 1.0, "max_scale": 2.0}),
        ("add_gamma", {"shape": 2.0, "scale": 1.0, "max_scale": 2.0}),
    ],
)
def test_delay_registry_rejects_duplicate_registration(method: str, args: dict[str, float]) -> None:
    registry = DelayFamilyRegistry()
    getattr(registry, method)(activity_type=3, **args)
    with pytest.raises(ValueError, match="activity type 3"):
        registry.add_constant(activity_type=3, factor=0.0)


@pytest.mark.parametrize(
    "values,probs,step,match",
    [
        ([0], [-1], 1, "non-negative"),
        ([float("nan")], [1], 1, "finite"),
        ([0], [float("inf")], 1, "finite"),
        ([0], [1], 0, "positive"),
        ([0], [1], 1.5, "integer"),
        ([0.5], [1], 1, "aligned"),
        ([0, 1], [0, 0], 1, "positive"),
        ([0, 1], [0.2, 0.2], 1, "sum to 1"),
    ],
)
def test_discrete_pmf_strict_validation(values, probs, step, match: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises((ValueError, TypeError), match=match):
        DiscretePMF(np.array(values, dtype=float), np.array(probs, dtype=float), step=step)


def test_discrete_pmf_aggregates_duplicate_support() -> None:
    pmf = DiscretePMF(np.array([1.0, 0.0, 1.0]), np.array([0.25, 0.5, 0.25]), step=1)
    np.testing.assert_allclose(pmf.values, [0.0, 1.0])
    np.testing.assert_allclose(pmf.probabilities, [0.5, 0.5])


def test_truncate_inserts_and_merges_boundary_bins_without_mass_loss() -> None:
    event = Event("b", EventTimestamp(0.0, 5.0, 0.0))
    ctx = AnalyticContext(
        events=(Event("a", EventTimestamp(-2.0, 100.0, -2.0)), event),
        activities={
            (0, 1): (0, AnalyticActivity(0, DiscretePMF(np.array([0.0, 5.0, 10.0]), np.array([0.2, 0.3, 0.5]), step=1)))
        },
        precedence_list=((1, ((0, 0),)),),
        step=1,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.TRUNCATE,
    )
    result = create_analytic_propagator(ctx).run()[1]
    np.testing.assert_allclose(result.pmf.values, [0.0, 3.0, 5.0])
    np.testing.assert_allclose(result.pmf.probabilities, [0.2, 0.3, 0.5])
    assert result.pmf.total_mass == pytest.approx(1.0)


def test_normal_analytic_propagation_is_quiet(capsys: pytest.CaptureFixture[str]) -> None:
    registry = DelayFamilyRegistry()
    AnalyticPropagator.from_context(
        _context(activity_type=99, duration=1.0),
        registry,
        step=1,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.TRUNCATE,
    ).run()
    assert capsys.readouterr().out == ""


def test_shared_context_validation_rejects_negative_duration_and_cycles() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _context(duration=-1.0)
    with pytest.raises(ValueError, match="cycle"):
        PropagationContext(
            events=(Event("a", EventTimestamp(0, 10, 0)), Event("b", EventTimestamp(0, 10, 0))),
            activities={(0, 1): Activity(0, 1, 1), (1, 0): Activity(1, 1, 1)},
            precedence_list=((1, ((0, 0),)), (0, ((1, 1),))),
        )


def test_seeded_monte_carlo_runs_are_reproducible() -> None:
    registry = DelayFamilyRegistry()
    registry.add_empirical(activity_type=1, values=[0, 10], weights=[0.5, 0.5])
    simulator = MonteCarloPropagator.from_context(_context(), registry)
    assert simulator.run(42).durations[0] == simulator.run(42).durations[0]
    observed = {simulator.run(seed).durations[0] for seed in range(20)}
    assert observed == {60.0, 70.0}
