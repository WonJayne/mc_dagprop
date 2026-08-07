from __future__ import annotations

import math

import pytest

import mc_dagprop as mc


def _events(*, latest: float = 100.0) -> tuple[mc.Event, mc.Event]:
    return (
        mc.Event("root", mc.EventTimestamp(0.0, latest, 0.0)),
        mc.Event("target", mc.EventTimestamp(0.0, latest, 0.0)),
    )


def _context(*, duration: float = 10.0, activity_type: int = 1, latest: float = 100.0) -> mc.PropagationContext:
    return mc.PropagationContext(
        _events(latest=latest), {(0, 1): mc.Activity(0, duration, activity_type)}, ((1, ((0, 0),)),)
    )


def test_timestamp_contract_uses_earliest_and_keeps_actual_as_metadata() -> None:
    context = mc.PropagationContext((mc.Event("root", mc.EventTimestamp(2.0, 10.0, 5.0)),), {}, ())
    registry = mc.DelayFamilyRegistry()

    monte_carlo = mc.MonteCarloPropagator.from_context(context, registry).run(seed=0)
    analytic = mc.AnalyticPropagator.from_context(
        context, registry, step=1, underflow_rule=mc.UnderflowRule.TRUNCATE, overflow_rule=mc.OverflowRule.TRUNCATE
    ).run()

    assert monte_carlo.realized.tolist() == [2.0]
    assert analytic[0].pmf.values.tolist() == [2.0]
    assert context.events[0].timestamp.actual == 5.0


@pytest.mark.parametrize(
    "timestamp,match",
    [
        (mc.EventTimestamp(2.0, 1.0, 1.0), "earliest > latest"),
        (mc.EventTimestamp(0.0, 1.0, -1.0), "actual"),
        (mc.EventTimestamp(0.0, 1.0, 2.0), "actual"),
        (mc.EventTimestamp(0.0, math.inf, 0.0), "finite"),
    ],
)
def test_timestamp_validation_is_shared(timestamp: mc.EventTimestamp, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        mc.PropagationContext((mc.Event("event", timestamp),), {}, ())


@pytest.mark.parametrize("activity_type", [-2, -1])
def test_all_negative_activity_types_are_rejected(activity_type: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _context(activity_type=activity_type)
    registry = mc.DelayFamilyRegistry()
    with pytest.raises(ValueError, match="non-negative"):
        registry.add_constant(activity_type, 0.0)


def test_activity_types_and_indices_use_non_negative_signed_32_bit_domain() -> None:
    maximum = 2**31 - 1
    mc.Activity(maximum, 0.0, maximum)
    registry = mc.DelayFamilyRegistry()
    registry.add_constant(maximum, 0.0)

    with pytest.raises(ValueError, match="2147483647"):
        mc.Activity(maximum + 1, 0.0, 0)
    with pytest.raises(ValueError, match="2147483647"):
        mc.Activity(0, 0.0, maximum + 1)
    with pytest.raises(ValueError, match="2147483647"):
        mc.DelayFamilyRegistry().add_constant(maximum + 1, 0.0)
    with pytest.raises(ValueError, match="2147483647"):
        mc.PropagationContext(_events(), {(0, maximum + 1): mc.Activity(0, 1.0, 0)}, ())


@pytest.mark.parametrize(
    "values,weights,match",
    [
        ([], [], "must not be empty"),
        ([0.0], [], "same length"),
        ([-1.0], [1.0], "non-negative"),
        ([math.nan], [1.0], "finite"),
        ([0.0], [-1.0], "non-negative"),
        ([0.0], [0.0], "positive total"),
    ],
)
def test_empirical_malformed_inputs_fail_in_shared_registry(
    values: list[float], weights: list[float], match: str
) -> None:
    registry = mc.DelayFamilyRegistry()
    with pytest.raises(ValueError, match=match):
        registry.add_empirical(1, values, weights)


def test_empirical_weights_are_normalized_without_overflow() -> None:
    registry = mc.DelayFamilyRegistry()
    registry.add_empirical(1, [0.0, 1.0], [1.0e308, 1.0e308])
    simulator = mc.MonteCarloPropagator.from_context(_context(duration=0.0), registry)

    assert {simulator.run(seed).durations[0] for seed in range(32)} == {0.0, 1.0}


def test_propagation_context_rejects_duplicate_predecessor_and_unused_activity() -> None:
    activities = {(0, 1): mc.Activity(0, 1.0, 0)}
    with pytest.raises(ValueError, match="duplicate predecessor"):
        mc.PropagationContext(_events(), activities, ((1, ((0, 0), (0, 0))),))
    with pytest.raises(ValueError, match="missing from precedence_list"):
        mc.PropagationContext(_events(), activities, ())


def test_exact_discrete_equivalence_domain_accepts_aligned_nonbinding_model() -> None:
    registry = mc.DelayFamilyRegistry()
    registry.add_empirical(1, [0.0, 2.0], [1.0, 3.0])
    mc.validate_equivalence_domain(_context(), registry, step=1)


def test_exact_discrete_equivalence_domain_rejects_continuous_family() -> None:
    registry = mc.DelayFamilyRegistry()
    registry.add_exponential(1, scale=1.0, max_scale=2.0)
    with pytest.raises(mc.EquivalenceDomainError, match="excludes exponential and gamma"):
        mc.validate_equivalence_domain(_context(), registry, step=1)


def test_equivalence_domain_rejects_binding_bounds() -> None:
    registry = mc.DelayFamilyRegistry()
    registry.add_empirical(1, [0.0, 2.0], [0.5, 0.5])
    with pytest.raises(mc.EquivalenceDomainError, match="bounds bind"):
        mc.validate_equivalence_domain(_context(duration=10.0, latest=11.0), registry, step=1)


def test_equivalence_domain_rejects_any_positive_binding_probability() -> None:
    registry = mc.DelayFamilyRegistry()
    registry.add_empirical(1, [0.0, 2.0], [1.0 - 1.0e-13, 1.0e-13])

    with pytest.raises(mc.EquivalenceDomainError, match="bounds bind"):
        mc.validate_equivalence_domain(_context(duration=0.0, latest=1.0), registry, step=1)


def test_relative_scale_units_are_dimensionless() -> None:
    registry = mc.DelayFamilyRegistry()
    registry.add_exponential(1, scale=2.0, max_scale=0.5)
    analytic = mc.AnalyticPropagator.from_context(
        _context(duration=20.0),
        registry,
        step=1,
        underflow_rule=mc.UnderflowRule.TRUNCATE,
        overflow_rule=mc.OverflowRule.TRUNCATE,
    )
    increment = analytic.context.activities[(0, 1)][1].pmf
    assert increment.values.min() == 20.0
    assert increment.values.max() <= 30.0


def test_finite_parameter_product_overflow_is_explicit() -> None:
    registry = mc.DelayFamilyRegistry()
    registry.add_constant(1, 1e308)
    with pytest.raises(OverflowError, match="finite floating-point range"):
        mc.AnalyticPropagator.from_context(
            _context(duration=1e308, latest=1e308),
            registry,
            step=1,
            underflow_rule=mc.UnderflowRule.TRUNCATE,
            overflow_rule=mc.OverflowRule.TRUNCATE,
        )
