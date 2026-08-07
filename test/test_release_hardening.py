from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

import mc_dagprop as mc
from mc_dagprop.analytic import (
    AnalyticActivity,
    AnalyticContext,
    DiscretePMF,
    OverflowRule,
    UnderflowRule,
    create_analytic_propagator,
)
from mc_dagprop.analytic.distributions import constant_pmf


def _event(event_id: str, earliest: int = 0, latest: int = 10_020) -> mc.Event:
    return mc.Event(event_id, mc.EventTimestamp(earliest, latest, earliest))


def test_remove_clipping_preserves_subprobability_through_chain() -> None:
    events = (_event("root"), _event("clipped", latest=0), _event("child", latest=10))
    first = DiscretePMF(np.array([0, 10]), np.array([0.5, 0.5]), step=1)
    second = constant_pmf(1, 1)
    context = AnalyticContext(
        events=events,
        activities={(0, 1): (0, AnalyticActivity(0, first)), (1, 2): (1, AnalyticActivity(1, second))},
        precedence_list=((1, ((0, 0),)), (2, ((1, 1),))),
        step=1,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.REMOVE,
    )
    result = create_analytic_propagator(context).run()
    assert result[1].pmf.total_mass == pytest.approx(0.5)
    assert result[1].overflow == pytest.approx(0.5)
    assert result[2].pmf.total_mass == pytest.approx(0.5)
    assert result[2].pmf.values.tolist() == [1.0]


def test_remove_merge_max_uses_product_mass_for_independent_marginals() -> None:
    events = (_event("a", latest=0), _event("b", latest=0), _event("merge", latest=10))
    clipped = DiscretePMF(np.array([0, 10]), np.array([0.5, 0.5]), step=1)
    zero = constant_pmf(0, 1)
    context = AnalyticContext(
        events=events,
        activities={(0, 2): (0, AnalyticActivity(0, zero)), (1, 2): (1, AnalyticActivity(1, zero))},
        precedence_list=((2, ((0, 0), (1, 1))),),
        step=1,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.REMOVE,
    )
    propagator = create_analytic_propagator(context)
    event_a = propagator._convert_to_simulated_event(clipped, 0, 0).pmf
    merged = event_a.maximum(event_a)
    assert merged.total_mass == pytest.approx(0.25)


def test_unaligned_analytic_grid_inputs_are_rejected() -> None:
    events = (_event("root", earliest=2, latest=8),)
    context = AnalyticContext(events, {}, (), 3, UnderflowRule.TRUNCATE, OverflowRule.TRUNCATE)
    with pytest.raises(ValueError, match="aligned"):
        create_analytic_propagator(context)


def test_zero_duration_relative_family_is_deterministic_zero() -> None:
    events = (_event("root"), _event("child"))
    context = mc.PropagationContext(events, {(0, 1): mc.Activity(0, 0, 1)}, ((1, ((0, 0),)),))
    registry = mc.DelayFamilyRegistry()
    registry.add_exponential(1, scale=1.0, max_scale=3.0)
    analytic = mc.AnalyticPropagator.from_context(
        context, registry, step=1, underflow_rule=mc.UnderflowRule.TRUNCATE, overflow_rule=mc.OverflowRule.TRUNCATE
    )
    assert analytic.context.activities[(0, 1)][1].pmf.values.tolist() == [0.0]
    result = analytic.run()
    assert result[1].pmf.values.tolist() == [0.0]
    assert mc.MonteCarloPropagator.from_context(context, registry).run(1).durations[0] == 0.0


@pytest.mark.parametrize("max_scale", [0.0, -1.0, math.inf, math.nan])
def test_relative_families_require_finite_positive_max_scale(max_scale: float) -> None:
    registry = mc.DelayFamilyRegistry()
    with pytest.raises(ValueError):
        registry.add_exponential(1, scale=1.0, max_scale=max_scale)
    registry = mc.DelayFamilyRegistry()
    with pytest.raises(ValueError):
        registry.add_gamma(1, shape=1.0, scale=1.0, max_scale=max_scale)


def test_openbus_fixed_precedence_resource_activity_hand_computed() -> None:
    events = (_event("A_release"), _event("B_own_ready"), _event("B_acquire"))
    release_delay = DiscretePMF(np.array([0, 60]), np.array([0.5, 0.5]), step=60)
    own_duration = constant_pmf(180, 60)
    separation = constant_pmf(120, 60)
    context = AnalyticContext(
        events=events,
        activities={(0, 2): (0, AnalyticActivity(0, separation)), (1, 2): (1, AnalyticActivity(1, own_duration))},
        precedence_list=((2, ((0, 0), (1, 1))),),
        step=60,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.TRUNCATE,
    )
    # Inject A_release's uncertainty by shifting root event 0 after propagation primitive.
    merged = release_delay.shift(120).maximum(DiscretePMF.delta(180, 60))
    positive = {
        value: probability
        for value, probability in zip(merged.values.tolist(), merged.probabilities.tolist(), strict=True)
        if probability > 0
    }
    assert positive == pytest.approx({180.0: 1.0})
    propagated = create_analytic_propagator(context).run()[2].pmf
    propagated_positive = {
        value: probability
        for value, probability in zip(propagated.values.tolist(), propagated.probabilities.tolist(), strict=True)
        if probability > 0
    }
    assert propagated_positive == pytest.approx({180.0: 1.0})


def test_low_level_malformed_inputs_raise_in_subprocess() -> None:
    code = """
import mc_dagprop as mc
ctx = mc.DagContext([mc.Event('e', mc.EventTimestamp(0, 1, 0))], {(0, 0): mc.Activity(-1, 0, 0)}, [])
mc.MonteCarloPropagator(ctx, mc.GenericDelayGenerator())
"""
    completed = subprocess.run([sys.executable, "-c", code], timeout=10, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "activity index" in completed.stderr
