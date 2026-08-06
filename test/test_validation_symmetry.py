from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

import mc_dagprop as mc
from mc_dagprop.analytic import (
    AnalyticActivity,
    AnalyticContext,
    DiscretePMF,
    OverflowRule,
    UnderflowRule,
    validate_exact_equivalence_domain,
)
from mc_dagprop.core import Activity as CoreActivity
from mc_dagprop.core import DagContext as CoreDagContext
from mc_dagprop.core import Event as CoreEvent
from mc_dagprop.core import EventTimestamp as CoreEventTimestamp


def _events(count: int = 2) -> tuple[mc.Event, ...]:
    return tuple(mc.Event(f"event-{index}", mc.EventTimestamp(0.0, 100.0, 0.0)) for index in range(count))


def _analytic_activity(index: int, *, step: int = 1) -> AnalyticActivity:
    return AnalyticActivity(index, DiscretePMF.delta(0.0, step=step))


def _analytic_context(
    *,
    events: tuple[mc.Event, ...] | None = None,
    activities: dict[tuple[int, int], tuple[int, AnalyticActivity]] | None = None,
    precedence: tuple[tuple[int, tuple[tuple[int, int], ...]], ...] | None = None,
    step: int | float = 1,
) -> AnalyticContext:
    if activities is None:
        activities = {(0, 1): (0, _analytic_activity(0))}
    if precedence is None:
        precedence = ((1, ((0, 0),)),)
    return AnalyticContext(
        events=_events() if events is None else events,
        activities=activities,
        precedence_list=precedence,
        step=step,  # type: ignore[arg-type]
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.TRUNCATE,
    )


def _duplicate_analytic_indices() -> AnalyticContext:
    return _analytic_context(
        events=_events(3),
        activities={(0, 1): (0, _analytic_activity(0)), (0, 2): (0, _analytic_activity(0))},
        precedence=((1, ((0, 0),)), (2, ((0, 0),))),
    )


def _analytic_cycle() -> AnalyticContext:
    return _analytic_context(
        activities={(0, 1): (0, _analytic_activity(0)), (1, 0): (1, _analytic_activity(1))},
        precedence=((1, ((0, 0),)), (0, ((1, 1),))),
    )


@pytest.mark.parametrize(
    ("context_factory", "match"),
    [
        (lambda: _analytic_context(events=(), activities={}, precedence=()), "at least one event"),
        (lambda: _analytic_context(step=1.5), "integer"),
        (
            lambda: _analytic_context(
                events=(mc.Event("bad", mc.EventTimestamp(0.0, math.inf, 0.0)),), activities={}, precedence=()
            ),
            "finite",
        ),
        (
            lambda: _analytic_context(
                events=(mc.Event("bad", mc.EventTimestamp(0.0, 1.0, 2.0)),), activities={}, precedence=()
            ),
            "actual",
        ),
        (lambda: _analytic_context(activities={(0, 2): (0, _analytic_activity(0))}), "invalid node"),
        (
            lambda: _analytic_context(activities={(0, 1): (-1, _analytic_activity(-1))}, precedence=((1, ((0, -1),)),)),
            "non-negative",
        ),
        (lambda: _analytic_context(activities={(0, 1): (0, _analytic_activity(1))}), "does not match context mapping"),
        (_duplicate_analytic_indices, "duplicate activity index"),
        (
            lambda: _analytic_context(activities={(0, 1): (1, _analytic_activity(1))}, precedence=((1, ((0, 1),)),)),
            "contiguous",
        ),
        (
            lambda: _analytic_context(activities={(0, 1): (0, _analytic_activity(0, step=1))}, step=2),
            "does not match context step",
        ),
        (
            lambda: _analytic_context(
                activities={
                    (0, 1): (
                        0,
                        AnalyticActivity(
                            0, DiscretePMF(np.array([0.0]), np.array([0.5]), step=1, allow_subprobability=True)
                        ),
                    )
                }
            ),
            "does not sum to 1",
        ),
        (lambda: _analytic_context(precedence=((1, ((0, 0),)), (1, ()))), "duplicate precedence entry"),
        (lambda: _analytic_context(precedence=((2, ((0, 0),)),)), "target index"),
        (lambda: _analytic_context(precedence=((1, ((2, 0),)),)), "predecessor index"),
        (lambda: _analytic_context(activities={}, precedence=((1, ((0, 0),)),)), "missing activity"),
        (lambda: _analytic_context(precedence=((1, ((0, 1),)),)), "does not match context mapping"),
        (lambda: _analytic_context(precedence=()), "missing from precedence list"),
        (_analytic_cycle, "cycle"),
    ],
)
def test_low_level_analytic_validation_rejects_malformed_inputs(
    context_factory: Callable[[], AnalyticContext], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        mc.create_analytic_propagator(context_factory())


def test_standalone_exact_domain_validator_handles_valid_and_cyclic_contexts() -> None:
    validate_exact_equivalence_domain(_analytic_context())
    with pytest.raises(ValueError, match="cycle"):
        validate_exact_equivalence_domain(_analytic_cycle())


def _shared_activity(index: int, source: int, target: int) -> tuple[tuple[int, int], mc.Activity]:
    return (source, target), mc.Activity(index, 1.0, 0)


def _duplicate_shared_indices() -> mc.PropagationContext:
    return mc.PropagationContext(
        _events(3), dict([_shared_activity(0, 0, 1), _shared_activity(0, 0, 2)]), ((1, ((0, 0),)), (2, ((0, 0),)))
    )


@pytest.mark.parametrize(
    ("context_factory", "match"),
    [
        (lambda: mc.PropagationContext((), {}, ()), "at least one event"),
        (
            lambda: mc.PropagationContext((mc.Event("duplicate", mc.EventTimestamp(0.0, 1.0, 0.0)),) * 2, {}, ()),
            "duplicate event id",
        ),
        (lambda: mc.PropagationContext(_events(), dict([_shared_activity(0, 0, 2)]), ()), "invalid event index"),
        (_duplicate_shared_indices, "duplicate activity index"),
        (lambda: mc.PropagationContext(_events(), dict([_shared_activity(1, 0, 1)]), ((1, ((0, 1),)),)), "contiguous"),
        (lambda: mc.PropagationContext(_events(), {(0, 1): mc.Activity(0, math.inf, 0)}, ((1, ((0, 0),)),)), "finite"),
        (
            lambda: mc.PropagationContext(_events(), {(True, 1): mc.Activity(0, 1.0, 0)}, ((1, ((True, 0),)),)),
            "source index",
        ),
        (
            lambda: mc.PropagationContext(_events(), {(0.5, 1): mc.Activity(0, 1.0, 0)}, ((1, ((0.5, 0),)),)),
            "source index",
        ),
        (
            lambda: mc.PropagationContext(_events(), dict([_shared_activity(0, 0, 1)]), ((True, ((0, 0),)),)),
            "target index",
        ),
        (
            lambda: mc.PropagationContext(_events(), dict([_shared_activity(0, 0, 1)]), ((1, ((0, 0.5),)),)),
            "activity index",
        ),
        (
            lambda: mc.PropagationContext(_events(), dict([_shared_activity(0, 0, 1)]), ((1, ((0, 0),)), (1, ()))),
            "duplicate precedence entry",
        ),
        (
            lambda: mc.PropagationContext(_events(), dict([_shared_activity(0, 0, 1)]), ((2, ((0, 0),)),)),
            "target index",
        ),
        (
            lambda: mc.PropagationContext(_events(), dict([_shared_activity(0, 0, 1)]), ((1, ((2, 0),)),)),
            "predecessor index",
        ),
        (lambda: mc.PropagationContext(_events(), {}, ((1, ((0, 0),)),)), "missing activity"),
        (
            lambda: mc.PropagationContext(_events(), dict([_shared_activity(0, 0, 1)]), ((1, ((0, 1),)),)),
            "does not match edge",
        ),
        (
            lambda: mc.PropagationContext(
                _events(),
                dict([_shared_activity(0, 0, 1), _shared_activity(1, 1, 0)]),
                ((1, ((0, 0),)), (0, ((1, 1),))),
            ),
            "cycle",
        ),
    ],
)
def test_shared_validation_rejects_malformed_inputs(
    context_factory: Callable[[], mc.PropagationContext], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        context_factory()


def test_compatibility_core_reexports_compiled_context_types() -> None:
    assert (CoreActivity, CoreDagContext, CoreEvent, CoreEventTimestamp) == (
        mc.Activity,
        mc.DagContext,
        mc.Event,
        mc.EventTimestamp,
    )
