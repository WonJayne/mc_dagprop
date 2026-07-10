"""Shared frontend helpers for constructing Monte Carlo and analytic propagators."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from mc_dagprop.analytic import AnalyticActivity, AnalyticContext, OverflowRule, UnderflowRule
from mc_dagprop.analytic.distributions import constant_pmf, empirical_pmf, exponential_pmf, gamma_pmf
from mc_dagprop.monte_carlo import Activity, DagContext, Event, GenericDelayGenerator, MonteCarloPropagator
from mc_dagprop.types import ActivityType, Second

from .analytic._pmf import DiscretePMF


@dataclass(frozen=True, slots=True)
class PropagationContext:
    """Logical propagation model shared by both backends."""

    events: tuple[Event, ...]
    activities: dict[tuple[int, int], Activity]
    precedence_list: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]

    def __init__(self, events, activities, precedence_list) -> None:  # type: ignore[no-untyped-def]
        object.__setattr__(self, "events", tuple(events))
        object.__setattr__(self, "activities", dict(activities))
        object.__setattr__(self, "precedence_list", tuple((int(t), tuple(p)) for t, p in precedence_list))
        validate_propagation_context(self)


@dataclass(frozen=True, slots=True)
class _DelayFamily:
    add_to_generator: Callable[[GenericDelayGenerator, ActivityType], None]
    to_extra_pmf: Callable[[Second, int], DiscretePMF]


class DelayFamilyRegistry:
    """Registry of stochastic extra-delay families keyed by activity type."""

    def __init__(self) -> None:
        self._families: dict[ActivityType, _DelayFamily] = {}

    def _register(self, activity_type: ActivityType, family: _DelayFamily) -> None:
        if activity_type in self._families:
            raise ValueError(f"delay family already registered for activity type {activity_type}")
        self._families[activity_type] = family

    def add_empirical(self, activity_type: ActivityType, values, weights) -> None:  # type: ignore[no-untyped-def]
        values_tuple = tuple(float(v) for v in values)
        weights_tuple = tuple(float(w) for w in weights)
        self._register(
            activity_type,
            _DelayFamily(
                lambda generator, t: generator.add_empirical_absolute(t, values_tuple, weights_tuple),
                lambda _base, step: empirical_pmf(values_tuple, weights_tuple, step),
            ),
        )

    def add_constant(self, activity_type: ActivityType, factor: float) -> None:
        self._register(
            activity_type,
            _DelayFamily(
                lambda generator, t: generator.add_constant(t, factor),
                lambda base, step: constant_pmf(base * factor, step),
            ),
        )

    def add_exponential(self, activity_type: ActivityType, scale: float, max_scale: float) -> None:
        self._register(
            activity_type,
            _DelayFamily(
                lambda generator, t: generator.add_exponential(t, scale, max_scale),
                lambda base, step: exponential_pmf(base * scale, step, 0, int(math.ceil(base * max_scale / step) * step)),
            ),
        )

    def add_gamma(self, activity_type: ActivityType, shape: float, scale: float, max_scale: float = math.inf) -> None:
        stop = max_scale if math.isfinite(max_scale) else shape * scale * 10.0
        self._register(
            activity_type,
            _DelayFamily(
                lambda generator, t: generator.add_gamma(t, shape, scale, max_scale),
                lambda base, step: gamma_pmf(shape, base * scale, step, 0, int(math.ceil(base * stop / step) * step)),
            ),
        )

    def to_generator(self) -> GenericDelayGenerator:
        generator = GenericDelayGenerator()
        for activity_type, family in self._families.items():
            family.add_to_generator(generator, activity_type)
        return generator

    def increment_pmf(self, activity: Activity, step: int) -> DiscretePMF:
        family = self._families.get(activity.activity_type)
        if family is None:
            return constant_pmf(activity.minimal_duration, step)
        return family.to_extra_pmf(activity.minimal_duration, step).shift(activity.minimal_duration)


def validate_propagation_context(context: PropagationContext | DagContext) -> None:
    """Validate shared DAG structure and deterministic durations."""
    events = tuple(context.events)
    activities = dict(context.activities)
    event_count = len(events)
    seen_event_ids: set[str] = set()
    for index, event in enumerate(events):
        if event.event_id in seen_event_ids:
            raise ValueError(f"duplicate event id {event.event_id!r}")
        seen_event_ids.add(event.event_id)
        ts = event.timestamp
        for name, value in (("earliest", ts.earliest), ("latest", ts.latest), ("actual", ts.actual)):
            if not math.isfinite(value):
                raise ValueError(f"event {index} {name} must be finite")
    seen_activity_indices: set[int] = set()
    for (src, dst), activity in activities.items():
        if not (0 <= src < event_count and 0 <= dst < event_count):
            raise ValueError(f"activity {(src, dst)} references invalid event index")
        if activity.idx in seen_activity_indices:
            raise ValueError(f"duplicate activity index {activity.idx}")
        seen_activity_indices.add(activity.idx)
        if not math.isfinite(activity.minimal_duration) or activity.minimal_duration < 0.0:
            raise ValueError(f"activity {activity.idx} minimal_duration must be finite and non-negative")
    adjacency: list[list[int]] = [[] for _ in range(event_count)]
    indegree = [0] * event_count
    for target, predecessors in context.precedence_list:
        if not (0 <= target < event_count):
            raise ValueError(f"target index {target} out of range")
        for source, activity_index in predecessors:
            if not (0 <= source < event_count):
                raise ValueError(f"predecessor index {source} out of range")
            edge = activities.get((source, target))
            if edge is None:
                raise ValueError(f"missing activity for {(source, target)}")
            if edge.idx != activity_index:
                raise ValueError(f"activity index {activity_index} does not match edge {(source, target)}")
            adjacency[source].append(target)
            indegree[target] += 1
    queue = [index for index, degree in enumerate(indegree) if degree == 0]
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != event_count:
        raise ValueError("precedence list contains a cycle")


def monte_carlo_from_context(context: PropagationContext, registry: DelayFamilyRegistry) -> MonteCarloPropagator:
    """Create a Monte Carlo propagator from shared logical inputs."""
    return MonteCarloPropagator(DagContext(context.events, context.activities, context.precedence_list), registry.to_generator())


def analytic_from_context(
    context: PropagationContext,
    registry: DelayFamilyRegistry,
    *,
    step: int,
    overflow_rule: OverflowRule,
    underflow_rule: UnderflowRule,
):
    """Create an analytic propagator, converting extra delays into increments."""
    if not isinstance(step, int) or step <= 0:
        raise ValueError("analytic step must be a positive integer")
    for index, event in enumerate(context.events):
        if event.timestamp.earliest > event.timestamp.latest:
            raise ValueError(f"event {index} has earliest > latest")
    analytic_activities = {
        edge: (activity.idx, AnalyticActivity(activity.idx, registry.increment_pmf(activity, step)))
        for edge, activity in context.activities.items()
    }
    from mc_dagprop.analytic import create_analytic_propagator

    analytic_context = AnalyticContext(
        events=context.events,
        activities=analytic_activities,
        precedence_list=context.precedence_list,
        step=step,
        underflow_rule=underflow_rule,
        overflow_rule=overflow_rule,
    )
    return create_analytic_propagator(analytic_context)
