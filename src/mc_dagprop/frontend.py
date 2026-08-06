"""Shared, validated construction of analytic and Monte Carlo propagators."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum, unique
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from mc_dagprop.analytic import AnalyticActivity, AnalyticContext, OverflowRule, UnderflowRule
from mc_dagprop.analytic.distributions import constant_pmf, empirical_pmf, exponential_pmf, gamma_pmf
from mc_dagprop.monte_carlo import Activity, DagContext, Event, GenericDelayGenerator, MonteCarloPropagator
from mc_dagprop.types import ActivityType, Second

from .analytic._pmf import DiscretePMF

if TYPE_CHECKING:
    from mc_dagprop.analytic import AnalyticPropagator

_MAX_MODEL_INDEX = 2**31 - 1
_FamilyKind = Literal["deterministic", "discrete", "continuous"]


@unique
class EquivalenceMode(StrEnum):
    """Cross-backend equivalence contract to validate.

    ``EXACT_DISCRETE`` covers grid-aligned deterministic and empirical delays.
    ``QUANTIZED_CONTINUOUS`` additionally covers exponential and gamma delays
    after every sampled stochastic activity extra delay is floored to the
    analytic grid before propagation.
    """

    EXACT_DISCRETE = "exact-discrete"
    QUANTIZED_CONTINUOUS = "quantized-continuous"


class EquivalenceDomainError(ValueError):
    """Raised when a model is outside a requested equivalence domain."""


@dataclass(frozen=True, slots=True)
class PropagationContext:
    """Immutable event-activity DAG shared by both propagation backends.

    ``earliest`` is the deterministic release time and the initial Monte Carlo
    event time. ``latest`` is an analytic hard bound and Monte Carlo metadata.
    ``actual`` is reference metadata for callers and is not used by either
    propagation algorithm. All three timestamps must be finite and satisfy
    ``earliest <= actual <= latest``.
    """

    events: tuple[Event, ...]
    activities: Mapping[tuple[int, int], Activity]
    precedence_list: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]

    def __init__(
        self,
        events: Iterable[Event],
        activities: Mapping[tuple[int, int], Activity],
        precedence_list: Iterable[tuple[int, Iterable[tuple[int, int]]]],
    ) -> None:
        object.__setattr__(self, "events", tuple(events))
        object.__setattr__(self, "activities", MappingProxyType(dict(activities)))
        object.__setattr__(
            self, "precedence_list", tuple((target, tuple(predecessors)) for target, predecessors in precedence_list)
        )
        validate_propagation_context(self)


@dataclass(frozen=True, slots=True)
class _DelayFamily:
    kind: _FamilyKind
    add_to_generator: Callable[[GenericDelayGenerator, ActivityType], None]
    to_extra_pmf: Callable[[Second, int], DiscretePMF]


def _require_activity_type(activity_type: object) -> None:
    if isinstance(activity_type, bool) or not isinstance(activity_type, int):
        raise TypeError("activity_type must be a non-negative integer")
    if activity_type < 0:
        raise ValueError("activity_type must be non-negative")
    if activity_type > _MAX_MODEL_INDEX:
        raise ValueError(f"activity_type must not exceed {_MAX_MODEL_INDEX}")


def _require_non_negative_index(value: object, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{label} must be non-negative")
    if value > _MAX_MODEL_INDEX:
        raise ValueError(f"{label} must not exceed {_MAX_MODEL_INDEX}")


def _require_finite_non_negative(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a real number, not bool")
    converted = float(value)
    if not math.isfinite(converted) or converted < 0.0:
        raise ValueError(f"{label} must be finite and non-negative")
    return converted


def _require_finite_positive(value: float, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be a real number, not bool")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return converted


def _require_positive_step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("analytic step must be an integer number of seconds")
    if value <= 0:
        raise ValueError("analytic step must be positive")
    return value


def _require_equivalence_mode(value: object) -> EquivalenceMode:
    if not isinstance(value, EquivalenceMode):
        raise TypeError("mode must be an EquivalenceMode")
    return value


def _checked_product(left: float, right: float, label: str) -> Second:
    product = left * right
    if not math.isfinite(product):
        raise OverflowError(f"{label} exceeds the finite floating-point range")
    return float(product)


def _validated_empirical_inputs(
    values: Iterable[float], weights: Iterable[float]
) -> tuple[tuple[Second, ...], tuple[float, ...]]:
    values_tuple = tuple(float(_require_finite_non_negative(value, "empirical value")) for value in values)
    weights_tuple = tuple(_require_finite_non_negative(weight, "empirical weight") for weight in weights)
    if not values_tuple:
        raise ValueError("empirical distribution must not be empty")
    if len(values_tuple) != len(weights_tuple):
        raise ValueError("empirical values and weights must have the same length")
    maximum_weight = max(weights_tuple)
    if maximum_weight <= 0.0:
        raise ValueError("empirical weights must have positive total mass")
    scaled_weights = tuple(weight / maximum_weight for weight in weights_tuple)
    scaled_total = math.fsum(scaled_weights)
    normalized_weights = tuple(weight / scaled_total for weight in scaled_weights)
    return values_tuple, normalized_weights


class DelayFamilyRegistry:
    """Extra-delay families keyed by a non-negative activity type.

    Activity base durations and empirical absolute values are seconds.
    Constant, exponential, and gamma parameters are dimensionless factors
    multiplied by the activity's base duration. Exponential ``scale`` is the
    mean factor; gamma ``scale`` is its scale factor; ``max_scale`` is the
    inclusive truncation factor for either continuous distribution.
    """

    def __init__(self) -> None:
        self._families: dict[ActivityType, _DelayFamily] = {}

    def _register(self, activity_type: ActivityType, family: _DelayFamily) -> None:
        _require_activity_type(activity_type)
        if activity_type in self._families:
            raise ValueError(f"delay family already registered for activity type {activity_type}")
        self._families[activity_type] = family

    def add_empirical(self, activity_type: ActivityType, values: Iterable[float], weights: Iterable[float]) -> None:
        """Register absolute extra delays in seconds with relative weights."""
        values_tuple, weights_tuple = _validated_empirical_inputs(values, weights)
        self._register(
            activity_type,
            _DelayFamily(
                "discrete",
                lambda generator, registered_type: generator.add_empirical_absolute(
                    registered_type, values_tuple, weights_tuple
                ),
                lambda _base, step: empirical_pmf(values_tuple, weights_tuple, step),
            ),
        )

    def add_constant(self, activity_type: ActivityType, factor: float) -> None:
        """Register deterministic extra delay ``base_duration * factor``."""
        factor_value = _require_finite_non_negative(factor, "constant delay factor")

        def to_constant_extra_pmf(base: Second, step: int) -> DiscretePMF:
            return constant_pmf(_checked_product(base, factor_value, "constant extra delay"), step)

        self._register(
            activity_type,
            _DelayFamily(
                "deterministic",
                lambda generator, registered_type: generator.add_constant(registered_type, factor_value),
                to_constant_extra_pmf,
            ),
        )

    def add_exponential(self, activity_type: ActivityType, scale: float, max_scale: float) -> None:
        """Register a truncated exponential extra-delay factor.

        The sampled dimensionless factor has mean parameter ``scale`` before
        conditioning on ``factor <= max_scale``. Extra delay in seconds is the
        sampled factor multiplied by the activity's base duration.
        """
        scale_value = _require_finite_positive(scale, "exponential scale factor")
        max_scale_value = _require_finite_positive(max_scale, "exponential max_scale factor")

        def to_exponential_extra_pmf(base: Second, step: int) -> DiscretePMF:
            if base == 0.0:
                return constant_pmf(0.0, step)
            scale_seconds = _checked_product(base, scale_value, "exponential scale in seconds")
            cutoff_seconds = _checked_product(base, max_scale_value, "exponential cutoff in seconds")
            return exponential_pmf(scale_seconds, step, 0.0, cutoff_seconds)

        self._register(
            activity_type,
            _DelayFamily(
                "continuous",
                lambda generator, registered_type: generator.add_exponential(
                    registered_type, scale_value, max_scale_value
                ),
                to_exponential_extra_pmf,
            ),
        )

    def add_gamma(self, activity_type: ActivityType, shape: float, scale: float, max_scale: float = 10.0) -> None:
        """Register a truncated gamma extra-delay factor."""
        shape_value = _require_finite_positive(shape, "gamma shape")
        scale_value = _require_finite_positive(scale, "gamma scale factor")
        max_scale_value = _require_finite_positive(max_scale, "gamma max_scale factor")

        def to_gamma_extra_pmf(base: Second, step: int) -> DiscretePMF:
            if base == 0.0:
                return constant_pmf(0.0, step)
            scale_seconds = _checked_product(base, scale_value, "gamma scale in seconds")
            cutoff_seconds = _checked_product(base, max_scale_value, "gamma cutoff in seconds")
            return gamma_pmf(shape_value, scale_seconds, step, 0.0, cutoff_seconds)

        self._register(
            activity_type,
            _DelayFamily(
                "continuous",
                lambda generator, registered_type: generator.add_gamma(
                    registered_type, shape_value, scale_value, max_scale_value
                ),
                to_gamma_extra_pmf,
            ),
        )

    def to_generator(self) -> GenericDelayGenerator:
        """Build an independent Monte Carlo generator from this registry."""
        generator = GenericDelayGenerator()
        for activity_type, family in self._families.items():
            family.add_to_generator(generator, activity_type)
        return generator

    def increment_pmf(self, activity: Activity, step: int) -> DiscretePMF:
        """Return the full base-plus-extra activity-duration PMF."""
        family = self._families.get(activity.activity_type)
        if family is None:
            return constant_pmf(activity.minimal_duration, step)
        return family.to_extra_pmf(activity.minimal_duration, step).shift(activity.minimal_duration)

    def has_continuous_family(self, activity_types: Iterable[ActivityType]) -> bool:
        """Return whether any selected activity type uses a continuous family."""
        selected_types = set(activity_types)
        return any(
            family.kind == "continuous"
            for activity_type, family in self._families.items()
            if activity_type in selected_types
        )


def validate_propagation_context(context: PropagationContext | DagContext) -> None:
    """Validate the complete shared DAG and public timestamp contract."""
    events = tuple(context.events)
    activities = dict(context.activities)
    precedence_list: tuple[tuple[int, tuple[tuple[int, int], ...]], ...] = tuple(
        (target, tuple(predecessors)) for target, predecessors in context.precedence_list
    )
    event_count = len(events)
    if event_count == 0:
        raise ValueError("propagation context must contain at least one event")

    seen_event_ids: set[str] = set()
    for index, event in enumerate(events):
        if event.event_id in seen_event_ids:
            raise ValueError(f"duplicate event id {event.event_id!r}")
        seen_event_ids.add(event.event_id)
        timestamp = event.timestamp
        for name, value in (
            ("earliest", timestamp.earliest),
            ("latest", timestamp.latest),
            ("actual", timestamp.actual),
        ):
            if not math.isfinite(value):
                raise ValueError(f"event {index} {name} must be finite")
        if timestamp.earliest > timestamp.latest:
            raise ValueError(f"event {index} has earliest > latest")
        if not timestamp.earliest <= timestamp.actual <= timestamp.latest:
            raise ValueError(f"event {index} actual must lie within [earliest, latest]")

    seen_activity_indices: set[int] = set()
    for (source, target), activity in activities.items():
        _require_non_negative_index(source, "activity source index")
        _require_non_negative_index(target, "activity target index")
        if not (0 <= source < event_count and 0 <= target < event_count):
            raise ValueError(f"activity {(source, target)} references invalid event index")
        _require_non_negative_index(activity.idx, "activity index")
        if activity.idx in seen_activity_indices:
            raise ValueError(f"duplicate activity index {activity.idx}")
        _require_activity_type(activity.activity_type)
        seen_activity_indices.add(activity.idx)
        if not math.isfinite(activity.minimal_duration) or activity.minimal_duration < 0.0:
            raise ValueError(f"activity {activity.idx} minimal_duration must be finite and non-negative")
    if seen_activity_indices != set(range(len(seen_activity_indices))):
        raise ValueError("activity indices must be contiguous from 0 to n-1")

    adjacency: list[list[int]] = [[] for _ in range(event_count)]
    indegree = [0] * event_count
    seen_targets: set[int] = set()
    referenced_edges: set[tuple[int, int]] = set()
    for target, predecessors in precedence_list:
        _require_non_negative_index(target, "precedence target index")
        if target in seen_targets:
            raise ValueError(f"duplicate precedence entry for target {target}")
        seen_targets.add(target)
        if not (0 <= target < event_count):
            raise ValueError(f"target index {target} out of range")
        seen_sources: set[int] = set()
        for source, activity_index in predecessors:
            _require_non_negative_index(source, "predecessor source index")
            _require_non_negative_index(activity_index, "predecessor activity index")
            if source in seen_sources:
                raise ValueError(f"duplicate predecessor {source} for target {target}")
            seen_sources.add(source)
            if not (0 <= source < event_count):
                raise ValueError(f"predecessor index {source} out of range")
            edge_key = (source, target)
            edge = activities.get(edge_key)
            if edge is None:
                raise ValueError(f"missing activity for {edge_key}")
            if edge.idx != activity_index:
                raise ValueError(f"activity index {activity_index} does not match edge {edge_key}")
            if edge_key in referenced_edges:
                raise ValueError(f"duplicate predecessor edge {edge_key}")
            referenced_edges.add(edge_key)
            adjacency[source].append(target)
            indegree[target] += 1

    unused_edges = set(activities).difference(referenced_edges)
    if unused_edges:
        raise ValueError(f"activities missing from precedence_list: {sorted(unused_edges)}")

    queue: deque[int] = deque(index for index, degree in enumerate(indegree) if degree == 0)
    visited = 0
    while queue:
        node = queue.popleft()
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != event_count:
        raise ValueError("precedence list contains a cycle")


def monte_carlo_from_context(context: PropagationContext, registry: DelayFamilyRegistry) -> MonteCarloPropagator:
    """Create a thread-safe Monte Carlo propagator from shared inputs."""
    validate_propagation_context(context)
    native_activities = {
        (int(source), int(target)): activity for (source, target), activity in context.activities.items()
    }
    native_precedence = tuple(
        (int(target), tuple((int(source), int(activity_index)) for source, activity_index in predecessors))
        for target, predecessors in context.precedence_list
    )
    native_context = DagContext(context.events, native_activities, native_precedence)
    return MonteCarloPropagator(native_context, registry.to_generator())


def analytic_from_context(
    context: PropagationContext,
    registry: DelayFamilyRegistry,
    *,
    step: int,
    overflow_rule: OverflowRule,
    underflow_rule: UnderflowRule,
) -> AnalyticPropagator:
    """Create an analytic propagator, converting extra delays into increments."""
    validate_propagation_context(context)
    step = _require_positive_step(step)
    analytic_activities = {
        (int(source), int(target)): (
            activity.idx,
            AnalyticActivity(activity.idx, registry.increment_pmf(activity, step)),
        )
        for (source, target), activity in context.activities.items()
    }
    from mc_dagprop.analytic import create_analytic_propagator  # noqa: PLC0415

    analytic_context = AnalyticContext(
        events=context.events,
        activities=analytic_activities,
        precedence_list=tuple(
            (int(target), tuple((int(source), int(activity_index)) for source, activity_index in predecessors))
            for target, predecessors in context.precedence_list
        ),
        step=step,
        underflow_rule=underflow_rule,
        overflow_rule=overflow_rule,
    )
    return create_analytic_propagator(analytic_context)


def validate_equivalence_domain(
    context: PropagationContext,
    registry: DelayFamilyRegistry,
    *,
    step: int,
    mode: EquivalenceMode = EquivalenceMode.EXACT_DISCRETE,
) -> None:
    """Validate the qualified analytic/Monte Carlo equivalence contract.

    Both modes require a valid grid-aligned DAG, independent stochastic
    ancestry at every merge, and event bounds that never bind. Exact mode also
    rejects continuous families. Quantized-continuous mode compares analytic
    bins with Monte Carlo propagation after each stochastic activity extra
    delay is floored to ``step``; its agreement is statistical rather than
    sample-by-sample.
    """
    mode = _require_equivalence_mode(mode)
    used_activity_types = {activity.activity_type for activity in context.activities.values()}
    if mode is EquivalenceMode.EXACT_DISCRETE and registry.has_continuous_family(used_activity_types):
        raise EquivalenceDomainError("exact-discrete equivalence excludes exponential and gamma families")

    try:
        propagator = analytic_from_context(
            context, registry, step=step, underflow_rule=UnderflowRule.REMOVE, overflow_rule=OverflowRule.REMOVE
        )
        result = propagator.run()
    except (ArithmeticError, RuntimeError, TypeError, ValueError) as error:
        raise EquivalenceDomainError(str(error)) from error

    binding_events = [
        index for index, event in enumerate(result) if float(event.underflow) > 0.0 or float(event.overflow) > 0.0
    ]
    if binding_events:
        raise EquivalenceDomainError(
            f"analytic event bounds bind at events {binding_events}; "
            "cross-backend equivalence requires nonbinding bounds"
        )
