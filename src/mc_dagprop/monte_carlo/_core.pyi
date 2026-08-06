# mc_dagprop/monte_carlo/_core.pyi
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from numpy._typing import NDArray

from mc_dagprop.frontend import DelayFamilyRegistry, PropagationContext
from mc_dagprop.types import ActivityIndex, ActivityType, EventId, EventIndex, Second

class EventTimestamp:
    """
    Event timestamp metadata.

    Propagation-context validation requires finite values satisfying
    ``earliest <= actual <= latest``. Monte Carlo initializes realized time at
    ``earliest``. ``actual`` is an external/reference timestamp and ``latest``
    is metadata; neither clips or otherwise changes Monte Carlo propagation.
    """

    def __init__(self, earliest: Second, latest: Second, actual: Second) -> None: ...
    @property
    def earliest(self) -> Second: ...
    @property
    def latest(self) -> Second: ...
    @property
    def actual(self) -> Second: ...

class Event:
    """
    Represents an event (node) with its earliest/latest window and actual timestamp.
    """

    def __init__(self, event_id: EventId, timestamp: EventTimestamp) -> None: ...
    @property
    def event_id(self) -> EventId: ...
    @property
    def timestamp(self) -> EventTimestamp: ...

class Activity:
    """
    Activity edge with non-negative signed-32-bit ``idx`` and ``activity_type``.
    """

    def __init__(self, idx: ActivityIndex, minimal_duration: Second, activity_type: ActivityType) -> None: ...
    @property
    def idx(self) -> ActivityIndex: ...
    @property
    def minimal_duration(self) -> Second: ...
    @property
    def activity_type(self) -> ActivityType: ...

class DagContext:
    """
    Wraps the DAG: a list of events, activities, and a precedence list.
    Every event/activity index must be in ``[0, 2**31 - 1]``.
    ``precedence_list`` can be in any order; ``Simulator`` sorts it
    topologically and raises ``RuntimeError`` on cycles.
    """

    def __init__(
        self,
        events: Sequence[Event],
        activities: Mapping[tuple[EventIndex, EventIndex], Activity],
        precedence_list: Sequence[tuple[EventIndex, Sequence[tuple[EventIndex, ActivityIndex]]]],
    ) -> None: ...
    @property
    def events(self) -> Sequence[Event]: ...
    @property
    def activities(self) -> Mapping[tuple[EventIndex, EventIndex], Activity]: ...
    @property
    def precedence_list(self) -> Sequence[tuple[EventIndex, list[tuple[EventIndex, ActivityIndex]]]]: ...

class SimResult:
    """
    The result of one run: realized times, per-activity delays, and causal predecessors.
    """

    @property
    def realized(self) -> NDArray[np.float64]: ...
    @property
    def durations(self) -> NDArray[np.float64]: ...
    @property
    def cause_event(self) -> NDArray[np.int32]: ...

class GenericDelayGenerator:
    """
    Configurable delay generator. Supports per-``activity_type`` distributions:
    constant, exponential, gamma, and empirical (absolute or relative).
    Activity types are signed-32-bit non-negative integers.
    """

    def __init__(self) -> None: ...
    def add_constant(self, activity_type: ActivityType, factor: float) -> None: ...
    def add_exponential(self, activity_type: ActivityType, scale: float, max_scale: float) -> None: ...
    def add_gamma(self, activity_type: ActivityType, shape: float, scale: float, max_scale: float = 10.0) -> None: ...
    def add_empirical_absolute(
        self, activity_type: ActivityType, values: Iterable[Second], weights: Iterable[float]
    ) -> None: ...
    def add_empirical_relative(
        self, activity_type: ActivityType, factors: Iterable[Second], weights: Iterable[float]
    ) -> None: ...

class MonteCarloPropagator:
    """
    Reentrant, thread-safe Monte Carlo DAG propagator.

    Within a fixed package build and platform, each result is determined by the
    immutable model and its explicit unsigned 64-bit seed. Concurrent calls on
    one instance are schedule independent. Cross-platform bit identity is not
    promised. ``run_many(seeds)`` is the ordered equivalent of independent
    ``run(seed)`` calls. A bool, negative integer, or integer above
    ``2**64 - 1`` is rejected as a seed.
    """

    def __init__(self, context: DagContext, generator: GenericDelayGenerator) -> None: ...
    @staticmethod
    def from_context(context: PropagationContext, registry: DelayFamilyRegistry) -> MonteCarloPropagator: ...
    def node_count(self) -> int: ...
    def activity_count(self) -> int: ...
    def run(self, seed: int) -> SimResult: ...
    def run_many(self, seeds: Iterable[int]) -> list[SimResult]: ...

Simulator = MonteCarloPropagator
