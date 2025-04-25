# mc_dagprop/_core.pyi
from collections.abc import Iterable, Mapping, Sequence
from typing import TypeAlias

from numpy._typing import NDArray

EventIndex: TypeAlias = int
ActivityIndex: TypeAlias = int
ActivityType: TypeAlias = int
EventId: TypeAlias = str
Second: TypeAlias = float

class SimEvent:
    """
    Represents an event (node) with its earliest/latest window and actual timestamp.
    """

    id: EventId
    timestamp: "EventTimestamp"

    def __init__(self, id_: EventId, timestamp: "EventTimestamp") -> None: ...

class EventTimestamp:
    """
    Holds the earliest/latest bounds and the actual (scheduled) time for an event.
    """

    earliest: Second
    latest: Second
    actual: Second

    def __init__(self, earliest: Second, latest: Second, actual: Second) -> None: ...

class SimActivity:
    """
    Represents an activity (edge) in the DAG, with its minimal duration and type.
    """

    minimal_duration: Second
    activity_type: ActivityType

    def __init__(self, minimal_duration: Second, activity_type: ActivityType) -> None: ...

class SimContext:
    """
    Wraps the DAG: a list of events, a map of activities, a precedence list, and a max?delay.
    """

    events: Sequence[SimEvent]
    activities: Mapping[tuple[EventIndex, EventIndex], tuple[ActivityIndex, SimActivity]]
    precedence_list: Sequence[tuple[EventIndex, list[tuple[EventIndex, ActivityIndex]]]]
    max_delay: Second

    def __init__(
        self,
        events: Sequence[SimEvent],
        activities: Mapping[tuple[EventIndex, EventIndex], tuple[ActivityIndex, SimActivity]],
        precedence_list: Sequence[tuple[EventIndex, list[tuple[EventIndex, ActivityIndex]]]],
        max_delay: Second,
    ) -> None: ...

class SimResult:
    """
    The result of one run: realized times, per-activity delays, and causal predecessors.
    """

    realized: NDArray[Second]
    durations: NDArray[Second]
    cause_event: NDArray[EventIndex]

class GenericDelayGenerator:
    """
    Configurable delay generator: constant or exponential per activity_type.
    """

    def __init__(self) -> None: ...
    def set_seed(self, seed: int) -> None: ...
    def add_constant(self, activity_type: ActivityType, factor: float) -> None: ...
    def add_exponential(self, activity_type: ActivityType, lambda_: float, max_scale: float) -> None: ...
    def add_gamma(
        self, activity_type: ActivityType, shape: float, scale: float, max_scale: float = float("inf")
    ) -> None: ...

class Simulator:
    """
    Monte Carlo DAG propagator: run single or batch simulations.
    """

    def __init__(self, context: SimContext, generator: GenericDelayGenerator) -> None: ...
    def run(self, seed: int) -> SimResult: ...
    def run_many(self, seeds: Iterable[int]) -> list[SimResult]: ...
    def run_many_arrays(self, seeds: Iterable[int]) -> tuple[NDArray[Second], NDArray[Second], NDArray[EventIndex]]: ...
