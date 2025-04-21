# mc_dagprop/_core.pyi
from collections.abc import Sequence
from typing import Iterable, Mapping

class SimEvent:
    """
    Represents an event (node) with its earliest/latest window and actual timestamp.
    """

    node_id: str
    timestamp: "EventTimestamp"

    def __init__(self, node_id: str, timestamp: "EventTimestamp") -> None: ...

class EventTimestamp:
    """
    Holds the earliest/latest bounds and the actual (scheduled) time for an event.
    """

    earliest: float
    latest: float
    actual: float

    def __init__(self, earliest: float, latest: float, actual: float) -> None: ...

class SimActivity:
    """
    Represents an activity (edge) in the DAG, with its minimal duration and type.
    """

    minimal_duration: float
    activity_type: int

    def __init__(self, minimal_duration: float, activity_type: int) -> None: ...

class SimContext:
    """
    Wraps the DAG: a list of events, a map of activities, a precedence list, and a max?delay.
    """

    events: Sequence[SimEvent]
    activities: Mapping[tuple[int, int], tuple[int, SimActivity]]
    precedence_list: Sequence[tuple[int, list[tuple[int, int]]]]
    max_delay: float

    def __init__(
        self,
        events: Sequence[SimEvent],
        activities: Mapping[tuple[int, int], tuple[int, SimActivity]],
        precedence_list: Sequence[tuple[int, list[tuple[int, int]]]],
        max_delay: float,
    ) -> None: ...

class SimResult:
    """
    The result of one run: realized times, per-activity delays, and causal predecessors.
    """

    realized: list[float]
    delays: list[float]
    cause_event: list[int]

class GenericDelayGenerator:
    """
    Configurable delay generator: constant or exponential per activity_type.
    """

    def __init__(self) -> None: ...
    def set_seed(self, seed: int) -> None: ...
    def add_constant(self, activity_type: int, factor: float) -> None: ...
    def add_exponential(self, activity_type: int, lambda_: float, max_scale: float) -> None: ...

class Simulator:
    """
    Monte Carlo DAG propagator: run single or batch simulations.
    """

    def __init__(self, context: SimContext, generator: GenericDelayGenerator) -> None: ...
    def run(self, seed: int) -> SimResult: ...
    def run_many(self, seeds: Iterable[int]) -> list[SimResult]: ...
