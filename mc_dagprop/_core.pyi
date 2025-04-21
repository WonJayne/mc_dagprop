from typing import Dict, List, Tuple

class SimulationTreeLink:
    minimal_duration: float
    link_type: int

    def __init__(self, minimal_duration: float, link_type: int) -> None: ...

class EventPointInTime:
    earliest: float
    latest: float
    actual: float

class SimContext:
    events: List[Tuple[str, Tuple[float, float, float]]]
    link_map: Dict[Tuple[int, int], Tuple[int, SimulationTreeLink]]
    precedence_list: List[Tuple[int, List[Tuple[int, int]]]]
    max_delay: float

    def __init__(
        self,
        events: List[Tuple[str, Tuple[float, float, float]]],
        link_map: Dict[Tuple[int, int], Tuple[int, SimulationTreeLink]],
        precedence_list: List[Tuple[int, List[Tuple[int, int]]]],
        max_delay: float,
    ) -> None: ...

class SimResult:
    realized: List[float]
    delays: List[float]
    cause_event: List[int]

class GenericDelayGenerator:
    def __init__(self) -> None: ...
    def add_constant(self, link_type: int, factor: float) -> None: ...
    def add_exponential(self, link_type: int, lambda_: float, max_scale: float) -> None: ...
    def set_seed(self, seed: int) -> None: ...

class Simulator:
    def __init__(self, context: SimContext, generator: GenericDelayGenerator) -> None: ...
    def run(self, seed: int) -> SimResult: ...
    def run_many(self, seeds: List[int]) -> List[SimResult]: ...
