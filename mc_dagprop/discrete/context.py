from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mc_dagprop import EventTimestamp, SimEvent

from .pmf import DiscretePMF

NodeIndex = int
EdgeIndex = int
Pred = tuple[NodeIndex, EdgeIndex]


@dataclass
class AnalyticEdge:
    pmf: DiscretePMF


@dataclass
class AnalyticEvent:
    id: str
    timestamp: EventTimestamp
    bounds: tuple[float, float] | None = None


@dataclass
class AnalyticContext:
    events: list[AnalyticEvent]
    activities: dict[tuple[NodeIndex, NodeIndex], tuple[EdgeIndex, AnalyticEdge]]
    precedence_list: list[tuple[NodeIndex, list[Pred]]]
    max_delay: float = 0.0
    step_size: float = 0.0

    def validate(self) -> None:
        for ev in self.events:
            if ev.bounds is None:
                ev.bounds = (ev.timestamp.earliest, ev.timestamp.latest)
        for _, edge in self.activities.values():
            if not np.isclose(edge.pmf.step, self.step_size):
                raise ValueError(f"edge PMF step {edge.pmf.step} does not match context step size {self.step_size}")
