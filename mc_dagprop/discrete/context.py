from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mc_dagprop import EventTimestamp
from .pmf import DiscretePMF, Probability, Second

NodeIndex = int
EdgeIndex = int
Pred = tuple[NodeIndex, EdgeIndex]

@dataclass(frozen=True, slots=True)
class AnalyticEdge:
    pmf: DiscretePMF

@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    id: str
    timestamp: EventTimestamp

    bounds: tuple[Second, Second] | None = None

    def __post_init__(self) -> None:
        if self.bounds is None:
            object.__setattr__(self, "bounds", (self.timestamp.earliest, self.timestamp.latest))

@dataclass(frozen=True, slots=True)
class SimulatedEvent:
    pmf: DiscretePMF
    underflow: Probability
    overflow: Probability

@dataclass(frozen=True, slots=True)
class AnalyticContext:
    events: tuple[ScheduledEvent, ...]
    activities: dict[tuple[NodeIndex, NodeIndex], tuple[EdgeIndex, AnalyticEdge]]
    precedence_list: tuple[tuple[NodeIndex, tuple[Pred, ...]], ...]
    max_delay: Second = 0.0
    step_size: Second = 0.0

    def __post_init__(self) -> None:
        # Accept sequences in the constructor but store tuples internally to
        # avoid accidental mutation of the context after creation.
        if not isinstance(self.events, tuple):
            object.__setattr__(self, "events", tuple(self.events))
        if not isinstance(self.precedence_list, tuple):
            fixed = []
            for target, preds in self.precedence_list:  # type: ignore[attr-defined]
                preds_tuple = tuple(preds)
                fixed.append((target, preds_tuple))
            object.__setattr__(self, "precedence_list", tuple(fixed))

    def validate(self) -> None:
        for _, edge in self.activities.values():
            if not np.isclose(edge.pmf.step, self.step_size):
                raise ValueError(f"edge PMF step {edge.pmf.step} does not match context step size {self.step_size}")
