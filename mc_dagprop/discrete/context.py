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
    """Edge with an associated delay distribution.

    Attributes:
        pmf: Probability mass function describing the delay on this edge.
    """

    pmf: DiscretePMF


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    """Timing information for a planned event.

    Attributes:
        id: Unique identifier of the event.
        timestamp: Earliest, latest and nominal time bounds.
        bounds: Optional lower and upper bounds used for simulation.
    """

    id: str
    timestamp: EventTimestamp

    bounds: tuple[Second, Second] | None = None

    def __post_init__(self) -> None:
        if self.bounds is None:
            object.__setattr__(self, "bounds", (self.timestamp.earliest, self.timestamp.latest))


@dataclass(frozen=True, slots=True)
class SimulatedEvent:
    """Result of propagating a scheduled event.

    Attributes:
        pmf: Distribution of simulated event times.
        underflow: Probability mass below the lower bound.
        overflow: Probability mass above the upper bound.
    """

    pmf: DiscretePMF
    underflow: Probability
    overflow: Probability


@dataclass(frozen=True, slots=True)
class AnalyticContext:
    """Container describing the analytic propagation network.

    Attributes:
        events: Immutable sequence of scheduled events.
        activities: Mapping from (source, target) node pairs to analytic edges.
        precedence_list: List of ``(target, predecessors)`` tuples.
        max_delay: Global cap on allowed propagation delay.
        step_size: Discrete time step shared by all distributions.
    """

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
            edge.pmf.validate_alignment(self.step_size)
