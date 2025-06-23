from __future__ import annotations

from dataclasses import dataclass
from enum import unique, IntEnum

import numpy as np

from mc_dagprop import EventTimestamp
from . import Second, ProbabilityMass, NodeIndex, EdgeIndex
from .pmf import DiscretePMF

PredecessorTuple = tuple[NodeIndex, EdgeIndex]


@dataclass(frozen=True, slots=True)
class AnalyticEdge:
    """Edge with an associated delay distribution.

    Attributes:
        pmf: Probability mass function describing the delay on this edge.
    """

    idx: EdgeIndex
    pmf: DiscretePMF


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    """Timing information for a planned event.

    Attributes:
        id: Unique identifier of the event.
        timestamp: Earliest, latest and nominal time bounds.
    """

    id: str
    timestamp: EventTimestamp

    @property
    def bounds(self) -> tuple[Second, Second]:
        """Return the lower and upper bounds of the event."""
        return self.timestamp.earliest, self.timestamp.latest


@dataclass(frozen=True, slots=True)
class SimulatedEvent:
    """Result of propagating a scheduled event.

    Attributes:
        pmf: Distribution of simulated event times.
        underflow: Probability mass below the lower bound.
        overflow: Probability mass above the upper bound.
    """

    pmf: DiscretePMF
    underflow: ProbabilityMass
    overflow: ProbabilityMass


@dataclass(frozen=True, slots=True)
class AnalyticContext:
    """Container describing the analytic propagation network.

    Attributes:
        events: Immutable sequence of scheduled events.
        activities: Mapping from (source, target) node pairs to analytic edges.
        precedence_list: List of ``(target, predecessors)`` tuples.
        step_size: Discrete time step shared by all distributions.
    """

    events: tuple[ScheduledEvent, ...]
    activities: dict[tuple[NodeIndex, NodeIndex], tuple[EdgeIndex, AnalyticEdge]]
    precedence_list: tuple[tuple[NodeIndex, tuple[PredecessorTuple, ...]], ...]
    step_size: Second
    underflow_rule: UnderflowRule
    overflow_rule: OverflowRule


@unique
class UnderflowRule(IntEnum):
    """How to handle probability mass below a lower bound."""

    # TODO: Explain meaning of each rule better

    TRUNCATE = 1
    REMOVE = 2
    REDISTRIBUTE = 3


@unique
class OverflowRule(IntEnum):
    """How to handle probability mass above an upper bound."""

    # TODO: Explain meaning of each rule better

    TRUNCATE = 1
    REMOVE = 2
    REDISTRIBUTE = 3


def validate_context(context: AnalyticContext) -> None:
    """Validate the AnalyticContext for correctness.

    This function checks that the context is well-formed, including:
    - All events have valid bounds.
    - All events have a valid timestamp.
    - All activities have valid PMFs.
    - Predecessors are correctly defined.
    - The precedence list does not contain cycles.
    """

    # TODO: Implement validation for events and their bounds.
    # TODO: Implement validation for the precedence list to ensure no cycles exist.
    # TODO: Implement validation for the activities to ensure all PMFs are well-defined.

    if len(context.activities) > 0:
        first_edge = next(iter(context.activities.values()))[1]
        base_step = first_edge.pmf.step
        for _, edge in context.activities.values():
            if not np.isclose(base_step, edge.pmf.step):
                raise ValueError(f"edge PMF step {edge.pmf.step} does not match context step size {base_step}")
            edge.pmf.validate_alignment(base_step)
