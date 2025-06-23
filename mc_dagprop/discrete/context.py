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
    """Policy for mass falling below the lower bound.

    ``TRUNCATE`` assigns it to the bound value, ``REMOVE`` drops it entirely and
    ``REDISTRIBUTE`` spreads it over the remaining probabilities.
    """

    TRUNCATE = 1
    REMOVE = 2
    REDISTRIBUTE = 3


@unique
class OverflowRule(IntEnum):
    """Policy for mass exceeding the upper bound.

    ``TRUNCATE`` moves the excess to the bound, ``REMOVE`` discards it and
    ``REDISTRIBUTE`` allocates it proportionally over the retained range.
    """

    TRUNCATE = 1
    REMOVE = 2
    REDISTRIBUTE = 3


def validate_context(context: AnalyticContext) -> None:
    """Validate that ``context`` is structurally correct.

    Checks scheduled event bounds, validates activity indices and common step
    size, and ensures the precedence list is free of cycles.
    """

    n_events = len(context.events)

    if context.step_size <= 0.0:
        raise ValueError("step_size must be positive")

    # Validate scheduled events
    for i, ev in enumerate(context.events):
        ts = ev.timestamp
        if ts.earliest > ts.latest:
            raise ValueError(f"event {i} has earliest > latest")
        if not (ts.earliest <= ts.actual <= ts.latest):
            raise ValueError(f"event {i} actual time outside bounds")

    # Validate activities and PMFs
    for (src, dst), (edge_idx, edge) in context.activities.items():
        if not (0 <= src < n_events and 0 <= dst < n_events):
            raise ValueError(f"activity {(src, dst)} references invalid node")
        edge.pmf.validate()
        if not np.isclose(edge.pmf.step, context.step_size):
            raise ValueError(
                f"edge {(src, dst)} step {edge.pmf.step} does not match context step size {context.step_size}"
            )
        edge.pmf.validate_alignment(context.step_size)

    # Validate precedence list and build topology for cycle check
    from collections import deque

    adjacency: list[list[int]] = [[] for _ in range(n_events)]
    indegree = [0] * n_events

    for target, preds in context.precedence_list:
        if not (0 <= target < n_events):
            raise ValueError(f"target index {target} out of range")
        for src, link in preds:
            if not (0 <= src < n_events):
                raise ValueError(f"predecessor index {src} out of range")
            edge = context.activities.get((src, target))
            if edge is None:
                raise ValueError(f"missing activity for {(src, target)}")
            if edge[0] != link:
                raise ValueError(
                    f"edge index {link} for {(src, target)} does not match context mapping {edge[0]}"
                )
            adjacency[src].append(target)
            indegree[target] += 1

    # Topological check for cycles
    q: deque[int] = deque(i for i, deg in enumerate(indegree) if deg == 0)
    visited = 0
    while q:
        node = q.popleft()
        visited += 1
        for dst in adjacency[node]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                q.append(dst)

    if visited != n_events:
        raise ValueError("precedence list contains a cycle")
