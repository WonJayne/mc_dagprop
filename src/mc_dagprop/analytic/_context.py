from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, unique

import numpy as np
from mc_dagprop import Event
from mc_dagprop.types import ActivityIndex, EventIndex, ProbabilityMass, Second

from ._pmf import DiscretePMF

PredecessorTuple = tuple[EventIndex, ActivityIndex]


@dataclass(frozen=True, slots=True)
class AnalyticActivity:
    """Edge with an associated delay distribution.

    Attributes:
        pmf: Probability mass function describing the delay on this edge.
    """

    idx: ActivityIndex
    pmf: DiscretePMF


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
        step: Discrete time step shared by all distributions.
    """

    events: tuple[Event, ...]
    activities: dict[tuple[EventIndex, EventIndex], tuple[ActivityIndex, AnalyticActivity]]
    precedence_list: tuple[tuple[EventIndex, tuple[PredecessorTuple, ...]], ...]
    step: Second
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

    if context.step <= 0.0:
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
        if not np.isclose(edge.pmf.step, context.step):
            raise ValueError(f"edge {(src, dst)} step {edge.pmf.step} does not match context step size {context.step}")
        edge.pmf.validate_alignment(context.step)
        if not np.isclose(edge.pmf.total_mass, 1.0):
            raise ValueError(f"activity {(src, dst)} PMF does not sum to 1, got {edge.pmf.total_mass}")

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
                raise ValueError(f"edge index {link} for {(src, target)} does not match context mapping {edge[0]}")
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
