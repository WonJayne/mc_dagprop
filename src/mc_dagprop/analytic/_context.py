from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import IntEnum, unique
from types import MappingProxyType
from typing import cast

import numpy as np

from mc_dagprop.monte_carlo import Event
from mc_dagprop.types import ActivityIndex, EventIndex, ProbabilityMass

from ._pmf import DiscretePMF

PredecessorTuple = tuple[EventIndex, ActivityIndex]
StochasticSource = tuple[EventIndex, EventIndex]
_MAX_SIGNED_INDEX = 2**31 - 1


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


def _require_step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"step must be an integer number of seconds, got {value!r}")
    if value <= 0:
        raise ValueError("step_size must be positive")
    return value


def _require_underflow_rule(value: object) -> UnderflowRule:
    if not isinstance(value, UnderflowRule):
        raise TypeError(f"underflow_rule must be an UnderflowRule, got {value!r}")
    return value


def _require_overflow_rule(value: object) -> OverflowRule:
    if not isinstance(value, OverflowRule):
        raise TypeError(f"overflow_rule must be an OverflowRule, got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class AnalyticContext:
    """Container describing the analytic propagation network.

    Attributes:
        events: Immutable sequence of scheduled events.
        activities: Mapping from (source, target) node pairs to analytic edges.
        precedence_list: List of ``(target, predecessors)`` tuples.
        step: Discrete time step shared by all distributions.
        underflow_rule: Rule for mass below event lower bounds.
        overflow_rule: Rule for mass above event upper bounds.
    """

    events: tuple[Event, ...]
    activities: Mapping[tuple[EventIndex, EventIndex], tuple[ActivityIndex, AnalyticActivity]]
    precedence_list: tuple[tuple[EventIndex, tuple[PredecessorTuple, ...]], ...]
    step: int
    underflow_rule: UnderflowRule
    overflow_rule: OverflowRule

    def __post_init__(self) -> None:
        """Snapshot all containers so the validated DAG cannot change later."""
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(
            self,
            "activities",
            MappingProxyType(
                {
                    (source, target): (activity_index, activity)
                    for (source, target), (activity_index, activity) in self.activities.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "precedence_list",
            tuple((target, tuple(predecessors)) for target, predecessors in self.precedence_list),
        )


def validate_context(context: AnalyticContext) -> None:
    """Validate that ``context`` is structurally correct.

    Checks scheduled event bounds, validates activity indices and common step
    size, and ensures the precedence list is free of cycles.
    """

    n_events = len(context.events)
    if n_events == 0:
        raise ValueError("analytic context must contain at least one event")

    _ = _require_step(context.step)
    _ = _require_underflow_rule(context.underflow_rule)
    _ = _require_overflow_rule(context.overflow_rule)

    def require_index(value: object, label: str) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{label} must be an integer, got {value!r}")
        if value > _MAX_SIGNED_INDEX:
            raise ValueError(f"{label} must not exceed {_MAX_SIGNED_INDEX}")

    def require_grid_aligned(value: float, label: str) -> None:
        if not np.isfinite(value):
            raise ValueError(f"{label} must be finite")
        remainder = cast(np.float64, np.mod(value, context.step))
        if not np.isclose(remainder, 0.0, rtol=0.0, atol=1e-9):
            raise ValueError(f"{label}={value!r} is not aligned to analytic step {context.step}")

    # Validate scheduled events
    for i, ev in enumerate(context.events):
        ts = ev.timestamp
        if ts.earliest > ts.latest:
            raise ValueError(f"event {i} has earliest > latest")
        if not (ts.earliest <= ts.actual <= ts.latest):
            raise ValueError(f"event {i} actual time outside bounds")
        require_grid_aligned(ts.earliest, f"event {i} earliest")
        require_grid_aligned(ts.latest, f"event {i} latest")
        require_grid_aligned(ts.actual, f"event {i} actual")

    # Validate activities and PMFs
    seen_activity_indices: set[int] = set()
    for (src, dst), (edge_idx, edge) in context.activities.items():
        require_index(src, "activity source index")
        require_index(dst, "activity target index")
        require_index(edge_idx, "activity index")
        require_index(edge.idx, "analytic activity index")
        if not (0 <= src < n_events and 0 <= dst < n_events):
            raise ValueError(f"activity {(src, dst)} references invalid node")
        if edge_idx < 0:
            raise ValueError(f"activity index {edge_idx} must be non-negative")
        if edge.idx != edge_idx:
            raise ValueError(f"activity index {edge.idx} for {(src, dst)} does not match context mapping {edge_idx}")
        if edge_idx in seen_activity_indices:
            raise ValueError(f"duplicate activity index {edge_idx}")
        seen_activity_indices.add(edge_idx)
        edge.pmf.validate()
        if np.any(edge.pmf.values < 0.0):
            raise ValueError(f"activity {(src, dst)} PMF support must be non-negative")
        if edge.pmf.step != context.step:
            raise ValueError(f"edge {(src, dst)} step {edge.pmf.step} does not match context step size {context.step}")
        edge.pmf.validate_alignment(context.step)
        if not np.isclose(edge.pmf.total_mass, 1.0, rtol=1e-12, atol=1e-15):
            raise ValueError(f"activity {(src, dst)} PMF does not sum to 1, got {edge.pmf.total_mass}")
    if seen_activity_indices != set(range(len(seen_activity_indices))):
        raise ValueError("activity indices must be contiguous from 0 to n-1")

    # Validate precedence list and build topology for cycle check
    adjacency: list[list[int]] = [[] for _ in range(n_events)]
    indegree = [0] * n_events
    seen_targets: set[int] = set()
    referenced_activities: set[tuple[int, int]] = set()

    for target, preds in context.precedence_list:
        require_index(target, "precedence target index")
        if not (0 <= target < n_events):
            raise ValueError(f"target index {target} out of range")
        if target in seen_targets:
            raise ValueError(f"duplicate precedence entry for target {target}")
        seen_targets.add(target)
        seen_predecessor_sources: set[int] = set()
        for src, link in preds:
            require_index(src, "predecessor source index")
            require_index(link, "predecessor activity index")
            if not (0 <= src < n_events):
                raise ValueError(f"predecessor index {src} out of range")
            if src in seen_predecessor_sources:
                raise ValueError(f"duplicate predecessor source {src} for target {target}")
            seen_predecessor_sources.add(src)
            edge = context.activities.get((src, target))
            if edge is None:
                raise ValueError(f"missing activity for {(src, target)}")
            if edge[0] != link:
                raise ValueError(f"edge index {link} for {(src, target)} does not match context mapping {edge[0]}")
            referenced_activities.add((src, target))
            adjacency[src].append(target)
            indegree[target] += 1

    unused_activities = set(context.activities).difference(referenced_activities)
    if unused_activities:
        unused_text = ", ".join(str(edge) for edge in sorted(unused_activities))
        raise ValueError(f"activities missing from precedence list: {unused_text}")

    # Topological check for cycles
    q: deque[int] = deque(i for i, deg in enumerate(indegree) if deg == 0)
    topological_order: list[int] = []
    while q:
        node = q.popleft()
        topological_order.append(node)
        for dst in adjacency[node]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                q.append(dst)

    if len(topological_order) != n_events:
        raise ValueError("precedence list contains a cycle")

    _validate_exact_equivalence_domain(context, topological_order)


def _is_stochastic(pmf: DiscretePMF) -> bool:
    """Return whether ``pmf`` contains more than one possible value."""
    return bool(np.count_nonzero(pmf.probabilities > 0.0) > 1)


def _validate_exact_equivalence_domain(context: AnalyticContext, topological_order: Iterable[int]) -> None:
    """Reject merges that combine dependent stochastic marginal distributions."""
    predecessors_by_target = dict(context.precedence_list)
    stochastic_ancestry: list[frozenset[StochasticSource]] = [frozenset() for _ in context.events]

    for target_index in topological_order:
        target = int(target_index)
        accumulated_sources: set[StochasticSource] = set()
        for source, _ in predecessors_by_target.get(target, ()):
            branch_sources = set(stochastic_ancestry[source])
            activity = context.activities[(source, target)][1]
            if _is_stochastic(activity.pmf):
                branch_sources.add((source, target))
            shared_sources = accumulated_sources.intersection(branch_sources)
            if shared_sources:
                shared_text = ", ".join(f"{src}->{dst}" for src, dst in sorted(shared_sources))
                raise ValueError(
                    f"analytic exactness requires disjoint stochastic ancestry at merge {target}; "
                    f"shared stochastic activities: {shared_text}"
                )
            accumulated_sources.update(branch_sources)
        stochastic_ancestry[target] = frozenset(accumulated_sources)


def validate_exact_equivalence_domain(context: AnalyticContext) -> None:
    """Validate the stochastic-independence domain required for exact propagation.

    ``validate_context`` performs this check automatically. This standalone
    helper is exposed for callers that need to state or inspect the exactness
    contract explicitly after structural validation.
    """
    event_count = len(context.events)
    adjacency: list[list[int]] = [[] for _ in range(event_count)]
    indegree = [0] * event_count
    for target, predecessors in context.precedence_list:
        indegree[target] += len(predecessors)
        for source, _ in predecessors:
            adjacency[source].append(target)
    queue: deque[int] = deque(index for index, degree in enumerate(indegree) if degree == 0)
    topological_order: list[int] = []
    while queue:
        source = queue.popleft()
        topological_order.append(source)
        for target in adjacency[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(topological_order) != event_count:
        raise ValueError("precedence list contains a cycle")
    _validate_exact_equivalence_domain(context, topological_order)
