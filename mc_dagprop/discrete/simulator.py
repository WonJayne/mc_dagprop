from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import cast

from .context import AnalyticContext, Pred, SimulatedEvent
from .pmf import (
    DiscretePMF,
    Probability,
    UnderflowRule,
    OverflowRule,
    apply_bounds,
)


def build_topology(
    context: AnalyticContext,
) -> tuple[list[tuple[Pred, ...] | None], list[int]]:
    """Return predecessor mapping and topological order for ``context``."""

    event_count = len(context.events)
    adjacency: list[list[int]] = [[] for _ in range(event_count)]
    indegree = [0] * event_count
    preds_by_target: list[tuple[Pred, ...] | None] = [None] * event_count

    for target, preds in context.precedence_list:
        preds_by_target[target] = preds
        indegree[target] = len(preds)
        for src, _ in preds:
            adjacency[src].append(target)

    order: list[int] = []
    q: deque[int] = deque(i for i, deg in enumerate(indegree) if deg == 0)

    while q:
        node = q.popleft()
        order.append(node)
        for dst in adjacency[node]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                q.append(dst)

    if len(order) != event_count:
        raise RuntimeError("Invalid DAG: cycle detected")

    return preds_by_target, order


def create_discrete_simulator(
    context: AnalyticContext,
    *,
    underflow_rule: UnderflowRule = UnderflowRule.TRUNCATE,
    overflow_rule: OverflowRule = OverflowRule.TRUNCATE,
) -> "DiscreteSimulator":
    """Return a :class:`DiscreteSimulator` with topology built for ``context``."""

    context.validate()
    preds, order = build_topology(context)
    return DiscreteSimulator(
        context=context,
        _preds_by_target=preds,
        order=order,
        underflow_rule=underflow_rule,
        overflow_rule=overflow_rule,
    )

@dataclass(frozen=True, slots=True)
class DiscreteSimulator:
    """Propagate discrete PMFs through a DAG.

    Probability mass outside an event's bounds can either be truncated to the
    nearest bound or removed entirely. The behaviour is controlled via the
    ``underflow_rule`` and ``overflow_rule`` attributes.
    """

    context: AnalyticContext
    _preds_by_target: list[tuple[Pred, ...] | None]
    order: list[int]
    underflow_rule: UnderflowRule = UnderflowRule.TRUNCATE
    overflow_rule: OverflowRule = OverflowRule.TRUNCATE

    def run(self) -> tuple[SimulatedEvent, ...]:
        """Propagate events through the DAG to compute node PMFs.

        Each node's distribution is derived from its predecessors and the result
        is returned as a tuple of :class:`SimulatedEvent` objects in original
        order. Nodes without incoming edges are deterministic and their PMF
        collapses to a delta at the event's earliest timestamp. Probability mass
        removed by ``apply_bounds`` is recorded per event.
        """
        n_events = len(self.context.events)
        # NOTE[codex]: We need index-based lookup for predecessors. Using a
        # simple append-only list would break because event indices are not
        # guaranteed to match the processing order.
        events: dict[int, SimulatedEvent] = {}
        for idx in self.order:
            ev = self.context.events[idx]
            base = DiscretePMF.delta(ev.timestamp.earliest)
            preds = self._preds_by_target[idx]
            if preds is None:
                pmf = base
            else:
                cur = None
                for src, link in preds:
                    edge_pmf = self.context.activities[(src, idx)][1].pmf
                    candidate = events[src].pmf.convolve(edge_pmf)
                    cur = candidate if cur is None else cur.maximum(candidate)
                pmf = cur if cur is not None else base
            lb, ub = ev.bounds
            pmf, u, o = apply_bounds(
                pmf,
                lb,
                ub,
                underflow_rule=self.underflow_rule,
                overflow_rule=self.overflow_rule,
            )
            events[idx] = SimulatedEvent(pmf, u, o)

        return tuple(events[i] for i in range(n_events))
