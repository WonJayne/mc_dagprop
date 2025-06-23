from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .context import AnalyticContext, Pred
from .pmf import DiscretePMF

@dataclass(frozen=True, slots=True)
class DiscreteSimulator:
    """Propagate discrete PMFs through a DAG."""

    context: AnalyticContext
    _preds_by_target: list[tuple[Pred, ...] | None] = field(init=False, repr=False)
    order: list[int] = field(init=False, repr=False)
    underflow: list[float] = field(init=False, repr=False)
    overflow: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.context.validate()
        object.__setattr__(self, "underflow", [])
        object.__setattr__(self, "overflow", [])
        self._build_topology()

    def _build_topology(self) -> None:
        # TODO: Move this on a factory or a separate method, that reutrns a DiscreteSimulator
        event_count = len(self.context.events)
        adjacency = [[] for _ in range(event_count)]
        indegree = [0] * event_count
        preds_by_target = [None] * event_count
        for target, predecessor in self.context.precedence_list:
            preds_by_target[target] = predecessor
            indegree[target] = len(predecessor)
            for src, _ in predecessor:
                adjacency[src].append(target)
        order: list[int] = []
        q = deque(i for i, deg in enumerate(indegree) if deg == 0)
        while q:
            n = q.popleft()
            order.append(n)
            for dst in adjacency[n]:
                indegree[dst] -= 1
                if indegree[dst] == 0:
                    q.append(dst)
        if len(order) != event_count:
            raise RuntimeError("Invalid DAG: cycle detected")
        object.__setattr__(self, "_preds_by_target", preds_by_target)
        object.__setattr__(self, "order", order)

    def run(self) -> tuple[DiscretePMF, ...]:
        # FIXME: here, I would expect to get a tuple of simulated Events, each with a PMF.

        n_events = len(self.context.events)
        event_pmfs: list[DiscretePMF] = [None] * n_events  # type: ignore
        under: list[float] = [0.0] * n_events
        over: list[float] = [0.0] * n_events
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
                    candidate = event_pmfs[src].convolve(edge_pmf)
                    cur = candidate if cur is None else cur.maximum(candidate)
                pmf = cur if cur is not None else base
            lb, ub = (
                ev.bounds
                if ev.bounds is not None
                else (ev.timestamp.earliest, ev.timestamp.latest)
            )
            pmf, u, o = pmf.truncate(lb, ub)
            under[idx] = u
            over[idx] = o
            event_pmfs[idx] = pmf
        # FIXME: Over and underflow should be given to the individual events, as this allows the user to
        #  investigate the propagation of the PMF in more detail.
        object.__setattr__(self, "underflow", under)
        object.__setattr__(self, "overflow", over)
        return event_pmfs
