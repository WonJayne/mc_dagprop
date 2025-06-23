from __future__ import annotations

from collections import deque
from typing import List

from .context import AnalyticContext
from .pmf import DiscretePMF


class DiscreteSimulator:
    """Propagate discrete PMFs through a DAG."""

    def __init__(self, context: AnalyticContext):
        self.context = context
        self.context.validate()
        self._build_topology()

    def _build_topology(self) -> None:
        event_count = len(self.context.events)
        adjacency = [[] for _ in range(event_count)]
        indegree = [0] * event_count
        preds_by_target = [None] * event_count
        for tgt, preds in self.context.precedence_list:
            preds_by_target[tgt] = preds
            indegree[tgt] = len(preds)
            for src, _ in preds:
                adjacency[src].append(tgt)
        order: List[int] = []
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
        self._preds_by_target = preds_by_target
        self.order = order

    def run(self) -> List[DiscretePMF]:
        event_pmfs: List[DiscretePMF] = [None] * len(self.context.events)  # type: ignore
        for idx in self.order:
            ev = self.context.events[idx]
            base = DiscretePMF.delta(ev.timestamp.earliest)
            preds = self._preds_by_target[idx]
            if preds is None:
                event_pmfs[idx] = base
                continue
            cur = None
            for src, link in preds:
                edge_pmf = self.context.activities[(src, idx)][1].pmf
                candidate = event_pmfs[src].convolve(edge_pmf)
                cur = candidate if cur is None else cur.maximum(candidate)
            event_pmfs[idx] = cur if cur is not None else base
        return event_pmfs
