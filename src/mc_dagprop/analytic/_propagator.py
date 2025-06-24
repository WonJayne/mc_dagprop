from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from mc_dagprop.types import ActivityIndex, EventIndex, ProbabilityMass, Second

from . import OverflowRule, UnderflowRule
from ._context import AnalyticContext, PredecessorTuple, SimulatedEvent, validate_context
from ._pmf import DiscretePMF


def _build_topology(
    context: AnalyticContext,
) -> tuple[tuple[tuple[tuple[EventIndex, ActivityIndex], ...] | None, ...], tuple[EventIndex, ...]]:
    """Return predecessor mapping and topological order for ``context``."""

    event_count = len(context.events)
    adjacency: list[list[int]] = [[] for _ in range(event_count)]
    indegree = [0] * event_count
    preds_by_target: list[tuple[PredecessorTuple, ...] | None] = [None] * event_count

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

    return tuple(preds_by_target), tuple(order)


def create_analytic_propagator(context: AnalyticContext, validate: bool = True) -> "AnalyticPropagator":
    """Return an :class:`AnalyticPropagator` with topology built for ``context``.

    Parameters
    ----------
    context:
        Analytic description of the DAG to simulate.
        How to handle probability mass outside event bounds.
    validate:
        When ``True`` (default), ``context.validate()`` is invoked before
        creating the simulator. Set to ``False`` if the caller guarantees that
        the context is already valid.
    """

    if validate:
        validate_context(context)
    predecessors, order = _build_topology(context)
    return AnalyticPropagator(context=context, _predecessors_by_target=predecessors, _topological_node_order=order)


@dataclass(frozen=True, slots=True)
class AnalyticPropagator:
    """Propagate discrete PMFs through a DAG.

    Probability mass outside an event's bounds can either be truncated to the
    nearest bound or removed entirely. The behaviour is controlled via the
    ``underflow_rule`` and ``overflow_rule`` attributes.
    """

    context: AnalyticContext
    _predecessors_by_target: tuple[tuple[PredecessorTuple, ...] | None, ...]
    _topological_node_order: tuple[EventIndex, ...]

    @property
    def underflow_rule(self) -> UnderflowRule:
        return self.context.underflow_rule

    @property
    def overflow_rule(self) -> OverflowRule:
        return self.context.overflow_rule

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
        for node_index in self._topological_node_order:
            ev = self.context.events[node_index]
            predecessors = self._predecessors_by_target[node_index]
            is_origin = predecessors is None
            if is_origin:
                events[node_index] = SimulatedEvent(
                    DiscretePMF.delta(ev.timestamp.earliest, self.context.step),
                    ProbabilityMass(0.0),
                    ProbabilityMass(0.0),
                )
                continue
            # ``validate_context`` guarantees that ``predecessors`` is non-empty for non-origin nodes.

            to_combine = []
            for i, (src, link) in enumerate(predecessors):
                edge_pmf = self.context.activities[(src, node_index)][1].pmf
                candidate = events[src].pmf.convolve(edge_pmf)
                to_combine.append(candidate)

            resulting_pmf = to_combine[0]
            if len(to_combine) > 1:
                for next_pmf in to_combine[1:]:
                    resulting_pmf = resulting_pmf.maximum(next_pmf)

            lb, ub = ev.timestamp.earliest, ev.timestamp.latest
            events[node_index] = self._convert_to_simulated_event(resulting_pmf, lb, ub)

        return tuple(events[i] for i in range(n_events))

    def _convert_to_simulated_event(self, pmf: DiscretePMF, min_value: Second, max_value: Second) -> SimulatedEvent:
        """Clip ``pmf`` to ``[min_value, max_value]`` according to the given rules."""

        if min_value > max_value:
            raise ValueError("min_value must not exceed max_value")
        vals = pmf.values
        probs = pmf.probabilities

        under_mask = vals < min_value
        over_mask = vals > max_value
        under_mass = ProbabilityMass(probs[under_mask].sum())
        over_mass = ProbabilityMass(probs[over_mask].sum())
        keep_mask = ~(under_mask | over_mask)

        new_vals = vals[keep_mask]
        new_probs = probs[keep_mask]

        # Move mass below the minimum bound to the bound itself when
        # ``TRUNCATE`` is active and there is mass to relocate.
        should_truncate_underflow = self.underflow_rule is UnderflowRule.TRUNCATE and float(under_mass) > 0.0
        if should_truncate_underflow:
            if new_vals.size and np.isclose(new_vals[0], min_value):
                new_probs[0] += under_mass
            else:
                new_vals = np.insert(new_vals, 0, min_value)
                new_probs = np.insert(new_probs, 0, float(under_mass))
            under_mass = ProbabilityMass(0.0)

        # Move mass above the maximum bound to the bound itself when
        # ``TRUNCATE`` is active and there is mass to relocate.
        should_truncate_overflow = self.overflow_rule is OverflowRule.TRUNCATE and float(over_mass) > 0.0
        if should_truncate_overflow:
            if new_vals.size and np.isclose(new_vals[-1], max_value):
                new_probs[-1] += over_mass
            else:
                new_vals = np.append(new_vals, max_value)
                new_probs = np.append(new_probs, float(over_mass))
            over_mass = ProbabilityMass(0.0)

        # Probability mass removed under ``REDISTRIBUTE`` is reallocated
        # proportionally across the remaining distribution.
        to_add = ProbabilityMass(0.0)
        should_redistribute_underflow = self.underflow_rule is UnderflowRule.REDISTRIBUTE and under_mass > 0.0
        if should_redistribute_underflow:
            to_add = under_mass
            under_mass = ProbabilityMass(0.0)

        should_redistribute_overflow = self.overflow_rule is OverflowRule.REDISTRIBUTE and over_mass > 0.0
        if should_redistribute_overflow:
            to_add += over_mass
            over_mass = ProbabilityMass(0.0)

        # Spread collected underflow/overflow mass according to the
        # relative probabilities of the remaining distribution.
        if to_add > 0.0:
            new_probs = new_probs + to_add * (new_probs / new_probs.sum())

        clipped = DiscretePMF(new_vals, new_probs, step=pmf.step)

        total_mass = float(clipped.probabilities.sum() + under_mass + over_mass)
        if total_mass > 1.0 and not np.isclose(total_mass, 1.0):
            raise ValueError("Total probability mass exceeds 1.0 after clipping")
        return SimulatedEvent(clipped, under_mass, over_mass)
