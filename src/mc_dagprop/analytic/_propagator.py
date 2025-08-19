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
        events: list[SimulatedEvent | None] = [None] * n_events
        for this_node in self._topological_node_order:
            ev = self.context.events[this_node]
            predecessors = self._predecessors_by_target[this_node]
            is_origin = predecessors is None or len(predecessors) == 0
            if is_origin:
                base = DiscretePMF.delta(
                    round(ev.timestamp.earliest / self.context.step) * self.context.step, self.context.step
                )
                assert np.isclose(base.total_mass, 1.0)
                events[this_node] = SimulatedEvent(base, ProbabilityMass(0.0), ProbabilityMass(0.0))
                continue

            to_combine = []
            for src, _ in predecessors:
                pred = events[src].pmf
                act = self.context.activities[(src, this_node)][1].pmf
                conv = pred.convolve(act)

                # Expect: mass(conv) ≈ mass(pred) * mass(act)
                m_pred, m_act, m_conv = pred.total_mass, act.total_mass, conv.total_mass
                if not np.isclose(m_conv, m_pred * m_act, rtol=1e-12, atol=1e-15):
                    print(
                        f"[LOSS in CONVOLVE] node={this_node} src={src} "
                        f"pred={m_pred:.17g} act={m_act:.17g} conv={m_conv:.17g}"
                    )
                to_combine.append(conv)

            resulting_pmf = to_combine[0]
            if len(to_combine) > 1:
                before = [p.total_mass for p in to_combine]
                for next_pmf in to_combine[1:]:
                    resulting_pmf = resulting_pmf.maximum(next_pmf)
                after = resulting_pmf.total_mass
                if not np.isclose(after, 1.0, rtol=1e-12, atol=1e-15) and all(np.isclose(b, 1.0) for b in before):
                    print(f"[LOSS in MAXIMUM] node={this_node} inputs={before} after={after:.17g}")

            lb, ub = np.round(ev.timestamp.earliest), np.round(ev.timestamp.latest)
            events[this_node] = self._convert_to_simulated_event(resulting_pmf, lb, ub)

            # Sanity: _convert shouldn't change mass when under/overflow=0
            assert np.isclose(
                events[this_node].pmf.total_mass + events[this_node].underflow + events[this_node].overflow,
                resulting_pmf.total_mass,
                rtol=1e-12,
                atol=1e-15,
            )

        assert all(events[i] is not None for i in range(n_events)), "Not all events were processed, check context"
        return tuple(events[i] for i in range(n_events))

    def _convert_to_simulated_event(self, pmf: DiscretePMF, min_value: int, max_value: int) -> SimulatedEvent:
        """Clip pmf to [min_value, max_value] and mass-correct depending on flow rules.

        Invariant enforced (up to numerical tolerance):
            clipped.pmf.total_mass + under_mass + over_mass == 1.0
        """
        if min_value > max_value:
            raise ValueError("min_value must not exceed max_value")

        vals = pmf.values
        probs = pmf.probabilities
        if vals.size == 0:
            raise ValueError("PMF must not be empty")

        # Partition mass by bound
        under_mask = vals < min_value
        over_mask = vals > max_value
        keep_mask = ~(under_mask | over_mask)

        under_mass = ProbabilityMass(probs[under_mask].sum())
        over_mass = ProbabilityMass(probs[over_mask].sum())

        new_vals = vals[keep_mask]
        new_probs = probs[keep_mask].copy()

        # ---- Handle UNDERFLOW rule
        to_redistribute_under = ProbabilityMass(0.0)
        if self.underflow_rule == UnderflowRule.TRUNCATE and under_mass > 0.0:
            # push underflow onto the lower bound bin
            if new_vals.size and np.isclose(new_vals[0], min_value):
                new_probs[0] += under_mass
            elif new_vals.size == 0:
                new_vals = np.array([min_value], dtype=float)
                new_probs = np.array([under_mass], dtype=float)
            else:
                raise ValueError(f"Underflow mass cannot be truncated: no lower-bound bin present. {new_vals=}")
            under_mass = ProbabilityMass(0.0)
        elif self.underflow_rule == UnderflowRule.REDISTRIBUTE and under_mass > 0.0:
            # keep record of mass but reinsert later proportionally
            to_redistribute_under = under_mass
            under_mass = ProbabilityMass(0.0)

        # ---- Handle OVERFLOW rule
        to_redistribute_over = ProbabilityMass(0.0)
        if self.overflow_rule == OverflowRule.TRUNCATE and over_mass > 0.0:
            # push overflow onto the upper bound bin
            if new_vals.size and np.isclose(new_vals[-1], max_value):
                new_probs[-1] += over_mass
            elif new_vals.size == 0:
                new_vals = np.array([max_value], dtype=float)
                new_probs = np.array([over_mass], dtype=float)
            else:
                raise ValueError("Overflow mass cannot be truncated: no upper-bound bin present.")
            over_mass = ProbabilityMass(0.0)
        elif self.overflow_rule == OverflowRule.REDISTRIBUTE and over_mass > 0.0:
            to_redistribute_over = over_mass
            over_mass = ProbabilityMass(0.0)

        # ---- Proportional redistribution (if enabled)
        to_redistribute = (to_redistribute_under + to_redistribute_over)
        base_inside = new_probs.sum()
        if to_redistribute > 0.0:
            if base_inside == 0.0:
                # FIXME: no inside mass to redistribute onto evenly distribute mass?
                anchor = min_value if np.isfinite(min_value) else max_value
                new_vals = np.array([anchor], dtype=float)
                new_probs = np.array([to_redistribute], dtype=float)
            else:
                # proportional to current inside mass
                new_probs = new_probs + to_redistribute * (new_probs / base_inside)

        # ---- Final mass correction: normalize → scale to target_inside
        if new_vals.size == 0:
            raise ValueError(
                f"PMF must not be empty after clipping, {under_mass=}, {over_mass=}, total_in={pmf.probabilities.sum()}"
            )

        # target inside mass = 1 - (currently accounted under/over)
        lost = under_mass + over_mass
        target_inside = max(0.0, 1.0 - lost)  # guard tiny negative from roundoff

        inside_sum = new_probs.sum()
        if inside_sum > 0.0:
            # Step 1: normalize inside to exactly 1.0 (eliminate drift)
            new_probs /= inside_sum
            # Step 2: scale to the desired inside mass
            new_probs *= target_inside
        else:
            # No inside support; ensure that we are consistent with target mass
            # If target_inside > 0, create a delta at nearest feasible bound
            if target_inside > 0.0:
                anchor = np.clip(min_value, min_value, max_value)  # use min bound
                new_vals = np.array([anchor], dtype=float)
                new_probs = np.array([target_inside], dtype=float)

        clipped = DiscretePMF(new_vals, new_probs, step=pmf.step)

        # Invariant sanity-check (tolerant)
        total = (clipped.total_mass + under_mass + over_mass)
        if not np.isclose(total, 1.0, rtol=1e-12, atol=1e-15):
            # Tighten by a last tiny rescale if we’re microscopically off due to casts
            corr = 1.0 / total if total > 0 else 1.0
            new_probs *= corr
            clipped = DiscretePMF(new_vals, new_probs, step=pmf.step)
            total = (clipped.total_mass + under_mass + over_mass)
            assert np.isclose(total, 1.0, rtol=1e-12, atol=1e-15), f"Mass mismatch after correction: {total=}"

        return SimulatedEvent(clipped, under_mass, over_mass)
