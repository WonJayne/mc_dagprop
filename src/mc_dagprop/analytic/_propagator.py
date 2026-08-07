from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

import numpy as np
import numpy.typing as npt

from mc_dagprop.monte_carlo import Activity, Event
from mc_dagprop.types import ActivityIndex, EventIndex, Second

from ._context import AnalyticContext, OverflowRule, PredecessorTuple, SimulatedEvent, UnderflowRule, validate_context
from ._pmf import DiscretePMF


class PropagationContextLike(Protocol):
    """Structural shared-context interface consumed by the analytic frontend."""

    @property
    def events(self) -> tuple[Event, ...]: ...

    @property
    def activities(self) -> Mapping[tuple[int, int], Activity]: ...

    @property
    def precedence_list(self) -> tuple[tuple[int, tuple[tuple[int, int], ...]], ...]: ...


class DelayFamilyRegistryLike(Protocol):
    """Structural delay-registry interface consumed by the analytic frontend."""

    def increment_pmf(self, activity: Activity, step: int) -> DiscretePMF: ...


class _AnalyticFactory(Protocol):
    def __call__(
        self,
        context: PropagationContextLike,
        registry: DelayFamilyRegistryLike,
        *,
        step: int,
        overflow_rule: OverflowRule,
        underflow_rule: UnderflowRule,
    ) -> AnalyticPropagator: ...


class _FrontendModule(Protocol):
    analytic_from_context: _AnalyticFactory


_LOGGER = logging.getLogger(__name__)


def _build_topology(
    context: AnalyticContext,
) -> tuple[tuple[tuple[tuple[EventIndex, ActivityIndex], ...] | None, ...], tuple[EventIndex, ...]]:
    """Return predecessor mapping and topological order for ``context``."""

    event_count = len(context.events)
    adjacency: list[list[EventIndex]] = [[] for _ in range(event_count)]
    indegree = [0] * event_count
    preds_by_target: list[tuple[PredecessorTuple, ...] | None] = [None] * event_count

    for target, preds in context.precedence_list:
        preds_by_target[target] = preds
        indegree[target] = len(preds)
        for src, _ in preds:
            adjacency[src].append(target)

    order: list[EventIndex] = []
    q: deque[EventIndex] = deque(int(i) for i, deg in enumerate(indegree) if deg == 0)

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


def create_analytic_propagator(context: AnalyticContext, validate: bool = True) -> AnalyticPropagator:
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

    @classmethod
    def from_context(
        cls,
        context: PropagationContextLike,
        registry: DelayFamilyRegistryLike,
        *,
        step: int,
        overflow_rule: OverflowRule,
        underflow_rule: UnderflowRule,
    ) -> AnalyticPropagator:
        """Create an analytic propagator from the shared logical frontend."""
        frontend = cast(_FrontendModule, cast(object, import_module("mc_dagprop.frontend")))
        factory: _AnalyticFactory = frontend.analytic_from_context
        propagator = factory(context, registry, step=step, overflow_rule=overflow_rule, underflow_rule=underflow_rule)
        if not isinstance(propagator, cls):
            raise TypeError("analytic frontend returned an incompatible propagator")
        return propagator

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
        events: list[SimulatedEvent | None] = [None] * n_events
        for this_node in self._topological_node_order:
            ev = self.context.events[this_node]
            predecessors = self._predecessors_by_target[this_node]
            if not predecessors:
                base = DiscretePMF.delta(ev.timestamp.earliest, self.context.step)
                if not np.isclose(base.total_mass, 1.0):
                    raise RuntimeError("root PMF construction did not produce unit mass")
                events[this_node] = SimulatedEvent(base, 0.0, 0.0)
                continue

            to_combine = []
            for src, _ in predecessors:
                predecessor_event = events[src]
                if predecessor_event is None:
                    raise RuntimeError(f"predecessor event {src} was not processed before event {this_node}")
                pred = predecessor_event.pmf
                act = self.context.activities[(src, this_node)][1].pmf
                conv = pred.convolve(act)

                # Expect: mass(conv) ≈ mass(pred) * mass(act)
                m_pred, m_act, m_conv = pred.total_mass, act.total_mass, conv.total_mass
                if not np.isclose(m_conv, m_pred * m_act, rtol=1e-12, atol=1e-15):
                    _LOGGER.debug(
                        "convolution mass drift: node=%s src=%s pred=%s act=%s conv=%s",
                        this_node,
                        src,
                        m_pred,
                        m_act,
                        m_conv,
                    )
                to_combine.append(conv)

            resulting_pmf = to_combine[0]
            if len(to_combine) > 1:
                before = [p.total_mass for p in to_combine]
                for next_pmf in to_combine[1:]:
                    resulting_pmf = resulting_pmf.maximum(next_pmf)
                after = resulting_pmf.total_mass
                if not np.isclose(after, 1.0, rtol=1e-12, atol=1e-15) and all(np.isclose(b, 1.0) for b in before):
                    _LOGGER.debug("maximum mass drift: node=%s inputs=%s after=%s", this_node, before, after)

            lower_bound, upper_bound = self._event_bounds(ev.timestamp.earliest, ev.timestamp.latest)
            simulated_event = self._convert_to_simulated_event(resulting_pmf, lower_bound, upper_bound)
            events[this_node] = simulated_event

            accounted_mass = simulated_event.pmf.total_mass + simulated_event.underflow + simulated_event.overflow
            if not np.isclose(accounted_mass, resulting_pmf.total_mass, rtol=1e-12, atol=1e-15):
                raise RuntimeError(
                    f"mass mismatch after clipping for event {this_node}: "
                    f"accounted={accounted_mass}, incoming={resulting_pmf.total_mass}"
                )

        if not all(events[i] is not None for i in range(n_events)):
            raise RuntimeError("Not all events were processed, check context")
        return tuple(event for event in events if event is not None)

    def _event_bounds(self, earliest: Second, latest: Second) -> tuple[int, int]:
        """Return rounded event bounds from the scheduled event window."""

        return round(earliest), round(latest)

    def _convert_to_simulated_event(self, pmf: DiscretePMF, min_value: int, max_value: int) -> SimulatedEvent:
        """Clip pmf to [min_value, max_value] and mass-correct depending on flow rules.

        Invariant enforced (up to numerical tolerance):
            clipped.pmf.total_mass + under_mass + over_mass == incoming mass.
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

        under_mass = float(cast(np.float64, probs[under_mask].sum()))
        over_mass = float(cast(np.float64, probs[over_mask].sum()))

        new_vals = vals[keep_mask]
        new_probs = probs[keep_mask].copy()

        # ---- Handle UNDERFLOW rule
        to_redistribute_under = 0.0
        if self.underflow_rule == UnderflowRule.TRUNCATE and under_mass > 0.0:
            # push underflow onto the lower bound bin
            if new_vals.size and np.isclose(cast(np.float64, new_vals[0]), min_value, rtol=0.0, atol=1.0e-9):
                new_probs[0] += under_mass
            elif new_vals.size == 0:
                new_vals = np.array([min_value], dtype=np.float64)
                new_probs = np.array([under_mass], dtype=np.float64)
            else:
                new_vals = np.insert(new_vals, 0, float(min_value))
                new_probs = np.insert(new_probs, 0, float(under_mass))
            under_mass = 0.0
        elif self.underflow_rule == UnderflowRule.REDISTRIBUTE and under_mass > 0.0:
            # keep record of mass but reinsert later proportionally
            to_redistribute_under = under_mass
            under_mass = 0.0

        # ---- Handle OVERFLOW rule
        to_redistribute_over = 0.0
        if self.overflow_rule == OverflowRule.TRUNCATE and over_mass > 0.0:
            # push overflow onto the upper bound bin
            if new_vals.size and np.isclose(cast(np.float64, new_vals[-1]), max_value, rtol=0.0, atol=1.0e-9):
                new_probs[-1] += over_mass
            elif new_vals.size == 0:
                new_vals = np.array([max_value], dtype=np.float64)
                new_probs = np.array([over_mass], dtype=np.float64)
            else:
                new_vals = cast(npt.NDArray[np.float64], np.append(new_vals, float(max_value)))
                new_probs = cast(npt.NDArray[np.float64], np.append(new_probs, float(over_mass)))
            over_mass = 0.0
        elif self.overflow_rule == OverflowRule.REDISTRIBUTE and over_mass > 0.0:
            to_redistribute_over = over_mass
            over_mass = 0.0

        # ---- Proportional redistribution (if enabled)
        to_redistribute = to_redistribute_under + to_redistribute_over
        base_inside = float(cast(np.float64, new_probs.sum()))
        if to_redistribute > 0.0:
            if base_inside == 0.0:
                # Preserve the side from which each redistributed mass originated.
                anchor_masses: dict[int, float] = {}
                if to_redistribute_under > 0.0:
                    anchor_masses[min_value] = to_redistribute_under
                if to_redistribute_over > 0.0:
                    anchor_masses[max_value] = anchor_masses.get(max_value, 0.0) + to_redistribute_over
                new_vals = np.array(tuple(anchor_masses), dtype=np.float64)
                new_probs = np.array(tuple(anchor_masses.values()), dtype=np.float64)
            else:
                # proportional to current inside mass
                new_probs = new_probs + to_redistribute * (new_probs / base_inside)

        if new_vals.size == 0:
            # REMOVE may legitimately eliminate all probability mass. Keep
            # zero-probability bound anchors so the sub-PMF remains composable.
            anchors: list[int] = []
            if under_mass > 0.0:
                anchors.append(min_value)
            if over_mass > 0.0 and max_value not in anchors:
                anchors.append(max_value)
            if not anchors:
                anchors.append(min_value)
            new_vals = np.array(anchors, dtype=np.float64)
            new_probs = np.zeros(len(anchors), dtype=np.float64)

        incoming_mass = float(pmf.total_mass)
        accounted_mass = float(cast(np.float64, new_probs.sum())) + under_mass + over_mass
        if not np.isclose(accounted_mass, incoming_mass, rtol=1e-12, atol=1e-15):
            raise RuntimeError(
                f"clipping mass mismatch: accounted={accounted_mass}, incoming={incoming_mass}, "
                f"underflow={under_mass}, overflow={over_mass}"
            )
        retained_mass = float(cast(np.float64, new_probs.sum()))
        clipped = DiscretePMF(
            new_vals,
            new_probs,
            step=pmf.step,
            allow_subprobability=not np.isclose(retained_mass, 1.0, rtol=1e-12, atol=1e-15),
        )
        total = clipped.total_mass + under_mass + over_mass
        if not np.isclose(total, incoming_mass, rtol=1e-12, atol=1e-15):
            raise RuntimeError(f"Mass mismatch after clipping: total={total}, incoming={incoming_mass}")

        return SimulatedEvent(clipped, under_mass, over_mass)
