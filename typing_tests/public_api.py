"""Static consumer of the supported root-package API."""

from __future__ import annotations

from mc_dagprop import (
    Activity,
    AnalyticPropagator,
    DelayFamilyRegistry,
    Event,
    EventTimestamp,
    MonteCarloPropagator,
    OverflowRule,
    PropagationContext,
    SimResult,
    SimulatedEvent,
    UnderflowRule,
    analytic_from_context,
    monte_carlo_from_context,
)
from mc_dagprop.types import ActivityIndex, ActivityType, EventId, EventIndex, ProbabilityMass, Second

event_id: EventId = EventId("root")
event_index: EventIndex = EventIndex(0)
activity_index: ActivityIndex = ActivityIndex(0)
activity_type: ActivityType = ActivityType(1)
duration: Second = Second(10.0)
unit_mass: ProbabilityMass = ProbabilityMass(1.0)

events = (Event(event_id, EventTimestamp(0.0, 100.0, 0.0)), Event("child", EventTimestamp(0.0, 100.0, 0.0)))
activities = {(event_index, 1): Activity(activity_index, duration, activity_type)}
precedence = ((1, ((0, 0),)),)
context = PropagationContext(events, activities, precedence)
registry = DelayFamilyRegistry()
registry.add_empirical(activity_type, [0.0, 5.0], [unit_mass, 0.0])

monte_carlo: MonteCarloPropagator = MonteCarloPropagator.from_context(context, registry)
analytic: AnalyticPropagator = AnalyticPropagator.from_context(
    context, registry, step=1, underflow_rule=UnderflowRule.TRUNCATE, overflow_rule=OverflowRule.TRUNCATE
)
monte_carlo_factory: MonteCarloPropagator = monte_carlo_from_context(context, registry)
analytic_factory: AnalyticPropagator = analytic_from_context(
    context, registry, step=1, underflow_rule=UnderflowRule.TRUNCATE, overflow_rule=OverflowRule.TRUNCATE
)

monte_carlo_result: SimResult = monte_carlo.run(seed=123)
monte_carlo_batch: list[SimResult] = monte_carlo.run_many([123, 456])
analytic_result: tuple[SimulatedEvent, ...] = analytic.run()
monte_carlo_node_count: int = monte_carlo_factory.node_count()
monte_carlo_activity_count: int = monte_carlo_factory.activity_count()
