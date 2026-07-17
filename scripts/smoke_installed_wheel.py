"""Smoke test an installed mc_dagprop wheel from outside the source tree."""
from __future__ import annotations

import mc_dagprop as mc

print(f"mc_dagprop version: {mc.__version__}")

events = (
    mc.Event("root", mc.EventTimestamp(0, 100, 0)),
    mc.Event("child", mc.EventTimestamp(0, 100, 0)),
)
activities = {(0, 1): mc.Activity(0, 10, 1)}
precedence = ((1, ((0, 0),)),)
context = mc.PropagationContext(events, activities, precedence)
registry = mc.DelayFamilyRegistry()
registry.add_empirical(1, [0, 5], [1, 0])

analytic = mc.AnalyticPropagator.from_context(
    context,
    registry,
    step=1,
    underflow_rule=mc.UnderflowRule.TRUNCATE,
    overflow_rule=mc.OverflowRule.TRUNCATE,
)
analytic_result = analytic.run()
assert {v: p for v, p in zip(analytic_result[1].pmf.values.tolist(), analytic_result[1].pmf.probabilities.tolist()) if p > 0} == {10.0: 1.0}

monte_carlo = mc.MonteCarloPropagator.from_context(context, registry)
mc_result = monte_carlo.run(123)
assert float(mc_result.realized[1]) == 10.0
assert float(mc_result.durations[0]) == 10.0
print("installed wheel smoke test passed")
