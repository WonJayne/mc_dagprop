from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mc_dagprop import AnalyticContext, DiscretePMF, Event, EventTimestamp, OverflowRule, UnderflowRule
from mc_dagprop.analytic import AnalyticActivity, exponential_pmf
from mc_dagprop.types import ActivityIndex, EventId, EventIndex, Second


@dataclass(frozen=True)
class ExampleConfig:
    """Configuration for the example DAG."""

    step_size: Second = 1
    pmf_stop: Second = 200.0


def build_example_context(cfg: ExampleConfig = ExampleConfig()) -> AnalyticContext:
    """Return an example context with ten events and twelve activities."""

    events = (
        Event(EventId("E0"), EventTimestamp(0.0, 0.0, 0.0)),
        Event(EventId("E1"), EventTimestamp(2.0, 500.0, 3.0)),
        Event(EventId("E2"), EventTimestamp(4.0, 800.0, 6.0)),
        Event(EventId("E3"), EventTimestamp(6.0, 1100.0, 8.0)),
        Event(EventId("E4"), EventTimestamp(7.0, 1300.0, 9.0)),
        Event(EventId("E5"), EventTimestamp(10.0, 1500.0, 12.0)),
        Event(EventId("E6"), EventTimestamp(8.0, 1200.0, 9.0)),
        Event(EventId("E7"), EventTimestamp(11.0, 1600.0, 13.0)),
        Event(EventId("E8"), EventTimestamp(12.0, 1800.0, 14.0)),
        Event(EventId("E9"), EventTimestamp(14.0, 2000.0, 16.0)),
    )

    step = cfg.step_size
    pmf_stop = cfg.pmf_stop

    def _exp(scale: Second) -> DiscretePMF:
        return exponential_pmf(scale=scale, step=step, start=0.0, stop=pmf_stop)

    activities = {
        (0, 1): (ActivityIndex(0), AnalyticActivity(ActivityIndex(0), _exp(2.0))),
        (1, 2): (ActivityIndex(1), AnalyticActivity(ActivityIndex(1), _exp(3.0))),
        (2, 3): (ActivityIndex(2), AnalyticActivity(ActivityIndex(2), _exp(4.0))),
        (3, 4): (ActivityIndex(3), AnalyticActivity(ActivityIndex(3), _exp(2.0))),
        (4, 5): (ActivityIndex(4), AnalyticActivity(ActivityIndex(4), _exp(3.5))),
        (1, 6): (ActivityIndex(5), AnalyticActivity(ActivityIndex(5), _exp(3.5))),
        (6, 7): (ActivityIndex(6), AnalyticActivity(ActivityIndex(6), _exp(2.5))),
        (7, 5): (ActivityIndex(7), AnalyticActivity(ActivityIndex(7), _exp(4.5))),
        (2, 8): (ActivityIndex(8), AnalyticActivity(ActivityIndex(8), _exp(5.0))),
        (8, 9): (ActivityIndex(9), AnalyticActivity(ActivityIndex(9), _exp(2.0))),
        (6, 8): (ActivityIndex(10), AnalyticActivity(ActivityIndex(10), _exp(2.5))),
        (4, 9): (ActivityIndex(11), AnalyticActivity(ActivityIndex(11), _exp(3.0))),
    }

    precedence_list = (
        (EventIndex(1), ((EventIndex(0), ActivityIndex(0)),)),
        (EventIndex(2), ((EventIndex(1), ActivityIndex(1)),)),
        (EventIndex(3), ((EventIndex(2), ActivityIndex(2)),)),
        (EventIndex(4), ((EventIndex(3), ActivityIndex(3)),)),
        (EventIndex(5), ((EventIndex(4), ActivityIndex(4)), (EventIndex(7), ActivityIndex(7)))),
        (EventIndex(6), ((EventIndex(1), ActivityIndex(5)),)),
        (EventIndex(7), ((EventIndex(6), ActivityIndex(6)),)),
        (EventIndex(8), ((EventIndex(2), ActivityIndex(8)), (EventIndex(6), ActivityIndex(10)))),
        (EventIndex(9), ((EventIndex(8), ActivityIndex(9)), (EventIndex(4), ActivityIndex(11)))),
    )

    return AnalyticContext(
        events=events,
        activities=activities,
        precedence_list=precedence_list,
        step=step,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.REMOVE,
    )
