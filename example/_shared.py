from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mc_dagprop import AnalyticContext, DiscretePMF, Event, EventTimestamp, OverflowRule, UnderflowRule
from mc_dagprop.analytic import AnalyticActivity
from mc_dagprop.types import ActivityIndex, EventId, EventIndex, Second


@dataclass(frozen=True)
class ExampleConfig:
    """Configuration for the example DAG."""

    step_size: Second = 1.0


def build_example_context(cfg: ExampleConfig = ExampleConfig()) -> AnalyticContext:
    """Return an :class:`AnalyticContext` with ten events and twelve activities."""

    events = (
        Event(EventId("E0"), EventTimestamp(0.0, 0.0, 0.0)),
        Event(EventId("E1"), EventTimestamp(2.0, 5.0, 3.0)),
        Event(EventId("E2"), EventTimestamp(4.0, 8.0, 6.0)),
        Event(EventId("E3"), EventTimestamp(6.0, 11.0, 8.0)),
        Event(EventId("E4"), EventTimestamp(7.0, 13.0, 9.0)),
        Event(EventId("E5"), EventTimestamp(10.0, 15.0, 12.0)),
        Event(EventId("E6"), EventTimestamp(8.0, 12.0, 9.0)),
        Event(EventId("E7"), EventTimestamp(11.0, 16.0, 13.0)),
        Event(EventId("E8"), EventTimestamp(12.0, 18.0, 14.0)),
        Event(EventId("E9"), EventTimestamp(14.0, 20.0, 16.0)),
    )

    step = cfg.step_size

    activities = {
        (0, 1): (0, AnalyticActivity(ActivityIndex(0), DiscretePMF(np.array([2.0]), np.array([1.0]), step))),
        (1, 2): (
            1,
            AnalyticActivity(ActivityIndex(1), DiscretePMF(np.array([1.0, 2.0, 3.0]), np.array([1 / 3, 1 / 3, 1 / 3]), step)),
        ),
        (2, 3): (
            ActivityIndex(2),
            AnalyticActivity(ActivityIndex(2), DiscretePMF(np.array([1.0, 2.0, 3.0]), np.array([0.2, 0.6, 0.2]), step)),
        ),
        (3, 4): (ActivityIndex(3), AnalyticActivity(ActivityIndex(3), DiscretePMF(np.array([2.0]), np.array([1.0]), step))),
        (4, 5): (
            ActivityIndex(4),
            AnalyticActivity(ActivityIndex(4), DiscretePMF(np.array([1.0, 2.0]), np.array([0.7, 0.3]), step)),
        ),
        (1, 6): (
            ActivityIndex(5),
            AnalyticActivity(
                ActivityIndex(5), DiscretePMF(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.4, 0.3, 0.2, 0.1]), step)
            ),
        ),
        (6, 7): (
            ActivityIndex(6),
            AnalyticActivity(ActivityIndex(6), DiscretePMF(np.array([1.0, 2.0, 3.0]), np.array([0.3, 0.5, 0.2]), step)),
        ),
        (7, 5): (
            ActivityIndex(7),
            AnalyticActivity(ActivityIndex(7), DiscretePMF(np.array([1.0, 3.0, 5.0]), np.array([0.5, 0.3, 0.2]), step)),
        ),
        (2, 8): (
            ActivityIndex(8),
            AnalyticActivity(ActivityIndex(8), DiscretePMF(np.array([2.0, 4.0]), np.array([0.6, 0.4]), step)),
        ),
        (8, 9): (9, AnalyticActivity(ActivityIndex(9), DiscretePMF(np.array([1.0]), np.array([1.0]), step))),
        (6, 8): (
            ActivityIndex(10),
            AnalyticActivity(ActivityIndex(10), DiscretePMF(np.array([3.0, 4.0, 5.0]), np.array([0.2, 0.5, 0.3]), step)),
        ),
        (4, 9): (
            ActivityIndex(11),
            AnalyticActivity(ActivityIndex(11), DiscretePMF(np.array([0.0, 1.0, 2.0]), np.array([0.3, 0.4, 0.3]), step)),
        ),
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
