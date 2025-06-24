from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from mc_dagprop import (
    AnalyticContext,
    DiscretePMF,
    EventTimestamp,
    ScheduledEvent,
    UnderflowRule,
    OverflowRule,
)
from mc_dagprop.analytic.context import AnalyticEdge


@dataclass(frozen=True)
class ExampleConfig:
    """Configuration for the example DAG."""

    step_size: float = 1.0


def build_example_context(cfg: ExampleConfig = ExampleConfig()) -> AnalyticContext:
    """Return an :class:`AnalyticContext` with ten events and twelve activities."""

    events = (
        ScheduledEvent("E0", EventTimestamp(0.0, 0.0, 0.0)),
        ScheduledEvent("E1", EventTimestamp(2.0, 5.0, 3.0)),
        ScheduledEvent("E2", EventTimestamp(4.0, 8.0, 6.0)),
        ScheduledEvent("E3", EventTimestamp(6.0, 11.0, 8.0)),
        ScheduledEvent("E4", EventTimestamp(7.0, 13.0, 9.0)),
        ScheduledEvent("E5", EventTimestamp(10.0, 15.0, 12.0)),
        ScheduledEvent("E6", EventTimestamp(8.0, 12.0, 9.0)),
        ScheduledEvent("E7", EventTimestamp(11.0, 16.0, 13.0)),
        ScheduledEvent("E8", EventTimestamp(12.0, 18.0, 14.0)),
        ScheduledEvent("E9", EventTimestamp(14.0, 20.0, 16.0)),
    )

    step = cfg.step_size

    activities = {
        (0, 1): (0, AnalyticEdge(0, DiscretePMF(np.array([2.0]), np.array([1.0]), step))),
        (1, 2): (
            1,
            AnalyticEdge(
                1,
                DiscretePMF(np.array([1.0, 2.0, 3.0]), np.array([1 / 3, 1 / 3, 1 / 3]), step),
            ),
        ),
        (2, 3): (
            2,
            AnalyticEdge(
                2,
                DiscretePMF(np.array([1.0, 2.0, 3.0]), np.array([0.2, 0.6, 0.2]), step),
            ),
        ),
        (3, 4): (3, AnalyticEdge(3, DiscretePMF(np.array([2.0]), np.array([1.0]), step))),
        (4, 5): (
            4,
            AnalyticEdge(4, DiscretePMF(np.array([1.0, 2.0]), np.array([0.7, 0.3]), step)),
        ),
        (1, 6): (
            5,
            AnalyticEdge(
                5,
                DiscretePMF(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.4, 0.3, 0.2, 0.1]), step),
            ),
        ),
        (6, 7): (
            6,
            AnalyticEdge(
                6,
                DiscretePMF(np.array([1.0, 2.0, 3.0]), np.array([0.3, 0.5, 0.2]), step),
            ),
        ),
        (7, 5): (
            7,
            AnalyticEdge(7, DiscretePMF(np.array([1.0, 3.0, 5.0]), np.array([0.5, 0.3, 0.2]), step)),
        ),
        (2, 8): (
            8,
            AnalyticEdge(8, DiscretePMF(np.array([2.0, 4.0]), np.array([0.6, 0.4]), step)),
        ),
        (8, 9): (9, AnalyticEdge(9, DiscretePMF(np.array([1.0]), np.array([1.0]), step))),
        (6, 8): (
            10,
            AnalyticEdge(10, DiscretePMF(np.array([3.0, 4.0, 5.0]), np.array([0.2, 0.5, 0.3]), step)),
        ),
        (4, 9): (
            11,
            AnalyticEdge(11, DiscretePMF(np.array([0.0, 1.0, 2.0]), np.array([0.3, 0.4, 0.3]), step)),
        ),
    }

    precedence_list = (
        (1, ((0, 0),)),
        (2, ((1, 1),)),
        (3, ((2, 2),)),
        (4, ((3, 3),)),
        (5, ((4, 4), (7, 7))),
        (6, ((1, 5),)),
        (7, ((6, 6),)),
        (8, ((2, 8), (6, 10))),
        (9, ((8, 9), (4, 11))),
    )

    return AnalyticContext(
        events=events,
        activities=activities,
        precedence_list=precedence_list,
        step_size=step,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.TRUNCATE,
    )
