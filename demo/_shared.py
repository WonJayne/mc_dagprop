from __future__ import annotations

from dataclasses import dataclass

from mc_dagprop import AnalyticContext, DiscretePMF, Event, EventTimestamp, OverflowRule, UnderflowRule
from mc_dagprop.analytic import AnalyticActivity, exponential_pmf
from mc_dagprop.types import Second


@dataclass(frozen=True)
class ExampleConfig:
    """Configuration for the example DAG."""

    step_size: int = 1
    pmf_stop: Second = 200.0


_DEFAULT_EXAMPLE_CONFIG = ExampleConfig()


def build_example_context(cfg: ExampleConfig = _DEFAULT_EXAMPLE_CONFIG) -> AnalyticContext:
    """Return an exact-domain example with three independent branches."""

    events = (
        Event("E0", EventTimestamp(0.0, 0.0, 0.0)),
        Event("E1", EventTimestamp(2.0, 2.0, 2.0)),
        Event("E2", EventTimestamp(4.0, 4.0, 4.0)),
        Event("E3", EventTimestamp(0.0, 2000.0, 0.0)),
        Event("E4", EventTimestamp(0.0, 2000.0, 0.0)),
        Event("E5", EventTimestamp(0.0, 2000.0, 0.0)),
        Event("E6", EventTimestamp(0.0, 2000.0, 0.0)),
        Event("E7", EventTimestamp(0.0, 2000.0, 0.0)),
        Event("E8", EventTimestamp(0.0, 2000.0, 0.0)),
        Event("E9", EventTimestamp(0.0, 2000.0, 0.0)),
    )

    step = cfg.step_size
    pmf_stop = cfg.pmf_stop

    def _exp(scale: Second) -> DiscretePMF:
        return exponential_pmf(scale=scale, step=step, start=0.0, stop=pmf_stop)

    activities = {
        (0, 3): (0, AnalyticActivity(0, _exp(2.0))),
        (3, 6): (1, AnalyticActivity(1, _exp(3.0))),
        (1, 4): (2, AnalyticActivity(2, _exp(4.0))),
        (4, 7): (3, AnalyticActivity(3, _exp(2.0))),
        (2, 5): (4, AnalyticActivity(4, _exp(3.5))),
        (5, 8): (5, AnalyticActivity(5, _exp(3.5))),
        (6, 9): (6, AnalyticActivity(6, _exp(2.5))),
        (7, 9): (7, AnalyticActivity(7, _exp(4.5))),
        (8, 9): (8, AnalyticActivity(8, _exp(5.0))),
    }

    precedence_list = (
        (3, ((0, 0),)),
        (4, ((1, 2),)),
        (5, ((2, 4),)),
        (6, ((3, 1),)),
        (7, ((4, 3),)),
        (8, ((5, 5),)),
        (9, ((6, 6), (7, 7), (8, 8))),
    )

    return AnalyticContext(
        events=events,
        activities=activities,
        precedence_list=precedence_list,
        step=step,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.REMOVE,
    )
