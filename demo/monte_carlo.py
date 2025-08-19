from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from demo._shared import ExampleConfig, build_example_context

from mc_dagprop import Activity, DagContext, Event, GenericDelayGenerator, Simulator
from mc_dagprop.types import ActivityType, EventIndex, Second


@dataclass(frozen=True)
class MonteCarloConfig(ExampleConfig):
    """Configuration for the Monte Carlo demonstration."""

    trials: int = 1000
    max_delay: Second = 1800.0


def build_mc_simulator(context_cfg: ExampleConfig, max_delay: Second) -> Simulator:
    """Return a :class:`Simulator` mirroring the analytic example."""

    analytic_ctx = build_example_context(context_cfg)
    events = [Event(ev.event_id, ev.timestamp) for ev in analytic_ctx.events]

    activities: dict[tuple[EventIndex, EventIndex], Activity] = {}
    generator = GenericDelayGenerator()

    for (src, dst), (_, edge) in analytic_ctx.activities.items():
        edge_idx = edge.idx
        activities[(src, dst)] = Activity(
            idx=edge_idx, minimal_duration=Second(0.0), activity_type=ActivityType(edge_idx)
        )
        pmf = edge.pmf
        generator.add_empirical_absolute(ActivityType(edge_idx), pmf.values.tolist(), pmf.probabilities.tolist())

    mc_ctx = DagContext(
        events=events, activities=activities, precedence_list=analytic_ctx.precedence_list, max_delay=max_delay
    )

    return Simulator(mc_ctx, generator)


def run_trials(sim: Simulator, seeds: Sequence[int]) -> np.ndarray:
    """Run ``sim`` for each seed and return realized times."""

    results = sim.run_many(seeds)
    return np.array([res.realized for res in results])


def main() -> None:
    cfg = MonteCarloConfig()
    sim = build_mc_simulator(cfg, cfg.max_delay)
    samples = run_trials(sim, range(cfg.trials))
    analytic = build_example_context(cfg)

    for idx, scheduled in enumerate(analytic.events):
        values, counts = np.unique(samples[:, idx], return_counts=True)
        probs = counts / cfg.trials
        print(f"{scheduled.event_id}:")
        print(f"  values: {values}")
        print(f"  probs:  {probs}\n")


if __name__ == "__main__":
    main()
