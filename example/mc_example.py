from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import numpy as np

from mc_dagprop import (
    GenericDelayGenerator,
    SimActivity,
    SimContext,
    SimEvent,
    Simulator,
)
from ._shared import ExampleConfig, build_example_context


@dataclass(frozen=True)
class MonteCarloConfig(ExampleConfig):
    """Configuration for the Monte Carlo demonstration."""

    trials: int = 1000
    max_delay: float = 20.0


def build_mc_simulator(context_cfg: ExampleConfig, max_delay: float) -> Simulator:
    """Return a :class:`Simulator` mirroring the analytic example."""

    analytic_ctx = build_example_context(context_cfg)
    events = [SimEvent(ev.id, ev.timestamp) for ev in analytic_ctx.events]

    activities: dict[tuple[int, int], tuple[int, SimActivity]] = {}
    generator = GenericDelayGenerator()

    for (src, dst), (edge_idx, edge) in analytic_ctx.activities.items():
        activities[(src, dst)] = (edge_idx, SimActivity(0.0, edge_idx))
        pmf = edge.pmf
        generator.add_empirical_absolute(
            edge_idx,
            pmf.values.tolist(),
            pmf.probabilities.tolist(),
        )

    mc_ctx = SimContext(
        events=events,
        activities=activities,
        precedence_list=analytic_ctx.precedence_list,
        max_delay=max_delay,
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

    for idx, sched in enumerate(analytic.events):
        values, counts = np.unique(samples[:, idx], return_counts=True)
        probs = counts / cfg.trials
        print(f"{sched.id}:")
        print(f"  values: {values}")
        print(f"  probs:  {probs}")
        print()


if __name__ == "__main__":
    main()
