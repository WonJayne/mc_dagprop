from __future__ import annotations

from ._shared import ExampleConfig, build_example_context
from .. import create_discrete_simulator


def main() -> None:
    """Run the analytic propagation demonstration."""

    ctx = build_example_context(ExampleConfig())
    sim = create_discrete_simulator(ctx)
    results = sim.run()

    for scheduled, result in zip(ctx.events, results):
        print(f"{scheduled.event_id}:")
        print(f"  values: {result.pmf.values}")
        print(f"  probs:  {result.pmf.probabilities}")
        print(f"  underflow: {float(result.underflow)} overflow: {float(result.overflow)}\n")


if __name__ == "__main__":
    main()
