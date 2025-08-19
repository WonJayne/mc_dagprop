from demo._shared import ExampleConfig, build_example_context

from mc_dagprop import create_analytic_propagator


def main() -> None:
    """Run the analytic propagation demonstration."""

    ctx = build_example_context(ExampleConfig())
    sim = create_analytic_propagator(ctx)
    results = sim.run()

    for scheduled, result in zip(ctx.events, results):
        print(f"{scheduled.event_id}:")
        print(f"  values: {result.pmf.values}")
        print(f"  probs:  {result.pmf.probabilities}")
        print(f"  underflow: {float(result.underflow)} overflow: {float(result.overflow)}\n")


if __name__ == "__main__":
    main()
