from __future__ import annotations

from mc_dagprop import create_discrete_simulator

from ._shared import ExampleConfig, build_example_context


def main() -> None:
    ctx = build_example_context()
    sim = create_discrete_simulator(ctx)
    results = sim.run()

    for sched, result in zip(ctx.events, results):
        print(f"{sched.id}:")
        print(f"  values: {result.pmf.values}")
        print(f"  probs:  {result.pmf.probabilities}")
        print(f"  underflow: {float(result.underflow)} overflow: {float(result.overflow)}\n")


if __name__ == "__main__":
    main()
