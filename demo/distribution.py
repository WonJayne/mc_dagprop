from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping

import plotly.graph_objects as go

from mc_dagprop import Event, EventTimestamp, Activity, DagContext, GenericDelayGenerator, Simulator
from mc_dagprop.types import ActivityType, ActivityIndex, Second


def simulate_and_collect(
    dist_name: str, params: Mapping[str, float], seeds: Iterable[int], base_duration: Second = 60.0
) -> list[float]:
    """Run a Monte Carlo simulation and return realized times."""

    events = [Event("A", EventTimestamp(0.0, 0.0, 0.0)), Event("B", EventTimestamp(0.0, 0.0, 0.0))]
    activities = {(0, 1): Activity(idx=ActivityIndex(0), minimal_duration=base_duration, activity_type=ActivityType(1))}
    precedence = [(1, [(0, 0)])]
    ctx = DagContext(events, activities, precedence, max_delay=1e6)

    gen = GenericDelayGenerator()
    if dist_name == "constant":
        gen.add_constant(ActivityType(1), factor=params["factor"])
    elif dist_name == "exponential":
        gen.add_exponential(ActivityType(1), lambda_=params["lambda"], max_scale=params["max_scale"])
    elif dist_name == "gamma":
        gen.add_gamma(
            ActivityType(1), shape=params["shape"], scale=params["scale"], max_scale=params.get("max_scale", 1e6)
        )
    else:
        raise ValueError(dist_name)

    sim = Simulator(ctx, gen)
    results = sim.run_many(seeds)
    return [res.realized[1] for res in results]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare different delay distributions")
    parser.add_argument("--trials", type=int, default=10000, help="number of Monte-Carlo runs per parameter set")
    args = parser.parse_args()

    seeds = tuple(range(args.trials))
    configs = {
        "constant": [{"factor": 1.0}, {"factor": 2.0}],
        "exponential": [
            {"lambda": 0.1, "max_scale": 5.0},
            {"lambda": 0.5, "max_scale": 5.0},
            {"lambda": 1.0, "max_scale": 5.0},
        ],
        "gamma": [
            {"shape": 0.1, "scale": 2.5, "max_scale": 3.0},
            {"shape": 2, "scale": 0.05, "max_scale": 5},
            {"shape": 0.5, "scale": 2.0},
            {"shape": 1.0, "scale": 1.0},
            {"shape": 2.0, "scale": 0.5},
            {"shape": 5.0, "scale": 0.2},
        ],
    }

    fig = go.Figure()
    for dist_name, param_list in configs.items():
        for params in param_list:
            label = f"{dist_name} " + ", ".join(f"{k}={v}" for k, v in params.items())
            data = simulate_and_collect(dist_name, params, seeds)
            fig.add_trace(go.Histogram(x=data, name=label, opacity=0.7, histnorm="probability density"))

    fig.update_layout(
        title="Distribution of Arrival Times at B (after base 60s)",
        xaxis_title="Realized time of B (s)",
        yaxis_title="Count",
        barmode="overlay",
    )
    fig.show()


if __name__ == "__main__":
    main()
