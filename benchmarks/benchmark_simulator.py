#!/usr/bin/env python3
"""Simple micro-benchmarks for the Simulator class."""

from __future__ import annotations

import random
import time
from collections.abc import Iterable

from mc_dagprop import EventTimestamp, GenericDelayGenerator, Activity, DagContext, Event, Simulator

# Use a reasonably large graph similar to the one used in the tests
N_NODES = 10_000


def build_context() -> DagContext:
    events = [Event(str(i), EventTimestamp(float(i), 100.0 + i, 0.0)) for i in range(N_NODES)]
    link_map = {(i, i + 1): (i, Activity(3.0 + random.random(), 1)) for i in range(N_NODES - 1)}
    precedence_list = [(i, [(i - 1, i)]) for i in range(1, N_NODES)]
    return DagContext(events=events, activities=link_map, precedence_list=precedence_list, max_delay=10.0)


def build_constant_sim(ctx: DagContext) -> Simulator:
    """Simulator with constant delay distribution."""
    gen = GenericDelayGenerator()
    gen.add_constant(activity_type=1, factor=1.0)
    return Simulator(ctx, gen)


def build_exponential_sim(ctx: DagContext) -> Simulator:
    """Simulator with exponential delay distribution."""
    gen = GenericDelayGenerator()
    gen.add_exponential(activity_type=1, lambda_=0.1, max_scale=1.0)
    return Simulator(ctx, gen)


def benchmark_run(sim: Simulator, seeds: Iterable[int]) -> float:
    start = time.perf_counter()
    for seed in seeds:
        sim.run(seed=seed)
    end = time.perf_counter()
    return end - start


def benchmark_run_many(sim: Simulator, batches: int, seeds: Iterable[int]) -> float:
    start = time.perf_counter()
    for _ in range(batches):
        sim.run_many(seeds)
    end = time.perf_counter()
    return end - start


def main() -> None:
    ctx = build_context()

    seeds = list(range(100))
    batches = 10

    constant_sim = build_constant_sim(ctx)
    c_single = benchmark_run(constant_sim, seeds * batches)
    print(f"Constant delay: {len(seeds) * batches} sequential runs took {c_single:.3f} s")
    c_batch = benchmark_run_many(constant_sim, batches, seeds)
    print(f"Constant delay: {batches} batched runs of {len(seeds)} seeds took {c_batch:.3f} s")

    exp_sim = build_exponential_sim(ctx)
    e_single = benchmark_run(exp_sim, seeds * batches)
    print(f"Exponential delay: {len(seeds) * batches} sequential runs took {e_single:.3f} s")
    e_batch = benchmark_run_many(exp_sim, batches, seeds)
    print(f"Exponential delay: {batches} batched runs of {len(seeds)} seeds took {e_batch:.3f} s")


if __name__ == "__main__":
    main()
