# mc_dagprop

[![PyPI version](https://img.shields.io/pypi/v/mc_dagprop.svg)](https://pypi.org/project/mc_dagprop/)  
[![Python Versions](https://img.shields.io/pypi/pyversions/mc_dagprop.svg)](https://pypi.org/project/mc_dagprop/)  
[![License](https://img.shields.io/pypi/l/mc_dagprop.svg)](https://github.com/WonJayne/mc_dagprop/blob/main/LICENSE)

**mc_dagprop** is a fast propagation toolkit for directed acyclic graphs (DAGs),
with a C++ Monte Carlo engine and a Python analytic engine.

The package provides two propagation modes:

- **Monte Carlo** simulation based on sampled edge delays.
- **Analytic** propagation of full discrete probability mass functions (PMFs).

Both engines share the same event/activity DAG model and expose aligned naming:
`MonteCarloPropagator` and `AnalyticPropagator`.

## Background

**mc\_dagprop** was developed as part of the
[SORRI project](https://www.ivt.ethz.ch/en/ts/projects/sorri.html) at
the Institute for Transport Planning and Systems (IVT), ETH Zurich. The SORRI project—
*Simulation-based Optimisation for Railway Robustness Improvement*
—focuses on learning real-life constraints and objectives to determine timetables optimized 
for robustness interactively. This research is supported by the
[SBB Research Fund](https://imp-sbb-lab.unisg.ch/de/research-fund/), 
which promotes innovative studies in transport management and the future of mobility in Switzerland.

---

## Features

- **High-performance Monte Carlo core** in C++ via pybind11.
- **Deterministic analytic propagator** for full event-time PMFs.
- Custom per-activity-type Monte Carlo delay distributions:
  - Constant
  - Exponential
  - Gamma
  - Empirical absolute/relative
- Single-run (`run(seed)`) and batched (`run_many(seeds)`) Monte Carlo APIs.
- Shared DAG concepts (`Event`, `Activity`, `DagContext`) and unified naming.
- Optional global `max_delay` cap available in both engines.

> **Note:** For Monte Carlo, configuring multiple distributions for the same
> `activity_type` overrides previous settings. Keep exactly one distribution per type.

---

## Installation

This library requires **Python 3.12** or newer.

```bash
# with poetry
poetry add mc-dagprop

# or with pip
pip install mc-dagprop
```

---

## Quickstart (Monte Carlo)

```python
from mc_dagprop import (
  Activity,
  DagContext,
  Event,
  EventTimestamp,
  GenericDelayGenerator,
  MonteCarloPropagator,
)

# 1) Build your DAG timing context
events = [
  Event("A", EventTimestamp(0.0, 100.0, 0.0)),
  Event("B", EventTimestamp(10.0, 100.0, 0.0)),
]

activities = {
  (0, 1): Activity(idx=0, minimal_duration=60.0, activity_type=1),
}

precedence = [
  (1, [(0, 0)]),
]

ctx = DagContext(
  events=events,
  activities=activities,
  precedence_list=precedence,
  max_delay=1800.0,
)

# 2) Configure a delay generator (one distribution per activity_type)
gen = GenericDelayGenerator()
gen.add_constant(activity_type=1, factor=1.5)

# 3) Run propagation
sim = MonteCarloPropagator(ctx, gen)
result = sim.run(seed=42)

print("Realized times:", result.realized)
print("Edge durations:", result.durations)
print("Causal predecessors:", result.cause_event)
```

`Simulator` remains available as a compatibility alias of
`MonteCarloPropagator`.

---

## Analytic Propagator

Use `AnalyticPropagator` to propagate discrete delay PMFs deterministically.

```python
from mc_dagprop import (
  AnalyticContext,
  Event,
  EventTimestamp,
  OverflowRule,
  UnderflowRule,
  create_analytic_propagator,
)
from mc_dagprop.analytic import AnalyticActivity, exponential_pmf

step = 1.0

delay_pmf = exponential_pmf(scale=10.0, step=step, start=0.0, stop=300.0)

events = (
  Event("A", EventTimestamp(0.0, 10.0, 0.0)),
  Event("B", EventTimestamp(0.0, 20.0, 0.0)),
)

activities = {
  (0, 1): (0, AnalyticActivity(idx=0, pmf=delay_pmf)),
}

precedence = (
  (1, ((0, 0),)),
)

ctx = AnalyticContext(
  events=events,
  activities=activities,
  precedence_list=precedence,
  step=step,
  underflow_rule=UnderflowRule.TRUNCATE,
  overflow_rule=OverflowRule.TRUNCATE,
  max_delay=None,
)

sim = create_analytic_propagator(ctx)
pmfs = sim.run()

print(pmfs[1].pmf.values)
print(pmfs[1].pmf.probabilities)
```

Notes:

- `step` is the shared PMF grid spacing for the analytic context.
- `create_analytic_propagator(..., validate=True)` validates PMF alignment,
  mass consistency, indices, and DAG acyclicity before running.
- `max_delay` mirrors Monte Carlo semantics by capping each event at
  `min(latest, earliest + max_delay)`.

---

## Package structure

- `mc_dagprop.monte_carlo` — compiled Monte Carlo core and delay generator.
- `mc_dagprop.analytic` — pure-Python PMF types, distributions, and propagator.
- `mc_dagprop.types` — shared typed aliases.
- `demo/` — runnable examples for analytic and Monte Carlo usage.

Install the distribution as **mc-dagprop** and import from `mc_dagprop`:

```python
from mc_dagprop import Simulator
```

---

## References

[^1]: Büker, T., et al. (2018). Delay propagation in stochastic railway networks.
[^2]: Subsequent extensions used in SORRI for timetable robustness analysis.
[^3]: De Wilde, B., et al. Event-based simulation approaches for railway delay analysis.
