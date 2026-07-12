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
`MonteCarloPropagator` and `AnalyticPropagator`. The recommended public construction path is `PropagationContext` plus `DelayFamilyRegistry`, followed by each backend's `from_context(...)` constructor.

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
- Custom per-activity-type stochastic extra-delay distributions:
  - Constant
  - Exponential (`scale` is the mean)
  - Gamma
  - Empirical absolute/relative
- Single-run (`run(seed)`) and batched (`run_many(seeds)`) Monte Carlo APIs.
- Shared DAG concepts (`Event`, `Activity`, `DagContext`) and unified naming.
- Documented backend semantics in `docs/semantics.md`, including analytic hard event bounds, Monte Carlo metadata treatment of `latest`, clipping-policy mass behavior, and marginal propagation limitations.

> **Note:** Configuring multiple stochastic delay families for the same
> `activity_type` is an error. Keep exactly one distribution per type.

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

## Quickstart (shared construction)

```python
from mc_dagprop import (
  Activity,
  AnalyticPropagator,
  DelayFamilyRegistry,
  Event,
  EventTimestamp,
  MonteCarloPropagator,
  OverflowRule,
  PropagationContext,
  UnderflowRule,
)

events = [
  Event("A", EventTimestamp(0.0, 100.0, 0.0)),
  Event("B", EventTimestamp(10.0, 100.0, 0.0)),
]
activities = {(0, 1): Activity(idx=0, minimal_duration=60.0, activity_type=1)}
precedence = [(1, [(0, 0)])]

context = PropagationContext(events=events, activities=activities, precedence_list=precedence)

# Delay families describe stochastic extra delay, not base duration.
registry = DelayFamilyRegistry()
registry.add_empirical(activity_type=1, values=[0.0, 10.0], weights=[0.8, 0.2])

mc = MonteCarloPropagator.from_context(context, registry)
analytic = AnalyticPropagator.from_context(
  context,
  registry,
  step=1,
  underflow_rule=UnderflowRule.TRUNCATE,
  overflow_rule=OverflowRule.TRUNCATE,
)

print(mc.run(seed=42).durations)      # base duration + sampled extra delay
print(analytic.run()[1].pmf.values)   # full edge-increment PMF shifted by base duration

# Unregistered activity types are deterministic: they add no stochastic extra delay.
# Duplicate registrations are rejected with an error identifying the activity type.
```

`Simulator` remains available as a compatibility alias of
`MonteCarloPropagator`. `max_delay` is no longer part of the public API. For exponential delay families, use `scale` as the mean; `lambda_` exists only as a deprecated compatibility alias.

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
)

sim = create_analytic_propagator(ctx)
pmfs = sim.run()

print(pmfs[1].pmf.values)
print(pmfs[1].pmf.probabilities)
```

Notes:

- Shared frontend delay families represent stochastic extra delay. Analytic construction shifts them by each activity's deterministic minimal duration to form full edge-increment PMFs.
- Unregistered activity types are deterministic in both backends.
- Duplicate stochastic registrations for the same activity type are rejected; use a fresh registry to change a family.
- `step` is the shared PMF grid spacing for the analytic context.
- `create_analytic_propagator(..., validate=True)` validates PMF alignment,
  mass consistency, indices, and DAG acyclicity before running.
- PMF binary operations (`convolve` and `maximum`) use numerically stable
  intermediates (`np.longdouble`) and a post-operation mass correction. This
  prevents tiny probabilities from being lost to cumulative floating-point
  drift in deep analytic propagation chains.
- The analytic backend bounds each event distribution to `[event.earliest, event.latest]`; `latest` is a hard bound. Use `step=1` for exact tests; a coarser `step=3` may be practical for examples when the timetable grid supports it.
- Clipping policies are explicit: `TRUNCATE` moves mass to the nearest boundary and preserves total mass; `REMOVE` reports removed mass and returns an explicit sub-probability PMF; `REDISTRIBUTE` conditionalizes the retained mass.
- The Monte Carlo backend treats `latest` as semantic metadata and does not cap realised event times.
- Analytic propagation is marginal PMF propagation. It is not generally exact on reconvergent DAGs with shared stochastic ancestry. A low-level conditional convolution primitive is tested for small PMFs, but full Büker-style route-conflict handling, interlinking/connection modelling, train priorities, and exact joint-distribution propagation are intentionally out of scope.

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


## Building wheels for Windows, macOS, and Linux

Cross-platform wheel builds are configured through GitHub Actions in
`.github/workflows/build-wheels.yml` using `cibuildwheel`.

- **Windows** (`windows-latest`)
- **macOS** (`macos-latest`, `x86_64` and `arm64`)
- **Linux** (`ubuntu-latest`)

To run a local source build without the CI workflow:

```bash
python -m pip install --upgrade build
python -m build
```

## References

[^1]: Büker, T., et al. (2018). Delay propagation in stochastic railway networks.
[^2]: Subsequent extensions used in SORRI for timetable robustness analysis.
[^3]: De Wilde, B., et al. Event-based simulation approaches for railway delay analysis.
