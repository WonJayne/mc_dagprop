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
  - Exponential (dimensionless mean factor `scale`)
  - Gamma (dimensionless `shape`, `scale`, and truncation factor)
  - Empirical absolute/relative
- Single-run (`run(seed)`) and batched (`run_many(seeds)`) Monte Carlo APIs.
- Shared DAG concepts (`Event`, `Activity`, `DagContext`) and unified naming.
- Documented backend semantics in `docs/semantics.md`, including analytic hard event bounds, Monte Carlo metadata treatment of `latest`, clipping-policy mass behavior, and marginal propagation limitations.

> **Note:** Configuring multiple stochastic delay families for the same
> `activity_type` is an error. Keep exactly one distribution per type.

---

## Installation

This library supports **CPython 3.12, 3.13, and 3.14**.

```bash
# with poetry
poetry add mc-dagprop

# or with pip
pip install mc-dagprop
```

PyPI does not select a release candidate while a stable release is available.
Install this candidate explicitly when validating `1.0.0rc1`:

```bash
pip install "mc-dagprop==1.0.0rc1"
# or
poetry add "mc-dagprop==1.0.0rc1"
```

Binary wheels are tested natively on Linux x86_64 and ARM64
(`manylinux_2_28`), Windows AMD64, and macOS x86_64 and ARM64. The source
distribution is installed and tested from outside the checkout on the same
platform matrix and all supported Python versions.

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
  Event("B", EventTimestamp(10.0, 100.0, 10.0)),
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
`MonteCarloPropagator`. Delay-family `scale` parameters are dimensionless
factors multiplied by the activity's minimal duration. The ambiguous `lambda_`
compatibility alias has been removed before 1.0.

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

step = 1

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
- Clipping policies are explicit: `TRUNCATE` moves mass to the nearest boundary and preserves total mass; `REMOVE` reports removed mass and returns an explicit sub-probability PMF, including a zero-mass PMF when everything is removed; `REDISTRIBUTE` conditionalizes retained mass or anchors each outside tail at its corresponding bound when none remains.
- The Monte Carlo backend treats `latest` as semantic metadata and does not cap realised event times.
- Analytic construction rejects reconvergent merges whose branches share
  stochastic activity ancestry. Use `validate_equivalence_domain(...)` before
  relying on exact-discrete or quantized-continuous cross-backend parity.

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

- **Windows AMD64** (`windows-latest`)
- **macOS x86_64 and ARM64** (native Intel and Apple Silicon runners)
- **Linux x86_64 and ARM64** (`manylinux_2_28`)

To run a local source build without the CI workflow:

```bash
python -m pip install --upgrade build
python -m build
```

The Python examples in this README and the non-interactive demo paths are
executed in CI. The repository uses the same Black, Ruff, BasedPyright, and
pytest check pattern as OpenBus:

```bash
./scripts/check.sh
python -m pytest --cov=mc_dagprop --cov-branch --cov-fail-under=85
```

The vendored C++ dependency notices are retained in
[`THIRD_PARTY_NOTICES.md`](https://github.com/WonJayne/mc_dagprop/blob/main/THIRD_PARTY_NOTICES.md).

## References

[^1]: Büker, T., et al. (2018). Delay propagation in stochastic railway networks.
[^2]: Subsequent extensions used in SORRI for timetable robustness analysis.
[^3]: De Wilde, B., et al. Event-based simulation approaches for railway delay analysis.


## Release-candidate semantics

`mc_dagprop` evaluates fixed-precedence event-activity DAGs. OpenBus constructs operational graphs upstream, resolves or selects resource precedence orders, and passes deterministic or stochastic separation activities into `PropagationContext`. The current kernel evaluates fixed precedence graphs. Dynamic conflict-order selection and dispatching policies are outside `mc_dagprop` and must be represented by alternative graphs or future policy layers.

The complete timestamp, bound, unit, activity-type, seed, overflow, thread-safety,
and qualified-equivalence contracts are frozen in
[`docs/semantics.md`](https://github.com/WonJayne/mc_dagprop/blob/main/docs/semantics.md).
In particular, analytic `latest` is a
hard bound while Monte Carlo retains it as metadata, and universal equivalence
outside the validated domain is not claimed.

See
[`docs/openbus_integration.md`](https://github.com/WonJayne/mc_dagprop/blob/main/docs/openbus_integration.md)
and [`RELEASE_NOTES.md`](https://github.com/WonJayne/mc_dagprop/blob/main/RELEASE_NOTES.md)
for integration and release details.
