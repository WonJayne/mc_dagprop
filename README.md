# mc_dagprop

[![PyPI version](https://img.shields.io/pypi/v/mc_dagprop.svg)](https://pypi.org/project/mc_dagprop/)  
[![Python Versions](https://img.shields.io/pypi/pyversions/mc_dagprop.svg)](https://pypi.org/project/mc_dagprop/)  
[![License](https://img.shields.io/pypi/l/mc_dagprop.svg)](https://github.com/WonJayne/mc_dagprop/blob/main/LICENSE)

**mc_dagprop** is a fast, Monte Carlo–style propagation simulator for directed acyclic graphs (DAGs),  
written in C++ with Python bindings via **pybind11**. It allows you to model timing networks  
(timetables, precedence graphs, etc.) and inject user-defined delay distributions on edges.

Under the hood, we leverage the high-performance [utl::random module](https://github.com/DmitriBogdanov/UTL/blob/master/docs/module_random.md)  
for all pseudo-random number generation—offering better speed and quality than the standard library.

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

- **Lightweight & high-performance** core in C++  
- Simple Python API via **poetry** or **pip**  
- Custom per-activity-type delay distributions:
  - **Constant** (linear scaling)
  - **Exponential** (scales base duration with cutoff)
  - **Gamma** (shape & scale, to scale base duration)
  - **Empirical** (absolute or relative)
    - **Absolute**: fixed values with weights
    - **Relative**: scaling factors with weights
  - Easily extendable (Weibull, etc.)  
- Single-run (`run(seed)`) and batch-run (`run_many([seeds])`), the latter releases the GIL, thus one can run it embarrassingly parallel with multithreading
- Returns a **SimResult**: realized times, per-edge durations, and causal predecessors  

> **Note:** Defining multiple distributions for the *same* `activity_type` will override previous settings.  
> Always set exactly one distribution per activity type.

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

## Quickstart

```python
from mc_dagprop import (
    EventTimestamp,
    SimEvent,
    SimActivity,
    SimContext,
    GenericDelayGenerator,
    Simulator,
)

# 1) Build your DAG timing context
events = [
    SimEvent("A", EventTimestamp(0.0, 100.0, 0.0)),
    SimEvent("B", EventTimestamp(10.0, 100.0, 0.0)),
]

activities = {
    (0, 1): (0, SimActivity(minimal_duration=60.0, activity_type=1)),
}

precedence = [
    (1, [(0, 0)]),
]

ctx = SimContext(
    events=events,
    activities=activities,
    precedence_list=precedence,
    max_delay=1800.0,
)

# 2) Configure a delay generator (one per activity_type)
gen = GenericDelayGenerator()
gen.add_constant(activity_type=1, factor=1.5)  # only one call for type=1

# 3) Create simulator and run
sim = Simulator(ctx, gen)
result = sim.run(seed=42)
print("Realized times:", result.realized)
print("Edge durations:", result.durations)
print("Causal predecessors:", result.cause_event)
```

---

## API Reference

### `EventTimestamp(earliest: float, latest: float, actual: float)`

Holds the scheduling window and actual time for one event (node):

- `earliest` – earliest possible occurrence  
- `latest`   – latest allowed occurrence  
- `actual`   – scheduled (baseline) timestamp  

### `SimEvent(id: str, timestamp: EventTimestamp)`

Wraps a DAG node with:

- `id`        – string key for the node  
- `timestamp` – an `EventTimestamp` instance  

### `SimActivity(minimal_duration: float, activity_type: int)`

Represents an edge in the DAG:

- `minimal_duration` – minimal (base) duration  
- `activity_type`    – integer type identifier  

### `SimContext(events, activities, precedence_list, max_delay)`

Container for your DAG:

- `events`:          `list[SimEvent]`
- `activities`:      `dict[(src_idx, dst_idx), (link_idx, SimActivity)]`
- `precedence_list`: `list[(target_idx, [(pred_idx, link_idx), …])]`
- `max_delay`:       overall cap on delay propagation
  - Can be given in any order. `Simulator` will sort topologically and raise
    a `RuntimeError` if cycles are detected.

### `GenericDelayGenerator`

Configurable delay factory (one distribution per `activity_type`):

- `.add_constant(activity_type, factor)`  
- `.add_exponential(activity_type, lambda_, max_scale)`  
- `.add_gamma(activity_type, shape, scale, max_scale=∞)`  
- `.add_empirical_absolute(activity_type, values, weights)`
- `.add_empirical_relative(activity_type, factors, weights)`
- `.set_seed(seed)`  

### `Simulator(context: SimContext, generator: GenericDelayGenerator)`

- `.run(seed: int) → SimResult`  
- `.run_many(seeds: Sequence[int]) → list[SimResult]`  

### `SimResult`

- `.realized`:   `NDArray[float]` – event times after propagation  
- `.durations`:  `NDArray[float]` – per-edge durations (base + extra)  
- `.cause_event`: `NDArray[int]` – which predecessor caused each event  

## Discrete Simulator

You can propagate discrete delay distributions analytically using `DiscreteSimulator`.
Define per-edge probability mass functions and build an `AnalyticContext`:

```python
from mc_dagprop import AnalyticContext, DiscretePMF, DiscreteSimulator, EventTimestamp, SimEvent

events = (
    SimEvent("A", EventTimestamp(0, 10, 0)),
    SimEvent("B", EventTimestamp(0, 10, 0)),
)
activities = {(0, 1): (0, DiscretePMF([1.0, 2.0], [0.5, 0.5]))}
precedence = (
    (1, ((0, 0),)),
)
ctx = AnalyticContext(
    events=events,
    activities=activities,
    precedence_list=precedence,
    step_size=1.0,
)

sim = DiscreteSimulator(ctx)
pmfs = sim.run()
print(pmfs[1].values, pmfs[1].probs)
```
This computes event-time PMFs deterministically without Monte-Carlo sampling.

The ``step_size`` sets the spacing for all values in the discrete PMFs.
``DiscreteSimulator`` invokes ``AnalyticContext.validate()`` at construction
time and will raise an error when any edge uses a different step.
Each ``ScheduledEvent`` may specify ``bounds=(lower, upper)`` to clip the
resulting distribution. The ``run()`` method returns a sequence of
``SimulatedEvent`` objects which hold the resulting PMF and the under- and
overflow mass for that event.
By default the step size is ``1.0`` second and typical delay deviations range
roughly from ``-180`` s up to ``+1800`` s.


---

## Visualization Demo

```bash
pip install mc-dagprop[plot]
python -m mc_dagprop.utils.demonstration
```

Displays histograms of realized times and delays.

---

## Benchmarks

A lightweight benchmark helps to measure raw execution speed for a large
simulation instance. Two delay generators are provided – one constant and
one exponential – so you can compare different implementations against the
same baseline and detect performance regressions.

```bash
python benchmarks/benchmark_simulator.py
```

---

## Development

```bash
git clone https://github.com/WonJayne/mc_dagprop.git
cd mc_dagprop
poetry install
```

---

## License

MIT — see [LICENSE](LICENSE)
