# mc_dagprop

[![PyPI version](https://img.shields.io/pypi/v/mc_dagprop.svg)](https://pypi.org/project/mc_dagprop/)  
[![Python Versions](https://img.shields.io/pypi/pyversions/mc_dagprop.svg)](https://pypi.org/project/mc_dagprop/)  
[![License](https://img.shields.io/pypi/l/mc_dagprop.svg)](https://github.com/WonJayne/mc_dagprop/blob/main/LICENSE)

**mc_dagprop** is a fast, Monte Carlo–style propagation simulator for directed acyclic graphs (DAGs), written in C++  
with Python bindings via **pybind11**. It allows you to model timing networks (timetables, precedence graphs, etc.)  
and inject user-defined delay distributions on edges.

---

## Features

- **Lightweight & high-performance** core in C++
- Simple Python API via **poetry** or **pip**
- Custom per-activity-type delay distributions:
    - **Constant** (linear scaling)
    - **Exponential** (with cutoff)
    - **Gamma** (shape & scale)
    - Easily extendable (Weibull, etc.)
- Single-run (`run(seed)`) and batch-run (`run_many([seeds])`)
- Returns a **SimResult**: realized times, per-edge delays, and causal predecessors

---

## Installation

```bash
# with poetry
poetry add mc_dagprop

# or with pip
pip install mc_dagprop
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
    SimEvent("A", EventTimestamp(0.0, 5.0, 2.0)),
    SimEvent("B", EventTimestamp(10.0, 15.0, 12.0)),
]

activities = {
    (0, 1): SimActivity(minimal_duration=60.0, activity_type=1),
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

# 2) Configure delay generator
gen = GenericDelayGenerator()
gen.add_constant(activity_type=1, factor=1.5)
gen.add_exponential(activity_type=1, lambda_=2.0, max_scale=5.0)
gen.add_gamma(activity_type=1, shape=2.0, scale=0.5)

# 3) Simulate
sim = Simulator(ctx, gen)
result = sim.run(seed=42)
print(result.realized, result.delays, result.cause_event)
```

---

## API Reference

### `EventTimestamp(earliest: float, latest: float, actual: float)`

Holds the scheduling window and actual time for one event (node):

- `earliest` – earliest possible occurrence
- `latest`   – latest allowed occurrence
- `actual`   – scheduled (baseline) timestamp

### `SimEvent(node_id: str, timestamp: EventTimestamp)`

Wraps a DAG node with its identifier and timing stamp:

- `node_id`   – string key for the node
- `timestamp` – an `EventTimestamp` instance

### `SimActivity(duration: float, activity_type: int)`

Represents an edge in the DAG:

- `minimal_duration`      – minimal (base) duration
- `activity_type` – integer type id

### `SimContext(events, activities, precedence_list, max_delay)`

Container for your DAG:

- `events`:          `List[SimEvent]`
- `activities`:      `Dict[(src_idx, dst_idx), SimActivity]`
- `precedence_list`: `List[(target_idx, [(pred_idx, act_idx), …])]`
- `max_delay`:       overall cap on delay propagation for an event

### `GenericDelayGenerator`

Configurable delay factory:

- `.add_constant(activity_type, factor)`
- `.add_exponential(activity_type, lambda_, max_scale)`
- `.add_gamma(activity_type, shape, scale)`
- `.set_seed(seed)`

### `Simulator(context: SimContext, generator: GenericDelayGenerator)`

- `.run(seed: int) → SimResult`
- `.run_many(seeds: Sequence[int]) → List[SimResult]`

### `SimResult`

- `.realized`:   `List[float]` – event times after propagation
- `.delays`:     `List[float]` – per-edge injected delays
- `.cause_event`: `List[int]` – which predecessor caused each event

---

## Visualization Demo

```bash
# install plotly to run the demo
pip install plotly

# then from your project root
python -m mc_dagprop.utils.demo_distributions
```

Displays histograms of the realized times under Constant, Exponential, and Gamma delays.

---

## Development

```bash
git clone https://github.com/WonJayne/mc_dagprop.git
cd mc_dagprop
poetry install
poetry run pytest
```

---

## License

MIT — see [LICENSE](LICENSE)  
