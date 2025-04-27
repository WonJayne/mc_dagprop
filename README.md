# mc_dagprop

[![PyPI version](https://img.shields.io/pypi/v/mc_dagprop.svg)](https://pypi.org/project/mc_dagprop/)  
[![Python Versions](https://img.shields.io/pypi/pyversions/mc_dagprop.svg)](https://pypi.org/project/mc_dagprop/)  
[![License](https://img.shields.io/pypi/l/mc_dagprop.svg)](https://github.com/WonJayne/mc_dagprop/blob/main/LICENSE)

**mc_dagprop** is a fast, Monte Carlo–style propagation simulator for directed acyclic graphs (DAGs),  
written in C++ with Python bindings via **pybind11**. It allows you to model timing networks  
(timetables, precedence graphs, etc.) and inject user-defined delay distributions on edges.

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
- Fast array-based batch mode (`run_many_arrays`)  
- Returns a **SimResult**: realized times, per-edge durations, and causal predecessors  

> **Note:** Defining multiple distributions for the *same* `activity_type` will override previous settings.  
> Always set exactly one distribution per activity type.

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
    SimEvent("A", EventTimestamp(0.0, 100.0, 0.0)),
    SimEvent("B", EventTimestamp(10.0, 100.0, 0.0)),
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

# 2) Configure a delay generator (one per activity_type)
gen = GenericDelayGenerator()
gen.add_constant(activity_type=1, factor=1.5)  # only one call for type=1

# 3) Create simulator and run
sim = Simulator(ctx, gen)
result = sim.run(seed=42)
print("Realized times:", result.realized)
print("Edge durations:", result.durations)
print("Causal predecessors:", result.cause_event)

# 4) Batch-run with fast array output
R = [1, 2, 3, 4, 5]
realized_arr, durations_arr, cause_arr = sim.run_many_arrays(R)
# shapes: realized_arr.shape == (N_nodes, len(R)), durations_arr.shape == (N_links, len(R))
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

- `events`:          `List[SimEvent]`  
- `activities`:      `Dict[(src_idx, dst_idx), SimActivity]`  
- `precedence_list`: `List[(target_idx, [(pred_idx, link_idx), …])]`  
- `max_delay`:       overall cap on delay propagation  

### `GenericDelayGenerator`

Configurable delay factory (one distribution per `activity_type`):

- `.add_constant(activity_type, factor)`  
- `.add_exponential(activity_type, lambda_, max_scale)`  
- `.add_gamma(activity_type, shape, scale, max_scale=∞)`  
- `.set_seed(seed)`  

### `Simulator(context: SimContext, generator: GenericDelayGenerator)`

- `.run(seed: int) → SimResult`  
- `.run_many(seeds: Sequence[int]) → List[SimResult]`  
- `.run_many_arrays(seeds: Sequence[int]) → tuple[
    realized: NDArray[float] (n_nodes × n_runs),
    durations: NDArray[float] (n_links × n_runs),
    cause_event: NDArray[int] (n_nodes × n_runs)
 ]`

### `SimResult`

- `.realized`:   `NDArray[float]` – event times after propagation  
- `.durations`:  `NDArray[float]` – per-edge durations (base + extra)  
- `.cause_event`: `NDArray[int]` – which predecessor caused each event  

---

## Visualization Demo

```bash
pip install plotly
python -m mc_dagprop.utils.demo_distributions
```

Displays histograms of realized times and delays.

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
