# mc_dagprop

[![PyPI version](https://img.shields.io/pypi/v/mc_dagprop.svg)](https://pypi.org/project/mc_dagprop/) [![Python Versions](https://img.shields.io/pypi/pyversions/mc_dagprop.svg)](https://pypi.org/project/mc_dagprop/) [![License](https://img.shields.io/pypi/l/mc_dagprop.svg)](https://github.com/WonJayne/mc_dagprop/blob/main/LICENSE)

**mc_dagprop** is a fast, Monte Carlo–style propagation simulator for directed acyclic graphs (DAGs), written in C++ with Python bindings via **pybind11**.  
It allows you to model timing networks (timetables, precedence graphs, etc.) and inject user‑defined delay distributions on links.  

---

## Features

- **Lightweight & high‑performance** core in C++  
- Expose a simple Python API via **poetry** or **pip**  
- Define custom delay distributions per link‑type:
  - **Constant** (linear scaling)
  - **Exponential** (with cutoff)
  - Easily extendable for Weibull, Gamma, …
- Single‑run (`run(seed)`) and batch‑run (`run_many([seeds])`) support  
- Returns a **SimResult** struct: realized times, link delays, and causal events  

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
    SimulationTreeLink,
    SimContext,
    GenericDelayGenerator,
    Simulator,
)

# 1) Build your DAG timing context
events = [
    ("A", (0.0, 100.0,  0.0)),
    ("B", (10.0, 100.0, 0.0)),
    ("C", (20.0, 100.0, 0.0)),
]

# Map (source_idx, target_idx) → (link_idx, SimulationTreeLink)
links = {
    (0, 1): (0, SimulationTreeLink(minimal_duration=5.0, link_type=1)),
    (1, 2): (1, SimulationTreeLink(minimal_duration=7.0, link_type=1)),
}

# Precedences: target_idx → [(predecessor_idx, link_idx)]
precedence = [
    (1, [(0, 0)]),
    (2, [(1, 1)]),
]

ctx = SimContext(
    events=events,
    link_map=links,
    precedence_list=precedence,
    max_delay=60.0,
)

# 2) Configure your delay generator
gen = GenericDelayGenerator()
gen.add_constant(link_type=1, factor=1.5)     # 50% extra on each link
gen.add_exponential(link_type=2, lambda_=2.0, max_scale=5.0)  

# 3) Create simulator and run
sim = Simulator(ctx, gen)

# Single seeded run
result = sim.run(seed=42)
print("Realized times:", result.realized)
print("Link delays:",    result.delays)
print("Causal events:",  result.cause_event)

# Batch runs
batch = sim.run_many([1,2,3,4,5])
```  

---

## API Reference

### `SimulationTreeLink(minimal_duration: float, link_type: int)`

Encapsulates the base duration and type id of a link.

### `SimContext(events, link_map, precedence_list, max_delay)`

Container for your DAG:
- `events`: list of `(node_id, (earliest, latest, actual))`
- `link_map`: dict `(src_idx, dst_idx) → (link_idx, SimulationTreeLink)`
- `precedence_list`: list of `(target_idx, [(pred_idx, link_idx), ...])`
- `max_delay`: overall cap on delay propagation

### `GenericDelayGenerator`

Configurable delay factory:
- `.add_constant(link_type, factor)`
- `.add_exponential(link_type, lambda_, max_scale)`
- `.set_seed(seed)`

### `Simulator(context: SimContext, generator: GenericDelayGenerator)`

- `.run(seed: int) → SimResult`
- `.run_many(seeds: Sequence[int]) → List[SimResult]`

### `SimResult`

- `.realized`: `List[float]` – event times after propagation  
- `.delays`:   `List[float]` – per‑link injected delays  
- `.cause_event`: `List[int]` – which predecessor caused each event

---

## Development

```bash
# clone & install dev dependencies
git clone https://github.com/yourorg/mc_dagprop.git
cd mc_dagprop
poetry install

# build & test
poetry run pytest
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

