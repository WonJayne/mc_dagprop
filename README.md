# mc_dagprop

[![PyPI version](https://img.shields.io/pypi/v/mc_dagprop.svg)](https://pypi.org/project/mc_dagprop/) [![Python Versions](https://img.shields.io/pypi/pyversions/mc_dagprop.svg)](https://pypi.org/project/mc_dagprop/) [![License](https://img.shields.io/pypi/l/mc_dagprop.svg)](./LICENSE)

**mc_dagprop** is a fast, Monte Carlo–style propagation simulator for directed acyclic graphs (DAGs), written in C++ with Python bindings via **pybind11**.  
It allows you to model timing networks (timetables, precedence graphs, etc.) and inject user‐defined delay distributions on links.  

---

## Features

- **Lightweight & high‑performance** core in C++  
- Expose a simple Python API via **poetry** or **pip**  
- Define custom delay distributions per link‐type:
  - **Constant** (linear scaling)
  - **Exponential** (with cutoff)
  - Easily extendable for Weibull, Gamma, …
- Single‐run (`run(seed)`) and batch‐run (`run_many([seeds])`) support  
- Returns a **SimResult** struct: realized times, link delays, and causal events  

---

## Installation

```bash
# with poetry
poetry add mc_dagprop

# or with pip
pip install mc_dagprop
