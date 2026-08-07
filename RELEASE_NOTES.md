# Release Notes

## 1.0.0rc2

- Removes the artificial Python `<3.15` package-metadata bound. The package now
  declares CPython 3.12 as its minimum without rejecting later interpreters.
- Replaces the wheel workflow's explicit CPython-version allowlist with a
  `cp3*` selector, while retaining the package minimum through
  `requires-python`; source-distribution installs are tested through CPython
  3.14 without rejecting later versions that are not yet available in CI.
- Explicitly excludes free-threaded builds until their concurrency semantics
  are validated.
- Removes the build-time setuptools upper bound so future Python source builds
  can select a compatible backend release.

## 1.0.0rc1

This release candidate freezes the first supported public semantics and is
intentionally allowed to break pre-1.0 behaviour.

- Defines timestamp roles, analytic bound policies, delay units, non-negative
  activity types, unsigned 64-bit seeds, overflow errors, and propagator thread
  safety in `docs/semantics.md`.
- Defines two qualified parity domains: exact grid-aligned discrete propagation
  and statistical parity for floor-quantized continuous delays.
- Rejects duplicate predecessors and stochastic shared-ancestry reconvergence
  instead of silently treating correlated branches as independent.
- Replaces unbounded exponential and gamma rejection loops with terminating
  truncated samplers and improves distribution-tail numerics.
- Makes Monte Carlo runs reentrant by giving each run independent RNG,
  distribution, and scratch state.
- Enforces common context and delay-family validation before either backend is
  constructed, equal PMF grid steps, and finite arithmetic throughout.
- Adds randomized exhaustive-enumerator parity, quantized-continuous parity,
  malformed-input symmetry, concurrency, sanitizer, executable documentation,
  and installed wheel/source-distribution tests.
- Aligns local formatting, linting, typing, and testing with OpenBus through
  Black, Ruff, BasedPyright, and pytest; restores the `py.typed` marker only
  with a passing public consumer type check.
- Tests CPython 3.12--3.14 wheels and source distributions outside the checkout
  on the documented Linux, Windows, and macOS architectures.
- Includes the vendored C++ header and its third-party MIT notice in source and
  binary distributions, and rejects release tags that do not match the package
  version.

## 0.10.0

- Clarified P0 fixed-precedence semantics for OpenBus-derived event-activity DAGs.
- Kept `max_delay` removed from the functional API.
- Documented analytic `latest` as a hard bound and Monte Carlo `latest` as metadata only.
- Added shared `PropagationContext` and `DelayFamilyRegistry` release semantics.
- Clarified extra-delay families: base/minimal duration is deterministic, registered families add stochastic extra delay, and unregistered activity types are deterministic zero-extra-delay.
- Rejected duplicate delay-family registrations.
- Fixed PMF validation and `REMOVE` clipping so sub-probability mass propagates through chains.
- Added strict analytic grid-alignment validation.
- Hardened C++/pybind Monte Carlo validation for malformed indices, references, distributions, and parameters.
- Added OpenBus fixed-precedence integration documentation.
- Fixed the optional `[plot]` extra metadata.
- Added installed-wheel smoke testing to the release pipeline.
- Removed the shipped `py.typed` marker until static typing/stubs are release-clean.

## 0.9.0

### Highlights

- Added a harmonized naming layer for the Monte Carlo engine:
  `MonteCarloPropagator` as the primary class name (with `Simulator` kept as a compatibility alias).
- Added broad parity tests between Monte Carlo and analytic propagation with
  configurable tolerance thresholds and larger test networks.
- Added event-bound overflow handling for analytic propagation.
- Improved test ergonomics so `pytest` works without explicitly setting
  `PYTHONPATH`.
