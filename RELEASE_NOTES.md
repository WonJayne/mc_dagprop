# Release Notes

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

## Unreleased

### Build and CI

- Added explicit platform-aware native extension flags in `setup.py` so the C++ core builds cleanly across Windows, macOS, and Linux.
- Added a GitHub Actions workflow (`.github/workflows/build-wheels.yml`) to build wheel artifacts for Windows, macOS (x86_64 + arm64), and Linux using `cibuildwheel`, plus an sdist build job.
- Hardened the wheel workflow for macOS/Windows by upgrading core build tooling before wheel creation and switching cibuildwheel to the `build` frontend.
- Pinned build-backend tooling to `setuptools<77` in `pyproject.toml` to avoid the `packaging.licenses` import requirement introduced by newer setuptools in isolated wheel builds.
- Normalized `project.license` to PEP 621 table form (`{ text = "MIT" }`) for compatibility with the pinned setuptools build backend range.
- Fixed `ApproxNormalDistribution::reset()` in `_custom_rng.hpp` to avoid references to non-existent cached-normal members, resolving clang/macOS wheel compile failures.
- Updated `_core.cpp` constructor moves to use explicit `std::move` to satisfy stricter clang diagnostics in macOS wheel builds.

### Documentation

- Documented the cross-platform wheel build workflow and local source-build command in the README.

### Documentation

- Synchronized the README with the current public API: analytic examples now use `AnalyticActivity`, explicit flow rules, and the `step` field name from `AnalyticContext`.
- Updated package-structure notes to reflect the current modules (`demo/` examples) and removed references to non-existent utilities.

## 0.9.0

### Highlights

- Added a harmonized naming layer for the Monte Carlo engine:
  `MonteCarloPropagator` as the primary class name (with `Simulator` kept as a compatibility alias).
- Added broad parity tests between Monte Carlo and analytic propagation with
  configurable tolerance thresholds and larger test networks.
- Added event-bound overflow handling for analytic propagation.
- Improved test ergonomics so `pytest` works without explicitly setting
  `PYTHONPATH`.

