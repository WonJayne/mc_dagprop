# Release Notes

## Unreleased

### Build and CI

- Added explicit platform-aware native extension flags in `setup.py` so the C++ core builds cleanly across Windows, macOS, and Linux.
- Added a GitHub Actions workflow (`.github/workflows/build-wheels.yml`) to build wheel artifacts for Windows, macOS (x86_64 + arm64), and Linux using `cibuildwheel`, plus an sdist build job.
- Hardened the wheel workflow for macOS/Windows by upgrading core build tooling before wheel creation and switching cibuildwheel to the `build` frontend.
- Pinned build-backend tooling to `setuptools<77` in `pyproject.toml` to avoid the `packaging.licenses` import requirement introduced by newer setuptools in isolated wheel builds.
- Normalized `project.license` to PEP 621 table form (`{ text = "MIT" }`) for compatibility with the pinned setuptools build backend range.

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
- Integrated `max_delay` semantics into both engines:
  - Monte Carlo now caps propagated event times by
    `event.earliest + max_delay`.
  - Analytic propagation now supports `AnalyticContext.max_delay` with the same
    cap, while keeping overflow handling governed by `overflow_rule`.
- Improved test ergonomics so `pytest` works without explicitly setting
  `PYTHONPATH`.

### Notes

- `max_delay` must be non-negative for both engines.
- Existing tests were updated to use explicit high `max_delay` values where no
  cap should apply.
