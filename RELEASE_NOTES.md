# Release Notes

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
