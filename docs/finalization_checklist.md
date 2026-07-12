# Finalization checklist

This note records the P3-P6 hardening status for the v0-style propagation API.

| Item | Status | Notes |
| --- | --- | --- |
| Duplicate delay-family registration | Implemented | `DelayFamilyRegistry` and the compiled generator reject a second stochastic family for the same activity type with an error naming the activity type. |
| Deterministic unregistered activity types | Implemented | Unregistered activity types contribute deterministic zero extra delay, so the activity increment is its minimal/base duration. |
| `activity_type = -1` handling | Implemented | `-1` is documented as a reserved deterministic no-delay sentinel. User stochastic registration for `-1` is rejected; leaving it unregistered remains deterministic. |
| Common frontend construction | Implemented | `PropagationContext` plus `DelayFamilyRegistry` can construct both `MonteCarloPropagator.from_context(...)` and `AnalyticPropagator.from_context(...)`. |
| Exponential `scale` naming | Implemented | `scale` is the preferred exponential mean parameter in docs and examples. |
| Exponential `lambda_` deprecation | Implemented with compatibility alias | `lambda_` remains available only as a deprecated compatibility alias and emits `DeprecationWarning`. |
| Analytic hard latest | Implemented | Analytic propagation clips event PMFs to `[earliest, latest]` according to explicit clipping policies. |
| Monte Carlo latest metadata | Implemented | Monte Carlo propagation does not cap realized times at `latest`. |
| Discrete PMF validation | Implemented | PMFs validate finite grid-aligned support, finite non-negative probabilities, positive integer step, nonzero normalized mass by default, and explicit sub-probability use. Duplicate support is aggregated deterministically. |
| Clipping policies | Implemented | `TRUNCATE` moves out-of-bound mass to the nearest boundary, `REMOVE` returns explicit sub-probability PMFs and records removed mass, and `REDISTRIBUTE` conditionalizes retained mass. Empty retained support under `REMOVE` is a clear error. |
| Conditional convolution primitive | Implemented as low-level primitive | `DiscretePMF.conditional_convolve_sum_le(...)` provides exact conditional sum convolution for tests and documentation. It is intentionally not wired into the main marginal propagator. |
| Full route-conflict handling | Documented out of scope | Büker-style route-conflict propagation, interlinking modelling, priorities, and exact joint-distribution propagation for reconvergent correlated DAGs remain follow-up model extensions. |
