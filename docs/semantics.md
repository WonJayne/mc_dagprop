# Propagation semantics

This document is the public behavioural contract for `mc_dagprop` 1.0 release
candidates. The shared `PropagationContext` and `DelayFamilyRegistry` are the
supported construction path when a model must be evaluated by both backends.

## Timestamps and event bounds

`EventTimestamp(earliest, latest, actual)` has one interpretation:

- `earliest` is the deterministic release time. Every Monte Carlo event starts
  at this value, and every analytic root is a Dirac mass at this value.
- `latest` is a hard upper bound in analytic propagation. It is metadata in
  Monte Carlo propagation and never clips a sampled event time.
- `actual` is validated reference metadata for callers. Neither backend uses it
  as a propagation input.

All timestamp values must be finite and satisfy
`earliest <= actual <= latest`. Analytic timestamps and activity increments must
also align with the positive integer `step`, expressed in seconds.

For a non-root event, propagation applies the maximum of its release time and
all incoming predecessor completion times. Analytic clipping implements this
lower release bound according to the configured underflow policy.

## Activities and delay units

Activity indices are contiguous integers starting at zero. Activity types are
non-negative integers. A type without a registered family has deterministic
zero extra delay; the activity still contributes its `minimal_duration`.
Negative activity types, duplicate family registrations, duplicate
predecessors, dangling activities, and cycles are rejected.

`minimal_duration` and empirical absolute delays are seconds. Other family
parameters are dimensionless:

| Family | Extra delay |
|---|---|
| Unregistered | `0` |
| Constant | `minimal_duration * factor` |
| Empirical absolute | sampled `value` in seconds |
| Empirical relative (low-level API) | `minimal_duration * sampled factor`; factors are dimensionless |
| Exponential | `minimal_duration * X`, where `X` has exponential mean parameter `scale` and is conditioned on `X <= max_scale` |
| Gamma | `minimal_duration * X`, where `X ~ Gamma(shape, scale)` and is conditioned on `X <= max_scale` |

The shared `DelayFamilyRegistry.add_empirical(...)` API registers empirical
absolute extra delays in seconds. Relative empirical factors are available only
through the low-level `GenericDelayGenerator.add_empirical_relative(...)` API.

All values and parameters must be finite; durations, factors, empirical values,
and weights must be non-negative; continuous distribution parameters must be
positive. A finite input combination that overflows during multiplication,
sampling, or propagation raises `OverflowError` instead of producing infinity.

## Analytic PMFs and bounds

Analytic PMFs live on a zero-origin integer grid. All binary PMF operations
require equal grid steps. Continuous families are converted to the distribution
of `floor(sample / step) * step`, conditional on their exact finite cutoff; a
partial final bin is retained when the cutoff is not grid-aligned.

Analytic bound policies have explicit mass semantics:

- `TRUNCATE` moves outside mass to the nearest bound and preserves total mass.
- `REMOVE` drops outside mass without renormalizing and reports it as event
  underflow or overflow. If all mass is removed, propagation continues with an
  explicit zero-mass sub-PMF anchored at the relevant bound or bounds.
- `REDISTRIBUTE` conditions on retained support. If no inside support exists,
  underflow is anchored at the lower bound and overflow at the upper bound.

## Qualified backend equivalence

Universal analytic/Monte Carlo equivalence is not claimed. Call
`validate_equivalence_domain(context, registry, step=..., mode=...)` before
relying on a parity guarantee.

`EquivalenceMode.EXACT_DISCRETE` requires:

1. grid-aligned deterministic or empirical activity increments;
2. independent stochastic activity ancestry on all branches entering a merge;
3. no duplicate predecessor; and
4. nonbinding analytic event bounds.

Within this domain, analytic event PMFs equal exhaustive enumeration up to
floating-point roundoff. Monte Carlo frequencies converge statistically to the
same PMFs.

`EquivalenceMode.QUANTIZED_CONTINUOUS` has the same structural and nonbinding-
bound requirements but also permits exponential and gamma families. Parity is
defined by flooring every sampled stochastic activity extra delay to the
analytic grid before event propagation. It is statistical, not sample-by-
sample. Flooring only a final event time is not equivalent on a multi-activity
path because floor quantization is not additive.

Reconvergent branches that share a stochastic ancestor are rejected by analytic
validation. Propagating only the incoming marginals would incorrectly treat
those correlated branches as independent. Deterministic shared ancestry remains
valid.

## Seeds and thread safety

Activity types, activity indices, and event indices use the non-negative
signed-32-bit domain `[0, 2**31 - 1]`. Boolean values are rejected rather than
coerced to integers. Seeds are a separate unsigned 64-bit domain.

`run(seed)` accepts an integer in `[0, 2**64 - 1]`. Within a fixed package build
and platform, the model and seed uniquely determine a result. Cross-platform
bit identity is not promised because standard-library distribution algorithms
may differ. `run_many(seeds)` preserves input order and is exactly the ordered
equivalent of independent `run(seed)` calls, including repeated seeds.

Each run owns its RNG, distribution state, and scratch buffers. Concurrent
`run` and `run_many` calls on the same `MonteCarloPropagator` are therefore safe,
schedule-independent, and reproducible within that build/platform. Contexts and
propagators are immutable after construction. Mutating a `DelayFamilyRegistry`
concurrently with model construction is not supported.
