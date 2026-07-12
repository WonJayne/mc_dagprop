# Propagation semantics

`mc_dagprop` provides two propagation backends with deliberately different
semantic roles.

## Analytic backend

The analytic propagator is a discrete-distribution implementation of the
Büker/OnTime-style analytical delay-propagation view. It propagates probability
mass functions (PMFs) over a DAG using marginal operations: edge delays are
convolved with predecessor event-time PMFs, and multiple predecessors are
combined with a marginal maximum operation.

For analytic propagation, each event's `latest` timestamp is a hard upper bound.
Analytic event distributions are bounded to `[event.earliest, event.latest]`
according to the configured underflow and overflow rules.

Reconvergent DAGs with shared stochastic ancestry can violate the independence
assumption implicit in marginal maximum formation. In those cases, analytic and
Monte Carlo outputs may differ. This is a documented limitation of the current
marginal implementation, not an implementation bug.

A low-level conditional convolution primitive is available for exact, small
PMF calculations, but it is deliberately not wired into the main propagator.
Full Büker-style route-conflict handling requires a complete conflict and
interlinking model in addition to conditional convolution; that conflict-aware
extension is intentionally out of scope for this pass.

## Monte Carlo backend

The Monte Carlo propagator samples realised edge delays and propagates realised
event times. It is a validation/reference backend for controlled cases, not a
semantic requirement for equivalence with the analytic backend on arbitrary
DAGs.

For Monte Carlo propagation, `latest` is semantic metadata only. Realised event
times are not capped by `latest`.

## Delay-family registration

Unregistered activity types are deterministic and add no stochastic extra delay:
the activity contributes only its configured minimal duration.

Each stochastic delay family may be registered at most once per activity type.
Registering a second family for the same activity type is an error. The
activity type `-1` is reserved as an internal deterministic no-delay sentinel:
users must not register stochastic delay families for `-1`, and an unregistered
`-1` activity remains deterministic like any other unregistered type.

## Activity durations and discrete PMFs

Every activity has a deterministic minimal duration. Registered delay families
model stochastic extra delay added on top of that minimal duration; unregistered
activity types have zero stochastic extra delay and therefore contribute only the
minimal duration. Monte Carlo samples realised edge durations as minimal duration
plus sampled extra delay. Shared frontend analytic construction converts the
same extra-delay families into full edge-increment PMFs by shifting them by the
minimal duration.

Analytic PMFs are strict discrete-grid objects: support values must be finite,
grid-aligned integer-step values and probabilities must be finite,
non-negative, and normalized unless a policy explicitly creates a documented
sub-probability result. Analytic `latest` remains a hard clipping bound; Monte
Carlo `latest` remains metadata and does not cap realised samples.


## Clipping policies

Analytic clipping policies have explicit mass semantics:

- `TRUNCATE` moves mass below/above the bound to the nearest boundary bin. If the
  boundary bin is missing it is inserted; if it already exists the mass is
  merged. Total PMF mass is preserved.
- `REMOVE` removes out-of-bound mass without renormalizing retained support. The
  resulting PMF is an explicit sub-probability PMF, and removed mass is reported
  as event underflow/overflow. If no support remains inside the event window, a
  clear error is raised instead of returning an empty PMF.
- `REDISTRIBUTE` removes out-of-bound mass and renormalizes/conditionalizes the
  retained support. If no inside support exists, mass is anchored at the lower
  bound.

Use `step=1` for exact unit tests. A coarser `step=3` can be practical for
examples or exploratory runs when that grid is appropriate for the timetable.

## Reproducibility

The recommended Monte Carlo reproducibility mechanism is passing `seed` to
`MonteCarloPropagator.run(seed=...)` or a deterministic seed sequence to
`run_many(...)`. The lower-level `GenericDelayGenerator.set_seed(...)` remains
available for direct generator use; a per-run seed resets the propagator's
reusable generator state for that run.

`max_delay` is no longer part of the public semantics or API.
