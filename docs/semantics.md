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

TODO: full Büker-style route-conflict handling requires conditional convolution
rather than only marginal convolution/maximum operations. That conflict-aware
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
Registering a second family for the same activity type is an error.
