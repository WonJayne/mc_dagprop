# 1.0 release-candidate checklist

This checklist is executable release policy. A final 1.0 tag may be created
only after the release-candidate branch satisfies every gate.

| Gate | Acceptance condition |
|---|---|
| Shared validation | Both backends are constructed from one validated `PropagationContext` and `DelayFamilyRegistry`; malformed-input symmetry tests pass. |
| Timestamp and bound semantics | `docs/semantics.md`, public docstrings, README examples, and implementation agree. |
| Delay semantics | Units, non-negative activity types, registrations, and finite overflow behaviour are tested for both backends. |
| Exact equivalence | Randomized aligned-discrete models equal exhaustive enumeration; correlated stochastic reconvergence and binding bounds are rejected by the equivalence validator. |
| Continuous parity | Quantized exponential and gamma results agree statistically with Monte Carlo, including non-grid-aligned truncation cutoffs. |
| Monte Carlo safety | Extreme truncation terminates, explicit seeds are reproducible within a fixed build/platform, and concurrent calls on one propagator are race-free and schedule-independent. |
| PMF and distribution numerics | Mixed grid steps fail; large-shape gamma and large-scale exponential reference cases meet their numeric tolerances. |
| Static checks | Black, Ruff, and BasedPyright pass; an installed public consumer type-checks before `py.typed` is shipped. |
| Coverage and sanitizers | Branch-aware Python coverage is at least 85%; the Ubuntu gcovr job covers at least 90% of project C++ lines and 50% of branches (excluding vendored UTL); AddressSanitizer and UndefinedBehaviorSanitizer tests pass. |
| Executable documentation | Every README Python block and all non-interactive demos execute in CI. |
| Distribution tests | Wheels and the source distribution install and run outside the checkout on every documented Python/platform target. |
| Packaging and legal | The sdist contains C++ headers; wheel/sdist metadata and contents pass smoke checks; third-party notices ship in both formats. |
| Release identity | `v<version>` exactly matches `project.version`; the check runs before publish jobs. |
| Release sequence | Publish `1.0.0rc2`, validate its artifacts from PyPI, then decide separately whether to create the final `1.0.0` tag. |

No migration guide is required for the pre-1.0 API because breaking cleanup is
explicitly allowed for this release candidate.
