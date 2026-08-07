from __future__ import annotations

import math
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from mc_dagprop import Activity, DagContext, Event, EventTimestamp, GenericDelayGenerator, MonteCarloPropagator


def _single_activity_context(*, duration: float = 1.0, activity_type: int = 1) -> DagContext:
    return DagContext(
        events=(Event("source", EventTimestamp(0.0, 1.0e308, 0.0)), Event("target", EventTimestamp(0.0, 1.0e308, 0.0))),
        activities={(0, 1): Activity(0, duration, activity_type)},
        precedence_list=((1, ((0, 0),)),),
    )


def _snapshot(result: object) -> tuple[tuple[float, ...], tuple[float, ...], tuple[int, ...]]:
    return (
        tuple(result.realized.tolist()),  # type: ignore[attr-defined]
        tuple(result.durations.tolist()),  # type: ignore[attr-defined]
        tuple(result.cause_event.tolist()),  # type: ignore[attr-defined]
    )


def test_extremely_truncated_exponential_completes_and_remains_bounded() -> None:
    generator = GenericDelayGenerator()
    generator.add_exponential(activity_type=1, scale=1.0, max_scale=1.0e-8)
    simulator = MonteCarloPropagator(_single_activity_context(), generator)

    extras = np.array([result.durations[0] - 1.0 for result in simulator.run_many(range(256))])

    assert np.all(extras >= 0.0)
    assert np.all(extras <= 1.0e-8)
    assert extras.mean() == pytest.approx(0.5e-8, rel=0.2)

    extreme_generator = GenericDelayGenerator()
    extreme_generator.add_exponential(activity_type=1, scale=1.0, max_scale=1.0e-300)
    extreme = MonteCarloPropagator(_single_activity_context(), extreme_generator).run(0)
    assert np.isfinite(extreme.durations[0])


def test_extremely_truncated_gamma_uses_finite_conditional_samples() -> None:
    generator = GenericDelayGenerator()
    generator.add_gamma(activity_type=1, shape=100.0, scale=1.0, max_scale=10.0)
    simulator = MonteCarloPropagator(_single_activity_context(), generator)

    extras = np.array([result.durations[0] - 1.0 for result in simulator.run_many(range(128))])

    assert np.all(np.isfinite(extras))
    assert np.all(extras >= 0.0)
    assert np.all(extras <= 10.0)
    assert np.unique(extras).size > 100


@pytest.mark.parametrize("standard_deviations", [6.0, 8.0])
def test_large_shape_gamma_fallback_completes_and_stays_bounded(standard_deviations: float) -> None:
    shape = 1.0e12
    maximum = shape - standard_deviations * math.sqrt(shape)
    generator = GenericDelayGenerator()
    generator.add_gamma(activity_type=1, shape=shape, scale=1.0, max_scale=maximum)
    simulator = MonteCarloPropagator(_single_activity_context(), generator)

    extras = np.array([result.durations[0] - 1.0 for result in simulator.run_many(range(32))])

    assert np.all(np.isfinite(extras))
    assert np.all(extras >= 0.0)
    assert np.all(extras <= maximum)
    assert np.unique(extras).size > 24


def test_deep_lower_tail_gamma_fallback_handles_underflowed_scale_ratio() -> None:
    generator = GenericDelayGenerator()
    generator.add_gamma(activity_type=1, shape=100.0, scale=1.0e308, max_scale=1.0e-320)
    simulator = MonteCarloPropagator(_single_activity_context(), generator)

    result = simulator.run(0)

    assert np.isfinite(result.durations[0])
    assert result.durations[0] == 1.0


def test_extreme_shape_gamma_rounds_unrepresentable_conditional_tail_inside_bound() -> None:
    generator = GenericDelayGenerator()
    generator.add_gamma(activity_type=1, shape=sys.float_info.max, scale=1.0, max_scale=8.0)
    simulator = MonteCarloPropagator(_single_activity_context(duration=5.0), generator)

    result = simulator.run(0)

    assert result.durations[0] == math.nextafter(45.0, 0.0)
    assert math.floor(result.durations[0] - 5.0) == 39


def test_registered_gamma_with_zero_base_duration_returns_zero() -> None:
    generator = GenericDelayGenerator()
    generator.add_gamma(activity_type=1, shape=2.0, scale=1.0, max_scale=10.0)
    simulator = MonteCarloPropagator(_single_activity_context(duration=0.0), generator)

    assert simulator.run(0).durations[0] == 0.0


def test_normal_gamma_batch_fast_path_smoke() -> None:
    generator = GenericDelayGenerator()
    generator.add_gamma(activity_type=1, shape=2.0, scale=1.0, max_scale=10.0)
    results = MonteCarloPropagator(_single_activity_context(), generator).run_many(range(2_000))

    assert len(results) == 2_000
    assert all(1.0 <= result.durations[0] <= 11.0 for result in results)


def test_gamma_upper_tail_inverse_fallback_is_finite_and_bounded() -> None:
    generator = GenericDelayGenerator()
    generator.add_gamma(activity_type=1, shape=100.0, scale=1.0, max_scale=101.0)
    simulator = MonteCarloPropagator(_single_activity_context(), generator)

    durations = np.array([result.durations[0] for result in simulator.run_many(range(4_096))])

    assert np.all(np.isfinite(durations))
    assert np.all(durations >= 1.0)
    assert np.all(durations <= 102.0)
    assert np.unique(durations).size > 4_000


def test_same_instance_is_schedule_independent_for_concurrent_calls() -> None:
    generator = GenericDelayGenerator()
    generator.add_gamma(activity_type=1, shape=2.0, scale=1.0, max_scale=10.0)
    simulator = MonteCarloPropagator(_single_activity_context(duration=10.0), generator)
    seeds = list(range(512))
    expected = {seed: _snapshot(simulator.run(seed)) for seed in seeds}
    batches = [seeds[offset::8] for offset in range(8)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent_batches = list(pool.map(simulator.run_many, batches))
        concurrent_runs = list(pool.map(simulator.run, reversed(seeds)))

    for batch_seeds, results in zip(batches, concurrent_batches, strict=True):
        assert [_snapshot(result) for result in results] == [expected[seed] for seed in batch_seeds]
    assert [_snapshot(result) for result in concurrent_runs] == [expected[seed] for seed in reversed(seeds)]


@pytest.mark.parametrize(
    ("call", "invalid_seed", "error_type"),
    [
        (lambda simulator, seed: simulator.run(seed), True, TypeError),
        (lambda simulator, seed: simulator.run(seed), -1, ValueError),
        (lambda simulator, seed: simulator.run(seed), 2**64, ValueError),
        (lambda simulator, seed: simulator.run_many([0, seed, 1]), True, TypeError),
        (lambda simulator, seed: simulator.run_many([0, seed, 1]), -1, ValueError),
        (lambda simulator, seed: simulator.run_many([0, seed, 1]), 2**64, ValueError),
    ],
)
def test_seed_domain_is_enforced(
    call: Callable[[MonteCarloPropagator, object], object], invalid_seed: object, error_type: type[Exception]
) -> None:
    simulator = MonteCarloPropagator(_single_activity_context(), GenericDelayGenerator())

    with pytest.raises(error_type, match=r"0\.\.2\*\*64-1"):
        call(simulator, invalid_seed)


def test_run_many_is_ordered_independent_runs_for_full_uint64_seed_domain() -> None:
    generator = GenericDelayGenerator()
    generator.add_empirical_absolute(activity_type=1, values=[0.0, 1.0], weights=[1.0, 1.0])
    simulator = MonteCarloPropagator(_single_activity_context(), generator)
    seeds = [2**64 - 1, 0, 42, 42, 1]

    batch = simulator.run_many(iter(seeds))

    assert [_snapshot(result) for result in batch] == [_snapshot(simulator.run(seed)) for seed in seeds]
    assert simulator.run_many([]) == []


@pytest.mark.parametrize(
    "generator_factory",
    [
        lambda: _constant_generator(factor=1.0e308),
        lambda: _relative_generator(factor=1.0e308),
        lambda: _absolute_generator(delay=1.0e308),
    ],
)
def test_non_finite_realized_duration_raises_overflow_error(
    generator_factory: Callable[[], GenericDelayGenerator],
) -> None:
    simulator = MonteCarloPropagator(_single_activity_context(duration=1.0e308), generator_factory())

    with pytest.raises(OverflowError, match=r"duration|delay"):
        simulator.run(0)


def _constant_generator(*, factor: float) -> GenericDelayGenerator:
    generator = GenericDelayGenerator()
    generator.add_constant(1, factor)
    return generator


def _relative_generator(*, factor: float) -> GenericDelayGenerator:
    generator = GenericDelayGenerator()
    generator.add_empirical_relative(1, [factor], [1.0])
    return generator


def _absolute_generator(*, delay: float) -> GenericDelayGenerator:
    generator = GenericDelayGenerator()
    generator.add_empirical_absolute(1, [delay], [1.0])
    return generator


def test_non_finite_propagated_time_raises_overflow_error() -> None:
    context = DagContext(
        events=(
            Event("source", EventTimestamp(1.0e308, 1.0e308, 1.0e308)),
            Event("target", EventTimestamp(0.0, 1.0e308, 0.0)),
        ),
        activities={(0, 1): Activity(0, 1.0e308, 99)},
        precedence_list=((1, ((0, 0),)),),
    )

    with pytest.raises(OverflowError, match="propagated time"):
        MonteCarloPropagator(context, GenericDelayGenerator()).run(0)


@pytest.mark.parametrize(
    "context",
    [
        DagContext((), {}, ()),
        DagContext((Event("event", EventTimestamp(1.0, 2.0, 0.0)),), {}, ()),
        DagContext((Event("duplicate", EventTimestamp(0.0, 1.0, 0.0)),) * 2, {}, ()),
        DagContext(
            (Event("source", EventTimestamp(0.0, 1.0, 0.0)), Event("target", EventTimestamp(0.0, 1.0, 0.0))),
            {(0, 1): Activity(0, 1.0, 1)},
            ((1, ((0, 0), (0, 0))),),
        ),
        DagContext(
            (Event("source", EventTimestamp(0.0, 1.0, 0.0)), Event("target", EventTimestamp(0.0, 1.0, 0.0))),
            {(0, 1): Activity(0, 1.0, 1)},
            (),
        ),
    ],
)
def test_low_level_context_rejects_malformed_structure(context: DagContext) -> None:
    with pytest.raises(RuntimeError):
        MonteCarloPropagator(context, GenericDelayGenerator())


def test_empirical_weights_are_scaled_before_accumulation() -> None:
    generator = GenericDelayGenerator()
    generator.add_empirical_absolute(1, (value for value in [0.0, 1.0]), (1.0e308 for _ in range(2)))
    simulator = MonteCarloPropagator(_single_activity_context(), generator)

    assert {simulator.run(seed).durations[0] for seed in range(32)} == {1.0, 2.0}


def test_generator_has_no_implicit_seed_or_deprecated_lambda_alias() -> None:
    generator = GenericDelayGenerator()

    assert not hasattr(generator, "set_seed")
    with pytest.raises(TypeError):
        generator.add_exponential(activity_type=1, lambda_=1.0, max_scale=2.0)


@pytest.mark.parametrize(
    ("method_name", "arguments"),
    [
        ("add_constant", (1, -1.0)),
        ("add_exponential", (1, 0.0, 1.0)),
        ("add_exponential", (1, 1.0, 0.0)),
        ("add_gamma", (1, 0.0, 1.0, 1.0)),
        ("add_gamma", (1, 1.0, 0.0, 1.0)),
        ("add_gamma", (1, 1.0, 1.0, 0.0)),
        ("add_empirical_absolute", (1, [], [])),
        ("add_empirical_absolute", (1, [0.0], [1.0, 1.0])),
        ("add_empirical_absolute", (1, [0.0, 1.0], [0.0, 0.0])),
    ],
)
def test_low_level_delay_families_reject_malformed_parameters(method_name: str, arguments: tuple[object, ...]) -> None:
    generator = GenericDelayGenerator()

    with pytest.raises(RuntimeError):
        getattr(generator, method_name)(*arguments)


def test_exponential_underflow_fallback_is_finite_and_bounded() -> None:
    generator = GenericDelayGenerator()
    generator.add_exponential(activity_type=1, scale=1.0e308, max_scale=1.0e-300)
    result = MonteCarloPropagator(_single_activity_context(), generator).run(0)

    assert np.isfinite(result.durations[0])
    assert 1.0 <= result.durations[0] <= 1.0 + 1.0e-300


def test_subnormal_exponential_truncation_retains_interior_samples() -> None:
    generator = GenericDelayGenerator()
    generator.add_exponential(activity_type=1, scale=1.0e308, max_scale=5.0e-16)
    simulator = MonteCarloPropagator(_single_activity_context(), generator)

    durations = np.array([result.durations[0] for result in simulator.run_many(range(10_000))])
    extras = durations - 1.0

    assert np.unique(durations).size >= 3
    assert np.all(extras >= 0.0)
    assert np.all(durations <= 1.0 + 5.0e-16)
    assert float(np.mean(extras)) == pytest.approx(2.5e-16, abs=0.3e-16)


@pytest.mark.parametrize("family", ["exponential", "gamma"])
def test_continuous_sampling_rescales_subnormal_factors_in_seconds(family: str) -> None:
    generator = GenericDelayGenerator()
    if family == "exponential":
        generator.add_exponential(activity_type=1, scale=1.0e-323, max_scale=5.0e-324)
    else:
        generator.add_gamma(activity_type=1, shape=2.0, scale=1.0e-323, max_scale=5.0e-324)
    simulator = MonteCarloPropagator(_single_activity_context(duration=1.0e308), generator)

    durations = np.array([result.durations[0] for result in simulator.run_many(range(32))])

    # The extra delay is sampled in seconds before it is added to the enormous
    # base duration. Binary64 cannot expose that sub-ULP addition publicly, but
    # the result must remain finite and within the conditioned support.
    assert np.all(np.isfinite(durations))
    assert np.all(durations == 1.0e308)


@pytest.mark.parametrize("family", ["exponential", "gamma"])
def test_continuous_sampling_avoids_overflowing_dimensionless_intermediates(family: str) -> None:
    generator = GenericDelayGenerator()
    if family == "exponential":
        generator.add_exponential(activity_type=1, scale=1.0e308, max_scale=1.0e308)
    else:
        generator.add_gamma(activity_type=1, shape=2.0, scale=1.0e308, max_scale=1.0e308)
    simulator = MonteCarloPropagator(_single_activity_context(duration=1.0e-308), generator)

    durations = np.array([result.durations[0] for result in simulator.run_many(range(64))])

    assert np.all(np.isfinite(durations))
    assert np.all(durations >= 1.0e-308)
    assert np.all(durations <= 1.0)
    assert np.unique(durations).size > 48


def _two_event_context(
    *,
    events: tuple[Event, ...] | None = None,
    activities: dict[tuple[int, int], Activity] | None = None,
    precedence_list: tuple[tuple[int, tuple[tuple[int, int], ...]], ...] = (),
) -> DagContext:
    if events is None:
        events = (Event("source", EventTimestamp(0.0, 10.0, 0.0)), Event("target", EventTimestamp(0.0, 10.0, 0.0)))
    return DagContext(events, {} if activities is None else activities, precedence_list)


@pytest.mark.parametrize(
    "context",
    [
        _two_event_context(
            events=(
                Event("source", EventTimestamp(0.0, float("inf"), 0.0)),
                Event("target", EventTimestamp(0.0, 10.0, 0.0)),
            )
        ),
        _two_event_context(activities={(0, 2): Activity(0, 1.0, 1)}),
        _two_event_context(
            events=(
                Event("source", EventTimestamp(0.0, 10.0, 0.0)),
                Event("target-a", EventTimestamp(0.0, 10.0, 0.0)),
                Event("target-b", EventTimestamp(0.0, 10.0, 0.0)),
            ),
            activities={(0, 1): Activity(0, 1.0, 1), (0, 2): Activity(0, 1.0, 1)},
        ),
        _two_event_context(activities={(0, 1): Activity(1, 1.0, 1)}),
        _two_event_context(precedence_list=((2, ()),)),
        _two_event_context(precedence_list=((1, ()), (1, ()))),
        _two_event_context(precedence_list=((1, ((2, 0),)),)),
        _two_event_context(precedence_list=((1, ((0, 0),)),)),
        _two_event_context(activities={(0, 1): Activity(0, 1.0, 1)}, precedence_list=((1, ((0, 1),)),)),
    ],
)
def test_low_level_context_rejects_each_malformed_index_relationship(context: DagContext) -> None:
    with pytest.raises(RuntimeError):
        MonteCarloPropagator(context, GenericDelayGenerator())


def test_bound_core_records_have_stable_readable_representations() -> None:
    timestamp = EventTimestamp(0.0, 10.0, 0.0)
    event = Event("event", timestamp)
    activity = Activity(0, 1.0, 1)

    assert repr(timestamp) == "EventTimestamp(earliest=0.0, latest=10.0, actual=0.0)"
    assert repr(event) == f"Event(event_id='event', timestamp={timestamp!r})"
    assert repr(activity) == "Activity(idx=0, minimal_duration=1.0, activity_type=1)"
    context = DagContext((event,), {}, ())
    assert repr(context) == f"DagContext(events=[{event!r}], activities={{}}, precedence_list=[])"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Activity(True, 1.0, 0),
        lambda: Activity(0, 1.0, False),
        lambda: Activity(0, True, 0),
        lambda: EventTimestamp(False, 1.0, 0.0),
        lambda: EventTimestamp(0.0, True, 0.0),
        lambda: EventTimestamp(0.0, 1.0, False),
    ],
)
def test_raw_scalar_constructors_reject_bool(factory: Callable[[], object]) -> None:
    with pytest.raises(TypeError, match="bool"):
        factory()


def test_raw_scalar_constructors_reject_unconvertible_values_and_huge_indices() -> None:
    with pytest.raises(TypeError, match="real number"):
        Activity(0, object(), 0)
    with pytest.raises(ValueError, match=str(2**31 - 1)):
        Activity(2**100, 0.0, 0)


@pytest.mark.parametrize(
    "register",
    [
        lambda generator: generator.add_constant(True, 0.0),
        lambda generator: generator.add_exponential(False, 1.0, 2.0),
        lambda generator: generator.add_gamma(True, 2.0, 1.0),
        lambda generator: generator.add_empirical_absolute(False, [0.0], [1.0]),
        lambda generator: generator.add_empirical_relative(True, [0.0], [1.0]),
        lambda generator: generator.add_constant(1, False),
        lambda generator: generator.add_exponential(1, True, 2.0),
        lambda generator: generator.add_gamma(1, 2.0, False),
        lambda generator: generator.add_empirical_absolute(1, [False], [1.0]),
        lambda generator: generator.add_empirical_relative(1, [0.0], [True]),
    ],
)
def test_raw_delay_registration_rejects_bool(register: Callable[[GenericDelayGenerator], None]) -> None:
    with pytest.raises(TypeError, match="bool"):
        register(GenericDelayGenerator())


def test_raw_dag_context_indices_reject_bool() -> None:
    events = (Event("source", EventTimestamp(0.0, 1.0, 0.0)), Event("target", EventTimestamp(0.0, 1.0, 0.0)))
    activity = Activity(0, 1.0, 0)

    with pytest.raises(TypeError, match="bool"):
        DagContext(events, {(False, 1): activity}, ((1, ((0, 0),)),))
    with pytest.raises(TypeError, match="bool"):
        DagContext(events, {(0, 1): activity}, ((True, ((0, 0),)),))
    with pytest.raises(TypeError, match="bool"):
        DagContext(events, {(0, 1): activity}, ((1, ((0, False),)),))


@pytest.mark.parametrize(
    ("activities", "precedence_list", "match"),
    [
        ({(0,): Activity(0, 1.0, 0)}, (), "activity keys must contain exactly two"),
        ({}, ((1,),), "precedence entries must contain a target"),
        ({(0, 1): Activity(0, 1.0, 0)}, ((1, ((0,),)),), "predecessor entries must contain a source"),
    ],
)
def test_raw_dag_context_rejects_malformed_tuple_shapes(
    activities: object, precedence_list: object, match: str
) -> None:
    events = (Event("source", EventTimestamp(0.0, 1.0, 0.0)), Event("target", EventTimestamp(0.0, 1.0, 0.0)))

    with pytest.raises(ValueError, match=match):
        DagContext(events, activities, precedence_list)


def test_low_level_activity_types_use_signed_32_bit_domain() -> None:
    maximum = 2**31 - 1
    Activity(maximum, 0.0, maximum)
    generator = GenericDelayGenerator()
    generator.add_constant(maximum, 0.0)

    with pytest.raises(ValueError, match=str(maximum)):
        Activity(maximum + 1, 0.0, 0)
    with pytest.raises(ValueError, match=str(maximum)):
        Activity(0, 0.0, maximum + 1)
    with pytest.raises(ValueError, match=str(maximum)):
        GenericDelayGenerator().add_constant(maximum + 1, 0.0)


def test_public_model_values_are_immutable() -> None:
    timestamp = EventTimestamp(0.0, 1.0, 0.0)
    event = Event("event", timestamp)
    activity = Activity(0, 1.0, 0)
    context = DagContext((event,), {}, ())

    for instance, attribute, replacement in (
        (timestamp, "earliest", 1.0),
        (event, "event_id", "changed"),
        (activity, "minimal_duration", 2.0),
        (context, "events", ()),
    ):
        with pytest.raises(AttributeError):
            setattr(instance, attribute, replacement)


def test_simulation_result_runtime_types_and_readonly_properties() -> None:
    simulator = MonteCarloPropagator(_single_activity_context(), GenericDelayGenerator())

    result = simulator.run(0)

    assert simulator.node_count() == 2
    assert simulator.activity_count() == 1
    assert result.realized.dtype == np.dtype(np.float64)
    assert result.durations.dtype == np.dtype(np.float64)
    assert result.cause_event.dtype == np.dtype(np.int32)
    result_buffer = memoryview(result)
    assert result_buffer.format == "d"
    assert result_buffer.shape == (2,)
    assert list(result_buffer) == result.realized.tolist()

    for attribute in ("realized", "durations", "cause_event"):
        with pytest.raises(AttributeError):
            setattr(result, attribute, np.array([], dtype=np.float64))
