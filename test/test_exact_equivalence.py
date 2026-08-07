from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
import pytest

import mc_dagprop as mc
from mc_dagprop.analytic import AnalyticActivity, AnalyticContext, DiscretePMF, OverflowRule, UnderflowRule


def _random_probability(rng: np.random.Generator) -> np.ndarray:
    first_probability = float(rng.uniform(0.15, 0.85))
    return np.array([first_probability, 1.0 - first_probability])


def _random_exact_context(seed: int) -> AnalyticContext:
    """Build an independently branching DAG in the qualified exact domain."""
    rng = np.random.default_rng(seed)
    branch_count = int(rng.integers(1, 4))
    events: list[mc.Event] = []
    activities: dict[tuple[int, int], tuple[int, AnalyticActivity]] = {}
    precedence: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    branch_ends: list[int] = []
    activity_index = 0

    for branch in range(branch_count):
        root_time = float(rng.integers(0, 4))
        root = len(events)
        events.append(mc.Event(f"root-{branch}", mc.EventTimestamp(root_time, 200.0, root_time)))
        previous = root
        for position in range(int(rng.integers(1, 4))):
            target = len(events)
            events.append(mc.Event(f"branch-{branch}-{position}", mc.EventTimestamp(0.0, 200.0, 0.0)))
            support = np.sort(rng.choice(np.arange(5, dtype=float), size=2, replace=False))
            pmf = DiscretePMF(support, _random_probability(rng), step=1)
            activities[(previous, target)] = (activity_index, AnalyticActivity(activity_index, pmf))
            precedence.append((target, ((previous, activity_index),)))
            previous = target
            activity_index += 1
        branch_ends.append(previous)

    if len(branch_ends) > 1:
        target = len(events)
        events.append(mc.Event("merge", mc.EventTimestamp(0.0, 200.0, 0.0)))
        merge_predecessors: list[tuple[int, int]] = []
        for source in branch_ends:
            support = np.sort(rng.choice(np.arange(4, dtype=float), size=2, replace=False))
            pmf = DiscretePMF(support, _random_probability(rng), step=1)
            activities[(source, target)] = (activity_index, AnalyticActivity(activity_index, pmf))
            merge_predecessors.append((source, activity_index))
            activity_index += 1
        precedence.append((target, tuple(merge_predecessors)))

    return AnalyticContext(
        events=tuple(events),
        activities=activities,
        precedence_list=tuple(precedence),
        step=1,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.TRUNCATE,
    )


def _enumerate_exact_event_pmfs(context: AnalyticContext) -> tuple[dict[float, float], ...]:
    """Enumerate every independent activity outcome in a small analytic DAG."""
    activities_by_index = sorted((edge.idx, edge.pmf) for _, edge in context.activities.values())
    choices = [tuple(zip(pmf.values, pmf.probabilities, strict=True)) for _, pmf in activities_by_index]
    predecessors_by_target = dict(context.precedence_list)
    accumulated: list[dict[float, float]] = [{} for _ in context.events]

    for outcome in itertools.product(*choices):
        durations = {
            activity_index: float(outcome[index][0]) for index, (activity_index, _) in enumerate(activities_by_index)
        }
        probability = math.prod(float(selected[1]) for selected in outcome)
        realized = [float(event.timestamp.earliest) for event in context.events]
        for target, event in enumerate(context.events):
            predecessors = predecessors_by_target.get(target, ())
            if predecessors:
                realized[target] = max(
                    float(event.timestamp.earliest),
                    *(realized[source] + durations[activity_index] for source, activity_index in predecessors),
                )
            accumulated[target][realized[target]] = accumulated[target].get(realized[target], 0.0) + probability
    return tuple(accumulated)


@pytest.mark.parametrize("seed", range(32))
def test_randomized_analytic_matches_exact_enumerator(seed: int) -> None:
    context = _random_exact_context(seed)

    analytic = mc.create_analytic_propagator(context).run()
    exact = _enumerate_exact_event_pmfs(context)

    for event_result, exact_probabilities in zip(analytic, exact, strict=True):
        analytic_probabilities = {
            float(value): float(probability)
            for value, probability in zip(event_result.pmf.values, event_result.pmf.probabilities, strict=True)
            if probability > 1.0e-15
        }
        assert analytic_probabilities.keys() == exact_probabilities.keys()
        for value, expected_probability in exact_probabilities.items():
            assert analytic_probabilities[value] == pytest.approx(expected_probability, abs=2.0e-14)


def _shared_model_from_analytic(
    analytic_context: AnalyticContext,
) -> tuple[mc.PropagationContext, mc.DelayFamilyRegistry]:
    registry = mc.DelayFamilyRegistry()
    activities: dict[tuple[int, int], mc.Activity] = {}
    for endpoints, (activity_index, analytic_activity) in analytic_context.activities.items():
        activity_type = activity_index
        activities[endpoints] = mc.Activity(activity_index, 0.0, activity_type)
        registry.add_empirical(
            activity_type, analytic_activity.pmf.values.tolist(), analytic_activity.pmf.probabilities.tolist()
        )
    context = mc.PropagationContext(analytic_context.events, activities, analytic_context.precedence_list)
    return context, registry


@pytest.mark.parametrize("model_seed", range(8))
def test_randomized_shared_model_matches_exact_enumerator_and_monte_carlo(model_seed: int) -> None:
    analytic_context = _random_exact_context(model_seed)
    exact = _enumerate_exact_event_pmfs(analytic_context)
    shared_context, registry = _shared_model_from_analytic(analytic_context)
    mc.validate_equivalence_domain(shared_context, registry, step=1)
    simulator = mc.MonteCarloPropagator.from_context(shared_context, registry)

    sample_count = 5_000
    results = simulator.run_many(model_seed * sample_count + sample for sample in range(sample_count))
    samples_by_event = np.array([result.realized for result in results])

    for event_index, exact_probabilities in enumerate(exact):
        samples = samples_by_event[:, event_index]
        assert set(np.unique(samples)).issubset(exact_probabilities)
        for value, expected_probability in exact_probabilities.items():
            observed_probability = float(np.mean(samples == value))
            assert observed_probability == pytest.approx(expected_probability, abs=0.04)
        exact_mean = sum(value * probability for value, probability in exact_probabilities.items())
        assert float(np.mean(samples)) == pytest.approx(exact_mean, abs=0.15)


def _continuous_chain_context(duration: float = 5.0) -> mc.PropagationContext:
    events = (
        mc.Event("root", mc.EventTimestamp(0.0, 100.0, 0.0)),
        mc.Event("middle", mc.EventTimestamp(0.0, 100.0, 0.0)),
        mc.Event("target", mc.EventTimestamp(0.0, 100.0, 0.0)),
    )
    return mc.PropagationContext(
        events,
        {
            (0, 1): mc.Activity(idx=0, minimal_duration=duration, activity_type=1),
            (1, 2): mc.Activity(idx=1, minimal_duration=duration, activity_type=2),
        },
        ((1, ((0, 0),)), (2, ((1, 1),))),
    )


def _minimal_analytic_context(
    *,
    activity_key: tuple[int, int] = (0, 1),
    mapping_index: int = 0,
    activity_index: int = 0,
    precedence_target: int = 1,
    predecessor_source: int = 0,
    predecessor_activity: int = 0,
    step: int = 1,
    underflow_rule: UnderflowRule = UnderflowRule.TRUNCATE,
    overflow_rule: OverflowRule = OverflowRule.TRUNCATE,
) -> AnalyticContext:
    events = (
        mc.Event("root", mc.EventTimestamp(0.0, 10.0, 0.0)),
        mc.Event("target", mc.EventTimestamp(0.0, 10.0, 0.0)),
    )
    activity = AnalyticActivity(activity_index, DiscretePMF.delta(1.0, step=1))
    return AnalyticContext(
        events,
        {activity_key: (mapping_index, activity)},
        ((precedence_target, ((predecessor_source, predecessor_activity),)),),
        step,
        underflow_rule,
        overflow_rule,
    )


@pytest.mark.parametrize(
    "register_family",
    [
        lambda registry: (
            registry.add_exponential(1, scale=0.4, max_scale=1.05),
            registry.add_exponential(2, scale=0.4, max_scale=1.05),
        ),
        lambda registry: (
            registry.add_gamma(1, shape=2.0, scale=0.25, max_scale=1.05),
            registry.add_gamma(2, shape=2.0, scale=0.25, max_scale=1.05),
        ),
    ],
    ids=["exponential", "gamma"],
)
def test_continuous_families_match_after_per_activity_floor_quantization(
    register_family: Callable[[mc.DelayFamilyRegistry], None],
) -> None:
    step = 1
    base_duration = 5.0
    context = _continuous_chain_context(duration=base_duration)
    registry = mc.DelayFamilyRegistry()
    register_family(registry)
    mc.validate_equivalence_domain(context, registry, step=step, mode=mc.EquivalenceMode.QUANTIZED_CONTINUOUS)
    analytic = (
        mc.AnalyticPropagator.from_context(
            context,
            registry,
            step=step,
            underflow_rule=mc.UnderflowRule.TRUNCATE,
            overflow_rule=mc.OverflowRule.TRUNCATE,
        )
        .run()[2]
        .pmf
    )
    simulator = mc.MonteCarloPropagator.from_context(context, registry)

    results = simulator.run_many(range(25_000))
    sampled_extra_delays = np.array([result.durations - base_duration for result in results])
    quantized_durations = 2.0 * base_duration + np.floor(sampled_extra_delays / step).sum(axis=1) * step
    final_only_quantization = 2.0 * base_duration + np.floor(sampled_extra_delays.sum(axis=1) / step) * step
    assert np.any(quantized_durations != final_only_quantization)
    empirical_probabilities = np.array(
        [np.mean(quantized_durations == value) for value in analytic.values], dtype=float
    )

    np.testing.assert_allclose(empirical_probabilities, analytic.probabilities, atol=0.012, rtol=0.0)
    analytic_mean = float(np.dot(analytic.values, analytic.probabilities))
    assert float(np.mean(quantized_durations)) == pytest.approx(analytic_mean, abs=0.035)


def test_deeply_truncated_gamma_matches_per_activity_floor_quantization() -> None:
    base_duration = 5.0
    context = mc.PropagationContext(
        (mc.Event("root", mc.EventTimestamp(0.0, 100.0, 0.0)), mc.Event("target", mc.EventTimestamp(0.0, 100.0, 0.0))),
        {(0, 1): mc.Activity(idx=0, minimal_duration=base_duration, activity_type=1)},
        ((1, ((0, 0),)),),
    )
    registry = mc.DelayFamilyRegistry()
    registry.add_gamma(1, shape=500.0, scale=0.2, max_scale=2.0)
    mc.validate_equivalence_domain(context, registry, step=1, mode=mc.EquivalenceMode.QUANTIZED_CONTINUOUS)
    analytic = (
        mc.AnalyticPropagator.from_context(
            context, registry, step=1, underflow_rule=mc.UnderflowRule.TRUNCATE, overflow_rule=mc.OverflowRule.TRUNCATE
        )
        .run()[1]
        .pmf
    )
    simulator = mc.MonteCarloPropagator.from_context(context, registry)

    sampled_durations = np.array([result.durations[0] for result in simulator.run_many(range(512))])
    quantized_durations = base_duration + np.floor((sampled_durations - base_duration) / 1.0)

    assert analytic.probabilities[-1] > 1.0 - 1.0e-12
    assert np.mean(quantized_durations == analytic.values[-1]) > 0.99


def test_duplicate_predecessor_sources_are_rejected() -> None:
    events = (
        mc.Event("root", mc.EventTimestamp(0.0, 10.0, 0.0)),
        mc.Event("target", mc.EventTimestamp(0.0, 10.0, 0.0)),
    )
    activity = AnalyticActivity(0, DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5]), step=1))
    context = AnalyticContext(
        events, {(0, 1): (0, activity)}, ((1, ((0, 0), (0, 0))),), 1, UnderflowRule.TRUNCATE, OverflowRule.TRUNCATE
    )

    with pytest.raises(ValueError, match="duplicate predecessor source 0"):
        mc.create_analytic_propagator(context)


@pytest.mark.parametrize(
    ("context", "label"),
    [
        (_minimal_analytic_context(activity_key=(True, 1)), "activity source index"),
        (_minimal_analytic_context(activity_key=(0, True)), "activity target index"),
        (_minimal_analytic_context(mapping_index=True), "activity index"),
        (_minimal_analytic_context(activity_index=True), "analytic activity index"),
        (_minimal_analytic_context(precedence_target=True), "precedence target index"),
        (_minimal_analytic_context(predecessor_source=True), "predecessor source index"),
        (_minimal_analytic_context(predecessor_activity=True), "predecessor activity index"),
    ],
)
def test_low_level_analytic_context_rejects_boolean_indices(context: AnalyticContext, label: str) -> None:
    with pytest.raises(TypeError, match=label):
        mc.create_analytic_propagator(context)


@pytest.mark.parametrize(
    "context",
    [
        _minimal_analytic_context(activity_key=(2**31, 1)),
        _minimal_analytic_context(activity_key=(0, 2**31)),
        _minimal_analytic_context(mapping_index=2**31),
        _minimal_analytic_context(activity_index=2**31),
        _minimal_analytic_context(precedence_target=2**31),
        _minimal_analytic_context(predecessor_source=2**31),
        _minimal_analytic_context(predecessor_activity=2**31),
    ],
)
def test_low_level_analytic_context_rejects_indices_above_signed_32_bit(context: AnalyticContext) -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        mc.create_analytic_propagator(context)


def test_low_level_analytic_context_rejects_boolean_step_and_rules() -> None:
    root = (mc.Event("root", mc.EventTimestamp(0.0, 10.0, 0.0)),)

    with pytest.raises(TypeError, match="step must be an integer"):
        mc.create_analytic_propagator(
            AnalyticContext(root, {}, (), True, UnderflowRule.TRUNCATE, OverflowRule.TRUNCATE)
        )
    with pytest.raises(TypeError, match="underflow_rule"):
        mc.create_analytic_propagator(AnalyticContext(root, {}, (), 1, True, OverflowRule.TRUNCATE))
    with pytest.raises(TypeError, match="overflow_rule"):
        mc.create_analytic_propagator(AnalyticContext(root, {}, (), 1, UnderflowRule.TRUNCATE, True))


def test_analytic_context_snapshots_caller_owned_collections() -> None:
    events = [
        mc.Event("root", mc.EventTimestamp(0.0, 10.0, 0.0)),
        mc.Event("target", mc.EventTimestamp(0.0, 10.0, 0.0)),
    ]
    activity = AnalyticActivity(0, DiscretePMF.delta(1.0, step=1))
    activities = {(0, 1): (0, activity)}
    predecessors = [(1, [(0, 0)])]
    context = AnalyticContext(events, activities, predecessors, 1, UnderflowRule.TRUNCATE, OverflowRule.TRUNCATE)

    events.clear()
    activities.clear()
    predecessors[0][1].clear()

    assert len(context.events) == 2
    assert dict(context.activities) == {(0, 1): (0, activity)}
    assert context.precedence_list == ((1, ((0, 0),)),)
    with pytest.raises(TypeError):
        context.activities[(1, 0)] = (1, activity)  # type: ignore[index]
    result = mc.create_analytic_propagator(context).run()
    np.testing.assert_array_equal(result[1].pmf.values, [1.0])


def test_discrete_pmf_snapshots_inputs_and_exposes_read_only_arrays() -> None:
    values = np.array([0.0, 1.0])
    probabilities = np.array([0.25, 0.75])
    pmf = DiscretePMF(values, probabilities, step=1)

    values[:] = 9.0
    probabilities[:] = 0.5

    np.testing.assert_array_equal(pmf.values, [0.0, 1.0])
    np.testing.assert_array_equal(pmf.probabilities, [0.25, 0.75])
    with pytest.raises(ValueError, match="read-only"):
        pmf.values[0] = 2.0
    with pytest.raises(ValueError, match="read-only"):
        pmf.probabilities[0] = 1.0
    with pytest.raises(ValueError, match="WRITEABLE"):
        pmf.values.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        pmf.probabilities.setflags(write=True)


def test_discrete_pmf_rejects_boolean_grid_steps() -> None:
    with pytest.raises(TypeError, match="step must be an integer"):
        DiscretePMF.delta(0.0, step=True)


def test_low_level_analytic_context_rejects_negative_activity_delays() -> None:
    events = (
        mc.Event("root", mc.EventTimestamp(0.0, 10.0, 0.0)),
        mc.Event("target", mc.EventTimestamp(0.0, 10.0, 0.0)),
    )
    activity = AnalyticActivity(0, DiscretePMF(np.array([-1.0, 0.0]), np.array([0.5, 0.5]), step=1))
    context = AnalyticContext(
        events, {(0, 1): (0, activity)}, ((1, ((0, 0),)),), 1, UnderflowRule.TRUNCATE, OverflowRule.TRUNCATE
    )

    with pytest.raises(ValueError, match="PMF support must be non-negative"):
        mc.create_analytic_propagator(context)


def test_deterministic_shared_ancestry_remains_in_exact_domain() -> None:
    events = tuple(mc.Event(f"E{index}", mc.EventTimestamp(0.0, 20.0, 0.0)) for index in range(5))
    deterministic = DiscretePMF.delta(1.0, step=1)
    left = DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5]), step=1)
    right = DiscretePMF(np.array([0.0, 2.0]), np.array([0.25, 0.75]), step=1)
    activities = {
        (0, 1): (0, AnalyticActivity(0, deterministic)),
        (1, 2): (1, AnalyticActivity(1, left)),
        (1, 3): (2, AnalyticActivity(2, right)),
        (2, 4): (3, AnalyticActivity(3, deterministic)),
        (3, 4): (4, AnalyticActivity(4, deterministic)),
    }
    context = AnalyticContext(
        events,
        activities,
        ((1, ((0, 0),)), (2, ((1, 1),)), (3, ((1, 2),)), (4, ((2, 3), (3, 4)))),
        1,
        UnderflowRule.TRUNCATE,
        OverflowRule.TRUNCATE,
    )

    result = mc.create_analytic_propagator(context).run()[4].pmf

    np.testing.assert_array_equal(result.values, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(result.probabilities, [0.125, 0.125, 0.75])
