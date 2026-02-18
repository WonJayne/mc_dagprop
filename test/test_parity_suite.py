from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mc_dagprop import (
    Activity,
    AnalyticContext,
    DagContext,
    DiscretePMF,
    Event,
    EventTimestamp,
    GenericDelayGenerator,
    MonteCarloPropagator,
    create_analytic_propagator,
)
from mc_dagprop.analytic import AnalyticActivity, OverflowRule, UnderflowRule


@dataclass(frozen=True)
class ParityTolerance:
    probability_atol: float
    mean_atol: float


DEFAULT_PARITY_TOLERANCE = ParityTolerance(probability_atol=0.03, mean_atol=0.08)


def _monte_carlo_probability(samples: np.ndarray, values: np.ndarray) -> np.ndarray:
    index_values = np.asarray(samples, dtype=int)
    counts = np.bincount(index_values, minlength=int(values.max()) + 1)
    return counts[values.astype(int)] / index_values.size


def _assert_distribution_parity(
    analytic_pmf: DiscretePMF, mc_samples: np.ndarray, tolerance: ParityTolerance = DEFAULT_PARITY_TOLERANCE
) -> None:
    mc_probabilities = _monte_carlo_probability(mc_samples, analytic_pmf.values)
    np.testing.assert_allclose(mc_probabilities, analytic_pmf.probabilities, atol=tolerance.probability_atol)

    mc_mean = float(np.mean(mc_samples))
    analytic_mean = float(np.dot(analytic_pmf.values, analytic_pmf.probabilities))
    assert abs(mc_mean - analytic_mean) <= tolerance.mean_atol


def test_large_chain_parity() -> None:
    values_a = np.array([0.0, 1.0, 2.0, 3.0])
    probs_a = np.array([0.1, 0.3, 0.4, 0.2])
    values_b = np.array([0.0, 1.0, 2.0, 3.0])
    probs_b = np.array([0.15, 0.35, 0.3, 0.2])

    events = (
        Event("E0", EventTimestamp(0.0, 120.0, 0.0)),
        Event("E1", EventTimestamp(0.0, 120.0, 0.0)),
        Event("E2", EventTimestamp(0.0, 120.0, 0.0)),
    )

    analytic_context = AnalyticContext(
        events=events,
        activities={
            (0, 1): (0, AnalyticActivity(0, DiscretePMF(values_a, probs_a, step=1))),
            (1, 2): (1, AnalyticActivity(1, DiscretePMF(values_b, probs_b, step=1))),
        },
        precedence_list=((1, ((0, 0),)), (2, ((1, 1),))),
        step=1,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.TRUNCATE,
    )
    analytic_output = create_analytic_propagator(analytic_context).run()[2].pmf

    mc_context = DagContext(
        events=list(events),
        activities={
            (0, 1): Activity(idx=0, minimal_duration=0.0, activity_type=1),
            (1, 2): Activity(idx=1, minimal_duration=0.0, activity_type=2),
        },
        precedence_list=[(1, [(0, 0)]), (2, [(1, 1)])],
        max_delay=120.0,
    )
    generator = GenericDelayGenerator()
    generator.add_empirical_absolute(1, values_a.tolist(), probs_a.tolist())
    generator.add_empirical_absolute(2, values_b.tolist(), probs_b.tolist())
    simulator = MonteCarloPropagator(mc_context, generator)

    samples = np.array([simulator.run(seed).realized[2] for seed in range(12_000)])
    _assert_distribution_parity(analytic_output, samples)


def test_large_branching_parity() -> None:
    values_left = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    probs_left = np.array([0.15, 0.2, 0.25, 0.25, 0.15])
    values_right = np.array([0.0, 1.0, 2.0, 3.0])
    probs_right = np.array([0.2, 0.35, 0.3, 0.15])

    events = (
        Event("E0", EventTimestamp(0.0, 200.0, 0.0)),
        Event("E1", EventTimestamp(0.0, 200.0, 0.0)),
        Event("E2", EventTimestamp(0.0, 200.0, 0.0)),
        Event("E3", EventTimestamp(0.0, 200.0, 0.0)),
    )

    analytic_context = AnalyticContext(
        events=events,
        activities={
            (0, 1): (0, AnalyticActivity(0, DiscretePMF(values_left, probs_left, step=1))),
            (0, 2): (1, AnalyticActivity(1, DiscretePMF(values_right, probs_right, step=1))),
            (1, 3): (2, AnalyticActivity(2, DiscretePMF(values_right, probs_right, step=1))),
            (2, 3): (3, AnalyticActivity(3, DiscretePMF(values_left, probs_left, step=1))),
        },
        precedence_list=((1, ((0, 0),)), (2, ((0, 1),)), (3, ((1, 2), (2, 3)))),
        step=1,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.TRUNCATE,
    )
    analytic_output = create_analytic_propagator(analytic_context).run()[3].pmf

    mc_context = DagContext(
        events=list(events),
        activities={
            (0, 1): Activity(idx=0, minimal_duration=0.0, activity_type=1),
            (0, 2): Activity(idx=1, minimal_duration=0.0, activity_type=2),
            (1, 3): Activity(idx=2, minimal_duration=0.0, activity_type=2),
            (2, 3): Activity(idx=3, minimal_duration=0.0, activity_type=1),
        },
        precedence_list=[(1, [(0, 0)]), (2, [(0, 1)]), (3, [(1, 2), (2, 3)])],
        max_delay=200.0,
    )
    generator = GenericDelayGenerator()
    generator.add_empirical_absolute(1, values_left.tolist(), probs_left.tolist())
    generator.add_empirical_absolute(2, values_right.tolist(), probs_right.tolist())
    simulator = MonteCarloPropagator(mc_context, generator)

    samples = np.array([simulator.run(seed).realized[3] for seed in range(14_000)])
    _assert_distribution_parity(analytic_output, samples)


def test_many_links_chain_parity() -> None:
    chain_length = 100
    max_delay = 1_000.0
    delay_values = np.array([0.0, 1.0, 2.0])
    delay_probabilities = np.array([0.2, 0.5, 0.3])

    events = tuple(Event(f"E{i}", EventTimestamp(0.0, max_delay, 0.0)) for i in range(chain_length + 1))
    analytic_activities: dict[tuple[int, int], tuple[int, AnalyticActivity]] = {}
    mc_activities: dict[tuple[int, int], Activity] = {}
    analytic_precedence: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    mc_precedence: list[tuple[int, list[tuple[int, int]]]] = []

    for edge_index in range(chain_length):
        arc = (edge_index, edge_index + 1)
        analytic_activities[arc] = (
            edge_index,
            AnalyticActivity(edge_index, DiscretePMF(delay_values, delay_probabilities, step=1)),
        )
        mc_activities[arc] = Activity(idx=edge_index, minimal_duration=0.0, activity_type=1)
        analytic_precedence.append((edge_index + 1, ((edge_index, edge_index),)))
        mc_precedence.append((edge_index + 1, [(edge_index, edge_index)]))

    analytic_context = AnalyticContext(
        events=events,
        activities=analytic_activities,
        precedence_list=tuple(analytic_precedence),
        step=1,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.TRUNCATE,
        max_delay=max_delay,
    )
    analytic_output = create_analytic_propagator(analytic_context).run()[-1].pmf

    mc_context = DagContext(
        events=list(events), activities=mc_activities, precedence_list=mc_precedence, max_delay=max_delay
    )
    generator = GenericDelayGenerator()
    generator.add_empirical_absolute(1, delay_values.tolist(), delay_probabilities.tolist())
    simulator = MonteCarloPropagator(mc_context, generator)

    samples = np.array([simulator.run(seed).realized[-1] for seed in range(10_000)])
    _assert_distribution_parity(analytic_output, samples, ParityTolerance(probability_atol=0.03, mean_atol=0.2))


def test_max_delay_parity_with_truncation() -> None:
    values = np.array([3.0, 4.0, 5.0, 6.0])
    probabilities = np.array([0.25, 0.25, 0.3, 0.2])
    max_delay = 4.0

    events = (Event("E0", EventTimestamp(10.0, 100.0, 10.0)), Event("E1", EventTimestamp(12.0, 80.0, 12.0)))

    analytic_context = AnalyticContext(
        events=events,
        activities={(0, 1): (0, AnalyticActivity(0, DiscretePMF(values, probabilities, step=1)))},
        precedence_list=((1, ((0, 0),)),),
        step=1,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.TRUNCATE,
        max_delay=max_delay,
    )

    analytic_result = create_analytic_propagator(analytic_context).run()[1]
    assert analytic_result.pmf.values.max() <= events[1].timestamp.earliest + max_delay
    assert float(analytic_result.overflow) == 0.0

    mc_context = DagContext(
        events=list(events),
        activities={(0, 1): Activity(idx=0, minimal_duration=0.0, activity_type=1)},
        precedence_list=[(1, [(0, 0)])],
        max_delay=max_delay,
    )
    generator = GenericDelayGenerator()
    generator.add_empirical_absolute(1, values.tolist(), probabilities.tolist())
    simulator = MonteCarloPropagator(mc_context, generator)

    sample_values = np.array([simulator.run(seed).realized[1] for seed in range(8_000)])
    assert float(sample_values.max()) <= events[1].timestamp.earliest + max_delay
    _assert_distribution_parity(analytic_result.pmf, sample_values, ParityTolerance(0.025, 0.05))
