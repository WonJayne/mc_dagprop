import unittest
from dataclasses import replace

import numpy as np
import pytest

from mc_dagprop import (
    Activity,
    AnalyticContext,
    DagContext,
    DiscretePMF,
    Event,
    EventTimestamp,
    GenericDelayGenerator,
    Simulator,
    create_analytic_propagator,
)
from mc_dagprop.analytic import OverflowRule, UnderflowRule
from mc_dagprop.analytic._context import AnalyticActivity, SimulatedEvent


TEST_ACTIVITY_VALUES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
TEST_ACTIVITY_PROBABILITIES = np.array([0.10, 0.15, 0.20, 0.25, 0.10, 0.20])
EXPECTED_INSIDE_VALUES = np.array([0.0, 1.0, 2.0])


def _run_rule_case(underflow_rule: UnderflowRule, overflow_rule: OverflowRule) -> SimulatedEvent:
    events = (
        Event("source", EventTimestamp(0.0, 0.0, 0.0)),
        Event("bounded_target", EventTimestamp(0.0, 2.0, 0.0)),
    )
    analytic_activity = AnalyticActivity(
        0,
        DiscretePMF(TEST_ACTIVITY_VALUES, TEST_ACTIVITY_PROBABILITIES, step=1),
    )
    context = AnalyticContext(
        events=events,
        activities={(0, 1): (0, analytic_activity)},
        precedence_list=((1, ((0, 0),)),),
        step=1,
        underflow_rule=underflow_rule,
        overflow_rule=overflow_rule,
    )
    return create_analytic_propagator(context).run()[1]


class TestDiscreteSimulator(unittest.TestCase):
    def setUp(self) -> None:
        self.events = (
            Event("0", EventTimestamp(0.0, 100.0, 0.0)),
            Event("1", EventTimestamp(0.0, 100.0, 0.0)),
            Event("2", EventTimestamp(0.0, 100.0, 0.0)),
        )
        self.mc_events = (
            Event("0", EventTimestamp(0.0, 100.0, 0.0)),
            Event("1", EventTimestamp(0.0, 100.0, 0.0)),
            Event("2", EventTimestamp(0.0, 100.0, 0.0)),
        )
        self.precedence = ((1, ((0, 0),)), (2, ((1, 1),)))

        act0 = AnalyticActivity(0, DiscretePMF(np.array([1.0, 2.0]), np.array([0.5, 0.5]), step=1))
        act1 = AnalyticActivity(1, DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5]), step=1))
        self.a_context = AnalyticContext(
            events=self.events,
            activities={(0, 1): (0, act0), (1, 2): (1, act1)},
            precedence_list=self.precedence,
            step=1,
            underflow_rule=UnderflowRule.TRUNCATE,
            overflow_rule=OverflowRule.TRUNCATE,
        )

        self.mc_context = DagContext(
            events=self.mc_events,
            activities={
                (0, 1): Activity(idx=0, minimal_duration=0.0, activity_type=1),
                (1, 2): Activity(idx=1, minimal_duration=0.0, activity_type=2),
            },
            precedence_list=self.precedence,
            max_delay=1e6,
        )
        gen = GenericDelayGenerator()
        gen.add_empirical_absolute(1, [1.0, 2.0], [0.5, 0.5])
        gen.add_empirical_absolute(2, [0.0, 1.0], [0.5, 0.5])
        self.mc_sim = Simulator(self.mc_context, gen)

    def test_compare_to_monte_carlo(self) -> None:
        ds = create_analytic_propagator(self.a_context)
        events = ds.run()
        self.assertTrue(all(isinstance(ev, SimulatedEvent) for ev in events))
        final = events[2].pmf
        samples = [self.mc_sim.run(seed=i).realized[2] for i in range(2000)]
        counts = np.bincount(np.array(samples, dtype=int))[1:4]
        mc_probs = counts / counts.sum()
        self.assertTrue(np.allclose(final.values, [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(final.probabilities, [0.25, 0.5, 0.25], atol=0.05))
        self.assertTrue(np.allclose(mc_probs, final.probabilities, atol=0.05))

    def test_event_without_predecessor(self) -> None:
        ds = create_analytic_propagator(self.a_context)
        events = ds.run()
        first = events[0].pmf
        earliest = self.events[0].timestamp.earliest
        self.assertTrue(np.allclose(first.values, [earliest]))
        self.assertTrue(np.allclose(first.probabilities, [1.0]))
        mc_res = self.mc_sim.run(seed=0)
        self.assertEqual(mc_res.cause_event[0], -1)

    def test_mismatched_step_size(self) -> None:
        act0 = AnalyticActivity(0, DiscretePMF(np.array([1.0, 2.0]), np.array([0.5, 0.5]), step=1))
        act1 = AnalyticActivity(1, DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5]), step=1))
        ctx = AnalyticContext(
            events=self.events,
            activities={(0, 1): (0, act0), (1, 2): (1, act1)},
            precedence_list=self.precedence,
            step=2,
            underflow_rule=UnderflowRule.TRUNCATE,
            overflow_rule=OverflowRule.TRUNCATE,
        )
        with self.assertRaises(ValueError):
            create_analytic_propagator(ctx)

    def test_non_positive_step_size(self) -> None:
        ctx = AnalyticContext(
            events=self.events,
            activities={},
            precedence_list=(),
            step=0,
            underflow_rule=UnderflowRule.TRUNCATE,
            overflow_rule=OverflowRule.TRUNCATE,
        )
        with self.assertRaises(ValueError):
            create_analytic_propagator(ctx)

    def test_negative_max_delay_rejected(self) -> None:
        ctx = AnalyticContext(
            events=self.events,
            activities={},
            precedence_list=(),
            step=1,
            underflow_rule=UnderflowRule.TRUNCATE,
            overflow_rule=OverflowRule.TRUNCATE,
            max_delay=-1,
        )
        with self.assertRaises(ValueError):
            create_analytic_propagator(ctx)

    def test_skip_validation(self) -> None:
        act0 = AnalyticActivity(0, DiscretePMF(np.array([1.0, 2.0]), np.array([0.5, 0.5]), step=1))
        ctx = AnalyticContext(
            events=self.events,
            activities={(0, 1): (0, act0)},
            precedence_list=((1, ((0, 0),)),),
            step=2,
            underflow_rule=UnderflowRule.TRUNCATE,
            overflow_rule=OverflowRule.TRUNCATE,
        )

        # Should not raise when validation is disabled
        sim = create_analytic_propagator(ctx, validate=False)
        result = sim.run()
        self.assertEqual(len(result), 3)

    def test_misaligned_values(self) -> None:
        act0 = AnalyticActivity(0, DiscretePMF(np.array([1.0, 2.5]), np.array([0.5, 0.5]), step=1))
        ctx = AnalyticContext(
            events=self.events,
            activities={(0, 1): (0, act0)},
            precedence_list=((1, ((0, 0),)),),
            step=1,
            underflow_rule=UnderflowRule.TRUNCATE,
            overflow_rule=OverflowRule.TRUNCATE,
        )
        with self.assertRaises(ValueError):
            create_analytic_propagator(ctx)

    def test_bounds_and_overflow(self) -> None:
        events = (
            Event("0", EventTimestamp(0.0, 0.0, 0.0)),
            Event("1", EventTimestamp(0.0, 2.0, 0.0)),
            Event("2", EventTimestamp(0.0, 1.0, 0.0)),
        )
        ctx = AnalyticContext(
            events=events,
            activities={
                (0, 1): (0, AnalyticActivity(0, DiscretePMF(np.array([1.0, 2.0]), np.array([0.5, 0.5]), step=1))),
                (1, 2): (1, AnalyticActivity(1, DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5]), step=1))),
            },
            precedence_list=self.precedence,
            step=1,
            underflow_rule=UnderflowRule.TRUNCATE,
            overflow_rule=OverflowRule.TRUNCATE,
        )
        ds = create_analytic_propagator(ctx)
        events_res = ds.run()
        self.assertTrue(all(isinstance(ev, SimulatedEvent) for ev in events_res))
        self.assertAlmostEqual(events_res[1].overflow, 0.0, places=6)
        self.assertAlmostEqual(events_res[2].overflow, 0.0, places=6)
        self.assertTrue(np.allclose(events_res[1].pmf.values, [1.0, 2.0]))
        self.assertTrue(np.all(events_res[1].pmf.values <= 2))
        self.assertTrue(np.all(events_res[2].pmf.values <= 1))
        self.assertAlmostEqual(events_res[2].pmf.probabilities.sum(), 1.0, places=6)

    def test_rule_combinations(self) -> None:
        events = (Event("0", EventTimestamp(0.0, 10.0, 0.0)), Event("1", EventTimestamp(0.0, 1.0, 0.0)))
        edge = AnalyticActivity(
            0, DiscretePMF(np.array([-1.0, 0.0, 1.0, 2.0]), np.array([0.5, 0.0, 0.0, 0.5]), step=1)
        )
        ctx = AnalyticContext(
            events=events,
            activities={(0, 1): (0, edge)},
            precedence_list=((1, ((0, 0),)),),
            step=1,
            underflow_rule=UnderflowRule.TRUNCATE,
            overflow_rule=OverflowRule.TRUNCATE,
        )

        ds_default = create_analytic_propagator(ctx)
        res_default = ds_default.run()[1]
        self.assertAlmostEqual(res_default.underflow, 0.0, places=6)
        self.assertAlmostEqual(res_default.overflow, 0.0, places=6)
        self.assertTrue(np.allclose(res_default.pmf.values, [0.0, 1.0]))

        ctx_remove_both = AnalyticContext(
            events=events,
            activities={(0, 1): (0, edge)},
            precedence_list=((1, ((0, 0),)),),
            step=1,
            underflow_rule=UnderflowRule.REMOVE,
            overflow_rule=OverflowRule.REMOVE,
        )
        create_analytic_propagator(ctx_remove_both).run()

        ctx_remove_under = replace(ctx, underflow_rule=UnderflowRule.REMOVE)
        ds_mixed1 = create_analytic_propagator(ctx_remove_under)
        res_mixed1 = ds_mixed1.run()[1]
        self.assertAlmostEqual(res_mixed1.underflow, 0.5, places=6)
        self.assertAlmostEqual(res_mixed1.overflow, 0.0, places=6)
        self.assertTrue(np.allclose(res_mixed1.pmf.values, [0.0, 1.0]))

        ctx_remove_over = replace(ctx, overflow_rule=OverflowRule.REMOVE)
        ds_mixed2 = create_analytic_propagator(ctx_remove_over)
        res_mixed2 = ds_mixed2.run()[1]
        self.assertAlmostEqual(res_mixed2.underflow, 0.0, places=6)
        self.assertAlmostEqual(res_mixed2.overflow, 0.5, places=6)
        self.assertTrue(np.allclose(res_mixed2.pmf.values, [0.0, 1.0]))

    def test_large_uniform_network(self) -> None:
        values = np.arange(-180.0, 1800.1, 1.0)
        probs = np.ones_like(values, dtype=float) / len(values)
        events = tuple(Event(str(i), EventTimestamp(0.0, 2000.0, 0.0)) for i in range(5))
        precedence = ((1, ((0, 0),)), (2, ((0, 1),)), (3, ((1, 2), (2, 3))), (4, ((2, 4), (3, 5))))
        activities = {
            (0, 1): (0, AnalyticActivity(0, DiscretePMF(values, probs, step=1))),
            (0, 2): (1, AnalyticActivity(1, DiscretePMF(values, probs, step=1))),
            (1, 3): (2, AnalyticActivity(2, DiscretePMF(values, probs, step=1))),
            (2, 3): (3, AnalyticActivity(3, DiscretePMF(values, probs, step=1))),
            (2, 4): (4, AnalyticActivity(4, DiscretePMF(values, probs, step=1))),
            (3, 4): (5, AnalyticActivity(5, DiscretePMF(values, probs, step=1))),
        }
        ctx = AnalyticContext(
            events=events,
            activities=activities,
            precedence_list=precedence,
            step=1,
            underflow_rule=UnderflowRule.TRUNCATE,
            overflow_rule=OverflowRule.TRUNCATE,
        )
        ds = create_analytic_propagator(ctx)
        events_res = ds.run()
        self.assertEqual(len(events_res), 5)
        for e in events_res[1:]:
            self.assertAlmostEqual(e.pmf.step, 1.0, places=6)
        self.assertTrue(all(e.underflow >= 0.0 for e in events_res))
        self.assertTrue(all(e.overflow >= 0.0 for e in events_res))

    def test_invalid_event_bounds(self) -> None:
        events = (Event("0", EventTimestamp(5.0, 4.0, 0.0)),)
        ctx = AnalyticContext(
            events=events,
            activities={},
            precedence_list=(),
            step=1,
            underflow_rule=UnderflowRule.TRUNCATE,
            overflow_rule=OverflowRule.TRUNCATE,
        )
        with self.assertRaises(ValueError):
            create_analytic_propagator(ctx)

    def test_cycle_detection(self) -> None:
        events = (Event("0", EventTimestamp(0.0, 10.0, 0.0)), Event("1", EventTimestamp(0.0, 10.0, 0.0)))
        edge = AnalyticActivity(0, DiscretePMF(np.array([1.0]), np.array([1.0]), step=1))
        ctx = AnalyticContext(
            events=events,
            activities={(0, 1): (0, edge), (1, 0): (1, edge)},
            precedence_list=((1, ((0, 0),)), (0, ((1, 1),))),
            step=1,
            underflow_rule=UnderflowRule.TRUNCATE,
            overflow_rule=OverflowRule.TRUNCATE,
        )
        with self.assertRaises(ValueError):
            create_analytic_propagator(ctx)


def test_run_returns_simulated_event_objects() -> None:
    events = (Event("0", EventTimestamp(0.0, 10.0, 0.0)), Event("1", EventTimestamp(0.0, 10.0, 0.0)))
    edge = AnalyticActivity(
        0, DiscretePMF(np.array([-1.0, 0.0, 1.0, 2.0]), np.array([0.25, 0.25, 0.25, 0.25]), step=1)
    )
    ctx = AnalyticContext(
        events=events,
        activities={(0, 1): (0, edge)},
        precedence_list=((1, ((0, 0),)),),
        step=1,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.TRUNCATE,
    )

    sim = create_analytic_propagator(ctx)
    result = sim.run()
    assert all(isinstance(ev, SimulatedEvent) for ev in result)


def test_clipping_tolerates_rounding_errors() -> None:
    vals = np.array([-1.0, 0.0, 1.0])
    probs = np.array([0.25, 0.25, 0.5 + 1e-12])
    pmf = DiscretePMF(vals, probs, step=1)
    ctx = AnalyticContext(
        events=(Event("e0", EventTimestamp(0.0, 1.0, 0.0)),),
        activities={},
        precedence_list=(),
        step=1,
        underflow_rule=UnderflowRule.TRUNCATE,
        overflow_rule=OverflowRule.TRUNCATE,
    )
    sim = create_analytic_propagator(ctx, validate=False)
    res = sim._convert_to_simulated_event(pmf, 0.0, 1.0)
    total = res.pmf.probabilities.sum() + float(res.underflow) + float(res.overflow)
    assert np.isclose(total, 1.0)



@pytest.mark.parametrize(
    ("underflow_rule", "overflow_rule", "expected_probabilities", "expected_underflow", "expected_overflow"),
    [
        pytest.param(
            UnderflowRule.TRUNCATE,
            OverflowRule.TRUNCATE,
            np.array([0.45, 0.25, 0.30]),
            0.0,
            0.0,
            id="truncate-under_truncate-over",
        ),
        pytest.param(
            UnderflowRule.TRUNCATE,
            OverflowRule.REMOVE,
            np.array([0.45, 0.25, 0.10]),
            0.0,
            0.20,
            id="truncate-under_remove-over",
        ),
        pytest.param(
            UnderflowRule.TRUNCATE,
            OverflowRule.REDISTRIBUTE,
            np.array([0.5625, 0.3125, 0.1250]),
            0.0,
            0.0,
            id="truncate-under_redistribute-over",
        ),
        pytest.param(
            UnderflowRule.REMOVE,
            OverflowRule.TRUNCATE,
            np.array([0.20, 0.25, 0.30]),
            0.25,
            0.0,
            id="remove-under_truncate-over",
        ),
        pytest.param(
            UnderflowRule.REMOVE,
            OverflowRule.REMOVE,
            np.array([0.20, 0.25, 0.10]),
            0.25,
            0.20,
            id="remove-under_remove-over",
        ),
        pytest.param(
            UnderflowRule.REMOVE,
            OverflowRule.REDISTRIBUTE,
            np.array([0.2727272727272727, 0.3409090909090909, 0.13636363636363635]),
            0.25,
            0.0,
            id="remove-under_redistribute-over",
        ),
        pytest.param(
            UnderflowRule.REDISTRIBUTE,
            OverflowRule.TRUNCATE,
            np.array([0.26666666666666666, 0.3333333333333333, 0.4000000000000001]),
            0.0,
            0.0,
            id="redistribute-under_truncate-over",
        ),
        pytest.param(
            UnderflowRule.REDISTRIBUTE,
            OverflowRule.REMOVE,
            np.array([0.2909090909090909, 0.36363636363636365, 0.14545454545454545]),
            0.0,
            0.20,
            id="redistribute-under_remove-over",
        ),
        pytest.param(
            UnderflowRule.REDISTRIBUTE,
            OverflowRule.REDISTRIBUTE,
            np.array([0.36363636363636365, 0.4545454545454546, 0.18181818181818182]),
            0.0,
            0.0,
            id="redistribute-under_redistribute-over",
        ),
    ],
)
def test_all_underflow_and_overflow_rule_combinations(
    underflow_rule: UnderflowRule,
    overflow_rule: OverflowRule,
    expected_probabilities: np.ndarray,
    expected_underflow: float,
    expected_overflow: float,
) -> None:
    result = _run_rule_case(underflow_rule, overflow_rule)

    np.testing.assert_allclose(result.pmf.values, EXPECTED_INSIDE_VALUES)
    np.testing.assert_allclose(result.pmf.probabilities, expected_probabilities, rtol=0.0, atol=1e-12)
    assert float(result.underflow) == pytest.approx(expected_underflow)
    assert float(result.overflow) == pytest.approx(expected_overflow)


@pytest.mark.parametrize(
    ("underflow_rule", "overflow_rule"),
    [(under_rule, over_rule) for under_rule in UnderflowRule for over_rule in OverflowRule],
)
def test_rule_combinations_preserve_total_mass_and_non_negative_probabilities(
    underflow_rule: UnderflowRule,
    overflow_rule: OverflowRule,
) -> None:
    result = _run_rule_case(underflow_rule, overflow_rule)

    assert np.all(result.pmf.probabilities >= 0.0)
    assert np.isclose(
        float(result.pmf.total_mass + result.underflow + result.overflow),
        1.0,
        rtol=1e-12,
        atol=1e-15,
    )


if __name__ == "__main__":
    unittest.main()
