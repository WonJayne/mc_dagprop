import unittest
from dataclasses import replace

import numpy as np

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
            max_delay=5.0,
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


if __name__ == "__main__":
    unittest.main()
