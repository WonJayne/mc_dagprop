import unittest
import numpy as np

from mc_dagprop import (
    AnalyticContext,
    AnalyticEvent,
    DiscretePMF,
    DiscreteSimulator,
    EventTimestamp,
    GenericDelayGenerator,
    SimActivity,
    SimContext,
    SimEvent,
    Simulator,
)
from mc_dagprop.discrete.context import AnalyticEdge


class TestDiscreteSimulator(unittest.TestCase):
    def setUp(self) -> None:
        self.events = [
            AnalyticEvent("0", EventTimestamp(0.0, 100.0, 0.0)),
            AnalyticEvent("1", EventTimestamp(0.0, 100.0, 0.0)),
            AnalyticEvent("2", EventTimestamp(0.0, 100.0, 0.0)),
        ]
        self.mc_events = [
            SimEvent("0", EventTimestamp(0.0, 100.0, 0.0)),
            SimEvent("1", EventTimestamp(0.0, 100.0, 0.0)),
            SimEvent("2", EventTimestamp(0.0, 100.0, 0.0)),
        ]
        self.precedence = [(1, [(0, 0)]), (2, [(1, 1)])]

        act0 = AnalyticEdge(DiscretePMF(np.array([1.0, 2.0]), np.array([0.5, 0.5])))
        act1 = AnalyticEdge(DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5])))
        self.a_context = AnalyticContext(
            events=self.events,
            activities={(0, 1): (0, act0), (1, 2): (1, act1)},
            precedence_list=self.precedence,
            max_delay=5.0,
            step_size=1.0,
        )

        self.mc_context = SimContext(
            events=self.mc_events,
            activities={(0, 1): (0, SimActivity(0.0, 1)), (1, 2): (1, SimActivity(0.0, 2))},
            precedence_list=self.precedence,
            max_delay=5.0,
        )
        gen = GenericDelayGenerator()
        gen.add_empirical_absolute(1, [1.0, 2.0], [0.5, 0.5])
        gen.add_empirical_absolute(2, [0.0, 1.0], [0.5, 0.5])
        self.mc_sim = Simulator(self.mc_context, gen)

    def test_compare_to_monte_carlo(self) -> None:
        ds = DiscreteSimulator(self.a_context)
        pmfs = ds.run()
        final = pmfs[2]
        samples = [self.mc_sim.run(seed=i).realized[2] for i in range(2000)]
        counts = np.bincount(np.array(samples, dtype=int))[1:4]
        mc_probs = counts / counts.sum()
        self.assertTrue(np.allclose(final.values, [1.0, 2.0, 3.0]))
        self.assertTrue(np.allclose(final.probs, [0.25, 0.5, 0.25], atol=0.05))
        self.assertTrue(np.allclose(mc_probs, final.probs, atol=0.05))

    def test_mismatched_step_size(self) -> None:
        act0 = AnalyticEdge(DiscretePMF(np.array([1.0, 2.0]), np.array([0.5, 0.5])))
        act1 = AnalyticEdge(DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5])))
        ctx = AnalyticContext(
            events=self.events,
            activities={(0, 1): (0, act0), (1, 2): (1, act1)},
            precedence_list=self.precedence,
            max_delay=5.0,
            step_size=2.0,
        )
        with self.assertRaises(ValueError):
            DiscreteSimulator(ctx)

    def test_bounds_and_overflow(self) -> None:
        events = [
            AnalyticEvent("0", EventTimestamp(0.0, 100.0, 0.0), bounds=(0.0, 0.0)),
            AnalyticEvent("1", EventTimestamp(0.0, 100.0, 0.0), bounds=(0.0, 1.5)),
            AnalyticEvent("2", EventTimestamp(0.0, 100.0, 0.0), bounds=(0.0, 1.8)),
        ]
        ctx = AnalyticContext(
            events=events,
            activities={(0, 1): (0, AnalyticEdge(DiscretePMF(np.array([1.0, 2.0]), np.array([0.5, 0.5])))),
                        (1, 2): (1, AnalyticEdge(DiscretePMF(np.array([0.0, 1.0]), np.array([0.5, 0.5]))))},
            precedence_list=self.precedence,
            max_delay=5.0,
            step_size=1.0,
        )
        ds = DiscreteSimulator(ctx)
        pmfs = ds.run()
        self.assertAlmostEqual(ds.overflow[1], 0.5, places=6)
        self.assertAlmostEqual(ds.overflow[2], 0.5, places=6)
        self.assertAlmostEqual(sum(ds.overflow), 1.0, places=6)
        self.assertTrue(np.all(pmfs[1].values <= 1.5))
        self.assertTrue(np.all(pmfs[2].values <= 1.8))
        self.assertAlmostEqual(pmfs[2].probs.sum(), 1.0, places=6)

    def test_large_uniform_network(self) -> None:
        values = np.arange(-180.0, 1800.1, 1.0)
        probs = np.ones_like(values, dtype=float) / len(values)
        events = [
            AnalyticEvent(str(i), EventTimestamp(0.0, 2000.0, 0.0))
            for i in range(5)
        ]
        precedence = [
            (1, [(0, 0)]),
            (2, [(0, 1)]),
            (3, [(1, 2), (2, 3)]),
            (4, [(2, 4), (3, 5)]),
        ]
        activities = {
            (0, 1): (0, AnalyticEdge(DiscretePMF(values, probs))),
            (0, 2): (1, AnalyticEdge(DiscretePMF(values, probs))),
            (1, 3): (2, AnalyticEdge(DiscretePMF(values, probs))),
            (2, 3): (3, AnalyticEdge(DiscretePMF(values, probs))),
            (2, 4): (4, AnalyticEdge(DiscretePMF(values, probs))),
            (3, 4): (5, AnalyticEdge(DiscretePMF(values, probs))),
        }
        ctx = AnalyticContext(
            events=events,
            activities=activities,
            precedence_list=precedence,
            max_delay=1800.0,
            step_size=1.0,
        )
        ds = DiscreteSimulator(ctx)
        pmfs = ds.run()
        self.assertEqual(len(pmfs), 5)
        for pmf in pmfs:
            self.assertAlmostEqual(pmf.step, 1.0, places=6)
        self.assertTrue(all(u == 0.0 for u in ds.underflow))
        self.assertTrue(all(o == 0.0 for o in ds.overflow))


if __name__ == "__main__":
    unittest.main()
