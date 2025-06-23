import unittest
import numpy as np

from mc_dagprop import (
    AnalyticContext,
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
            events=self.events,
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


if __name__ == "__main__":
    unittest.main()
