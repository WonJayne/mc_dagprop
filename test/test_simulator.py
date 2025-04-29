import unittest

import numpy as np
from mc_dagprop import EventTimestamp, GenericDelayGenerator, SimActivity, SimContext, SimEvent, Simulator
from plotly import graph_objects


class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.events = [
            SimEvent("0", EventTimestamp(0.0, 100.0, 0.0)),
            SimEvent("1", EventTimestamp(5.0, 100.0, 0.0)),
            SimEvent("2", EventTimestamp(10.0, 100.0, 0.0)),
            SimEvent("3", EventTimestamp(22.0, 100.0, 0.0)),
            SimEvent("4", EventTimestamp(20.0, 100.0, 0.0)),
            SimEvent("5", EventTimestamp(100.0, 100.0, 0.0)),
        ]

        # 2 links: (src, dst) ? SimActivity)
        self.link_map = {
            (0, 1): (0, SimActivity(3.0, 1)),
            (1, 2): (1, SimActivity(5.0, 1)),
            (1, 3): (2, SimActivity(5.0, 1)),
            (2, 4): (3, SimActivity(15.0, 2)),
            (3, 4): (4, SimActivity(10.0, 3)),
        }

        # Precedence: node_idx ? [(pred_idx, link_idx)]
        self.precedence_list = [(1, [(0, 0)]), (2, [(1, 1)]), (3, [(1, 2)]), (4, [(2, 3), (3, 4)])]

        self.context = SimContext(
            events=self.events, activities=self.link_map, precedence_list=self.precedence_list, max_delay=10.0
        )

    def test_constant_via_generic(self):
        gen = GenericDelayGenerator()
        gen.add_constant(activity_type=1, factor=1.0)
        sim = Simulator(self.context, gen)

        res = sim.run(seed=7)
        r = res.realized
        d = res.durations

        batch = sim.run_many([1, 2, 3])
        self.assertEqual(len(batch), 3)
        for b in batch:
            self.assertIsInstance(b, type(res))

        self.assertAlmostEqual(r[1], 6.0, places=6)
        self.assertAlmostEqual(r[2], 16.0, places=6)

        # durations array length == #links
        self.assertEqual(len(d), 5)
        self.assertEqual(len(res.durations), 5)
        self.assertEqual(len(res.realized), 6)
        self.assertEqual(len(res.cause_event), 6)

        self.assertEqual(res.cause_event[0], -1)  # None
        self.assertEqual(res.cause_event[1], 0)
        self.assertEqual(res.cause_event[2], 1)

    def test_exponential_via_generic(self):
        gen = GenericDelayGenerator()
        gen.add_exponential(1, 1000.0, max_scale=1.0)
        sim = Simulator(self.context, gen)
        for idx, res in enumerate(sim.run_many(tuple(range(3)))):
            r = list(res.realized)
            deltas = list(res.durations)

            # Node0 always 0
            self.assertAlmostEqual(r[0], 0.0, places=6)

            # Node1 = earliest1 + delay_on_link0
            # delay_on_link0 = 3.0 * some exp_sample, <= 3.0 * 10.0
            self.assertGreaterEqual(r[1], 5.0)

            # Node2 likewise
            self.assertGreaterEqual(r[2], r[1])
            self.assertLessEqual(r[2], r[1] + 5.0 * 10.0 + 1e-6)

            # durations vector has two entries
            self.assertEqual(len(deltas), 5)

    def test_propagation(self):
        gen = GenericDelayGenerator()
        gen.add_constant(activity_type=1, factor=1.0)
        gen.add_constant(activity_type=3, factor=3.0)
        sim = Simulator(self.context, gen)

        # run with a single event
        res = sim.run_many(tuple(range(5)))[3]

        # check that the durations are all 0
        self.assertEqual(res.durations[0], 6.0)
        self.assertEqual(res.durations[1], 10.0)
        self.assertEqual(res.durations[2], 10.0)
        self.assertEqual(res.durations[3], 15.0)
        self.assertEqual(res.durations[4], 40.0)

        # check that the realized times are correct
        self.assertEqual(res.realized[0], 0.0)
        self.assertEqual(res.realized[1], 6.0)
        self.assertEqual(res.realized[2], 16.0)
        self.assertEqual(res.realized[3], 22.0)
        self.assertEqual(res.realized[4], 62.0)
        self.assertEqual(res.realized[5], 100.0)

        # check that the cause events are correct
        self.assertEqual(res.cause_event[0], -1)
        self.assertEqual(res.cause_event[1], 0)
        self.assertEqual(res.cause_event[2], 1)
        self.assertEqual(res.cause_event[3], -1)
        self.assertEqual(res.cause_event[4], 3)
        self.assertEqual(res.cause_event[5], -1)

    def test_empirical_absolute(self) -> None:
        gen = GenericDelayGenerator()
        gen.add_empirical_absolute(activity_type=1, values=[10, 20, 40, 50], weights=[0.1, 0.2, 0.3, 0.4])
        sim = Simulator(self.context, gen)
        res = sim.run(seed=7)

        self.assertEqual(res.realized[3], 68.0)
        self.assertEqual(res.realized[5], 100.0)

    def test_empirical_relative(self) -> None:
        gen = GenericDelayGenerator()
        gen.add_empirical_relative(activity_type=1, factors=[1.2, 1.3, 1.35, 4.5], weights=[0.1, 0.2, 0.3, 0.4])
        sim = Simulator(self.context, gen)
        res = sim.run(seed=7)

        self.assertAlmostEqual(res.realized[3], 34.10, places=4)
        self.assertEqual(res.realized[5], 100.0)

    def test_empirical_relative_with_exponential(self) -> None:
        values = []
        np.random.seed(7)
        while len(values) < 1000000:
            value = 10
            while value > 5.0:
                value = np.random.exponential(3.0)
            values.append(value)
        generator = GenericDelayGenerator()
        hist, bin_edges = np.histogram(values, bins=1000, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        generator.add_empirical_relative(activity_type=1, factors=bin_centers, weights=hist)
        simulator = Simulator(self.context, generator)
        result = simulator.run(seed=7)
        self.assertAlmostEqual(result.realized[3], 23.062461393412335, places=3)
        self.assertEqual(result.realized[5], 100.0)


if __name__ == "__main__":
    unittest.main()
