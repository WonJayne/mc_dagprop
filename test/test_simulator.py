import unittest

from mc_dagprop import EventTimestamp, GenericDelayGenerator, SimActivity, SimContext, SimEvent, Simulator


class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.events = [
            SimEvent("0", EventTimestamp(0.0, 100.0, 0.0)),
            SimEvent("1", EventTimestamp(5.0, 100.0, 0.0)),
            SimEvent("2", EventTimestamp(10.0, 100.0, 0.0)),
        ]

        # 2 links: (src, dst) ? SimActivity)
        self.link_map = {(0, 1): (SimActivity(3.0, 1)), (1, 2): (SimActivity(5.0, 1))}

        # Precedence: node_idx ? [(pred_idx, link_idx)]
        self.precedence_list = [(1, [(0, 0)]), (2, [(1, 1)])]

        self.context = SimContext(
            events=self.events, activities=self.link_map, precedence_list=self.precedence_list, max_delay=10.0
        )

    def test_constant_via_generic(self):
        # Constant factor = 2.0 for link_type==1
        gen = GenericDelayGenerator()
        gen.add_constant(activity_type=1, factor=1.0)
        sim = Simulator(self.context, gen)

        res = sim.run(seed=7)
        r = tuple(res.realized)
        d = tuple(res.delays)

        batch = sim.run_many([1, 2, 3])
        self.assertEqual(len(batch), 3)
        for b in batch:
            self.assertIsInstance(b, type(res))

        self.assertAlmostEqual(r[1], 6.0, places=6)
        self.assertAlmostEqual(r[2], 16.0, places=6)

        # delays array length == #links
        self.assertEqual(len(d), 2)

        self.assertEqual(res.cause_event[0], -1) # None
        self.assertEqual(res.cause_event[1], 0)
        self.assertEqual(res.cause_event[2], 1)

    def test_exponential_via_generic(self):
        gen = GenericDelayGenerator()
        # exponential on link_type==1: lambda=1.0, max_scale=10.0
        gen.add_exponential(1, 1000.0, max_scale=1.0)
        sim = Simulator(self.context, gen)

        # generate multiple runs, see they produce non-negative delays <= max_scale*duration
        results = sim.run_many((range(3)))

        for idx, res in enumerate(results):
            r = list(res.realized)
            deltas = list(res.delays)

            # Node0 always 0
            self.assertAlmostEqual(r[0], 0.0, places=6)

            # Node1 = earliest1 + delay_on_link0
            # delay_on_link0 = 3.0 * some exp_sample, <= 3.0 * 10.0
            self.assertGreaterEqual(r[1], 5.0)

            # Node2 likewise
            self.assertGreaterEqual(r[2], r[1])
            self.assertLessEqual(r[2], r[1] + 5.0 * 10.0 + 1e-6)

            # delays vector has two entries
            self.assertEqual(len(deltas), 2)

    def test_run_many_array(self) -> None:
        # Test the run_many method with an array of seeds
        gen = GenericDelayGenerator()
        gen.add_constant(activity_type=1, factor=1.0)
        sim = Simulator(self.context, gen)

        seeds = tuple(range(100_000))
        results = sim.run_many_arrays(seeds)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].shape[1], len(seeds))
        self.assertEqual(results[1].shape[1], len(seeds))
        self.assertEqual(results[2].shape[1], len(seeds))

        self.assertEqual(results[0].shape[0], 3)
        self.assertEqual(results[1].shape[0], 2)
        self.assertEqual(results[2].shape[0], 3)


if __name__ == "__main__":
    unittest.main()
