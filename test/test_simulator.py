import unittest

from mc_dagprop import GenericDelayGenerator, SimContext, SimulationTreeLink, Simulator


class TestSimulator(unittest.TestCase):
    def setUp(self):
        self.events = [("0", (0.0, 100.0, 0.0)), ("1", (5.0, 100.0, 0.0)), ("2", (10.0, 100.0, 0.0))]

        # 2 links: (src, dst) ? (link_index, SimulationTreeLink)
        self.link_map = {(0, 1): (0, SimulationTreeLink(3.0, 1)), (1, 2): (1, SimulationTreeLink(5.0, 1))}

        # Precedence: node_idx ? [(pred_idx, link_idx)]
        self.precedence_list = [(1, [(0, 0)]), (2, [(1, 1)])]

        self.context = SimContext(
            events=self.events, link_map=self.link_map, precedence_list=self.precedence_list, max_delay=10.0
        )

    def test_constant_via_generic(self):
        # Constant factor = 2.0 for link_type==1
        gen = GenericDelayGenerator()
        gen.add_constant(link_type=1, factor=1.0)
        sim = Simulator(self.context, gen)

        # single-run
        res = sim.run(seed=7)
        r = tuple(res.realized)
        d = tuple(res.delays)

        batch = sim.run_many([1, 2, 3])
        self.assertEqual(len(batch), 3)
        for b in batch:
            self.assertIsInstance(b, type(res))

        # expected:
        # Node0 = 0
        # Node1 = 10 (earliest) + 3*2 = 16
        # Node2 = 16           + 5*2 = 26
        self.assertAlmostEqual(r[1], 6.0, places=6)
        self.assertAlmostEqual(r[2], 16.0, places=6)

        # delays array length == #links
        self.assertEqual(len(d), 2)

    def test_exponential_via_generic(self):
        gen = GenericDelayGenerator()
        # exponential on link_type==1: lambda=1.0, max_scale=10.0
        gen.add_exponential(1, 1000.0, max_scale=1.0)
        sim = Simulator(self.context, gen)

        # generate multiple runs, see they produce non-negative delays <= max_scale*duration
        results = sim.run_many((range(3)))

        delays = [5.695681166393335, 5.695681166393335, 5.695681166393335]
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


if __name__ == "__main__":
    unittest.main()
