import unittest
import numpy as np

from mc_dagprop import Activity, DagContext, Event, EventTimestamp, GenericDelayGenerator, Simulator


class BaseContextMixin:
    def create_context(self) -> DagContext:
        events = [
            Event("0", EventTimestamp(0.0, 100.0, 0.0)),
            Event("1", EventTimestamp(0.0, 100.0, 0.0)),
            Event("2", EventTimestamp(0.0, 100.0, 0.0)),
        ]
        activities = {
            (0, 1): Activity(idx=0, minimal_duration=1.0, activity_type=1),
            (1, 2): Activity(idx=1, minimal_duration=2.0, activity_type=1),
        }
        precedence = [(1, [(0, 0)]), (2, [(1, 1)])]
        return DagContext(events=events, activities=activities, precedence_list=precedence, max_delay=5.0)


class TestDelayDistributions(BaseContextMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.context = self.create_context()

    def test_constant_distribution(self) -> None:
        gen = GenericDelayGenerator()
        gen.add_constant(activity_type=1, factor=1.0)
        sim = Simulator(self.context, gen)

        result = sim.run(seed=42)
        np.testing.assert_allclose(result.durations, [2.0, 4.0])
        np.testing.assert_allclose(result.realized, [0.0, 2.0, 6.0])

    def test_exponential_distribution(self) -> None:
        gen = GenericDelayGenerator()
        gen.add_exponential(activity_type=1, lambda_=2.0, max_scale=0.5)
        sim = Simulator(self.context, gen)
        res = sim.run(seed=0)
        self.assertGreaterEqual(res.durations[0], 1.0)
        self.assertLessEqual(res.durations[0], 1.5)
        self.assertGreaterEqual(res.durations[1], 2.0)
        self.assertLessEqual(res.durations[1], 3.0)
        self.assertGreaterEqual(res.realized[2], res.realized[1])

    def test_gamma_distribution(self) -> None:
        gen = GenericDelayGenerator()
        gen.add_gamma(activity_type=1, shape=2.0, scale=1.0, max_scale=0.5)
        sim = Simulator(self.context, gen)
        res = sim.run(seed=1)
        self.assertGreaterEqual(res.durations[0], 1.0)
        self.assertLessEqual(res.durations[0], 1.5)
        self.assertGreaterEqual(res.durations[1], 2.0)
        self.assertLessEqual(res.durations[1], 3.0)
        self.assertGreaterEqual(res.realized[2], res.realized[1])


class TestErrorConditions(BaseContextMixin, unittest.TestCase):
    def test_cycle_detection(self) -> None:
        events = [
            Event("0", EventTimestamp(0.0, 100.0, 0.0)),
            Event("1", EventTimestamp(0.0, 100.0, 0.0)),
        ]
        activities = {
            (0, 1): Activity(idx=0, minimal_duration=1.0, activity_type=1),
            (1, 0): Activity(idx=1, minimal_duration=1.0, activity_type=1),
        }
        precedence = [(1, [(0, 0)]), (0, [(1, 1)])]
        context = DagContext(
            events=events, activities=activities, precedence_list=precedence, max_delay=5.0
        )
        gen = GenericDelayGenerator()
        gen.add_constant(1, 0.0)
        with self.assertRaises(RuntimeError):
            Simulator(context, gen)

    def test_reserved_activity_type(self) -> None:
        context = self.create_context()
        gen = GenericDelayGenerator()
        gen.add_constant(-1, 0.0)
        with self.assertRaises(RuntimeError):
            Simulator(context, gen)


class TestRunMany(BaseContextMixin, unittest.TestCase):
    def setUp(self) -> None:
        self.context = self.create_context()
        gen = GenericDelayGenerator()
        gen.add_exponential(activity_type=1, lambda_=1.0, max_scale=0.5)
        self.sim = Simulator(self.context, gen)

    def test_run_many_matches_individual_runs(self) -> None:
        seeds = list(range(5))
        batch = self.sim.run_many(seeds)
        solo = [self.sim.run(seed) for seed in seeds]
        self.assertEqual(len(batch), len(solo))
        for b, s in zip(batch, solo):
            np.testing.assert_allclose(b.realized, s.realized)
            np.testing.assert_allclose(b.durations, s.durations)
            np.testing.assert_array_equal(b.cause_event, s.cause_event)


if __name__ == "__main__":
    unittest.main()
