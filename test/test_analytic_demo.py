import unittest

from demo import analytic


class TestAnalyticDemo(unittest.TestCase):
    def test_demo_runs(self) -> None:
        analytic.main()


if __name__ == "__main__":
    unittest.main()
