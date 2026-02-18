from __future__ import annotations

from mc_dagprop import MonteCarloPropagator, Simulator


def test_monte_carlo_class_name_is_primary() -> None:
    assert MonteCarloPropagator.__name__ == "MonteCarloPropagator"
    assert Simulator is MonteCarloPropagator
