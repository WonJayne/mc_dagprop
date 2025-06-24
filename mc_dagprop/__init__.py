"""Public interface for the :mod:`mc_dagprop` package."""

from importlib.metadata import version

try:
    from .monte_carlo import Activity, DagContext, Event, EventTimestamp, GenericDelayGenerator, SimResult, Simulator
except ModuleNotFoundError as exc:  # pragma: no cover - compiled module missing
    raise ImportError(
        "mc_dagprop requires the compiled extension 'mc_dagprop.monte_carlo._core'. "
        "Install the package from source to build it."
    ) from exc
from .analytic import (
    AnalyticContext,
    AnalyticPropagator,
    DiscretePMF,
    DiscreteSimulator,
    OverflowRule,
    SimulatedEvent,
    UnderflowRule,
    create_analytic_propagator,
    create_discrete_simulator,
)
from .utils.inspection import plot_activity_delays, retrieve_absolute_and_relative_delays

__version__ = version("mc-dagprop")

__all__ = [
    "GenericDelayGenerator",
    "DagContext",
    "SimResult",
    "Event",
    "Activity",
    "Simulator",
    "EventTimestamp",
    "DiscretePMF",
    "SimulatedEvent",
    "UnderflowRule",
    "OverflowRule",
    "AnalyticContext",
    "AnalyticPropagator",
    "create_analytic_propagator",
    "DiscreteSimulator",
    "create_discrete_simulator",
    "plot_activity_delays",
    "retrieve_absolute_and_relative_delays",
    "__version__",
]
