"""Public interface for the :mod:`mc_dagprop` package."""

from importlib.metadata import version

try:
    from ._core import EventTimestamp, GenericDelayGenerator, SimActivity, SimContext, SimEvent, SimResult, Simulator
except ModuleNotFoundError as exc:  # pragma: no cover - compiled module missing
    raise ImportError(
        "mc_dagprop requires the compiled extension 'mc_dagprop._core'. " "Install the package from source to build it."
    ) from exc
from .discrete import (
    AnalyticContext,
    ScheduledEvent,
    SimulatedEvent,
    DiscretePMF,
    DiscreteSimulator,
    create_discrete_simulator,
)
from . import UnderflowRule, OverflowRule
from .utils.inspection import plot_activity_delays, retrieve_absolute_and_relative_delays

__version__ = version("mc-dagprop")

__all__ = [
    "GenericDelayGenerator",
    "SimContext",
    "SimResult",
    "SimEvent",
    "SimActivity",
    "Simulator",
    "EventTimestamp",
    "DiscretePMF",
    "ScheduledEvent",
    "SimulatedEvent",
    "UnderflowRule",
    "OverflowRule",
    "AnalyticContext",
    "DiscreteSimulator",
    "create_discrete_simulator",
    "plot_activity_delays",
    "retrieve_absolute_and_relative_delays",
    "__version__",
]
