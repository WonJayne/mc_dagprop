from importlib.metadata import version

try:
    from ._core import (
        EventTimestamp,
        GenericDelayGenerator,
        SimActivity,
        SimContext,
        SimEvent,
        SimResult,
        Simulator,
    )
except ModuleNotFoundError:  # pragma: no cover - optional native module
    try:  # fallback to extension from an installed package
        import importlib.util
        import glob
        import os
        import sys

        core_path = None
        for _p in sys.path[1:]:
            cand = os.path.join(_p, "mc_dagprop")
            if os.path.isdir(cand):
                matches = glob.glob(os.path.join(cand, "_core.*"))
                if matches:
                    core_path = matches[0]
                    break
        if core_path is None:
            raise FileNotFoundError

        spec = importlib.util.spec_from_file_location("mc_dagprop._core", core_path)
        if spec is None or spec.loader is None:
            raise ImportError
        _c = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_c)

        EventTimestamp = _c.EventTimestamp
        GenericDelayGenerator = _c.GenericDelayGenerator
        SimActivity = _c.SimActivity
        SimContext = _c.SimContext
        SimEvent = _c.SimEvent
        SimResult = _c.SimResult
        Simulator = _c.Simulator
    except Exception:  # pragma: no cover - give up
        EventTimestamp = GenericDelayGenerator = SimActivity = SimContext = SimEvent = Simulator = SimResult = None  # type: ignore
from .discrete import (
    AnalyticContext,
    ScheduledEvent,
    SimulatedEvent,
    DiscretePMF,
    DiscreteSimulator,
    create_discrete_simulator,
)
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
    "AnalyticContext",
    "DiscreteSimulator",
    "create_discrete_simulator",
    "plot_activity_delays",
    "retrieve_absolute_and_relative_delays",
    "__version__",
]
