"""Public interface for the :mod:`mc_dagprop` package."""

from importlib.metadata import version

try:
    from .monte_carlo import (
        Activity,
        DagContext,
        Event,
        EventTimestamp,
        GenericDelayGenerator,
        MonteCarloPropagator,
        SimResult,
        Simulator,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - compiled module missing
    raise ImportError(
        "mc_dagprop requires the compiled extension 'mc_dagprop.monte_carlo._core'. "
        "Install the package from source to build it."
    ) from exc
from .frontend import (
    DelayFamilyRegistry,
    PropagationContext,
    analytic_from_context,
    monte_carlo_from_context,
    validate_propagation_context,
)
from .analytic import (
    AnalyticContext,
    AnalyticPropagator,
    DiscretePMF,
    OverflowRule,
    SimulatedEvent,
    UnderflowRule,
    create_analytic_propagator,
)

__version__ = version("mc-dagprop")

__all__ = [
    "DelayFamilyRegistry",
    "PropagationContext",
    "validate_propagation_context",
    "analytic_from_context",
    "monte_carlo_from_context",
    "GenericDelayGenerator",
    "DagContext",
    "SimResult",
    "Event",
    "Activity",
    "Simulator",
    "MonteCarloPropagator",
    "EventTimestamp",
    "DiscretePMF",
    "SimulatedEvent",
    "UnderflowRule",
    "OverflowRule",
    "AnalyticContext",
    "AnalyticPropagator",
    "create_analytic_propagator",
    "__version__",
]

MonteCarloPropagator.from_context = classmethod(
    lambda cls, context, registry: monte_carlo_from_context(context, registry)
)
AnalyticPropagator.from_context = classmethod(
    lambda cls, context, registry, *, step, overflow_rule, underflow_rule: analytic_from_context(
        context,
        registry,
        step=step,
        overflow_rule=overflow_rule,
        underflow_rule=underflow_rule,
    )
)
