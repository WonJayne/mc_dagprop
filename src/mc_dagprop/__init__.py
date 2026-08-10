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
from .analytic import (
    AnalyticContext,
    AnalyticPropagator,
    DiscretePMF,
    OverflowRule,
    SimulatedEvent,
    UnderflowRule,
    create_analytic_propagator,
    validate_exact_equivalence_domain,
)
from .frontend import (
    DelayFamilyRegistry,
    EquivalenceDomainError,
    EquivalenceMode,
    PropagationContext,
    analytic_from_context,
    monte_carlo_from_context,
    validate_equivalence_domain,
    validate_propagation_context,
)

__version__ = version("mc-dagprop")

__all__ = [
    "Activity",
    "AnalyticContext",
    "AnalyticPropagator",
    "DagContext",
    "DelayFamilyRegistry",
    "DiscretePMF",
    "EquivalenceDomainError",
    "EquivalenceMode",
    "Event",
    "EventTimestamp",
    "GenericDelayGenerator",
    "MonteCarloPropagator",
    "OverflowRule",
    "PropagationContext",
    "SimResult",
    "SimulatedEvent",
    "Simulator",
    "UnderflowRule",
    "__version__",
    "analytic_from_context",
    "create_analytic_propagator",
    "monte_carlo_from_context",
    "validate_equivalence_domain",
    "validate_exact_equivalence_domain",
    "validate_propagation_context",
]
