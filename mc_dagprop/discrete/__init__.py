from __future__ import annotations

from typing import NewType

from .context import AnalyticContext, ScheduledEvent, SimulatedEvent, UnderflowRule, OverflowRule
from .pmf import DiscretePMF
from .simulator import DiscreteSimulator, create_discrete_simulator

Second = NewType("Second", float)
ProbabilityMass = NewType("ProbabilityMass", float)
NodeIndex = NewType("NodeIndex", int)
EdgeIndex = NewType("EdgeIndex", int)

__all__ = [
    "DiscretePMF",
    "ScheduledEvent",
    "SimulatedEvent",
    "UnderflowRule",
    "OverflowRule",
    "AnalyticContext",
    "DiscreteSimulator",
    "create_discrete_simulator",
    "Second",
    "ProbabilityMass",
    "NodeIndex",
    "EdgeIndex",
]
