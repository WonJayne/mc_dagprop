from __future__ import annotations

from typing import NewType

__all__ = [
    "Second",
    "ProbabilityMass",
    "NodeIndex",
    "EdgeIndex",
    "EventIndex",
    "ActivityIndex",
    "ActivityType",
    "EventId",
]

Second = NewType("Second", float)
ProbabilityMass = NewType("ProbabilityMass", float)

NodeIndex = NewType("NodeIndex", int)
EdgeIndex = NewType("EdgeIndex", int)

EventIndex = NewType("EventIndex", int)
ActivityIndex = NewType("ActivityIndex", int)
ActivityType = NewType("ActivityType", int)
EventId = NewType("EventId", str)
