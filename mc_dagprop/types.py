from __future__ import annotations

from typing import NewType

__all__ = ["Second", "ProbabilityMass", "ActivityIndex", "EventIndex", "ActivityIndex", "ActivityType", "EventId"]

Second = NewType("Second", float)
ProbabilityMass = NewType("ProbabilityMass", float)

ActivityIndex = NewType("ActivityIndex", int)

EventIndex = NewType("EventIndex", int)
ActivityIndex = NewType("ActivityIndex", int)
ActivityType = NewType("ActivityType", int)
EventId = NewType("EventId", str)
