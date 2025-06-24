from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

from .types import ActivityIndex, ActivityType, EventId, EventIndex, Second


__all__ = ["EventTimestamp", "Event", "Activity", "DagContext"]


@dataclass(slots=True, frozen=True)
class EventTimestamp:
    """Scheduling bounds and baseline timestamp for an event."""

    earliest: Second
    latest: Second
    actual: Second


@dataclass(slots=True, frozen=True)
class Event:
    """Node with identifier and timing information."""

    event_id: EventId
    timestamp: EventTimestamp


@dataclass(slots=True, frozen=True)
class Activity:
    """Edge definition with minimal duration and type identifier."""

    minimal_duration: Second
    activity_type: ActivityType


@dataclass(slots=True, frozen=True)
class DagContext:
    """Container describing a DAG for simulation."""

    events: Sequence[Event]
    activities: Mapping[tuple[EventIndex, EventIndex], tuple[ActivityIndex, Activity]]
    precedence_list: Sequence[tuple[EventIndex, Sequence[tuple[EventIndex, ActivityIndex]]]]
    max_delay: Second
