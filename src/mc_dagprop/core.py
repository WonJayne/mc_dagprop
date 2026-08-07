"""Compatibility layer re-exporting the core dataclasses from the C++ module."""

from __future__ import annotations

from .monte_carlo import Activity, DagContext, Event, EventTimestamp

__all__ = ["Activity", "DagContext", "Event", "EventTimestamp"]
