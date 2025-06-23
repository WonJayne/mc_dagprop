from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from mc_dagprop import SimEvent

from .pmf import DiscretePMF

NodeIndex = int
EdgeIndex = int
Pred = tuple[NodeIndex, EdgeIndex]


@dataclass
class AnalyticEdge:
    pmf: DiscretePMF


@dataclass
class AnalyticContext:
    events: List[SimEvent]
    activities: Dict[tuple[NodeIndex, NodeIndex], tuple[EdgeIndex, AnalyticEdge]]
    precedence_list: List[tuple[NodeIndex, List[Pred]]]
    max_delay: float = 0.0
