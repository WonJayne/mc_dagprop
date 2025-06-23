from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from mc_dagprop import SimEvent

from .pmf import DiscretePMF

NodeIndex = int
EdgeIndex = int
Pred = Tuple[NodeIndex, EdgeIndex]


@dataclass
class AnalyticEdge:
    pmf: DiscretePMF


@dataclass
class AnalyticContext:
    events: List[SimEvent]
    activities: Dict[Tuple[NodeIndex, NodeIndex], Tuple[EdgeIndex, AnalyticEdge]]
    precedence_list: List[Tuple[NodeIndex, List[Pred]]]
    max_delay: float = 0.0
