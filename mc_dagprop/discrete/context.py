from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

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
    step_size: float = 0.0

    def validate(self) -> None:
        for _, edge in self.activities.values():
            if not np.isclose(edge.pmf.step, self.step_size):
                raise ValueError(
                    f"edge PMF step {edge.pmf.step} does not match context step size {self.step_size}"
                )
