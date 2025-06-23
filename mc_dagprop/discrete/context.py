from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

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
    step_size: float = 0.0

    def validate(self) -> None:
        for _, edge in self.activities.values():
            if not np.isclose(edge.pmf.step, self.step_size):
                raise ValueError(
                    f"edge PMF step {edge.pmf.step} does not match context step size {self.step_size}"
                )
