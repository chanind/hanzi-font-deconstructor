from dataclasses import dataclass
from typing import Tuple

@dataclass
class TransformedStroke:
    translate: Tuple[float, float]
    rotate: float
    skewX: float
    skewY: float
    scale: Tuple[float, float]
    path: str