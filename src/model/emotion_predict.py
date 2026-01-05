
from dataclasses import dataclass
from typing import List


@dataclass
class EmotionPredict:
    label: str
    conf: float
    probs: List[float]