from dataclasses import dataclass
from typing import Tuple


@dataclass
class RecognitionResult:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    det_score: float  # Detection confidence score
    name: str  # Recognized person's name
    similarity: float  # Similarity score with the known face
    percent: float  # Confidence percentage of the recognition
