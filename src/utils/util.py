import math

import numpy as np


def l2norm(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2 Normalization of a numpy array."""
    n = float(np.linalg.norm(x))
    return x / (n + eps)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two numpy arrays."""
    a_norm = l2norm(a)
    b_norm = l2norm(b)
    return float(np.dot(a_norm, b_norm))


def sim_to_percent(similarity: float, center: float = 0.47, steep: float = 16.0) -> float:
    """Convert similarity score to confidence percentage use sigmod method."""
    # sim <= thresh => 0%, sim = 1 => 100%
    return 100.0 / (1.0 + math.exp(-steep * (similarity - center)))
