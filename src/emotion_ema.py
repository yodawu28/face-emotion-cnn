import numpy as np


class EmotionEMA:
    """
    Exponential Moving Average for emotion probabilities.
    
    Uses adaptive alpha: higher confidence predictions get more weight.
    This helps when the model is uncertain (small gaps between emotions).
    """
    
    def __init__(self, n_classes: int, alpha: float = 0.25,
                 strong_conf: float = 0.85, strong_gap: float = 0.40):
        self.base_alpha = alpha
        self.alpha = alpha
        self.ema = None
        self.n = n_classes
        self.strong_conf = strong_conf
        self.strong_gap = strong_gap
        
        # Track stability
        self._last_top_emotion: int = -1
        self._stability_count: int = 0

    def update(self, probs: list[float]) -> tuple[int, float, np.ndarray]:
        p = np.asarray(probs, dtype=np.float32)

        # Compute raw signal metrics
        top2 = np.argsort(p)[-2:][::-1]
        raw_top1 = float(p[top2[0]])
        raw_gap = float(p[top2[0]] - p[top2[1]])
        raw_idx = int(top2[0])

        # Snap immediately if very confident
        if self.ema is None or (raw_top1 >= self.strong_conf and raw_gap >= self.strong_gap):
            self.ema = p.copy()
            self._last_top_emotion = raw_idx
            self._stability_count = 1
        else:
            # Adaptive alpha based on gap (uncertainty measure)
            # gap < 0.05 → very uncertain → alpha = 0.15 * base
            # gap > 0.20 → confident → alpha = 1.0 * base
            if raw_gap < 0.05:
                confidence_factor = 0.15  # Heavy smoothing for uncertain
            elif raw_gap < 0.10:
                confidence_factor = 0.35
            elif raw_gap < 0.15:
                confidence_factor = 0.60
            else:
                confidence_factor = min(1.0, raw_gap / 0.20)
            
            adaptive_alpha = self.base_alpha * confidence_factor
            
            # Extra smoothing if prediction keeps changing
            if raw_idx != self._last_top_emotion:
                self._stability_count = 0
                adaptive_alpha *= 0.5  # Even more smoothing for changes
            else:
                self._stability_count += 1
                # Gradually increase alpha as same emotion persists
                if self._stability_count > 3:
                    adaptive_alpha = min(adaptive_alpha * 1.5, self.base_alpha)
            
            self._last_top_emotion = raw_idx
            self.ema = (1 - adaptive_alpha) * self.ema + adaptive_alpha * p

        idx = int(np.argmax(self.ema))
        conf = float(self.ema[idx])
        return idx, conf, self.ema