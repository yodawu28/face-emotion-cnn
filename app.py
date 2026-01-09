"""
Face Recognition & Emotion Detection App
=========================================
A Streamlit app using InsightFace for face recognition
and a custom CNN for emotion classification.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, List

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from src.emotion_classifier import EmotionClassifier
from src.emotion_ema import EmotionEMA
from src.emotion_infer import crop_with_margin
from src.emotion_ui import draw_emotion_row_on_frame, load_icons_as_badges
from src.face_engine import FaceEngine, draw_overlays


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class EmotionConfig:
    """Configuration for emotion detection and switching logic."""
    model_path: str = "models/emotion_resnet_fer_model_ts.pth"  # FER+ trained model (76.76% val acc)
    device: str = "cpu"
    
    # Multi-frame aggregation (NEW)
    use_multi_frame: bool = True      # Enable temporal pooling
    aggregate_window: float = 1.0     # Aggregate over 1 second
    min_frames_required: int = 3      # Need at least 3 frames (first, middle, last)
    aggregation_method: str = "mean"  # "mean", "max", or "voting"
    use_first_middle_last: bool = True  # Sample only first, middle, last (not all frames)
    use_recency_weighting: bool = True  # Weight recent predictions more heavily
    adaptive_window: bool = True        # Use shorter window during transitions
    transition_gap_threshold: float = 0.10  # Lower threshold to detect transitions
    snap_on_strong_new: bool = True     # Snap immediately on very strong new emotion
    strong_new_gap: float = 0.35        # Gap threshold for strong new emotion
    
    # Two-stage smoothing: Buffer → EMA (NEW)
    use_buffer_ema: bool = True       # Apply EMA on top of buffer aggregation
    buffer_ema_alpha: float = 0.5     # EMA alpha for aggregated output (0.5 = moderate smoothing)
    
    # EMA smoothing (used standalone when multi-frame disabled)
    ema_alpha: float = 0.40           # Alpha for legacy EMA-only mode
    inference_interval: float = 0.10  # Sample more frequently (10 FPS)
    
    # Thresholds (Tuned for FER+ model)
    min_confidence: float = 0.30      # Accept predictions > 30% confidence
    use_gap_filter: bool = False      # Disable gap filter - FER+ model has smaller gaps
    min_gap: float = 0.03             # Only 3% gap needed (rarely used)
    
    # Switching (reduced with multi-frame stability)
    snap_confidence: float = 0.75     # High for instant snap
    snap_gap: float = 0.45            # Clear prediction needed
    switch_streak_required: int = 2   # Reduced to 2 for faster UI response
    switch_min_gap: float = 0.08      # Lowered to allow easier switching (8% gap)
    
    # Debug visualization
    debug_save_faces: bool = True    # Save preprocessed faces for inspection


@dataclass
class FaceConfig:
    """Configuration for face detection quality gates."""
    min_face_size: int = 120          # Lower to reduce skips (was 130)
    min_detection_score: float = 0.75  # Increased for better face quality (was 0.65)
    bbox_ttl: float = 1.0
    crop_margin: float = 0.35


@dataclass
class UIConfig:
    """Configuration for UI elements."""
    icon_dir: str = "assets/icons"
    badge_padding: int = 8
    badge_bg_alpha: int = 160
    badge_anchor: str = "top-right"
    badge_margin: int = 12
    badge_gap: int = 10
    dim_opacity: float = 0.18


# =============================================================================
# Streamlit Setup
# =============================================================================

st.set_page_config(page_title="Face Recognition Demo", layout="wide")


@st.cache_resource
def get_face_engine() -> FaceEngine:
    """Initialize and cache the face recognition engine."""
    return FaceEngine(
        model_name="buffalo_s",
        providers=["CPUExecutionProvider"],
        det_size=(320, 320),  # buffalo_l needs larger det_size for accuracy
        db_dir="db/embeddings",
        threshold=0.35,
        unknown_percent_cutoff=30.0,
    )


# =============================================================================
# Multi-Frame Prediction Buffer
# =============================================================================

class PredictionBuffer:
    """
    Buffer to store recent predictions for temporal aggregation.
    
    Stores (timestamp, probs, confidence) tuples.
    """
    
    def __init__(self, window_seconds: float = 1.0, max_size: int = 30):
        self.window = window_seconds
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, probs: np.ndarray, confidence: float):
        """Add a new prediction to the buffer."""
        with self.lock:
            self.buffer.append((time.time(), probs.copy(), confidence))
    
    def get_recent(self, window: Optional[float] = None) -> List[tuple]:
        """Get predictions within the time window."""
        if window is None:
            window = self.window
        
        cutoff = time.time() - window
        
        with self.lock:
            return [(ts, p, c) for ts, p, c in self.buffer if ts >= cutoff]
    
    def get_first_middle_last(self, window: Optional[float] = None) -> List[tuple]:
        """Get only first, middle, and last predictions from the window."""
        recent = self.get_recent(window)
        
        if len(recent) < 3:
            return recent
        
        # Sample first, middle, last
        first = recent[0]
        middle = recent[len(recent) // 2]
        last = recent[-1]
        
        return [first, middle, last]
    
    def get_recent_weighted(self, window: Optional[float] = None, use_recent_half: bool = False) -> List[tuple]:
        """
        Get recent predictions, optionally focusing on most recent half.
        
        Args:
            window: Time window in seconds
            use_recent_half: If True, only return most recent 50% of predictions
        """
        recent = self.get_recent(window)
        
        if use_recent_half and len(recent) >= 4:
            # Only use second half for faster adaptation
            start_idx = len(recent) // 2
            return recent[start_idx:]
        
        return recent
    
    def get_last_n(self, n: int = 2) -> List[tuple]:
        """
        Get only the last N predictions (ignoring time window).
        Used for very fast transitions when new emotion is strong.
        """
        with self.lock:
            if len(self.buffer) <= n:
                return list(self.buffer)
            return list(self.buffer)[-n:]
    
    def clear(self):
        """Clear the buffer."""
        with self.lock:
            self.buffer.clear()
    
    @staticmethod
    def aggregate_mean(predictions: List[tuple], use_recency_weighting: bool = False) -> tuple:
        """
        Aggregate predictions using weighted mean.
        
        Args:
            predictions: List of (timestamp, probs, confidence) tuples
            use_recency_weighting: Weight recent predictions more heavily
        
        Returns: (aggregated_probs, mean_confidence)
        """
        if not predictions:
            return None, 0.0
        
        # Extract probs and confidences
        all_probs = np.array([p for _, p, _ in predictions])
        all_confs = np.array([c for _, _, c in predictions])
        
        if use_recency_weighting:
            # Apply exponential decay: recent predictions get much more weight
            # timestamps are already sorted (oldest to newest)
            n = len(predictions)
            # Create exponential weights: older=0.2, newer=1.0 (5x more weight to recent)
            time_weights = np.linspace(0.2, 1.0, n) ** 2  # Square for more aggressive decay
            
            # Combine confidence and recency weights
            weights = all_confs * time_weights
            weights = weights / (weights.sum() + 1e-8)
        else:
            # Original: weight only by confidence
            weights = all_confs / (all_confs.sum() + 1e-8)
        
        agg_probs = np.average(all_probs, axis=0, weights=weights)
        mean_conf = float(np.mean(all_confs))
        
        return agg_probs, mean_conf
    
    @staticmethod
    def aggregate_max(predictions: List[tuple]) -> tuple:
        """
        Aggregate by taking max probability for each class.
        
        Returns: (aggregated_probs, max_confidence)
        """
        if not predictions:
            return None, 0.0
        
        all_probs = np.array([p for _, p, _ in predictions])
        all_confs = np.array([c for _, _, c in predictions])
        
        # Max pooling across time
        agg_probs = np.max(all_probs, axis=0)
        max_conf = float(np.max(all_confs))
        
        return agg_probs, max_conf
    
    @staticmethod
    def aggregate_voting(predictions: List[tuple]) -> tuple:
        """
        Aggregate by majority voting of top predictions.
        
        Returns: (aggregated_probs, vote_confidence)
        """
        if not predictions:
            return None, 0.0
        
        # Get top prediction from each frame
        votes = [int(np.argmax(p)) for _, p, _ in predictions]
        all_probs = np.array([p for _, p, _ in predictions])
        
        # Count votes
        vote_counts = np.bincount(votes, minlength=all_probs.shape[1])
        winner = int(np.argmax(vote_counts))
        vote_ratio = vote_counts[winner] / len(votes)
        
        # Create prob vector with winner having vote_ratio
        agg_probs = np.zeros(all_probs.shape[1], dtype=np.float32)
        agg_probs[winner] = vote_ratio
        
        # Distribute remaining prob to other classes proportionally
        if vote_ratio < 1.0:
            num_classes = all_probs.shape[1]
            other_mask = np.arange(num_classes) != winner
            other_probs = np.mean(all_probs[:, other_mask], axis=0)
            other_probs = other_probs / (other_probs.sum() + 1e-8) * (1.0 - vote_ratio)
            agg_probs[other_mask] = other_probs
        
        return agg_probs, float(vote_ratio)


# =============================================================================
# Video Processor
# =============================================================================

class VideoProcessor(VideoProcessorBase):
    """
    Process video frames for face recognition and emotion detection.

    Pipeline:
    1. Receive frame from webcam
    2. Run face detection/recognition (async, throttled)
    3. Crop face ROI and predict emotion (with EMA smoothing)
    4. Draw overlays and emotion badges
    5. Return processed frame
    """

    def __init__(self):
        # Load configs
        self.emotion_cfg = EmotionConfig()
        self.face_cfg = FaceConfig()
        self.ui_cfg = UIConfig()

        # Thread safety
        self.lock = threading.Lock()

        # Face engine (shared via Streamlit cache)
        self.engine = get_face_engine()

        # Emotion classifier
        self.emotion_model = EmotionClassifier(
            model_path=self.emotion_cfg.model_path,
            labels_path=None,
            device=self.emotion_cfg.device,
            debug_save_faces=self.emotion_cfg.debug_save_faces,
        )

        # EMA smoother (optional, used when multi-frame is disabled)
        self.ema = EmotionEMA(n_classes=7, alpha=self.emotion_cfg.ema_alpha)
        
        # Multi-frame prediction buffer (NEW)
        self.pred_buffer = PredictionBuffer(
            window_seconds=self.emotion_cfg.aggregate_window,
            max_size=30
        )
        
        # EMA smoother for aggregated output (two-stage: Buffer → EMA)
        self.agg_ema = EmotionEMA(n_classes=7, alpha=self.emotion_cfg.buffer_ema_alpha)

        # State: face detection
        self._last_bbox: Optional[tuple] = None
        self._last_bbox_ts: float = 0.0
        self._last_det_score: float = 0.0
        self._last_recognitions: List = []

        # State: emotion
        self._last_emotion: Optional[str] = None
        self._last_emotion_conf: float = 0.0
        self._last_emotion_ts: float = 0.0

        # State: emotion switching
        self._switch_candidate: Optional[str] = None
        self._switch_streak: int = 0

        # State: inference throttling
        self._last_infer_ts: float = 0.0
        self._infer_running: bool = False

        # State: UI badges
        self._badges: Optional[dict] = None
        self._badge_size: Optional[int] = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Process a single video frame (called by streamlit-webrtc)."""
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        # Get cached state (thread-safe)
        with self.lock:
            recognitions = self._last_recognitions
            selected_emotion = self._last_emotion

        # Draw face recognition overlays
        vis = draw_overlays(img, recognitions)

        # Draw emotion badges
        h, w = vis.shape[:2]
        self._ensure_badges_cached(h, w)

        if self._badges:
            vis = draw_emotion_row_on_frame(
                vis,
                badges=self._badges,
                selected=selected_emotion,
                anchor=self.ui_cfg.badge_anchor,
                margin=self.ui_cfg.badge_margin,
                gap=self.ui_cfg.badge_gap,
                dim_opacity=self.ui_cfg.dim_opacity,
            )

        # Kick off async inference (throttled)
        self._maybe_start_inference(img, now)

        return av.VideoFrame.from_ndarray(vis, format="bgr24")

    # -------------------------------------------------------------------------
    # Inference (runs in background thread)
    # -------------------------------------------------------------------------

    def _maybe_start_inference(self, img: np.ndarray, now: float) -> None:
        """Start inference thread if enough time has passed."""
        interval = self.emotion_cfg.inference_interval

        if (now - self._last_infer_ts) >= interval and not self._infer_running:
            self._infer_running = True
            self._last_infer_ts = now
            thread = threading.Thread(
                target=self._run_inference,
                args=(img.copy(),),
                daemon=True
            )
            thread.start()

    def _run_inference(self, img_bgr: np.ndarray) -> None:
        """Run face recognition and emotion detection."""
        try:
            now = time.time()

            # Step 1: Face detection/recognition
            recognitions = self.engine.recognize(img_bgr, max_faces=1)
            bbox, det_score = self._get_face_bbox(recognitions, now)

            # Step 2: Emotion prediction
            emotion_label, emotion_conf = self._predict_emotion(
                img_bgr, bbox, det_score, now
            )

            # Step 3: Commit state (thread-safe)
            with self.lock:
                self._last_recognitions = recognitions
                if emotion_label is not None:
                    if emotion_label != self._last_emotion:
                        print(
                            f"[STATE] Emotion: {self._last_emotion} → {emotion_label}")
                    self._last_emotion = emotion_label
                    self._last_emotion_conf = emotion_conf

        finally:
            self._infer_running = False

    def _get_face_bbox(self, recognitions: List, now: float) -> tuple:
        """Get face bounding box with TTL caching."""
        if recognitions:
            r = recognitions[0]
            bbox = r.bbox
            det_score = float(getattr(r, "det_score", 0.0))

            # Cache for future frames
            self._last_bbox = bbox
            self._last_bbox_ts = now
            self._last_det_score = det_score

            return bbox, det_score

        # Use cached bbox if within TTL
        if self._last_bbox and (now - self._last_bbox_ts) <= self.face_cfg.bbox_ttl:
            return self._last_bbox, self._last_det_score

        return None, 0.0

    def _predict_emotion(
        self,
        img_bgr: np.ndarray,
        bbox: Optional[tuple],
        det_score: float,
        now: float
    ) -> tuple:
        """
        Predict emotion from face ROI with multi-frame temporal aggregation.
        
        NEW: Uses PredictionBuffer to aggregate predictions over time instead
        of single-frame + EMA smoothing.
        """
        cfg = self.emotion_cfg

        # Check if we should run emotion inference
        if bbox is None:
            return None, 0.0

        if (now - self._last_emotion_ts) < cfg.inference_interval:
            return None, 0.0

        # Check face quality gates
        x1, y1, x2, y2 = bbox
        w, h = int(x2 - x1), int(y2 - y1)

        if w < self.face_cfg.min_face_size or h < self.face_cfg.min_face_size:
            print(f"[EMO] skip: face too small ({w}x{h})")
            return None, 0.0

        if det_score < self.face_cfg.min_detection_score:
            print(f"[EMO] skip: det_score={det_score:.2f}")
            return None, 0.0

        # Crop face ROI
        roi = crop_with_margin(img_bgr, bbox, margin=self.face_cfg.crop_margin)
        if roi is None:
            return None, 0.0

        # Get raw prediction
        pred = self.emotion_model.predict_from_roi(roi)
        raw_probs = np.asarray(pred.probs, dtype=np.float32).ravel()
        raw_idx = int(np.argmax(raw_probs))
        raw_conf = float(raw_probs[raw_idx])
        raw_label = self.emotion_model.labels[raw_idx]
        raw_gap = self._compute_top2_gap(raw_probs)

        # Add to buffer
        self.pred_buffer.add(raw_probs, raw_conf)

        # Multi-frame aggregation (NEW)
        if cfg.use_multi_frame:
            # Detect transition: RAW shows different emotion from current with any gap
            is_different = (self._last_emotion is not None and 
                           raw_label != self._last_emotion)
            
            # Strong new emotion: RAW shows NEW emotion with high confidence gap
            is_strong_new = is_different and raw_gap >= cfg.strong_new_gap
            
            # Normal transition: RAW differs with moderate confidence
            in_transition = is_different and raw_gap >= cfg.transition_gap_threshold
            
            # Determine sampling strategy based on transition strength
            if is_strong_new and cfg.snap_on_strong_new:
                # Very strong new emotion - use only last 2 predictions for fast snap
                recent = self.pred_buffer.get_last_n(2)
                mode_label = "SNAP"
            elif cfg.adaptive_window and in_transition:
                # Normal transition - use shorter window and recent half
                window = cfg.aggregate_window * 0.5
                recent = self.pred_buffer.get_recent_weighted(window, use_recent_half=True)
                mode_label = "TRANS"
            elif cfg.use_first_middle_last:
                # Steady state - use FML sampling
                recent = self.pred_buffer.get_first_middle_last(cfg.aggregate_window)
                mode_label = "FML"
            else:
                recent = self.pred_buffer.get_recent(cfg.aggregate_window)
                mode_label = "ALL"
            
            # Need minimum frames (lower requirement during transitions)
            min_required = 2 if (in_transition or is_strong_new) else cfg.min_frames_required
            if len(recent) < min_required:
                print(f"[EMO] Buffering... {len(recent)}/{min_required}")
                return None, 0.0
            
            # Aggregate based on method
            # Use aggressive recency weighting during transitions
            use_recency = cfg.use_recency_weighting or in_transition
            
            if cfg.aggregation_method == "mean":
                agg_probs, agg_conf = PredictionBuffer.aggregate_mean(
                    recent, 
                    use_recency_weighting=use_recency
                )
            elif cfg.aggregation_method == "max":
                agg_probs, agg_conf = PredictionBuffer.aggregate_max(recent)
            elif cfg.aggregation_method == "voting":
                agg_probs, agg_conf = PredictionBuffer.aggregate_voting(recent)
            else:
                raise ValueError(f"Unknown aggregation method: {cfg.aggregation_method}")
            
            if agg_probs is None:
                return None, 0.0
            
            # Stage 2: Apply EMA on aggregated output (NEW two-stage smoothing)
            if cfg.use_buffer_ema:
                # Snap EMA on strong transitions for faster response
                if is_strong_new:
                    self.agg_ema.ema = agg_probs.copy()  # Keep as numpy array
                else:
                    self.agg_ema.update(agg_probs.tolist())
                
                # Use EMA-smoothed aggregation
                final_probs = np.asarray(self.agg_ema.ema, dtype=np.float32).ravel()
            else:
                # No EMA, use raw aggregation
                final_probs = agg_probs
            
            # Get final prediction
            final_idx = int(np.argmax(final_probs))
            final_label = self.emotion_model.labels[final_idx]
            final_conf = float(final_probs[final_idx])
            final_gap = self._compute_top2_gap(final_probs)
            
            # Update timestamp
            self._last_emotion_ts = now
            
            # Debug log
            trans_marker = "*" if is_strong_new else ("~" if in_transition else "")
            ema_tag = "+EMA" if cfg.use_buffer_ema else ""
            print(
                f"[EMO] RAW: {raw_label} {raw_conf:.3f} gap={raw_gap:.3f}{trans_marker} | "
                f"AGG ({cfg.aggregation_method}-{mode_label}{ema_tag}, n={len(recent)}): {final_label} {final_conf:.3f} gap={final_gap:.3f} | "
                f"det={det_score:.2f}"
            )
            
            # Apply switching logic with final smoothed values
            return self._apply_emotion_switching(
                final_label, final_conf, final_gap,
                raw_conf, raw_gap
            )
        
        else:
            # Legacy EMA smoothing path
            self.ema.update(raw_probs.tolist())
            self._maybe_snap_ema(raw_probs, raw_idx, raw_conf, raw_gap)

            # Get smoothed prediction
            ema_probs = np.asarray(self.ema.ema, dtype=np.float32).ravel()
            ema_idx = int(np.argmax(ema_probs))
            ema_conf = float(ema_probs[ema_idx])
            ema_label = self.emotion_model.labels[ema_idx]
            ema_gap = self._compute_top2_gap(ema_probs)

            # Update timestamp
            self._last_emotion_ts = now

            # Debug log
            print(
                f"[EMO] RAW: {raw_label} {raw_conf:.3f} gap={raw_gap:.3f} | "
                f"EMA: {ema_label} {ema_conf:.3f} | det={det_score:.2f}"
            )

            # Apply switching logic
            return self._apply_emotion_switching(
                ema_label, ema_conf, ema_gap,
                raw_conf, raw_gap
            )

    def _apply_emotion_switching(
        self,
        candidate: str,
        cand_conf: float,
        ema_gap: float,
        raw_conf: float,
        raw_gap: float
    ) -> tuple:
        """Apply streak-based debounce logic for emotion switching."""
        cfg = self.emotion_cfg

        # Check acceptance threshold
        if cand_conf < cfg.min_confidence:
            print(f"[REJECT] Low confidence: {candidate} {cand_conf:.3f} < {cfg.min_confidence}")
            return None, 0.0

        if cfg.use_gap_filter and ema_gap < cfg.min_gap:
            print(f"[REJECT] Low gap: {candidate} gap={ema_gap:.3f} < {cfg.min_gap}")
            return None, 0.0
            return None, 0.0

        current = self._last_emotion

        # Case 1: No current emotion → accept immediately
        if current is None:
            return candidate, cand_conf

        # Case 2: Same emotion → refresh confidence, reset switch state
        if candidate == current:
            self._switch_candidate = None
            self._switch_streak = 0
            return candidate, cand_conf

        # Case 3: Different emotion → apply switching logic

        # Require minimum gap to even consider switching (prevents noise)
        min_switch_gap = getattr(cfg, 'switch_min_gap', 0.04)
        if ema_gap < min_switch_gap:
            # Gap too small - prediction is uncertain, keep current
            return current, self._last_emotion_conf

        hard_snap = (
            raw_conf >= cfg.snap_confidence and
            raw_gap >= cfg.snap_gap
        )

        # Count consecutive detections of same candidate
        if self._switch_candidate == candidate:
            self._switch_streak += 1
        else:
            self._switch_candidate = candidate
            self._switch_streak = 1

        # Switch if hard snap OR enough consecutive detections
        if hard_snap or self._switch_streak >= cfg.switch_streak_required:
            print(
                f"[SWITCH] {current} → {candidate} (snap={hard_snap}, streak={self._switch_streak}, gap={ema_gap:.3f})")
            self._switch_candidate = None
            self._switch_streak = 0
            return candidate, cand_conf

        # Keep current emotion while waiting
        return current, self._last_emotion_conf

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_top2_gap(probs: np.ndarray) -> float:
        """Compute gap between top-2 probabilities."""
        p = np.asarray(probs, dtype=np.float32).ravel()
        if p.size < 2:
            return 0.0
        top2 = np.partition(p, -2)[-2:]
        return float(np.max(top2) - np.min(top2))

    def _maybe_snap_ema(
        self,
        raw_probs: np.ndarray,
        raw_idx: int,
        raw_conf: float,
        raw_gap: float
    ) -> None:
        """Snap EMA to raw prediction if confidence is very high."""
        if self.ema.ema is None:
            return

        ema_arr = np.asarray(self.ema.ema, dtype=np.float32).ravel()
        ema_idx = int(np.argmax(ema_arr))

        cfg = self.emotion_cfg
        should_snap = (
            raw_conf >= cfg.snap_confidence and
            raw_gap >= cfg.snap_gap and
            raw_idx != ema_idx
        )

        if should_snap:
            self.ema.ema = raw_probs.copy()

    def _ensure_badges_cached(self, frame_h: int, frame_w: int) -> None:
        """Load badges if not cached or size changed significantly."""
        short_side = min(frame_w, frame_h)
        target_size = int(max(36, min(80, short_side * 0.07)))

        should_reload = (
            self._badges is None or
            self._badge_size is None or
            abs(target_size - self._badge_size) >= 4
        )

        if should_reload:
            self._badge_size = target_size
            self._badges = load_icons_as_badges(
                icon_dir=self.ui_cfg.icon_dir,
                badge_size=self._badge_size,
                pad=self.ui_cfg.badge_padding,
                bg_alpha=self.ui_cfg.badge_bg_alpha,
                try_remove_white_bg=True,
            )


# =============================================================================
# Sidebar UI
# =============================================================================

def render_sidebar(engine: FaceEngine) -> None:
    """Render the enrollment and database controls in sidebar."""
    with st.sidebar:
        st.header("📸 Enroll User")

        name = st.text_input("Name", placeholder="Enter name...")
        uploads = st.file_uploader(
            "Upload photos (jpg/png)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if st.button("Enroll", type="primary"):
            _handle_enrollment(engine, name, uploads)

        st.divider()

        st.header("📁 Database")
        if st.button("Reload"):
            engine.reload_db()
            st.success("Database reloaded!")

        known_users = list(engine.known.keys())
        if known_users:
            st.write("**Known users:**", ", ".join(known_users))
        else:
            st.write("No users enrolled yet.")


def _handle_enrollment(engine: FaceEngine, name: str, uploads) -> None:
    """Process user enrollment from uploaded images."""
    if not name.strip():
        st.error("Please enter a name.")
        return

    if not uploads:
        st.error("Please upload at least one photo.")
        return

    images = []
    for f in uploads:
        data = np.frombuffer(f.read(), dtype=np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is not None:
            images.append(bgr)

    if not images:
        st.error("Could not decode any images.")
        return

    try:
        emb = engine.enroll(name.strip(), images)
        st.success(f"✓ Enrolled '{name}' (embedding dim={emb.shape[0]})")
    except Exception as e:
        st.error(f"Enrollment failed: {e}")


# =============================================================================
# Camera Stream
# =============================================================================

def render_camera() -> None:
    """Render the camera stream with WebRTC."""
    st.subheader("📹 Camera")

    webrtc_streamer(
        key="face-emotion-cam",
        video_processor_factory=VideoProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": {
                "width": {"min": 480, "ideal": 640, "max": 800},
                "height": {"min": 360, "ideal": 480, "max": 600},
                "frameRate": {"min": 15, "ideal": 24, "max": 30},
            },
            "audio": False,
        },
        async_processing=True,
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main application entry point."""
    st.title("🎭 Face Recognition & Emotion Detection")

    engine = get_face_engine()

    render_sidebar(engine)
    render_camera()


if __name__ == "__main__":
    main()
