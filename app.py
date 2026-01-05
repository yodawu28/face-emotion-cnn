"""
Face Recognition & Emotion Detection App
=========================================
A Streamlit app using InsightFace for face recognition
and a custom CNN for emotion classification.
"""

import threading
import time
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
    model_path: str = "models/emotion_resnet_model_ts.pth"
    device: str = "cpu"
    ema_alpha: float = 0.25           # Much more smoothing
    inference_interval: float = 0.33
    min_confidence: float = 0.45      # Only accept confident predictions
    use_gap_filter: bool = True
    min_gap: float = 0.18             # Require clear winner
    snap_confidence: float = 0.80     # Very high for instant snap
    snap_gap: float = 0.55            # Very clear prediction needed
    switch_streak_required: int = 6   # Need 6 consecutive frames
    switch_min_gap: float = 0.25      # High bar for considering switch


@dataclass
class FaceConfig:
    """Configuration for face detection quality gates."""
    min_face_size: int = 120          # Lower to reduce skips (was 130)
    min_detection_score: float = 0.65
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
        det_size=(320, 320),  # Smaller det_size = faster detection
        db_dir="db/embeddings",
        threshold=0.35,
        unknown_percent_cutoff=30.0,
    )


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
        )

        # EMA smoother
        self.ema = EmotionEMA(n_classes=7, alpha=self.emotion_cfg.ema_alpha)

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
        """Predict emotion from face ROI with EMA smoothing."""
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

        # Update EMA
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
            return None, 0.0

        if cfg.use_gap_filter and ema_gap < cfg.min_gap:
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
