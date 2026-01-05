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
    model_path: str = "models/emotion_model_ts.pth"
    device: str = "cpu"
    ema_alpha: float = 0.75           # Higher = faster response
    inference_interval: float = 0.30
    min_confidence: float = 0.28
    use_gap_filter: bool = True
    min_gap: float = 0.05
    snap_confidence: float = 0.70     # Hard snap threshold
    snap_gap: float = 0.20
    switch_streak_required: int = 3   # Consecutive detections to switch


@dataclass
class FaceConfig:
    """Configuration for face detection quality gates."""
    min_face_size: int = 140
    min_detection_score: float = 0.70
    bbox_ttl: float = 0.8
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
        det_size=(640, 640),
        db_dir="db/embeddings",
        threshold=0.35,
        unknown_percent_cutoff=30.0,
    )


# =============================================================================
# Sidebar UI Functions
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


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # ----- overlay state -----
        self.last_frame = None
        self.last_recognize = []
        self.lock = threading.Lock()

        # ---- badge -----
        self.badges = None
        self.badge_size = None

        # ----- emotion model -----
        self.emotion = EmotionClassifier(
            model_path="models/emotion_model_ts.pth",
            labels_path=None,
            device="cpu",
        )

        # ====== Emotion smoothing/throttle params ======
        self.emo_ema = EmotionEMA(n_classes=7, alpha=0.75)  # Higher = faster response
        self.emotion_interval = 0.30
        self.emotion_conf = 0.28   # cho model “mềm” dễ accept hơn chút

        self.USE_GAP_FILTER = True
        self.MIN_GAP_ACCEPT = 0.05  # 0.03~0.08

        # snap nhanh hơn
        self.SNAP_RAW_CONF = 0.70  # lowered from 0.75
        self.SNAP_RAW_GAP  = 0.20  # lowered from 0.25

        # hysteresis / debounce
        self.SWITCH_STREAK_N = 3          # need 3 consecutive detections to switch
        self.SWITCH_MIN_DELTA = 0.05      # (not used anymore, kept for reference)
        self.switch_candidate = None
        self.switch_streak = 0


        # ----- bbox TTL + quality gate -----
        self.last_face_bbox = None
        self.last_face_ts = 0.0
        self.face_ttl = 0.8

        self.min_face = 140        # thử 120/140/160
        self.min_det  = 0.70       # thử 0.65~0.80
        self.last_det_score = 0.0  # giữ det score lần cuối detect thành công

        # ----- emotion state -----
        self.last_emotion_ts = 0.0
        self.last_emotion_conf = 0.0
        self.last_emotion = None

        # ----- inference throttle -----
        self.infer_interval = 0.4
        self.last_infer_ts = 0.0
        self.infer_running = False

        # ----- badges cache (IMPORTANT: đừng load trong recv mỗi frame) -----
        self.badges = None
        self.badge_size = None

    def get_last_emotion(self):
        with self.lock:
            return self.last_emotion

    @staticmethod
    def _top2_gap(probs: np.ndarray) -> float:
        p = np.asarray(probs, dtype=np.float32).reshape(-1)
        if p.size < 2:
            return 0.0
        top2 = np.partition(p, -2)[-2:]  # 2 giá trị lớn nhất
        return float(np.max(top2) - np.min(top2))

    def _maybe_snap_ema(self, raw_probs: np.ndarray, raw_idx: int, raw_conf: float) -> None:
        """
        Nếu raw cực chắc + khác EMA hiện tại => cho EMA "nhảy" theo raw.
        """
        if self.emo_ema.ema is None:
            return

        ema = np.asarray(self.emo_ema.ema, dtype=np.float32).reshape(-1)
        if ema.size == 0:
            return

        ema_idx = int(np.argmax(ema))
        raw_gap = self._top2_gap(raw_probs)

        if (raw_conf >= self.SNAP_RAW_CONF) and (raw_gap >= self.SNAP_RAW_GAP) and (raw_idx != ema_idx):
            self.emo_ema.ema = np.asarray(raw_probs, dtype=np.float32).reshape(-1)

    def _ema_gap(self) -> float:
        ema = self.emo_ema.ema
        if ema is None:
            return 0.0
        ema = np.asarray(ema, dtype=np.float32).reshape(-1)
        return self._top2_gap(ema)

    def _infer_async(self, img_bgr):
        now = time.time()
        try:
            recognizes = engine.recognize(img_bgr, max_faces=1)

            # --- 1) bbox with TTL (không hack det_score=1.0) ---
            bbox = None
            det_score = 0.0

            if recognizes:
                r0 = recognizes[0]
                bbox = r0.bbox
                det_score = float(getattr(r0, "det_score", 0.0))

                self.last_face_bbox = bbox
                self.last_face_ts = now
                self.last_det_score = det_score
            else:
                if self.last_face_bbox is not None and (now - self.last_face_ts) <= self.face_ttl:
                    bbox = self.last_face_bbox
                    det_score = float(self.last_det_score)  # dùng score cũ, không set 1.0

            # --- 2) emotion prediction (throttled) ---
            emotion_label = None
            emotion_conf = 0.0

            if bbox is not None and (now - self.last_emotion_ts) >= self.emotion_interval:
                x1, y1, x2, y2 = bbox
                w = int(x2 - x1)
                h = int(y2 - y1)

                # gate: face size + det score
                if (w >= self.min_face and h >= self.min_face) and (det_score >= self.min_det):
                    roi = crop_with_margin(img_bgr, bbox, margin=0.35)

                    if roi is not None:
                        # --- RAW ---
                        pred = self.emotion.predict_from_roi(roi)
                        raw_probs = np.asarray(pred.probs, dtype=np.float32).reshape(-1)
                        raw_idx = int(np.argmax(raw_probs))
                        raw_conf = float(raw_probs[raw_idx])
                        raw_label = self.emotion.labels[raw_idx] if raw_idx < len(self.emotion.labels) else str(raw_idx)
                        raw_gap = self._top2_gap(raw_probs)

                        # --- EMA update ---
                        self.emo_ema.update(raw_probs.tolist())

                        # snap nếu raw cực chắc
                        self._maybe_snap_ema(raw_probs, raw_idx, raw_conf)

                        # lấy EMA hiện tại (sau snap)
                        ema = self.emo_ema.ema
                        if ema is None:
                            ema_idx = raw_idx
                            ema_conf = raw_conf
                        else:
                            ema = np.asarray(ema, dtype=np.float32).reshape(-1)
                            ema_idx = int(np.argmax(ema))
                            ema_conf = float(ema[ema_idx])

                        ema_label = self.emotion.labels[ema_idx] if ema_idx < len(self.emotion.labels) else str(ema_idx)

                        # IMPORTANT: update timestamp ngay khi đã chạy emotion inference
                        self.last_emotion_ts = now

                        # DEBUG log
                        print(
                            f"[EMO] RAW: {raw_label} {raw_conf:.3f} gap={raw_gap:.3f} | "
                            f"EMA: {ema_label} {ema_conf:.3f} | det={det_score:.2f} bbox=({w}x{h})"
                        )

                        ema_gap = self._ema_gap()

                        # accept condition
                        accept = (ema_conf >= self.emotion_conf)
                        
                        if self.USE_GAP_FILTER:
                            accept = accept and (ema_gap >= self.MIN_GAP_ACCEPT)

                        if accept:
                            candidate = ema_label
                            cand_conf = float(ema_conf)

                            # current state
                            cur = self.last_emotion
                            cur_conf = float(self.last_emotion_conf or 0.0)

                            # 1) nếu chưa có current -> set luôn
                            if cur is None:
                                emotion_label, emotion_conf = candidate, cand_conf

                            # 2) nếu candidate giống current -> refresh conf luôn
                            elif candidate == cur:
                                emotion_label, emotion_conf = candidate, cand_conf
                                self.switch_candidate = None
                                self.switch_streak = 0

                            # 3) candidate khác current -> streak-based switching
                            else:
                                # hard snap nếu raw rất chắc (bắt chuyển nhanh)
                                raw_gap = self._top2_gap(raw_probs)
                                hard_snap = (raw_conf >= self.SNAP_RAW_CONF and raw_gap >= self.SNAP_RAW_GAP)

                                # Count consecutive frames with same candidate
                                # (removed confidence comparison - just use streak)
                                if self.switch_candidate == candidate:
                                    self.switch_streak += 1
                                else:
                                    self.switch_candidate = candidate
                                    self.switch_streak = 1

                                # Switch if hard_snap OR enough consecutive detections
                                if hard_snap or self.switch_streak >= self.SWITCH_STREAK_N:
                                    emotion_label, emotion_conf = candidate, cand_conf
                                    self.switch_candidate = None
                                    self.switch_streak = 0
                                    print(f"[SWITCH] {cur} → {candidate} (hard_snap={hard_snap}, streak={self.switch_streak})")
                                else:
                                    # Keep showing current emotion while waiting for switch
                                    emotion_label, emotion_conf = cur, cur_conf
                else:
                    # optional debug
                    print(f"[EMO] skip: det={det_score:.2f} bbox=({w}x{h})")
                    pass

            # --- 3) commit state ---
            with self.lock:
                self.last_recognize = recognizes
                if emotion_label is not None:
                    if emotion_label != self.last_emotion:
                        print(f"[STATE] Emotion changed: {self.last_emotion} → {emotion_label}")
                    self.last_emotion = emotion_label
                    self.last_emotion_conf = emotion_conf

        finally:
            self.infer_running = False

    def _ensure_badges(self, frame_h: int, frame_w: int):
        """
        Cache badges theo kích thước khung hình để tránh load lại mỗi frame.
        """
        short_side = min(frame_w, frame_h)
        target_badge = int(max(36, min(80, short_side * 0.07)))  # 7% short-side

        # chỉ rebuild khi thay đổi đủ lớn (tránh rebuild liên tục)
        if self.badges is None or self.badge_size is None or abs(target_badge - self.badge_size) >= 4:
            self.badge_size = target_badge
            self.badges = load_icons_as_badges(
                icon_dir="assets/icons",
                badge_size=self.badge_size,
                pad=8,
                bg_alpha=160,
                try_remove_white_bg=True,
            )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        now = time.time()

        # overlay from cached results
        with self.lock:
            recognize = self.last_recognize
            selected = self.last_emotion

        if selected is not None:
            print(f"[DISPLAY] Drawing icon for: {selected}")

        if self.badges is not None and selected is not None:
            if selected not in self.badges:
                print("[BADGE] selected not in badges:", selected, "keys=", list(self.badges.keys()))

        vis = draw_overlays(img, recognize)

        # badges cached (không load mỗi frame)
        h, w = vis.shape[:2]
        self._ensure_badges(h, w)

        if self.badges is not None:
            vis = draw_emotion_row_on_frame(
                vis,
                badges=self.badges,
                selected=selected,  # None => all dim
                anchor="top-right",
                margin=12,
                gap=10,
                dim_opacity=0.18,
            )

        # kick inference thread
        if (now - self.last_infer_ts) >= self.infer_interval and not self.infer_running:
            self.infer_running = True
            self.last_infer_ts = now
            threading.Thread(target=self._infer_async, args=(img.copy(),), daemon=True).start()

        return av.VideoFrame.from_ndarray(vis, format="bgr24")

st.subheader("Camera")
webrtc_ctx = webrtc_streamer(
    key="cam",
    video_processor_factory=VideoProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {
            "width": {"exact": 1280},
            "height": {"exact": 720},
            "frameRate": {"ideal": 30, "max": 30},
        },
        "audio": False,
    },
    async_processing=True,
)
