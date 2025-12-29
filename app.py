import threading
import time

import av
import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer

from src.emotion_ui import load_icons_as_badges, draw_emotion_row_on_frame
from src.face_engine import FaceEngine, draw_overlays

st.set_page_config(page_title="Face Recognition Demo", layout="wide")


@st.cache_resource
def get_engine():
    # đổi buffalo_l nếu muốn “xịn” hơn
    return FaceEngine(
        model_name="buffalo_s",
        providers=["CPUExecutionProvider"],
        det_size=(640, 640),
        db_dir="db/embeddings",
        threshold=0.40,
        unknown_percent_cutoff=30.0,
    )


engine = get_engine()

st.title("Module 1 - Face Recognition (InsightFace)")

with st.sidebar:
    st.header("Enroll user")
    name = st.text_input("Tên user", value="")
    uploads = st.file_uploader("Upload 1+ ảnh (jpg/png)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if st.button("Enroll"):
        if not name.strip():
            st.error("Bạn chưa nhập tên.")
        elif not uploads:
            st.error("Bạn chưa upload ảnh.")
        else:
            imgs = []
            for f in uploads:
                data = np.frombuffer(f.read(), dtype=np.uint8)
                bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if bgr is not None:
                    imgs.append(bgr)

            try:
                emb = engine.enroll(name.strip(), imgs)
                st.success(f"Enroll OK: {name} (saved embedding dim={emb.shape[0]})")
            except Exception as e:
                st.error(f"Enroll failed: {e}")

    st.divider()
    st.header("Database")
    if st.button("Reload DB"):
        engine.reload_db()
        st.success("Reloaded embeddings.")

    st.write("Known users:", list(engine.known.keys()))


class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Kết quả overlay gần nhất
        self.last_frame = None
        self.last_recognize = []
        self.lock = threading.Lock()

        self.badges = load_icons_as_badges(
            icon_dir="assets/icons",
            badge_size=56,
            pad=8,
            bg_alpha=160,
            try_remove_white_bg=True,
        )

        self.last_emotion = None

        # Throttle inference (giây/lần) -> bạn chỉnh 0.3 ~ 0.6 cho Mac
        self.infer_interval = 0.4
        self.last_infer_ts = 0.0
        self.infer_running = False

    def get_last_emotion(self):
        with self.lock:
            return self.last_emotion

    def _infer_async(self, img_bgr):
        try:
            recognizes = engine.recognize(img_bgr, max_faces=2)  # giảm max_faces cho nhẹ

            emotion = None

            with self.lock:
                self.last_recognize = recognizes
                self.last_emotion = emotion
        finally:
            self.infer_running = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        now = time.time()

        # Luôn vẽ overlay từ kết quả gần nhất (nhanh)
        with self.lock:
            recognize = self.last_recognize
            selected = self.last_emotion
        vis = draw_overlays(img, recognize)

        # draw emotion icons directly on video frame
        vis = draw_emotion_row_on_frame(
            vis,
            badges=self.badges,
            selected=selected,  # None => all dim
            anchor="top-right",
            margin=12,
            gap=10,
            dim_opacity=0.18,
        )

        # Nếu đến thời điểm thì kick inference sang thread (không block recv)
        if (now - self.last_infer_ts) >= self.infer_interval and not self.infer_running:
            self.infer_running = True
            self.last_infer_ts = now
            threading.Thread(target=self._infer_async, args=(img.copy(),), daemon=True).start()

        return av.VideoFrame.from_ndarray(vis, format="bgr24")


st.subheader("Camera")
webrtc_ctx = webrtc_streamer(
    key="cam",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 1280},
            "height": {"ideal": 720},
            "frameRate": {"ideal": 30, "max": 30},
        },
        "audio": False,
    },
    async_processing=True,
)
