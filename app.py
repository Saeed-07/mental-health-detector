"""
Mental Health Tracker — real-time face detection, drowsiness monitoring
and emotion recognition, served as a Streamlit web app.

Pipeline:
    1. Webcam feed input          -> streamlit-webrtc (browser -> server)
    2. Face & landmark detection  -> MediaPipe Face Mesh
    3. Feature extraction         -> EAR-based drowsiness, FER/DeepFace emotion
    4. Dashboard & analytics      -> live Streamlit charts + mood score

Educational project only — not a medical or diagnostic tool.
"""
import time

import av
import cv2
import pandas as pd
import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

from modules.drowsiness import DrowsinessMonitor
from modules.emotion_detector import EmotionDetector, mood_score_from_emotions
from modules.face_mesh_detector import FaceMeshDetector
from modules.metrics_store import MetricsStore

st.set_page_config(page_title="Mental Health Tracker", layout="wide", page_icon="🧠")

# STUN server so the browser<->server webcam connection can traverse NAT —
# needed once this is deployed off localhost (e.g. Streamlit Community Cloud).
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

if "metrics_store" not in st.session_state:
    st.session_state.metrics_store = MetricsStore()
METRICS = st.session_state.metrics_store

st.title("🧠 Real-Time Mental Health Tracker")
st.caption(
    "Face-landmark detection, drowsiness monitoring and emotion recognition "
    "from your webcam, live in the browser."
)

with st.sidebar:
    st.header("Settings")
    backend = st.selectbox(
        "Emotion model",
        ["fer", "deepface"],
        index=0,
        help="Pick this before pressing START — switching backends mid-session "
        "requires stopping and restarting the stream.",
    )
    ear_threshold = st.slider("Eye-closed threshold (EAR)", 0.15, 0.30, 0.21, 0.01)
    emotion_every_n = st.slider("Run emotion model every N frames", 1, 15, 5)
    st.markdown("---")
    st.caption(
        "⚠️ Video is processed only for this session; nothing is stored "
        "unless you use the CSV download button. Not a clinical or "
        "diagnostic tool."
    )


class VideoProcessor(VideoProcessorBase):
    def __init__(self, backend="fer", ear_threshold=0.21, emotion_every_n=5):
        self.face_detector = FaceMeshDetector()
        self.drowsiness = DrowsinessMonitor(ear_threshold=ear_threshold)
        self.emotion_detector = EmotionDetector(backend=backend)
        self.emotion_every_n = emotion_every_n
        self.frame_idx = 0
        self.last_dominant = None
        self.last_probs = {}

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_idx += 1

        landmarks_px, results = self.face_detector.process(img)
        img = self.face_detector.draw(img, results)

        if landmarks_px is not None:
            d = self.drowsiness.update(landmarks_px)

            x1, y1, x2, y2 = self.face_detector.bounding_box(landmarks_px, img.shape)
            face_crop = img[y1:y2, x1:x2]
            if self.frame_idx % self.emotion_every_n == 0 and face_crop.size > 0:
                dominant, probs = self.emotion_detector.detect(face_crop)
                if probs is not None:
                    self.last_dominant, self.last_probs = dominant, probs

            mood = mood_score_from_emotions(self.last_probs) if self.last_probs else None

            METRICS.push(
                mood_score=mood,
                emotion=self.last_dominant,
                ear=d["ear"],
                blink_count=d["blink_count"],
                drowsy_alert=d["is_drowsy"],
                emotion_probs=self.last_probs,
            )

            label = self.last_dominant or "detecting..."
            color = (0, 0, 255) if d["is_drowsy"] else (0, 200, 0)
            cv2.putText(img, f"Emotion: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(
                img, f"EAR: {d['ear']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
            )
            if d["is_drowsy"]:
                cv2.putText(
                    img,
                    "DROWSINESS ALERT",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
        else:
            cv2.putText(
                img, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def make_processor():
    return VideoProcessor(backend=backend, ear_threshold=ear_threshold, emotion_every_n=emotion_every_n)


col_video, col_dash = st.columns([3, 2])

with col_video:
    ctx = webrtc_streamer(
        key="mental-health-tracker",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=make_processor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# Live-tunable settings: push slider changes into the already-running processor.
if ctx.video_processor:
    ctx.video_processor.drowsiness.ear_threshold = ear_threshold
    ctx.video_processor.emotion_every_n = emotion_every_n

with col_dash:
    st.subheader("Live analytics")
    m1, m2, m3 = st.columns(3)
    mood_metric = m1.empty()
    blink_metric = m2.empty()
    alert_metric = m3.empty()
    mood_chart = st.empty()
    emotion_chart = st.empty()

download_slot = st.empty()

if ctx.state.playing:
    while ctx.state.playing:
        snap = METRICS.snapshot()

        mood_metric.metric(
            "Mood score",
            f"{snap['mood_scores'][-1]:.0f}/100" if snap["mood_scores"] else "—",
        )
        blink_metric.metric("Blinks", snap["blink_count"])
        alert_metric.metric("Drowsiness", "⚠️ Alert" if snap["drowsy_alert"] else "OK")

        if snap["mood_scores"]:
            df = pd.DataFrame({"time": snap["timestamps"], "mood": snap["mood_scores"]})
            df["elapsed_s"] = (df["time"] - snap["session_start"]).round(1)
            mood_chart.line_chart(df.set_index("elapsed_s")["mood"])

            csv = df[["time", "mood"]].to_csv(index=False)
            download_slot.download_button(
                "Download session data (CSV)",
                csv,
                file_name="mental_health_session.csv",
                mime="text/csv",
            )

        if snap["last_emotion_probs"]:
            probs_df = pd.DataFrame({"probability": snap["last_emotion_probs"]})
            emotion_chart.bar_chart(probs_df)

        time.sleep(0.5)
else:
    st.info("Click **START** above and allow camera access to begin.")
