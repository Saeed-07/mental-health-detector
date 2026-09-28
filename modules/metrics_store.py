"""
Thread-safe rolling store for real-time metrics.

streamlit-webrtc runs video processing (face detection, EAR, emotion) on
its own background thread, separate from the thread that renders the
Streamlit dashboard. This object is the hand-off point between the two:
the video thread pushes new readings in, the dashboard loop polls
snapshots out.
"""
import threading
import time
from collections import deque


class MetricsStore:
    def __init__(self, max_len=600):
        self._lock = threading.Lock()
        self.timestamps = deque(maxlen=max_len)
        self.mood_scores = deque(maxlen=max_len)
        self.emotions = deque(maxlen=max_len)
        self.ear_values = deque(maxlen=max_len)
        self.blink_count = 0
        self.drowsy_alert = False
        self.last_emotion_probs = {}
        self.session_start = time.time()

    def push(
        self,
        mood_score=None,
        emotion=None,
        ear=None,
        blink_count=None,
        drowsy_alert=None,
        emotion_probs=None,
    ):
        with self._lock:
            now = time.time()
            if mood_score is not None:
                self.timestamps.append(now)
                self.mood_scores.append(mood_score)
            if emotion is not None:
                self.emotions.append(emotion)
            if ear is not None:
                self.ear_values.append(ear)
            if blink_count is not None:
                self.blink_count = blink_count
            if drowsy_alert is not None:
                self.drowsy_alert = drowsy_alert
            if emotion_probs is not None:
                self.last_emotion_probs = emotion_probs

    def snapshot(self):
        with self._lock:
            return {
                "timestamps": list(self.timestamps),
                "mood_scores": list(self.mood_scores),
                "emotions": list(self.emotions),
                "ear_values": list(self.ear_values),
                "blink_count": self.blink_count,
                "drowsy_alert": self.drowsy_alert,
                "last_emotion_probs": dict(self.last_emotion_probs),
                "session_start": self.session_start,
            }
