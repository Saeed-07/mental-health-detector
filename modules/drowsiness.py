"""
Eye-Aspect-Ratio (EAR) based drowsiness and blink detection.

EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
(Soukupova & Cech, 2016 — the standard low-cost drowsiness heuristic:
EAR stays roughly constant while the eye is open and drops sharply when
it closes.)
"""
import time
from collections import deque

import numpy as np

from .face_mesh_detector import LEFT_EYE, RIGHT_EYE


def _eye_aspect_ratio(eye_pts):
    p1, p2, p3, p4, p5, p6 = eye_pts
    vertical_1 = np.linalg.norm(p2 - p6)
    vertical_2 = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class DrowsinessMonitor:
    def __init__(self, ear_threshold=0.21, closed_frames_for_alert=15, history_seconds=60):
        """
        ear_threshold:         EAR below this counts as "eyes closed"
        closed_frames_for_alert: consecutive closed frames before raising
                                  a drowsiness alert (~0.5-1s at typical
                                  webcam frame rates)
        history_seconds:       how much EAR history to keep for the
                                  rolling "percent time eyes closed" stat
        """
        self.ear_threshold = ear_threshold
        self.closed_frames_for_alert = closed_frames_for_alert
        self.history_seconds = history_seconds

        self._consecutive_closed = 0
        self._was_closed = False
        self.blink_count = 0
        self.ear_history = deque()  # (timestamp, ear)
        self.drowsy_alert = False

    def update(self, landmarks_px):
        left = landmarks_px[LEFT_EYE]
        right = landmarks_px[RIGHT_EYE]
        ear = (_eye_aspect_ratio(left) + _eye_aspect_ratio(right)) / 2.0

        now = time.time()
        self.ear_history.append((now, ear))
        cutoff = now - self.history_seconds
        while self.ear_history and self.ear_history[0][0] < cutoff:
            self.ear_history.popleft()

        is_closed = ear < self.ear_threshold
        if is_closed:
            self._consecutive_closed += 1
        else:
            # A short closure that ends before the drowsiness threshold
            # counts as a blink rather than a drowsy episode.
            if self._was_closed and 2 <= self._consecutive_closed < self.closed_frames_for_alert:
                self.blink_count += 1
            self._consecutive_closed = 0
        self._was_closed = is_closed

        self.drowsy_alert = self._consecutive_closed >= self.closed_frames_for_alert

        if self.ear_history:
            closed_ratio = sum(1 for _, e in self.ear_history if e < self.ear_threshold) / len(
                self.ear_history
            )
        else:
            closed_ratio = 0.0

        return {
            "ear": ear,
            "is_drowsy": self.drowsy_alert,
            "blink_count": self.blink_count,
            "eye_closed_ratio": closed_ratio,
        }
