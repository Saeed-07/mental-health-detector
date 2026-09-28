"""
Facial emotion recognition with two interchangeable backends:

  "fer"       - the `fer` package (lighter, faster)
  "deepface"  - the `deepface` package (heavier, several model choices,
                downloads pretrained weights on first use — needs internet
                the first time it runs)

Both return the same normalized {emotion: probability} dict so the rest
of the app doesn't need to know which one is active.
"""

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Weights used to collapse an emotion-probability distribution into a
# single 0-100 "mood score". Tune these to taste for your rubric.
MOOD_WEIGHTS = {
    "happy": 1.0,
    "surprise": 0.4,
    "neutral": 0.1,
    "sad": -0.8,
    "fear": -0.7,
    "angry": -1.0,
    "disgust": -0.9,
}


def mood_score_from_emotions(emotion_probs):
    """Map an emotion probability dict to a 0-100 mood score."""
    if not emotion_probs:
        return None
    raw = sum(MOOD_WEIGHTS.get(e, 0) * p for e, p in emotion_probs.items())
    # raw is roughly in [-1, 1] given the weights above; rescale to [0, 100]
    score = (raw + 1.0) / 2.0 * 100.0
    return max(0.0, min(100.0, score))


class EmotionDetector:
    def __init__(self, backend="fer"):
        self.backend = backend
        if backend == "fer":
            from fer import FER

            # mtcnn=False -> use OpenCV's Haar cascade for the internal
            # face check (faster; we already crop to a detected face
            # ourselves before calling this).
            self._detector = FER(mtcnn=False)
        elif backend == "deepface":
            # DeepFace loads its models lazily on first .analyze() call.
            self._detector = None
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

    def detect(self, bgr_frame):
        """Returns (dominant_emotion, emotion_probs_dict), or (None, None)
        if no face/emotion could be read from the given crop."""
        if self.backend == "fer":
            results = self._detector.detect_emotions(bgr_frame)
            if not results:
                return None, None
            probs = results[0]["emotions"]
            dominant = max(probs, key=probs.get)
            return dominant, probs

        # backend == "deepface"
        from deepface import DeepFace

        try:
            analysis = DeepFace.analyze(
                bgr_frame, actions=["emotion"], enforce_detection=False, silent=True
            )
        except Exception:
            return None, None
        result = analysis[0] if isinstance(analysis, list) else analysis
        raw_probs = result.get("emotion", {})
        total = sum(raw_probs.values()) or 1.0
        probs = {k.lower(): v / total for k, v in raw_probs.items()}
        dominant = result.get("dominant_emotion", max(probs, key=probs.get) if probs else None)
        return dominant, probs
