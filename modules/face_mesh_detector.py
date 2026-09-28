"""
Face & landmark detection, powered by MediaPipe Face Mesh.

MediaPipe returns 468 (or 478, with iris refinement) 3D landmarks per face.
That's denser than the classic dlib 68-point model and — importantly for a
web app — needs no extra ~100MB model file to download, so it's a better
fit here. The indices below map onto the same eye-contour points the
classic 68-point EAR method uses, so the drowsiness math is unchanged.

If your assignment specifically requires the dlib 68-point predictor,
swap this module for a dlib.get_frontal_face_detector() +
dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") pair — the
rest of the pipeline (drowsiness.py, emotion_detector.py) only needs the
six eye-contour points per eye, so it will keep working with either
landmark source as long as you update LEFT_EYE / RIGHT_EYE accordingly.
"""
import cv2
import mediapipe as mp
import numpy as np

# Eye-contour landmark indices (MediaPipe's 468-point mesh) used for EAR.
# Order per eye: [outer_corner, top_1, top_2, inner_corner, bottom_1, bottom_2]
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


class FaceMeshDetector:
    def __init__(
        self,
        max_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self._mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def process(self, bgr_frame):
        """Run detection on a BGR frame.

        Returns (landmarks_px, mediapipe_results):
          landmarks_px is an (N, 2) float array of pixel coordinates, or
          None if no face was found.
        """
        h, w = bgr_frame.shape[:2]
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None, results
        landmarks = results.multi_face_landmarks[0]
        pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark])
        return pts, results

    def draw(self, frame, results):
        """Overlay the face mesh contours on `frame` (in place, returns it)."""
        if not results.multi_face_landmarks:
            return frame
        for face_landmarks in results.multi_face_landmarks:
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self._mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_styles.get_default_face_mesh_contours_style(),
            )
        return frame

    @staticmethod
    def bounding_box(landmarks_px, frame_shape, margin=20):
        """Pixel bounding box around the face, for cropping to the emotion model."""
        h, w = frame_shape[:2]
        x_min, y_min = landmarks_px.min(axis=0)
        x_max, y_max = landmarks_px.max(axis=0)
        x1 = max(int(x_min) - margin, 0)
        y1 = max(int(y_min) - margin, 0)
        x2 = min(int(x_max) + margin, w)
        y2 = min(int(y_max) + margin, h)
        return x1, y1, x2, y2

    def close(self):
        self.face_mesh.close()
