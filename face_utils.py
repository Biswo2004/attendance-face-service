"""
OpenCV helper functions.
Detection: OpenCV DNN SSD face detector (res10_300x300_ssd_iter_140000)
Embedding: OpenCV DNN + OpenFace model (nn4.small2.v1.t7) -> 128-d vector
Liveness: MediaPipe Face Mesh landmarks + real eye-aspect-ratio (EAR) across frames
Model files go in /models — see models/README.md for download links.
"""
import os
import cv2
import numpy as np
import mediapipe as mp

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- Face detector (Caffe SSD) ---
PROTOTXT = os.path.join(MODELS_DIR, "deploy.prototxt")
DETECTOR_WEIGHTS = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe(PROTOTXT, DETECTOR_WEIGHTS)

# --- Embedding model (Torch, OpenFace) ---
EMBEDDER_WEIGHTS = os.path.join(MODELS_DIR, "nn4.small2.v1.t7")
embedder_net = cv2.dnn.readNetFromTorch(EMBEDDER_WEIGHTS)

# --- Eye detector for liveness (legacy Haar cascade — kept but no longer
# used by /verify; superseded by MediaPipe-based get_real_ear below) ---
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# --- MediaPipe Face Mesh for real eye-aspect-ratio (EAR) liveness check ---
# static_image_mode=True because we're checking independent frames, not a
# continuous video stream.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)

# Standard MediaPipe Face Mesh landmark indices for each eye's 6 key points
# (corner, top, top, corner, bottom, bottom) — same convention as the classic
# dlib 6-point eye-aspect-ratio (EAR) formula.
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]


def detect_face(img: np.ndarray, conf_threshold: float = 0.6):
    """Returns (x1, y1, x2, y2) box for the most confident face, or None."""
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    face_net.setInput(blob)
    detections = face_net.forward()
    best_box, best_conf = None, 0.0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold and confidence > best_conf:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            best_box, best_conf = (x1, y1, x2, y2), confidence
    return best_box


def get_embedding(img: np.ndarray, face_box) -> np.ndarray:
    (x1, y1, x2, y2) = face_box
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        raise ValueError("Empty face crop")
    face_blob = cv2.dnn.blobFromImage(
        face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False
    )
    embedder_net.setInput(face_blob)
    vec = embedder_net.forward()
    vec = vec.flatten()
    return vec / np.linalg.norm(vec)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.dot(a, b))


def detect_eyes(img: np.ndarray, face_box):
    """Legacy Haar-cascade eye detector — kept for reference, no longer
    used by /verify's liveness check (see get_real_ear instead)."""
    (x1, y1, x2, y2) = face_box
    face_gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6)
    return eyes


def eye_aspect_ratio(eyes) -> float:
    """
    Legacy Haar-cascade box-shape proxy — kept for reference only.
    NOT used by /verify anymore: Haar eye boxes don't shrink as an eye
    closes, so this always returns ~1.0 regardless of blinking. Superseded
    by get_real_ear(), which uses actual facial landmark geometry instead.
    """
    if len(eyes) == 0:
        return 0.3
    ratios = [h / w for (_, _, w, h) in eyes if w > 0]
    return float(np.mean(ratios)) if ratios else 0.3


def _euclidean(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def _ear_from_points(pts):
    """Classic EAR formula: (vertical1 + vertical2) / (2 * horizontal)."""
    p1, p2, p3, p4, p5, p6 = pts
    vertical1 = _euclidean(p2, p6)
    vertical2 = _euclidean(p3, p5)
    horizontal = _euclidean(p1, p4)
    if horizontal == 0:
        return 0.3
    return (vertical1 + vertical2) / (2.0 * horizontal)


def get_real_ear(img: np.ndarray):
    """
    Returns the average real eye-aspect-ratio across both eyes using
    MediaPipe face-mesh landmarks, or None if no face/landmarks found.
    A real blink causes this number to genuinely dip (unlike the old
    Haar-cascade box-shape approach, which never changed).
    """
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark

    def pts_for(idxs):
        return [(landmarks[i].x * w, landmarks[i].y * h) for i in idxs]

    left_ear = _ear_from_points(pts_for(LEFT_EYE_IDX))
    right_ear = _ear_from_points(pts_for(RIGHT_EYE_IDX))
    return (left_ear + right_ear) / 2.0
