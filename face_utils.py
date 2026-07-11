"""
OpenCV helper functions.
Detection: OpenCV DNN SSD face detector (res10_300x300_ssd_iter_140000)
Embedding: OpenCV DNN + OpenFace model (nn4.small2.v1.t7) -> 128-d vector
Liveness: Haar-cascade eye detection + eye-aspect-ratio across frames
Model files go in /models — see models/README.md for download links.
"""

import os
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- Face detector (Caffe SSD) ---
PROTOTXT = os.path.join(MODELS_DIR, "deploy.prototxt")
DETECTOR_WEIGHTS = os.path.join(MODELS_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
face_net = cv2.dnn.readNetFromCaffe(PROTOTXT, DETECTOR_WEIGHTS)

# --- Embedding model (Torch, OpenFace) ---
EMBEDDER_WEIGHTS = os.path.join(MODELS_DIR, "nn4.small2.v1.t7")
embedder_net = cv2.dnn.readNetFromTorch(EMBEDDER_WEIGHTS)

# --- Eye detector for liveness ---
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


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
    (x1, y1, x2, y2) = face_box
    face_gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6)
    return eyes


def eye_aspect_ratio(eyes) -> float:
    """
    Rough proxy for eye-openness using Haar eye-box height/width ratio,
    averaged across detected eyes. Lower value ~= more closed.
    """
    if len(eyes) == 0:
        return 0.3  # treat as "closed / not visible"
    ratios = [h / w for (_, _, w, h) in eyes if w > 0]
    return float(np.mean(ratios)) if ratios else 0.3
