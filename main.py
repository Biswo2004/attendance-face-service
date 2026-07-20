"""
Face Recognition Microservice — pure OpenCV
Endpoints:
  POST /enroll  -> takes student_id + 2-3 images, stores averaged embedding in Supabase
  POST /verify  -> takes student_id + timetable_id + a short burst of live frames,
                    runs liveness (blink check) + face match against stored embedding,
                    and — on a real match — writes the attendance row itself
  GET  /health  -> simple health check for Render/HF Spaces
"""

import os
import io
import base64
from datetime import date as date_cls
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

from face_utils import (
    detect_face,
    get_embedding,
    cosine_similarity,
    eye_aspect_ratio,
    detect_eyes,
)

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
MATCH_THRESHOLD = float(os.environ.get("MATCH_THRESHOLD", "0.75"))

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_URL else None

app = FastAPI(title="Attendance Face Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your Vercel/Netlify domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class EnrollRequest(BaseModel):
    student_id: str
    images_base64: List[str]  # 2-3 signup photos


class VerifyRequest(BaseModel):
    student_id: str
    timetable_id: str          # which class session this scan is for
    frames_base64: List[str]   # short burst (5-10 frames) captured from webcam


def b64_to_cv2_image(b64_string: str) -> np.ndarray:
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")
    return img


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/enroll")
def enroll(req: EnrollRequest):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    if len(req.images_base64) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 photos")

    embeddings = []
    for b64 in req.images_base64:
        img = b64_to_cv2_image(b64)
        face_box = detect_face(img)
        if face_box is None:
            raise HTTPException(status_code=400, detail="No face detected in one of the photos")
        embedding = get_embedding(img, face_box)
        embeddings.append(embedding)

    # average embedding across the enrollment photos for robustness
    avg_embedding = np.mean(embeddings, axis=0)
    avg_embedding = (avg_embedding / np.linalg.norm(avg_embedding)).tolist()

    supabase.table("face_embeddings").insert(
        {"student_id": req.student_id, "embedding": avg_embedding}
    ).execute()

    return {"status": "enrolled", "student_id": req.student_id}


@app.post("/verify")
def verify(req: VerifyRequest, authorization: str = Header(None)):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    # --- Confirm the caller is really logged in as this student ---
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing login session")
    token = authorization.split(" ", 1)[1]
    try:
        user_resp = supabase.auth.get_user(token)
        caller_id = user_resp.user.id
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired login session")
    if caller_id != req.student_id:
        raise HTTPException(status_code=403, detail="Login session does not match student_id")

    if len(req.frames_base64) < 3:
        raise HTTPException(status_code=400, detail="Provide at least 3 frames for liveness check")

    # --- Liveness: look for at least one blink across the frame burst ---
    ear_values = []
    last_face_img = None
    last_face_box = None

    for b64 in req.frames_base64:
        img = b64_to_cv2_image(b64)
        face_box = detect_face(img)
        if face_box is None:
            continue
        last_face_img, last_face_box = img, face_box
        eyes = detect_eyes(img, face_box)
        if eyes is not None and len(eyes) > 0:
            ear_values.append(eye_aspect_ratio(eyes))

    if last_face_img is None:
        raise HTTPException(status_code=400, detail="No face detected in the captured frames")

    is_live = False
    if len(ear_values) >= 3:
        # a blink shows up as a dip in eye-aspect-ratio partway through the burst
        min_ear = min(ear_values)
        max_ear = max(ear_values)
        is_live = (max_ear - min_ear) > 0.08  # empirical threshold, tune after testing

    if not is_live:
        raise HTTPException(
            status_code=400,
            detail="Liveness check failed — please blink naturally during capture",
        )

    # --- Recognition: compare against stored embedding(s) ---
    live_embedding = get_embedding(last_face_img, last_face_box)

    result = (
        supabase.table("face_embeddings")
        .select("embedding")
        .eq("student_id", req.student_id)
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=404, detail="No enrolled face found for this student")

    best_score = -1.0
    for row in result.data:
        stored_embedding = np.array(row["embedding"])
        score = cosine_similarity(live_embedding, stored_embedding)
        best_score = max(best_score, score)

    matched = best_score >= MATCH_THRESHOLD

    if not matched:
        return {
            "matched": False,
            "confidence": round(float(best_score), 4),
            "student_id": req.student_id,
        }

    # --- Real match confirmed — write the attendance row ourselves (server-authority) ---
    try:
        supabase.table("attendance").insert({
            "student_id": req.student_id,
            "timetable_id": req.timetable_id,
            "date": date_cls.today().isoformat(),
            "status": "present",
            "confidence_score": round(float(best_score), 4),
        }).execute()
        return {
            "matched": True,
            "confidence": round(float(best_score), 4),
            "student_id": req.student_id,
            "already_marked": False,
        }
    except Exception as e:
        if "23505" in str(e) or "duplicate key" in str(e).lower():
            return {
                "matched": True,
                "confidence": round(float(best_score), 4),
                "student_id": req.student_id,
                "already_marked": True,
            }
        raise HTTPException(status_code=500, detail=f"Verified but failed to record attendance: {e}")
