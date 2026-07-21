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
from datetime import date as date_cls, datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))
LATE_GRACE_MINUTES = 5  # how many minutes after class end_time marking is still allowed
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
    detect_eyes,
    get_real_ear,
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

    # --- Liveness: real eye-aspect-ratio (EAR) via MediaPipe face-mesh
    # landmarks. A genuine blink causes this value to actually dip, unlike
    # the earlier Haar-cascade box-shape approach. ---
    last_face_img = None
    last_face_box = None
    frames_with_face = 0
    ear_values = []

    for idx, b64 in enumerate(req.frames_base64):
        img = b64_to_cv2_image(b64)
        face_box = detect_face(img)
        if face_box is None:
            print(f"[verify-debug] frame {idx}: no face detected")
            continue
        frames_with_face += 1
        last_face_img, last_face_box = img, face_box
        ear = get_real_ear(img)
        if ear is not None:
            ear_values.append(ear)
            print(f"[verify-debug] frame {idx}: face OK, ear={ear:.4f}")
        else:
            print(f"[verify-debug] frame {idx}: face OK, no landmarks/ear")

    print(f"[verify-debug] TOTAL frames_with_face={frames_with_face} ear_values={ear_values}")

    if last_face_img is None:
        raise HTTPException(status_code=400, detail="No face detected in the captured frames")

    is_live = False
    if len(ear_values) >= 3:
        min_ear = min(ear_values)
        max_ear = max(ear_values)
        # Self-calibrating check: instead of one fixed number for every
        # person (which breaks down for people with naturally
        # smaller/larger eye-openness ranges), we measure each person's
        # blink RELATIVE TO THEIR OWN open-eye baseline (max_ear) captured
        # in this same burst. A real blink should dip at least ~25% below
        # that person's own baseline. We also sanity-check that the
        # baseline itself looks like a genuinely open eye (not a face
        # turned away or a bad-quality frame throughout).
        baseline_is_plausible = max_ear > 0.15
        relative_dip = (max_ear - min_ear) / max_ear if max_ear > 0 else 0
        is_live = baseline_is_plausible and relative_dip > 0.25
        print(f"[verify-debug] min_ear={min_ear:.4f} max_ear={max_ear:.4f} relative_dip={relative_dip:.3f} baseline_ok={baseline_is_plausible} is_live={is_live}")
    else:
        print(f"[verify-debug] not enough ear_values to evaluate liveness (need >=3, got {len(ear_values)})")

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
        raw_embedding = row["embedding"]
        if isinstance(raw_embedding, str):
            # Supabase/PostgREST can return pgvector columns as a text
            # representation like "[0.1,0.2,...]" rather than a real list —
            # parse it back into actual numbers before doing any math.
            raw_embedding = raw_embedding.strip("[]")
            stored_embedding = np.array([float(x) for x in raw_embedding.split(",")])
        else:
            stored_embedding = np.array(raw_embedding)
        score = cosine_similarity(live_embedding, stored_embedding)
        best_score = max(best_score, score)

    matched = best_score >= MATCH_THRESHOLD

    if not matched:
        return {
            "matched": False,
            "confidence": round(float(best_score), 4),
            "student_id": req.student_id,
        }

    # --- Enforce that marking only happens during the class window (+grace),
    # using the institute's local time (IST), not the server's UTC clock ---
    timetable_result = (
        supabase.table("timetable")
        .select("start_time, end_time, institute_id, department, batch_id, section")
        .eq("id", req.timetable_id)
        .single()
        .execute()
    )
    if not timetable_result.data:
        raise HTTPException(status_code=404, detail="Class session not found")

    now_ist = datetime.now(IST)
    today_ist = now_ist.date()

    # --- Enforce that today falls within this section's currently
    # configured semester dates — prevents marking before a semester has
    # started or after it has ended, until an admin sets up the next one ---
    tt = timetable_result.data
    section_result = (
        supabase.table("sections")
        .select("semester_start_date, semester_end_date")
        .eq("institute_id", tt["institute_id"])
        .eq("department", tt["department"])
        .eq("batch_id", tt["batch_id"])
        .eq("name", tt["section"])
        .single()
        .execute()
    )
    if not section_result.data:
        raise HTTPException(status_code=404, detail="Section configuration not found")

    sem_start = section_result.data.get("semester_start_date")
    sem_end = section_result.data.get("semester_end_date")

    if sem_start and today_ist < datetime.strptime(sem_start, "%Y-%m-%d").date():
        raise HTTPException(
            status_code=400,
            detail="This semester hasn't started yet — attendance isn't open",
        )
    if sem_end and today_ist > datetime.strptime(sem_end, "%Y-%m-%d").date():
        raise HTTPException(
            status_code=400,
            detail="This semester has ended — attendance is closed until the next semester's timetable is set up",
        )

    start_time_str = tt["start_time"]  # e.g. "09:00:00"
    end_time_str = tt["end_time"]        # e.g. "10:00:00"
    start_dt = datetime.combine(today_ist, datetime.strptime(start_time_str, "%H:%M:%S").time(), tzinfo=IST)
    end_dt = datetime.combine(today_ist, datetime.strptime(end_time_str, "%H:%M:%S").time(), tzinfo=IST)
    grace_end_dt = end_dt + timedelta(minutes=LATE_GRACE_MINUTES)

    if now_ist < start_dt:
        raise HTTPException(
            status_code=400,
            detail=f"This class hasn't started yet — attendance opens at {start_time_str[:5]} IST",
        )
    if now_ist > grace_end_dt:
        raise HTTPException(
            status_code=400,
            detail=f"Attendance window has closed — this class ended at {end_time_str[:5]} IST",
        )

    # --- Real match confirmed — write the attendance row ourselves (server-authority) ---
    try:
        supabase.table("attendance").insert({
            "student_id": req.student_id,
            "timetable_id": req.timetable_id,
            "date": today_ist.isoformat(),
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
