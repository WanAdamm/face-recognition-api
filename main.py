import os
import cv2
import face_recognition
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from typing import Annotated

# =========================
# FastAPI setup
# =========================
app = FastAPI(title="Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Load known faces ONCE
# =========================
print("Loading known faces...")


known_face_encodings = []
known_face_names = []
known_face_empID = []

FACE_DIR = "known_faces"

for file in os.listdir(FACE_DIR):
    if not file.lower().endswith((".jpg", ".png")):
        continue

    stem = os.path.splitext(file)[0]  #
    
    parts = stem.rsplit(" ", 1)
    
    if len(parts) != 2 or not parts[1].isdigit():
        raise ValueError(f"Invalid filename format: {file}")
    
    name = parts[0]
    emp_id = parts[1]

    image = face_recognition.load_image_file(os.path.join(FACE_DIR, file))
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_face_encodings.append(encodings[0])
        known_face_names.append(name)
        known_face_empID.append(emp_id)

print("Known faces loaded.")

# =========================
# Routes
# =========================

@app.get("/")
def root():
    return {
        "message": "Face Recognition API",
        "endpoint": "/recognize",
        "method": "POST",
        "content_type": "multipart/form-data"
    }


@app.post("/recognize")

async def recognize_face(file: Annotated[UploadFile, File(...)]):
    contents = await file.read()

    # Decode image
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid image"}
        )

    rgb_frame = frame[:, :, ::-1]
    # Resize for stability and performance
    height, width = rgb_frame.shape[:2]

    MAX_WIDTH = 800
    if width > MAX_WIDTH:
        scale = MAX_WIDTH / width
        rgb_frame = cv2.resize(
            rgb_frame,
            (int(width * scale), int(height * scale))
        )

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_frame, face_locations
    )

    results = []

    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):
        name = "Unknown"
        emp_id = "Unknown"

        if known_face_encodings:
            distances = face_recognition.face_distance(
                known_face_encodings, face_encoding
            )
            best_match = np.argmin(distances)

            if distances[best_match] < 0.6:
                name = known_face_names[best_match]
                emp_id = known_face_empID[best_match]

        results.append({
            "name": name,
            "empID": emp_id,
            "box": {
                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left
            }
        })

    return {
        "faces_detected": len(results),
        "results": results
    }