from __future__ import annotations

import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.recognition import ArtworkRecognizer, RecognitionError

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "6000000"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI(title="RijksLens Recognition Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

recognizer: ArtworkRecognizer | None = None


class HealthResponse(BaseModel):
    ok: bool
    referenceCount: int
    detector: str


@app.on_event("startup")
def startup() -> None:
    global recognizer
    recognizer = ArtworkRecognizer()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if recognizer is None:
        return HealthResponse(ok=False, referenceCount=0, detector="not-loaded")

    detector_name = type(recognizer.detector).__name__

    return HealthResponse(
        ok=True,
        referenceCount=recognizer.reference_count,
        detector=detector_name,
    )


@app.post("/reload")
def reload_references() -> dict:
    if recognizer is None:
        raise HTTPException(status_code=500, detail="Recognizer is not initialized.")

    recognizer.load_references()

    return {
        "ok": True,
        "referenceCount": recognizer.reference_count,
    }


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)) -> dict:
    if recognizer is None:
        raise HTTPException(status_code=500, detail="Recognizer is not initialized.")

    image_bytes = await file.read()

    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image is too large. Upload a smaller photo.",
        )

    try:
        return recognizer.recognize_bytes(image_bytes)
    except RecognitionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {exc}") from exc
