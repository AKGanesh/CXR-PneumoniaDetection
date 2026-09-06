"""FastAPI REST API for chest X-ray pneumonia detection."""

from __future__ import annotations

import base64
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from cxr.gradcam import make_gradcam_heatmap, overlay_heatmap
from cxr.predict import load_inference_bundle, predict_bytes

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("CXR_MODELS_DIR", ROOT / "models"))

_models: list[Any] | None = None
_metadata: dict | None = None


class PredictionResponse(BaseModel):
    label: str = Field(description="Predicted class: healthy or pneumonia")
    probability: float = Field(description="Raw pneumonia probability")
    confidence: float = Field(description="Confidence in the predicted label")
    threshold: float = Field(description="Decision threshold used for classification")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    ensemble_members: list[str] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    ensemble_members: list[str]
    member_weights: list[float]
    threshold: float
    use_clahe: bool
    image_size: list[int]


class GradCamResponse(PredictionResponse):
    gradcam_overlay_base64: str = Field(description="Base64-encoded PNG of Grad-CAM overlay")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _models, _metadata
    try:
        _models, _metadata = load_inference_bundle(MODELS_DIR)
    except FileNotFoundError:
        _models = None
        _metadata = None
    yield


app = FastAPI(
    title="CXR Pneumonia Detection API",
    description="REST API for pneumonia detection from chest X-ray images.",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_models() -> tuple[list[Any], dict]:
    if _models is None or _metadata is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"No trained ensemble found in {MODELS_DIR}. "
                "Run scripts/train.py before starting the API."
            ),
        )
    return _models, _metadata


def _validate_upload(file: UploadFile) -> None:
    if file.content_type not in {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "application/octet-stream",
    }:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Upload a PNG or JPEG image.",
        )


def _encode_png(image: np.ndarray) -> str:
    success, encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode Grad-CAM image.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_models is not None,
        ensemble_members=_metadata.get("members", []) if _metadata else [],
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    _, metadata = _require_models()
    return ModelInfoResponse(
        ensemble_members=metadata.get("members", []),
        member_weights=metadata.get("member_weights", []),
        threshold=float(metadata.get("threshold", 0.5)),
        use_clahe=bool(metadata.get("use_clahe", True)),
        image_size=list(metadata.get("image_size", [224, 224])),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)) -> PredictionResponse:
    """Classify an uploaded chest X-ray as healthy or pneumonia."""
    _require_models()
    _validate_upload(file)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result, _ = predict_bytes(
            data,
            models=_models,
            metadata=_metadata,
            models_dir=MODELS_DIR,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(**result)


@app.post("/predict/gradcam", response_model=GradCamResponse)
async def predict_gradcam(
    file: UploadFile = File(...),
    alpha: float = Query(0.4, ge=0.0, le=1.0),
) -> GradCamResponse:
    """Classify an X-ray and return a Grad-CAM explanation overlay."""
    models, metadata = _require_models()
    _validate_upload(file)

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result, image = predict_bytes(
            data,
            models=models,
            metadata=metadata,
            models_dir=MODELS_DIR,
        )
        gradcam_error = None
        overlay = None
        for model in models:
            try:
                heatmap = make_gradcam_heatmap(model, image)
                overlay = overlay_heatmap(image, heatmap, alpha=alpha)
                break
            except ValueError as exc:
                gradcam_error = str(exc)
        if overlay is None:
            raise ValueError(gradcam_error or "Grad-CAM generation failed.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return GradCamResponse(
        **result,
        gradcam_overlay_base64=_encode_png(overlay),
    )
