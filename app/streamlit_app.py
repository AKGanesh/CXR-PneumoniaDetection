"""Streamlit app for pneumonia detection from chest X-rays."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr.gradcam import make_gradcam_heatmap, overlay_heatmap
from cxr.models import ensemble_predict
from cxr.predict import load_inference_bundle
from cxr.preprocessing import apply_clahe, label_from_probability, to_display_image


st.set_page_config(
    page_title="CXR Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
)

MODELS_DIR = ROOT / "models"


@st.cache_resource
def load_models():
    return load_inference_bundle(MODELS_DIR)


def preprocess_upload(uploaded_file, use_clahe: bool, image_size: tuple[int, int]) -> np.ndarray:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Could not decode uploaded image.")

    if use_clahe:
        image = apply_clahe(image_bgr)
    else:
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, image_size)
    return image.astype(np.float32) / 255.0


def main() -> None:
    st.title("Chest X-Ray Pneumonia Detection")
    st.caption(
        "Upload a chest X-ray to classify it as healthy or pneumonia. "
        "The ensemble uses EfficientNet + VGG16 with CLAHE preprocessing."
    )

    if not (MODELS_DIR / "ensemble_metadata.json").exists():
        st.error(
            "No trained model found. Run `python scripts/create_demo_dataset.py` "
            "and `python scripts/train.py` first, or place trained weights in `models/`."
        )
        st.stop()

    models, metadata = load_models()
    image_size = tuple(metadata.get("image_size", [224, 224]))
    use_clahe = metadata.get("use_clahe", True)
    threshold = float(metadata.get("threshold", 0.5))

    col_left, col_right = st.columns([1, 1])
    with col_left:
        uploaded = st.file_uploader(
            "Upload chest X-ray (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
        )
        show_gradcam = st.checkbox("Show Grad-CAM explanation", value=True)

    if uploaded is None:
        st.info("Upload an image to run inference.")
        return

    image = preprocess_upload(uploaded, use_clahe=use_clahe, image_size=image_size)
    batch = np.expand_dims(image, axis=0)
    member_weights = metadata.get("member_weights")
    probability = float(
        ensemble_predict(models, batch, weights=member_weights).numpy()[0][0]
    )
    label = label_from_probability(probability, threshold=threshold)
    confidence = probability if label == "pneumonia" else 1.0 - probability

    with col_right:
        st.subheader("Prediction")
        if label == "pneumonia":
            st.error(f"**{label.title()}** ({confidence * 100:.1f}% confidence)")
        else:
            st.success(f"**{label.title()}** ({confidence * 100:.1f}% confidence)")

        st.metric("Pneumonia probability", f"{probability:.3f}")
        st.metric("Decision threshold", f"{threshold:.3f}")
        st.write("Ensemble members:", ", ".join(metadata.get("members", [])))

    display = to_display_image(image)
    st.image(display, caption="Preprocessed X-ray", use_container_width=True)

    if show_gradcam:
        st.subheader("Grad-CAM (primary model)")
        primary_model = models[0]
        heatmap = make_gradcam_heatmap(primary_model, batch[0])
        overlay = overlay_heatmap(display, heatmap)
        st.image(overlay, caption="Model attention overlay", use_container_width=True)


if __name__ == "__main__":
    main()
