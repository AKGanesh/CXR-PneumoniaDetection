from __future__ import annotations

from typing import Literal

import cv2
import numpy as np


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE contrast enhancement, common for chest X-rays."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray.astype(np.uint8))
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def load_image(
    path: str,
    size: tuple[int, int] = (224, 224),
    use_clahe: bool = True,
) -> np.ndarray:
    """Load and preprocess a single image for model input."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")

    return preprocess_image_array(image, size=size, use_clahe=use_clahe)


def preprocess_image_array(
    image: np.ndarray,
    size: tuple[int, int] = (224, 224),
    use_clahe: bool = True,
) -> np.ndarray:
    """Preprocess a single image array for model input."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

    if use_clahe:
        image = apply_clahe(image)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, size)
    return image.astype(np.float32) / 255.0


def preprocess_image_bytes(
    data: bytes,
    size: tuple[int, int] = (224, 224),
    use_clahe: bool = True,
) -> np.ndarray:
    """Decode uploaded image bytes and preprocess for model input."""
    encoded = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to decode uploaded image bytes.")
    return preprocess_image_array(image, size=size, use_clahe=use_clahe)


def preprocess_batch(
    images: np.ndarray,
    use_clahe: bool = True,
    size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Preprocess a batch of images in HWC uint8/float format."""
    processed = []
    for image in images:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        if use_clahe:
            image = apply_clahe(image)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(image, size)
        processed.append(image.astype(np.float32) / 255.0)
    return np.asarray(processed)


def to_display_image(image: np.ndarray) -> np.ndarray:
    """Convert a normalized RGB image to uint8 for plotting."""
    if image.max() <= 1.0:
        image = (image * 255.0).clip(0, 255)
    return image.astype(np.uint8)


def label_from_probability(
    probability: float,
    threshold: float = 0.5,
) -> Literal["healthy", "pneumonia"]:
    return "pneumonia" if probability >= threshold else "healthy"
