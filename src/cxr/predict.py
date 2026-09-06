from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from cxr.config import TrainConfig
from cxr.models import ensemble_predict
from cxr.preprocessing import label_from_probability, load_image


def load_inference_bundle(models_dir: Path | None = None) -> tuple[list[tf.keras.Model], dict]:
    """Load ensemble models and metadata for inference."""
    config = TrainConfig(models_dir=models_dir or Path("models"))
    metadata_path = config.models_dir / "ensemble_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No trained ensemble found at {metadata_path}. Run scripts/train.py first."
        )

    metadata = json.loads(metadata_path.read_text())
    models = [
        tf.keras.models.load_model(config.models_dir / f"{name}_best.keras")
        for name in metadata["members"]
    ]
    return models, metadata


def predict_array(
    image: np.ndarray,
    models: list[tf.keras.Model],
    metadata: dict,
) -> dict:
    """Run ensemble inference on a preprocessed image array."""
    if image.ndim == 3:
        batch = np.expand_dims(image, axis=0)
    else:
        batch = image

    member_weights = metadata.get("member_weights")
    probability = float(
        ensemble_predict(models, batch, weights=member_weights).numpy()[0][0]
    )
    threshold = float(metadata.get("threshold", 0.5))
    label = label_from_probability(probability, threshold=threshold)

    return {
        "label": label,
        "probability": probability,
        "confidence": probability if label == "pneumonia" else 1.0 - probability,
        "threshold": threshold,
    }


def predict_image(
    image_path: str | Path,
    models: list[tf.keras.Model] | None = None,
    metadata: dict | None = None,
    models_dir: Path | None = None,
) -> dict:
    """Run ensemble inference on a single chest X-ray."""
    if models is None or metadata is None:
        models, metadata = load_inference_bundle(models_dir)

    image = load_image(
        str(image_path),
        size=tuple(metadata.get("image_size", [224, 224])),
        use_clahe=metadata.get("use_clahe", True),
    )
    return predict_array(image, models, metadata)


def predict_bytes(
    data: bytes,
    models: list[tf.keras.Model] | None = None,
    metadata: dict | None = None,
    models_dir: Path | None = None,
) -> tuple[dict, np.ndarray]:
    """Run ensemble inference on uploaded image bytes."""
    from cxr.preprocessing import preprocess_image_bytes

    if models is None or metadata is None:
        models, metadata = load_inference_bundle(models_dir)

    image = preprocess_image_bytes(
        data,
        size=tuple(metadata.get("image_size", [224, 224])),
        use_clahe=metadata.get("use_clahe", True),
    )
    return predict_array(image, models, metadata), image


def predict_directory_to_csv(
    test_dir: Path,
    output_csv: Path,
    models_dir: Path | None = None,
) -> pd.DataFrame:
    """Generate Kaggle-style submission CSV from a folder of test images."""
    models, metadata = load_inference_bundle(models_dir)
    config = TrainConfig(models_dir=models_dir or Path("models"))

    image_paths = sorted(
        path
        for path in test_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    rows = []
    for index, image_path in enumerate(image_paths):
        result = predict_image(image_path, models=models, metadata=metadata)
        rows.append({"ID": index, "class": result["label"]})

    submission = pd.DataFrame(rows)
    submission.to_csv(output_csv, index=False)
    return submission
