from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from cxr.config import TrainConfig
from cxr.data import build_generators, prepare_class_folders
from cxr.metrics import evaluate_predictions, find_best_threshold, save_model_metadata
from cxr.models import build_transfer_model, compile_model, ensemble_predict, unfreeze_top_layers


def _callbacks(model_path: Path) -> list:
    return [
        ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_auc",
            patience=4,
            mode="max",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def train_single_backbone(
    backbone: str,
    train_generator,
    validation_generator,
    class_weight_dict: dict[int, float],
    config: TrainConfig,
) -> tuple[tf.keras.Model, dict]:
    """Train one backbone with transfer learning and fine-tuning."""
    model, base_model = build_transfer_model(
        backbone=backbone,
        input_shape=(*config.image_size, 3),
        dropout=config.dropout,
    )
    compile_model(model, learning_rate=config.learning_rate)

    model_path = config.models_dir / f"{backbone}_best.keras"
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=config.epochs,
        class_weight=class_weight_dict,
        callbacks=_callbacks(model_path),
        verbose=1,
    )

    unfreeze_top_layers(base_model, config.unfreeze_layers)
    compile_model(model, learning_rate=config.fine_tune_learning_rate)
    model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=config.fine_tune_epochs,
        class_weight=class_weight_dict,
        callbacks=_callbacks(model_path),
        verbose=1,
    )

    model.save(model_path)
    val_probs = model.predict(validation_generator, verbose=0).flatten()
    val_labels = validation_generator.classes
    threshold, metrics = find_best_threshold(val_labels, val_probs, metric="f1")

    metadata = {
        "backbone": backbone,
        "threshold": threshold,
        "metrics": metrics.to_dict(),
        "use_clahe": config.use_clahe,
        "image_size": list(config.image_size),
    }
    save_model_metadata(config.models_dir / f"{backbone}_metadata.json", metadata)
    return model, metadata


def train_ensemble(
    metadata_csv: Path,
    source_dir: Path,
    working_dir: Path,
    config: TrainConfig | None = None,
) -> dict:
    """Train all ensemble members and persist ensemble metadata."""
    config = config or TrainConfig()
    config.ensure_dirs()

    class_dir = prepare_class_folders(metadata_csv, source_dir, working_dir / "processed_train_data")
    _, train_gen, val_gen, class_weights = build_generators(class_dir, config)

    member_models: list[tf.keras.Model] = []
    member_metadata: list[dict] = []

    for backbone in config.ensemble_members:
        model, metadata = train_single_backbone(
            backbone=backbone,
            train_generator=train_gen,
            validation_generator=val_gen,
            class_weight_dict=class_weights,
            config=config,
        )
        member_models.append(model)
        member_metadata.append(metadata)

    val_probs = ensemble_predict(member_models, val_gen).numpy().flatten()
    val_labels = val_gen.classes
    threshold, ensemble_metrics = find_best_threshold(val_labels, val_probs, metric="f1")

    ensemble_meta = {
        "members": [meta["backbone"] for meta in member_metadata],
        "member_weights": [meta["metrics"]["f1"] for meta in member_metadata],
        "threshold": threshold,
        "metrics": ensemble_metrics.to_dict(),
        "use_clahe": config.use_clahe,
        "image_size": list(config.image_size),
    }
    save_model_metadata(config.models_dir / "ensemble_metadata.json", ensemble_meta)
    return ensemble_meta


def load_ensemble_models(config: TrainConfig | None = None) -> tuple[list[tf.keras.Model], dict]:
    config = config or TrainConfig()
    metadata = json.loads((config.models_dir / "ensemble_metadata.json").read_text())
    models = [
        tf.keras.models.load_model(config.models_dir / f"{name}_best.keras")
        for name in metadata["members"]
    ]
    return models, metadata
