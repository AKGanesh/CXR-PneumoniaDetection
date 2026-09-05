from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from cxr.config import TrainConfig


def prepare_class_folders(
    metadata_csv: Path,
    source_dir: Path,
    output_dir: Path,
) -> Path:
    """Organize flat image folders into class subdirectories for Keras generators."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for class_name in ("healthy", "pneumonia"):
        (output_dir / class_name).mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_csv)
    for _, row in metadata.iterrows():
        class_name = row["class"]
        if class_name not in {"healthy", "pneumonia"}:
            continue
        src = source_dir / row["path"]
        dst = output_dir / class_name / row["path"]
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    return output_dir


def build_generators(
    data_dir: Path,
    config: TrainConfig,
) -> tuple[ImageDataGenerator, object, object, dict[int, float]]:
    """Create augmented train/validation generators and class weights."""
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        fill_mode="nearest",
        rescale=1.0 / 255.0,
        validation_split=config.validation_split,
    )

    train_generator = datagen.flow_from_directory(
        str(data_dir),
        target_size=config.image_size,
        batch_size=config.batch_size,
        class_mode="binary",
        shuffle=True,
        seed=config.seed,
        subset="training",
    )
    validation_generator = datagen.flow_from_directory(
        str(data_dir),
        target_size=config.image_size,
        batch_size=config.batch_size,
        class_mode="binary",
        shuffle=False,
        seed=config.seed,
        subset="validation",
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_generator.classes),
        y=train_generator.classes,
    )
    class_weight_dict = dict(enumerate(class_weights))
    return datagen, train_generator, validation_generator, class_weight_dict


def build_test_generator(
    test_dir: Path,
    config: TrainConfig,
    batch_size: int = 1,
):
    """Generator for unlabeled test images."""
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    return test_datagen.flow_from_directory(
        str(test_dir),
        target_size=config.image_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )
