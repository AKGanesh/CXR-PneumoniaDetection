from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications import EfficientNetB0, VGG16
from tensorflow.keras.layers import (
    BatchNormalization,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)
from tensorflow.keras.optimizers import Nadam


def build_transfer_model(
    backbone: str,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    dropout: float = 0.3,
) -> tuple[Model, Model]:
    """Build a transfer-learning model with a frozen backbone."""
    backbone = backbone.lower()
    if backbone == "efficientnet":
        base_model = EfficientNetB0(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
    elif backbone == "vgg16":
        base_model = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=input_shape,
        )
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=outputs, name=f"{backbone}_pneumonia")
    for layer in base_model.layers:
        layer.trainable = False

    return model, base_model


def compile_model(
    model: Model,
    learning_rate: float = 1e-3,
) -> Model:
    model.compile(
        optimizer=Nadam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def unfreeze_top_layers(base_model: Model, num_layers: int) -> None:
    for layer in base_model.layers[-num_layers:]:
        layer.trainable = True


def ensemble_predict(
    models: list[Model],
    inputs: tf.Tensor | tf.data.Dataset,
    weights: list[float] | None = None,
) -> tf.Tensor:
    """Average predictions from multiple models, optionally weighted."""
    predictions = [model.predict(inputs, verbose=0) for model in models]
    stacked = tf.stack(predictions, axis=0)
    if weights is None:
        return tf.reduce_mean(stacked, axis=0)

    weight_tensor = tf.reshape(tf.constant(weights, dtype=tf.float32), (-1, 1, 1))
    weighted = stacked * weight_tensor
    return tf.reduce_sum(weighted, axis=0) / tf.reduce_sum(weight_tensor)


def get_last_conv_layer(model: Model) -> tf.keras.layers.Layer:
    """Find the last convolutional layer for Grad-CAM."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                return get_last_conv_layer(layer)
            except ValueError:
                continue
        if len(layer.output_shape) == 4:
            return layer
    raise ValueError("No convolutional layer found for Grad-CAM.")
