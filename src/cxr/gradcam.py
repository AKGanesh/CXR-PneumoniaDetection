from __future__ import annotations

import numpy as np
import tensorflow as tf


def make_gradcam_heatmap(
    model: tf.keras.Model,
    image: np.ndarray,
    conv_layer_name: str | None = None,
    pred_index: int = 0,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap for a single preprocessed image."""
    if image.ndim == 3:
        image_batch = np.expand_dims(image, axis=0)
    else:
        image_batch = image

    if conv_layer_name is None:
        conv_layer = _find_last_conv_layer(model)
    else:
        conv_layer = model.get_layer(conv_layer_name)

    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image_batch)
        if predictions.shape[-1] == 1:
            loss = predictions[:, 0]
        else:
            loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Overlay heatmap on an RGB image for visualization."""
    import cv2

    if image.max() <= 1.0:
        display = (image * 255.0).astype(np.uint8)
    else:
        display = image.astype(np.uint8)

    heatmap_resized = cv2.resize(heatmap, (display.shape[1], display.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(display, 1 - alpha, colored, alpha, 0)
    return overlay


def _find_last_conv_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.SeparableConv2D,
        tf.keras.layers.DepthwiseConv2D,
    )
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                return _find_last_conv_layer(layer)
            except ValueError:
                continue
        if isinstance(layer, conv_types):
            return layer
        output_shape = getattr(layer, "output_shape", None)
        if output_shape and len(output_shape) == 4:
            return layer
    raise ValueError("No convolutional layer found for Grad-CAM.")
