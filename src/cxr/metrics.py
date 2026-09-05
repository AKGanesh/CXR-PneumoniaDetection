from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float | None
    threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> EvaluationResult:
    y_pred = (y_prob >= threshold).astype(int)
    roc_auc = None
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))

    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        threshold=threshold,
    )


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> tuple[float, EvaluationResult]:
    """Search thresholds that maximize the chosen metric."""
    best_threshold = 0.5
    best_result = evaluate_predictions(y_true, y_prob, threshold=best_threshold)

    for threshold in np.linspace(0.1, 0.9, 81):
        result = evaluate_predictions(y_true, y_prob, threshold=float(threshold))
        score = getattr(result, metric)
        best_score = getattr(best_result, metric)
        if score > best_score:
            best_threshold = float(threshold)
            best_result = result

    return best_threshold, best_result


def save_model_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))


def load_model_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())
