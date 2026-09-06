from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainConfig:
    """Training and inference configuration."""

    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 15
    fine_tune_epochs: int = 10
    validation_split: float = 0.2
    seed: int = 42
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    unfreeze_layers: int = 5
    dropout: float = 0.3
    use_clahe: bool = True
    class_names: tuple[str, str] = ("healthy", "pneumonia")
    models_dir: Path = field(default_factory=lambda: Path("models"))
    ensemble_members: tuple[str, ...] = ("efficientnet", "vgg16")
    decision_threshold: float = 0.5

    def ensure_dirs(self) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_CONFIG = TrainConfig()
