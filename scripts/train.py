#!/usr/bin/env python3
"""Train the pneumonia detection ensemble."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr.config import TrainConfig
from cxr.train import train_ensemble


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CXR pneumonia ensemble.")
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=Path("data/demo/train_metadata.csv"),
        help="CSV with columns: path, class",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/demo/train"),
        help="Directory containing training images",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("data/working"),
        help="Directory for processed class folders",
    )
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["efficientnet", "vgg16"],
        help="Backbones to include in the ensemble",
    )
    args = parser.parse_args()

    config = TrainConfig(
        models_dir=args.models_dir,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        batch_size=args.batch_size,
        ensemble_members=tuple(args.backbones),
    )

    metadata = train_ensemble(
        metadata_csv=args.metadata_csv,
        source_dir=args.source_dir,
        working_dir=args.working_dir,
        config=config,
    )
    print("Training complete.")
    print(f"Ensemble F1: {metadata['metrics']['f1']:.4f}")
    print(f"Optimal threshold: {metadata['threshold']:.3f}")


if __name__ == "__main__":
    main()
