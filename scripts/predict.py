#!/usr/bin/env python3
"""Run inference on a single chest X-ray image."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cxr.predict import predict_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict pneumonia from a chest X-ray.")
    parser.add_argument("image", type=Path, help="Path to PNG/JPG image")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    args = parser.parse_args()

    result = predict_image(args.image, models_dir=args.models_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
