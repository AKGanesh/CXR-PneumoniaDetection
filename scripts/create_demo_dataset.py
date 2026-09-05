#!/usr/bin/env python3
"""Create a small synthetic dataset for local pipeline validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def _make_image(label: str, seed: int, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(120, 25, size=(size, size)).clip(0, 255).astype(np.uint8)
    if label == "pneumonia":
        for _ in range(rng.integers(3, 8)):
            center = rng.integers(20, size - 20, size=2)
            radius = int(rng.integers(10, 35))
            cv2.circle(base, tuple(center), radius, int(rng.integers(160, 230)), -1)
    else:
        cv2.line(base, (0, size // 2), (size, size // 2), 180, 2)
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB)
    return rgb


def create_demo_dataset(output_dir: Path, samples_per_class: int = 40) -> None:
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    seed = 0
    for label in ("healthy", "pneumonia"):
        for idx in range(samples_per_class):
            filename = f"{label}_{idx:03d}.png"
            image = _make_image(label, seed=seed)
            cv2.imwrite(str(train_dir / filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            rows.append({"path": filename, "class": label})
            seed += 1

    pd.DataFrame(rows).to_csv(output_dir / "train_metadata.csv", index=False)
    print(f"Created demo dataset at {output_dir} with {len(rows)} images.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic CXR demo dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/demo"))
    parser.add_argument("--samples-per-class", type=int, default=40)
    args = parser.parse_args()
    create_demo_dataset(args.output_dir, samples_per_class=args.samples_per_class)


if __name__ == "__main__":
    main()
