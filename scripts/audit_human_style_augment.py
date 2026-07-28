#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "code"))

from tsbir.data import human_style_augment, stroke_dropout  # noqa: E402


def fit(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_coco_clip_humanstyle_consistency_10ep_auto2gpu.yaml"),
    )
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/human_style_augmentation_audit.jpg"),
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    options = config["data"]["human_style"]
    records = []
    with Path(config["data"]["train_manifest"]).open() as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
            if len(records) == args.samples:
                break

    tile = 192
    header = 34
    columns = 4
    canvas = Image.new("RGB", (columns * tile, header + len(records) * tile), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(("ORIGINAL", "PIXEL A", "C2 VIEW 1", "C2 VIEW 2")):
        draw.text((column * tile + 8, 9), label, fill="black")

    for row, record in enumerate(records):
        original = Image.open(record["synthetic_sketch"]).convert("L")
        random.seed(2026 + row)
        np.random.seed(2026 + row)
        pixel = stroke_dropout(original)
        random.seed(2026 + row)
        np.random.seed(2026 + row)
        target_keep = random.uniform(float(options["minimum_keep"]), 1.0)
        first, first_keep = human_style_augment(original, options, target_keep)
        second, second_keep = human_style_augment(original, options, target_keep)
        for column, image in enumerate((original, pixel, first, second)):
            canvas.paste(fit(image, tile), (column * tile, header + row * tile))
        draw.text(
            (2 * tile + 5, header + row * tile + 5),
            f"keep={first_keep:.2f}",
            fill="#c00000",
        )
        draw.text(
            (3 * tile + 5, header + row * tile + 5),
            f"keep={second_keep:.2f}",
            fill="#c00000",
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=92)
    print(f"saved {len(records)} samples to {args.output}")


if __name__ == "__main__":
    main()
