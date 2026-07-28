#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "code"))

from tsbir.data import stroke_dropout, stroke_segment_dropout  # noqa: E402


DEFAULT_COCO_IDS = [391895, 483108, 561100, 334321, 368117]
FONT_CANDIDATES = [
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render matched pixel- and stroke-segment-dropout examples.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifests/test.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dropout_examples"))
    parser.add_argument("--coco-ids", type=int, nargs="+", default=DEFAULT_COCO_IDS)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--minimum-keep", type=float, default=0.6)
    parser.add_argument(
        "--maximum-target-keep",
        type=float,
        default=0.75,
        help="Skip seeds above this target keep ratio so the visual deletion remains obvious.",
    )
    parser.add_argument("--tile-size", type=int, default=240)
    return parser.parse_args()


def resolve_from_repo(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def load_records(manifest: Path, coco_ids: list[int]) -> list[dict[str, Any]]:
    wanted = set(coco_ids)
    found: dict[int, dict[str, Any]] = {}
    with resolve_from_repo(manifest).open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            coco_id = int(record["coco_id"])
            if coco_id in wanted:
                found[coco_id] = record
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"COCO ids missing from manifest: {sorted(missing)}")
    return [found[coco_id] for coco_id in coco_ids]


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def square(image: Image.Image, size: int) -> Image.Image:
    return image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)


def foreground_keep_ratio(before: Image.Image, after: Image.Image) -> float:
    before_mask = np.asarray(before.convert("L")) < 245
    after_mask = np.asarray(after.convert("L")) < 245
    count = int(before_mask.sum())
    return float(after_mask.sum() / count) if count else 1.0


def choose_example_seeds(
    base_seed: int,
    count: int,
    minimum_keep: float,
    maximum_target_keep: float,
) -> list[int]:
    seeds: list[int] = []
    candidate = base_seed
    while len(seeds) < count:
        target_keep = random.Random(candidate).uniform(minimum_keep, 1.0)
        if target_keep <= maximum_target_keep:
            seeds.append(candidate)
        candidate += 1
    return seeds


def center_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    value: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str = "#202124",
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), value, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        (left + (right - left - width) / 2, top + (bottom - top - height) / 2 - bounds[1]),
        value,
        font=font,
        fill=fill,
    )


def render_sheet(
    title: str,
    headers: list[str],
    rows: list[list[Image.Image]],
    footers: list[str],
    tile_size: int,
    output: Path,
) -> None:
    title_height = 58
    header_height = 48
    footer_height = 34
    row_height = tile_size + footer_height
    width = tile_size * len(headers)
    height = title_height + header_height + row_height * len(rows)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(25)
    header_font = load_font(20)
    footer_font = load_font(15)

    center_text(draw, (0, 0, width, title_height), title, title_font, "#111111")
    draw.line((0, title_height - 1, width, title_height - 1), fill="#d8dce1", width=1)
    for column, header in enumerate(headers):
        left = column * tile_size
        center_text(
            draw,
            (left, title_height, left + tile_size, title_height + header_height),
            header,
            header_font,
        )

    image_top = title_height + header_height
    for row_index, (images, footer) in enumerate(zip(rows, footers)):
        top = image_top + row_index * row_height
        for column, image in enumerate(images):
            left = column * tile_size
            canvas.paste(square(image, tile_size), (left, top))
            draw.rectangle(
                (left, top, left + tile_size - 1, top + tile_size - 1),
                outline="#e2e5e9",
                width=1,
            )
        footer_top = top + tile_size
        if row_index % 2 == 0:
            draw.rectangle((0, footer_top, width, footer_top + footer_height), fill="#f7f8fa")
        center_text(
            draw,
            (0, footer_top, width, footer_top + footer_height),
            footer,
            footer_font,
            "#555b65",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, optimize=True)


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.minimum_keep <= 1.0:
        raise ValueError("--minimum-keep must be in [0, 1]")
    if not args.minimum_keep <= args.maximum_target_keep <= 1.0:
        raise ValueError("--maximum-target-keep must be in [minimum_keep, 1]")
    if args.minimum_keep < 1.0 and args.maximum_target_keep == args.minimum_keep:
        raise ValueError("--maximum-target-keep must exceed --minimum-keep")
    records = load_records(args.manifest, args.coco_ids)
    output_dir = resolve_from_repo(args.output_dir)
    row_seeds = choose_example_seeds(
        args.seed,
        len(records),
        args.minimum_keep,
        args.maximum_target_keep,
    )

    pixel_rows: list[list[Image.Image]] = []
    segment_rows: list[list[Image.Image]] = []
    comparison_rows: list[list[Image.Image]] = []
    pixel_footers: list[str] = []
    segment_footers: list[str] = []
    comparison_footers: list[str] = []
    metadata: list[dict[str, Any]] = []

    for row_index, record in enumerate(records):
        image = Image.open(resolve_from_repo(Path(record["image"]))).convert("RGB")
        synthetic = Image.open(resolve_from_repo(Path(record["synthetic_sketch"]))).convert("L")
        human = Image.open(resolve_from_repo(Path(record["human_sketch"]))).convert("RGB")
        row_seed = row_seeds[row_index]

        random.seed(row_seed)
        np.random.seed(row_seed)
        target_keep = random.Random(row_seed).uniform(args.minimum_keep, 1.0)
        pixel = stroke_dropout(synthetic, minimum_keep=args.minimum_keep)
        random.seed(row_seed)
        np.random.seed(row_seed)
        segment = stroke_segment_dropout(synthetic, minimum_keep=args.minimum_keep)

        pixel_keep = foreground_keep_ratio(synthetic, pixel)
        segment_keep = foreground_keep_ratio(synthetic, segment)
        coco_id = int(record["coco_id"])
        pixel_rows.append([image, synthetic, pixel, human])
        segment_rows.append([image, synthetic, segment, human])
        comparison_rows.append([image, synthetic, pixel, segment, human])
        pixel_footers.append(f"COCO {coco_id}  ·  实际保留 {pixel_keep:.1%}")
        segment_footers.append(f"COCO {coco_id}  ·  实际保留 {segment_keep:.1%}")
        comparison_footers.append(
            f"COCO {coco_id}  ·  Pixel {pixel_keep:.1%}  ·  Segment {segment_keep:.1%}",
        )
        metadata.append(
            {
                "coco_id": coco_id,
                "seed": row_seed,
                "target_keep": target_keep,
                "pixel_keep": pixel_keep,
                "segment_keep": segment_keep,
                "image": record["image"],
                "synthetic_sketch": record["synthetic_sketch"],
                "human_sketch": record["human_sketch"],
            },
        )

    render_sheet(
        "方法一：像素级随机删除",
        ["原始图像", "合成草图", "Pixel Dropout", "人工草图"],
        pixel_rows,
        pixel_footers,
        args.tile_size,
        output_dir / "pixel_dropout_examples.png",
    )
    render_sheet(
        "方法二：骨架连通笔画段删除",
        ["原始图像", "合成草图", "Segment Dropout", "人工草图"],
        segment_rows,
        segment_footers,
        args.tile_size,
        output_dir / "segment_dropout_examples.png",
    )
    render_sheet(
        "两种草图 Dropout 方法对比",
        ["原始图像", "合成草图", "Pixel Dropout", "Segment Dropout", "人工草图"],
        comparison_rows,
        comparison_footers,
        args.tile_size,
        output_dir / "dropout_comparison.png",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "examples.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"rendered {len(records)} examples in {output_dir}")


if __name__ == "__main__":
    main()
