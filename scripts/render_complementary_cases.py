#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from tsbir.data import read_jsonl
from tsbir.evaluate import load_font, text_tile


def load_rows(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def join_results(results_dir: Path) -> list[dict[str, Any]]:
    sketch_rows = {
        int(row["coco_id"]): row
        for row in load_rows(results_dir / "sketch_top5.jsonl")
    }
    text_rows = load_rows(results_dir / "text_top5.jsonl")
    mixed_rows = load_rows(results_dir / "mixed_top5.jsonl")
    if len(text_rows) != len(mixed_rows):
        raise RuntimeError("text and mixed result files have different lengths")

    joined = []
    for text_row, mixed_row in zip(text_rows, mixed_rows):
        if (
            int(text_row["coco_id"]) != int(mixed_row["coco_id"])
            or text_row["caption"] != mixed_row["caption"]
        ):
            raise RuntimeError("text and mixed rows are not aligned")
        sketch_row = sketch_rows[int(text_row["coco_id"])]
        joined.append(
            {
                "coco_id": int(text_row["coco_id"]),
                "caption": text_row["caption"],
                "text_rank": int(text_row["rank"]),
                "sketch_rank": int(sketch_row["rank"]),
                "mixed_rank": int(mixed_row["rank"]),
                "text_top5_indices": text_row["top5_indices"],
                "text_top5_coco_ids": text_row["top5_coco_ids"],
                "sketch_top5_indices": sketch_row["top5_indices"],
                "sketch_top5_coco_ids": sketch_row["top5_coco_ids"],
                "mixed_top5_indices": mixed_row["top5_indices"],
                "mixed_top5_coco_ids": mixed_row["top5_coco_ids"],
            }
        )
    return joined


def select_distinct(
    rows: list[dict[str, Any]],
    predicate,
    sort_key,
    count: int,
) -> list[dict[str, Any]]:
    selected = []
    seen_targets: set[int] = set()
    for row in sorted((row for row in rows if predicate(row)), key=sort_key):
        if row["coco_id"] in seen_targets:
            continue
        selected.append(row)
        seen_targets.add(row["coco_id"])
        if len(selected) == count:
            break
    return selected


def image_tile(path: str, size: int) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").resize((size, size))


def bordered(tile: Image.Image, correct: bool) -> Image.Image:
    result = tile.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle(
        (3, 3, result.width - 4, result.height - 4),
        outline="#00a651" if correct else "#d13c3c",
        width=7,
    )
    return result


def render_comparison(
    row: dict[str, Any],
    records: list[dict[str, Any]],
    record_by_id: dict[int, int],
    output: Path,
    sketch_field: str,
) -> None:
    tile_size = 224
    margin = 24
    gap = 14
    row_label_width = 190
    row_gap = 22
    header_height = 135
    tile_label_height = 34
    target_id = int(row["coco_id"])
    record_index = record_by_id[target_id]
    target_record = records[record_index]

    canvas_width = 2 * margin + row_label_width + 5 * tile_size + 4 * gap
    query_height = tile_label_height + tile_size
    result_height = tile_label_height + tile_size
    canvas_height = (
        header_height
        + query_height
        + row_gap
        + 3 * result_height
        + 2 * row_gap
        + margin
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), "#f7f7f7")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(29, bold=True)
    caption_font = load_font(25)
    row_font = load_font(23, bold=True)
    label_font = load_font(20, bold=True)

    title = (
        f"COCO {target_id}   |   Text rank: {row['text_rank']}   |   "
        f"Sketch rank: {row['sketch_rank']}   |   Mixed rank: {row['mixed_rank']}"
    )
    draw.text((margin, 16), title, fill="#111111", font=title_font)
    caption = "\n".join(textwrap.wrap(row["caption"], width=105)[:2])
    draw.multiline_text((margin, 58), caption, fill="#303030", font=caption_font, spacing=5)

    content_x = margin + row_label_width
    y = header_height
    draw.text((margin, y + 92), "QUERY / GT", fill="#333333", font=row_font)
    query_tiles = [
        (image_tile(target_record[sketch_field], tile_size), "SKETCH QUERY"),
        (text_tile(row["caption"], tile_size), "TEXT QUERY"),
        (bordered(image_tile(target_record["image"], tile_size), True), "CORRECT IMAGE"),
    ]
    for index, (tile, label) in enumerate(query_tiles):
        x = content_x + index * (tile_size + gap)
        color = "#00843d" if label == "CORRECT IMAGE" else "#333333"
        draw.text((x + 8, y + 4), label, fill=color, font=label_font)
        canvas.paste(tile, (x, y + tile_label_height))

    y += query_height + row_gap
    modes = (
        ("TEXT TOP-5", "text", row["text_rank"]),
        ("SKETCH TOP-5", "sketch", row["sketch_rank"]),
        ("MIXED TOP-5", "mixed", row["mixed_rank"]),
    )
    for row_index, (row_label, prefix, rank) in enumerate(modes):
        draw.multiline_text(
            (margin, y + 85),
            f"{row_label}\nGT rank: {rank}",
            fill="#333333",
            font=row_font,
            spacing=5,
        )
        indices = row[f"{prefix}_top5_indices"]
        coco_ids = row[f"{prefix}_top5_coco_ids"]
        for top_index, (gallery_index, coco_id) in enumerate(zip(indices, coco_ids), 1):
            x = content_x + (top_index - 1) * (tile_size + gap)
            draw.text((x + 78, y + 4), f"TOP-{top_index}", fill="#333333", font=label_font)
            tile = bordered(
                image_tile(records[int(gallery_index)]["image"], tile_size),
                int(coco_id) == target_id,
            )
            canvas.paste(tile, (x, y + tile_label_height))
        y += result_height
        if row_index < len(modes) - 1:
            y += row_gap

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95, subsampling=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("outputs/test_clip_10ep_gpu12_last"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifests/test.jsonl"),
    )
    parser.add_argument("--sketch-field", default="human_sketch")
    parser.add_argument("--cases-per-group", type=int, default=6)
    args = parser.parse_args()

    records = read_jsonl(args.manifest)
    record_by_id = {
        int(record["coco_id"]): index
        for index, record in enumerate(records)
    }
    rows = join_results(args.results)

    groups = {
        "both_bad_mixed_good": select_distinct(
            rows,
            lambda row: (
                row["text_rank"] > 20
                and row["sketch_rank"] > 20
                and row["mixed_rank"] <= 5
            ),
            lambda row: (
                row["mixed_rank"],
                -min(row["text_rank"], row["sketch_rank"]),
            ),
            args.cases_per_group,
        ),
        "text_bad_sketch_rescues": select_distinct(
            rows,
            lambda row: (
                row["text_rank"] > 100
                and row["sketch_rank"] <= 5
                and row["mixed_rank"] <= 5
            ),
            lambda row: (row["mixed_rank"], -row["text_rank"]),
            args.cases_per_group,
        ),
        "sketch_bad_text_rescues": select_distinct(
            rows,
            lambda row: (
                row["sketch_rank"] > 100
                and row["text_rank"] <= 5
                and row["mixed_rank"] <= 5
            ),
            lambda row: (row["mixed_rank"], -row["sketch_rank"]),
            args.cases_per_group,
        ),
    }

    output_dir = args.results / "cases" / "complementary"
    selection = {}
    for group_name, selected in groups.items():
        selection[group_name] = []
        for index, row in enumerate(selected, 1):
            filename = f"{group_name}_{index:02d}_coco_{row['coco_id']}.jpg"
            render_comparison(
                row,
                records,
                record_by_id,
                output_dir / filename,
                args.sketch_field,
            )
            selection[group_name].append(
                {
                    "file": filename,
                    "coco_id": row["coco_id"],
                    "caption": row["caption"],
                    "text_rank": row["text_rank"],
                    "sketch_rank": row["sketch_rank"],
                    "mixed_rank": row["mixed_rank"],
                }
            )

    with (output_dir / "selection.json").open("w") as handle:
        json.dump(selection, handle, indent=2, ensure_ascii=False)
    print(
        f"rendered {sum(len(group) for group in groups.values())} complementary cases "
        f"in {output_dir}"
    )


if __name__ == "__main__":
    main()
