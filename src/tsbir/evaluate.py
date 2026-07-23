from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))

from clip.clip import tokenize  # noqa: E402
from clip.model import CLIP  # noqa: E402
from tsbir.data import eval_transform, read_jsonl  # noqa: E402


class ImageFieldDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]], field: str, image_size: int) -> None:
        self.records = records
        self.field = field
        self.transform = eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return self.transform(Image.open(record[self.field])), index


class CaptionDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.pairs = [
            (record_index, caption_index)
            for record_index, record in enumerate(records)
            for caption_index in range(len(record["captions"]))
        ]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        record_index, caption_index = self.pairs[index]
        caption = self.records[record_index]["captions"][caption_index]
        return tokenize(caption)[0], record_index, caption


def load_model(checkpoint_path: Path, device: torch.device) -> CLIP:
    with (REPO_ROOT / "code" / "training" / "model_configs" / "ViT-B-16.json").open() as handle:
        config = json.load(handle)
    model = CLIP(
        **config,
        weight_sharing=True,
        feature_fusion="avg",
        num_class=90,
        normalize_fused_query=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    state_dict = checkpoint["state_dict"]
    if next(iter(state_dict)).startswith("module."):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def encode_visual(model, records, field, image_size, batch_size, workers, device, sketch=False):
    loader = DataLoader(
        ImageFieldDataset(records, field, image_size),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    features = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            encoded = model.encode_sketch(images) if sketch else model.encode_image(images)
            features.append(F.normalize(encoded.float(), dim=-1))
    return torch.cat(features, dim=0)


def ranks_and_top_five(query: torch.Tensor, gallery: torch.Tensor, targets: torch.Tensor):
    with torch.autocast("cuda", enabled=False):
        scores = query.float() @ gallery.float().t()
    target_scores = scores.gather(1, targets[:, None])
    ranks = (scores > target_scores).sum(dim=1) + 1
    top_five = scores.topk(5, dim=1).indices
    return ranks.cpu().tolist(), top_five.cpu().tolist()


def metrics(ranks: list[int]) -> dict[str, float | int]:
    values = np.asarray(ranks)
    return {
        "queries": int(values.size),
        "R@1": float(np.mean(values <= 1)),
        "R@5": float(np.mean(values <= 5)),
        "R@10": float(np.mean(values <= 10)),
        "MRR": float(np.mean(1.0 / values)),
        "median_rank": float(np.median(values)),
        "mean_rank": float(np.mean(values)),
    }


def text_tile(text: str, size: int = 256) -> Image.Image:
    tile = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(tile)
    lines = textwrap.wrap(text or "[empty text]", width=30)
    draw.multiline_text((12, 12), "\n".join(lines[:12]), fill="black", spacing=5)
    return tile


def case_grid(row: dict[str, Any], records: list[dict[str, Any]], output: Path) -> None:
    mode = row["mode"]
    target_id = int(row["coco_id"])
    if mode == "text":
        query_tile = text_tile(row["caption"])
    else:
        target_record = records[row["record_index"]]
        query_tile = Image.open(target_record["human_sketch"]).convert("RGB").resize((256, 256))

    result_tiles = []
    for gallery_index in row["top5_indices"]:
        record = records[gallery_index]
        tile = Image.open(record["image"]).convert("RGB").resize((256, 256))
        draw = ImageDraw.Draw(tile)
        color = "#00b050" if int(record["coco_id"]) == target_id else "#cc3333"
        draw.rectangle((3, 3, 252, 252), outline=color, width=6)
        result_tiles.append(tile)

    title_height = 90
    canvas = Image.new("RGB", (256 * 6, 256 + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"mode={mode}  target={target_id}  rank={row['rank']}"
    if row["caption"]:
        title += "\n" + "\n".join(textwrap.wrap(row["caption"], width=120)[:2])
    draw.multiline_text((10, 8), title, fill="black", spacing=4)
    for index, tile in enumerate([query_tile, *result_tiles]):
        canvas.paste(tile, (index * 256, title_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def save_results(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w") as handle:
        for row in rows:
            serialized = {key: value for key, value in row.items() if key != "record_index"}
            handle.write(json.dumps(serialized, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/taskformer_coco_official_finetune/last_weights.pt"),
    )
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifests/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/test_last_epoch"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--visual-batch-size", type=int, default=256)
    parser.add_argument("--text-batch-size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", 0)
    records = read_jsonl(args.manifest)
    if len(records) != 5000:
        raise RuntimeError(f"expected 5000 test images, found {len(records)}")
    model = load_model(args.checkpoint, device)

    print("encoding 5k image gallery", flush=True)
    gallery = encode_visual(
        model, records, "image", args.image_size, args.visual_batch_size, args.workers, device, sketch=False
    )
    print("encoding 5k human sketches", flush=True)
    sketch_features = encode_visual(
        model,
        records,
        "human_sketch",
        args.image_size,
        args.visual_batch_size,
        args.workers,
        device,
        sketch=True,
    )

    transform = eval_transform(args.image_size)
    blank_sketch = transform(Image.new("RGB", (args.image_size, args.image_size), "white")).unsqueeze(0).to(device)
    empty_tokens = tokenize("").to(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        blank_sketch_feature = F.normalize(model.encode_sketch(blank_sketch).float(), dim=-1)
        empty_text_feature = F.normalize(model.encode_text(empty_tokens).float(), dim=-1)

    target_indices = torch.arange(len(records), device=device)
    sketch_query = F.normalize((sketch_features + empty_text_feature) / 2.0, dim=-1)
    sketch_ranks, sketch_top = ranks_and_top_five(sketch_query, gallery, target_indices)
    mode_rows: dict[str, list[dict[str, Any]]] = {"sketch": [], "text": [], "mixed": []}
    for record_index, (rank, top_indices) in enumerate(zip(sketch_ranks, sketch_top)):
        record = records[record_index]
        mode_rows["sketch"].append(
            {
                "mode": "sketch",
                "record_index": record_index,
                "coco_id": int(record["coco_id"]),
                "caption": "",
                "rank": rank,
                "top5_indices": top_indices,
                "top5_coco_ids": [int(records[index]["coco_id"]) for index in top_indices],
            }
        )

    print("encoding 25,010 captions and scoring text/mixed queries", flush=True)
    caption_loader = DataLoader(
        CaptionDataset(records),
        batch_size=args.text_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for tokens, record_indices, captions in caption_loader:
            record_indices_gpu = record_indices.to(device, non_blocking=True)
            text_features = F.normalize(model.encode_text(tokens.to(device, non_blocking=True)).float(), dim=-1)
            current_sketches = sketch_features[record_indices_gpu]
            queries = {
                "text": F.normalize((text_features + blank_sketch_feature) / 2.0, dim=-1),
                "mixed": F.normalize((text_features + current_sketches) / 2.0, dim=-1),
            }
            for mode, query in queries.items():
                current_ranks, current_top = ranks_and_top_five(query, gallery, record_indices_gpu)
                for record_index, caption, rank, top_indices in zip(
                    record_indices.tolist(), captions, current_ranks, current_top
                ):
                    record = records[record_index]
                    mode_rows[mode].append(
                        {
                            "mode": mode,
                            "record_index": record_index,
                            "coco_id": int(record["coco_id"]),
                            "caption": caption,
                            "rank": rank,
                            "top5_indices": top_indices,
                            "top5_coco_ids": [int(records[index]["coco_id"]) for index in top_indices],
                        }
                    )

    all_metrics = {mode: metrics([row["rank"] for row in rows]) for mode, rows in mode_rows.items()}
    with (args.output / "metrics.json").open("w") as handle:
        json.dump(all_metrics, handle, indent=2)
    for mode, rows in mode_rows.items():
        save_results(rows, args.output / f"{mode}_top5.jsonl")
        successes = [row for row in rows if row["rank"] == 1][:3]
        failures = sorted((row for row in rows if row["rank"] > 5), key=lambda row: row["rank"], reverse=True)[:3]
        for case_type, cases in (("success", successes), ("failure", failures)):
            for index, row in enumerate(cases, 1):
                case_grid(row, records, args.output / "cases" / f"{mode}_{case_type}_{index}.jpg")
    print(json.dumps(all_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
