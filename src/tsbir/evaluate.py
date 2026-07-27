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
from PIL import Image, ImageDraw, ImageFont
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


def load_model(checkpoint_path: Path, device: torch.device, feature_fusion: str = "auto") -> CLIP:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=False)
    state_dict = checkpoint["state_dict"]
    if next(iter(state_dict)).startswith("module."):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

    state_fusion = "gate" if any(key.startswith("gate.") for key in state_dict) else "avg"
    saved_fusion = checkpoint.get("config", {}).get("model", {}).get("feature_fusion")
    if saved_fusion is not None and saved_fusion != state_fusion:
        raise RuntimeError(
            f"checkpoint metadata says feature_fusion={saved_fusion!r}, "
            f"but its state_dict implies {state_fusion!r}"
        )
    checkpoint_fusion = saved_fusion or state_fusion
    if feature_fusion != "auto" and feature_fusion != checkpoint_fusion:
        raise ValueError(
            f"--feature-fusion={feature_fusion!r} does not match checkpoint "
            f"fusion mode {checkpoint_fusion!r}"
        )

    resolved_fusion = checkpoint_fusion if feature_fusion == "auto" else feature_fusion
    saved_model_config = checkpoint.get("config", {}).get("model", {})
    gate_hidden_dim = int(saved_model_config.get("gate_hidden_dim", 256))
    gate_mixed_only = bool(saved_model_config.get("gate_mixed_only", False))
    if resolved_fusion == "gate" and "gate.0.weight" in state_dict:
        state_hidden_dim = int(state_dict["gate.0.weight"].shape[0])
        if "gate_hidden_dim" in saved_model_config and gate_hidden_dim != state_hidden_dim:
            raise RuntimeError(
                f"checkpoint gate_hidden_dim metadata is {gate_hidden_dim}, "
                f"but gate.0.weight implies {state_hidden_dim}"
            )
        gate_hidden_dim = state_hidden_dim

    with (REPO_ROOT / "code" / "training" / "model_configs" / "ViT-B-16.json").open() as handle:
        config = json.load(handle)
    model = CLIP(
        **config,
        weight_sharing=True,
        feature_fusion=resolved_fusion,
        num_class=90,
        normalize_fused_query=True,
        gate_hidden_dim=gate_hidden_dim,
        gate_mixed_only=gate_mixed_only,
    )
    model.load_state_dict(state_dict, strict=True)
    print(
        f"[load_model] feature_fusion={resolved_fusion}, loaded {len(state_dict)} tensors "
        f"from {checkpoint_path}",
        flush=True,
    )
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


def alpha_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    values = np.asarray([row["alpha"] for row in rows], dtype=np.float32)
    return {
        "queries": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p10": float(np.quantile(values, 0.1)),
        "p50": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
    }


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    try:
        return ImageFont.truetype(str(path), size=size)
    except OSError:
        return ImageFont.load_default()


def text_tile(text: str, size: int = 288) -> Image.Image:
    tile = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(tile)
    body_font = load_font(25)
    lines = textwrap.wrap(text or "[empty text]", width=20)
    draw.multiline_text(
        (22, 22), "\n".join(lines[:8]), fill="#111111", font=body_font, spacing=10
    )
    return tile


def case_grid(
    row: dict[str, Any],
    records: list[dict[str, Any]],
    output: Path,
    sketch_field: str = "human_sketch",
) -> None:
    tile_size = 288
    margin = 24
    gap = 16
    header_height = 148
    label_height = 42
    mode = row["mode"]
    target_id = int(row["coco_id"])
    if mode == "text":
        query_tile = text_tile(row["caption"], tile_size)
    else:
        target_record = records[row["record_index"]]
        with Image.open(target_record[sketch_field]) as image:
            query_tile = image.convert("RGB").resize((tile_size, tile_size))

    result_tiles = []
    for gallery_index in row["top5_indices"]:
        record = records[gallery_index]
        with Image.open(record["image"]) as image:
            tile = image.convert("RGB").resize((tile_size, tile_size))
        draw = ImageDraw.Draw(tile)
        color = "#00b050" if int(record["coco_id"]) == target_id else "#cc3333"
        draw.rectangle((3, 3, tile_size - 4, tile_size - 4), outline=color, width=7)
        result_tiles.append(tile)

    tiles = [query_tile, *result_tiles]
    labels = ["QUERY", *[f"TOP-{rank}" for rank in range(1, 6)]]
    if int(row["rank"]) > 5:
        target_record = records[row["record_index"]]
        with Image.open(target_record["image"]) as image:
            correct_tile = image.convert("RGB").resize((tile_size, tile_size))
        correct_draw = ImageDraw.Draw(correct_tile)
        correct_draw.rectangle(
            (3, 3, tile_size - 4, tile_size - 4),
            outline="#00b050",
            width=7,
        )
        tiles.append(correct_tile)
        labels.append("CORRECT IMAGE")

    canvas_width = 2 * margin + len(tiles) * tile_size + (len(tiles) - 1) * gap
    canvas_height = header_height + label_height + tile_size + margin
    canvas = Image.new("RGB", (canvas_width, canvas_height), "#f7f7f7")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(30, bold=True)
    caption_font = load_font(30)
    label_font = load_font(23, bold=True)
    title = f"{mode.upper()} RETRIEVAL   |   Target COCO {target_id}   |   Ground-truth rank: {row['rank']}"
    draw.text((margin, 18), title, fill="#111111", font=title_font)
    if row["caption"]:
        caption = "\n".join(textwrap.wrap(row["caption"], width=125)[:2])
        draw.multiline_text((margin, 62), caption, fill="#303030", font=caption_font, spacing=6)

    for index, (tile, label) in enumerate(zip(tiles, labels)):
        x = margin + index * (tile_size + gap)
        label_box = draw.textbbox((0, 0), label, font=label_font)
        label_width = label_box[2] - label_box[0]
        label_color = "#00843d" if label == "CORRECT IMAGE" else "#333333"
        draw.text(
            (x + (tile_size - label_width) / 2, header_height + 6),
            label,
            fill=label_color,
            font=label_font,
        )
        canvas.paste(tile, (x, header_height + label_height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95, subsampling=0)


def random_case_selection(
    rows: list[dict[str, Any]],
    cases_per_type: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select reproducible, diverse Top-5 successes and failures."""

    rng = np.random.default_rng(seed)

    def sample(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates or cases_per_type == 0:
            return []
        order = rng.permutation(len(candidates)).tolist()
        selected: list[dict[str, Any]] = []
        selected_indices: set[int] = set()
        seen_targets: set[int] = set()

        # Text and mixed evaluation have several captions per image. Prefer
        # distinct target images so that the visual examples stay diverse.
        for candidate_index in order:
            row = candidates[candidate_index]
            target_id = int(row["coco_id"])
            if target_id in seen_targets:
                continue
            selected.append(row)
            selected_indices.add(candidate_index)
            seen_targets.add(target_id)
            if len(selected) == cases_per_type:
                return selected

        # This fallback matters only when fewer distinct images than requested
        # are available; it still samples rows without replacement.
        for candidate_index in order:
            if candidate_index in selected_indices:
                continue
            selected.append(candidates[candidate_index])
            if len(selected) == cases_per_type:
                break
        return selected

    successes = sample([row for row in rows if int(row["rank"]) <= 5])
    failures = sample([row for row in rows if int(row["rank"]) > 5])
    return successes, failures


def render_case_images(
    output: Path,
    records: list[dict[str, Any]],
    mode_rows: dict[str, list[dict[str, Any]]],
    sketch_field: str,
    cases_per_type: int,
    case_seed: int,
) -> None:
    cases_dir = output / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    selection_rows = []
    for mode_index, mode in enumerate(("sketch", "text", "mixed")):
        successes, failures = random_case_selection(
            mode_rows[mode],
            cases_per_type,
            case_seed + mode_index,
        )
        for case_type, cases in (("success", successes), ("failure", failures)):
            for index, row in enumerate(cases, 1):
                filename = f"{mode}_{case_type}_{index}.jpg"
                case_grid(row, records, cases_dir / filename, sketch_field)
                selection_rows.append(
                    {
                        "file": filename,
                        "mode": mode,
                        "case_type": case_type,
                        "coco_id": int(row["coco_id"]),
                        "caption": row["caption"],
                        "rank": int(row["rank"]),
                        "top5_coco_ids": row["top5_coco_ids"],
                    }
                )

    with (cases_dir / "selection.json").open("w") as handle:
        json.dump(
            {
                "seed": case_seed,
                "cases_per_type": cases_per_type,
                "success_definition": "ground-truth rank <= 5",
                "failure_definition": "ground-truth rank > 5",
                "cases": selection_rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )


def render_saved_cases(
    output: Path,
    records: list[dict[str, Any]],
    sketch_field: str = "human_sketch",
    cases_per_type: int = 6,
    case_seed: int = 2026,
) -> None:
    record_indices = {int(record["coco_id"]): index for index, record in enumerate(records)}
    mode_rows: dict[str, list[dict[str, Any]]] = {}
    for mode in ("sketch", "text", "mixed"):
        rows = read_jsonl(output / f"{mode}_top5.jsonl")
        for row in rows:
            row["record_index"] = record_indices[int(row["coco_id"])]
        mode_rows[mode] = rows
    render_case_images(
        output,
        records,
        mode_rows,
        sketch_field,
        cases_per_type,
        case_seed,
    )


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
    parser.add_argument(
        "--sketch-field",
        default="human_sketch",
        help="manifest field used as the sketch query source (human_sketch or synthetic_sketch)",
    )
    parser.add_argument(
        "--feature-fusion",
        default="auto",
        choices=["auto", "avg", "gate"],
        help="query fusion mode; auto detects it from the checkpoint and rejects mismatches",
    )
    parser.add_argument(
        "--gate-checkpoint",
        type=Path,
        help="optional small gate-only checkpoint overlaid after loading the full model",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="redraw case images from existing *_top5.jsonl files without evaluating the model",
    )
    parser.add_argument(
        "--cases-per-type",
        type=int,
        default=6,
        help="number of random success and failure examples rendered for each retrieval mode",
    )
    parser.add_argument(
        "--case-seed",
        type=int,
        default=2026,
        help="random seed used for reproducible case selection",
    )
    args = parser.parse_args()
    if args.cases_per_type < 1:
        parser.error("--cases-per-type must be at least 1")
    args.output.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(args.manifest)
    if len(records) != 5000:
        raise RuntimeError(f"expected 5000 test images, found {len(records)}")
    if args.render_only:
        render_saved_cases(
            args.output,
            records,
            args.sketch_field,
            args.cases_per_type,
            args.case_seed,
        )
        print(f"redrew case images in {args.output / 'cases'}", flush=True)
        return

    device = torch.device("cuda", 0)
    model = load_model(args.checkpoint, device, feature_fusion=args.feature_fusion)
    if args.gate_checkpoint is not None:
        if model.feature_fusion != "gate":
            raise ValueError("--gate-checkpoint requires a gate-enabled full checkpoint")
        gate_checkpoint = torch.load(
            args.gate_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        gate_state_dict = gate_checkpoint.get("gate_state_dict")
        if gate_state_dict is None:
            raise RuntimeError(f"{args.gate_checkpoint} has no gate_state_dict")
        model.gate.load_state_dict(gate_state_dict, strict=True)
        print(f"[load_model] overlaid gate parameters from {args.gate_checkpoint}", flush=True)

    print("encoding 5k image gallery", flush=True)
    gallery = encode_visual(
        model, records, "image", args.image_size, args.visual_batch_size, args.workers, device, sketch=False
    )
    print(f"encoding 5k sketches (field={args.sketch_field})", flush=True)
    sketch_features = encode_visual(
        model,
        records,
        args.sketch_field,
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
    with torch.inference_mode():
        if model.feature_fusion == "gate" and model.gate_mixed_only:
            sketch_query = sketch_features
            sketch_alpha = torch.zeros(
                (sketch_features.shape[0], 1),
                device=sketch_features.device,
                dtype=sketch_features.dtype,
            )
        else:
            sketch_query, sketch_alpha = model.feature_fuse(
                empty_text_feature,
                sketch_features,
                return_alpha=True,
            )
        sketch_ranks, sketch_top = ranks_and_top_five(sketch_query, gallery, target_indices)
    sketch_alphas = sketch_alpha.flatten().cpu().tolist()
    mode_rows: dict[str, list[dict[str, Any]]] = {"sketch": [], "text": [], "mixed": []}
    for record_index, (rank, top_indices, alpha) in enumerate(
        zip(sketch_ranks, sketch_top, sketch_alphas)
    ):
        record = records[record_index]
        mode_rows["sketch"].append(
            {
                "mode": "sketch",
                "record_index": record_index,
                "coco_id": int(record["coco_id"]),
                "caption": "",
                "alpha": alpha,
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
            if model.feature_fusion == "gate" and model.gate_mixed_only:
                text_alpha = torch.ones(
                    (text_features.shape[0], 1),
                    device=text_features.device,
                    dtype=text_features.dtype,
                )
                text_query = (text_features, text_alpha)
            else:
                text_query = model.feature_fuse(
                    text_features,
                    blank_sketch_feature,
                    return_alpha=True,
                )
            queries = {
                "text": text_query,
                "mixed": model.feature_fuse(
                    text_features,
                    current_sketches,
                    return_alpha=True,
                ),
            }
            for mode, (query, alpha) in queries.items():
                current_ranks, current_top = ranks_and_top_five(query, gallery, record_indices_gpu)
                for record_index, caption, rank, top_indices, alpha_value in zip(
                    record_indices.tolist(),
                    captions,
                    current_ranks,
                    current_top,
                    alpha.flatten().cpu().tolist(),
                ):
                    record = records[record_index]
                    mode_rows[mode].append(
                        {
                            "mode": mode,
                            "record_index": record_index,
                            "coco_id": int(record["coco_id"]),
                            "caption": caption,
                            "alpha": alpha_value,
                            "rank": rank,
                            "top5_indices": top_indices,
                            "top5_coco_ids": [int(records[index]["coco_id"]) for index in top_indices],
                        }
                    )

    all_metrics = {mode: metrics([row["rank"] for row in rows]) for mode, rows in mode_rows.items()}
    all_alpha_metrics = {mode: alpha_metrics(rows) for mode, rows in mode_rows.items()}
    with (args.output / "metrics.json").open("w") as handle:
        json.dump(all_metrics, handle, indent=2)
    with (args.output / "alpha_metrics.json").open("w") as handle:
        json.dump(all_alpha_metrics, handle, indent=2)
    for mode, rows in mode_rows.items():
        save_results(rows, args.output / f"{mode}_top5.jsonl")
    render_case_images(
        args.output,
        records,
        mode_rows,
        args.sketch_field,
        args.cases_per_type,
        args.case_seed,
    )
    print(json.dumps({"retrieval": all_metrics, "alpha": all_alpha_metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
