from __future__ import annotations

import heapq
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomAffine,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from torchvision.transforms import functional as transform_functional

from clip.clip import tokenize


CLIP_NORMALIZE = Normalize(
    (0.48145466, 0.4578275, 0.40821073),
    (0.26862954, 0.26130258, 0.27577711),
)


def _rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB")


def train_image_transform(image_size: int) -> Compose:
    return Compose(
        [
            RandomAffine(
                degrees=30,
                translate=(0.3, 0.3),
                shear=(-30, 30, -30, 30),
                scale=(1.0, 2.0),
                fill=255,
                interpolation=InterpolationMode.BICUBIC,
            ),
            RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
            _rgb,
            ToTensor(),
            CLIP_NORMALIZE,
        ]
    )


def train_sketch_transform(image_size: int) -> Compose:
    return Compose(
        [
            RandomAffine(
                degrees=30,
                translate=(0.3, 0.3),
                shear=(-30, 30, -30, 30),
                scale=(1.0, 2.0),
                fill=255,
                interpolation=InterpolationMode.BICUBIC,
            ),
            RandomResizedCrop(image_size, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
            _rgb,
            ToTensor(),
            CLIP_NORMALIZE,
        ]
    )


def eval_transform(image_size: int) -> Compose:
    return Compose(
        [
            Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            CenterCrop(image_size),
            _rgb,
            ToTensor(),
            CLIP_NORMALIZE,
        ]
    )


def stroke_dropout(image: Image.Image, minimum_keep: float = 0.6) -> Image.Image:
    gray = np.asarray(image.convert("L")).copy()
    foreground = gray < 245
    keep_probability = random.uniform(minimum_keep, 1.0)
    removed = foreground & (np.random.random(gray.shape) > keep_probability)
    gray[removed] = 255
    return Image.fromarray(gray, mode="L")


def _zhang_suen_skeleton(foreground: np.ndarray) -> np.ndarray:
    """Return a one-pixel skeleton without consuming any random state."""
    skeleton = np.pad(foreground.astype(np.uint8), 1)
    while True:
        changed = False
        for first_subiteration in (True, False):
            center = skeleton[1:-1, 1:-1]
            p2 = skeleton[:-2, 1:-1]
            p3 = skeleton[:-2, 2:]
            p4 = skeleton[1:-1, 2:]
            p5 = skeleton[2:, 2:]
            p6 = skeleton[2:, 1:-1]
            p7 = skeleton[2:, :-2]
            p8 = skeleton[1:-1, :-2]
            p9 = skeleton[:-2, :-2]
            neighbours = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transition_count = sum(
                (left == 0) & (right == 1)
                for left, right in (
                    (p2, p3), (p3, p4), (p4, p5), (p5, p6),
                    (p6, p7), (p7, p8), (p8, p9), (p9, p2),
                )
            )
            if first_subiteration:
                topology = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                topology = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            remove = center.astype(bool) & (neighbours >= 2) & (neighbours <= 6)
            remove &= transition_count == 1
            remove &= topology
            if remove.any():
                center[remove] = 0
                changed = True
        if not changed:
            break
    return skeleton[1:-1, 1:-1].astype(bool)


def _ordered_component_pixels(component: np.ndarray) -> list[tuple[int, int]]:
    """Trace an unbranched skeleton component in spatial order."""
    coordinates = [tuple(point) for point in np.argwhere(component)]
    if len(coordinates) <= 1:
        return coordinates
    coordinate_set = set(coordinates)

    def adjacent(point: tuple[int, int]) -> list[tuple[int, int]]:
        row, column = point
        return sorted(
            (row + dr, column + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
            if (dr or dc) and (row + dr, column + dc) in coordinate_set
        )

    endpoints = [point for point in coordinates if len(adjacent(point)) <= 1]
    start = min(endpoints or coordinates)
    ordered: list[tuple[int, int]] = []
    visited: set[tuple[int, int]] = set()
    stack = [start]
    while stack:
        point = stack.pop()
        if point in visited:
            continue
        visited.add(point)
        ordered.append(point)
        # Reversed push makes the traversal's next point lexicographically
        # smallest and therefore deterministic.
        stack.extend(reversed([item for item in adjacent(point) if item not in visited]))
    if len(visited) != len(coordinates):
        ordered.extend(sorted(coordinate_set - visited))
    return ordered


def _skeleton_segment_labels(skeleton: np.ndarray, maximum_length: int) -> np.ndarray:
    neighbour_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbour_kernel[1, 1] = 0
    neighbour_count = ndimage.convolve(skeleton.astype(np.uint8), neighbour_kernel, mode="constant")
    nodes = skeleton & (neighbour_count != 2)
    paths = skeleton & ~nodes
    component_labels, component_count = ndimage.label(paths, structure=np.ones((3, 3), dtype=np.uint8))
    segment_labels = np.zeros(skeleton.shape, dtype=np.int32)
    next_label = 1
    component_slices = ndimage.find_objects(component_labels)
    for component_id, component_slice in enumerate(component_slices, start=1):
        if component_slice is None:
            continue
        local_component = component_labels[component_slice] == component_id
        ordered = _ordered_component_pixels(local_component)
        row_offset = component_slice[0].start
        column_offset = component_slice[1].start
        ordered = [(row + row_offset, column + column_offset) for row, column in ordered]
        chunk_count = max(1, math.ceil(len(ordered) / maximum_length))
        for chunk in np.array_split(np.asarray(ordered, dtype=np.int32), chunk_count):
            if not len(chunk):
                continue
            segment_labels[chunk[:, 0], chunk[:, 1]] = next_label
            next_label += 1

    # Junction/end pixels were cut to separate paths. Attach each of them to
    # its nearest path segment. A tiny dot with no path becomes one segment.
    if next_label == 1:
        if skeleton.any():
            segment_labels[skeleton] = 1
        return segment_labels
    _, nearest = ndimage.distance_transform_edt(segment_labels == 0, return_indices=True)
    nearest_labels = segment_labels[nearest[0], nearest[1]]
    segment_labels[nodes] = nearest_labels[nodes]
    return segment_labels


def stroke_segment_dropout(
    image: Image.Image,
    minimum_keep: float = 0.6,
    maximum_segment_length: int = 64,
    maximum_segment_area_ratio: float = 0.2,
    minimum_segments_remaining: int = 3,
) -> Image.Image:
    """Delete coherent skeleton segments under the paper's keep-ratio budget.

    This intentionally consumes the same two global RNG calls as
    :func:`stroke_dropout`. All later choices are derived from ``random_field``
    so changing only the dropout mode does not shift subsequent augmentations.
    """
    gray = np.asarray(image.convert("L")).copy()
    foreground = gray < 245
    keep_probability = random.uniform(minimum_keep, 1.0)
    random_field = np.random.random(gray.shape)
    foreground_area = int(foreground.sum())
    if foreground_area == 0:
        return Image.fromarray(gray, mode="L")

    skeleton = _zhang_suen_skeleton(foreground)
    segment_labels = _skeleton_segment_labels(skeleton, maximum_segment_length)
    if not segment_labels.any():
        return Image.fromarray(gray, mode="L")

    # Expand the one-pixel labels back over the anti-aliased/thick foreground.
    _, nearest = ndimage.distance_transform_edt(segment_labels == 0, return_indices=True)
    foreground_segments = segment_labels[nearest[0], nearest[1]]
    foreground_labels = foreground_segments[foreground]
    segment_count = int(segment_labels.max())
    segment_areas = np.bincount(foreground_labels, minlength=segment_count + 1)

    priorities = np.full(segment_count + 1, -1.0, dtype=np.float64)
    flat_labels = segment_labels.ravel()
    labelled_indices = np.flatnonzero(flat_labels)
    represented_labels, first_positions = np.unique(
        flat_labels[labelled_indices], return_index=True,
    )
    representative_indices = labelled_indices[first_positions]
    priorities[represented_labels] = random_field.ravel()[representative_indices]

    removal_budget = int(math.floor(foreground_area * (1.0 - keep_probability)))
    maximum_segment_area = int(math.floor(foreground_area * maximum_segment_area_ratio))
    required_remaining = min(minimum_segments_remaining, segment_count)
    removed_area = 0
    removed_segments: list[int] = []
    for segment_id in np.argsort(priorities[1:])[::-1] + 1:
        area = int(segment_areas[segment_id])
        if area <= 0 or area > maximum_segment_area:
            continue
        if segment_count - len(removed_segments) <= required_remaining:
            break
        if removed_area + area > removal_budget:
            continue
        removed_segments.append(int(segment_id))
        removed_area += area

    if removed_segments:
        removed = foreground & np.isin(foreground_segments, removed_segments)
        gray[removed] = 255
    return Image.fromarray(gray, mode="L")


DEFAULT_HUMAN_STYLE_CONFIG = {
    "minimum_keep": 0.6,
    "width_probability": 0.35,
    "elastic_probability": 0.30,
    "elastic_amplitude": [1.0, 4.0],
    "elastic_sigma": [8.0, 16.0],
    "gap_probability": 0.30,
    "gap_count": [1, 4],
    "gap_length": [4.0, 16.0],
    "gap_width": [2.0, 6.0],
    "simplify_probability": 0.20,
    "simplify_scale": [0.55, 0.85],
    "small_structure_probability": 0.15,
    "small_segment_maximum_length": 24,
    "segment_probability": 0.10,
    "maximum_segment_length": 64,
    "maximum_structural_area_ratio": 0.10,
}


def _probability(config: dict[str, Any], key: str) -> bool:
    probability = float(config[key])
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"{key} must be in [0, 1], got {probability}")
    return random.random() < probability


def _range_pair(config: dict[str, Any], key: str) -> tuple[float, float]:
    values = config[key]
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{key} must contain exactly two values")
    lower, upper = float(values[0]), float(values[1])
    if lower > upper:
        raise ValueError(f"{key} lower bound exceeds upper bound: {values}")
    return lower, upper


def _elastic_warp(gray: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    amplitude = random.uniform(*_range_pair(config, "elastic_amplitude"))
    sigma = random.uniform(*_range_pair(config, "elastic_sigma"))
    fields = []
    for _ in range(2):
        field = ndimage.gaussian_filter(
            np.random.uniform(-1.0, 1.0, gray.shape),
            sigma=sigma,
            mode="reflect",
        )
        maximum = float(np.abs(field).max())
        fields.append(field * (amplitude / maximum) if maximum > 1e-8 else field)
    rows, columns = np.meshgrid(
        np.arange(gray.shape[0], dtype=np.float32),
        np.arange(gray.shape[1], dtype=np.float32),
        indexing="ij",
    )
    warped = ndimage.map_coordinates(
        gray,
        [rows + fields[0], columns + fields[1]],
        order=1,
        mode="constant",
        cval=255,
    )
    return np.clip(warped, 0, 255).astype(np.uint8)


def _simplify_raster(gray: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    scale = random.uniform(*_range_pair(config, "simplify_scale"))
    height, width = gray.shape
    reduced = Image.fromarray(gray, mode="L").resize(
        (max(8, round(width * scale)), max(8, round(height * scale))),
        resample=Image.Resampling.BILINEAR,
    )
    return np.asarray(
        reduced.resize((width, height), resample=Image.Resampling.BICUBIC),
        dtype=np.uint8,
    ).copy()


def _vary_line_width(gray: np.ndarray) -> np.ndarray:
    operation = random.choice(("thin", "thicken"))
    radius = random.choice((1, 1, 2))
    size = 2 * radius + 1
    if operation == "thicken":
        changed = ndimage.grey_erosion(gray, size=(size, size), mode="constant", cval=255)
    else:
        changed = ndimage.grey_dilation(gray, size=(size, size), mode="constant", cval=255)
        # Do not let thinning erase most of a sparse drawing.
        if np.count_nonzero(changed < 245) < 0.5 * max(np.count_nonzero(gray < 245), 1):
            return gray
    return changed.astype(np.uint8)


def _segment_map(foreground: np.ndarray, maximum_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    skeleton = _zhang_suen_skeleton(foreground)
    segment_labels = _skeleton_segment_labels(skeleton, maximum_length)
    if not segment_labels.any():
        empty = np.zeros(1, dtype=np.int64)
        return segment_labels, empty, empty
    _, nearest = ndimage.distance_transform_edt(segment_labels == 0, return_indices=True)
    expanded = segment_labels[nearest[0], nearest[1]]
    count = int(segment_labels.max())
    areas = np.bincount(expanded[foreground], minlength=count + 1)
    lengths = np.bincount(segment_labels.ravel(), minlength=count + 1)
    return expanded, areas, lengths


def _delete_one_segment(
    gray: np.ndarray,
    target_pixels: int,
    maximum_area: int,
    maximum_length: int,
    small_only: bool,
    small_maximum_length: int,
) -> np.ndarray:
    foreground = gray < 245
    expanded, areas, lengths = _segment_map(foreground, maximum_length)
    candidates = [
        segment_id
        for segment_id in range(1, len(areas))
        if 0 < areas[segment_id] <= maximum_area
        and np.count_nonzero(foreground) - areas[segment_id] >= target_pixels
        and (
            lengths[segment_id] <= small_maximum_length
            if small_only
            else lengths[segment_id] >= 8
        )
    ]
    if candidates:
        selected = random.choice(candidates)
        gray[foreground & (expanded == selected)] = 255
    return gray


def _delete_local_gaps(
    gray: np.ndarray,
    target_pixels: int,
    maximum_area: int,
    config: dict[str, Any],
) -> np.ndarray:
    foreground = gray < 245
    points = np.argwhere(_zhang_suen_skeleton(foreground))
    if not len(points):
        return gray
    minimum_count, maximum_count = (int(value) for value in _range_pair(config, "gap_count"))
    for _ in range(random.randint(minimum_count, maximum_count)):
        center_row, center_column = points[np.random.randint(len(points))]
        half_length = random.uniform(*_range_pair(config, "gap_length")) / 2.0
        half_width = random.uniform(*_range_pair(config, "gap_width")) / 2.0
        angle = random.uniform(0.0, math.pi)
        radius = math.ceil(max(half_length, half_width))
        row_start, row_stop = max(0, center_row - radius), min(gray.shape[0], center_row + radius + 1)
        col_start, col_stop = max(0, center_column - radius), min(gray.shape[1], center_column + radius + 1)
        rows, columns = np.meshgrid(
            np.arange(row_start, row_stop) - center_row,
            np.arange(col_start, col_stop) - center_column,
            indexing="ij",
        )
        major = rows * math.sin(angle) + columns * math.cos(angle)
        minor = rows * math.cos(angle) - columns * math.sin(angle)
        local_gap = (major / max(half_length, 1e-6)) ** 2 + (minor / max(half_width, 1e-6)) ** 2 <= 1.0
        foreground = gray < 245
        candidate = np.zeros(gray.shape, dtype=bool)
        candidate[row_start:row_stop, col_start:col_stop] = local_gap
        candidate &= foreground
        candidate_area = int(candidate.sum())
        if (
            0 < candidate_area <= maximum_area
            and int(foreground.sum()) - candidate_area >= target_pixels
        ):
            gray[candidate] = 255
    return gray


def human_style_augment(
    image: Image.Image,
    config: dict[str, Any] | None = None,
    target_keep: float | None = None,
) -> tuple[Image.Image, float]:
    """Create one human-style view with a shared 60--100% deletion budget."""
    options = {**DEFAULT_HUMAN_STYLE_CONFIG, **(config or {})}
    minimum_keep = float(options["minimum_keep"])
    if not 0.0 < minimum_keep <= 1.0:
        raise ValueError(f"minimum_keep must be in (0, 1], got {minimum_keep}")
    if target_keep is None:
        target_keep = random.uniform(minimum_keep, 1.0)
    if not minimum_keep <= target_keep <= 1.0:
        raise ValueError(f"target_keep must be in [{minimum_keep}, 1], got {target_keep}")

    gray = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    if _probability(options, "elastic_probability"):
        gray = _elastic_warp(gray, options)
    if _probability(options, "simplify_probability"):
        gray = _simplify_raster(gray, options)
    if _probability(options, "width_probability"):
        gray = _vary_line_width(gray)

    reference_foreground = gray < 245
    reference_pixels = int(reference_foreground.sum())
    if reference_pixels == 0:
        return Image.fromarray(gray, mode="L"), 1.0
    target_pixels = max(
        math.ceil(minimum_keep * reference_pixels),
        math.floor(target_keep * reference_pixels),
    )
    maximum_structural_area = max(
        1,
        math.floor(float(options["maximum_structural_area_ratio"]) * reference_pixels),
    )

    if _probability(options, "small_structure_probability"):
        gray = _delete_one_segment(
            gray,
            target_pixels,
            maximum_structural_area,
            int(options["maximum_segment_length"]),
            small_only=True,
            small_maximum_length=int(options["small_segment_maximum_length"]),
        )
    if _probability(options, "segment_probability"):
        gray = _delete_one_segment(
            gray,
            target_pixels,
            maximum_structural_area,
            int(options["maximum_segment_length"]),
            small_only=False,
            small_maximum_length=int(options["small_segment_maximum_length"]),
        )
    if _probability(options, "gap_probability"):
        gray = _delete_local_gaps(
            gray,
            target_pixels,
            maximum_structural_area,
            options,
        )

    # Pixel-level deletion fills the remainder of the same completeness
    # budget, rather than stacking another independent 40% removal on top.
    foreground = gray < 245
    delete_count = max(0, int(foreground.sum()) - target_pixels)
    if delete_count:
        foreground_indices = np.flatnonzero(foreground)
        priorities = np.random.random(len(foreground_indices))
        selected = foreground_indices[np.argpartition(priorities, -delete_count)[-delete_count:]]
        gray.ravel()[selected] = 255
    actual_keep = np.count_nonzero(gray < 245) / reference_pixels
    if actual_keep + 1e-8 < minimum_keep:
        raise RuntimeError(f"human-style augmentation violated keep floor: {actual_keep:.4f}")
    return Image.fromarray(gray, mode="L"), float(actual_keep)


def paired_train_sketch_transform(
    first: Image.Image,
    second: Image.Image,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the exact same global affine and crop to two style views."""
    affine_parameters = RandomAffine.get_params(
        degrees=(-30.0, 30.0),
        translate=(0.3, 0.3),
        scale_ranges=(1.0, 2.0),
        shears=(-30.0, 30.0, -30.0, 30.0),
        img_size=list(first.size),
    )

    def affine(image: Image.Image) -> Image.Image:
        return transform_functional.affine(
            image,
            *affine_parameters,
            interpolation=InterpolationMode.BICUBIC,
            fill=255,
        )

    first, second = affine(first), affine(second)
    crop_parameters = RandomResizedCrop.get_params(
        first,
        scale=(0.8, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
    )

    def finish(image: Image.Image) -> torch.Tensor:
        image = transform_functional.resized_crop(
            image,
            *crop_parameters,
            size=[image_size, image_size],
            interpolation=InterpolationMode.BICUBIC,
        ).convert("RGB")
        return CLIP_NORMALIZE(transform_functional.to_tensor(image))

    return finish(first), finish(second)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


class CaptionSketchDataset(Dataset):
    def __init__(
        self,
        manifest: str | Path,
        image_size: int = 224,
        training: bool = True,
        use_human_sketch: bool = False,
        query_dropout: float = 0.2,
        sketch_dropout_mode: str = "pixel",
        sketch_views: int = 1,
        human_style_config: dict[str, Any] | None = None,
    ) -> None:
        self.records = read_jsonl(manifest)
        self.pairs = [
            (record_index, caption_index)
            for record_index, record in enumerate(self.records)
            for caption_index in range(len(record["captions"]))
        ]
        self.training = training
        self.use_human_sketch = use_human_sketch
        self.query_dropout = query_dropout if training else 0.0
        if sketch_dropout_mode not in {"pixel", "segment", "human_style"}:
            raise ValueError(f"unsupported sketch_dropout_mode: {sketch_dropout_mode}")
        if sketch_views not in {1, 2}:
            raise ValueError(f"sketch_views must be 1 or 2, got {sketch_views}")
        if sketch_dropout_mode == "human_style" and (not training or sketch_views != 2):
            raise ValueError("human_style mode requires training=True and sketch_views=2")
        self.sketch_dropout_mode = sketch_dropout_mode
        self.sketch_views = sketch_views
        self.human_style_config = human_style_config or {}
        self.image_size = image_size
        if training:
            self.image_transform = train_image_transform(image_size)
            self.sketch_transform = train_sketch_transform(image_size)
        else:
            self.image_transform = eval_transform(image_size)
            self.sketch_transform = eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, caption_index = self.pairs[index]
        record = self.records[record_index]
        caption = record["captions"][caption_index]
        image = Image.open(record["image"]).convert("RGB")
        sketch_key = "human_sketch" if self.use_human_sketch else "synthetic_sketch"
        sketch_path = record[sketch_key]
        if not sketch_path:
            raise RuntimeError(f"record {record['coco_id']} has no {sketch_key}")
        sketch = Image.open(sketch_path).convert("L")
        second_sketch = None
        sketch_keep_ratio = 1.0
        second_sketch_keep_ratio = 1.0
        if self.training:
            if self.sketch_dropout_mode == "pixel":
                sketch = stroke_dropout(sketch)
            elif self.sketch_dropout_mode == "segment":
                sketch = stroke_segment_dropout(sketch)
            else:
                minimum_keep = float(
                    self.human_style_config.get(
                        "minimum_keep",
                        DEFAULT_HUMAN_STYLE_CONFIG["minimum_keep"],
                    )
                )
                target_keep = random.uniform(minimum_keep, 1.0)
                original_sketch = sketch
                sketch, sketch_keep_ratio = human_style_augment(
                    original_sketch,
                    self.human_style_config,
                    target_keep=target_keep,
                )
                second_sketch, second_sketch_keep_ratio = human_style_augment(
                    original_sketch,
                    self.human_style_config,
                    target_keep=target_keep,
                )

        drop_draw = random.random()
        drop_sketch = self.training and drop_draw < self.query_dropout / 2
        drop_text = self.training and self.query_dropout / 2 <= drop_draw < self.query_dropout
        if drop_sketch:
            sketch = Image.new("L", sketch.size, color=255)
            if second_sketch is not None:
                second_sketch = Image.new("L", second_sketch.size, color=255)
        query_caption = "" if drop_text else caption

        labels = torch.zeros(90, dtype=torch.float32)
        for category_id in record["category_ids"]:
            labels[category_id - 1] = 1.0

        item = {
            "image": self.image_transform(image),
            "text_tokens": tokenize(query_caption)[0],
            "caption_tokens": tokenize(caption)[0],
            "labels": labels,
            "coco_id": int(record["coco_id"]),
            "caption": caption,
            "sketch_present": not drop_sketch,
        }
        if second_sketch is None:
            item["sketch"] = self.sketch_transform(sketch)
        else:
            item["sketch"], item["sketch_view2"] = paired_train_sketch_transform(
                sketch,
                second_sketch,
                self.image_size,
            )
            item["sketch_keep_ratio"] = sketch_keep_ratio
            item["sketch_view2_keep_ratio"] = second_sketch_keep_ratio
        return item


class DistributedUniqueImageBatchSampler(Sampler[list[int]]):
    """Build DDP batches in which every COCO image occurs at most once.

    Uniqueness is enforced across the full batch on all ranks, not only within
    one rank. All caption pairs remain in the dataset and are scheduled once
    per epoch, except for at most ``num_replicas - 1`` final pairs when the
    epoch size is not divisible by the number of DDP processes.
    """

    def __init__(
        self,
        dataset: CaptionSketchDataset,
        batch_size_per_rank: int,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 0,
    ) -> None:
        if batch_size_per_rank <= 0:
            raise ValueError("batch_size_per_rank must be positive")
        if num_replicas <= 0 or not 0 <= rank < num_replicas:
            raise ValueError(f"invalid distributed sampler settings: rank={rank}, replicas={num_replicas}")
        self.dataset = dataset
        self.batch_size_per_rank = batch_size_per_rank
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = 0
        self.global_batch_size = batch_size_per_rank * num_replicas

        self.pairs_by_record: dict[int, list[int]] = {}
        for pair_index, (record_index, _) in enumerate(dataset.pairs):
            self.pairs_by_record.setdefault(record_index, []).append(pair_index)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _global_batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        pairs_by_record = {record: pairs.copy() for record, pairs in self.pairs_by_record.items()}
        for pairs in pairs_by_record.values():
            rng.shuffle(pairs)

        # Records with more remaining captions are scheduled first. A record
        # is pushed back only after the current batch is complete, so it cannot
        # occur twice in that batch. Random tie breakers reshuffle equal-sized
        # groups between epochs.
        active = [(-len(pairs), rng.random(), record) for record, pairs in pairs_by_record.items()]
        heapq.heapify(active)
        batches: list[list[int]] = []
        while active:
            selected = [heapq.heappop(active) for _ in range(min(self.global_batch_size, len(active)))]
            global_batch: list[int] = []
            for _, _, record in selected:
                global_batch.append(pairs_by_record[record].pop())

            usable = len(global_batch) - len(global_batch) % self.num_replicas
            if usable:
                batches.append(global_batch[:usable])

            for _, _, record in selected:
                if pairs_by_record[record]:
                    heapq.heappush(active, (-len(pairs_by_record[record]), rng.random(), record))

        scheduled = sum(len(batch) for batch in batches)
        if len(self.dataset) - scheduled >= self.num_replicas:
            raise RuntimeError("caption distribution cannot form equal, globally unique DDP batches")
        return batches

    def __iter__(self):
        for global_batch in self._global_batches():
            local_batch = global_batch[self.rank :: self.num_replicas]
            yield local_batch

    def __len__(self) -> int:
        usable_pairs = len(self.dataset) - len(self.dataset) % self.num_replicas
        return math.ceil(usable_pairs / self.global_batch_size)
