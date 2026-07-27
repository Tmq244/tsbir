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
        if sketch_dropout_mode not in {"pixel", "segment"}:
            raise ValueError(f"unsupported sketch_dropout_mode: {sketch_dropout_mode}")
        self.sketch_dropout_mode = sketch_dropout_mode
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
        if self.training:
            if self.sketch_dropout_mode == "pixel":
                sketch = stroke_dropout(sketch)
            else:
                sketch = stroke_segment_dropout(sketch)

        drop_draw = random.random()
        drop_sketch = self.training and drop_draw < self.query_dropout / 2
        drop_text = self.training and self.query_dropout / 2 <= drop_draw < self.query_dropout
        if drop_sketch:
            sketch = Image.new("L", sketch.size, color=255)
        query_caption = "" if drop_text else caption

        labels = torch.zeros(90, dtype=torch.float32)
        for category_id in record["category_ids"]:
            labels[category_id - 1] = 1.0

        return {
            "image": self.image_transform(image),
            "sketch": self.sketch_transform(sketch),
            "text_tokens": tokenize(query_caption)[0],
            "caption_tokens": tokenize(caption)[0],
            "labels": labels,
            "coco_id": int(record["coco_id"]),
            "caption": caption,
        }


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
