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
            sketch = stroke_dropout(sketch)

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
