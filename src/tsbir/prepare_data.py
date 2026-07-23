from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_category_ids(annotation_files: list[Path]) -> dict[int, list[int]]:
    labels: dict[int, set[int]] = defaultdict(set)
    for path in annotation_files:
        with path.open() as handle:
            annotation = json.load(handle)
        for item in annotation["annotations"]:
            labels[int(item["image_id"])].add(int(item["category_id"]))
    return {image_id: sorted(category_ids) for image_id, category_ids in labels.items()}


def prepare(root: Path) -> None:
    karpathy_path = root / "karpathy" / "dataset_coco.json"
    annotation_files = [
        root / "annotations" / "instances_train2014.json",
        root / "annotations" / "instances_val2014.json",
    ]
    for required in [karpathy_path, *annotation_files]:
        if not required.is_file():
            raise FileNotFoundError(required)

    with karpathy_path.open() as handle:
        karpathy = json.load(handle)
    category_ids = load_category_ids(annotation_files)

    output_dir = root / "processed" / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {split: output_dir / f"{split}.jsonl" for split in ("train", "val", "test")}
    handles = {split: path.open("w") for split, path in output_paths.items()}
    image_counts: Counter[str] = Counter()
    caption_counts: Counter[str] = Counter()
    missing_images: list[str] = []
    missing_human_sketches: list[str] = []

    try:
        for item in karpathy["images"]:
            source_split = item["split"]
            split = "train" if source_split in {"train", "restval"} else source_split
            if split not in handles:
                continue

            filename = item["filename"]
            image_path = root / "coco" / item["filepath"] / filename
            if not image_path.is_file():
                missing_images.append(str(image_path))

            human_sketch = root / "human_sketches" / "sketch_jpg" / filename
            if split == "test" and not human_sketch.is_file():
                missing_human_sketches.append(str(human_sketch))

            captions = [sentence["raw"].strip() for sentence in item["sentences"]]
            record = {
                "coco_id": int(item["cocoid"]),
                "filename": filename,
                "image": str(image_path),
                "synthetic_sketch": str(root / "synthetic_sketches" / filename),
                "human_sketch": str(human_sketch) if split == "test" else None,
                "captions": captions,
                "category_ids": category_ids.get(int(item["cocoid"]), []),
            }
            handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            image_counts[split] += 1
            caption_counts[split] += len(captions)
    finally:
        for handle in handles.values():
            handle.close()

    expected_images = {"train": 113287, "val": 5000, "test": 5000}
    expected_test_sketches = 5000
    if dict(image_counts) != expected_images:
        raise RuntimeError(f"unexpected Karpathy split counts: {dict(image_counts)}")
    if missing_images:
        raise RuntimeError(f"{len(missing_images)} COCO images are missing; first: {missing_images[0]}")
    if len(missing_human_sketches) != 0:
        raise RuntimeError(
            f"{len(missing_human_sketches)} test sketches are missing; first: {missing_human_sketches[0]}"
        )
    actual_test_sketches = len(list((root / "human_sketches" / "sketch_jpg").glob("*.jpg")))
    if actual_test_sketches != expected_test_sketches:
        raise RuntimeError(f"expected 5000 test sketches, found {actual_test_sketches}")

    stats = {
        "images": dict(image_counts),
        "captions": dict(caption_counts),
        "test_sketches": actual_test_sketches,
        "classes": 90,
    }
    with (output_dir / "stats.json").open("w") as handle:
        json.dump(stats, handle, indent=2)
    print(json.dumps(stats, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    args = parser.parse_args()
    prepare(args.root)


if __name__ == "__main__":
    main()
