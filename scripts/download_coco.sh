#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-data}"
RAW="$ROOT/raw"
mkdir -p "$RAW"

download() {
    local url="$1"
    local output="$2"
    if [[ -s "$output" ]]; then
        echo "[skip] $output already exists"
        return
    fi
    wget -c "$url" -O "$output"
}

download "http://images.cocodataset.org/zips/train2014.zip" "$RAW/train2014.zip" &
pid_train=$!
download "http://images.cocodataset.org/zips/val2014.zip" "$RAW/val2014.zip" &
pid_val=$!
download "http://images.cocodataset.org/annotations/annotations_trainval2014.zip" "$RAW/annotations_trainval2014.zip" &
pid_ann=$!
download "https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip" "$RAW/caption_datasets.zip" &
pid_caps=$!

if [[ -s sketch.zip ]]; then
    cp -n sketch.zip "$RAW/sketch.zip"
else
    download "https://patsorn.me/projects/tsbir/data/sketch.zip" "$RAW/sketch.zip"
fi

wait "$pid_train" "$pid_val" "$pid_ann" "$pid_caps"

mkdir -p "$ROOT/coco" "$ROOT/annotations" "$ROOT/karpathy" "$ROOT/human_sketches"
unzip -q -n "$RAW/train2014.zip" -d "$ROOT/coco"
unzip -q -n "$RAW/val2014.zip" -d "$ROOT/coco"
unzip -q -n "$RAW/annotations_trainval2014.zip" -d "$ROOT"
unzip -q -n "$RAW/caption_datasets.zip" -d "$ROOT/karpathy"
unzip -q -n "$RAW/sketch.zip" -d "$ROOT/human_sketches"

echo "[done] COCO 2014, annotations, Karpathy splits, and TASK-former test sketches are ready under $ROOT"
