#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="src:code${PYTHONPATH:+:$PYTHONPATH}"
CONFIG="configs/finetune_coco_ddp.yaml"
LOG_DIR="outputs/taskformer_coco_official_finetune"
mkdir -p "$LOG_DIR"

echo "[$(date --iso-8601=seconds)] waiting for COCO extraction" | tee -a "$LOG_DIR/pipeline.log"
until [[ -f data/karpathy/dataset_coco.json ]] \
    && [[ -f data/annotations/instances_train2014.json ]] \
    && [[ -d data/coco/train2014 ]] \
    && [[ -d data/coco/val2014 ]]; do
    sleep 60
done

echo "[$(date --iso-8601=seconds)] preparing Karpathy manifests" | tee -a "$LOG_DIR/pipeline.log"
python -m tsbir.prepare_data --root data 2>&1 | tee "$LOG_DIR/prepare_manifest.log"

echo "[$(date --iso-8601=seconds)] generating PhotoSketch training sketches on GPU 5" | tee -a "$LOG_DIR/pipeline.log"
CUDA_VISIBLE_DEVICES=5 python -m tsbir.generate_sketches \
    --input data/coco/train2014 \
    --input data/coco/val2014 \
    --output data/synthetic_sketches \
    --batch-size 128 \
    --workers 8 2>&1 | tee "$LOG_DIR/generate_sketches.log"

actual_sketches=$(find data/synthetic_sketches -maxdepth 1 -type f -name '*.jpg' | wc -l)
if [[ "$actual_sketches" -ne 123287 ]]; then
    echo "expected 123287 synthetic sketches, found $actual_sketches" >&2
    exit 1
fi

wait_for_gpu() {
    local gpu="$1"
    local threshold_mib=2048
    while true; do
        local used
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
        if [[ "$used" -lt "$threshold_mib" ]]; then
            break
        fi
        echo "[$(date --iso-8601=seconds)] GPU $gpu still uses ${used} MiB; waiting" | tee -a "$LOG_DIR/pipeline.log"
        sleep 60
    done
}

wait_for_gpu 1
wait_for_gpu 5

export CUDA_VISIBLE_DEVICES=1,5
export OMP_NUM_THREADS=8
echo "[$(date --iso-8601=seconds)] starting 2-GPU smoke run, per-GPU batch 48" | tee -a "$LOG_DIR/pipeline.log"
torchrun --standalone --nproc-per-node=2 -m tsbir.train \
    --config "$CONFIG" --batch-size 48 --smoke 2>&1 | tee "$LOG_DIR/smoke_launch.log"

echo "[$(date --iso-8601=seconds)] smoke passed; starting full 2-GPU fine-tuning on GPU 1,5" | tee -a "$LOG_DIR/pipeline.log"
torchrun --standalone --nproc-per-node=2 -m tsbir.train \
    --config "$CONFIG" --batch-size 48 2>&1 | tee "$LOG_DIR/train_launch.log"
