#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=4,5
export PYTHONPATH="src:code${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8

mkdir -p outputs
log=outputs/train_frozen_gate_grid_3ep_gpu45.log
echo "[$(date --iso-8601=seconds)] starting frozen-encoder mixed-only gate training on GPU 4,5"
torchrun --standalone --nproc-per-node=2 -m tsbir.train_gate \
  --config configs/train_coco_frozen_gate_grid_gpu45.yaml \
  2>&1 | tee "$log"
status=${PIPESTATUS[0]}
echo "[$(date --iso-8601=seconds)] training exited with code $status"
exit "$status"
