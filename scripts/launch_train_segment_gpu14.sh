#!/usr/bin/env bash
# Launch experiment B (skeleton-segment dropout) on physical GPUs 1 and 4.
set -euo pipefail
cd /home/tangmingqiang/cir/tsbir
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tsbir
export CUDA_VISIBLE_DEVICES=1,4
export PYTHONPATH=src:code
export OMP_NUM_THREADS=8
mkdir -p outputs
echo "[$(date --iso-8601=seconds)] starting segment-dropout B on GPU 1,4"
torchrun --standalone --nproc-per-node=2 -m tsbir.train \
  --config configs/train_coco_clip_segment_10ep_gpu14.yaml \
  2>&1 | tee outputs/train_clip_segment_10ep_gpu14.log
exit_code=${PIPESTATUS[0]}
echo "[$(date --iso-8601=seconds)] training exited with code ${exit_code}"
exit "${exit_code}"
