#!/usr/bin/env bash
# Launch the formal C2 run on physical GPUs 5 and 6 after a full-batch smoke.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=5,6
export PYTHONPATH="src:code${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=8
mkdir -p outputs
echo "[$(date --iso-8601=seconds)] starting formal human-style consistency training on GPU 5,6"
set +e
torchrun --standalone --nproc-per-node=2 -m tsbir.train \
  --config configs/train_coco_clip_humanstyle_consistency_10ep_auto2gpu.yaml \
  2>&1 | tee outputs/train_humanstyle_consistency_10ep_auto2gpu.log
train_status=${PIPESTATUS[0]}
set -e
echo "[$(date --iso-8601=seconds)] formal training exited with code ${train_status}"
exit "$train_status"
