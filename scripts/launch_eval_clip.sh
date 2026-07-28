#!/usr/bin/env bash
# Evaluate CLIP-pretrained TSBIR (epoch 10 / last) on the 5000-image test set.
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=2     # GPU 2 currently free; GPU 1 is contended
export PYTHONPATH="src:code${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p outputs
echo "[$(date --iso-8601=seconds)] starting evaluation on GPU 2"
python -m tsbir.evaluate \
  --checkpoint outputs/taskformer_coco_clip_10ep_gpu12_bs96/last_weights.pt \
  --manifest data/processed/manifests/test.jsonl \
  --output outputs/test_clip_10ep_gpu12_last \
  2>&1 | tee outputs/eval_clip_10ep_gpu12_last.log
echo "[$(date --iso-8601=seconds)] evaluation exited with code ${PIPESTATUS[0]}"
