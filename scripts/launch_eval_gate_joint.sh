#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export PYTHONPATH="src:code${PYTHONPATH:+:$PYTHONPATH}"

python -m tsbir.evaluate \
  --checkpoint outputs/taskformer_coco_clip_gate_joint_10ep_gpu12/last_weights.pt \
  --manifest data/processed/manifests/test.jsonl \
  --output outputs/test_gate_joint_10ep_gpu12_last \
  --workers 8
