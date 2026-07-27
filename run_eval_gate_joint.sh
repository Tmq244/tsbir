#!/bin/bash
set -e
cd /home/tangmingqiang/cir/tsbir
source /home/tangmingqiang/miniconda3/etc/profile.d/conda.sh
conda activate tsbir
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=src:code
python -m tsbir.evaluate \
  --checkpoint outputs/taskformer_coco_clip_gate_joint_10ep_gpu12/last_weights.pt \
  --manifest data/processed/manifests/test.jsonl \
  --output outputs/test_gate_joint_10ep_gpu12_last \
  --workers 8
