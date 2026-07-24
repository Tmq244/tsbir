#!/usr/bin/env bash
# Launch CLIP-pretrained training (10 epochs) on physical GPU 1,2 inside a tmux session.
set -euo pipefail
cd /home/tangmingqiang/cir/tsbir
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tsbir
export CUDA_VISIBLE_DEVICES=1,2
export PYTHONPATH=src:code
export OMP_NUM_THREADS=8
mkdir -p outputs
echo "[$(date --iso-8601=seconds)] starting CLIP-pretrained training on GPU 1,2 (gradient_checkpointing=false)"
torchrun --standalone --nproc-per-node=2 -m tsbir.train \
  --config configs/train_coco_clip_10ep_gpu12.yaml \
  2>&1 | tee outputs/train_clip_10ep_gpu12.log
echo "[$(date --iso-8601=seconds)] training exited with code ${PIPESTATUS[0]}"
