#!/usr/bin/env bash
# Wait for the formal C2 epoch-10 checkpoint, then evaluate human and
# synthetic sketches on the released training GPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:code${PYTHONPATH:+:$PYTHONPATH}"

readonly checkpoint=outputs/taskformer_coco_clip_humanstyle_consistency_10ep_auto2gpu/epoch_010_weights.pt
readonly config_pattern='tsbir.train --config configs/train_coco_clip_humanstyle_consistency_10ep_auto2gpu.yaml'
readonly watcher_log=outputs/wait_eval_humanstyle_epoch10.log
exec > >(tee -a "$watcher_log") 2>&1

echo "[$(date --iso-8601=seconds)] waiting for ${checkpoint}"
while [[ ! -s "$checkpoint" ]]; do
  if ! tmux has-session -t tsbir_c2_gpu56 2>/dev/null; then
    echo "[$(date --iso-8601=seconds)] training tmux ended before epoch 10 was saved" >&2
    exit 2
  fi
  sleep 60
done

echo "[$(date --iso-8601=seconds)] epoch 10 found; waiting for training processes to release GPUs"
while pgrep -f "$config_pattern" >/dev/null; do
  sleep 10
done

echo "[$(date --iso-8601=seconds)] starting final human/synthetic evaluations"
CUDA_VISIBLE_DEVICES=5 python -m tsbir.evaluate \
  --checkpoint "$checkpoint" \
  --manifest data/processed/manifests/test.jsonl \
  --output outputs/test_clip_humanstyle_epoch010_human \
  --sketch-field human_sketch \
  > outputs/eval_humanstyle_epoch010_human.log 2>&1 &
human_pid=$!

CUDA_VISIBLE_DEVICES=6 python -m tsbir.evaluate \
  --checkpoint "$checkpoint" \
  --manifest data/processed/manifests/test.jsonl \
  --output outputs/test_clip_humanstyle_epoch010_synthetic \
  --sketch-field synthetic_sketch \
  > outputs/eval_humanstyle_epoch010_synthetic.log 2>&1 &
synthetic_pid=$!

human_status=0
synthetic_status=0
wait "$human_pid" || human_status=$?
wait "$synthetic_pid" || synthetic_status=$?
echo "[$(date --iso-8601=seconds)] evaluations finished: human=${human_status}, synthetic=${synthetic_status}"
if (( human_status != 0 || synthetic_status != 0 )); then
  exit 3
fi
