#!/usr/bin/env bash
# Wait for two GPUs with >30 GiB free, verify full-batch C2 with a smoke run,
# then launch the formal 10-epoch experiment on those same physical GPUs.
set -euo pipefail

cd /home/tangmingqiang/cir/tsbir
source /home/tangmingqiang/miniconda3/etc/profile.d/conda.sh
conda activate tsbir

readonly minimum_free_mib=30720
readonly poll_seconds=60
readonly stable_seconds=10
readonly config_path=configs/train_coco_clip_humanstyle_consistency_10ep_auto2gpu.yaml
readonly output_dir=outputs/taskformer_coco_clip_humanstyle_consistency_10ep_auto2gpu
readonly wait_log=outputs/wait_train_humanstyle_consistency_auto2gpu.log
readonly train_log=outputs/train_humanstyle_consistency_10ep_auto2gpu.log

mkdir -p outputs
exec > >(tee -a "$wait_log") 2>&1

available_gpus() {
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F, -v threshold="$minimum_free_mib" '
        { gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2) }
        $2 > threshold { print $1 }
      '
}

choose_pair() {
  local -a available=()
  mapfile -t available < <(available_gpus)
  if (( ${#available[@]} >= 2 )); then
    echo "${available[0]},${available[1]}"
  fi
}

echo "[$(date --iso-8601=seconds)] watcher started; waiting for two GPUs with free memory > ${minimum_free_mib} MiB"
while true; do
  first_pair="$(choose_pair)"
  if [[ -n "$first_pair" ]]; then
    echo "[$(date --iso-8601=seconds)] candidate GPUs ${first_pair}; checking stability for ${stable_seconds}s"
    sleep "$stable_seconds"
    second_pair="$(choose_pair)"
    if [[ "$first_pair" == "$second_pair" ]]; then
      selected_pair="$first_pair"
      break
    fi
    echo "[$(date --iso-8601=seconds)] free-memory set changed (${first_pair} -> ${second_pair:-none}); resuming wait"
  else
    free_snapshot="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr '\n' ';')"
    echo "[$(date --iso-8601=seconds)] waiting; free MiB: ${free_snapshot}"
  fi
  sleep "$poll_seconds"
done

if [[ -e "$output_dir/epoch_001_weights.pt" ]]; then
  echo "[$(date --iso-8601=seconds)] refusing to overwrite an existing formal checkpoint in ${output_dir}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$selected_pair"
export PYTHONPATH=src:code
export OMP_NUM_THREADS=8
echo "[$(date --iso-8601=seconds)] selected physical GPUs ${selected_pair}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader

echo "[$(date --iso-8601=seconds)] running full-batch two-step smoke"
if ! torchrun --standalone --nproc-per-node=2 -m tsbir.train \
  --config "$config_path" --smoke; then
  echo "[$(date --iso-8601=seconds)] full-batch smoke failed; formal training was not started" >&2
  exit 3
fi

echo "[$(date --iso-8601=seconds)] smoke passed; starting formal 10-epoch C2 training"
set +e
torchrun --standalone --nproc-per-node=2 -m tsbir.train \
  --config "$config_path" \
  2>&1 | tee "$train_log"
train_status=${PIPESTATUS[0]}
set -e
echo "[$(date --iso-8601=seconds)] formal training exited with code ${train_status}"
exit "$train_status"
