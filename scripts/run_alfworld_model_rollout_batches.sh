#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/SkillRL}"
MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/skillrl-data/models/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/skillrl-data/rollouts/alfworld_qwen7b_base}"
RUN_DIR="${RUN_DIR:-/root/autodl-tmp/skillrl-runs}"

TARGET_SUCCESS_ENVS="${TARGET_SUCCESS_ENVS:-113}"
TARGET_FAIL_ENVS="${TARGET_FAIL_ENVS:-110}"
MAX_BATCHES="${MAX_BATCHES:-40}"
BATCH_SIZE="${BATCH_SIZE:-100}"
TRAJS_PER_ENV="${TRAJS_PER_ENV:-3}"
MAX_STEPS="${MAX_STEPS:-50}"
HISTORY_TURNS="${HISTORY_TURNS:-8}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
NUM_CPUS_PER_WORKER="${NUM_CPUS_PER_WORKER:-0.05}"

mkdir -p "$OUTPUT_DIR" "$RUN_DIR"
cd "$REPO_DIR"

export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

next_offset() {
  python - "$OUTPUT_DIR" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
max_id = -1
for path in root.glob("env*"):
    if not path.is_dir():
        continue
    match = re.fullmatch(r"env(\d+)", path.name)
    if match:
        max_id = max(max_id, int(match.group(1)))
print(max_id + 1)
PY
}

summary_counts() {
  python - "$OUTPUT_DIR/rollout_summary.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print("0 0 0 0")
    raise SystemExit
data = json.loads(path.read_text(encoding="utf-8"))
envs = data.get("envs", [])
success = sum(1 for item in envs if item.get("status") == "all_success")
fail = sum(1 for item in envs if item.get("status") == "all_fail")
mixed = sum(1 for item in envs if item.get("status") == "mixed")
print(success, fail, mixed, len(envs))
PY
}

summarize() {
  python scripts/summarize_alfworld_rollouts.py \
    --input_dir "$OUTPUT_DIR" \
    --target_success_envs "$TARGET_SUCCESS_ENVS" \
    --target_fail_envs "$TARGET_FAIL_ENVS" \
    --min_trajs_per_env "$TRAJS_PER_ENV"
}

for ((batch = 0; batch < MAX_BATCHES; batch++)); do
  summarize
  read -r success_count fail_count mixed_count env_count < <(summary_counts)
  log "summary envs=$env_count all_success=$success_count all_fail=$fail_count mixed=$mixed_count"

  if (( success_count >= TARGET_SUCCESS_ENVS && fail_count >= TARGET_FAIL_ENVS )); then
    log "targets reached; stopping"
    break
  fi

  offset="$(next_offset)"
  seed="$offset"
  batch_name="$(printf 'batch_offset_%06d' "$offset")"
  batch_log="$RUN_DIR/alfworld_qwen7b_base_${batch_name}_tp${TENSOR_PARALLEL_SIZE}.log"

  log "starting $batch_name env_num=$BATCH_SIZE seed=$seed log=$batch_log"
  ray stop --force >/tmp/skillrl_ray_stop.log 2>&1 || true

  python scripts/generate_alfworld_model_rollouts.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --env_num "$BATCH_SIZE" \
    --env_offset "$offset" \
    --trajs_per_env "$TRAJS_PER_ENV" \
    --max_steps "$MAX_STEPS" \
    --history_turns "$HISTORY_TURNS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --num_cpus_per_worker "$NUM_CPUS_PER_WORKER" \
    --seed "$seed" \
    2>&1 | tee "$batch_log"
done

summarize
log "finished batch loop"
