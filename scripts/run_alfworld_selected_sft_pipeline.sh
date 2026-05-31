#!/usr/bin/env bash
# Continue the ALFWorld SFT-data pipeline from a trajectory-level selected
# processed_trajectories.json file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SFT_DIR="${REPO_DIR}/examples/sft_data_generation"
ARTIFACTS="${SFT_DIR}/audit_artifacts.py"

WORK_DIR="${WORK_DIR:?Set WORK_DIR to the selected ALFWorld run directory}"
RUN_ID="${RUN_ID:-$(basename "${WORK_DIR}")}"
ROLLOUT_DIR="${ROLLOUT_DIR:-/root/autodl-tmp/skillrl-data/rollouts/alfworld_qwen7b_base}"
TRACE_LLM="${TRACE_LLM:-1}"
MEMORY_MODEL="${MEMORY_MODEL:-deepseek-v4-pro}"
AGGREGATE_MODEL="${AGGREGATE_MODEL:-deepseek-v4-pro}"
DISTILL_MODEL="${DISTILL_MODEL:-deepseek-v4-pro}"
TARGET_SUCCESS="${TARGET_SUCCESS:-113}"
TARGET_FAIL="${TARGET_FAIL:-110}"
SELECTED_INPUT="${SELECTED_INPUT:-${WORK_DIR}/01_processed/processed_trajectories.json}"

TRACE_ARGS=()
case "${TRACE_LLM}" in
  1|true|TRUE|yes|YES|on|ON)
    TRACE_ARGS=(--trace_llm)
    ;;
esac

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

stage_dir() {
  printf '%s/%s' "${WORK_DIR}" "$1"
}

if [ ! -f "${SELECTED_INPUT}" ]; then
  echo "Missing selected processed trajectories: ${SELECTED_INPUT}" >&2
  echo "Run scripts/select_alfworld_rollout_trajectories.py first." >&2
  exit 1
fi

mkdir -p \
  "$(stage_dir 00_raw_rollouts)" \
  "$(stage_dir 01_processed)" \
  "$(stage_dir 02_memory)" \
  "$(stage_dir 03_skill_bank)" \
  "$(stage_dir 04_distillation)" \
  "$(stage_dir 05_sft_data)" \
  "$(stage_dir 06_report)"

log "Checking selected trajectory counts"
python - "${SELECTED_INPUT}" "${TARGET_SUCCESS}" "${TARGET_FAIL}" <<'PY'
import json
import sys
from collections import Counter

path, target_success, target_fail = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(path, encoding="utf-8") as f:
    data = json.load(f)
counts = Counter(item.get("type") for item in data)
success = counts.get("all_success", 0)
fail = counts.get("all_fail", 0)
print(f"selected records={len(data)} all_success={success} all_fail={fail}")
if success != target_success or fail != target_fail:
    raise SystemExit(
        f"Expected all_success={target_success}, all_fail={target_fail}; got {success}/{fail}."
    )
PY

log "Writing manifest"
python "${ARTIFACTS}" manifest \
  --run-id "${RUN_ID}" \
  --repo-dir "${REPO_DIR}" \
  --work-dir "${WORK_DIR}" \
  --rollout-dir "${ROLLOUT_DIR}" \
  --memory-model "${MEMORY_MODEL}" \
  --aggregate-model "${AGGREGATE_MODEL}" \
  --distill-model "${DISTILL_MODEL}" \
  --limit "${TARGET_SUCCESS}+${TARGET_FAIL}" \
  --trace-llm "${TRACE_LLM}"

log "[1/5] Sampling selected processed trajectories"
python "${ARTIFACTS}" sample-json \
  --input "${SELECTED_INPUT}" \
  --stage-dir "$(stage_dir 01_processed)" \
  --name processed_trajectories \
  --limit 3

log "[2/5] Removing repeated-loop steps"
python "${SFT_DIR}/preprocess/dedupe_repetitions.py" \
  --input_file "${SELECTED_INPUT}" \
  --output_file "$(stage_dir 01_processed)/processed_trajectories_cleaned.json"

python "${ARTIFACTS}" sample-json \
  --input "$(stage_dir 01_processed)/processed_trajectories_cleaned.json" \
  --stage-dir "$(stage_dir 01_processed)" \
  --name processed_trajectories_cleaned \
  --limit 3

log "[3/5] Generating per-trajectory teacher memories"
python "${SFT_DIR}/skill_memory/generate_memory_alfworld.py" \
  --input_file "$(stage_dir 01_processed)/processed_trajectories_cleaned.json" \
  --output_file "$(stage_dir 02_memory)/generated_memories.json" \
  --model "${MEMORY_MODEL}" \
  --artifact_dir "$(stage_dir 02_memory)" \
  "${TRACE_ARGS[@]}"

python "${ARTIFACTS}" sample-json \
  --input "$(stage_dir 02_memory)/generated_memories.json" \
  --stage-dir "$(stage_dir 02_memory)" \
  --name generated_memories \
  --limit 3

log "[4/5] Aggregating memories into skill bank"
python "${SFT_DIR}/skill_memory/aggregate_skills.py" \
  --input_file "$(stage_dir 02_memory)/generated_memories.json" \
  --output_file "$(stage_dir 03_skill_bank)/skill_bank.json" \
  --env alfworld \
  --model "${AGGREGATE_MODEL}" \
  --artifact_dir "$(stage_dir 03_skill_bank)" \
  "${TRACE_ARGS[@]}"

python "${ARTIFACTS}" skill-bank \
  --input "$(stage_dir 03_skill_bank)/skill_bank.json" \
  --stage-dir "$(stage_dir 03_skill_bank)"

log "[5/5] Distilling successful trajectories and flattening to SFT pairs"
python "${SFT_DIR}/distillation/distill_alfworld.py" \
  --input_file "$(stage_dir 01_processed)/processed_trajectories_cleaned.json" \
  --skill_bank_file "$(stage_dir 03_skill_bank)/skill_bank.json" \
  --output_file "$(stage_dir 04_distillation)/distilled_trajectories.json" \
  --model "${DISTILL_MODEL}" \
  --artifact_dir "$(stage_dir 04_distillation)" \
  "${TRACE_ARGS[@]}"

python "${ARTIFACTS}" sample-json \
  --input "$(stage_dir 04_distillation)/distilled_trajectories.json" \
  --stage-dir "$(stage_dir 04_distillation)" \
  --name distilled_trajectories \
  --limit 3

python "${SFT_DIR}/postprocess/sharegpt_to_pairs.py" \
  --input_file "$(stage_dir 04_distillation)/distilled_trajectories.json" \
  --output_file "$(stage_dir 05_sft_data)/alfworld_sft_data.json"

python "${ARTIFACTS}" sample-json \
  --input "$(stage_dir 05_sft_data)/alfworld_sft_data.json" \
  --stage-dir "$(stage_dir 05_sft_data)" \
  --name alfworld_sft_data \
  --limit 3

log "Writing Chinese summary report"
python "${ARTIFACTS}" report \
  --work-dir "${WORK_DIR}" \
  --run-id "${RUN_ID}" \
  --rollout-dir "${ROLLOUT_DIR}" \
  --output-dir "$(stage_dir 06_report)" \
  --memory-model "${MEMORY_MODEL}" \
  --aggregate-model "${AGGREGATE_MODEL}" \
  --distill-model "${DISTILL_MODEL}"

log "Packaging run artifacts"
PACKAGE_PATH="$(python "${ARTIFACTS}" package --work-dir "${WORK_DIR}" --run-id "${RUN_ID}")"

log "Done"
echo "Work dir: ${WORK_DIR}"
echo "Summary: ${WORK_DIR}/06_report/summary.md"
echo "Package: ${PACKAGE_PATH}"
