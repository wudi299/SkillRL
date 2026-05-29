#!/usr/bin/env bash
# Run a small auditable ALFWorld SFT-data pipeline.
#
# Required:
#   export OPENAI_API_KEY=...
#   export ROLLOUT_DIR=/root/autodl-tmp/skillrl-data/rollouts/alfworld
#
# Optional:
#   export WORK_DIR=/root/autodl-tmp/skillrl-runs/alfworld_smoke_001
#   LIMIT=3 TRACE_LLM=1 bash scripts/run_alfworld_audit_pipeline.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SFT_DIR="${REPO_DIR}/examples/sft_data_generation"
ARTIFACTS="${SFT_DIR}/audit_artifacts.py"

ROLLOUT_DIR="${ROLLOUT_DIR:?Set ROLLOUT_DIR to your ALFWorld rollout txt directory}"
LIMIT="${LIMIT:-3}"
TRACE_LLM="${TRACE_LLM:-1}"
MEMORY_MODEL="${MEMORY_MODEL:-gpt-4o}"
AGGREGATE_MODEL="${AGGREGATE_MODEL:-gpt-4o}"
DISTILL_MODEL="${DISTILL_MODEL:-o3}"

if [ -z "${WORK_DIR:-}" ]; then
  RUN_ID="${RUN_ID:-alfworld_$(date '+%Y%m%d_%H%M%S')}"
  WORK_DIR="/root/autodl-tmp/skillrl-runs/${RUN_ID}"
else
  RUN_ID="${RUN_ID:-$(basename "${WORK_DIR}")}"
fi

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

mkdir -p \
  "$(stage_dir 00_raw_rollouts)" \
  "$(stage_dir 01_processed)" \
  "$(stage_dir 02_memory)" \
  "$(stage_dir 03_skill_bank)" \
  "$(stage_dir 04_distillation)" \
  "$(stage_dir 05_sft_data)" \
  "$(stage_dir 06_report)"

log "Writing manifest"
python "${ARTIFACTS}" manifest \
  --run-id "${RUN_ID}" \
  --repo-dir "${REPO_DIR}" \
  --work-dir "${WORK_DIR}" \
  --rollout-dir "${ROLLOUT_DIR}" \
  --memory-model "${MEMORY_MODEL}" \
  --aggregate-model "${AGGREGATE_MODEL}" \
  --distill-model "${DISTILL_MODEL}" \
  --limit "${LIMIT}" \
  --trace-llm "${TRACE_LLM}"

log "Capturing raw rollout samples"
python "${ARTIFACTS}" raw \
  --rollout-dir "${ROLLOUT_DIR}" \
  --stage-dir "$(stage_dir 00_raw_rollouts)" \
  --limit "${LIMIT}" \
  --sample-limit 3

log "[1/6] Parsing ALFWorld rollouts"
python "${SFT_DIR}/preprocess/parse_alfworld.py" \
  --input_dir "${ROLLOUT_DIR}" \
  --output_file "$(stage_dir 01_processed)/processed_trajectories.json" \
  --n_success_envs "${LIMIT}" \
  --n_fail_envs "${LIMIT}" \
  --n_trajs_per_env 1 \
  --min_trajs_per_env 1

python "${ARTIFACTS}" sample-json \
  --input "$(stage_dir 01_processed)/processed_trajectories.json" \
  --stage-dir "$(stage_dir 01_processed)" \
  --name processed_trajectories \
  --limit 3

log "[2/6] Removing repeated-loop steps"
python "${SFT_DIR}/preprocess/dedupe_repetitions.py" \
  --input_file "$(stage_dir 01_processed)/processed_trajectories.json" \
  --output_file "$(stage_dir 01_processed)/processed_trajectories_cleaned.json"

python "${ARTIFACTS}" sample-json \
  --input "$(stage_dir 01_processed)/processed_trajectories_cleaned.json" \
  --stage-dir "$(stage_dir 01_processed)" \
  --name processed_trajectories_cleaned \
  --limit 3

log "[3/6] Generating per-trajectory teacher memories"
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

log "[4/6] Aggregating memories into skill bank"
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

log "[5/6] Distilling ShareGPT trajectories"
python "${SFT_DIR}/distillation/distill_alfworld.py" \
  --input_file "$(stage_dir 01_processed)/processed_trajectories_cleaned.json" \
  --skill_bank_file "$(stage_dir 03_skill_bank)/skill_bank.json" \
  --output_file "$(stage_dir 04_distillation)/distilled_trajectories.json" \
  --model "${DISTILL_MODEL}" \
  --limit "${LIMIT}" \
  --artifact_dir "$(stage_dir 04_distillation)" \
  "${TRACE_ARGS[@]}"

python "${ARTIFACTS}" sample-json \
  --input "$(stage_dir 04_distillation)/distilled_trajectories.json" \
  --stage-dir "$(stage_dir 04_distillation)" \
  --name distilled_trajectories \
  --limit 3

log "[6/6] Flattening to cold-start SFT pairs"
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
