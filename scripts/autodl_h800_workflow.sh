#!/usr/bin/env bash
# Run the recommended AutoDL H800 setup and optional ALFWorld smoke pipeline.

set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/SkillRL}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/skillrl-data}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/root/autodl-tmp/envs/skillrl}"
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-$(dirname "${CONDA_ENV_PREFIX}")}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${DATA_DIR}/rollouts/alfworld}"
WORK_DIR="${WORK_DIR:-/root/autodl-tmp/skillrl-runs/alfworld_smoke_001}"
LIMIT="${LIMIT:-1}"
TRACE_LLM="${TRACE_LLM:-1}"
INSTALL_GPU_DEPS="${INSTALL_GPU_DEPS:-1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
FLASH_ATTN_REQUIRED="${FLASH_ATTN_REQUIRED:-1}"
MAX_JOBS="${MAX_JOBS:-8}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/autodl_h800_workflow.sh all
  bash scripts/autodl_h800_workflow.sh sync
  bash scripts/autodl_h800_workflow.sh env
  bash scripts/autodl_h800_workflow.sh verify
  bash scripts/autodl_h800_workflow.sh smoke

Modes:
  all     git pull, shell syntax checks, full H800 env install, then verify.
          Set RUN_ALFWORLD_SMOKE=1 to run the paid/API smoke pipeline too.
  sync    git pull and bash -n checks only.
  env     install full H800 environment only.
  verify  import-check the installed environment only.
  smoke   run the auditable ALFWorld pipeline only.

Useful overrides:
  REPO_DIR=/root/autodl-tmp/SkillRL
  DATA_DIR=/root/autodl-tmp/skillrl-data
  CONDA_ENVS_DIR=/root/autodl-tmp/envs
  CONDA_ENV_PREFIX=/root/autodl-tmp/envs/skillrl
  INSTALL_GPU_DEPS=1
  INSTALL_FLASH_ATTN=1
  FLASH_ATTN_REQUIRED=0
  MAX_JOBS=4
  ROLLOUT_DIR=/root/autodl-tmp/skillrl-data/rollouts/alfworld
  WORK_DIR=/root/autodl-tmp/skillrl-runs/alfworld_smoke_001
  LIMIT=1
  TRACE_LLM=1
EOF
}

load_runtime_env() {
  if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "conda not found. AutoDL images usually provide /root/miniconda3." >&2
    exit 1
  fi

  conda activate "${CONDA_ENV_PREFIX}"

  if [ -f "${DATA_DIR}/env.sh" ]; then
    # shellcheck disable=SC1090
    source "${DATA_DIR}/env.sh"
  fi
}

sync_code() {
  if [ ! -d "${REPO_DIR}/.git" ]; then
    echo "Repository not found at ${REPO_DIR}" >&2
    echo "Clone first: git clone https://github.com/wudi299/SkillRL.git ${REPO_DIR}" >&2
    exit 1
  fi

  cd "${REPO_DIR}"
  log "Pulling latest code from origin/main"
  git pull --ff-only origin main

  log "Checking shell script syntax"
  bash -n scripts/prepare_autodl_env.sh
  bash -n scripts/run_alfworld_audit_pipeline.sh
  bash -n scripts/autodl_h800_workflow.sh
}

install_env() {
  cd "${REPO_DIR}"
  log "Installing full H800 environment"
  REPO_DIR="${REPO_DIR}" DATA_DIR="${DATA_DIR}" \
    CONDA_ENVS_DIR="${CONDA_ENVS_DIR}" CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX}" \
    INSTALL_GPU_DEPS="${INSTALL_GPU_DEPS}" INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN}" \
    FLASH_ATTN_REQUIRED="${FLASH_ATTN_REQUIRED}" MAX_JOBS="${MAX_JOBS}" \
    bash scripts/prepare_autodl_env.sh full
}

verify_env() {
  cd "${REPO_DIR}"
  log "Verifying SkillRL environment"
  REPO_DIR="${REPO_DIR}" DATA_DIR="${DATA_DIR}" \
    CONDA_ENVS_DIR="${CONDA_ENVS_DIR}" CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX}" \
    bash scripts/prepare_autodl_env.sh verify
}

run_smoke() {
  if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "OPENAI_API_KEY is required for smoke mode." >&2
    exit 1
  fi

  mkdir -p "${ROLLOUT_DIR}" "$(dirname "${WORK_DIR}")"
  first_rollout="$(find "${ROLLOUT_DIR}" -type f -name '*.txt' -print -quit || true)"
  if [ -z "${first_rollout}" ]; then
    echo "No rollout .txt files found under ${ROLLOUT_DIR}" >&2
    echo "Expected layout: ${ROLLOUT_DIR}/env000/test1.txt" >&2
    exit 1
  fi

  load_runtime_env
  cd "${REPO_DIR}"
  log "Running auditable ALFWorld smoke pipeline"
  export ROLLOUT_DIR WORK_DIR LIMIT TRACE_LLM
  bash scripts/run_alfworld_audit_pipeline.sh
}

case "${1:-all}" in
  all|setup)
    sync_code
    install_env
    verify_env
    if [ "${RUN_ALFWORLD_SMOKE:-0}" = "1" ]; then
      run_smoke
    else
      log "Skipping ALFWorld smoke run. Use RUN_ALFWORLD_SMOKE=1 or run the smoke mode after adding rollouts."
    fi
    ;;
  sync)
    sync_code
    ;;
  env)
    install_env
    ;;
  verify)
    verify_env
    ;;
  smoke)
    run_smoke
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
