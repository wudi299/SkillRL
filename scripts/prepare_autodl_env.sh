#!/usr/bin/env bash
# Prepare a SkillRL Conda environment on AutoDL.
#
# Current no-GPU instance:
#   bash scripts/prepare_autodl_env.sh base
#
# H800/GPU instance:
#   INSTALL_GPU_DEPS=1 bash scripts/prepare_autodl_env.sh full

set -euo pipefail

REPO_DIR="${REPO_DIR:-/root/autodl-tmp/SkillRL}"
DATA_DIR="${DATA_DIR:-/root/autodl-tmp/skillrl-data}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/root/autodl-tmp/envs/skillrl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
MAX_JOBS="${MAX_JOBS:-8}"
INSTALL_GPU_DEPS="${INSTALL_GPU_DEPS:-auto}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/prepare_autodl_env.sh base
  bash scripts/prepare_autodl_env.sh full
  bash scripts/prepare_autodl_env.sh verify

Modes:
  base    Install Conda env, PyTorch cu124, project dependencies except flash-attn.
  full    Run base, then install vLLM and flash-attn if GPU deps are enabled.
  verify  Import-check installed packages and print CUDA status.

Useful overrides:
  REPO_DIR=/root/autodl-tmp/SkillRL
  DATA_DIR=/root/autodl-tmp/skillrl-data
  CONDA_ENV_PREFIX=/root/autodl-tmp/envs/skillrl
  INSTALL_GPU_DEPS=1       Force vLLM and flash-attn install.
  INSTALL_GPU_DEPS=0       Skip vLLM and flash-attn.
  VLLM_VERSION=0.11.0
  FLASH_ATTN_VERSION=2.7.4.post1
EOF
}

load_conda() {
  if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
  elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  else
    echo "conda not found. AutoDL images usually provide /root/miniconda3." >&2
    exit 1
  fi
}

write_env_file() {
  mkdir -p "${DATA_DIR}/hf-cache"
  cat > "${DATA_DIR}/env.sh" <<EOF
export SKILLRL_REPO_DIR="${REPO_DIR}"
export SKILLRL_DATA_DIR="${DATA_DIR}"
export HF_HOME="${DATA_DIR}/hf-cache"
export HUGGINGFACE_HUB_CACHE="${DATA_DIR}/hf-cache/hub"
export TRANSFORMERS_CACHE="${DATA_DIR}/hf-cache/transformers"
export ALFWORLD_DATA="${DATA_DIR}/alfworld"
export SEARCH_R1_DIR="${DATA_DIR}/searchR1"
export SEARCH_R1_PROCESSED_DIR="${DATA_DIR}/searchR1_processed_direct"
EOF
}

activate_env() {
  load_conda
  if [ ! -d "${CONDA_ENV_PREFIX}" ]; then
    log "Creating Conda environment: ${CONDA_ENV_PREFIX}"
    conda create -p "${CONDA_ENV_PREFIX}" "python=${PYTHON_VERSION}" -y
  fi
  conda activate "${CONDA_ENV_PREFIX}"
}

has_gpu() {
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1
}

should_install_gpu_deps() {
  case "${INSTALL_GPU_DEPS}" in
    1|true|TRUE|yes|YES)
      return 0
      ;;
    0|false|FALSE|no|NO)
      return 1
      ;;
    auto)
      has_gpu
      return
      ;;
    *)
      echo "Invalid INSTALL_GPU_DEPS=${INSTALL_GPU_DEPS}; use auto, 1, or 0." >&2
      exit 1
      ;;
  esac
}

install_base() {
  if [ ! -d "${REPO_DIR}" ]; then
    echo "Repository not found at ${REPO_DIR}" >&2
    echo "Clone it first: git clone https://github.com/wudi299/SkillRL.git ${REPO_DIR}" >&2
    exit 1
  fi

  write_env_file
  # shellcheck disable=SC1090
  source "${DATA_DIR}/env.sh"
  activate_env

  cd "${REPO_DIR}"
  log "Installing build tooling"
  pip install -U pip setuptools wheel ninja packaging

  log "Installing PyTorch ${TORCH_VERSION} from ${TORCH_CUDA_INDEX}"
  pip install "torch==${TORCH_VERSION}" torchvision torchaudio --index-url "${TORCH_CUDA_INDEX}"

  log "Installing SkillRL requirements except flash-attn"
  grep -v -E '^[[:space:]]*flash-attn([=<> ].*)?$' requirements.txt > /tmp/skillrl-requirements-no-flash.txt
  pip install -r /tmp/skillrl-requirements-no-flash.txt

  log "Installing SkillRL editable package and lightweight extras"
  pip install openai huggingface_hub
  pip install -e .
}

install_gpu_deps() {
  # shellcheck disable=SC1090
  [ -f "${DATA_DIR}/env.sh" ] && source "${DATA_DIR}/env.sh"
  activate_env
  cd "${REPO_DIR}"

  if ! should_install_gpu_deps; then
    log "Skipping vLLM and flash-attn. Set INSTALL_GPU_DEPS=1 on the H800 instance to force installation."
    return
  fi

  log "Installing vLLM ${VLLM_VERSION}"
  pip install "vllm==${VLLM_VERSION}"

  log "Installing flash-attn ${FLASH_ATTN_VERSION}"
  MAX_JOBS="${MAX_JOBS}" pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation --no-cache-dir
}

verify_env() {
  # shellcheck disable=SC1090
  [ -f "${DATA_DIR}/env.sh" ] && source "${DATA_DIR}/env.sh"
  activate_env
  cd "${REPO_DIR}"

  python - <<'PY'
import importlib.util

mods = [
    "torch",
    "verl",
    "agent_system",
    "ray",
    "transformers",
    "datasets",
    "hydra",
    "peft",
    "tensordict",
    "wandb",
    "qwen_vl_utils",
    "vllm",
    "flash_attn",
]

for name in mods:
    print(f"{name}: {bool(importlib.util.find_spec(name))}")

import torch
print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_version:", torch.version.cuda)
print("gpu_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu_name:", torch.cuda.get_device_name(0))
PY

  test -f "${REPO_DIR}/memory_data/alfworld/claude_style_skills.json"
  test -f "${REPO_DIR}/memory_data/webshop/claude_style_skills.json"
  test -f "${REPO_DIR}/memory_data/search/claude_style_skills_search.json"
  log "Verification command completed"
}

case "${1:-}" in
  base)
    install_base
    verify_env
    ;;
  full)
    install_base
    install_gpu_deps
    verify_env
    ;;
  verify)
    verify_env
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
