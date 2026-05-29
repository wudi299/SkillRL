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
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-/root/autodl-tmp/envs}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${CONDA_ENVS_DIR}/skillrl}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.6.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.21.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.6.0}"
TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-https://download.pytorch.org/whl/cu124}"
VLLM_VERSION="${VLLM_VERSION:-0.8.5.post1}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
FLASH_ATTN_REQUIRED="${FLASH_ATTN_REQUIRED:-1}"
MAX_JOBS="${MAX_JOBS:-8}"
INSTALL_GPU_DEPS="${INSTALL_GPU_DEPS:-auto}"
REGISTER_CONDA_ENV_NAME="${REGISTER_CONDA_ENV_NAME:-1}"

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
  CONDA_ENVS_DIR=/root/autodl-tmp/envs
  CONDA_ENV_PREFIX=/root/autodl-tmp/envs/skillrl
  REGISTER_CONDA_ENV_NAME=1  Register envs_dir so 'conda activate skillrl' works.
  INSTALL_GPU_DEPS=1       Force vLLM and flash-attn install.
  INSTALL_GPU_DEPS=0       Skip vLLM and flash-attn.
  INSTALL_FLASH_ATTN=0      Skip flash-attn even when GPU deps are enabled.
  FLASH_ATTN_REQUIRED=0     Continue if flash-attn fails, useful when GitHub is unstable.
  VLLM_VERSION=0.8.5.post1
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

register_conda_envs_dir() {
  case "${REGISTER_CONDA_ENV_NAME}" in
    1|true|TRUE|yes|YES|on|ON)
      local envs_dir
      envs_dir="$(dirname "${CONDA_ENV_PREFIX}")"
      if ! conda config --show envs_dirs | grep -Fq "${envs_dir}"; then
        log "Registering Conda envs_dir: ${envs_dir}"
        conda config --append envs_dirs "${envs_dir}" >/dev/null
      fi
      ;;
  esac
}

write_env_file() {
  mkdir -p "${DATA_DIR}/hf-cache"
  cat > "${DATA_DIR}/env.sh" <<EOF
export SKILLRL_REPO_DIR="${REPO_DIR}"
export SKILLRL_DATA_DIR="${DATA_DIR}"
export SKILLRL_CONDA_ENVS_DIR="${CONDA_ENVS_DIR}"
export SKILLRL_CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX}"
export SKILLRL_CONDA_ENV_NAME="$(basename "${CONDA_ENV_PREFIX}")"
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
  register_conda_envs_dir
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

truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
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
  pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_CUDA_INDEX}"

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
  cat > /tmp/skillrl-autodl-constraints.txt <<EOF
torch==${TORCH_VERSION}
torchvision==${TORCHVISION_VERSION}
torchaudio==${TORCHAUDIO_VERSION}
EOF
  pip install "vllm==${VLLM_VERSION}" \
    --constraint /tmp/skillrl-autodl-constraints.txt \
    --extra-index-url "${TORCH_CUDA_INDEX}"

  if ! truthy "${INSTALL_FLASH_ATTN}"; then
    log "Skipping flash-attn because INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN}"
    return
  fi

  log "Installing flash-attn ${FLASH_ATTN_VERSION}"
  set +e
  if [ -n "${FLASH_ATTN_WHEEL_URL}" ]; then
    MAX_JOBS="${MAX_JOBS}" pip install --no-cache-dir "${FLASH_ATTN_WHEEL_URL}"
  else
    MAX_JOBS="${MAX_JOBS}" pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation --no-cache-dir
  fi
  flash_status=$?
  set -e

  if [ "${flash_status}" -ne 0 ]; then
    if truthy "${FLASH_ATTN_REQUIRED}"; then
      echo "flash-attn installation failed. Set FLASH_ATTN_REQUIRED=0 to continue without it." >&2
      exit "${flash_status}"
    fi
    log "flash-attn installation failed, continuing because FLASH_ATTN_REQUIRED=${FLASH_ATTN_REQUIRED}"
  fi
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
