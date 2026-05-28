#!/usr/bin/env bash
# Prepare SkillRL code, data, and CPU-verifiable dependencies on a no-GPU Linux server.
#
# Typical host usage:
#   bash scripts/prepare_nogpu_server.sh host
#
# Container-only usage, if you already entered the container:
#   bash scripts/prepare_nogpu_server.sh container

set -euo pipefail

IMAGE="${IMAGE:-whatcanyousee/verl:ngc-cu124-vllm0.8.5-sglang0.4.6-mcore0.12.0-te2.3}"
REPO_DIR="${REPO_DIR:-/workspace/SkillRL}"
DATA_DIR="${DATA_DIR:-/data/skillrl}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${DATA_DIR}/hf-cache}"
CONTAINER_NAME="${CONTAINER_NAME:-skillrl-prep}"
WEBSHOP_DATA_SIZE="${WEBSHOP_DATA_SIZE:-all}"
SKIP_WEBSHOP="${SKIP_WEBSHOP:-0}"
SKIP_SEARCH_INDEX="${SKIP_SEARCH_INDEX:-0}"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/prepare_nogpu_server.sh host
  bash scripts/prepare_nogpu_server.sh container
  bash scripts/prepare_nogpu_server.sh verify

Environment overrides:
  IMAGE=...                 Docker image to use.
  REPO_DIR=/workspace/SkillRL
  DATA_DIR=/data/skillrl
  HF_CACHE_DIR=/data/skillrl/hf-cache
  CONTAINER_NAME=skillrl-prep
  WEBSHOP_DATA_SIZE=all     Use "small" if Google Drive/full data is unstable.
  SKIP_WEBSHOP=1            Skip WebShop setup.
  SKIP_SEARCH_INDEX=1       Skip large Search index/corpus download.
EOF
}

write_env_file() {
  mkdir -p "${DATA_DIR}"
  cat > "${DATA_DIR}/env.sh" <<EOF
export SKILLRL_REPO_DIR="${REPO_DIR}"
export SKILLRL_DATA_DIR="${DATA_DIR}"
export HF_HOME="${HF_CACHE_DIR}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export ALFWORLD_DATA="${DATA_DIR}/alfworld"
export SEARCH_R1_DIR="${DATA_DIR}/searchR1"
export SEARCH_R1_PROCESSED_DIR="${DATA_DIR}/searchR1_processed_direct"
EOF
}

prepare_host() {
  if ! command -v docker >/dev/null 2>&1; then
    echo "docker command not found. Install Docker first on the no-GPU server." >&2
    exit 1
  fi
  if [ ! -d "${REPO_DIR}" ]; then
    echo "REPO_DIR does not exist: ${REPO_DIR}" >&2
    echo "Sync or clone this repository there first, or set REPO_DIR=/path/to/SkillRL." >&2
    exit 1
  fi

  mkdir -p "${DATA_DIR}" "${HF_CACHE_DIR}"
  write_env_file

  log "Pulling Docker image: ${IMAGE}"
  docker pull "${IMAGE}"

  if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    echo "Container ${CONTAINER_NAME} already exists." >&2
    echo "Remove it manually or set CONTAINER_NAME=skillrl-prep-2." >&2
    exit 1
  fi

  log "Running container preparation without GPU"
  docker run --name "${CONTAINER_NAME}" --net=host --shm-size=10g --cap-add=SYS_ADMIN \
    -v "${REPO_DIR}:${REPO_DIR}" \
    -v "${DATA_DIR}:${DATA_DIR}" \
    -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
    -e REPO_DIR="${REPO_DIR}" \
    -e DATA_DIR="${DATA_DIR}" \
    -e HF_CACHE_DIR="${HF_CACHE_DIR}" \
    -e WEBSHOP_DATA_SIZE="${WEBSHOP_DATA_SIZE}" \
    -e SKIP_WEBSHOP="${SKIP_WEBSHOP}" \
    -e SKIP_SEARCH_INDEX="${SKIP_SEARCH_INDEX}" \
    "${IMAGE}" \
    bash "${REPO_DIR}/scripts/prepare_nogpu_server.sh" container

  log "Preparation container exited. Data and caches are in mounted directories."
  log "Open an interactive shell with the same mounts when you need to inspect or retry:"
  cat <<EOF
docker run --rm -it --net=host --shm-size=10g --cap-add=SYS_ADMIN \\
  -v "${REPO_DIR}:${REPO_DIR}" \\
  -v "${DATA_DIR}:${DATA_DIR}" \\
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \\
  -e REPO_DIR="${REPO_DIR}" \\
  -e DATA_DIR="${DATA_DIR}" \\
  -e HF_CACHE_DIR="${HF_CACHE_DIR}" \\
  "${IMAGE}" bash
EOF
}

prepare_alfworld_cache_link() {
  mkdir -p "${DATA_DIR}/alfworld" /root/.cache
  if [ -e /root/.cache/alfworld ] && [ ! -L /root/.cache/alfworld ]; then
    log "Using existing /root/.cache/alfworld; ALFWorld files may not be under ${DATA_DIR}/alfworld"
  elif [ ! -e /root/.cache/alfworld ]; then
    ln -s "${DATA_DIR}/alfworld" /root/.cache/alfworld
  fi
}

ensure_java() {
  if command -v java >/dev/null 2>&1; then
    java -version
    return
  fi
  if command -v apt-get >/dev/null 2>&1; then
    log "Java not found; installing default-jre for WebShop setup"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y default-jre
  else
    echo "Java is required for WebShop setup, and apt-get is not available in this container." >&2
    exit 1
  fi
}

prepare_container() {
  if [ ! -d "${REPO_DIR}" ]; then
    echo "Repository not mounted at ${REPO_DIR}" >&2
    exit 1
  fi

  mkdir -p "${DATA_DIR}" "${HF_CACHE_DIR}" "${HF_CACHE_DIR}/hub" "${HF_CACHE_DIR}/transformers"
  write_env_file
  # shellcheck disable=SC1090
  source "${DATA_DIR}/env.sh"

  cd "${REPO_DIR}"

  log "Installing SkillRL editable package without dependency churn"
  pip install --no-deps -e .
  pip install openai huggingface_hub

  log "Preparing ALFWorld"
  pip install alfworld gymnasium==0.29.1 stable-baselines3==2.6.0
  prepare_alfworld_cache_link
  ALFWORLD_DATA="${DATA_DIR}/alfworld" alfworld-download -f
  python -c "import alfworld, gymnasium; print('alfworld ok')"

  if [ "${SKIP_WEBSHOP}" = "1" ]; then
    log "Skipping WebShop because SKIP_WEBSHOP=1"
  else
    ensure_java
    log "Preparing WebShop with -d ${WEBSHOP_DATA_SIZE}"
    cd "${REPO_DIR}/agent_system/environments/env_package/webshop/webshop"
    if ! ./setup.sh -d "${WEBSHOP_DATA_SIZE}"; then
      log "WebShop setup failed. Retry with WEBSHOP_DATA_SIZE=small or manually place failed Google Drive files into the WebShop data directory."
      exit 1
    fi
    python run_web_agent_text_env.py
    cd "${REPO_DIR}"
  fi

  log "Preparing Search environment and SearchR1 processed parquet"
  cd "${REPO_DIR}"
  pip install -e agent_system/environments/env_package/search/third_party
  pip install gym==0.26.2 faiss-cpu
  python examples/data_preprocess/preprocess_search_r1_dataset.py \
    --local_dir "${DATA_DIR}/searchR1_processed_direct"

  if [ "${SKIP_SEARCH_INDEX}" = "1" ]; then
    log "Skipping large Search index/corpus download because SKIP_SEARCH_INDEX=1"
  else
    log "Downloading Search index and corpus"
    mkdir -p "${DATA_DIR}/searchR1"
    python examples/search/searchr1_download.py \
      --local_dir "${DATA_DIR}/searchR1"
    cd "${DATA_DIR}/searchR1"
    if [ -f part_aa ] && [ -f part_ab ]; then
      cat part_aa part_ab > e5_Flat.index
    fi
    if [ -f wiki-18.jsonl.gz ] && [ ! -f wiki-18.jsonl ]; then
      gunzip -k wiki-18.jsonl.gz
    fi
    cd "${REPO_DIR}"
  fi

  verify_container
  log "No-GPU preparation complete"
}

verify_container() {
  # shellcheck disable=SC1090
  [ -f "${DATA_DIR}/env.sh" ] && source "${DATA_DIR}/env.sh"
  cd "${REPO_DIR}"

  log "Verifying repo and task dependencies"
  python -c "import verl; import agent_system; print('repo imports ok')"
  python -c "import alfworld; print('alfworld import ok')"
  python -c "import gym; print('gym import ok')"

  log "Verifying SkillBank files"
  test -f "${REPO_DIR}/memory_data/alfworld/claude_style_skills.json"
  test -f "${REPO_DIR}/memory_data/webshop/claude_style_skills.json"
  test -f "${REPO_DIR}/memory_data/search/claude_style_skills_search.json"

  log "Verifying prepared Search data"
  test -f "${DATA_DIR}/searchR1_processed_direct/train.parquet"
  if [ "${SKIP_SEARCH_INDEX}" != "1" ]; then
    test -f "${DATA_DIR}/searchR1/e5_Flat.index"
    test -f "${DATA_DIR}/searchR1/wiki-18.jsonl"
  fi

  log "Verification complete"
}

case "${1:-}" in
  host)
    prepare_host
    ;;
  container)
    prepare_container
    ;;
  verify)
    verify_container
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
