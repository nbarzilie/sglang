#!/usr/bin/env bash
set -euo pipefail

export SGLANG_SOURCE_DIR="${SGLANG_SOURCE_DIR:-/workspace/sglang}"
export PYTHONPATH="${SGLANG_SOURCE_DIR}/python:${PYTHONPATH:-}"

LOG_DIR="${LOG_DIR:-/tmp/sglang-functest-logs}"
mkdir -p "${LOG_DIR}"

PIDS=()

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

kill_tree() {
  local pid="$1"
  python3 - "${pid}" <<'PY' || kill "${pid}" >/dev/null 2>&1 || true
import sys

from sglang.srt.utils import kill_process_tree

kill_process_tree(int(sys.argv[1]), wait_timeout=30)
PY
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [ "${#PIDS[@]}" -gt 0 ]; then
    log "Cleaning up ${#PIDS[@]} background process(es)"
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill_tree "${pid}"
      fi
    done
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill -9 "${pid}" >/dev/null 2>&1 || true
      fi
    done
  fi
  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

check_python_deps() {
  python3 - <<'PY'
import importlib
import torch

checks = [
    "sglang",
    "sglang_router.launch_router",
    "nixl._api",
    "requests",
]
for name in checks:
    importlib.import_module(name)

print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_devices={torch.cuda.device_count()}")
print("sglang, router, nixl imports OK")
PY
}

wait_http_ready() {
  local url="$1"
  local name="$2"
  local timeout="${3:-900}"
  local start
  start="$(date +%s)"
  log "Waiting for ${name}: ${url}"
  while true; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      log "${name} is ready"
      return 0
    fi
    local now
    now="$(date +%s)"
    if [ $(( now - start )) -ge "${timeout}" ]; then
      log "Timed out waiting for ${name}"
      return 1
    fi
    sleep 2
  done
}

post_generate_smoke() {
  local base_url="$1"
  local prompt="${2:-The capital of France is}"
  local max_new_tokens="${3:-16}"
  log "Sending /generate smoke request to ${base_url}"
  curl -fsS \
    -H 'Content-Type: application/json' \
    -d "{\"text\":\"${prompt}\",\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":${max_new_tokens}}}" \
    "${base_url}/generate"
  printf '\n'
}

append_extra_args() {
  local extra="${1:-}"
  if [ -n "${extra}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=(${extra})
  else
    EXTRA_ARGS=()
  fi
}
