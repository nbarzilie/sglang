#!/usr/bin/env bash
set -euo pipefail

source /usr/local/bin/sglang_functest_common.sh

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
HOST="${HOST:-0.0.0.0}"
PORT="${REGULAR_PORT:-30000}"
TP="${TP:-1}"
TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
KEEP_ALIVE="${KEEP_ALIVE:-1}"

export SGLANG_DISAGGREGATION_NIXL_BACKEND="${SGLANG_DISAGGREGATION_NIXL_BACKEND:-UCX}"
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS="${SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS:-{}}"

check_python_deps

append_extra_args "${REGULAR_EXTRA_ARGS:-}"

log "Starting regular single-server SGLang Qwen3 run"
log "Model: ${MODEL_PATH}"
log "NIXL import/backend env is verified, but regular non-PD serving does not use the NIXL transfer backend."
python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp "${TP}" \
  --trust-remote-code \
  "${EXTRA_ARGS[@]}" \
  >"${LOG_DIR}/regular_qwen3.log" 2>&1 &
PIDS+=("$!")

wait_http_ready "http://127.0.0.1:${PORT}/health" "regular-qwen3" "${TIMEOUT}"
post_generate_smoke "http://127.0.0.1:${PORT}" "The capital of France is" 16

log "Regular Qwen3 smoke passed. Logs: ${LOG_DIR}/regular_qwen3.log"

if [ "${KEEP_ALIVE}" = "1" ]; then
  log "KEEP_ALIVE=1; server stays up at http://127.0.0.1:${PORT} until Ctrl-C"
  wait "${PIDS[0]}"
fi
