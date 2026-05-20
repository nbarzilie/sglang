#!/usr/bin/env bash
set -euo pipefail

source /usr/local/bin/sglang_functest_common.sh

MODEL_PATH="${CPU_MODEL_PATH:-Qwen/Qwen3-0.6B}"
HOST="${HOST:-0.0.0.0}"
PORT="${CPU_HEALTH_PORT:-31000}"
CPU_TP="${CPU_TP:-1}"
TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
KEEP_ALIVE="${KEEP_ALIVE:-0}"

export SGLANG_USE_CPU_ENGINE=1

check_python_deps

append_extra_args "${CPU_EXTRA_ARGS:-}"

log "Starting CPU SGLang health server on port ${PORT}"
log "Model: ${MODEL_PATH}"
python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --load-format dummy \
  --device cpu \
  --host "${HOST}" \
  --port "${PORT}" \
  --tp "${CPU_TP}" \
  --trust-remote-code \
  --disable-overlap-schedule \
  --context-length 512 \
  "${EXTRA_ARGS[@]}" \
  >"${LOG_DIR}/cpu_health_server.log" 2>&1 &
PIDS+=("$!")

wait_http_ready "http://127.0.0.1:${PORT}/health" "cpu-health-server" "${TIMEOUT}"
post_generate_smoke "http://127.0.0.1:${PORT}" "hello" 4

log "CPU SGLang health check passed. Logs: ${LOG_DIR}/cpu_health_server.log"

if [ "${KEEP_ALIVE}" = "1" ]; then
  log "KEEP_ALIVE=1; server stays up until Ctrl-C"
  wait "${PIDS[0]}"
fi
