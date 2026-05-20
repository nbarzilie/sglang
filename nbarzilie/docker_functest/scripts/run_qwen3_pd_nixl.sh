#!/usr/bin/env bash
set -euo pipefail

source /usr/local/bin/sglang_functest_common.sh

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-32B}"
HOST="${HOST:-0.0.0.0}"
PREFILL_PORT="${PREFILL_PORT:-30100}"
DECODE_PORT="${DECODE_PORT:-30200}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-30500}"
TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
KEEP_ALIVE="${KEEP_ALIVE:-1}"

export SGLANG_DISAGGREGATION_NIXL_BACKEND="${SGLANG_DISAGGREGATION_NIXL_BACKEND:-UCX}"
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS="${SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS:-{}}"

check_python_deps

IB_ARGS=()
if [ -n "${DISAGG_IB_DEVICES:-}" ]; then
  IB_ARGS=(--disaggregation-ib-device "${DISAGG_IB_DEVICES}")
fi

append_extra_args "${PREFILL_EXTRA_ARGS:-}"
PREFILL_EXTRA=("${EXTRA_ARGS[@]}")
append_extra_args "${DECODE_EXTRA_ARGS:-}"
DECODE_EXTRA=("${EXTRA_ARGS[@]}")

log "Starting PD NIXL Qwen3 run"
log "Model: ${MODEL_PATH}"
log "Prefill: GPU 0, port ${PREFILL_PORT}; Decode: GPU 1, port ${DECODE_PORT}; Router: port ${ROUTER_PORT}"

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PREFILL_PORT}" \
  --tp 1 \
  --trust-remote-code \
  --disaggregation-mode prefill \
  --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
  --disaggregation-transfer-backend nixl \
  "${IB_ARGS[@]}" \
  "${PREFILL_EXTRA[@]}" \
  >"${LOG_DIR}/pd_prefill.log" 2>&1 &
PIDS+=("$!")

python3 -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${DECODE_PORT}" \
  --tp 1 \
  --base-gpu-id 1 \
  --trust-remote-code \
  --disaggregation-mode decode \
  --disaggregation-bootstrap-port "${BOOTSTRAP_PORT}" \
  --disaggregation-transfer-backend nixl \
  "${IB_ARGS[@]}" \
  "${DECODE_EXTRA[@]}" \
  >"${LOG_DIR}/pd_decode.log" 2>&1 &
PIDS+=("$!")

wait_http_ready "http://127.0.0.1:${PREFILL_PORT}/health" "pd-prefill" "${TIMEOUT}"
wait_http_ready "http://127.0.0.1:${DECODE_PORT}/health" "pd-decode" "${TIMEOUT}"

python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --mini-lb \
  --prefill "http://127.0.0.1:${PREFILL_PORT}" \
  --decode "http://127.0.0.1:${DECODE_PORT}" \
  --host "${HOST}" \
  --port "${ROUTER_PORT}" \
  >"${LOG_DIR}/pd_router.log" 2>&1 &
PIDS+=("$!")

wait_http_ready "http://127.0.0.1:${ROUTER_PORT}/health" "pd-router" "${TIMEOUT}"
post_generate_smoke "http://127.0.0.1:${ROUTER_PORT}" "The capital of France is" 16

log "PD NIXL Qwen3 smoke passed."
log "Logs: ${LOG_DIR}/pd_prefill.log ${LOG_DIR}/pd_decode.log ${LOG_DIR}/pd_router.log"

if [ "${KEEP_ALIVE}" = "1" ]; then
  log "KEEP_ALIVE=1; router stays up at http://127.0.0.1:${ROUTER_PORT} until Ctrl-C"
  wait
fi
