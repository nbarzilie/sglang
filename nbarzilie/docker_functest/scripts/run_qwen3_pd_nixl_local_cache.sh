#!/usr/bin/env bash
set -euo pipefail

source /usr/local/bin/sglang_functest_common.sh

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
HOST="${HOST:-0.0.0.0}"
PREFILL_PORT="${PREFILL_PORT:-30100}"
DECODE_PORT="${DECODE_PORT:-30200}"
ROUTER_PORT="${ROUTER_PORT:-30000}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-30500}"
TIMEOUT="${SERVER_READY_TIMEOUT:-1200}"
KEEP_ALIVE="${KEEP_ALIVE:-0}"

export HF_HOME
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export SGLANG_DISAGGREGATION_NIXL_BACKEND="${SGLANG_DISAGGREGATION_NIXL_BACKEND:-UCX}"
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS="${SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS:-{}}"

# Keep generated caches off the container writable overlay.
export TMPDIR="${TMPDIR:-${LOG_DIR}/tmp}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${LOG_DIR}/xdg-cache}"
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-${LOG_DIR}/sglang-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${LOG_DIR}/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${LOG_DIR}/torchinductor-cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${LOG_DIR}/nv-cache}"
mkdir -p \
  "${TMPDIR}" \
  "${XDG_CACHE_HOME}" \
  "${SGLANG_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}"

resolve_hf_snapshot() {
  local model="$1"

  if [ -d "${model}" ]; then
    printf '%s\n' "${model}"
    return 0
  fi

  local cache_name
  cache_name="models--${model//\//--}"
  local snapshots_dir="${HF_HOME}/hub/${cache_name}/snapshots"

  if [ ! -d "${snapshots_dir}" ]; then
    echo "Missing HF snapshots directory: ${snapshots_dir}" >&2
    echo "Mount your downloaded cache to ${HF_HOME}, or pass MODEL_PATH as a local snapshot path." >&2
    return 1
  fi

  local snapshot
  while IFS= read -r snapshot; do
    if [ -f "${snapshot}/config.json" ] && \
       find "${snapshot}" -maxdepth 1 -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) | grep -q .; then
      printf '%s\n' "${snapshot}"
      return 0
    fi
  done < <(find "${snapshots_dir}" -mindepth 1 -maxdepth 1 -type d | sort -r)

  echo "No complete local HF snapshot found for ${model} under ${snapshots_dir}" >&2
  echo "Expected config.json plus at least one *.safetensors, *.bin, or *.pt file." >&2
  find "${snapshots_dir}" -maxdepth 2 -type f | sed 's#^#  #' | head -80 >&2 || true
  return 1
}

MODEL_LOCAL_PATH="$(resolve_hf_snapshot "${MODEL_PATH}")"

check_python_deps

python3 - "${MODEL_LOCAL_PATH}" <<'PY'
import os
import sys

model_path = sys.argv[1]
required = ["config.json"]
missing = [name for name in required if not os.path.exists(os.path.join(model_path, name))]
weights = [
    name for name in os.listdir(model_path)
    if name.endswith((".safetensors", ".bin", ".pt"))
]
if missing or not weights:
    raise SystemExit(
        f"Incomplete model snapshot: {model_path}; missing={missing}; weights={len(weights)}"
    )
print(f"Using local model snapshot: {model_path}")
print(f"Found {len(weights)} weight file(s)")
PY

IB_ARGS=()
if [ -n "${DISAGG_IB_DEVICES:-}" ]; then
  IB_ARGS=(--disaggregation-ib-device "${DISAGG_IB_DEVICES}")
fi

DEFAULT_EXTRA_ARGS="${DEFAULT_EXTRA_ARGS:---disable-cuda-graph --mem-fraction-static 0.55}"
append_extra_args "${DEFAULT_EXTRA_ARGS} ${PREFILL_EXTRA_ARGS:-}"
PREFILL_EXTRA=("${EXTRA_ARGS[@]}")
append_extra_args "${DEFAULT_EXTRA_ARGS} ${DECODE_EXTRA_ARGS:-}"
DECODE_EXTRA=("${EXTRA_ARGS[@]}")

log "Starting offline local-cache PD NIXL Qwen3 run"
log "Requested model: ${MODEL_PATH}"
log "Resolved model path: ${MODEL_LOCAL_PATH}"
log "HF_HOME: ${HF_HOME}"
log "Prefill: GPU 0, port ${PREFILL_PORT}; Decode: GPU 1, port ${DECODE_PORT}; Router: port ${ROUTER_PORT}"

python3 -m sglang.launch_server \
  --model-path "${MODEL_LOCAL_PATH}" \
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
  --model-path "${MODEL_LOCAL_PATH}" \
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

log "Offline local-cache PD NIXL Qwen3 smoke passed."
log "Logs: ${LOG_DIR}/pd_prefill.log ${LOG_DIR}/pd_decode.log ${LOG_DIR}/pd_router.log"

if [ "${KEEP_ALIVE}" = "1" ]; then
  log "KEEP_ALIVE=1; router stays up at http://127.0.0.1:${ROUTER_PORT} until Ctrl-C"
  wait
fi
