# PD Transfer Backend Benchmark Guide

This guide assumes:

- `$MY/sqshs/sglang_pd_transfer_united.sqsh` already exists.
- `/logs/scripts/pd_bench/fingerprint.sh` already exists inside the container,
  mounted from `$MY/logs/scripts/pd_bench/fingerprint.sh`.
- You compare only runtime backend selection: `BACKEND=nixl` vs
  `BACKEND=mooncake`.
- You can run at most two active Slurm jobs.
- Each job is limited to 2 hours.
- Each node has 8 GPUs.

The guide covers missions 2, 3, and 4 from `set_plan.md`: fingerprints and log
layout, automated benchmark phases, and `srun`/`sbatch` execution.

## Phase Plan

Phase 0 smoke:

```text
nodes: 1
backends: nixl, mooncake
cases: Ptp2_Dtp2_Pdp1_Ddp1
datasets: rand
concurrency: 1 8 32
reps: 1
```

Phase 1 reduced main one-node sweep:

```text
nodes: 1
backends: nixl, mooncake
cases:
  Ptp2_Dtp2_Pdp1_Ddp1
  Ptp4_Dtp4_Pdp1_Ddp1
  Ptp2_Dtp4_Pdp1_Ddp1
  Ptp4_Dtp2_Pdp1_Ddp1
datasets: rand sharegpt radixcache
concurrency: 1 8 32
reps: 5
```

Phase 2 high-concurrency two-node sweep:

```text
nodes: 2
backends: nixl, mooncake
cases:
  Ptp2_Dtp2_Pdp1_Ddp1
  Ptp4_Dtp4_Pdp1_Ddp1
  Ptp2_Dtp4_Pdp1_Ddp1
  Ptp4_Dtp2_Pdp1_Ddp1
datasets: rand sharegpt radixcache
concurrency: 64 128 256
reps: 5
```

Run Phase 2 only after Phase 1 has complete JSONL files, clean fingerprints, and
no recurring server launch failures.

DP attention note: current smoke failures show SGLang crashing during PD prefill
warmup with `dp_size=2` and `enable_dp_attention=True`. Until that is fixed,
keep benchmark cases TP-only with `pdp=1` and `ddp=1`. This still compares the
transfer backends across symmetric TP and asymmetric prefill/decode TP layouts,
without mixing in DP-attention scheduler instability.

## Host Setup

Run this from the SGLang checkout on the host before submitting jobs.

```bash
export HF_TOKEN=<HF_TOKEN>
export MY=<cluster_workspace_root>

mkdir -p "$MY/logs/scripts/pd_bench"
mkdir -p "$MY/logs/pd_bench"
mkdir -p "$MY/logs/pd_bench_tmp"

# fingerprint.sh is assumed to exist, but keep this as a local repair path.
if [ -f nbarzilie/benching/fingerprint.sh ]; then
  cp nbarzilie/benching/fingerprint.sh "$MY/logs/scripts/pd_bench/fingerprint.sh"
fi
chmod +x "$MY/logs/scripts/pd_bench/fingerprint.sh"
```

The mounted paths inside the container will be:

```text
$MY/logs/scripts/pd_bench -> /logs/scripts/pd_bench
$MY/logs/pd_bench         -> /logs/pd_bench
$MY/logs/pd_bench_tmp     -> /logs/pd_bench_tmp
```

## Output Layout

Successful benchmark artifacts:

```text
/logs/pd_bench/<run_id>/
  fingerprint_job.txt
  pd_env.sh
  <backend>/<case>/<dataset>/
    run_meta.json
    fingerprint.txt
    env.txt
    commands.sh
    health_before.json
    health_after.json
    results/
      c<concurrency>_r<rep>.jsonl
    bench_logs/
      c<concurrency>_r<rep>.log
```

Temporary server stdout/stderr:

```text
/logs/pd_bench_tmp/<run_id>/<backend>/<case>/<dataset>/
  prefill.log
  decode.log
  router.log
```

On successful batch completion, temporary server logs are removed unless
`KEEP_SUCCESS_LOGS=1`. On failure, they are preserved at:

```text
/logs/pd_bench/<run_id>/<backend>/<case>/<dataset>/failed_server_logs/
```

## Script: One-Node Matrix Runner

Create this on the host:

```bash
cat > "$MY/logs/scripts/pd_bench/run_pd_backend_matrix.sh" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT="${SCRIPT_ROOT:-/logs/scripts/pd_bench}"
LOG_ROOT="${LOG_ROOT:-/logs/pd_bench}"
TMP_LOG_ROOT="${TMP_LOG_ROOT:-/logs/pd_bench_tmp}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
IMAGE="${IMAGE:-unknown}"
IB_DEV="${IB_DEV:-mlx5_0}"
BACKENDS="${BACKENDS:-nixl mooncake}"
CASES="${CASES:-Ptp2_Dtp2_Pdp1_Ddp1 Ptp4_Dtp4_Pdp1_Ddp1 Ptp2_Dtp4_Pdp1_Ddp1 Ptp4_Dtp2_Pdp1_Ddp1}"
DATASETS="${DATASETS:-rand sharegpt radixcache}"
CONCURRENCY_VALUES="${CONCURRENCY_VALUES:-1 8 32}"
REPS="${REPS:-5}"
MIN_PROMPTS="${MIN_PROMPTS:-100}"
PROMPTS_PER_CONCURRENCY="${PROMPTS_PER_CONCURRENCY:-10}"
OUTPUT_DETAILS="${OUTPUT_DETAILS:-0}"
CACHE_MODE="${CACHE_MODE:-warm}"
TOTAL_GPUS_PER_NODE="${TOTAL_GPUS_PER_NODE:-8}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"

PREFILL_HOST="${PREFILL_HOST:-0.0.0.0}"
PREFILL_PORT="${PREFILL_PORT:-30000}"
PREFILL_URL="${PREFILL_URL:-http://127.0.0.1:30000}"
DECODE_HOST="${DECODE_HOST:-0.0.0.0}"
DECODE_PORT="${DECODE_PORT:-30001}"
DECODE_URL="${DECODE_URL:-http://127.0.0.1:30001}"
ROUTER_HOST="${ROUTER_HOST:-0.0.0.0}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:8000}"

PREFILL_PID=""
DECODE_PID=""
ROUTER_PID=""
OUT_DIR=""
TMP_SERVER_LOG_DIR=""
DATASET=""
BACKEND=""
CASE_NAME=""

write_fingerprint() {
  local out="$1"
  mkdir -p "$(dirname "$out")"
  if [ -x "$SCRIPT_ROOT/fingerprint.sh" ]; then
    "$SCRIPT_ROOT/fingerprint.sh" > "$out" 2>&1
  else
    echo "fingerprint.sh missing at $SCRIPT_ROOT/fingerprint.sh" > "$out"
    env | sort >> "$out"
  fi
}

write_json_meta() {
  local out="$1"
  python3 - "$out" <<'PY'
import json
import os
import sys

def split_ints(name):
    return [int(x) for x in os.environ.get(name, "").split() if x]

out = sys.argv[1]
data = {
    "run_id": os.environ.get("RUN_ID"),
    "backend": os.environ.get("BACKEND"),
    "model": os.environ.get("MODEL"),
    "case_name": os.environ.get("CASE_NAME"),
    "dataset": os.environ.get("DATASET"),
    "image": os.environ.get("IMAGE"),
    "nodes": int(os.environ.get("NODES", "1")),
    "prefill_tp": int(os.environ.get("PREFILL_TP_SIZE", "0")),
    "decode_tp": int(os.environ.get("DECODE_TP_SIZE", "0")),
    "prefill_dp": int(os.environ.get("PREFILL_DP_SIZE", "0")),
    "decode_dp": int(os.environ.get("DECODE_DP_SIZE", "0")),
    "prefill_base_gpu_id": int(os.environ.get("PREFILL_BASE_GPU_ID", "0")),
    "decode_base_gpu_id": int(os.environ.get("DECODE_BASE_GPU_ID", "0")),
    "prefill_resolved_gpu_count": int(os.environ.get("PREFILL_RESOLVED_GPU_COUNT", "0")),
    "decode_resolved_gpu_count": int(os.environ.get("DECODE_RESOLVED_GPU_COUNT", "0")),
    "prefill_gpu_range": os.environ.get("PREFILL_GPU_RANGE"),
    "decode_gpu_range": os.environ.get("DECODE_GPU_RANGE"),
    "total_gpus_per_node": int(os.environ.get("TOTAL_GPUS_PER_NODE", "8")),
    "concurrency_values": split_ints("CONCURRENCY_VALUES"),
    "repetitions": int(os.environ.get("REPS", "0")),
    "min_prompts": int(os.environ.get("MIN_PROMPTS", "0")),
    "prompts_per_concurrency": int(os.environ.get("PROMPTS_PER_CONCURRENCY", "0")),
    "cache_mode": os.environ.get("CACHE_MODE", "warm"),
    "output_details": os.environ.get("OUTPUT_DETAILS", "0") == "1",
}
with open(out, "w") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
PY
}

set_case_env() {
  local case_name="$1"
  export CASE_NAME="$case_name"
  case "$case_name" in
    Ptp2_Dtp2_Pdp1_Ddp1)
      export PREFILL_TP_SIZE=2 DECODE_TP_SIZE=2 PREFILL_DP_SIZE=1 DECODE_DP_SIZE=1
      export PREFILL_BASE_GPU_ID=0 DECODE_BASE_GPU_ID=2
      ;;
    Ptp4_Dtp4_Pdp1_Ddp1)
      export PREFILL_TP_SIZE=4 DECODE_TP_SIZE=4 PREFILL_DP_SIZE=1 DECODE_DP_SIZE=1
      export PREFILL_BASE_GPU_ID=0 DECODE_BASE_GPU_ID=4
      ;;
    Ptp2_Dtp4_Pdp1_Ddp1)
      export PREFILL_TP_SIZE=2 DECODE_TP_SIZE=4 PREFILL_DP_SIZE=1 DECODE_DP_SIZE=1
      export PREFILL_BASE_GPU_ID=0 DECODE_BASE_GPU_ID=2
      ;;
    Ptp4_Dtp2_Pdp1_Ddp1)
      export PREFILL_TP_SIZE=4 DECODE_TP_SIZE=2 PREFILL_DP_SIZE=1 DECODE_DP_SIZE=1
      export PREFILL_BASE_GPU_ID=0 DECODE_BASE_GPU_ID=4
      ;;
    *)
      echo "unknown case: $case_name" >&2
      return 1
      ;;
  esac
}

preflight_gpu_fit() {
  local prefill_gpus="$PREFILL_TP_SIZE"
  local decode_gpus="$DECODE_TP_SIZE"
  local prefill_start="$PREFILL_BASE_GPU_ID"
  local decode_start="$DECODE_BASE_GPU_ID"
  local prefill_end=$((prefill_start + prefill_gpus - 1))
  local decode_end=$((decode_start + decode_gpus - 1))

  if [ $((prefill_end + 1)) -gt "$TOTAL_GPUS_PER_NODE" ]; then
    echo "prefill gpu range exceeds node: ${prefill_start}-${prefill_end}" >&2
    return 1
  fi
  if [ $((decode_end + 1)) -gt "$TOTAL_GPUS_PER_NODE" ]; then
    echo "decode gpu range exceeds node: ${decode_start}-${decode_end}" >&2
    return 1
  fi
  if [ "$prefill_start" -le "$decode_end" ] && [ "$decode_start" -le "$prefill_end" ]; then
    echo "gpu ranges overlap: prefill=${prefill_start}-${prefill_end} decode=${decode_start}-${decode_end}" >&2
    return 1
  fi

  export PREFILL_RESOLVED_GPU_COUNT="$prefill_gpus"
  export DECODE_RESOLVED_GPU_COUNT="$decode_gpus"
  export PREFILL_GPU_RANGE="${prefill_start}-${prefill_end}"
  export DECODE_GPU_RANGE="${decode_start}-${decode_end}"
}

prepare_batch_dirs() {
  OUT_DIR="$LOG_ROOT/$RUN_ID/$BACKEND/$CASE_NAME/$DATASET"
  TMP_SERVER_LOG_DIR="$TMP_LOG_ROOT/$RUN_ID/$BACKEND/$CASE_NAME/$DATASET"
  mkdir -p "$OUT_DIR/results" "$OUT_DIR/bench_logs" "$TMP_SERVER_LOG_DIR"
}

preserve_server_logs_on_failure() {
  local rc="$1"
  if [ -z "${OUT_DIR:-}" ] || [ -z "${TMP_SERVER_LOG_DIR:-}" ]; then
    return 0
  fi
  if [ "$rc" -eq 0 ]; then
    if [ "${KEEP_SUCCESS_LOGS:-0}" != "1" ]; then
      rm -rf "$TMP_SERVER_LOG_DIR"
    fi
    return 0
  fi
  mkdir -p "$OUT_DIR/failed_server_logs"
  cp -a "$TMP_SERVER_LOG_DIR"/. "$OUT_DIR/failed_server_logs/" 2>/dev/null || true
  echo "server logs preserved in $OUT_DIR/failed_server_logs" >&2
}

cleanup_servers() {
  local rc="${1:-$?}"
  set +e
  for pid in ${ROUTER_PID:-} ${DECODE_PID:-} ${PREFILL_PID:-}; do
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      pkill -TERM -P "$pid" || true
      kill -TERM "$pid" || true
    fi
  done
  sleep 5
  for pid in ${ROUTER_PID:-} ${DECODE_PID:-} ${PREFILL_PID:-}; do
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      pkill -KILL -P "$pid" || true
      kill -KILL "$pid" || true
    fi
  done
  preserve_server_logs_on_failure "$rc"
  set -e
}
trap 'cleanup_servers $?' EXIT

wait_health() {
  local url="$1"
  local name="$2"
  local pid="${3:-}"
  local deadline=$((SECONDS + 900))
  until curl -fsS "$url/health" >/dev/null; do
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "$name process exited before health check passed" >&2
      if [ -n "${TMP_SERVER_LOG_DIR:-}" ]; then
        echo "last lines from $name log:" >&2
        tail -200 "$TMP_SERVER_LOG_DIR/$name.log" >&2 || true
      fi
      return 1
    fi
    if [ "$SECONDS" -gt "$deadline" ]; then
      echo "timeout waiting for $name at $url" >&2
      if [ -n "${TMP_SERVER_LOG_DIR:-}" ]; then
        echo "last lines from $name log:" >&2
        tail -200 "$TMP_SERVER_LOG_DIR/$name.log" >&2 || true
      fi
      return 1
    fi
    sleep 5
  done
}

capture_health() {
  local out="$1"
  {
    echo "{"
    echo "\"prefill_health\": $(curl -fsS "$PREFILL_URL/health" 2>/dev/null | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),"
    echo "\"decode_health\": $(curl -fsS "$DECODE_URL/health" 2>/dev/null | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'),"
    echo "\"router_health\": $(curl -fsS "$ROUTER_URL/health" 2>/dev/null | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
    echo "}"
  } > "$out" || true
}

flush_caches() {
  for url in "$PREFILL_URL" "$DECODE_URL" "$ROUTER_URL"; do
    curl -fsS -X POST "$url/flush_cache" >/dev/null 2>&1 || true
  done
}

launch_prefill() {
  local dp_args=()
  if [ "$PREFILL_DP_SIZE" -gt 1 ]; then dp_args=(--enable-dp-attention); fi
  local extra_args=()
  if [ -n "$SERVER_EXTRA_ARGS" ]; then read -r -a extra_args <<< "$SERVER_EXTRA_ARGS"; fi
  sglang serve \
    --model-path "$MODEL" \
    --host "$PREFILL_HOST" \
    --port "$PREFILL_PORT" \
    --base-gpu-id "$PREFILL_BASE_GPU_ID" \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend "$BACKEND" \
    --disaggregation-ib-device "$IB_DEV" \
    --tp-size "$PREFILL_TP_SIZE" \
    --dp-size "$PREFILL_DP_SIZE" \
    "${dp_args[@]}" \
    "${extra_args[@]}" \
    > "$TMP_SERVER_LOG_DIR/prefill.log" 2>&1 &
  PREFILL_PID=$!
}

launch_decode() {
  local dp_args=()
  if [ "$DECODE_DP_SIZE" -gt 1 ]; then dp_args=(--enable-dp-attention); fi
  local extra_args=()
  if [ -n "$SERVER_EXTRA_ARGS" ]; then read -r -a extra_args <<< "$SERVER_EXTRA_ARGS"; fi
  sglang serve \
    --model-path "$MODEL" \
    --host "$DECODE_HOST" \
    --port "$DECODE_PORT" \
    --base-gpu-id "$DECODE_BASE_GPU_ID" \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend "$BACKEND" \
    --disaggregation-ib-device "$IB_DEV" \
    --tp-size "$DECODE_TP_SIZE" \
    --dp-size "$DECODE_DP_SIZE" \
    "${dp_args[@]}" \
    "${extra_args[@]}" \
    > "$TMP_SERVER_LOG_DIR/decode.log" 2>&1 &
  DECODE_PID=$!
}

launch_router() {
  python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "$PREFILL_URL" \
    --decode "$DECODE_URL" \
    --host "$ROUTER_HOST" \
    --port "$ROUTER_PORT" \
    > "$TMP_SERVER_LOG_DIR/router.log" 2>&1 &
  ROUTER_PID=$!
}

dataset_args_for() {
  local dataset="$1"
  local num_prompts="$2"
  case "$dataset" in
    rand)
      DATASET_ARGS=(--dataset-name random --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 --num-prompts "$num_prompts")
      ;;
    sharegpt)
      DATASET_ARGS=(--dataset-name sharegpt --sharegpt-output-len 1024 --num-prompts "$num_prompts")
      ;;
    radixcache)
      local groups=1
      local limit="$num_prompts"
      if [ "$limit" -gt 64 ]; then limit=64; fi
      for ((candidate=limit; candidate>=1; candidate--)); do
        if [ $((num_prompts % candidate)) -eq 0 ]; then
          groups="$candidate"
          break
        fi
      done
      local per_group=$((num_prompts / groups))
      DATASET_ARGS=(--dataset-name generated-shared-prefix --gsp-num-groups "$groups" --gsp-prompts-per-group "$per_group" --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 1024 --gsp-range-ratio 1.0)
      ;;
    *)
      echo "unknown dataset: $dataset" >&2
      return 1
      ;;
  esac
}

run_one_bench() {
  local max_conc="$1"
  local rep="$2"
  local num_prompts=$((max_conc * PROMPTS_PER_CONCURRENCY))
  if [ "$num_prompts" -lt "$MIN_PROMPTS" ]; then num_prompts="$MIN_PROMPTS"; fi

  dataset_args_for "$DATASET" "$num_prompts"

  local tag="${BACKEND}_${CASE_NAME}_${DATASET}_c${max_conc}_r${rep}"
  local jsonl="$OUT_DIR/results/c${max_conc}_r${rep}.jsonl"
  local log="$OUT_DIR/bench_logs/c${max_conc}_r${rep}.log"
  local detail_args=()
  if [ "$OUTPUT_DETAILS" = "1" ]; then detail_args=(--output-details); fi

  python3 -m sglang.bench_serving \
    --backend sglang \
    --base-url "$ROUTER_URL" \
    "${DATASET_ARGS[@]}" \
    --max-concurrency "$max_conc" \
    --pd-separated \
    --warmup-requests 2 \
    "${detail_args[@]}" \
    --output-file "$jsonl" \
    --tag "$tag" \
    > "$log" 2>&1
}

mkdir -p "$LOG_ROOT/$RUN_ID"
write_fingerprint "$LOG_ROOT/$RUN_ID/fingerprint_job.txt"
env | sort > "$LOG_ROOT/$RUN_ID/job_env.txt"

for BACKEND in $BACKENDS; do
  export BACKEND
  for CASE_NAME in $CASES; do
    set_case_env "$CASE_NAME"
    preflight_gpu_fit
    for DATASET in $DATASETS; do
      export DATASET
      prepare_batch_dirs
      write_fingerprint "$OUT_DIR/fingerprint.txt"
      env | sort > "$OUT_DIR/env.txt"
      write_json_meta "$OUT_DIR/run_meta.json"
      {
        echo "BACKEND=$BACKEND CASE_NAME=$CASE_NAME DATASET=$DATASET"
        echo "CONCURRENCY_VALUES=$CONCURRENCY_VALUES REPS=$REPS CACHE_MODE=$CACHE_MODE"
      } > "$OUT_DIR/commands.sh"

      PREFILL_PID=""; DECODE_PID=""; ROUTER_PID=""
      launch_prefill
      wait_health "$PREFILL_URL" prefill "$PREFILL_PID"
      launch_decode
      wait_health "$DECODE_URL" decode "$DECODE_PID"
      launch_router
      wait_health "$ROUTER_URL" router "$ROUTER_PID"
      capture_health "$OUT_DIR/health_before.json"

      for max_conc in $CONCURRENCY_VALUES; do
        if [ "$CACHE_MODE" != "cold" ]; then flush_caches; fi
        for ((rep=1; rep<=REPS; rep++)); do
          if [ "$CACHE_MODE" = "cold" ]; then flush_caches; fi
          run_one_bench "$max_conc" "$rep"
        done
      done

      capture_health "$OUT_DIR/health_after.json"
      cleanup_servers 0
      PREFILL_PID=""; DECODE_PID=""; ROUTER_PID=""
    done
  done
done
BASH

chmod +x "$MY/logs/scripts/pd_bench/run_pd_backend_matrix.sh"
```

## Script: One-Node sbatch Wrapper

Create:

```bash
cat > "$MY/logs/scripts/pd_bench/sbatch_pd_1node.sh" <<'BASH'
#!/usr/bin/env bash
#SBATCH -A network_research_advdev
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH -J pdbench1
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

: "${MY:?Set MY}"
: "${HF_TOKEN:?Set HF_TOKEN}"

IMAGE="$MY/sqshs/sglang_pd_transfer_united.sqsh"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}}"
HOST_RUN_DIR="$MY/logs/pd_bench/$RUN_ID"
mkdir -p "$HOST_RUN_DIR"

srun \
  --container-image="$IMAGE" \
  --container-workdir=/sgl-workspace/sglang \
  --container-mounts="$MY/.cache/huggingface:/root/.cache/huggingface,$MY/logs:/logs,$MY:/host_my" \
  bash -lc "
set -euo pipefail
export PYTHONPATH=/sgl-workspace/sglang/python:\${PYTHONPATH:-}
export HF_TOKEN='$HF_TOKEN'
export RUN_ID='$RUN_ID'
export IMAGE='$IMAGE'
export LOG_ROOT=/logs/pd_bench
mkdir -p /logs/pd_bench/\$RUN_ID

cat > /logs/pd_bench/\$RUN_ID/pd_env.sh <<'ENV'
export MODEL=meta-llama/Llama-3.1-8B-Instruct
export IB_DEV=mlx5_0
export LOG_ROOT=/logs/pd_bench
export TMP_LOG_ROOT=/logs/pd_bench_tmp
export SCRIPT_ROOT=/logs/scripts/pd_bench
export MIN_PROMPTS=100
export PROMPTS_PER_CONCURRENCY=10
export PREFILL_HOST=0.0.0.0
export PREFILL_PORT=30000
export PREFILL_URL=http://127.0.0.1:30000
export DECODE_HOST=0.0.0.0
export DECODE_PORT=30001
export DECODE_URL=http://127.0.0.1:30001
export ROUTER_HOST=0.0.0.0
export ROUTER_PORT=8000
export ROUTER_URL=http://127.0.0.1:8000
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS='{}'
ENV

source /logs/pd_bench/\$RUN_ID/pd_env.sh
export BACKENDS=\"${BACKENDS:-nixl mooncake}\"
export CASES=\"${CASES:-Ptp2_Dtp2_Pdp1_Ddp1 Ptp4_Dtp4_Pdp1_Ddp1 Ptp2_Dtp4_Pdp1_Ddp1 Ptp4_Dtp2_Pdp1_Ddp1}\"
export DATASETS=\"${DATASETS:-rand sharegpt radixcache}\"
export CONCURRENCY_VALUES=\"${CONCURRENCY_VALUES:-1 8 32}\"
export REPS=\"${REPS:-5}\"
export OUTPUT_DETAILS=\"${OUTPUT_DETAILS:-0}\"
export CACHE_MODE=\"${CACHE_MODE:-warm}\"
export SERVER_EXTRA_ARGS=\"${SERVER_EXTRA_ARGS:-}\"
bash /logs/scripts/pd_bench/run_pd_backend_matrix.sh
"

cp "slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$HOST_RUN_DIR/sbatch_stdout.log" 2>/dev/null || true
BASH

chmod +x "$MY/logs/scripts/pd_bench/sbatch_pd_1node.sh"
```

## Optional Script: Two-Node sbatch Placeholder

Phase 2 needs a dedicated two-node runner. Create this wrapper only after
`run_pd_backend_matrix_2node.sh` is implemented and validated.

```bash
cat > "$MY/logs/scripts/pd_bench/sbatch_pd_2node_high_conc.sh" <<'BASH'
#!/usr/bin/env bash
#SBATCH -A network_research_advdev
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --gpus-per-node=8
#SBATCH -t 02:00:00
#SBATCH -J pdbench2
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

: "${MY:?Set MY}"
: "${HF_TOKEN:?Set HF_TOKEN}"

if [ ! -x "$MY/logs/scripts/pd_bench/run_pd_backend_matrix_2node.sh" ]; then
  echo "run_pd_backend_matrix_2node.sh is required before Phase 2" >&2
  exit 1
fi

IMAGE="$MY/sqshs/sglang_pd_transfer_united.sqsh"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_${SLURM_JOB_ID}}"
HOST_RUN_DIR="$MY/logs/pd_bench/$RUN_ID"
mkdir -p "$HOST_RUN_DIR"

srun \
  --container-image="$IMAGE" \
  --container-workdir=/sgl-workspace/sglang \
  --container-mounts="$MY/.cache/huggingface:/root/.cache/huggingface,$MY/logs:/logs,$MY:/host_my" \
  bash -lc "
set -euo pipefail
export PYTHONPATH=/sgl-workspace/sglang/python:\${PYTHONPATH:-}
export HF_TOKEN='$HF_TOKEN'
export RUN_ID='$RUN_ID'
export IMAGE='$IMAGE'
export LOG_ROOT=/logs/pd_bench
export TMP_LOG_ROOT=/logs/pd_bench_tmp
export SCRIPT_ROOT=/logs/scripts/pd_bench
export BACKENDS=\"${BACKENDS:-nixl mooncake}\"
export CASES=\"${CASES:-Ptp2_Dtp2_Pdp1_Ddp1 Ptp4_Dtp4_Pdp1_Ddp1 Ptp2_Dtp4_Pdp1_Ddp1 Ptp4_Dtp2_Pdp1_Ddp1}\"
export DATASETS=\"${DATASETS:-rand sharegpt radixcache}\"
export CONCURRENCY_VALUES=\"${CONCURRENCY_VALUES:-64 128 256}\"
export REPS=\"${REPS:-5}\"
export OUTPUT_DETAILS=\"${OUTPUT_DETAILS:-0}\"
export CACHE_MODE=\"${CACHE_MODE:-warm}\"
bash /logs/scripts/pd_bench/run_pd_backend_matrix_2node.sh
"

cp "slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$HOST_RUN_DIR/sbatch_stdout.log" 2>/dev/null || true
BASH

chmod +x "$MY/logs/scripts/pd_bench/sbatch_pd_2node_high_conc.sh"
```

## Interactive Phase 0 Smoke

Use this when debugging before `sbatch`.

```bash
srun \
  -A network_research_advdev \
  -t 02:00:00 \
  -N 1 \
  -p interactive \
  --gpus-per-node=8 \
  --container-image="$MY/sqshs/sglang_pd_transfer_united.sqsh" \
  --container-workdir=/sgl-workspace/sglang \
  --container-mounts="$MY/.cache/huggingface:/root/.cache/huggingface,$MY/logs:/logs,$MY:/host_my" \
  --pty bash
```

Inside the container:

```bash
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}
export HF_TOKEN=<HF_TOKEN>
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_smoke"
export BACKENDS="nixl mooncake"
export CASES="Ptp2_Dtp2_Pdp1_Ddp1"
export DATASETS="rand"
export CONCURRENCY_VALUES="1 8 32"
export REPS=1
export OUTPUT_DETAILS=1
export CACHE_MODE=warm
export SERVER_EXTRA_ARGS="--attention-backend triton --mem-fraction-static 0.60 --chunked-prefill-size 2048"
bash /logs/scripts/pd_bench/run_pd_backend_matrix.sh
```

## Batch Runs

Phase 0:

```bash
RUN_ID=llama_phase0_smoke \
BACKENDS="nixl mooncake" \
CASES="Ptp2_Dtp2_Pdp1_Ddp1" \
DATASETS="rand" \
CONCURRENCY_VALUES="1 8 32" \
REPS=1 \
OUTPUT_DETAILS=1 \
SERVER_EXTRA_ARGS="--attention-backend triton --mem-fraction-static 0.60 --chunked-prefill-size 2048" \
sbatch "$MY/logs/scripts/pd_bench/sbatch_pd_1node.sh"
```

Phase 1 shard A:

```bash
RUN_ID=llama_phase1_a \
BACKENDS="nixl mooncake" \
CASES="Ptp2_Dtp2_Pdp1_Ddp1 Ptp2_Dtp4_Pdp1_Ddp1" \
DATASETS="rand sharegpt radixcache" \
CONCURRENCY_VALUES="1 8 32" \
REPS=5 \
OUTPUT_DETAILS=0 \
CACHE_MODE=warm \
SERVER_EXTRA_ARGS="--attention-backend triton --mem-fraction-static 0.60 --chunked-prefill-size 2048" \
sbatch "$MY/logs/scripts/pd_bench/sbatch_pd_1node.sh"
```

Phase 1 shard B:

```bash
RUN_ID=llama_phase1_b \
BACKENDS="mooncake nixl" \
CASES="Ptp4_Dtp4_Pdp1_Ddp1 Ptp4_Dtp2_Pdp1_Ddp1" \
DATASETS="rand sharegpt radixcache" \
CONCURRENCY_VALUES="1 8 32" \
REPS=5 \
OUTPUT_DETAILS=0 \
CACHE_MODE=warm \
SERVER_EXTRA_ARGS="--attention-backend triton --mem-fraction-static 0.60 --chunked-prefill-size 2048" \
sbatch "$MY/logs/scripts/pd_bench/sbatch_pd_1node.sh"
```

Cold radix transfer-cost point:

```bash
RUN_ID=llama_phase1_radix_cold \
BACKENDS="nixl mooncake" \
CASES="Ptp2_Dtp2_Pdp1_Ddp1" \
DATASETS="radixcache" \
CONCURRENCY_VALUES="32" \
REPS=5 \
OUTPUT_DETAILS=0 \
CACHE_MODE=cold \
SERVER_EXTRA_ARGS="--attention-backend triton --mem-fraction-static 0.60 --chunked-prefill-size 2048" \
sbatch "$MY/logs/scripts/pd_bench/sbatch_pd_1node.sh"
```

Phase 2 examples, after the two-node runner is ready:

```bash
RUN_ID=llama_phase2_a \
BACKENDS="nixl mooncake" \
CASES="Ptp2_Dtp2_Pdp1_Ddp1 Ptp2_Dtp4_Pdp1_Ddp1" \
DATASETS="rand sharegpt radixcache" \
CONCURRENCY_VALUES="64 128 256" \
REPS=5 \
OUTPUT_DETAILS=0 \
CACHE_MODE=warm \
SERVER_EXTRA_ARGS="--attention-backend triton --mem-fraction-static 0.60 --chunked-prefill-size 2048" \
sbatch "$MY/logs/scripts/pd_bench/sbatch_pd_2node_high_conc.sh"
```

```bash
RUN_ID=llama_phase2_b \
BACKENDS="mooncake nixl" \
CASES="Ptp4_Dtp4_Pdp1_Ddp1 Ptp4_Dtp2_Pdp1_Ddp1" \
DATASETS="rand sharegpt radixcache" \
CONCURRENCY_VALUES="64 128 256" \
REPS=5 \
OUTPUT_DETAILS=0 \
CACHE_MODE=warm \
SERVER_EXTRA_ARGS="--attention-backend triton --mem-fraction-static 0.60 --chunked-prefill-size 2048" \
sbatch "$MY/logs/scripts/pd_bench/sbatch_pd_2node_high_conc.sh"
```

## Known Startup Failure: FlashAttention KV View

If prefill fails during PD warmup with:

```text
RuntimeError: view size is not compatible with input tensor's size and stride
...
flashattention_backend.py
memory_pool.py
k.view(-1, row_dim)
```

the server is dying inside the FlashAttention prefill/KV-cache write path, not
inside the benchmark client. For backend comparison, use the same stable
attention backend for both Mooncake and NIXL:

```bash
export SERVER_EXTRA_ARGS="--attention-backend triton --mem-fraction-static 0.60 --chunked-prefill-size 2048"
```

This changes the absolute serving performance, but the comparison remains fair
because both transfer backends use the same model, topology, dataset, and
attention backend. If you later want default-attention numbers, first verify a
single backend/case can pass PD warmup without this flag.

## Failure Inspection

For a failed batch:

```bash
find "$MY/logs/pd_bench/<run_id>" -path '*failed_server_logs*' -type f -maxdepth 8
```

Common files:

```text
failed_server_logs/prefill.log
failed_server_logs/decode.log
failed_server_logs/router.log
bench_logs/c<concurrency>_r<rep>.log
fingerprint.txt
env.txt
run_meta.json
```

Quick checks:

```bash
grep -R "Traceback\|Error\|ERR\|timeout\|NIXL\|mooncake" "$MY/logs/pd_bench/<run_id>" | head -200
find "$MY/logs/pd_bench/<run_id>" -name '*.jsonl' -size +0
```

## Success Criteria

Phase 0 is successful when:

- Both backends complete all three smoke concurrency points.
- Each backend/case/dataset has `run_meta.json` and `fingerprint.txt`.
- JSONL files are non-empty.
- No `failed_server_logs/` directory is present.

Phase 1 is successful when:

- Both shard jobs complete.
- Every `backend x case x dataset x concurrency` has 5 JSONL files.
- `CACHE_MODE` and `OUTPUT_DETAILS` are recorded in `run_meta.json`.
- Temporary server logs are removed, unless `KEEP_SUCCESS_LOGS=1`.

Phase 2 is successful when:

- The dedicated two-node runner has already passed a small two-node smoke.
- Both nodes write logs to `/logs`.
- Routable prefill/decode/router URLs are recorded.
- Cleanup works on both nodes.
