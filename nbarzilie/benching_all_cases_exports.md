# PD Benchmark All Cases Export Runbook

This file keeps one reusable command flow for all requested PD benchmark cases.
Pick one backend, one model, one dataset, and one TP/DP case by exporting values,
then run the same four terminals.

## Slurm Cluster Initiation

NIXL image:

```bash
export HF_TOKEN=<HF_TOKEN>

srun \
  -A network_research_advdev \
  -t 02:00:00 \
  -N 1 \
  -p interactive \
  --gpus-per-node=8 \
  --container-image=$MY/sqshs/sglang_nixl.sqsh \
  --container-workdir=/sgl-workspace/sglang \
  --container-mounts=$MY/.cache/huggingface:/root/.cache/huggingface,$MY/logs:/logs \
  --pty bash
```

For Mooncake, replace:

```bash
--container-image=$MY/sqshs/sglang_nixl.sqsh
```

with:

```bash
--container-image=./sglang_mooncake.sqsh
```

Use four terminals or four `tmux` panes inside the same allocation. Do not run
NIXL and Mooncake on the same ports at the same time.

## General Exports

Use these values either directly in a terminal or inside the shared `/logs/pd_env.sh` file below.

```bash
export IB_DEV=mlx5_0
export LOG_DIR=/logs

export PREFILL_HOST=0.0.0.0
export PREFILL_URL=http://127.0.0.1:30000
export PREFILL_PORT=30000
export PREFILL_BASE_GPU_ID=0

export DECODE_HOST=0.0.0.0
export DECODE_URL=http://127.0.0.1:30001
export DECODE_PORT=30001
export DECODE_BASE_GPU_ID=2

export ROUTER_HOST=0.0.0.0
export ROUTER_URL=http://127.0.0.1:8000
export ROUTER_PORT=8000
```

For two-node runs, replace `127.0.0.1` with routable node IPs or hostnames and
usually set both `PREFILL_BASE_GPU_ID=0` and `DECODE_BASE_GPU_ID=0`.

## Recommended tmux Export Strategy

Each `tmux` pane or window has its own shell environment. Exports from one pane
do not automatically appear in the other panes. Create one shared environment
file with `nano`, then source it in every pane.

Create the file once:

```bash
nano /logs/pd_env.sh
```

Paste the general exports, one backend export, one model export, and one TP/DP
case export into that file. Example:

```bash
export BACKEND=nixl
export MODEL=meta-llama/Llama-3.1-8B-Instruct
export IB_DEV=mlx5_0
export LOG_DIR=/logs
export CONCURRENCY_VALUES="1 2 4 8 16 32 64 128 256"
export MIN_PROMPTS=100
export PROMPTS_PER_CONCURRENCY=10

export CASE_NAME=Ptp2_Dtp2_Pdp2_Ddp2
export PREFILL_TP_SIZE=2
export DECODE_TP_SIZE=2
export PREFILL_DP_SIZE=2
export DECODE_DP_SIZE=2

export PREFILL_HOST=0.0.0.0
export PREFILL_URL=http://127.0.0.1:30000
export PREFILL_PORT=30000
export PREFILL_BASE_GPU_ID=0

export DECODE_HOST=0.0.0.0
export DECODE_URL=http://127.0.0.1:30001
export DECODE_PORT=30001
export DECODE_BASE_GPU_ID=2

export ROUTER_HOST=0.0.0.0
export ROUTER_URL=http://127.0.0.1:8000
export ROUTER_PORT=8000

export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS='{}'
```

In `nano`, save with `Ctrl-O`, press `Enter`, then exit with `Ctrl-X`.

Then run this in every tmux pane before starting prefill, decode, router, or
benchmark:

```bash
source /logs/pd_env.sh
echo "$BACKEND $CASE_NAME $MODEL"
```

When changing cases, edit the same file again:

```bash
nano /logs/pd_env.sh
```

Then re-source it in every pane.


## TP/DP Cases

SGLang requires `tp_size % dp_size == 0` when `--enable-dp-attention` is used.
Do not launch a side where DP is larger than TP or where TP is not divisible by
DP. Otherwise the server can fail at startup with:

```text
AssertionError: assert self.tp_size % self.dp_size == 0
```

For that reason, two originally requested shapes are corrected here:

```text
Ptp1_Dtp1_Pdp4_Ddp4 -> Ptp4_Dtp4_Pdp4_Ddp4
Ptp2_Dtp2_Pdp1_Ddp4 -> Ptp2_Dtp4_Pdp1_Ddp4
```

In this DP-attention layout, the GPU count for each side is the side's `tp_size`;
`dp_size` divides those ranks into DP attention groups. For one-node PD runs,
`DECODE_BASE_GPU_ID` is set after the prefill TP range.

Before launching, verify the active values in each tmux pane:

```bash
source /logs/pd_env.sh
printf "prefill: tp=%s dp=%s base_gpu=%s\n" "$PREFILL_TP_SIZE" "$PREFILL_DP_SIZE" "$PREFILL_BASE_GPU_ID"
printf "decode:  tp=%s dp=%s base_gpu=%s\n" "$DECODE_TP_SIZE" "$DECODE_DP_SIZE" "$DECODE_BASE_GPU_ID"
```

You can also dump the relevant environment in sorted form:

```bash
env | grep -E "^(BACKEND|MODEL|CASE_NAME|PREFILL_|DECODE_|ROUTER_|IB_DEV|LOG_DIR|CONCURRENCY_VALUES|MIN_PROMPTS|PROMPTS_PER_CONCURRENCY|SGLANG_DISAGGREGATION_)" | sort
```


| Case | Prefill TP | Decode TP | Prefill DP | Decode DP | Total GPUs in this PD layout |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Ptp4_Dtp4_Pdp4_Ddp4` | 4 | 4 | 4 | 4 | 8 |
| `Ptp2_Dtp4_Pdp1_Ddp1` | 2 | 4 | 1 | 1 | 6 |
| `Ptp4_Dtp2_Pdp1_Ddp1` | 4 | 2 | 1 | 1 | 6 |
| `Ptp2_Dtp4_Pdp1_Ddp4` | 2 | 4 | 1 | 4 | 6 |
| `Ptp2_Dtp2_Pdp2_Ddp2` | 2 | 2 | 2 | 2 | 4 |
| `Ptp4_Dtp4_Pdp1_Ddp1` | 4 | 4 | 1 | 1 | 8 |


Select exactly one case:

```bash
# Case 1
export CASE_NAME=Ptp4_Dtp4_Pdp4_Ddp4
export PREFILL_TP_SIZE=4
export DECODE_TP_SIZE=4
export PREFILL_DP_SIZE=4
export DECODE_DP_SIZE=4
export PREFILL_BASE_GPU_ID=0
export DECODE_BASE_GPU_ID=4
```

```bash
# Case 2
export CASE_NAME=Ptp2_Dtp4_Pdp1_Ddp1
export PREFILL_TP_SIZE=2
export DECODE_TP_SIZE=4
export PREFILL_DP_SIZE=1
export DECODE_DP_SIZE=1
export PREFILL_BASE_GPU_ID=0
export DECODE_BASE_GPU_ID=2
```

```bash
# Case 3
export CASE_NAME=Ptp4_Dtp2_Pdp1_Ddp1
export PREFILL_TP_SIZE=4
export DECODE_TP_SIZE=2
export PREFILL_DP_SIZE=1
export DECODE_DP_SIZE=1
export PREFILL_BASE_GPU_ID=0
export DECODE_BASE_GPU_ID=4
```

```bash
# Case 4
export CASE_NAME=Ptp2_Dtp4_Pdp1_Ddp4
export PREFILL_TP_SIZE=2
export DECODE_TP_SIZE=4
export PREFILL_DP_SIZE=1
export DECODE_DP_SIZE=4
export PREFILL_BASE_GPU_ID=0
export DECODE_BASE_GPU_ID=2
```

```bash
# Case 5
export CASE_NAME=Ptp2_Dtp2_Pdp2_Ddp2
export PREFILL_TP_SIZE=2
export DECODE_TP_SIZE=2
export PREFILL_DP_SIZE=2
export DECODE_DP_SIZE=2
export PREFILL_BASE_GPU_ID=0
export DECODE_BASE_GPU_ID=2
```

```bash
# Case 6
export CASE_NAME=Ptp4_Dtp4_Pdp1_Ddp1
export PREFILL_TP_SIZE=4
export DECODE_TP_SIZE=4
export PREFILL_DP_SIZE=1
export DECODE_DP_SIZE=1
export PREFILL_BASE_GPU_ID=0
export DECODE_BASE_GPU_ID=4
```

## Backend Options

Run each case once with each backend:

```bash
export BACKEND=nixl
```

```bash
export BACKEND=mooncake
```

For NIXL, the default plugin is usually UCX:

```bash
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS='{}'
```

## Dataset Options

The benchmark script accepts `rand`, `sharegpt`, or `radixcache`.

| Script value | `bench_serving` dataset | Path/export |
| --- | --- | --- |
| `rand` | `random` | no dataset file needed |
| `sharegpt` | `sharegpt` | no dataset path; selected by `--dataset-name sharegpt` |
| `radixcache` | `generated-shared-prefix` | no dataset file needed |

Do not set `--dataset-path` for these standard profiles. Select the workload by `--dataset-name` only.

Important: `generated-shared-prefix` does not use `--num-prompts` to size the dataset. Its request count is `--gsp-num-groups * --gsp-prompts-per-group`. The benchmark script below recomputes those two GSP arguments inside the loop so the printed `num-prompts` value matches the actual request count.

## Model Options

Select one model:

```bash
# Llama 8B instruct
export MODEL=meta-llama/Llama-3.1-8B-Instruct
```

```bash
# Qwen 32B
export MODEL=Qwen/Qwen3-32B
```

```bash
# DeepSeek small MoE
export MODEL=deepseek-ai/DeepSeek-V2-Lite-Chat
```

## Remaining Fill-In Exports



Set the concurrency sweep and prompt-count scaling. The benchmark script below computes `num_prompts = max(MIN_PROMPTS, PROMPTS_PER_CONCURRENCY * max_concurrency)` for each concurrency point. This keeps low-concurrency runs short while giving high-concurrency runs enough requests for stable batching:

```bash
export CONCURRENCY_VALUES="1 2 4 8 16 32 64 128 256"
export MIN_PROMPTS=100
export PROMPTS_PER_CONCURRENCY=10
```

## Terminal 1: Prefill

```bash
PREFILL_DP_ARGS=()
if [ "$PREFILL_DP_SIZE" -gt 1 ]; then
  PREFILL_DP_ARGS=(--enable-dp-attention)
fi

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
  "${PREFILL_DP_ARGS[@]}"
```

## Terminal 2: Decode

```bash
DECODE_DP_ARGS=()
if [ "$DECODE_DP_SIZE" -gt 1 ]; then
  DECODE_DP_ARGS=(--enable-dp-attention)
fi

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
  "${DECODE_DP_ARGS[@]}"
```

## Terminal 3: Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill "$PREFILL_URL" \
  --decode "$DECODE_URL" \
  --host "$ROUTER_HOST" \
  --port "$ROUTER_PORT"
```

## Terminal 4: Benchmark

Create `bench_pd_case.sh`:

```bash
#!/bin/bash
set -euo pipefail

DATASET="${1:-rand}"  # rand, sharegpt, radixcache
BACKEND=$BACKEND
CASE_NAME="${CASE_NAME:?Set CASE_NAME from the TP/DP case exports}"
LOG_DIR="${LOG_DIR:-/logs}"
ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:8000}"

mkdir -p "$LOG_DIR"

case "$DATASET" in
  rand)
    DATASET_LABEL=rand
    ;;
  sharegpt)
    DATASET_LABEL=sharegpt
    ;;
  radixcache)
    DATASET_LABEL=radixcache
    ;;
  *)
    echo "Usage: $0 rand|sharegpt|radixcache"
    exit 1
    ;;
esac

LOG_FILE="$LOG_DIR/${BACKEND}_${CASE_NAME}_${DATASET_LABEL}.log"
: > "$LOG_FILE"
CONCURRENCY_VALUES="${CONCURRENCY_VALUES:-1 2 4 8 16 32 64 128 256}"
MIN_PROMPTS="${MIN_PROMPTS:-100}"
PROMPTS_PER_CONCURRENCY="${PROMPTS_PER_CONCURRENCY:-10}"

for max_conc in $CONCURRENCY_VALUES
do
  num_prompts=$((max_conc * PROMPTS_PER_CONCURRENCY))
  if [ "$num_prompts" -lt "$MIN_PROMPTS" ]; then
    num_prompts="$MIN_PROMPTS"
  fi

  case "$DATASET" in
    rand)
      DATASET_ARGS=(--dataset-name random --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 --num-prompts "$num_prompts")
      ;;
    sharegpt)
      DATASET_ARGS=(--dataset-name sharegpt --sharegpt-output-len 1024 --num-prompts "$num_prompts")
      ;;
    radixcache)
      GSP_NUM_GROUPS=1
      GSP_GROUP_LIMIT="$num_prompts"
      if [ "$GSP_GROUP_LIMIT" -gt 64 ]; then
        GSP_GROUP_LIMIT=64
      fi
      for ((candidate=GSP_GROUP_LIMIT; candidate>=1; candidate--)); do
        if [ $((num_prompts % candidate)) -eq 0 ]; then
          GSP_NUM_GROUPS="$candidate"
          break
        fi
      done
      GSP_PROMPTS_PER_GROUP=$((num_prompts / GSP_NUM_GROUPS))
      DATASET_ARGS=(--dataset-name generated-shared-prefix --gsp-num-groups "$GSP_NUM_GROUPS" --gsp-prompts-per-group "$GSP_PROMPTS_PER_GROUP" --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 1024 --gsp-range-ratio 1.0)
      ;;
  esac

  echo "========== max-concurrency: $max_conc num-prompts: $num_prompts dataset: $DATASET_LABEL ==========" | tee -a "$LOG_FILE"
  if [ "$DATASET" = "radixcache" ]; then
    echo "GSP actual prompts: $((GSP_NUM_GROUPS * GSP_PROMPTS_PER_GROUP)) ($GSP_NUM_GROUPS groups x $GSP_PROMPTS_PER_GROUP prompts/group)" | tee -a "$LOG_FILE"
  fi
  for run in 1 2
  do
    echo "--- Run $run ---" | tee -a "$LOG_FILE"
    CMD=(python3 -m sglang.bench_serving
      --backend sglang
      --base-url "$ROUTER_URL"
      "${DATASET_ARGS[@]}"
      --max-concurrency "$max_conc"
      --pd-separated
      --warmup-requests 2
      --output-details)
    "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
  done
done

echo "========== flushing caches ==========" | tee -a "$LOG_FILE"
for url in "${PREFILL_URL:-}" "${DECODE_URL:-}" "${ROUTER_URL:-}"
do
  if [ -n "$url" ]; then
    echo "POST $url/flush_cache" | tee -a "$LOG_FILE"
    curl -sS -X POST "$url/flush_cache" 2>&1 | tee -a "$LOG_FILE" || true
    echo "" | tee -a "$LOG_FILE"
  fi
done

echo "Logs saved to: $LOG_FILE"
```

Run one dataset:

```bash
chmod +x bench_pd_case.sh
./bench_pd_case.sh rand
./bench_pd_case.sh sharegpt
./bench_pd_case.sh radixcache
```

Expected log names:

```text
nixl_Ptp4_Dtp4_Pdp4_Ddp4_rand.log
mooncake_Ptp2_Dtp4_Pdp1_Ddp1_sharegpt.log
nixl_Ptp4_Dtp4_Pdp1_Ddp1_radixcache.log
```

## Quick Checks

```bash
curl "$PREFILL_URL/health"
curl "$DECODE_URL/health"
curl "$ROUTER_URL/health"
```

```bash
curl -s "$ROUTER_URL/generate" \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":16}}'
```
