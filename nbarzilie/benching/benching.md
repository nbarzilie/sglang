# PD Disaggregation Benchmark Guide

This guide launches one single-node PD disaggregation setup and benchmarks it through the router. Use four terminals on the same node or one interactive shell with `tmux` panes.

The examples assume:

```bash
export MY=$HOME
export LOG_DIR=/logs
```

Use the NIXL image for the NIXL flow:

```bash
$MY/sqshs/sglang_fresh.sqsh
```

Use the Mooncake image for the Mooncake flow:

```bash
./fresh-sglang-mooncake-cuda13.sqsh
```

If you enter containers manually, use the same mount/cache setup for every terminal:

```bash
srun \
  -A network_research_advdev \
  -t 02:00:00 \
  -N 1 \
  -p interactive \
  --gpus-per-node=8 \
  --container-image=$MY/sqshs/sglang_fresh.sqsh \
  --container-workdir=/sgl-workspace/sglang \
  --container-mounts=$MY/.cache/huggingface:/root/.cache/huggingface,$MY/logs:/logs \
  --pty bash
```

For Mooncake, replace `--container-image` with `./fresh-sglang-mooncake-cuda13.sqsh`.

Do not run the NIXL and Mooncake flows at the same time on the same ports.

## NIXL Backend

### Terminal 1: Prefill Node

```bash
sglang serve \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --disaggregation-ib-device mlx5_0 \
  --tp 2
```

### Terminal 2: Decode Node

```bash
sglang serve \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 2 \
  --disaggregation-transfer-backend nixl \
  --disaggregation-ib-device mlx5_0 \
  --tp 4
```

### Terminal 3: Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 \
  --port 8000
```

### Terminal 4: Bench Script

Create `bench_nixl.sh`:

```bash
#!/bin/bash

LOG_DIR="/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/bench_nixl_$TIMESTAMP.log"

for num_prompts in 1 2 4 8 16 32 64 128 256 512 1024
do
    echo "========== num-prompts: $num_prompts ==========" | tee -a "$LOG_FILE"

    for run in 1 2
    do
        echo "--- Run $run ---" | tee -a "$LOG_FILE"

        python3 -m sglang.bench_serving \
            --backend sglang \
            --base-url http://127.0.0.1:8000 \
            --dataset-name random \
            --num-prompts $num_prompts \
            --random-input 1024 \
            --random-output 1024 \
            --pd-separated \
            --warmup-requests 2 \
            --random-range-ratio 1.0 \
            --output-details 2>&1 | tee -a "$LOG_FILE"

        echo "" | tee -a "$LOG_FILE"
    done

    echo "" | tee -a "$LOG_FILE"
done

echo "✅ Logs saved to: $LOG_FILE"
```

Run it:

```bash
chmod +x bench_nixl.sh
./bench_nixl.sh
```

## Mooncake Backend

Use this flow from the Mooncake image:

```bash
./fresh-sglang-mooncake-cuda13.sqsh
```

### Terminal 1: Prefill Node

```bash
sglang serve \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0 \
  --tp 2
```

### Terminal 2: Decode Node

```bash
sglang serve \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 2 \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-ib-device mlx5_0 \
  --tp 4
```

### Terminal 3: Router

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 \
  --port 8000
```

### Terminal 4: Bench Script

Create `bench_mooncake.sh`:

```bash
#!/bin/bash

LOG_DIR="/logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/bench_mooncake_$TIMESTAMP.log"

for num_prompts in 1 2 4 8 16 32 64 128 256 512 1024
do
    echo "========== num-prompts: $num_prompts ==========" | tee -a "$LOG_FILE"

    for run in 1 2
    do
        echo "--- Run $run ---" | tee -a "$LOG_FILE"

        python3 -m sglang.bench_serving \
            --backend sglang \
            --base-url http://127.0.0.1:8000 \
            --dataset-name random \
            --num-prompts $num_prompts \
            --random-input 1024 \
            --random-output 1024 \
            --pd-separated \
            --warmup-requests 2 \
            --random-range-ratio 1.0 \
            --output-details 2>&1 | tee -a "$LOG_FILE"

        echo "" | tee -a "$LOG_FILE"
    done

    echo "" | tee -a "$LOG_FILE"
done

echo "✅ Logs saved to: $LOG_FILE"
```

Run it:

```bash
chmod +x bench_mooncake.sh
./bench_mooncake.sh
```

## Quick Checks

Check server health before benchmarking:

```bash
curl http://127.0.0.1:30000/health
curl http://127.0.0.1:30001/health
curl http://127.0.0.1:8000/health
```

Send a small request through the router:

```bash
curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"temperature":0,"max_new_tokens":16}}'
```

If Mooncake or NIXL cannot find `mlx5_0`, replace it with the active device on the node. List available devices with:

```bash
ibv_devices
```
