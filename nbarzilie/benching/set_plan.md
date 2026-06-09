# Mooncake vs NIXL PD Benchmark Plan

## Goal

Create a repeatable benchmark workflow that compares the Mooncake and NIXL
disaggregation transfer backends under the same SGLang source, same container,
same model, same TP/DP shape, same dataset, same concurrency, and same node/GPU
conditions.

The benchmark must produce:

- A single unified `.sqsh` image that contains both transfer stacks.
- Per-run environment fingerprints from the staged
  `/logs/scripts/pd_bench/fingerprint.sh`.
- Structured JSONL benchmark results from `sglang.bench_serving`.
- Plain logs for debugging server launch, router launch, and benchmark failures.
  Server stdout/stderr must first be written to a mounted temporary location and
  copied into permanent results only on failure, or when explicitly requested.
- Enough repetition to compare medians, means, variability, and SLO-constrained
  throughput.
- A JSONL-first analysis workflow that can replace the older cleaned-log parser.

## Design Principles

Use one image for both backends. The current `refresh_mooncake.sh` and
`refresh_nixl.sh` install nearly the same stack and already install both NIXL and
Mooncake packages. Keeping separate images makes image contents a confounder.
The unified image should be saved once from `sglang-fresh.sqsh` and used for
both `BACKEND=mooncake` and `BACKEND=nixl`.

Use JSONL as source of truth. `sglang.bench_serving` supports `--output-file`.
Text logs are still useful for debugging, but analysis should read JSONL records
directly.

Assume the runtime image can be a public SGLang image or checkout that does not
contain the `nbarzilie/benching` helper files. Therefore every runnable helper
script must be staged on the mounted log filesystem under:

```text
/logs/scripts/pd_bench/
```

All run commands should use `SCRIPT_ROOT=${SCRIPT_ROOT:-/logs/scripts/pd_bench}`
and execute scripts from that directory. The local repository paths in this plan
are source paths for authoring only.

Benchmark at two layers:

1. Serving-level benchmark: `sglang.bench_serving` against the PD router.
2. Transfer-aware evidence: capture fingerprints, server logs, backend env, and
   any NIXL/Mooncake transfer metrics that SGLang logs. If deeper transfer
   telemetry is added later, write it as sidecar JSONL under the same run
   directory.

Compare at the same concurrency and also by SLO. Peak throughput alone is not
enough. The analysis must identify the best backend at the highest concurrency
that satisfies selected latency SLOs.

## Mission 1: Create a Unified SQSH Refresh Script

Create:

```text
source path: nbarzilie/benching/refresh_united.sh
runtime path: /logs/scripts/pd_bench/refresh_united.sh
```

The script should be based on:

- `nbarzilie/benching/refresh_mooncake.sh`
- `nbarzilie/benching/refresh_nixl.sh`

Target image:

```text
$MY/sqshs/sglang_pd_transfer_united.sqsh
```

The script should:

1. Start from `$MY/sqshs/sglang-fresh.sqsh`.
2. Save to `$MY/sqshs/sglang_pd_transfer_united.sqsh`.
3. Pull the SGLang source to the latest fast-forwardable commit.
4. Install the common SGLang runtime stack.
5. Install both transfer stacks in the same image.
6. Install/refresh the `sglang` CLI wrapper so source code is used from
   `/sgl-workspace/sglang/python`.
7. Run import checks for `sglang`, `sgl_kernel`, `nixl._api`, and
   `mooncake.engine`.
8. Run `fingerprint.sh` during refresh and save the output
   under `/logs/image_fingerprints/`.
9. Leave an interactive shell open before final save so the image can be
   inspected. Pyxis saves when the shell exits.

Template:

```bash
#!/usr/bin/env bash
set -u

: "${MY:?Set MY to the cluster workspace root}"

SRC_IMAGE="$MY/sqshs/sglang-fresh.sqsh"
OUT_IMAGE="$MY/sqshs/sglang_pd_transfer_united.sqsh"

srun \
  -A network_research_advdev \
  -t 02:00:00 \
  -N 1 \
  -p interactive \
  --gpus-per-node=8 \
  --container-image="$SRC_IMAGE" \
  --container-save="$OUT_IMAGE" \
  --container-writable \
  --container-remap-root \
  --container-workdir=/sgl-workspace/sglang \
  --container-mounts="$MY/.cache/huggingface:/root/.cache/huggingface,$MY/logs:/logs,$MY:/host_my" \
  --pty bash -lc '
set -euo pipefail

echo "===== refresh source ====="
git status --short
git pull --ff-only
git rev-parse HEAD

echo
echo "===== install common stack ====="
python3 -m pip install --upgrade pip setuptools wheel --break-system-packages
python3 -m pip install --force-reinstall "sglang-kernel==0.4.3" --break-system-packages
python3 -m pip install --upgrade sglang-router --break-system-packages
python3 -m pip uninstall -y sglang --break-system-packages || true

printf "%s\n" "export PYTHONPATH=/sgl-workspace/sglang/python:\${PYTHONPATH:-}" > /etc/profile.d/sglang-source.sh
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}

printf "%s\n" \
  "#!/usr/bin/env bash" \
  "export PYTHONPATH=/sgl-workspace/sglang/python:\${PYTHONPATH:-}" \
  "exec python3 -c '\''from sglang.cli.main import main; main()'\'' \"\$@\"" \
  > /usr/local/bin/sglang
chmod +x /usr/local/bin/sglang

echo
echo "===== install both transfer stacks ====="
python3 -m pip install --upgrade nixl nixl-cu13 --no-deps --break-system-packages
python3 -m pip install --upgrade "cuda-python==13.2.0" --break-system-packages
python3 -m pip uninstall -y mooncake-transfer-engine --break-system-packages || true
python3 -m pip install --upgrade mooncake-transfer-engine-cuda13 --break-system-packages

echo
echo "===== import checks ====="
python3 - <<'"'"'PY'"'"'
import importlib.metadata as md
import importlib.util
import subprocess
import sys

required_packages = [
    "sglang-kernel",
    "sglang-router",
    "torch",
    "triton",
    "nixl",
    "nixl-cu13",
    "mooncake-transfer-engine-cuda13",
]
optional_packages = [
    "sglang",
    "flashinfer-python",
    "flashinfer-cubin",
    "mooncake-transfer-engine",
]
required_modules = ["sglang", "sgl_kernel", "nixl._api", "mooncake.engine"]

print("git", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
print("python", sys.version.replace("\n", " "))
missing_packages = []
for pkg in required_packages + optional_packages:
    try:
        print(pkg, md.version(pkg))
    except md.PackageNotFoundError:
        print(pkg, "NOT_INSTALLED")
        if pkg in required_packages:
            missing_packages.append(pkg)

missing_modules = []
for mod in required_modules:
    spec = importlib.util.find_spec(mod)
    print("module", mod, spec)
    if spec is None:
        missing_modules.append(mod)

if missing_packages or missing_modules:
    raise SystemExit(
        "refresh validation failed: "
        f"missing_packages={missing_packages} missing_modules={missing_modules}"
    )
PY

echo
echo "===== gpu / ib summary ====="
nvidia-smi -L || true
ls -l /sys/class/infiniband || true

echo
echo "===== refresh fingerprint ====="
mkdir -p /logs/image_fingerprints
SCRIPT_ROOT="${SCRIPT_ROOT:-/logs/scripts/pd_bench}"
if [ -x "$SCRIPT_ROOT/fingerprint.sh" ]; then
  "$SCRIPT_ROOT/fingerprint.sh" \
    > "/logs/image_fingerprints/united_$(date -u +%Y%m%dT%H%M%SZ).txt" 2>&1
else
  echo "fingerprint script not found or not executable at $SCRIPT_ROOT/fingerprint.sh"
fi

echo
echo "Unified PD transfer image is ready to inspect. Exit this shell to save:"
echo "  '"$OUT_IMAGE"'"
exec bash -i
'
```

Validation after saving:

```bash
srun \
  -A network_research_advdev \
  -t 00:20:00 \
  -N 1 \
  -p interactive \
  --gpus-per-node=8 \
  --container-image="$MY/sqshs/sglang_pd_transfer_united.sqsh" \
  --container-workdir=/sgl-workspace/sglang \
  --container-mounts="$MY/.cache/huggingface:/root/.cache/huggingface,$MY/logs:/logs" \
  --pty bash -lc '
set -euo pipefail
export PYTHONPATH=/sgl-workspace/sglang/python:${PYTHONPATH:-}
python3 - <<PY
import importlib.util
missing = []
for mod in ["sglang", "sgl_kernel", "nixl._api", "mooncake.engine"]:
    spec = importlib.util.find_spec(mod)
    print(mod, spec)
    if spec is None:
        missing.append(mod)
if missing:
    raise SystemExit(f"missing required modules: {missing}")
PY
python3 -m sglang.bench_serving --help >/dev/null && echo bench_serving_ok
'
```

## Mission 2: Use Fingerprint Results in Logs

`nbarzilie/benching/fingerprint.sh` should be copied into every result tree.
At runtime, use the staged copy:

```text
/logs/scripts/pd_bench/fingerprint.sh
```

Run it in two places:

1. Once at job start:

```text
/logs/pd_bench/<run_id>/fingerprint_job.txt
```

2. Once per backend/case/dataset batch before launching servers:

```text
/logs/pd_bench/<run_id>/<backend>/<case>/<dataset>/fingerprint.txt
```

Also store a machine-readable wrapper metadata file:

```text
/logs/pd_bench/<run_id>/<backend>/<case>/<dataset>/run_meta.json
```

Minimum `run_meta.json` fields:

```json
{
  "run_id": "20260609T120000Z",
  "backend": "nixl",
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "case_name": "Ptp2_Dtp4_Pdp1_Ddp1",
  "dataset": "rand",
  "image": "$MY/sqshs/sglang_pd_transfer_united.sqsh",
  "nodes": 1,
  "prefill_tp": 2,
  "decode_tp": 4,
  "prefill_dp": 1,
  "decode_dp": 1,
  "concurrency_values": [1, 2, 4, 8, 16, 32],
  "repetitions": 5,
  "min_prompts": 100,
  "prompts_per_concurrency": 10,
  "timestamp_utc": "2026-06-09T12:00:00Z"
}
```

Fingerprint integration template:

```bash
write_fingerprint() {
  local out="$1"
  mkdir -p "$(dirname "$out")"
  local script_root="${SCRIPT_ROOT:-/logs/scripts/pd_bench}"
  if [ -x "$script_root/fingerprint.sh" ]; then
    "$script_root/fingerprint.sh" > "$out" 2>&1
  else
    echo "fingerprint.sh missing or not executable at $script_root/fingerprint.sh" > "$out"
    env | sort >> "$out"
  fi
}
```

## Mission 3: Automated Benchmark Method

### Workloads

Run all three datasets:

- `rand`: random fixed-length prefill/decode, good for baseline scaling.
- `sharegpt`: more realistic conversational distribution.
- `radixcache`: generated shared-prefix workload, critical for cache/full-hit
  behavior and aux-only transfer paths.

Dataset arguments:

```bash
rand:
  --dataset-name random
  --random-input-len 1024
  --random-output-len 1024
  --random-range-ratio 1.0

sharegpt:
  --dataset-name sharegpt
  --sharegpt-output-len 1024

radixcache:
  --dataset-name generated-shared-prefix
  --gsp-system-prompt-len 2048
  --gsp-question-len 128
  --gsp-output-len 1024
  --gsp-range-ratio 1.0
```

For `radixcache`, compute:

```bash
GSP_NUM_GROUPS <= 64
GSP_NUM_GROUPS * GSP_PROMPTS_PER_GROUP == num_prompts
```

### Repetitions

Use 5 measured repetitions for each
`backend x case x dataset x concurrency`.

Use a small warmup:

```bash
--warmup-requests 2
```

If time is tight, preserve 5 reps for high-signal points and reduce the matrix,
not the repetition count. Repetition is needed because PD serving measurements
can vary due to queueing, launch state, and network noise.

### Metrics

Primary canonical metrics after JSONL normalization:

- `successful_requests`
- `request_throughput`
- `input_throughput`
- `output_throughput`
- `total_throughput`
- `mean_ttft_ms`
- `median_ttft_ms`
- `p99_ttft_ms`
- `mean_tpot_ms`
- `median_tpot_ms`
- `p99_tpot_ms`
- `mean_itl_ms`
- `median_itl_ms`
- `p95_itl_ms`
- `p99_itl_ms`
- `max_itl_ms`
- `benchmark_duration_s`

The raw `bench_serving` JSON currently writes some different names, such as
`duration` and `completed`. Normalize raw keys immediately in the parser and use
canonical names everywhere after ingestion. Missing optional fields should be
represented as null/empty values, not as parser failures.

Analysis must compute:

- Mean, median, standard deviation, min, max across the 5 reps.
- Coefficient of variation for key metrics.
- Backend speedup ratios at the same case/dataset/concurrency.
- SLO-constrained throughput.

Recommended initial SLOs:

```text
TTFT p99 <= 5000 ms
TPOT p99 <= 200 ms
ITL p99 <= 200 ms
success rate == 100%
```

Keep SLOs configurable in the analysis script. These values are starting points,
not universal product targets.

### Two-Layer Benchmark

Layer 1: serving benchmark.

Command:

```bash
DETAIL_ARGS=()
if [ "${OUTPUT_DETAILS:-0}" = "1" ]; then
  DETAIL_ARGS=(--output-details)
fi

python3 -m sglang.bench_serving \
  --backend sglang \
  --base-url "$ROUTER_URL" \
  "${DATASET_ARGS[@]}" \
  --num-prompts "$num_prompts" \
  --max-concurrency "$max_conc" \
  --pd-separated \
  --warmup-requests 2 \
  "${DETAIL_ARGS[@]}" \
  --output-file "$jsonl_file" \
  --tag "$tag"
```

Default `OUTPUT_DETAILS=0` for Phase 1 and Phase 2. `--output-details` includes
per-request lengths, TTFTs, ITLs, errors, and full generated texts; with
1024-token outputs across hundreds of runs, it can inflate JSONL storage
substantially. Use `OUTPUT_DETAILS=1` for Phase 0 smoke, failure repros, or
small targeted investigations. If details are enabled in a broad run, the
analysis script should strip or ignore `generated_texts` before producing
summary artifacts.

Layer 2: transfer-aware sidecar.

Immediate version:

- Save server logs for prefill, decode, and router.
- Save `fingerprint.txt`.
- Save exact commands.
- Save backend environment variables.
- Save `/server_info` and `/health` outputs from prefill, decode, and router.

Future version:

- Add SGLang-side transfer event JSONL if available or if instrumentation is
  later added.
- For NIXL, include transfer telemetry fields if SGLang exposes them.
- For Mooncake, include comparable transfer latency/bytes if exposed.

### Job Constraints

Assumptions:

- At most 2 active Slurm jobs.
- Every job is at most 2 hours.
- Every node has 8 GPUs.
- Every job can use up to 2 nodes.
- High concurrency points `64 128 256` may need 2 nodes.

This means the full matrix cannot be run as one monolithic job:

```text
2 backends x 6 cases x 3 datasets x 9 concurrencies x 5 reps = 1620 bench_serving invocations
```

The automation should split the matrix into bounded shards.

Recommended phases:

#### Phase 0: Smoke Tests

Purpose: prove the image and scripts work.

Run on 1 node:

```text
model: Llama-3.1-8B-Instruct
backends: nixl, mooncake
cases: Ptp2_Dtp2_Pdp2_Ddp2
datasets: rand
concurrency: 1 8 32
reps: 1
```

If this fails, do not run the full benchmark.

Expected size:

```text
2 backends x 1 case x 1 dataset x 3 concurrencies x 1 rep = 6 bench_serving invocations
```

This should fit comfortably in one 2-hour interactive or batch job unless model
download or server launch is broken.

#### Phase 1: Reduced Main One-Node Sweep

Purpose: compare the most important one-node PD topologies without exploding the
matrix.

Run on 1 node:

```text
model: Llama-3.1-8B-Instruct
backends: nixl, mooncake
cases:
  Ptp2_Dtp2_Pdp2_Ddp2   # 2-2-2-2
  Ptp4_Dtp4_Pdp1_Ddp1   # 4-4-1-1
  Ptp2_Dtp4_Pdp1_Ddp1   # 2-4-1-1
  Ptp4_Dtp2_Pdp1_Ddp1   # 4-2-1-1
datasets: rand, sharegpt, radixcache
concurrency: 1 8 32
reps: 5
```

Expected size:

```text
2 backends x 4 cases x 3 datasets x 3 concurrencies x 5 reps = 360 bench_serving invocations
```

This is the default comparison set. It is much more practical than the full
matrix and should be split into 2-hour shards. With at most two active jobs,
start with two shards:

```text
job 1: cases Ptp2_Dtp2_Pdp2_Ddp2, Ptp2_Dtp4_Pdp1_Ddp1
job 2: cases Ptp4_Dtp4_Pdp1_Ddp1, Ptp4_Dtp2_Pdp1_Ddp1
```

If either shard exceeds 2 hours, split by dataset:

```text
shard key: case subset x dataset subset
```

Do not add the dropped one-node cases back into Phase 1 unless Phase 1 results
are already stable and the extra wall-clock cost is justified.

#### Phase 2: High-Concurrency Two-Node PD Sweep

Purpose: test saturation behavior.

Run on 2 nodes:

```text
model: Llama-3.1-8B-Instruct
backends: nixl, mooncake
cases:
  Ptp2_Dtp2_Pdp2_Ddp2   # 2-2-2-2
  Ptp4_Dtp4_Pdp1_Ddp1   # 4-4-1-1
  Ptp2_Dtp4_Pdp1_Ddp1   # 2-4-1-1
  Ptp4_Dtp2_Pdp1_Ddp1   # 4-2-1-1
datasets: rand, radixcache, sharegpt
concurrency: 64 128 256
reps: 5
```

Expected size:

```text
2 backends x 4 cases x 3 datasets x 3 concurrencies x 5 reps = 360 bench_serving invocations
```

Phase 2 should use the same case/dataset/backend structure as Phase 1 so the
curves join cleanly at the analysis layer:

```text
Phase 1 concurrency: 1, 8, 32
Phase 2 concurrency: 64, 128, 256
```

Run Phase 2 only after Phase 1 has clean fingerprints, complete JSONL records,
and no systematic server failures.

### Backend Ordering

Avoid always running all NIXL first and all Mooncake second. Node state can drift.

Preferred order per case/dataset:

```text
nixl c=1 reps
mooncake c=1 reps
mooncake c=2 reps
nixl c=2 reps
nixl c=4 reps
mooncake c=4 reps
...
```

Simpler acceptable order:

```text
nixl then mooncake for one case/dataset
mooncake then nixl for the next case/dataset
```

Record actual order in `run_meta.json`.

## Mission 4: Comprehensive srun/sbatch Guide for the Unified SQSH

### Script Staging

Before launching any interactive or batch job, copy every benchmark helper into
the mounted log script directory on the host:

```bash
mkdir -p "$MY/logs/scripts/pd_bench"
cp nbarzilie/benching/refresh_united.sh "$MY/logs/scripts/pd_bench/"
cp nbarzilie/benching/fingerprint.sh "$MY/logs/scripts/pd_bench/"
cp nbarzilie/benching/run_pd_backend_matrix.sh "$MY/logs/scripts/pd_bench/"
cp nbarzilie/benching/run_pd_backend_matrix_2node.sh "$MY/logs/scripts/pd_bench/" 2>/dev/null || true
cp nbarzilie/benching/sbatch_pd_1node.sh "$MY/logs/scripts/pd_bench/"
cp nbarzilie/benching/sbatch_pd_2node_high_conc.sh "$MY/logs/scripts/pd_bench/" 2>/dev/null || true
chmod +x "$MY/logs/scripts/pd_bench/"*.sh
```

Inside the container these files appear at:

```text
/logs/scripts/pd_bench/
```

Do not rely on `nbarzilie/benching/*` being present inside the image. Public
SGLang images and clean checkouts may not include these local helper files.

### Directory Layout

Stage scripts here before starting any job:

```text
/logs/scripts/pd_bench/
  refresh_united.sh
  fingerprint.sh
  run_pd_backend_matrix.sh
  run_pd_backend_matrix_2node.sh
  sbatch_pd_1node.sh
  sbatch_pd_2node_high_conc.sh
```

This matters when the container is based on public SGLang. The mounted `/logs`
directory is the portable control plane for the benchmark scripts.

Use this output layout:

```text
/logs/pd_bench/
  <run_id>/
    fingerprint_job.txt
    job_meta.json
    sbatch_stdout.log
    nixl/
      <case>/
        <dataset>/
          run_meta.json
          fingerprint.txt
          env.txt
          commands.sh
          health_before.json
          health_after.json
          failed_server_logs/
            prefill.log
            decode.log
            router.log
          results/
            c1_r1.jsonl
            c1_r2.jsonl
            ...
          bench_logs/
            c1_r1.log
            c1_r2.log
            ...
    mooncake/
      ...
/logs/pd_bench_tmp/
  <run_id>/
    <backend>/
      <case>/
        <dataset>/
          prefill.log
          decode.log
          router.log
```

Server stdout/stderr always goes to `/logs/pd_bench_tmp/...` first. On success,
delete that temporary server-log directory by default. On failure, copy or move
it to `failed_server_logs/` before cleanup so launch errors, Python tracebacks,
NIXL/Mooncake transport errors, and router failures remain inspectable.

### Shared Environment File

Create:

```text
/logs/pd_bench/<run_id>/pd_env.sh
```

Template:

```bash
export HF_TOKEN=<HF_TOKEN>
export MODEL=meta-llama/Llama-3.1-8B-Instruct
export IB_DEV=mlx5_0
export LOG_ROOT=/logs/pd_bench
export TMP_LOG_ROOT=/logs/pd_bench_tmp
export SCRIPT_ROOT=/logs/scripts/pd_bench
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"

export CONCURRENCY_VALUES="1 8 32"
export HIGH_CONCURRENCY_VALUES="64 128 256"
export REPS=5
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
```

For 2-node jobs, override URLs to routable hostnames or node IPs. Use
`scontrol show hostnames "$SLURM_JOB_NODELIST"` to derive them.

### Case Definitions

Create a helper function in the runner:

```bash
set_case_env() {
  local case_name="$1"
  export CASE_NAME="$case_name"
  case "$case_name" in
    Ptp4_Dtp4_Pdp4_Ddp4)
      export PREFILL_TP_SIZE=4 DECODE_TP_SIZE=4 PREFILL_DP_SIZE=4 DECODE_DP_SIZE=4
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
    Ptp2_Dtp4_Pdp1_Ddp4)
      export PREFILL_TP_SIZE=2 DECODE_TP_SIZE=4 PREFILL_DP_SIZE=1 DECODE_DP_SIZE=4
      export PREFILL_BASE_GPU_ID=0 DECODE_BASE_GPU_ID=2
      ;;
    Ptp2_Dtp2_Pdp2_Ddp2)
      export PREFILL_TP_SIZE=2 DECODE_TP_SIZE=2 PREFILL_DP_SIZE=2 DECODE_DP_SIZE=2
      export PREFILL_BASE_GPU_ID=0 DECODE_BASE_GPU_ID=2
      ;;
    Ptp4_Dtp4_Pdp1_Ddp1)
      export PREFILL_TP_SIZE=4 DECODE_TP_SIZE=4 PREFILL_DP_SIZE=1 DECODE_DP_SIZE=1
      export PREFILL_BASE_GPU_ID=0 DECODE_BASE_GPU_ID=4
      ;;
    *)
      echo "unknown case: $case_name" >&2
      return 1
      ;;
  esac
}
```

Add a GPU-fit preflight immediately after `set_case_env` and before server
launch. DP attention does not multiply the process GPU count by `dp_size`; for
these one-node PD cases the GPU count per side is `tp_size` when `pp_size=1`.
The names are easy to misread, so the runner must record the resolved GPU
ranges explicitly.

```bash
preflight_gpu_fit() {
  local total_visible="${TOTAL_GPUS_PER_NODE:-8}"
  local prefill_gpus="${PREFILL_TP_SIZE}"
  local decode_gpus="${DECODE_TP_SIZE}"
  local prefill_start="${PREFILL_BASE_GPU_ID}"
  local decode_start="${DECODE_BASE_GPU_ID}"
  local prefill_end=$((prefill_start + prefill_gpus - 1))
  local decode_end=$((decode_start + decode_gpus - 1))

  if [ "$prefill_gpus" -le 0 ] || [ "$decode_gpus" -le 0 ]; then
    echo "invalid gpu count: prefill=$prefill_gpus decode=$decode_gpus" >&2
    return 1
  fi
  if [ $((prefill_end + 1)) -gt "$total_visible" ]; then
    echo "prefill gpu range exceeds node: ${prefill_start}-${prefill_end} of $total_visible" >&2
    return 1
  fi
  if [ $((decode_end + 1)) -gt "$total_visible" ]; then
    echo "decode gpu range exceeds node: ${decode_start}-${decode_end} of $total_visible" >&2
    return 1
  fi
  if [ "$prefill_start" -le "$decode_end" ] && [ "$decode_start" -le "$prefill_end" ]; then
    echo "prefill/decode gpu ranges overlap: prefill=${prefill_start}-${prefill_end} decode=${decode_start}-${decode_end}" >&2
    return 1
  fi

  export PREFILL_RESOLVED_GPU_COUNT="$prefill_gpus"
  export DECODE_RESOLVED_GPU_COUNT="$decode_gpus"
  export PREFILL_GPU_RANGE="${prefill_start}-${prefill_end}"
  export DECODE_GPU_RANGE="${decode_start}-${decode_end}"
}
```

`run_meta.json` must include:

```json
{
  "prefill_resolved_gpu_count": 2,
  "decode_resolved_gpu_count": 4,
  "prefill_gpu_range": "0-1",
  "decode_gpu_range": "2-5",
  "total_gpus_per_node": 8
}
```

### Automated Runner Script

Create:

```text
source path: nbarzilie/benching/run_pd_backend_matrix.sh
runtime path: /logs/scripts/pd_bench/run_pd_backend_matrix.sh
```

Responsibilities:

1. Source `/logs/pd_bench/<run_id>/pd_env.sh` or accept equivalent env vars.
2. Select cases, datasets, concurrencies, and backends from env.
3. For each backend/case/dataset:
   - Create output directory.
   - Create temporary server-log directory under
     `/logs/pd_bench_tmp/<run_id>/<backend>/<case>/<dataset>/`.
   - Write `run_meta.json`.
   - Run `fingerprint.sh`.
   - Launch prefill server in background with stdout/stderr in the temporary
     server-log directory.
   - Launch decode server in background with stdout/stderr in the temporary
     server-log directory.
   - Launch router in background with stdout/stderr in the temporary server-log
     directory.
   - Wait for health.
   - Run benchmark reps and write one JSONL per run.
   - Flush cache before each concurrency point, unless intentionally measuring
     steady-state cache reuse.
   - Capture health after benchmark.
   - Kill prefill/decode/router process trees.
   - On success, remove temporary server logs unless `KEEP_SUCCESS_LOGS=1`.
   - On failure, copy temporary server logs to `failed_server_logs/`.
4. Fail fast on server launch failure.
5. Continue to the next shard only after cleanup is complete.

Temporary server-log setup:

```bash
prepare_batch_dirs() {
  OUT_DIR="$LOG_ROOT/$RUN_ID/$BACKEND/$CASE_NAME/$DATASET"
  TMP_SERVER_LOG_DIR="${TMP_LOG_ROOT:-/logs/pd_bench_tmp}/$RUN_ID/$BACKEND/$CASE_NAME/$DATASET"
  mkdir -p "$OUT_DIR" "$TMP_SERVER_LOG_DIR"
}

preserve_server_logs_on_failure() {
  local rc="$1"
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
```

Important cleanup pattern:

```bash
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
  set -e
  preserve_server_logs_on_failure "$rc"
}
trap 'cleanup_servers $?' EXIT
```

Server launch commands:

```bash
launch_prefill() {
  local log="$1"
  local dp_args=()
  if [ "$PREFILL_DP_SIZE" -gt 1 ]; then
    dp_args=(--enable-dp-attention)
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
    "${dp_args[@]}" \
    > "$log" 2>&1 &
  PREFILL_PID=$!
}

launch_decode() {
  local log="$1"
  local dp_args=()
  if [ "$DECODE_DP_SIZE" -gt 1 ]; then
    dp_args=(--enable-dp-attention)
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
    "${dp_args[@]}" \
    > "$log" 2>&1 &
  DECODE_PID=$!
}

launch_router() {
  local log="$1"
  python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "$PREFILL_URL" \
    --decode "$DECODE_URL" \
    --host "$ROUTER_HOST" \
    --port "$ROUTER_PORT" \
    > "$log" 2>&1 &
  ROUTER_PID=$!
}
```

Health helper:

```bash
wait_health() {
  local url="$1"
  local name="$2"
  local deadline=$((SECONDS + 900))
  until curl -fsS "$url/health" >/dev/null; do
    if [ "$SECONDS" -gt "$deadline" ]; then
      echo "timeout waiting for $name at $url" >&2
      return 1
    fi
    sleep 5
  done
}
```

Benchmark command:

```bash
run_one_bench() {
  local dataset="$1"
  local max_conc="$2"
  local rep="$3"
  local out_dir="$4"

  local num_prompts=$((max_conc * PROMPTS_PER_CONCURRENCY))
  if [ "$num_prompts" -lt "$MIN_PROMPTS" ]; then
    num_prompts="$MIN_PROMPTS"
  fi

  local dataset_args=()
  case "$dataset" in
    rand)
      dataset_args=(--dataset-name random --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 --num-prompts "$num_prompts")
      ;;
    sharegpt)
      dataset_args=(--dataset-name sharegpt --sharegpt-output-len 1024 --num-prompts "$num_prompts")
      ;;
    radixcache)
      local gsp_num_groups=1
      local limit="$num_prompts"
      if [ "$limit" -gt 64 ]; then limit=64; fi
      for ((candidate=limit; candidate>=1; candidate--)); do
        if [ $((num_prompts % candidate)) -eq 0 ]; then
          gsp_num_groups="$candidate"
          break
        fi
      done
      local gsp_prompts_per_group=$((num_prompts / gsp_num_groups))
      dataset_args=(--dataset-name generated-shared-prefix --gsp-num-groups "$gsp_num_groups" --gsp-prompts-per-group "$gsp_prompts_per_group" --gsp-system-prompt-len 2048 --gsp-question-len 128 --gsp-output-len 1024 --gsp-range-ratio 1.0)
      ;;
    *)
      echo "unknown dataset: $dataset" >&2
      return 1
      ;;
  esac

  mkdir -p "$out_dir/results" "$out_dir/bench_logs"
  local tag="${BACKEND}_${CASE_NAME}_${dataset}_c${max_conc}_r${rep}"
  local jsonl="$out_dir/results/c${max_conc}_r${rep}.jsonl"
  local log="$out_dir/bench_logs/c${max_conc}_r${rep}.log"
  local detail_args=()
  if [ "${OUTPUT_DETAILS:-0}" = "1" ]; then
    detail_args=(--output-details)
  fi

  python3 -m sglang.bench_serving \
    --backend sglang \
    --base-url "$ROUTER_URL" \
    "${dataset_args[@]}" \
    --max-concurrency "$max_conc" \
    --pd-separated \
    --warmup-requests 2 \
    "${detail_args[@]}" \
    --output-file "$jsonl" \
    --tag "$tag" \
    > "$log" 2>&1
}
```

### Interactive srun

Use for smoke tests and debugging:

```bash
export HF_TOKEN=<HF_TOKEN>

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
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)_smoke"
mkdir -p "/logs/pd_bench/$RUN_ID"

# Create or source the env file.
source /logs/pd_bench/$RUN_ID/pd_env.sh

BACKENDS="nixl mooncake" \
CASES="Ptp2_Dtp2_Pdp2_Ddp2" \
DATASETS="rand" \
CONCURRENCY_VALUES="1 8 32" \
REPS=1 \
bash /logs/scripts/pd_bench/run_pd_backend_matrix.sh
```

### sbatch for 1-Node Shards

Create:

```text
source path: nbarzilie/benching/sbatch_pd_1node.sh
runtime path: /logs/scripts/pd_bench/sbatch_pd_1node.sh
```

Template:

```bash
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
if [ -n "${SLURM_JOB_ID:-}" ] && [ -f "slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" ]; then
  cp "slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$HOST_RUN_DIR/sbatch_stdout_start.log" || true
fi

srun \
  --container-image="$IMAGE" \
  --container-workdir=/sgl-workspace/sglang \
  --container-mounts="$MY/.cache/huggingface:/root/.cache/huggingface,$MY/logs:/logs,$MY:/host_my" \
  bash -lc "
set -euo pipefail
export PYTHONPATH=/sgl-workspace/sglang/python:\${PYTHONPATH:-}
export HF_TOKEN='$HF_TOKEN'
export RUN_ID='$RUN_ID'
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
export CASES=\"${CASES:-Ptp2_Dtp2_Pdp2_Ddp2 Ptp4_Dtp4_Pdp1_Ddp1 Ptp2_Dtp4_Pdp1_Ddp1 Ptp4_Dtp2_Pdp1_Ddp1}\"
export DATASETS=\"${DATASETS:-rand sharegpt radixcache}\"
export CONCURRENCY_VALUES=\"${CONCURRENCY_VALUES:-1 8 32}\"
export REPS=\"${REPS:-5}\"
export OUTPUT_DETAILS=\"${OUTPUT_DETAILS:-0}\"
export CACHE_MODE=\"${CACHE_MODE:-warm}\"
bash /logs/scripts/pd_bench/run_pd_backend_matrix.sh
"

cp "slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$HOST_RUN_DIR/sbatch_stdout.log" 2>/dev/null || true
```

Submit examples:

Phase 0 smoke:

```bash
RUN_ID=llama_phase0_smoke \
BACKENDS="nixl mooncake" \
CASES="Ptp2_Dtp2_Pdp2_Ddp2" \
DATASETS="rand" \
CONCURRENCY_VALUES="1 8 32" \
REPS=1 \
sbatch /logs/scripts/pd_bench/sbatch_pd_1node.sh
```

Phase 1 reduced main sweep, shard A:

```bash
RUN_ID=llama_phase1_a \
BACKENDS="nixl mooncake" \
CASES="Ptp2_Dtp2_Pdp2_Ddp2 Ptp2_Dtp4_Pdp1_Ddp1" \
DATASETS="rand sharegpt radixcache" \
CONCURRENCY_VALUES="1 8 32" \
REPS=5 \
sbatch /logs/scripts/pd_bench/sbatch_pd_1node.sh
```

Phase 1 reduced main sweep, shard B:

```bash
RUN_ID=llama_phase1_b \
BACKENDS="mooncake nixl" \
CASES="Ptp4_Dtp4_Pdp1_Ddp1 Ptp4_Dtp2_Pdp1_Ddp1" \
DATASETS="rand sharegpt radixcache" \
CONCURRENCY_VALUES="1 8 32" \
REPS=5 \
sbatch /logs/scripts/pd_bench/sbatch_pd_1node.sh
```

Run at most two active jobs at a time.

### sbatch for 2-Node High-Concurrency Shards

Create:

```text
source path: nbarzilie/benching/sbatch_pd_2node_high_conc.sh
runtime path: /logs/scripts/pd_bench/sbatch_pd_2node_high_conc.sh
```

Key differences:

- `#SBATCH -N 2`
- `CONCURRENCY_VALUES="64 128 256"`
- Use routable node hostnames or IPs.
- Usually set `PREFILL_BASE_GPU_ID=0` and `DECODE_BASE_GPU_ID=0` because each
  side can run on a separate node.

Important: two-node process placement needs explicit design. The simplest robust
approach is one container task per node:

- Node 0 runs prefill and router.
- Node 1 runs decode.
- Benchmark client can run on node 0.

The runner script should detect:

```bash
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
PREFILL_NODE="${NODES[0]}"
DECODE_NODE="${NODES[1]}"
```

Then use routable URLs:

```bash
export PREFILL_URL="http://${PREFILL_NODE}:30000"
export DECODE_URL="http://${DECODE_NODE}:30001"
export ROUTER_URL="http://${PREFILL_NODE}:8000"
```

Before Phase 2 is run, create and validate a dedicated:

```text
/logs/scripts/pd_bench/run_pd_backend_matrix_2node.sh
```

Phase 2 is blocked until that script proves:

- One container task launches on the prefill/router node.
- One container task launches on the decode node.
- Prefill, decode, and router URLs are routable across nodes.
- Both tasks write to the shared `/logs` mount.
- Cleanup kills server processes on both nodes.
- Failure preserves server stdout/stderr from both nodes.

Do not run Phase 2 with the one-node runner. Do not fake two-node mode by
launching both servers on one node.

Two-node launch sketch:

```bash
srun --nodes=1 --nodelist="$PREFILL_NODE" --ntasks=1 bash -lc 'launch_prefill_and_router' &
srun --nodes=1 --nodelist="$DECODE_NODE" --ntasks=1 bash -lc 'launch_decode' &
```

The two-node script must still write results to the shared `/logs` mount.

Phase 2 high-concurrency two-node examples:

```bash
RUN_ID=llama_phase2_a \
BACKENDS="nixl mooncake" \
CASES="Ptp2_Dtp2_Pdp2_Ddp2 Ptp2_Dtp4_Pdp1_Ddp1" \
DATASETS="rand sharegpt radixcache" \
CONCURRENCY_VALUES="64 128 256" \
REPS=5 \
sbatch /logs/scripts/pd_bench/sbatch_pd_2node_high_conc.sh
```

```bash
RUN_ID=llama_phase2_b \
BACKENDS="mooncake nixl" \
CASES="Ptp4_Dtp4_Pdp1_Ddp1 Ptp4_Dtp2_Pdp1_Ddp1" \
DATASETS="rand sharegpt radixcache" \
CONCURRENCY_VALUES="64 128 256" \
REPS=5 \
sbatch /logs/scripts/pd_bench/sbatch_pd_2node_high_conc.sh
```

### Cache Policy

Default policy:

- Flush prefill, decode, and router caches before each concurrency point.
- Do not flush between the five reps for the same concurrency unless measuring
  cold-cache behavior.
- Record `CACHE_MODE` in `run_meta.json` for every batch.

Rationale:

- For `rand` and `sharegpt`, flush removes cross-point state.
- For `radixcache`, keeping cache across reps helps measure steady-state
  shared-prefix behavior. If cold-cache radix behavior is required, add
  `CACHE_MODE=cold` and flush before every rep.
- Warm radix reps are not statistically independent samples of cold-start
  transfer cost. They measure steady-state shared-prefix/cache behavior.
- Add at least one cold radix point for transfer-cost comparison:

```text
dataset: radixcache
cache_mode: cold
case: Ptp2_Dtp2_Pdp2_Ddp2
concurrency: 32
reps: 5
```

Cache flush helper:

```bash
flush_caches() {
  for url in "$PREFILL_URL" "$DECODE_URL" "$ROUTER_URL"; do
    curl -fsS -X POST "$url/flush_cache" >/dev/null 2>&1 || true
  done
}
```

Runner policy:

```bash
if [ "${CACHE_MODE:-warm}" = "cold" ]; then
  flush_caches  # before every rep
else
  flush_caches  # before each concurrency point
fi
```

## Mission 5: Analysis Guide for New JSONL Logs

The current `nbarzilie/benching/analysis.md` workflow:

- Reads raw `.log` files from `input/`.
- Cleans text blocks into `cleaned_input/`.
- Converts cleaned text to CSV.
- Plots mean TTFT and ITL.

The new workflow should be JSONL-first:

```text
/logs/pd_bench/<run_id>/**/results/*.jsonl
  -> csv/benchmark_results_jsonl.csv
  -> csv/benchmark_summary.csv
  -> csv/backend_pairwise.csv
  -> result_png/*.png
  -> result_html or markdown report
```

### JSONL Parser Requirements

Create or update an analysis script to recursively read:

```text
/logs/pd_bench/<run_id>/<backend>/<case>/<dataset>/results/c<concurrency>_r<rep>.jsonl
```

Each JSONL file may contain one or more records because `bench_serving` appends.
The parser should read all JSON lines and use the last complete record unless
the file contains multiple intentionally separate records.

Extract metadata from directory path:

- `run_id`
- `backend`
- `case`
- `dataset`
- `concurrency`
- `rep`

Also load `run_meta.json` from the parent directory and merge:

- model
- TP/DP sizes
- image
- node count
- backend order
- prompt scaling values
- timestamp

Extract raw result fields directly from JSON:

- `duration`
- `completed`
- `total_input_tokens`
- `total_output_tokens`
- `total_output_tokens_retokenized`
- `request_throughput`
- `input_throughput`
- `output_throughput`
- `output_throughput_retokenized`
- `total_throughput`
- `total_throughput_retokenized`
- `mean_ttft_ms`
- `median_ttft_ms`
- `std_ttft_ms`
- `p99_ttft_ms`
- `mean_tpot_ms`
- `median_tpot_ms`
- `std_tpot_ms`
- `p99_tpot_ms`
- `mean_itl_ms`
- `median_itl_ms`
- `std_itl_ms`
- `p95_itl_ms`
- `p99_itl_ms`
- `max_itl_ms`
- `max_output_tokens_per_s`
- `errors`

Normalize immediately into this canonical schema:

```text
duration                         -> benchmark_duration_s
completed                        -> successful_requests
total_input_tokens               -> input_tokens
total_output_tokens              -> output_tokens
total_output_tokens_retokenized  -> output_tokens_retokenized
request_throughput               -> request_throughput_req_s
input_throughput                 -> input_token_throughput_tok_s
output_throughput                -> output_token_throughput_tok_s
output_throughput_retokenized    -> output_token_throughput_retokenized_tok_s
total_throughput                 -> total_token_throughput_tok_s
total_throughput_retokenized     -> total_token_throughput_retokenized_tok_s
mean_ttft_ms                     -> mean_ttft_ms
median_ttft_ms                   -> median_ttft_ms
p99_ttft_ms                      -> p99_ttft_ms
mean_tpot_ms                     -> mean_tpot_ms
median_tpot_ms                   -> median_tpot_ms
p99_tpot_ms                      -> p99_tpot_ms
mean_itl_ms                      -> mean_itl_ms
median_itl_ms                    -> median_itl_ms
p95_itl_ms                       -> p95_itl_ms
p99_itl_ms                       -> p99_itl_ms
max_itl_ms                       -> max_itl_ms
max_output_tokens_per_s          -> max_output_tokens_per_s
```

After normalization, downstream CSVs, plots, and reports should use only the
canonical names. Treat missing optional keys as expected nulls. Treat missing
required comparison keys, such as `successful_requests`, throughput, or p99
latency metrics, as data-quality failures.

### Data Quality Checks

The analysis script should fail loudly or emit a clear warning for:

- Missing `run_meta.json`.
- Missing `fingerprint.txt`.
- Missing backend pair for a case/dataset/concurrency.
- Fewer than 5 reps in a completed main run.
- Any failed request.
- Any non-empty `errors` list.
- Mixed model values inside one comparison group.
- Mixed image values inside one comparison group.
- Mixed git SHA values if parsed from fingerprints.

Comparison group key:

```text
model, case, dataset, concurrency
```

Only compare `nixl` and `mooncake` within the same group.

### Aggregation

Produce one per-run CSV:

```text
csv/benchmark_results_jsonl.csv
```

Produce one summary CSV with:

```text
group columns:
  run_id
  model
  case
  dataset
  backend
  concurrency

metric columns for each selected metric:
  mean
  median
  std
  min
  max
  count
  coeff_var
```

For example:

```text
mean_ttft_ms_mean
mean_ttft_ms_median
mean_ttft_ms_std
p99_ttft_ms_mean
p99_ttft_ms_median
output_throughput_mean
output_throughput_median
```

Use median as the primary displayed value. Use mean and std as supporting
stability indicators.

### Pairwise Backend Comparison

Create:

```text
csv/backend_pairwise.csv
```

One row per:

```text
model, case, dataset, concurrency
```

Columns:

- `nixl_output_throughput_median`
- `mooncake_output_throughput_median`
- `output_throughput_ratio_nixl_over_mooncake`
- `nixl_p99_ttft_ms_median`
- `mooncake_p99_ttft_ms_median`
- `p99_ttft_ratio_nixl_over_mooncake`
- `nixl_p99_itl_ms_median`
- `mooncake_p99_itl_ms_median`
- `p99_itl_ratio_nixl_over_mooncake`
- `winner_throughput`
- `winner_latency`
- `winner_slo`

Ratio conventions:

- Throughput ratio greater than 1 means NIXL is faster.
- Latency ratio less than 1 means NIXL is lower latency.

### SLO Analysis

Create:

```text
csv/slo_summary.csv
```

Inputs:

```text
TTFT p99 SLO
TPOT p99 SLO
ITL p99 SLO
success rate SLO
```

For each:

```text
model, case, dataset, backend
```

Find:

- Highest concurrency satisfying all SLOs.
- Median output throughput at that concurrency.
- Median total throughput at that concurrency.
- Headroom to the SLO for TTFT/TPOT/ITL.

This is the most important decision table. It answers:

```text
Which backend serves more traffic before violating latency?
```

### Plots

Generate at least these plots per `model x case x dataset`:

1. Output throughput vs concurrency.
2. Total throughput vs concurrency.
3. p99 TTFT vs concurrency.
4. p99 TPOT vs concurrency.
5. p99 ITL vs concurrency.
6. SLO pass/fail vs concurrency.

Use log2 x-axis for concurrency:

```text
1, 2, 4, 8, 16, 32, 64, 128, 256
```

Plot median as the line. Show min/max or stddev as shaded error bands when
there are at least 3 reps.

### Report

Generate a Markdown report:

```text
report.md
```

Sections:

1. Run metadata: model, git SHA, image, nodes, date, package versions.
2. Data completeness: expected vs observed reps.
3. SLO summary table.
4. Pairwise backend winners.
5. Dataset-specific findings.
6. Case-specific findings.
7. Outliers and failed runs.
8. Links to plots and raw JSONL.

### Backward Compatibility

Keep the old log workflow as fallback:

- `clean_bench_logs.py`
- `cleaned_to_csv.py`
- `plot_bench_comparison.py`

But mark it as legacy for new runs. New automation should always pass
`--output-file` to `bench_serving`.

### Minimal Python Agent Modification Checklist

The Python analysis agent should:

- Stop treating `.log` as primary input for new runs.
- Recursively discover `results/*.jsonl`.
- Parse `backend/case/dataset/concurrency/rep` from paths.
- Merge `run_meta.json`.
- Preserve `fingerprint.txt` path in every row.
- Build per-run CSV.
- Build summary CSV using 5 reps.
- Build pairwise backend comparison CSV.
- Build SLO summary CSV.
- Plot median and variability, not only mean.
- Flag missing reps and failed requests before plotting.

## Done Definition

This benchmarking setup is done when:

1. `refresh_united.sh` creates a saved unified `.sqsh` with both backends.
2. The image validates imports for NIXL and Mooncake.
3. `fingerprint.sh` output is attached to every job and every benchmark batch.
4. The runner can execute at least the smoke matrix fully automatically.
5. The runner writes JSONL results with stable naming.
6. The runner cleans up prefill/decode/router processes after every batch.
7. The Slurm guide supports both interactive `srun` and `sbatch`.
8. The plan respects the maximum of two active jobs, two hours per job, and up
   to two nodes per job.
9. The analysis workflow reads JSONL directly and produces per-run, summary,
   pairwise, SLO, and plot outputs.
10. Logs are sufficient to reproduce or reject each benchmark result.
