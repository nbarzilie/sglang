# NIXL-SGLang Cluster Functional Test Summary

## Environment

- Cluster launch path: Slurm + Pyxis/Enroot SQSH container.
- Container image: `$MY/sqshs/sglang-nixl-functest.sqsh`.
- Mounted Hugging Face cache: `$MY/.cache/huggingface:/root/.cache/huggingface`.
- Mounted logs/cache scratch: `$MY/logs:/logs`.
- SGLang source inside container: `/workspace/sglang`.
- Branch: `feature/nixl-testing-suite`.
- Commit observed in runs: `b658340`.
- GPU node observed: 8 x A100-SXM4-80GB.
- Python environment confirmed:
  - `torch=2.11.0+cu130`
  - `torch.cuda.is_available() == True`
  - `torch.cuda.device_count() == 8`
  - `sglang`, `sglang_router`, and `nixl._api` import successfully.

## Working Interactive Container Command

This command successfully starts the SQSH container, mounts the HF cache and logs,
and redirects runtime/JIT caches away from the container writable overlay.

```bash
srun -A network_research_advdev \
     -N 1 \
     -p interactive \
     -t 2:00:00 \
     --gpus-per-node=8 \
     --cpus-per-task=32 \
     --mem=0 \
     --container-image=$MY/sqshs/sglang-nixl-functest.sqsh \
     --container-workdir=/workspace/sglang \
     --container-mounts=$MY/.cache/huggingface:/root/.cache/huggingface,$MY/logs:/logs \
     --pty bash -lc '
       mkdir -p /logs/{tmp,xdg-cache,sglang-cache,triton-cache,torchinductor-cache,nv-cache,tvm-ffi-cache} && \
       export HF_HOME=/root/.cache/huggingface \
              XDG_CACHE_HOME=/logs/xdg-cache \
              SGLANG_CACHE_DIR=/logs/sglang-cache \
              TRITON_CACHE_DIR=/logs/triton-cache \
              TORCHINDUCTOR_CACHE_DIR=/logs/torchinductor-cache \
              TVM_FFI_CACHE_DIR=/logs/tvm-ffi-cache \
              CUDA_CACHE_PATH=/logs/nv-cache \
              TMPDIR=/logs/tmp \
              LOG_DIR=/logs && \
       exec bash
     '
```

The important extra cache variable was:

```bash
TVM_FFI_CACHE_DIR=/logs/tvm-ffi-cache
```

Without it, SGLang JIT kernel compilation failed with:

```text
OSError: [Errno 122] Disk quota exceeded
```

The failing write came from `tvm_ffi/cpp/extension.py` while compiling a SGLang
JIT kernel during PD warmup.

## Issues Found And Fixes

### 1. SQSH Startup Failure

Initial Pyxis startup failed with:

```text
pyxis: container exited too soon
spank_pyxis.so: task_init() failed
```

The SQSH was valid but was created with LZO compression:

```text
Squashfs filesystem ... lzo compressed
```

The likely compatibility issue was the cluster's Enroot/Pyxis support for that
SquashFS compression. Recreating the SQSH with a compatible compressor fixed
container startup.

### 2. HF Hub Rate Limit

Initial SGLang launches failed while loading Qwen because the mounted HF cache was
incomplete and the container made unauthenticated requests:

```text
429 Too Many Requests
Local HF snapshot ... has no files matching ['*.safetensors', '*.bin', '*.pt']
```

Fix:

- Populate `$MY/.cache/huggingface` with the required model weights before the
  cluster run.
- Mount that cache to `/root/.cache/huggingface`.
- Use offline mode for cluster tests:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

### 3. Container Overlay / Cache Quota

After the HF cache was fixed, Qwen3-8B loaded successfully but startup failed
during warmup with:

```text
OSError: [Errno 122] Disk quota exceeded
```

The failure happened in SGLang JIT compilation:

```text
sglang/jit_kernel/resolve_future_token_ids.py
tvm_ffi/cpp/extension.py
```

Fix:

- Redirect all runtime caches and temp paths to `/logs`, which is mounted to the
  user's Lustre-backed log directory:

```bash
XDG_CACHE_HOME=/logs/xdg-cache
SGLANG_CACHE_DIR=/logs/sglang-cache
TRITON_CACHE_DIR=/logs/triton-cache
TORCHINDUCTOR_CACHE_DIR=/logs/torchinductor-cache
TVM_FFI_CACHE_DIR=/logs/tvm-ffi-cache
CUDA_CACHE_PATH=/logs/nv-cache
TMPDIR=/logs/tmp
LOG_DIR=/logs
```

### 4. 32B Memory Pool Failure

Qwen3-32B weights loaded successfully:

```text
Load weight end ... mem usage=61.04 GB
```

but SGLang failed when allocating the KV memory pool:

```text
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```

This was not a NIXL failure. It was SGLang memory pool sizing after loading a
large model on a single A100-80GB per PD worker.

Suggested next attempts:

```bash
--mem-fraction-static 0.75
```

or:

```bash
--mem-fraction-static 0.80 --context-length 8192
```

## Test 1: Qwen3-8B PD NIXL Smoke

Command run inside the container:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
KEEP_ALIVE=0 \
MODEL_PATH=Qwen/Qwen3-8B \
SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX \
DISAGG_IB_DEVICES=mlx5_0,mlx5_1 \
PREFILL_EXTRA_ARGS="--disable-cuda-graph --mem-fraction-static 0.55" \
DECODE_EXTRA_ARGS="--disable-cuda-graph --mem-fraction-static 0.55" \
run_qwen3_pd_nixl.sh
```

Result: passed.

What this validated:

- The container can import `sglang`, `sglang_router`, and `nixl._api`.
- Qwen3-8B loads from the mounted local HF cache.
- Prefill SGLang server starts on GPU 0.
- Decode SGLang server starts on GPU 1.
- PD router starts.
- NIXL UCX backend is instantiated.
- `NIXL KVManager initialized with backend: UCX` appears.
- A `/generate` request through the PD router succeeds.
- `KEEP_ALIVE=0` exits cleanly after the smoke request.

This is the main successful NIXL-SGLang functional validation so far.

## Test 2: Registered Basic NIXL Test

Original test:

```text
test/registered/disaggregation/test_disaggregation_nixl_basic.py
```

Initial result: failed in offline mode because the test uses:

```python
DEFAULT_SMALL_MODEL_NAME_FOR_TEST = "meta-llama/Llama-3.2-1B-Instruct"
```

The mounted cache only contained Qwen models.

Temporary local edit inside the container:

```python
cls.model = "Qwen/Qwen3-8B"
```

Command:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX \
run_disaggregation_nixl_basic_test.sh
```

Result after edit: passed.

Observed success:

```text
Ran 1 test in 108.898s
OK
```

What this validated:

- The registered test launches real SGLang PD servers.
- It starts prefill, decode, and router.
- It performs real inference through `/generate`.
- It verifies HTTP 200, non-empty generated text, and post-request liveness of
  router, prefill, and decode workers.
- NIXL UCX is initialized during the server runs.

Limitations:

- The edit was made inside the live container and is not persistent in the SQSH.
- The test class sets `cls.rdma_devices = []`, so it does not explicitly validate
  `DISAGG_IB_DEVICES=mlx5_0,mlx5_1`.
- Output correctness is only checked as non-empty text, not accuracy.

## Test 3: Decode Radix Cache NIXL Test

Test file:

```text
test/registered/distributed/test_disaggregation_decode_radix_cache.py
```

The NIXL class is skipped by default:

```python
@unittest.skip("Temporarily disabled until nixl backend is stable.")
class TestDisaggregationDecodeRadixCacheNixl(...)
```

For local experimentation, the skip was removed and the model was changed to:

```python
cls.model = "Qwen/Qwen3-8B"
```

The targeted method was:

```text
TestDisaggregationDecodeRadixCacheNixl.test_decode_radix_cache_hits_and_workers_stay_alive
```

Observed progress:

- Qwen3-8B loaded from local HF cache.
- Prefill and decode servers started.
- NIXL UCX backend instantiated.
- Router started.
- `/server_info` checks succeeded.
- Cache flush requests succeeded.

Failure:

```text
Cannot reach https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json: offline mode is enabled.
```

Root cause:

- `run_multiturn_cache_hit_test` samples prompts from the ShareGPT dataset.
- That dataset file was not present in the mounted HF cache.
- Offline mode correctly blocked network access.

Required extra cached artifact:

```bash
HF_HOME=$MY/.cache/huggingface \
huggingface-cli download \
  anon8231489123/ShareGPT_Vicuna_unfiltered \
  ShareGPT_V3_unfiltered_cleaned_split.json \
  --repo-type dataset
```

Then verify inside the container:

```bash
find /root/.cache/huggingface -name 'ShareGPT_V3_unfiltered_cleaned_split.json' | head
```

Status:

- Server startup and NIXL initialization for this heavier test succeeded.
- The actual cache-hit workload did not run because the ShareGPT dataset was
  missing from offline cache.

## Current Validation Status

Confirmed working:

- SQSH container starts on the cluster.
- GPU visibility works.
- SGLang and NIXL import correctly.
- Qwen3-8B loads from local mounted HF cache.
- PD NIXL smoke with Qwen3-8B passes.
- Registered basic NIXL test passes after changing the model to cached Qwen3-8B.
- Runtime/JIT cache quota issue is fixed by routing caches to `/logs`, including
  `TVM_FFI_CACHE_DIR`.

Not yet fully validated:

- Qwen3-32B PD NIXL inference, because memory pool sizing failed with
  `--mem-fraction-static 0.55`.
- Decode radix cache cache-hit workload, because ShareGPT dataset is missing from
  offline HF cache.
- NIXL with explicit RDMA devices inside the registered basic test, because that
  test overrides RDMA device args.
- Accuracy tests such as GSM8K two-pass in `test_disaggregation_decode_radix_cache.py`.

## Recommended Next Steps

1. Cache the ShareGPT dataset file if continuing decode radix cache testing.
2. Re-run only:

```bash
python3 test/registered/distributed/test_disaggregation_decode_radix_cache.py \
  TestDisaggregationDecodeRadixCacheNixl.test_decode_radix_cache_hits_and_workers_stay_alive
```

3. Retry Qwen3-32B PD smoke with a larger memory fraction:

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
KEEP_ALIVE=0 \
MODEL_PATH=Qwen/Qwen3-32B \
SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX \
DISAGG_IB_DEVICES=mlx5_0,mlx5_1 \
PREFILL_EXTRA_ARGS="--disable-cuda-graph --mem-fraction-static 0.80 --context-length 8192" \
DECODE_EXTRA_ARGS="--disable-cuda-graph --mem-fraction-static 0.80 --context-length 8192" \
run_qwen3_pd_nixl.sh
```

4. If these edits need to be repeatable, update the source branch and rebuild the
   Docker image and SQSH. In-container `nano` edits are only session-local.

## What Was Actually Proven

This section separates evidence-backed conclusions from things that were only
partially exercised.

### Container And Cluster Runtime

Proven working:

- Slurm can allocate the interactive job with the SQSH image.
- Pyxis/Enroot can start the SQSH container after using a compatible SQSH image.
- The container starts in `/workspace/sglang`.
- The baked SGLang checkout exists in the container.
- The mounted Hugging Face cache is visible at `/root/.cache/huggingface`.
- The mounted log/cache directory is visible at `/logs`.
- Runtime writes can be redirected to `/logs`, avoiding the container overlay
  quota.

Evidence:

```text
Using SGLang source: /workspace/sglang
SGLang git branch: feature/nixl-testing-suite
SGLang git commit: b658340
```

and successful server startup after setting:

```bash
TVM_FFI_CACHE_DIR=/logs/tvm-ffi-cache
```

### GPU And Python Runtime

Proven working:

- CUDA is visible inside the container.
- All 8 GPUs are visible to PyTorch.
- SGLang imports.
- SGLang router imports.
- NIXL Python bindings import.

Evidence:

```text
torch=2.11.0+cu130 cuda_available=True cuda_devices=8
sglang, router, nixl imports OK
```

This proves the container is suitable for running GPU SGLang jobs and that the
NIXL Python package is installed and importable.

### Hugging Face Offline Model Cache

Proven working for:

- `Qwen/Qwen3-8B`

Evidence:

```text
Found local HF snapshot for Qwen/Qwen3-8B at /root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/...; skipping download.
Load weight end. elapsed=...
```

Proven partially for:

- `Qwen/Qwen3-32B`

Evidence:

```text
Found local HF snapshot for Qwen/Qwen3-32B ...
Multi-thread loading shards: 100% Completed | 17/17
Load weight end ...
```

This proves the 32B weights are cached and readable. It does not prove 32B PD
serving works, because the run later failed during KV memory-pool sizing.

Not proven / missing:

- `meta-llama/Llama-3.2-1B-Instruct` was not available in the mounted cache.
- `meta-llama/Llama-3.1-8B-Instruct` was not available in the mounted cache.
- The ShareGPT dataset used by the decode radix cache test was not available in
  the mounted cache.

### SGLang Single-Model Loading

Proven working:

- Two independent SGLang server processes can load Qwen3-8B concurrently, one on
  GPU 0 and one on GPU 1.
- Each process can allocate model weights and KV cache.
- CUDA graph capture can complete when cache paths are redirected to `/logs`.
- SGLang warmup requests can complete.

Evidence from the registered basic test and radix-cache test startup:

```text
Load weight end ... type=Qwen3ForCausalLM
KV Cache is allocated
Capture cuda graph end
The server is fired up and ready to roll!
```

In the explicit PD smoke command, CUDA graph was disabled for the faster/smaller
smoke path:

```bash
--disable-cuda-graph --mem-fraction-static 0.55
```

In the edited registered tests, CUDA graph capture did run and completed for
Qwen3-8B after the cache directory fix.

### PD Disaggregation Server Startup

Proven working:

- Prefill server starts.
- Decode server starts.
- Decode uses GPU 1 via `--base-gpu-id 1`.
- Prefill and decode both become healthy.
- Router starts in PD mode.
- Router becomes healthy.

Evidence:

```text
Uvicorn running on http://127.0.0.1:21100
Uvicorn running on http://127.0.0.1:21200
Server http://127.0.0.1:21100/health is ready
Server http://127.0.0.1:21200/health is ready
Uvicorn running on http://127.0.0.1:21000
Server http://127.0.0.1:21000/health is ready
```

The functional script used ports:

```text
prefill: 30100
decode:  30200
router:  30000
bootstrap: 30500
```

The registered tests used fixture ports:

```text
prefill: 21100
decode:  21200
router:  21000
bootstrap: 21500
```

Both port layouts worked.

### NIXL Backend Initialization

Proven working:

- `nixl._api` imports.
- NIXL agents are created during SGLang PD server startup.
- UCX backend is instantiated by NIXL.
- SGLang's `NIXL KVManager` initializes successfully with backend `UCX`.

Evidence:

```text
NIXL INFO _api.py:247 Initialized NIXL agent: ...
NIXL INFO _api.py:369 Backend UCX was instantiated
NIXL KVManager initialized with backend: UCX
```

This proves the SGLang-NIXL integration gets through import, agent creation,
backend creation, and manager initialization.

### PD NIXL Inference Path

Proven working:

- A request sent to the PD router completes successfully with Qwen3-8B.
- The request crosses the PD path: router -> prefill/decode servers.
- Prefill and decode workers remain alive after the request.
- The registered basic NIXL test saw a real `/generate` response and passed.

Evidence:

```text
POST /generate HTTP/1.1" 200 OK
Ran 1 test in 108.898s
OK
```

The registered basic test verified:

- HTTP status `200`
- response contains `"text"`
- generated text length is greater than zero
- load balancer health remains OK
- prefill health remains OK
- decode health remains OK

Therefore, for Qwen3-8B on this cluster/container setup, basic SGLang PD
inference with NIXL backend is confirmed.

### Explicit RDMA Device Path

Partially proven:

- The standalone PD smoke script accepted:

```bash
DISAGG_IB_DEVICES=mlx5_0,mlx5_1
```

- The servers started and the Qwen3-8B PD smoke passed with this argument.

Not fully proven:

- The registered basic test did not validate explicit RDMA devices because the
  test class forces:

```python
cls.rdma_devices = []
```

- We did not independently inspect UCX/NIXL transport selection to prove traffic
  used a specific NIC or RDMA path rather than another UCX transport.

So the correct conclusion is:

- SGLang accepts the explicit device argument in the functional script.
- NIXL UCX initializes successfully with that setup.
- The end-to-end request works.
- Specific wire-level RDMA/NIC usage was not proven.

### Decode Radix Cache Test

Partially proven:

- After local edits, the NIXL decode radix cache test class could launch servers.
- Decode radix cache test startup reached server health and router health.
- `/server_info` checks succeeded.
- cache flush endpoints succeeded.

Evidence:

```text
TestDisaggregationDecodeRadixCacheNixl.test_decode_radix_cache_hits_and_workers_stay_alive
GET /server_info HTTP/1.1" 200 OK
Cache flushed successfully!
```

Not proven:

- Actual multiturn cache-hit workload did not run.
- `total_cached_tokens > 0` was not evaluated.
- post-workload liveness was not evaluated.
- GSM8K two-pass accuracy was not run.

Reason:

The test attempted to load the ShareGPT dataset:

```text
anon8231489123/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json
```

but offline mode was enabled and the dataset was missing from the mounted cache.

### Qwen3-32B

Proven working:

- Qwen3-32B model files are present in the mounted cache.
- Qwen3-32B weights can be loaded by SGLang on one A100-80GB process.

Evidence:

```text
Multi-thread loading shards: 100% Completed | 17/17
Load weight end ... mem usage=61.04 GB
```

Not proven:

- Qwen3-32B PD serving did not complete.
- Qwen3-32B NIXL transfer path did not run.

Reason:

SGLang failed during KV memory-pool allocation:

```text
RuntimeError: Not enough memory. Please try to increase --mem-fraction-static.
```

This is a memory sizing/configuration issue, not a NIXL initialization issue.

### What Works For Sure Now

Based on successful runs, the following can be stated confidently:

- The SQSH image can run on the cluster.
- The cache/log mount strategy works.
- The container sees GPUs and can run CUDA SGLang.
- NIXL is installed and importable.
- NIXL UCX backend can be instantiated inside SGLang.
- Qwen3-8B can be loaded offline from the mounted HF cache.
- Qwen3-8B PD prefill/decode servers can start.
- The PD router can route to prefill and decode servers.
- A real `/generate` request through PD disaggregation succeeds with Qwen3-8B.
- The edited registered basic NIXL test passes end-to-end with Qwen3-8B.
- The quota problem is solved by putting `TVM_FFI_CACHE_DIR` and other caches on
  `/logs`.

### What Still Needs Work

- Cache the ShareGPT dataset before rerunning decode radix cache tests offline.
- Either cache Meta Llama models or keep editing tests to use Qwen.
- Tune Qwen3-32B memory settings before expecting PD inference to pass.
- If exact RDMA/NIC validation is required, add UCX/NIXL transport logging or a
  lower-level transfer/traffic check.
