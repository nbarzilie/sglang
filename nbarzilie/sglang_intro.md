# SGLang Development Intro for Coding Agents

This is the minimum practical context an agent should have before changing SGLang. Treat it as an onboarding map, not a replacement for reading nearby code.

## 1. Source Of Truth To Read First

Start with these files before making broad changes:

- `README.md`: project purpose, major features, install links, contribution entry points.
- `python/sglang/README.md`: high-level Python package structure.
- `docs/README.md`: documentation workflow and docs style rules.
- `docs/developer_guide/contribution_guide.md`: contribution workflow, tests, code style, CI triggering, kernel update process.
- `docs/get_started/install.md`: source, pip/uv, Docker, platform install patterns.
- `test/README.md`, `test/registered/README.md`, `test/registered/unit/README.md`: test layout, local execution, registration rules, CI suites.
- `docs/developer_guide/benchmark_and_profiling.md`: benchmark tools and profiler workflows.
- `docs/advanced_features/server_arguments.md`: launch flags and runtime configuration.
- `docs/references/environment_variables.md`: environment variables used for tuning, debugging, profiling, and platform behavior.
- `docs/supported_models/extending/support_new_models.md`: model extension workflow.
- `docs/developer_guide/development_jit_kernel_guide.md` and `sgl-kernel/README.md`: JIT and AOT kernel development.
- `docs/advanced_features/sgl_model_gateway.md` and `sgl-model-gateway/README.md`: Rust gateway/router architecture.
- `docs/diffusion/index.md`, `docs/diffusion/contributing.md`, `docs/diffusion/support_new_models.md`: diffusion stack and contribution rules.

For tests, also read `nbarzilie/sglang_test_guide.md`; it is more detailed than this intro.

## 2. Project Mental Model

SGLang is a high-performance serving framework for LLMs, VLMs, embedding/rerank/classification models, and diffusion models. The core runtime goal is low-latency and high-throughput inference, so small overheads in request scheduling, model forward, KV cache management, sampling, and tokenization matter.

Major pieces:

- `python/sglang/lang`: frontend language APIs.
- `python/sglang/srt`: SGLang Runtime, the main backend engine for local model serving.
- `python/sglang/multimodal_gen`: SGLang Diffusion image/video generation framework.
- `python/sglang/jit_kernel`: runtime-compiled CUDA/C++ kernels.
- `sgl-kernel`: separately packaged AOT kernel library; Python import path is `sgl_kernel`.
- `sgl-model-gateway`: Rust model gateway/router for worker management, load balancing, OpenAI-compatible routing, gRPC routing, PD routing, MCP, metrics, and history connectors.
- `docs`: Sphinx documentation, mostly notebooks for executable examples.
- `test/registered`: CI-discovered tests.
- `benchmark`: accuracy and performance benchmark scripts.
- `docker`, `scripts/ci`, `scripts/release`: deployment, CI, and release automation.

The default serving path is:

```text
sglang serve / python -m sglang.launch_server
  -> sglang.srt.server_args.prepare_server_args
  -> sglang.srt.entrypoints.http_server.launch_server
  -> TokenizerManager / TemplateManager
  -> scheduler subprocesses
  -> ScheduleBatch / Req state
  -> ModelRunner
  -> model forward / attention backend / KV cache / sampler
  -> detokenizer and HTTP/OpenAI-compatible response streaming
```

Important source entry points:

- `python/sglang/launch_server.py`: legacy module entrypoint; `sglang serve` is the recommended CLI, but this path still matters.
- `python/sglang/srt/server_args.py`: argument definitions, validation, auto-selection, feature flags.
- `python/sglang/srt/entrypoints/http_server.py`: FastAPI HTTP/OpenAI/Ollama/Anthropic/admin endpoints.
- `python/sglang/srt/entrypoints/engine.py`: engine process wiring.
- `python/sglang/srt/managers/tokenizer_manager.py`: request intake, tokenization, API-level routing into the backend.
- `python/sglang/srt/managers/scheduler.py`: central scheduling loop for a tensor-parallel worker.
- `python/sglang/srt/managers/schedule_batch.py`: request and batch state.
- `python/sglang/srt/managers/io_struct.py`: request/response/control message structures.
- `python/sglang/srt/model_executor/model_runner.py`: model loading, distributed setup, CUDA graph capture, forward execution.
- `python/sglang/srt/models/registry.py` and `python/sglang/srt/models/*.py`: model implementations.
- `python/sglang/srt/layers/attention`: attention backend implementations and selection.
- `python/sglang/srt/mem_cache`: KV memory pools and radix/prefix cache.
- `python/sglang/srt/disaggregation`: prefill/decode disaggregation and KV transfer backends.

## 3. Local Setup

Common source install:

```bash
pip install --upgrade pip
pip install -e "python"
```

For test dependencies:

```bash
pip install -e "python[test]"
```

For diffusion:

```bash
pip install -e "python[diffusion]"
```

The docs recommend `uv` for faster installs. For GPU development, the dev Docker image is often the cleanest environment:

```bash
docker run -itd --shm-size 32g --gpus all \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/src/sglang:/sgl-workspace/sglang \
  --ipc=host --network=host --privileged \
  --name sglang_dev lmsysorg/sglang:dev /bin/zsh
docker exec -it sglang_dev /bin/zsh
```

Use `--shm-size`/`--ipc=host` for Docker because SGLang uses shared memory for process communication. Mount Hugging Face cache to avoid repeated downloads. For B300/GB300/CUDA 13, prefer the documented CUDA 13 images and do not reinstall editable dependencies inside those images unless you know the dependency implications.

Basic server launch:

```bash
python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --host 0.0.0.0 --port 30000
```

Useful common flags:

- `--tp N`: tensor parallelism.
- `--dp N`: data parallelism, usually through `sglang_router.launch_server` / model gateway.
- `--mem-fraction-static 0.7`: lower KV/static memory use when hitting OOM.
- `--chunked-prefill-size 4096`: reduce prefill memory pressure for long prompts.
- `--attention-backend triton|flashinfer|fa3|fa4|trtllm_mla|...`: force attention backend.
- `--kv-cache-dtype fp8_e4m3|fp8_e5m2`: quantized KV cache.
- `--enable-multimodal`: enable multimodal pipelines for VLMs.
- `--trust-remote-code`: needed for some HF models, but use deliberately.
- `--load-format dummy`: benchmark configs without real weights.
- `--json-model-override-args '{"num_hidden_layers": 1}'`: shrink configs for fast profiling.

Run `python -m sglang.launch_server --help` for the complete current argument list.

## 4. Code Style And Runtime Constraints

SGLang code is performance-critical. Follow the contribution guide's rules:

- Avoid code duplication. Extract shared helpers when the same nontrivial block appears repeatedly.
- Minimize CPU-GPU synchronization. Avoid `tensor.item()`, `tensor.cpu()`, and shape/value reads on hot paths unless required.
- Cache invariant runtime checks in `__init__` or setup paths, especially inside model forward code.
- Keep functions as pure as practical. Avoid mutating arguments unless the local pattern expects it.
- Keep files concise. If a file grows beyond roughly 2,000 lines, split by responsibility.
- Put core data structures near the top of a file and utilities near the bottom.
- Keep tests fast. Split test files over 500 seconds or CI jobs over 30 minutes.
- Never deserialize untrusted or network data with `pickle.loads`, `pickle.load`, or `recv_pyobj`; use safe formats such as JSON or msgpack/msgspec.
- When adding hardware-specific behavior, prefer new focused files and keep the common NVIDIA/existing path first in conditionals.
- Respect existing abstractions. Prefer nearby helper APIs over inventing a new layer.

Before pushing:

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```

If pre-commit modifies files or fails after fixing, run it again.

## 5. Testing Rules

Use the smallest test that can observe the behavior.

Placement:

- Pure component logic: `test/registered/unit/<module>/`.
- Server/API/model integration: category under `test/registered/`, such as `core`, `openai_server`, `lora`, `spec`, `distributed`, `disaggregation`, `vlm`.
- Manual/debug-only scripts: `test/manual/`.
- JIT kernel tests: `python/sglang/jit_kernel/tests/test_*.py`.
- AOT `sgl-kernel` tests: `sgl-kernel/tests/`.

Unit tests should not launch a server or load model weights. Mirror the source tree where possible:

```text
python/sglang/srt/mem_cache/radix_cache.py
  -> test/registered/unit/mem_cache/test_radix_cache.py
```

Every CI-discovered test file needs module-level registration with literal values so `run_suite.py` can parse it:

```python
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-small")
```

Use `CustomTestCase` for new unittest-style tests:

```python
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestThing(CustomTestCase):
    def test_behavior(self):
        self.assertEqual(1 + 1, 2)


if __name__ == "__main__":
    unittest.main()
```

Direct execution matters. End files with exactly a standard `unittest.main()` or `pytest.main([__file__])` block. Do not add custom argparse or mutate `sys.argv`; CI appends failfast flags.

Local test commands:

```bash
pytest test/registered/unit/ -v
pytest test/registered/unit/mem_cache/ -v
python3 test/registered/core/test_srt_endpoint.py
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode
python3 python/sglang/jit_kernel/tests/test_add_constant.py
python3 test/run_suite.py --hw cpu --suite base-a-test-cpu
python3 test/run_suite.py --hw cuda --suite base-b-test-1-gpu-small
```

Coverage commands:

```bash
pytest test/registered/unit/ --cov --cov-config=.coveragerc -v
pytest test/registered/unit/ --cov --cov-config=.coveragerc --cov-report=xml
diff-cover coverage.xml --compare-branch=origin/main --fail-under=60
```

CI suite selection:

- CPU/no GPU: `base-a-test-cpu`.
- Most 1-GPU tests: `base-b-test-1-gpu-small`.
- Large memory or Hopper features: `base-b-test-1-gpu-large`.
- Multi-GPU: `base-b-test-2-gpu-large`, `base-c-test-*`.
- Long or experimental: nightly suites.

When launching subprocesses in tests, clean them defensively even if setup fails.

## 6. Benchmarking And Profiling

Benchmark tools:

- `sglang.bench_serving`: default choice for realistic online serving with TTFT, TPOT, ITL, throughput. Requires a running server.
- `sglang.bench_one_batch_server`: one HTTP batch to a running server; useful but not steady-state by default.
- `sglang.bench_offline_throughput`: in-process `Engine`, no HTTP overhead, measures max throughput.
- `sglang.bench_one_batch`: lowest-level static batch via `ModelRunner`, good for kernel profiling, unrealistic for scheduler behavior.

Default benchmark command:

```bash
python3 -m sglang.bench_serving \
  --backend sglang \
  --max-concurrency 16 \
  --num-prompts 80 \
  --random-input-len 256 \
  --random-output-len 32 \
  --dataset-name random
```

For `bench_serving`, use `num-prompts >= 5 * max-concurrency` to reduce warmup bias.

PyTorch profiler:

```bash
export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_profiles
python -m sglang.bench_serving --backend sglang --num-prompts 10 --profile
```

Direct profiler control:

```bash
python -m sglang.profiler --num-steps 10 --cpu --gpu --output-dir /tmp/sglang_profiles
curl -X POST http://127.0.0.1:30000/start_profile -H "Content-Type: application/json" -d '{"num_steps": 10}'
curl -X POST http://127.0.0.1:30000/stop_profile
```

Use `--disable-cuda-graph` when you need source-level CUDA kernel attribution or layerwise NVTX visibility. View traces in Perfetto or Chrome tracing.

Nsight example:

```bash
nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node \
  python3 -m sglang.bench_one_batch \
  --model meta-llama/Meta-Llama-3-8B \
  --batch-size 64 \
  --input-len 512
```

## 7. Model Development

For a new text model, usually add one file under:

```text
python/sglang/srt/models/
```

Start from a similar existing implementation, often Llama/Qwen/DeepSeek. SGLang reuses many vLLM-style interfaces, but do not rely on vLLM components in new SGLang model files.

Porting checklist:

- Replace vLLM `Attention` with SGLang `RadixAttention` and pass `layer_id`.
- Use SGLang `LogitsProcessor`.
- Use SGLang layers such as RMSNorm/SiluAndMul equivalents.
- Remove vLLM `Sample`.
- Implement the SGLang `forward` signature and add `forward_batch` behavior where required.
- Add `EntryClass = YourModelClass` at the end of the file.
- Register/support the architecture through the existing registry path.
- Update supported model docs.
- Add model tests and report accuracy/performance in the PR.

For multimodal models:

- Mark the architecture as multimodal in model config logic.
- Add or select a chat template if the tokenizer template cannot accept images.
- Implement/register a `BaseMultimodalProcessor`.
- Implement `pad_input_ids` so multimodal tokens are expanded and padded with multimodal data hashes for RadixAttention.
- Implement image/video/audio feature extraction as needed.
- Use `VisionAttention` for vision attention paths.

Testing a new model:

```bash
python3 scripts/playground/reference_hf.py --model-path <new-model> --model-type text
python3 -m sglang.bench_one_batch --correct --model <new-model>
ONLY_RUN=<model-id> python3 -m unittest test_generation_models.TestGenerationModels.test_others
```

Accuracy/eval commands commonly used:

```bash
python -m sglang.test.few_shot_gsm8k --host 127.0.0.1 --port 30000 --num-questions 200 --num-shots 5
python -m sglang.test.run_eval --eval-name mmlu --port 30000 --num-examples 1000 --max-tokens 8192
python benchmark/hellaswag/bench_sglang.py --host 127.0.0.1 --port 30000 --num-questions 200 --num-shots 20
```

For external/private model implementations, use `SGLANG_EXTERNAL_MODEL_PACKAGE`. For external multimodal models, also use `SGLANG_EXTERNAL_MM_MODEL_ARCH` and `SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`.

## 8. Attention, Memory, And Scheduling

SGLang has many attention backends. Let auto-selection work unless the change is backend-specific or you are debugging:

- MHA CUDA defaults: Hopper often `fa3`, Blackwell often `trtllm_mha`, other CUDA often `flashinfer` then `triton`.
- MLA CUDA defaults: Hopper often `fa3`, Blackwell often optimized DeepSeek paths, other CUDA often `triton`.
- Multimodal attention is selected with `--mm-attention-backend`.
- Hybrid prefill/decode attention can be controlled with `--prefill-attention-backend` and `--decode-attention-backend`.
- Page size affects prefix cache reuse and attention kernel performance. `page_size=1` maximizes token-level reuse; larger page sizes can improve kernels but reduce prefix-cache granularity.

Important runtime areas:

- `mem_cache`: KV pool allocation, radix cache, prefix caching, offloading.
- `schedule_policy`: request ordering and batching policy.
- `scheduler_components`: metrics, streaming, profiling, request receiver, watchdog, flush, DP attention adapters.
- `disaggregation`: prefill/decode modes and transfer backends (`mooncake`, `nixl`, `ascend`, `fake`, `mori`).
- `lora`: adapter registration, batching, eviction, overlap loading.
- `constrained`: structured output and grammar backends (`xgrammar`, `outlines`, `llguidance`).
- `sampling`: sampling params and sampling backend behavior.

If a change touches scheduler state, KV cache, attention, batch construction, or model forward, assume it can affect correctness, latency, throughput, memory, and distributed behavior.

## 9. Kernel Development

There are two kernel stacks:

- `sgl-kernel`: AOT package, built separately and released as `sglang-kernel`; import path `sgl_kernel`.
- `python/sglang/jit_kernel`: runtime JIT kernels loaded through `load_jit`.

For `sgl-kernel`:

1. Implement under `sgl-kernel/csrc`.
2. Expose in `sgl-kernel/include/sgl_kernel_ops.h`.
3. Bind in `sgl-kernel/csrc/common_extension.cc`.
4. Update `CMakeLists.txt`.
5. Expose Python interface under `sgl-kernel/python/sgl_kernel`.
6. Add tests and benchmarks.

Build:

```bash
cd sgl-kernel
make build
make build MAX_JOBS=2 CMAKE_ARGS="-DSGL_KERNEL_COMPILE_THREADS=1"
```

For JIT kernels:

- C++/CUDA source: `python/sglang/jit_kernel/csrc`.
- Shared headers: `python/sglang/jit_kernel/include`.
- Python wrappers: `python/sglang/jit_kernel`.
- Use `load_jit` to compile/load and `cache_once` instead of `functools.lru_cache`.
- Validate dtype/device/shape in Python before launching when possible.
- Use `TensorMatcher`, `RuntimeCheck`, `RuntimeDeviceCheck`, and `LaunchKernel` helpers.

Generate clangd config:

```bash
python -m sglang.jit_kernel
```

Kernel package update process is multi-PR: first update `sgl-kernel` source without using it from SGLang, then bump/release `sglang-kernel`, then update `python/pyproject.toml` and callers to consume it.

## 10. Docs Workflow

Most docs live under `docs/`. For common features, prefer executable Jupyter notebooks; for complex/distributed features, Markdown is acceptable.

Docs dependencies:

```bash
cd docs
pip install -r requirements.txt
# Linux: apt-get install pandoc parallel retry
# macOS: brew install pandoc parallel retry
```

Build:

```bash
make compile
make html
make markdown
bash serve.sh
```

Rules:

- Update `index.rst` or the relevant `.rst` file when adding pages.
- Prefer relative links, not absolute `https://docs.sglang.io/...` links.
- Keep notebooks fast: small models, reused servers, minimal expensive cells.
- Clean notebook outputs before PRs:

```bash
pip install nbstripout
find . -name '*.ipynb' -exec nbstripout {} \;
```

## 11. Model Gateway

The Rust gateway in `sgl-model-gateway` is a separate but integrated control/data plane:

- Worker lifecycle, health, registry, load monitoring.
- HTTP, PD, gRPC, and OpenAI-compatible routing.
- Load balancing policies: random, round robin, cache aware, power of two, bucket/manual variants.
- Reliability: retries, circuit breakers, token bucket rate limiting, queues.
- OpenAI-compatible APIs including chat, completions, responses, embeddings, rerank, classify, tokenize/detokenize.
- Native Rust tokenizer/reasoning/tool parser path for gRPC routing.
- MCP integration, history connectors, WASM middleware, Prometheus/OpenTelemetry.

Build:

```bash
cd sgl-model-gateway
cargo build --release
```

Common launch:

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --policy cache_aware \
  --host 0.0.0.0 --port 30000
```

PD routing:

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://prefill1:30001 \
  --decode http://decode1:30011 \
  --policy cache_aware
```

If changing gateway code, expect Rust tests under `sgl-model-gateway/tests` and Python launcher compatibility to matter.

## 12. Diffusion Stack

Diffusion lives in `python/sglang/multimodal_gen`. It provides `sglang generate`, `sglang serve`, OpenAI-compatible image/video APIs, diffusers backend support, optimized kernels, scheduler improvements, and caching acceleration.

New diffusion models use a pipeline architecture:

- `ComposedPipeline`: model-level pipeline orchestration.
- `PipelineStage`: modular stage abstraction.
- Common stages: denoising and decoding.
- Model-specific pre-processing can be a single `{Model}BeforeDenoisingStage` or a composition of standard stages.
- Static model config: `PipelineConfig`.
- Runtime parameters: `SamplingParams`.
- Components: DiT/transformer, encoders, VAE, scheduler, processors.

Implementation paths:

- DiT/transformer: `python/sglang/multimodal_gen/runtime/models/dits/`.
- Encoders: `runtime/models/encoders/`.
- VAEs: `runtime/models/vaes/`.
- Schedulers: `runtime/models/schedulers/`.
- Pipeline configs: `configs/pipeline_configs/`.
- Sampling params: `configs/sample/`.
- Registry: `python/sglang/multimodal_gen/registry.py`.

Diffusion PRs affecting latency, throughput, or memory should include a perf comparison:

```bash
sglang generate --model-path <model> --prompt "A benchmark prompt" --perf-dump-path baseline.json
sglang generate --model-path <model> --prompt "A benchmark prompt" --perf-dump-path new.json
python python/sglang/multimodal_gen/benchmarks/compare_perf.py baseline.json new.json
```

Diffusion commit messages should use:

```text
[diffusion] <scope>: <subject>
```

## 13. Environment Variables Worth Knowing

General:

- `SGLANG_USE_MODELSCOPE`: use ModelScope.
- `SGLANG_HOST_IP`, `SGLANG_PORT`: server host/port.
- `SGLANG_CACHE_DIR`: SGLang cache directory.
- `SGLANG_HEALTH_CHECK_TIMEOUT`: health check timeout.

Performance/debug:

- `SGLANG_IS_FLASHINFER_AVAILABLE`: override FlashInfer availability.
- `SGLANG_SKIP_P2P_CHECK`: skip peer access checks.
- `SGLANG_SET_CPU_AFFINITY`: CPU affinity.
- `SGLANG_RECORD_STEP_TIME`: record scheduler step timing.
- `SGLANG_DEBUG_MEMORY_POOL`: memory pool debugging.
- `SGLANG_PROFILE_WITH_STACK`, `SGLANG_PROFILE_RECORD_SHAPES`: profiler detail controls.

Profiling:

- `SGLANG_TORCH_PROFILER_DIR`: profiler output directory.

Distributed:

- `SGLANG_BLOCK_NONZERO_RANK_CHILDREN`.
- `SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS`.
- `SGLANG_PP_LAYER_PARTITION`.

Kernels:

- `SGLANG_USE_CUSTOM_TRITON_KERNEL_CACHE`.
- `SGLANG_KERNEL_API_LOGLEVEL`, `SGLANG_KERNEL_API_LOGDEST`, `SGLANG_KERNEL_API_DUMP_DIR`.
- `SGLANG_ENABLE_JIT_DEEPGEMM`, `SGLANG_DG_CACHE_DIR`.

Tool/reasoning:

- `SGLANG_TOOL_STRICT_LEVEL`.
- `SGLANG_FORWARD_UNKNOWN_TOOLS`.

There are many feature-specific variables for DeepGEMM, DeepEP, MoRI, NSA, multimodal hashing, quantization, PD staging buffers, and platform backends. Check `docs/references/environment_variables.md` before adding or changing env-driven behavior.

## 14. Agent Workflow Before Editing

Use this checklist for most tasks:

1. Identify the user-facing behavior and the smallest source area.
2. Read the requested files plus nearby source, tests, and docs.
3. Check for existing patterns with `rg` before adding new abstractions.
4. If touching `srt`, decide whether the change is API, scheduler, model, cache, attention, sampling, tokenizer, distributed, or observability.
5. Add or update the lightest test that observes the behavior.
6. Register CI tests with literal registration values.
7. Run targeted tests first; run broader suites only when risk justifies it.
8. Run pre-commit if files or docs changed.
9. For performance-sensitive changes, run an appropriate benchmark and report methodology.
10. For docs changes, keep links relative and update indexes.

High-risk areas that require extra care:

- Scheduler request lifecycle.
- KV cache allocation/release and radix cache.
- Model forward signatures and logits processing.
- Attention backend dispatch and page-size assumptions.
- CUDA graph capture paths.
- Distributed initialization and rank-specific code.
- PD disaggregation metadata and KV transfer.
- LoRA loading/eviction/batching.
- Structured output parsing and tool/reasoning parsers.
- Admin endpoints, auth, and serialization.

## 15. Quick Command Reference

```bash
# Search
rg "pattern" python/sglang test docs
rg --files | rg "scheduler|radix|server_args"

# Install
pip install -e "python"
pip install -e "python[test]"
pip install -e "python[diffusion]"

# Serve
python -m sglang.launch_server --model-path <model> --host 0.0.0.0 --port 30000
sglang serve --model-path <model> --host 0.0.0.0 --port 30000

# Test
pytest test/registered/unit/ -v
python3 test/registered/<category>/<file>.py
python3 test/run_suite.py --hw cpu --suite base-a-test-cpu
python3 test/run_suite.py --hw cuda --suite base-b-test-1-gpu-small

# Benchmark
python -m sglang.bench_serving --backend sglang --dataset-name random --num-prompts 80 --max-concurrency 16
python -m sglang.bench_offline_throughput --model-path <model> --num-prompts 10
python -m sglang.bench_one_batch --model-path <model> --batch-size 32 --input-len 256 --output-len 32

# Profile
export SGLANG_TORCH_PROFILER_DIR=/tmp/sglang_profiles
python -m sglang.bench_serving --backend sglang --num-prompts 10 --profile
python -m sglang.profiler --num-steps 10 --cpu --gpu --output-dir /tmp/sglang_profiles

# Docs
cd docs
make compile
make html
make markdown
bash serve.sh

# Formatting
pre-commit run --all-files
```
