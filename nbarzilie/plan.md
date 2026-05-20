# NIXL Backend Stabilization Test Plan for SGLang

## Goal

Stabilize the SGLang NIXL disaggregation backend by building layered coverage:

1. CPU unit tests for NIXL-specific protocol, status, parsing, descriptor, and failure logic.
2. Small real PD/NIXL smoke tests that prove a basic prefill-to-decode transfer completes.
3. Functional PD tests on the planned 2-GPU H100 suite.
4. Large scale decode-radix-cache and disaggregation tests on the planned 8-GPU H20 suite.
5. Optional backend/hardware-specific tests for XPU, staging, hybrid state, and transport plugin behavior.

The default rule is to test logic with CPU unit tests and reserve real server tests for behavior that requires separate prefill/decode processes or actual transport.

## Current Repository Signals

- Main NIXL implementation: `python/sglang/srt/disaggregation/nixl/conn.py`.
- Existing wire tests: `test/registered/unit/disaggregation/test_disaggregation_wire.py`.
- Existing bootstrap registration tests: `test/registered/unit/disaggregation/test_register_to_bootstrap.py`.
- Existing NIXL XPU E2E: `test/registered/disaggregation/test_disaggregation_xpu.py`, currently disabled for standard CUDA CI and XPU-only.
- Existing decode radix cache E2E: `test/registered/distributed/test_disaggregation_decode_radix_cache.py`.
- `TestDisaggregationDecodeRadixCacheNixl` is currently skipped until NIXL is stable.
- Planned functional/basic suite: `base-b-test-2-gpu-large` on `2-gpu-h100`.
- Planned large-scale suite: `base-c-test-8-gpu-h20` on `8-gpu-h20`.

## Testing Guidelines

### Unit Tests

Unit tests should live under `test/registered/unit/disaggregation/`, use `CustomTestCase`, register with `register_cpu_ci(est_time=..., suite="base-a-test-cpu")`, avoid importing real NIXL when possible, and avoid `NixlKVManager.__init__` unless all side effects are mocked.

Recommended unit files:

- `test/registered/unit/disaggregation/test_nixl_transfer_info.py`
- `test/registered/unit/disaggregation/test_nixl_transfer_status.py`
- `test/registered/unit/disaggregation/test_nixl_notifications.py`
- `test/registered/unit/disaggregation/test_nixl_backend_config.py`
- `test/registered/unit/disaggregation/test_nixl_descriptor_building.py`
- `test/registered/unit/disaggregation/test_nixl_receiver_poll.py`
- `test/registered/unit/disaggregation/test_nixl_node_failure.py`
- `test/registered/unit/disaggregation/test_nixl_staging.py`
- `test/registered/unit/disaggregation/test_nixl_hybrid_state.py`

Core unit cases:

- `TransferInfo.from_zmq` parses room, endpoint, agent name, KV indices, aux index, required response count, state indices, and `decode_prefix_len`.
- `TransferInfo.is_dummy()` returns false for decode radix full-hit cases where `dst_kv_indices` is empty but `decode_prefix_len > 0`.
- `KVArgsRegisterInfo.from_zmq` preserves large unsigned pointers, optional state fields, optional staging fields, GPU ID, TP size, rank, and item lengths.
- `TransferStatus.is_done()` handles normal KV chunks, zero-KV aux-only completion, multi-PP expected counts, state-required completion, and failure completion.
- Notification parsing handles:
  - `{room}_kv_{chunk_id}_{is_last}_{pp_rank}`
  - `{room}_stg_{chunk_id}_{is_last}_{pp_rank}_{chunk_idx}_{page_start}_{num_pages}_{agent_name}`
  - `{room}_aux`
  - `{room}_aux_nokv_{pp_rank}`
  - `{room}_state_{pp_rank}`
- Staging notification parsing must preserve agent names with underscores by relying on bounded splitting.
- Aux `nokv` must set expected KV count to zero for that PP rank and allow completion when aux is received.
- State notifications must mark state arrival per PP rank.
- Backend params validation must reject JSON that is not an object or has non-string keys/values.
- Missing NIXL import should raise actionable install guidance.
- Missing requested plugin should raise an error that includes requested backend and available plugins.
- `register_buffer_to_engine` should register KV, aux, and state buffers with expected memory type and fail on empty registration descriptors.
- Descriptor construction for KV, aux, state, Mamba state, and sliced KV transfers should use correct memory type, address arithmetic, notification tags, and `WRITE` operation.
- Pointer math should preserve unsigned 64-bit addresses, including addresses with bit 63 set.
- `NixlKVReceiver.poll` should handle waiting timeout, failed manager status, completion, and late failure without resurrecting cleared room state.
- `_handle_node_failure` should remove failed prefill info, clean connection state, and mark affected rooms failed.
- `NixlKVSender.should_send_kv_chunk(0, last_chunk=True)` should return true for decode-radix full-hit aux-only completion.

### Basic PD/NIXL Functional Tests

These tests should launch real prefill, decode, and router processes using `PDDisaggregationServerBase`. They belong under `test/registered/disaggregation/` unless they become large distributed coverage.

Recommended file:

- `test/registered/disaggregation/test_disaggregation_nixl_basic.py`

Registration:

```python
register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")
```

Core cases:

- NIXL dependency probe with `@unittest.skipUnless(_has_nixl(), "NIXL is required for this test.")`.
- Force `cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]`.
- Use the smallest practical model, preferably `DEFAULT_SMALL_MODEL_NAME_FOR_TEST`.
- Set `cls.rdma_devices = []` unless the target NIXL/UCX setup requires explicit devices.
- Send one deterministic `/generate` request through the load balancer.
- Assert HTTP 200, response contains non-empty `text`, and all three processes are still healthy.
- Add a second request with `return_logprob` only if basic generation is stable and the extra assertion catches NIXL-specific behavior.

### Basic DP/PD and Rank-Mapping Tests

Use CPU unit tests for pure rank mapping and pointer slicing. Use real E2E only when the bug requires actual worker layout.

Recommended unit targets:

- `CommonKVManager._resolve_rank_mapping`
- `get_mha_kv_ptrs_with_pp`
- `get_mla_kv_ptrs_with_pp`
- NIXL `send_kvcache_slice`
- NIXL staging fallback path for heterogeneous TP non-MLA

Recommended E2E files, only after unit coverage:

- `test/registered/distributed/test_disaggregation_nixl_different_tp.py`
- `test/registered/distributed/test_disaggregation_nixl_pp.py`
- `test/registered/distributed/test_disaggregation_nixl_dp_attention.py`

Registration for early functional coverage:

```python
register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")
```

Registration for broader large distributed coverage:

```python
register_cuda_ci(est_time=300, stage="base-c", runner_config="8-gpu-h20")
```

Cases:

- Prefill TP equals decode TP.
- Prefill TP larger than decode TP.
- Decode TP larger than prefill TP.
- CP rank filtering.
- PP same layout.
- Decode PP size 1 with prefill PP larger.
- Non-MLA heterogeneous TP staging path.
- MLA path does not use staging unnecessarily.

### Decode Radix Cache Tests

Decode radix cache is a high-risk NIXL path because full cache hits can have zero KV pages while aux/state still must complete.

Required unit coverage before enabling NIXL E2E:

- `TransferInfo.is_dummy()` full-hit behavior.
- `NixlKVSender.should_send_kv_chunk(0, last_chunk=True)`.
- `_handle_aux_notification` with `aux_nokv`.
- `TransferStatus.is_done()` with `num_pp_ranks_expected=1`, aux received, and expected KV count zero.
- Receiver `poll()` returns `KVPoll.Success` for aux-only full hit.

Large E2E target:

- Existing `test/registered/distributed/test_disaggregation_decode_radix_cache.py`.

Planned NIXL class:

- Unskip `TestDisaggregationDecodeRadixCacheNixl` only after the basic NIXL smoke test is stable in target CI.

Registration:

```python
register_cuda_ci(est_time=300, stage="base-c", runner_config="8-gpu-h20")
```

Cases already covered by the mixin and should remain:

- Decode server reports radix cache enabled.
- Multi-turn workload produces cached tokens.
- Prefill, decode, and load balancer stay healthy.
- Two GSM8K passes stay above threshold and second pass does not regress by more than the configured bound.

### Staging Buffer Tests

Staging is important for heterogeneous TP non-MLA paths. Start with unit tests, then add E2E only when runner support and runtime are known.

Unit cases:

- Staging registration uses `VRAM` and fails clearly when registration returns empty descriptors.
- `_prefetch_staging_reqs` is a no-op when staging is disabled or KV tensors are absent.
- `_do_staging_transfer` requeues when staging allocation/watermark is not ready.
- `_do_staging_transfer` falls back to sliced KV transfer when staging buffer is too small.
- `send_kvcache_staged` validates prefill and decode staging sizes and uses one bulk NIXL `WRITE`.
- Staging notification with agent name containing underscores is parsed correctly.
- Last scatter is submitted only after aux and all staging chunks arrive.

Potential E2E file:

- `test/registered/distributed/test_disaggregation_nixl_staging.py`

Suite:

- Start disabled or gated on `base-c-test-8-gpu-h20` until stability is proven.

### Hybrid State Tests

Hybrid models add Mamba/SWA/NSA state transfers. Cover dispatch and descriptors first.

Unit cases:

- `maybe_send_extra` sends nothing for empty state indices.
- Mamba homogeneous TP calls `_send_mamba_state`.
- Mamba heterogeneous TP calls `_send_mamba_state_slice`.
- Mamba state descriptor offsets are correct for both source and destination indices.
- Missing state dimension metadata falls back or fails according to current code behavior.
- SWA/NSA homogeneous paths use `_send_kvcache_generic` for state-like payloads.
- Unsupported hybrid state under NIXL raises actionable errors.
- State notification marks per-PP completion and gates `TransferStatus.is_done()`.

Potential larger E2E:

- Add NIXL variants of existing hybrid attention PD tests only after unit descriptor/state coverage passes.

### Failure and Recovery Tests

Most failures should be unit-tested without real transport.

Cases:

- ImportError for missing NIXL package includes install guidance.
- Invalid backend params fail before backend creation.
- Missing plugin reports available plugin list.
- `initialize_xfer` returning false raises a clear exception.
- `agent.transfer()` returning `ERR` raises and records failed transfer state.
- `agent.check_xfer_state()` returning `ERR` records failure and prevents room success.
- `failure_exception()` returns the original exception when available.
- Waiting timeout marks room failed.
- Prefill node heartbeat failure marks related rooms failed.
- Late failed notification after room clear does not affect future room reuse.

Potential E2E:

- Keep failure injection on Mooncake for now unless a deterministic NIXL failure mode can be created without flakiness.

### Backend and Plugin Coverage

The default planned NIXL backend should be explicit through environment:

```bash
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS='{}'
```

Unit coverage should not assume UCX exists. It can mock plugin lists.

Real NIXL tests must skip when `nixl._api` cannot be imported and should fail clearly when the configured plugin is absent.

Optional backend-specific tests:

- UCX for VRAM-to-VRAM PD transfer.
- LIBFABRIC only if CI image and hardware guarantee it.
- GDS/GDS_MT only for storage/offload paths, not basic PD KV transfer.
- XPU coverage remains disabled from standard CUDA CI unless an XPU runner exists.

## Flow Plan

### Phase 1: Local Single-Script Unit Validation

Write the first unit test file and run it directly:

```bash
python3 test/registered/unit/disaggregation/test_nixl_transfer_info.py
python3 test/registered/unit/disaggregation/test_nixl_transfer_status.py
python3 test/registered/unit/disaggregation/test_nixl_notifications.py
```

Use this phase for fast iteration. No server, model, GPU, or NIXL install should be required for pure logic tests.

### Phase 2: Local Unit Suite Validation

Run the relevant unit directory:

```bash
pytest test/registered/unit/disaggregation/ -v
```

Then run CPU suite discovery:

```bash
python3 test/run_suite.py --hw cpu --suite base-a-test-cpu
```

Expected result: new unit files are discovered, registered, executable directly, and pass.

### Phase 3: Local NIXL Smoke Script

Before committing a CI E2E file, run the basic PD/NIXL flow locally as a direct test:

```bash
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS='{}'
python3 test/registered/disaggregation/test_disaggregation_nixl_basic.py
```

If RDMA devices are required in the local setup:

```bash
export SGLANG_TEST_PD_DISAGG_BACKEND=nixl
export SGLANG_TEST_PD_DISAGG_DEVICES=mlx5_0,mlx5_1
python3 test/registered/disaggregation/test_disaggregation_nixl_basic.py
```

Expected result: prefill, decode, and router launch; one request completes; all processes remain healthy.

### Phase 4: Local CI Suite Validation for Basic Functional Tests

Run the planned functional/basic suite locally on a 2-GPU H100 machine:

```bash
python3 test/run_suite.py --hw cuda --suite base-b-test-2-gpu-large
```

For faster iteration, run just the target file first:

```bash
python3 test/registered/disaggregation/test_disaggregation_nixl_basic.py
```

Expected result: the file passes on the target machine and suite discovery accepts registration.

### Phase 5: GitHub Actions Basic Functional CI

Add the basic NIXL E2E test to:

```text
test/registered/disaggregation/test_disaggregation_nixl_basic.py
```

Register:

```python
register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")
```

Open PR and use normal SGLang CI flow. If needed, use `/rerun-test test/registered/disaggregation/test_disaggregation_nixl_basic.py` after fixing flakes.

Required before enabling as a gating test:

- CI image has `nixl` installed.
- Required NIXL plugin exists.
- Runner has enough GPU memory.
- Failure mode for missing NIXL is skip, not hard failure.
- Runtime is measured and `est_time` is updated.

### Phase 6: Local Large-Scale Radix Cache Validation

Run the existing decode radix cache file after unskipping or temporarily targeting the NIXL class:

```bash
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS='{}'
python3 test/registered/distributed/test_disaggregation_decode_radix_cache.py
```

Target hardware: 8-GPU H20 for the planned suite.

Expected result:

- Decode radix cache is enabled.
- Cache-hit workload reports cached tokens.
- Two-pass GSM8K remains within accuracy thresholds.
- All workers remain healthy after async cleanup.

### Phase 7: GitHub Actions Large-Scale CI

Keep large NIXL decode radix cache coverage in:

```text
test/registered/distributed/test_disaggregation_decode_radix_cache.py
```

Register:

```python
register_cuda_ci(est_time=300, stage="base-c", runner_config="8-gpu-h20")
```

Enable `TestDisaggregationDecodeRadixCacheNixl` only after:

- Unit full-hit/aux-only tests are merged.
- Basic NIXL smoke is stable.
- CI NIXL dependency and plugin availability are confirmed.
- At least one local 8-GPU H20 run passes.

## Local Setup Guide

### CPU Unit Tests

Requirements:

- Python environment with editable SGLang install.
- Test dependencies installed.
- No GPU required.
- No NIXL package required if tests mock NIXL-facing interactions.

Setup:

```bash
pip install -e "python[test]"
```

Run:

```bash
python3 test/registered/unit/disaggregation/test_nixl_transfer_info.py
pytest test/registered/unit/disaggregation/ -v
python3 test/run_suite.py --hw cpu --suite base-a-test-cpu
```

### Basic NIXL PD Functional Tests

Requirements:

- 2 NVIDIA GPUs, planned target `2-gpu-h100`.
- `nixl` Python package installed.
- Required NIXL plugin installed, normally UCX for remote VRAM transfer.
- Router available through the normal SGLang test environment.
- Hugging Face model cache or network access for the small model.

Setup:

```bash
pip install -e "python[test]"
pip install nixl
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS='{}'
```

Optional RDMA device setup:

```bash
export SGLANG_TEST_PD_DISAGG_BACKEND=nixl
export SGLANG_TEST_PD_DISAGG_DEVICES=mlx5_0,mlx5_1
```

Run:

```bash
python3 test/registered/disaggregation/test_disaggregation_nixl_basic.py
python3 test/run_suite.py --hw cuda --suite base-b-test-2-gpu-large
```

### Large NIXL Decode Radix Cache Tests

Requirements:

- 8 NVIDIA GPUs, planned target `8-gpu-h20`.
- `nixl` package and required plugin installed.
- Model cache for `DEFAULT_MODEL_NAME_FOR_TEST`.
- Enough runtime for cache-hit workload and two-pass GSM8K.

Setup:

```bash
pip install -e "python[test]"
pip install nixl
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS='{}'
```

Run:

```bash
python3 test/registered/distributed/test_disaggregation_decode_radix_cache.py
python3 test/run_suite.py --hw cuda --suite base-c-test-8-gpu-h20
```

### XPU-Specific NIXL Tests

Requirements:

- Intel XPU machine.
- `torch.xpu.is_available()` returns true.
- NIXL installed with XPU-compatible transport.

Current file:

```text
test/registered/disaggregation/test_disaggregation_xpu.py
```

Notes:

- This test is disabled from standard CUDA CI.
- It currently installs `sglang-router` inside the test. Before making it a normal CI test, remove runtime package installation and move dependency setup to the CI image.
- Add request timeouts to all `requests.post` calls before promoting it.

## CI Registration Rules for This Plan

Use literal registration values.

CPU unit files:

```python
register_cpu_ci(est_time=3, suite="base-a-test-cpu")
```

Basic NIXL functional files:

```python
register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")
```

Large radix/distributed NIXL files:

```python
register_cuda_ci(est_time=300, stage="base-c", runner_config="8-gpu-h20")
```

If a real NIXL test cannot run in the target CI image yet, use disabled registration with a concrete reason:

```python
register_cuda_ci(
    est_time=300,
    stage="base-b",
    runner_config="2-gpu-large",
    disabled="requires NIXL-enabled CI runner with UCX plugin",
)
```

## Promotion Criteria

Promote from unit to E2E only when:

- Unit tests cover the pure logic branch.
- The behavior crosses process boundaries or requires real model/transport behavior.
- The test has explicit dependency skips.
- The test checks process health.
- Requests have timeouts.
- Runtime is measured.

Promote NIXL E2E into gating CI only when:

- The target CI image has NIXL installed.
- The target runner exposes the required NIXL backend plugin.
- The file passes direct local execution on equivalent hardware.
- Suite discovery passes.
- Repeated local or CI runs show no obvious flake.

## Known Risks and Follow-Ups

- Do not unskip the large NIXL decode radix cache class before basic NIXL smoke is stable.
- Do not assume UCX is installed on every CUDA runner.
- Do not make tests install packages at runtime.
- Do not use real NIXL for pure parser/status/descriptor logic.
- Do not start `NixlKVManager` background threads in unit tests unless the constructor itself is under test and threads are fully controlled.
- Add timeouts to all request calls in new E2E tests.
- Keep model choice small for basic smoke tests.
- Keep large model/accuracy coverage only in the 8-GPU large suite.
- Treat staging and hybrid state as separate layers after core NIXL transfer is stable.
- Keep failure messages specific: include backend, room, URL, response text, or plugin list where relevant.
