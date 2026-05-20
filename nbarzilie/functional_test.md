# PD/NIXL Functional Test Summary and Running Guide

## Tests Created

### `test/registered/disaggregation/test_disaggregation_nixl_basic.py`

Basic PD/NIXL smoke test.

- CI suite: `base-b-test-2-gpu-large`
- Registration: `register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")`
- Launches prefill, decode, and mini load balancer through `PDDisaggregationServerBase`.
- Forces `--disaggregation-transfer-backend nixl`.
- Uses `DEFAULT_SMALL_MODEL_NAME_FOR_TEST`.
- Sends one deterministic `/generate` request through the load balancer.
- Asserts:
  - HTTP 200.
  - Response contains non-empty `text`.
  - Load balancer, prefill, and decode processes are still alive.
  - `/health` succeeds for all three services.
- Skips when `nixl._api` cannot be imported.

### `test/registered/distributed/test_disaggregation_nixl_different_tp.py`

NIXL PD functional coverage for different tensor-parallel layouts.

- CI suite: `base-c-test-8-gpu-h20`
- Registration: `register_cuda_ci(est_time=300, stage="base-c", runner_config="8-gpu-h20")`
- Forces `--disaggregation-transfer-backend nixl`.
- Uses `DEFAULT_SMALL_MODEL_NAME_FOR_TEST`.
- Covers:
  - Prefill TP larger than decode TP: prefill `tp=2`, decode `tp=1`.
  - Decode TP larger than prefill TP: prefill `tp=1`, decode `tp=2`.
- Sends one `/generate` request per layout.
- Asserts non-empty output and process health.
- Skips when `nixl._api` cannot be imported.

### `test/registered/distributed/test_disaggregation_nixl_pp.py`

NIXL PD functional coverage for pipeline-parallel layouts.

- CI suite: `base-c-test-8-gpu-h20`
- Registration: `register_cuda_ci(est_time=300, stage="base-c", runner_config="8-gpu-h20")`
- Forces `--disaggregation-transfer-backend nixl`.
- Uses `DEFAULT_SMALL_MODEL_NAME_FOR_TEST`.
- Covers:
  - Prefill PP with decode PP size 1.
  - Matching prefill/decode PP size 2.
- Sends one `/generate` request per layout.
- Asserts non-empty output and process health.
- Skips when `nixl._api` cannot be imported.

### `test/registered/distributed/test_disaggregation_nixl_dp_attention.py`

NIXL PD functional coverage for DP attention.

- CI suite: `base-c-test-8-gpu-h20`
- Registration: `register_cuda_ci(est_time=300, stage="base-c", runner_config="8-gpu-h20")`
- Forces `--disaggregation-transfer-backend nixl`.
- Uses `DEFAULT_MODEL_NAME_FOR_TEST_MLA`.
- Disables JIT DeepGEMM for the test class.
- Launches prefill and decode with:
  - `--tp 2`
  - `--dp 2`
  - `--enable-dp-attention`
  - `--load-balance-method auto`
- Sends one `/generate` request.
- Asserts non-empty output and process health.
- Skips when `nixl._api` cannot be imported.

## Environment Requirements

Basic requirements:

```bash
pip install -e "python[test]"
pip install nixl
```

NIXL backend configuration:

```bash
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS='{}'
```

Optional RDMA device configuration, if the local NIXL/UCX setup requires explicit devices:

```bash
export SGLANG_TEST_PD_DISAGG_BACKEND=nixl
export SGLANG_TEST_PD_DISAGG_DEVICES=mlx5_0,mlx5_1
```

The new tests set `cls.rdma_devices = []` internally, so only use explicit RDMA device configuration if you adapt the files or local fixture behavior for an environment that requires it.

## Direct Run Commands

Run the basic 2-GPU NIXL smoke test:

```bash
python3 test/registered/disaggregation/test_disaggregation_nixl_basic.py
```

Run different-TP NIXL coverage:

```bash
python3 test/registered/distributed/test_disaggregation_nixl_different_tp.py
```

Run PP NIXL coverage:

```bash
python3 test/registered/distributed/test_disaggregation_nixl_pp.py
```

Run DP-attention NIXL coverage:

```bash
python3 test/registered/distributed/test_disaggregation_nixl_dp_attention.py
```

## Suite Run Commands

Run the planned basic functional suite:

```bash
python3 test/run_suite.py --hw cuda --suite base-b-test-2-gpu-large
```

Run the planned large distributed NIXL suite:

```bash
python3 test/run_suite.py --hw cuda --suite base-c-test-8-gpu-h20
```

For partitioned CI-style runs:

```bash
python3 test/run_suite.py --hw cuda --suite base-c-test-8-gpu-h20 \
  --auto-partition-id 0 --auto-partition-size 2
```

## Lightweight Validation

Syntax-check the new files without launching servers:

```bash
python3 -m py_compile \
  test/registered/disaggregation/test_disaggregation_nixl_basic.py \
  test/registered/distributed/test_disaggregation_nixl_different_tp.py \
  test/registered/distributed/test_disaggregation_nixl_pp.py \
  test/registered/distributed/test_disaggregation_nixl_dp_attention.py
```

Check CI registration parsing without importing the full `sglang` package:

```bash
python3 -c 'import importlib.util; spec=importlib.util.spec_from_file_location("ci_register", "python/sglang/test/ci/ci_register.py"); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); files=["test/registered/disaggregation/test_disaggregation_nixl_basic.py","test/registered/distributed/test_disaggregation_nixl_different_tp.py","test/registered/distributed/test_disaggregation_nixl_pp.py","test/registered/distributed/test_disaggregation_nixl_dp_attention.py"]; regs=mod.collect_tests(files, sanity_check=True); print("\n".join(f"{r.filename}: {r.effective_suite}" for r in regs))'
```

Expected registration output:

```text
test/registered/disaggregation/test_disaggregation_nixl_basic.py: base-b-test-2-gpu-large
test/registered/distributed/test_disaggregation_nixl_different_tp.py: base-c-test-8-gpu-h20
test/registered/distributed/test_disaggregation_nixl_pp.py: base-c-test-8-gpu-h20
test/registered/distributed/test_disaggregation_nixl_dp_attention.py: base-c-test-8-gpu-h20
```
