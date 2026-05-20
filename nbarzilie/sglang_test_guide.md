# SGLang Test Guide for Coding Agents

This guide equips coding agents to write, register, run, and review SGLang tests, with extra depth for PD disaggregation and the NIXL transfer backend.

It is intentionally practical. Use it as a local playbook before changing test code.

## 000. Required Local Context

Read these files before authoring SGLang tests:

- @.claude/skills/write-sglang-test/SKILL.md
- @.claude/skills/ci-workflow-guide/SKILL.md
- @test/README.md
- @test/run_suite.py
- @python/sglang/test/ci/ci_register.py
- @python/sglang/test/test_utils.py
- @python/sglang/test/server_fixtures/disaggregation_fixture.py

Read these files before authoring PD disaggregation or NIXL tests:

- @docs/advanced_features/pd_disaggregation.md
- @docs/advanced_features/server_arguments.md
- @test/registered/distributed/test_disaggregation_decode_radix_cache.py
- @test/registered/disaggregation/test_disaggregation_basic.py
- @test/registered/disaggregation/test_disaggregation_xpu.py
- @test/registered/unit/disaggregation/test_disaggregation_wire.py
- @test/registered/unit/disaggregation/test_register_to_bootstrap.py
- @python/sglang/srt/disaggregation/base/conn.py
- @python/sglang/srt/disaggregation/common/conn.py
- @python/sglang/srt/disaggregation/nixl/conn.py

Read these files when choosing real model coverage:

- @README.md
- @docs/get_started/install.md
- @docs/basic_usage/qwen3_5.md
- @docs/basic_usage/deepseek_v3.md
- @docs_new/README.md

Relevant skills:

- @.claude/skills/write-sglang-test/SKILL.md for test authoring.
- @.claude/skills/ci-workflow-guide/SKILL.md for CI stage, suite, partition, rerun, and failure behavior.
- Use normal prose for test files and docs, even if a terse communication skill is active.

## 001. Agent Mission

Your job when writing SGLang tests is not to maximize the number of tests.

Your job is to preserve runtime behavior under realistic serving conditions while keeping CI cost low.

The best SGLang test usually has these traits:

- It targets one behavior.
- It uses the smallest fixture that can observe that behavior.
- It registers to the lightest CI suite that can run it.
- It uses `CustomTestCase`.
- It cleans up subprocesses even when setup fails.
- It uses real server launches only when mocks cannot observe the bug.
- It avoids backend duplication unless the backend itself is the subject.
- It gives future agents enough failure signal to debug quickly.

## 002. SGLang Testing Mental Model

SGLang is a serving system.

Many bugs are not pure function bugs.

Common bug categories:

- CLI/server argument validation.
- Scheduler state transition.
- Memory pool accounting.
- Radix cache correctness.
- PD disaggregation bootstrap routing.
- KV transfer metadata packing.
- Tensor pointer arithmetic.
- GPU, RDMA, or XPU memory registration.
- Request lifecycle, abort, retry, and failure cleanup.
- Accuracy regression under a feature flag.
- Performance regression under a representative workload.

Choose test type by observability:

- If a unit can observe the behavior, write a unit test.
- If one launched server can observe the behavior, launch one server.
- If prefill/decode separation is required, use PD fixture.
- If actual NIXL transport is required, gate by dependency and hardware.
- If large model accuracy is required, use eval mixins or `run_eval` only in larger suites.

## 003. Test Placement Rules

General CI-discovered tests live under:

```text
test/registered/
```

Unit tests live under:

```text
test/registered/unit/
```

Server or integration tests live under a category:

```text
test/registered/core/
test/registered/disaggregation/
test/registered/distributed/
test/registered/openai_server/
test/registered/perf/
```

JIT kernel tests are exceptions:

```text
python/sglang/jit_kernel/tests/
python/sglang/jit_kernel/benchmark/
```

Manual debug tests do not belong in CI:

```text
test/manual/
```

Decision rule:

- Pure component logic: `test/registered/unit/...`
- Disaggregation wire encoding: `test/registered/unit/disaggregation/...`
- PD launch and inference: `test/registered/disaggregation/...`
- Multi-GPU distributed PD behavior: `test/registered/distributed/...`
- Backend-specific GPU kernels: `test/registered/kernels/...` or JIT kernel path.
- One-off repro scripts: `test/manual/...`

## 004. Always Use CustomTestCase

Use:

```python
from sglang.test.test_utils import CustomTestCase
```

Do not use raw `unittest.TestCase` for new CI tests unless the file already has a documented exception.

Why:

- `CustomTestCase` is the local base with SGLang test behavior.
- It supports CI retry conventions.
- It prevents common lifecycle leaks.
- The skill @.claude/skills/write-sglang-test/SKILL.md requires it.

Preferred skeleton:

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

## 005. Defensive Cleanup

Any launched process must be cleaned defensively.

Good:

```python
@classmethod
def tearDownClass(cls):
    if hasattr(cls, "process") and cls.process:
        kill_process_tree(cls.process.pid)
```

Bad:

```python
@classmethod
def tearDownClass(cls):
    kill_process_tree(cls.process.pid)
```

Setup can fail halfway.

CI can cancel at awkward points.

Resource leaks are expensive in GPU CI.

PD fixtures usually own prefill, decode, and load balancer processes. If you subclass `PDDisaggregationServerBase`, call `super().tearDownClass()` when overriding cleanup.

## 006. Main Entry Rule

Every registered file must be executable directly.

For unittest:

```python
if __name__ == "__main__":
    unittest.main()
```

For pytest:

```python
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
```

Do not add custom `argparse` to a test file.

Do not mutate `sys.argv` before `unittest.main()`.

The CI runner appends failfast flags.

## 007. CI Registration Rules

Every CI-discovered test file needs a top-level registration call.

Current registration API supports two shapes:

```python
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
```

and:

```python
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
```

Use `stage=` plus `runner_config=` for CUDA suites that follow:

```text
{stage}-test-{runner_config}
```

Examples:

```python
register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_cuda_ci(est_time=300, stage="base-c", runner_config="8-gpu-h20")
register_cuda_ci(est_time=375, stage="base-c", runner_config="8-gpu-h200")
```

Use `suite=` for CPU, AMD, NPU, nightly, stress, weekly, and other suites that do not fit the pair form:

```python
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")
register_cuda_ci(est_time=200, suite="nightly-1-gpu", nightly=True)
```

The AST parser in @python/sglang/test/ci/ci_register.py requires literal values.

Do not write:

```python
SUITE = "base-a-test-cpu"
register_cpu_ci(est_time=5, suite=SUITE)
```

Do write:

```python
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
```

## 008. Registration Compatibility Note

Some older docs and tests use:

```python
register_cuda_ci(est_time=80, suite="base-b-test-1-gpu-small")
```

Some current tests use:

```python
register_cuda_ci(est_time=80, stage="base-b", runner_config="1-gpu-small")
```

Both can parse, but new CUDA per-commit tests should prefer the stage/runner form when the suite follows the `{stage}-test-{runner_config}` pattern.

When uncertain, inspect nearby files and @test/run_suite.py.

## 009. CI Suite Selection

Use the lightest suite that can run the test.

Common choices:

```text
base-a-test-cpu
base-a-test-1-gpu-small
base-b-test-1-gpu-small
base-b-test-1-gpu-large
base-b-test-2-gpu-large
base-c-test-4-gpu-h100
base-c-test-8-gpu-h20
base-c-test-8-gpu-h200
base-c-test-deepep-4-gpu-h100
```

CPU-only unit tests:

```python
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
```

Small CUDA unit or server tests:

```python
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
```

H100 memory or Hopper-specific test:

```python
register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")
```

Two-GPU behavior:

```python
register_cuda_ci(est_time=180, stage="base-b", runner_config="2-gpu-large")
```

Large distributed PD behavior:

```python
register_cuda_ci(est_time=300, stage="base-c", runner_config="8-gpu-h20")
```

Extra label-gated CI:

```python
register_cuda_ci(est_time=310, stage="extra-b", runner_config="8-gpu-h200")
```

Nightly:

```python
register_cuda_ci(est_time=600, suite="nightly-8-gpu-h200", nightly=True)
```

## 010. CI Backend Selection

Do not register a backend unless the test gives backend-specific signal.

Bad:

```python
register_cuda_ci(...)
register_amd_ci(...)
register_npu_ci(...)
```

if the test only checks OpenAI response shape.

Good:

```python
register_amd_ci(..., suite="stage-b-test-1-gpu-small-amd")
```

when testing ROCm-specific code.

Good:

```python
register_cuda_ci(..., stage="base-c", runner_config="8-gpu-h20")
```

when testing CUDA distributed PD behavior.

For NIXL:

- NIXL is a transfer backend.
- It may depend on installed NIXL plugins.
- It may require UCX/libfabric/GDS/UCCL availability.
- It may require hardware or network topology.
- Do not blindly run NIXL E2E on standard per-commit CUDA unless CI guarantees NIXL availability.

## 011. Local Run Commands

Single file:

```bash
python3 test/registered/unit/disaggregation/test_disaggregation_wire.py
```

Single test method:

```bash
python3 test/registered/core/test_srt_endpoint.py TestSRTEndpoint.test_simple_decode
```

One suite:

```bash
python3 test/run_suite.py --hw cpu --suite base-a-test-cpu
```

CUDA suite:

```bash
python3 test/run_suite.py --hw cuda --suite base-b-test-1-gpu-small
```

Nightly suite:

```bash
python3 test/run_suite.py --hw cuda --suite nightly-1-gpu --nightly
```

Auto partition:

```bash
python3 test/run_suite.py --hw cuda --suite base-b-test-1-gpu-small \
  --auto-partition-id 0 --auto-partition-size 8
```

Run a disaggregation file:

```bash
python3 test/registered/disaggregation/test_disaggregation_basic.py
```

Run the decode radix cache file:

```bash
python3 test/registered/distributed/test_disaggregation_decode_radix_cache.py
```

## 012. How test/run_suite.py Works

@test/run_suite.py:

- Maps `--hw` to hardware backend enum.
- Defines per-commit suites.
- Defines nightly suites.
- Defines other suites.
- Scans `test/registered/**/*.py`.
- Ignores `conftest.py`, `__init__.py`, and a few helper files.
- Adds JIT kernel correctness tests.
- Adds JIT kernel benchmarks.
- Uses `collect_tests(..., sanity_check=True)`.
- Validates suite names for CUDA and CPU.
- Filters by backend, suite, and nightly.
- Applies LPT auto-partition if requested.
- Runs files through `run_unittest_files`.

Consequences:

- A missing registration fails discovery.
- A non-literal registration fails discovery.
- An invalid CUDA suite fails validation.
- A missing `if __name__ == "__main__"` fails sanity checks.
- A disabled registration is reported but skipped.

## 013. CI Fast-Fail Model

From @.claude/skills/ci-workflow-guide/SKILL.md:

- Method failure stops the file through failfast.
- File failure stops the suite in PR mode unless `--continue-on-error`.
- Job failures fast-fail peer jobs through health checks.
- Stage failures skip later stages in PR mode.
- Scheduled runs use different continue behavior.

Agent implication:

- Put cheap tests in early suites.
- Put broad expensive tests in later suites.
- Make failure messages specific.
- Avoid tests that fail late after doing unnecessary setup.

## 014. Est Time

`est_time` is used for partitioning.

Choose honest values.

If a file starts three servers and runs eval, do not claim `est_time=10`.

Rough guidance:

- Pure unit test: 1 to 15 seconds.
- Small one-server smoke test: 30 to 120 seconds.
- PD launch with one model and a few requests: 180 to 500 seconds.
- Eval with hundreds of examples: 300+ seconds.
- Multi-GPU large model: usually base-c or nightly.

Update `est_time` when adding slow test methods.

## 015. Disabled Tests

Temporary disable:

```python
register_cuda_ci(
    est_time=300,
    stage="base-c",
    runner_config="8-gpu-h20",
    disabled="flaky NIXL backend bootstrap timeout; see #12345",
)
```

Class-level skip:

```python
@unittest.skip("Temporarily disabled until nixl backend is stable.")
class TestDisaggregationDecodeRadixCacheNixl(...):
    ...
```

Use skip when:

- Dependency is not installed.
- Hardware is unavailable.
- Backend is intentionally not stable.
- Test remains useful for local/manual enablement.

Use disabled registration when:

- The file should be discovered but not run in CI.

Use both only with a clear reason.

## 016. Unit Test Principles

Unit tests should not launch:

- SGLang server.
- Router.
- Engine.
- Real model.
- Real NIXL transport.

Unit tests should check:

- Pure serialization.
- Metadata parsing.
- Status state transitions.
- Retry and backoff logic.
- Failure propagation.
- Argument validation.
- Pointer arithmetic helper logic.
- Notification tag parsing.
- Topology mapping logic.

Example sources:

- @test/registered/unit/disaggregation/test_disaggregation_wire.py
- @test/registered/unit/disaggregation/test_register_to_bootstrap.py
- @test/registered/unit/server_args/test_server_args.py

## 017. Unit Test Template for Disaggregation Helpers

```python
"""Unit tests for srt/disaggregation/<module>."""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDisaggregationHelper(CustomTestCase):
    def test_expected_branch(self):
        self.assertEqual("actual", "actual")


if __name__ == "__main__":
    unittest.main()
```

## 018. Mocking Expensive Managers

Many manager methods read attributes but do not require full initialization.

Pattern from @test/registered/unit/disaggregation/test_register_to_bootstrap.py:

```python
from unittest.mock import MagicMock

from sglang.srt.disaggregation.common.conn import CommonKVManager

mgr = MagicMock(spec=CommonKVManager)
mgr.register_to_bootstrap = CommonKVManager.register_to_bootstrap.__get__(
    mgr, CommonKVManager
)
mgr.bootstrap_host = "127.0.0.1"
mgr.bootstrap_port = 8765
```

Use this pattern when:

- The method is real.
- The constructor is expensive.
- You can set all read attributes explicitly.

Do not use it when:

- Constructor side effects are the behavior under test.
- Thread startup is required.
- Real sockets or memory registration are required.

## 019. CPU Stubs

Some imports pull GPU-only packages.

Use:

```python
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()
```

before importing modules that transitively import `sgl_kernel`.

Do not mutate `sys.modules` globally without cleanup.

Prefer `patch.dict` or existing helper functions.

## 020. E2E Test Principles

Use an E2E test when:

- HTTP behavior is the subject.
- Scheduler/server lifecycle is the subject.
- Model output is required.
- KV transfer requires real workers.
- Router behavior is required.

Avoid E2E when:

- You only need argument validation.
- You only need serialization.
- You only need branch coverage for one function.
- You can assert behavior by mocking dependencies.

E2E tests are expensive because model launch dominates runtime.

Group multiple related assertions under one server setup when reasonable.

## 021. Single Server Template

```python
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    CustomTestCase,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")


class TestMyServerFeature(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--disable-radix-cache"],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_generate(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 8},
            },
            timeout=30,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("text", response.json())


if __name__ == "__main__":
    unittest.main()
```

## 022. Server Fixture Reuse

Use fixture bases when they fit.

Common fixture:

```python
from sglang.test.server_fixtures.default_fixture import DefaultServerBase
```

PD fixture:

```python
from sglang.test.server_fixtures.disaggregation_fixture import PDDisaggregationServerBase
```

Fixtures reduce boilerplate and enforce cleanup.

Do not fork a fixture casually.

Subclass and override class attributes or class setup.

## 023. PD Disaggregation Background

PD disaggregation separates:

- Prefill phase: compute-intensive processing of input prompt.
- Decode phase: memory-intensive token generation using KV cache.

Unified serving can suffer:

- Prefill interruptions delaying decode.
- DP attention imbalance when some workers prefill while others decode.

PD mode runs separate servers:

- A prefill-only server.
- A decode-only server.
- A router/load balancer.

KV cache is transferred from prefill to decode through a transfer backend.

Supported transfer backends include:

- `mooncake`
- `nixl`
- `ascend`
- `fake`

Relevant docs:

- @docs/advanced_features/pd_disaggregation.md
- @docs/advanced_features/server_arguments.md

## 024. PD Server Arguments

Core flags:

```text
--disaggregation-mode prefill
--disaggregation-mode decode
--disaggregation-transfer-backend nixl
--disaggregation-bootstrap-port 8998
--disaggregation-ib-device mlx5_0,mlx5_1
```

Decode radix cache:

```text
--disaggregation-decode-enable-radix-cache
```

Decode offload:

```text
--disaggregation-decode-enable-offload-kvcache
```

Different TP sizes:

```text
--tp 4
--dp 4
--enable-dp-attention
```

Staging buffer environment:

```bash
export SGLANG_DISAGG_STAGING_BUFFER=1
export SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=64
export SGLANG_DISAGG_STAGING_POOL_SIZE_MB=4096
```

NIXL backend selection:

```bash
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS='{}'
```

## 025. PDDisaggregationServerBase

@python/sglang/test/server_fixtures/disaggregation_fixture.py provides `PDDisaggregationServerBase`.

It sets:

- `base_host`
- `lb_port`
- `prefill_port`
- `decode_port`
- `bootstrap_port`
- `prefill_url`
- `decode_url`
- `lb_url`
- `base_url`
- `process_lb`
- `process_decode`
- `process_prefill`

It starts:

- Prefill server through `start_prefill`.
- Decode server through `start_decode`.
- Router through `launch_lb`.

It uses:

- `popen_launch_pd_server`.
- `wait_for_http_ready`.
- `kill_process_tree`.

It sets transfer backend:

- In CI: Mooncake plus RDMA devices.
- Locally: `SGLANG_TEST_PD_DISAGG_BACKEND` and `SGLANG_TEST_PD_DISAGG_DEVICES`.

If you force NIXL in a subclass, override `cls.transfer_backend` after `super().setUpClass()`.

## 026. PD Fixture Template

```python
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")


class TestDisaggregationFeature(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.extra_prefill_args = ["--some-prefill-flag"]
        cls.extra_decode_args = ["--some-decode-flag"]
        cls.launch_all()

    def test_health(self):
        for url in [self.lb_url, self.prefill_url, self.decode_url]:
            response = requests.get(url + "/health", timeout=10)
            self.assertEqual(response.status_code, 200, response.text)


if __name__ == "__main__":
    unittest.main()
```

## 027. NIXL Background

NIXL is a transfer backend for disaggregation.

SGLang uses NIXL to transfer:

- KV cache in VRAM.
- Auxiliary metadata in DRAM.
- Hybrid model state in VRAM.
- Staging buffers for heterogeneous TP non-MLA cases.

NIXL setup occurs in @python/sglang/srt/disaggregation/nixl/conn.py.

Important classes:

- `NixlKVManager`
- `NixlKVSender`
- `NixlKVReceiver`
- `NixlKVBootstrapServer`
- `TransferInfo`
- `TransferKVChunk`
- `KVArgsRegisterInfo`
- `TransferStatus`

The backend imports:

```python
from nixl._api import nixl_agent, nixl_agent_config
```

When not installed, `NixlKVManager` raises an ImportError with install guidance.

## 028. NIXL Configuration

Environment variables:

```text
SGLANG_DISAGGREGATION_NIXL_BACKEND
SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS
SGLANG_DISAGG_STAGING_BUFFER
SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB
SGLANG_DISAGG_STAGING_POOL_SIZE_MB
SGLANG_DISAGGREGATION_QUEUE_SIZE
SGLANG_DISAGGREGATION_WAITING_TIMEOUT
SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE
```

Default backend is sourced from env config.

Known NIXL plugin choices from docs:

- `UCX`
- `LIBFABRIC`
- Any installed NIXL plugin.

Code also contains thread parameter adjustments for:

- `UCX`
- `OBJ`
- `GDS_MT`
- `UCCL`

Test implication:

- Validate bad `SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS` as a unit test.
- Gate real backend tests by plugin availability.
- Never assume UCX is installed on every CI runner unless CI image documents it.

## 029. NIXL Data Flow

Decode side registers base pointers with prefill:

- `NixlKVReceiver._register_kv_args`.
- Sends `GUARD`, `room=None`, endpoint, port, agent name, agent metadata, KV ptrs, aux ptrs, state ptrs, GPU id, TP size, rank, item lengths, staging info.
- Prefill bootstrap thread parses `KVArgsRegisterInfo.from_zmq`.
- Prefill adds remote peer through `_add_remote_peer`.

Decode side sends per-request metadata:

- `NixlKVReceiver.send_metadata`.
- Sends room, endpoint, rank port, agent name, destination KV indices, aux index, required dst info count, state indices, decode prefix length.
- Prefill parses `TransferInfo.from_zmq`.
- Once all required dst info arrives, request status becomes `WaitingForInput`.

Prefill side sends data:

- `NixlKVSender.send`.
- Queues `TransferKVChunk`.
- Worker picks chunk in `NixlKVManager.transfer_worker`.
- Worker posts NIXL RDMA transfers.
- Worker sends KV, state, and aux notifications.

Decode side polls:

- `NixlKVReceiver.poll`.
- Reads `NixlKVManager.update_transfer_status`.
- Uses `TransferStatus.is_done`.
- Returns `KVPoll.Success` or `KVPoll.Failed`.

## 030. NIXL Status Model

`KVPoll` states in @python/sglang/srt/disaggregation/base/conn.py:

```python
class KVPoll:
    Failed = 0
    Bootstrapping = 1
    WaitingForInput = 2
    Transferring = 3
    Success = 4
```

Status update rules:

- Failed can override later states.
- Non-failed updates are monotonic.
- Late Failed after room clear should not resurrect the entry.

Unit tests should cover:

- Failed wins.
- Success remains success after duplicate transfer notifications.
- Late abort does not pollute future reused room.
- Waiting timeout records failure.

## 031. NIXL Notification Tags

Notification formats in @python/sglang/srt/disaggregation/nixl/conn.py:

```text
{room}_kv_{chunk_id}_{is_last}_{pp_rank}
{room}_stg_{chunk_id}_{is_last}_{pp_rank}_{chunk_idx}_{page_start}_{num_pages}_{agent_name}
{room}_aux
{room}_aux_nokv_{pp_rank}
{room}_state_{pp_rank}
```

Agent names may contain underscores.

The parser uses `split("_", 8)` for staging tags.

Regression test target:

- Staging tag with agent name containing underscores.
- Aux `nokv` tag marks zero expected KV chunks for a PP rank.
- State tag marks state received for that PP rank.
- Last KV tag sets expected chunk count.

## 032. NIXL Wire Packing

Existing wire utility tests:

- @test/registered/unit/disaggregation/test_disaggregation_wire.py

Good coverage:

- `pack_int_lists`
- `unpack_int_lists`
- Empty outer list.
- Empty inner list.
- NumPy arrays.
- Buffer list roundtrip.

Additional NIXL-specific wire coverage can target:

- `TransferInfo.from_zmq`.
- `KVArgsRegisterInfo.from_zmq`.
- Optional staging tail fields.
- Empty state fields.
- `decode_prefix_len` default behavior.
- Large unsigned pointer values.

## 033. NIXL Pointer Arithmetic Risks

NIXL uses device pointers and item lengths.

On Intel XPU, addresses can have bit 63 set.

@test/registered/disaggregation/test_disaggregation_xpu.py documents why `np.uint64` matters.

Risky areas:

- `_send_kvcache_generic`
- `send_kvcache_slice`
- `send_kvcache_staged`
- `_send_mamba_state_slice`

Unit tests should assert arrays are built with unsigned-compatible behavior when possible.

Integration tests should verify the real backend on XPU only when XPU is available.

## 034. NIXL Transfer Modes

NIXL has several KV paths:

- `send_kvcache`: homogeneous TP or MLA.
- `send_kvcache_slice`: heterogeneous TP fallback.
- `send_kvcache_staged`: staging buffer gather and bulk RDMA.
- `maybe_send_extra`: hybrid state dispatch.
- `_send_mamba_state`: homogeneous Mamba state.
- `_send_mamba_state_slice`: heterogeneous Mamba state.
- `_send_kvcache_generic`: shared MHA/MLA/SWA/NSA transfer descriptor logic.

Test each mode at the cheapest level that can observe it.

Examples:

- Descriptor construction: mock agent unit test.
- State dispatch selection: unit test.
- Real RDMA post completion: gated integration test.
- Accuracy under PD: expensive E2E.

## 035. NIXL Staging Buffer Risks

Staging exists for heterogeneous TP non-MLA models.

Prefill:

- Initializes per-worker staging buffers.
- Builds a worker-local staging strategy lazily.
- Prefetches STAGING_REQ before enqueue.
- Routes all chunks for a room to `room % len(transfer_queues)`.

Decode:

- Initializes staging allocator.
- Handles STAGING_REQ.
- Sends STAGING_RSP.
- Tracks watermarks.
- Submits final scatter once all KV and aux are done.

Test risks:

- Shared staging strategy across workers would race.
- Missing prefetch can delay transfer.
- Oversized chunk should raise actionable error.
- Decode buffer too small should fall back.
- `nokv` aux path should still complete.
- `agent_name` with underscores must parse.

## 036. NIXL Failure Risks

Important failure paths:

- NIXL import missing.
- Backend params invalid JSON or wrong shape.
- Plugin missing.
- Memory registration returns empty descs.
- Transfer init returns false.
- Transfer post returns `ERR`.
- `check_xfer_state` returns `ERR`.
- Prefill node heartbeat failures.
- Waiting timeout.
- Bootstrap info cannot be fetched.
- Staging allocation oversized.

Good tests assert:

- Failure is recorded with useful reason.
- Status moves to `KVPoll.Failed`.
- Poll returns failed.
- `failure_exception()` raises the original exception when available.
- Pending room state is removed or not resurrected incorrectly.

## 037. NIXL Unit Test Template: TransferInfo

```python
import unittest

import numpy as np

from sglang.srt.disaggregation.nixl.conn import TransferInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNixlTransferInfo(CustomTestCase):
    def test_decode_radix_full_hit_is_not_dummy(self):
        msg = [
            b"7",
            b"127.0.0.1",
            b"12345",
            b"agent_a",
            np.array([], dtype=np.int32).tobytes(),
            b"3",
            b"1",
            b"",
            b"12",
        ]
        info = TransferInfo.from_zmq(msg)

        self.assertEqual(info.room, 7)
        self.assertEqual(info.decode_prefix_len, 12)
        self.assertFalse(info.is_dummy())


if __name__ == "__main__":
    unittest.main()
```

## 038. NIXL Unit Test Template: Notifications

```python
import unittest
from collections import defaultdict

from sglang.srt.disaggregation.nixl.conn import NixlKVManager, TransferStatus
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class FakeAgent:
    def __init__(self, messages):
        self.messages = messages

    def get_new_notifs(self):
        return {"peer": [m.encode("ascii") for m in self.messages]}


class TestNixlNotifications(CustomTestCase):
    def test_staging_agent_name_can_contain_underscores(self):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = FakeAgent(["5_stg_0_1_0_2_0_8_agent_with_underscores"])
        mgr.transfer_statuses = defaultdict(TransferStatus)
        mgr.required_prefill_response_num_table = {5: 1}
        mgr.enable_staging = False

        called = []
        mgr._handle_staging_chunk_arrived = lambda *args: called.append(args)

        mgr.update_transfer_status()

        self.assertEqual(called[0][-1], "agent_with_underscores")
        self.assertEqual(mgr.transfer_statuses[5].expected_kvs_per_pp[0], 1)


if __name__ == "__main__":
    unittest.main()
```

This template avoids `NixlKVManager.__init__` and sets only the attributes that
`update_transfer_status()` reads for this path.

## 039. NIXL Unit Test Template: Backend Params

```python
import os
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNixlBackendParams(CustomTestCase):
    @patch.dict(os.environ, {"SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS": "[]"})
    def test_backend_params_must_be_object(self):
        # Prefer testing the exact parser helper if one exists.
        # If no helper exists, consider extracting one before adding fragile constructor tests.
        with self.assertRaises(ValueError):
            self._construct_minimal_manager()

    def _construct_minimal_manager(self):
        raise NotImplementedError("Fill with minimal mocked construction.")


if __name__ == "__main__":
    unittest.main()
```

Better engineering option:

- Extract a small helper that parses and validates backend params.
- Unit-test the helper.
- Keep `NixlKVManager.__init__` integration coverage small.

## 040. NIXL E2E Template

```python
import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST

register_cuda_ci(
    est_time=300,
    stage="base-b",
    runner_config="2-gpu-large",
    disabled="requires validated NIXL install on CI runner",
)


def _has_nixl():
    try:
        import nixl._api  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_has_nixl(), "NIXL is required for this test.")
class TestDisaggregationNixlBasic(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
        cls.rdma_devices = []
        cls.launch_all()

    def test_completion_returns_text(self):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("text", response.json())
        self.assertGreater(len(response.json()["text"]), 0)


if __name__ == "__main__":
    unittest.main()
```

Use this only when the behavior needs real NIXL.

## 041. Decode Radix Cache Test Pattern

Reference:

- @test/registered/distributed/test_disaggregation_decode_radix_cache.py

It tests:

- Decode radix cache enabled in server info.
- Multi-turn requests produce cached tokens.
- Workers stay alive after requests.
- GSM8K accuracy remains above threshold.
- Second pass accuracy does not drop too much.

It uses:

- `PDDisaggregationServerBase`
- `run_multiturn_cache_hit_test`
- `run_eval`
- `DEFAULT_MODEL_NAME_FOR_TEST`
- Transfer backend mixin.

NIXL class is currently skipped:

```python
@unittest.skip("Temporarily disabled until nixl backend is stable.")
class TestDisaggregationDecodeRadixCacheNixl(...):
    transfer_backend_name = "nixl"
```

Agent rule:

- Do not unskip NIXL radix cache coverage without proving stability locally or in the target CI environment.
- Add unit tests for the NIXL radix full-hit transfer path before relying on E2E.

## 042. Decode Radix Cache NIXL Specifics

NIXL handles decode-side radix cache full hits specially.

Relevant code:

- `TransferInfo.is_dummy`
- `NixlKVSender.should_send_kv_chunk`
- `NixlKVManager.transfer_worker`
- `_handle_aux_notification`

When `dst_kv_indices` is empty and `decode_prefix_len > 0`:

- Transfer is not dummy.
- KV RDMA may send no pages.
- Aux still must be sent.
- Aux notification uses `aux_nokv`.
- Decode marks expected KV count as zero for that PP rank.
- Request can complete.

Minimum regression tests:

- `TransferInfo.is_dummy()` returns false for empty dst indices plus decode prefix.
- `NixlKVSender.should_send_kv_chunk(num_pages=0, last_chunk=True)` returns true.
- `_handle_aux_notification` with `aux_nokv` sets `expected_kvs_per_pp[pp_rank] = 0`.
- `TransferStatus.is_done()` succeeds when aux arrived, expected count is zero, and no state expected.

## 043. Health Checks in E2E

When a test exercises failure-sensitive behavior, check all processes after requests.

Pattern:

```python
def _assert_process_healthy(self, name, process, url):
    self.assertIsNotNone(process, f"{name} process was not started")
    self.assertIsNone(
        process.poll(),
        f"{name} exited unexpectedly with code {process.returncode}",
    )
    response = requests.get(f"{url}/health", timeout=10)
    response.raise_for_status()
```

Check:

- Load balancer.
- Prefill server.
- Decode server.

Add a short idle sleep only if the bug manifests after async cleanup.

## 044. Accuracy Tests

Use accuracy tests sparingly.

Good accuracy coverage:

- Feature can silently corrupt model output.
- Unit assertions cannot observe the behavior.
- The failure mode is realistic under serving.

Bad accuracy coverage:

- Pure argument parser change.
- Logging-only change.
- Simple serialization change.

Example with `run_eval`:

```python
from types import SimpleNamespace
from sglang.test.run_eval import run_eval

args = SimpleNamespace(
    base_url=self.base_url,
    eval_name="gsm8k",
    api="completion",
    max_tokens=512,
    num_examples=200,
    num_threads=128,
)
metrics = run_eval(args)
self.assertGreater(metrics["score"], 0.62)
```

For repeated-run cache tests:

```python
accuracy_drop = metrics_first["score"] - metrics_second["score"]
self.assertLessEqual(accuracy_drop, 0.03)
```

## 045. Cache Hit Tests

Use cache-hit kit when testing radix behavior:

```python
from sglang.test.kits.cache_hit_kit import run_multiturn_cache_hit_test
```

Example:

```python
result = run_multiturn_cache_hit_test(
    base_url=self.base_url,
    model_path=self.model,
    num_clients=4,
    num_rounds=3,
    request_length=384,
    output_length=64,
    max_parallel=4,
)
self.assertGreater(result["overall"]["total_cached_tokens"], 0)
```

For NIXL transfer-backend tests, cache-hit assertions are useful because they stress:

- Decode prefix length.
- Empty KV chunk handling.
- Aux transfer completion.
- Worker liveness after partial transfer.

## 046. Model Choice

Use the smallest model that gives signal.

Common constants from `sglang.test.test_utils`:

- `DEFAULT_SMALL_MODEL_NAME_FOR_TEST`
- `DEFAULT_SMALL_MODEL_NAME_FOR_TEST_BASE`
- `DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN`
- `DEFAULT_MODEL_NAME_FOR_TEST`
- `DEFAULT_MOE_MODEL_NAME_FOR_TEST`
- `DEFAULT_DRAFT_MODEL_EAGLE3`
- `DEFAULT_TARGET_MODEL_EAGLE3`

Guidance:

- Unit tests: no model.
- Small server smoke: `DEFAULT_SMALL_MODEL_NAME_FOR_TEST`.
- Qwen-specific parser or model behavior: Qwen test constant.
- General accuracy/performance: `DEFAULT_MODEL_NAME_FOR_TEST`.
- MoE/DeepSeek behavior: use specific large model only when required.

Use `try_cached_model` when large model download cost matters:

```python
from sglang.test.test_utils import try_cached_model

cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST)
```

## 047. Qwen and DeepSeek Context

@docs/basic_usage/qwen3_5.md highlights Qwen 3.5:

- Hybrid attention.
- MoE with shared experts.
- Multimodal support.
- AMD-specific attention backend guidance.
- Reasoning and tool-call parser flags.

@docs/basic_usage/deepseek_v3.md highlights DeepSeek:

- MLA optimizations.
- DP attention.
- Multi-node tensor parallelism.
- FP8 details.
- Speculative decoding/MTP.
- Reasoning and tool calling.

Testing implication:

- Do not use giant Qwen/DeepSeek models for generic behavior.
- Use them only when the feature depends on their architecture.
- For DeepSeek DP attention or MLA-specific PD transfer, pick suites with enough GPUs.
- For non-MLA heterogeneous TP staging, use a model with MHA/GQA-style KV layout.

## 048. Install and Dependency Context

@docs/get_started/install.md documents:

- pip/uv install.
- source install.
- Docker.
- CUDA 13 notes.
- FlashInfer common issues.
- shared memory requirements.

NIXL docs in @docs/advanced_features/pd_disaggregation.md document:

```bash
pip install nixl
```

or source build:

```bash
git clone https://github.com/ai-dynamo/nixl.git
cd nixl
pip install . --config-settings=setup-args="-Ducx_path=/path/to/ucx"
```

Agent rule:

- Do not make CI tests install dependencies at runtime.
- Tests can skip if dependency missing.
- CI images should own dependency installation.
- Local docs can mention install commands.

## 049. Docs Context

@docs_new/README.md says new docs are Mintlify `.mdx` files and recommends concise active voice.

When adding docs for tests:

- Keep commands runnable.
- Prefer active voice.
- Include exact file paths.
- Add navigation only if adding a docs page under `docs_new`.

For this guide, root-level Markdown is fine because it is an agent playbook, not public docs navigation.

## 050. Server Argument Tests

Server argument tests should usually be unit tests.

Example target:

- `--disaggregation-transfer-backend`
- `--disaggregation-mode`
- `--disaggregation-bootstrap-port`
- `--disaggregation-ib-device`
- `--disaggregation-decode-enable-offload-kvcache`
- `--disaggregation-decode-polling-interval`
- `--page-size`
- `--kv-cache-dtype`

Use @test/registered/unit/server_args/test_server_args.py as nearby style.

Do not launch a server just to validate parsing.

## 051. Environment Variable Tests

Environment variables control many disaggregation paths.

Use `patch.dict`:

```python
import os
from unittest.mock import patch

@patch.dict(os.environ, {"SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1"})
def test_timeout(self):
    ...
```

Clean up manual env changes in `tearDownClass`.

Bad:

```python
os.environ["SOME_FLAG"] = "1"
```

without cleanup.

Good:

```python
@classmethod
def tearDownClass(cls):
    os.environ.pop("SOME_FLAG", None)
    super().tearDownClass()
```

## 052. Local Environment for PD Tests

Local PD fixture can read:

```text
SGLANG_TEST_PD_DISAGG_BACKEND
SGLANG_TEST_PD_DISAGG_DEVICES
```

Example:

```bash
export SGLANG_TEST_PD_DISAGG_BACKEND=nixl
export SGLANG_TEST_PD_DISAGG_DEVICES=mlx5_0,mlx5_1
python3 test/registered/disaggregation/test_disaggregation_basic.py
```

If no devices are specified, fixture warns and uses defaults.

For NIXL tests that do not need RDMA devices, a subclass can set:

```python
cls.rdma_devices = []
```

## 053. RDMA Device Handling

@python/sglang/test/server_fixtures/disaggregation_fixture.py auto-detects RDMA devices for CI.

It checks:

- `/sys/class/infiniband`
- active port state.
- link rate.
- device name filters.
- `CUDA_VISIBLE_DEVICES`.
- `SGLANG_CI_RDMA_ALL_DEVICES`.

Do not hardcode `mlx5_0` in general CI tests.

Do hardcode only in manual repro instructions or dedicated environment tests.

## 054. Router in PD Tests

`PDDisaggregationServerBase.launch_lb()` runs:

```bash
python3 -m sglang_router.launch_router \
  --pd-disaggregation \
  --mini-lb \
  --prefill <prefill_url> \
  --decode <decode_url> \
  --host <base_host> \
  --port <lb_port>
```

Requests should usually go to:

```python
self.base_url
```

or:

```python
self.lb_url
```

Health checks should include both backend servers when testing crash/leak behavior.

## 055. Request Patterns

Basic generate:

```python
response = requests.post(
    self.base_url + "/generate",
    json={
        "text": "The capital of France is",
        "sampling_params": {"temperature": 0, "max_new_tokens": 16},
    },
    timeout=60,
)
self.assertEqual(response.status_code, 200, response.text)
```

Logprob request:

```python
response = requests.post(
    self.lb_url + "/generate",
    json={
        "text": "The capital of france is ",
        "sampling_params": {"temperature": 0},
        "return_logprob": True,
        "return_input_logprob": True,
        "logprob_start_len": 0,
    },
    timeout=60,
)
```

OpenAI client:

```python
import openai

client = openai.Client(api_key="empty", base_url=f"{self.lb_url}/v1")
res = client.completions.create(model="dummy", prompt="Hello").model_dump()
```

Always set timeouts for `requests`.

## 056. Assertions

Good assertions include context:

```python
self.assertEqual(response.status_code, 200, response.text)
```

Good:

```python
self.assertGreater(
    result["overall"]["total_cached_tokens"],
    0,
    "expected decode radix cache to reuse at least some tokens",
)
```

Bad:

```python
assert response.ok
```

Prefer `self.assert...` methods in unittest classes.

## 057. Avoid Bare pytest Main Issues

SGLang has a unit test guarding test entrypoints:

- @test/registered/unit/test_no_bare_pytest_main.py

Do not create files that break runner assumptions.

Use the local patterns.

## 058. Test Naming

Files:

```text
test_<feature>.py
```

Classes:

```python
class TestFeature(CustomTestCase):
```

Methods:

```python
def test_specific_behavior(self):
```

For mixins:

```python
class DisaggregationDecodeRadixCacheTestMixin:
```

Mixin should not inherit `CustomTestCase` directly if combined with fixture base.

Concrete class should inherit:

```python
class TestThing(Mixin, PDDisaggregationServerBase):
    ...
```

## 059. Mixins

Use mixins to share behavior across transfer backends.

Example:

```python
class DisaggregationDecodeRadixCacheTestMixin:
    transfer_backend_name = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.transfer_backend = [
            "--disaggregation-transfer-backend",
            cls.transfer_backend_name,
        ]
        cls.launch_all()
```

Concrete backends:

```python
class TestFeatureMooncake(Mixin, PDDisaggregationServerBase):
    transfer_backend_name = "mooncake"


class TestFeatureNixl(Mixin, PDDisaggregationServerBase):
    transfer_backend_name = "nixl"
```

Skip each backend separately if dependencies differ.

## 060. Skips and Dependency Probes

Dependency probe:

```python
def _has_nixl():
    try:
        import nixl._api  # noqa: F401
    except ImportError:
        return False
    return True
```

Use:

```python
@unittest.skipUnless(_has_nixl(), "NIXL is required for this test.")
```

For Mooncake:

```python
def _has_mooncake():
    try:
        import mooncake.engine  # noqa: F401
    except ImportError:
        return False
    return True
```

CI-specific logic:

```python
from sglang.test.test_utils import is_in_ci

@unittest.skipUnless(is_in_ci() or _has_mooncake(), "Mooncake is required.")
```

Do not skip silently.

The reason must say what is missing.

## 061. Testing Missing Dependency Behavior

If testing missing NIXL import:

- Patch import path carefully.
- Prefer testing a small helper if available.
- Avoid making the test environment depend on actually missing packages.

Example approach:

```python
with patch.dict("sys.modules", {"nixl._api": None}):
    ...
```

But this can be fragile if module import already happened.

Better:

- Extract helper `load_nixl_agent_classes()`.
- Unit-test helper with patching.
- Keep constructor behavior covered by one test.

## 062. Testing NIXL Agent Interactions

Mock agent should support:

- `register_memory`
- `create_backend`
- `get_plugin_list`
- `add_remote_agent`
- `get_xfer_descs`
- `initialize_xfer`
- `transfer`
- `check_xfer_state`
- `get_new_notifs`
- `get_agent_metadata`
- `name`

Small fake:

```python
class FakeAgent:
    name = "fake_agent"

    def __init__(self):
        self.created_backend = None
        self.registered = []

    def create_backend(self, backend, params):
        self.created_backend = (backend, params)

    def get_plugin_list(self):
        return ["UCX"]

    def register_memory(self, addrs, kind):
        self.registered.append((addrs, kind))
        return ["desc"]
```

Use fake agents to verify SGLang descriptor construction without requiring NIXL runtime.

## 063. Testing Transfer Descriptor Construction

Targets:

- `_send_kvcache_generic`
- `send_kvcache_slice`
- `send_aux`
- `_send_mamba_state`
- `_send_mamba_state_slice`

Assertions:

- Calls `get_xfer_descs` with `VRAM` for KV/state.
- Calls `get_xfer_descs` with `DRAM` for aux.
- Uses `WRITE`.
- Encodes notification as ASCII.
- Groups contiguous indices.
- Uses `np.uint64`-safe pointer math.
- Returns transfer handle.
- Raises when handle missing or transfer returns `ERR`.

Keep these tests CPU if they use fake pointers only.

## 064. Testing Transfer Worker

Transfer worker is threaded and loops forever.

Avoid direct thread tests unless necessary.

Prefer extracting helper logic or testing one iteration through a fake queue with controlled exception.

If you must test worker behavior:

- Use a fake queue that returns one chunk then raises a sentinel exception.
- Patch methods to avoid real NIXL.
- Assert status and recorded failures.
- Do not leave daemon threads running if avoidable.

Better test units:

- `_do_staging_transfer`
- `_handle_aux_notification`
- `_track_kv_arrival`
- `_maybe_submit_last_scatter`
- `TransferStatus.is_done`

## 065. Testing Heartbeat Failure

`_start_heartbeat_checker_thread` loops forever.

Do not start it in unit tests.

Test `_handle_node_failure` directly.

Set:

- `connection_pool`
- `prefill_info_table`
- `addr_to_rooms_tracker`
- `transfer_statuses`
- `request_status`

Assert:

- Connection pool keys removed.
- Prefill info removed.
- Rooms marked failed.
- Status updated to `KVPoll.Failed`.

## 066. Testing Waiting Timeout

Target:

- `NixlKVReceiver.poll`

Set:

- `started_transfer = True`
- `init_time` in past.
- `kv_mgr.waiting_timeout`
- `kv_mgr.record_failure`
- `kv_mgr.update_status`

Assert:

- `KVPoll.Failed`.
- Failure reason includes room and elapsed.

Use `patch("time.time")` for deterministic tests.

## 067. Testing Bootstrap Info Fetch

Common manager method:

- `try_ensure_parallel_info`

Test:

- HTTP 200 caches info.
- Non-200 returns false.
- Request exception returns false.
- Page size mismatch raises.
- KV cache dtype mismatch raises.
- Rank mapping is resolved.

Patch:

```python
@patch("sglang.srt.disaggregation.common.conn.requests.get")
```

Use `MagicMock` manager object if constructor is expensive.

## 068. Testing Rank Mapping

Target:

- `CommonKVManager._resolve_rank_mapping`

Cases:

- Same TP size.
- Decode TP larger than prefill TP.
- Prefill TP larger than decode TP.
- MLA backend.
- Non-MLA warning path.
- CP size mapping.
- `SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER`.
- PP size same.
- Decode PP size 1 with prefill PP larger.
- Invalid decode CP size assertion.
- Invalid PP relationship assertion.

Rank mapping bugs produce wrong KV routing, so unit tests here are valuable.

## 069. Testing PP Pointer Slicing

Targets:

- `get_mha_kv_ptrs_with_pp`
- `get_mla_kv_ptrs_with_pp`
- `_mla_slice_ptrs_for_pp`

Cases:

- Same PP layout.
- Decode has full-model list while prefill stage has subrange.
- Draft model KV layout.
- Compressed-MLA kv_data layout.
- Compressed-MLA state_data layout.
- Missing `prefill_end_layer` raises.
- Unexpected lengths raise.

These should be CPU unit tests with fake pointers.

## 070. Testing Hybrid State

State types:

- `MAMBA`
- `SWA`
- `NSA`

Target:

- `maybe_send_extra`

Cases:

- Empty state indices sends nothing.
- Mamba homogeneous TP calls `_send_mamba_state`.
- Mamba heterogeneous TP calls `_send_mamba_state_slice`.
- SWA/NSA non-MLA heterogeneous TP raises.
- SWA/NSA index length mismatch raises.
- Unknown state type raises.

Use mocks, not real GPU memory.

## 071. Testing Aux Transfer

Aux transfer must complete even when no KV pages transfer.

Target:

- `send_aux`
- `_handle_aux_notification`
- `TransferStatus.is_done`

Cases:

- Normal aux notification sets `received_aux`.
- `aux_nokv` sets expected KV count zero.
- Done requires aux.
- Done requires expected KV count for every PP rank.
- Done requires state if expected.

## 072. NIXL E2E Gate Checklist

Before adding or enabling a real NIXL E2E test:

- [ ] NIXL dependency installed in target environment.
- [ ] Required plugin installed.
- [ ] Backend env defaults known.
- [ ] Runner has required GPU or XPU.
- [ ] Test uses smallest possible model.
- [ ] Test has dependency skip.
- [ ] Test has explicit registration disabled if CI cannot run it.
- [ ] Test has health checks.
- [ ] Test has request timeouts.
- [ ] Test cleans all processes.
- [ ] Test does not install packages during CI.
- [ ] Test failure message distinguishes missing dependency from behavior failure.

## 073. PD E2E Checklist

Before adding a PD E2E test:

- [ ] Feature cannot be covered by unit tests.
- [ ] Uses `PDDisaggregationServerBase`.
- [ ] Calls `super().setUpClass()`.
- [ ] Sets `cls.model`.
- [ ] Sets feature args before `launch_all()`.
- [ ] Calls `cls.launch_all()`.
- [ ] Sends requests to load balancer.
- [ ] Checks `/health` on relevant processes.
- [ ] Uses timeouts for requests.
- [ ] Registers to a suite with enough GPUs.
- [ ] Uses `try_cached_model` when appropriate.
- [ ] Avoids extra backends unless feature is backend-specific.

## 074. Unit Test Checklist

Before adding a unit test:

- [ ] Located under `test/registered/unit/`.
- [ ] Mirrors source tree where practical.
- [ ] Uses `CustomTestCase`.
- [ ] Does not launch server or engine.
- [ ] Does not load real model weights.
- [ ] Mocks expensive dependencies.
- [ ] Registers to `base-a-test-cpu` unless GPU required.
- [ ] Uses literal registration values.
- [ ] Has `if __name__ == "__main__"`.
- [ ] Has specific assertion messages for non-obvious failures.

## 075. Review Checklist

Review every test as if it will run in CI under load.

Check:

- Does it leak processes?
- Does it rely on global state?
- Does it mutate environment without cleanup?
- Does it assume a port?
- Does it assume dependency presence?
- Does it assume model download?
- Does it over-register across backends?
- Does it use a bigger model than needed?
- Does it run too long for its suite?
- Does it have deterministic assertions?
- Does it fail with useful output?
- Does it use current registration style?
- Does `test/run_suite.py` discover it?

## 076. Common Bad Patterns

Bad: launching a server in unit tests.

Bad: using raw `unittest.TestCase`.

Bad: missing `tearDownClass`.

Bad: non-literal CI registration.

Bad: registering AMD for a backend-independent HTTP test.

Bad: using DeepSeek 671B for generic JSON validation.

Bad: checking only response code and not response body.

Bad: no timeout on requests.

Bad: modifying env vars without cleanup.

Bad: installing packages in test setup.

Bad: sleeping long durations instead of waiting on a condition.

Bad: unskipping flaky NIXL tests without targeted unit coverage.

## 077. Common Good Patterns

Good: unit-test wire packing and rank mapping.

Good: use PD fixture for real PD lifecycle.

Good: split backend mixin from concrete backend classes.

Good: skip NIXL class when NIXL dependency is absent.

Good: health-check prefill/decode/router after stress requests.

Good: assert accuracy does not degrade between cache runs.

Good: keep CI suite small and targeted.

Good: test failure cleanup paths directly with mocks.

Good: use `try_cached_model` for large model tests.

Good: make skip reasons explicit.

## 078. Example: Add NIXL Full-Hit Unit Coverage

Goal:

Protect decode radix cache full-hit path without launching NIXL.

Suggested file:

```text
test/registered/unit/disaggregation/test_nixl_decode_radix_full_hit.py
```

Registration:

```python
register_cpu_ci(est_time=3, suite="base-a-test-cpu")
```

Test cases:

- Empty `dst_kv_indices` plus `decode_prefix_len` is not dummy.
- `should_send_kv_chunk(0, last_chunk=True)` is true.
- `aux_nokv` notification marks expected KV count zero.
- `TransferStatus.is_done()` completes after aux for zero expected KV.

Why this belongs in unit CI:

- It covers logic in @python/sglang/srt/disaggregation/nixl/conn.py.
- It requires no real NIXL plugin.
- It catches regressions before expensive distributed E2E.

## 079. Example: Add NIXL Plugin Validation Unit Coverage

Goal:

Validate operator error when backend plugin is missing.

Suggested approach:

- Patch `nixl_agent`.
- Fake `get_plugin_list()` to return `["UCX"]`.
- Set env backend to `LIBFABRIC`.
- Assert `ValueError` includes available plugins.

Prefer extracting constructor parsing if current constructor is too expensive.

Expected message shape:

```text
NIXL backend 'LIBFABRIC' not found. Available: ['UCX'].
```

## 080. Example: Add NIXL Staging Notification Coverage

Goal:

Ensure staging notification parsing handles agent names with underscores.

Suggested file:

```text
test/registered/unit/disaggregation/test_nixl_notifications.py
```

Cases:

- `5_stg_0_1_0_2_0_8_agent_with_underscores`
- Assert agent name is `agent_with_underscores`.
- Assert KV arrival tracked.
- Assert expected chunks set on last chunk.

This guards the `split("_", 8)` behavior.

## 081. Example: Add PD NIXL Smoke Test

Goal:

Verify real NIXL transfer backend completes one request.

Suggested location:

```text
test/registered/disaggregation/test_disaggregation_nixl_basic.py
```

Registration:

```python
register_cuda_ci(
    est_time=300,
    stage="base-b",
    runner_config="2-gpu-large",
    disabled="requires NIXL-enabled CI runner",
)
```

Class:

```python
@unittest.skipUnless(_has_nixl(), "NIXL is required for this test.")
class TestDisaggregationNixlBasic(PDDisaggregationServerBase):
    ...
```

Request:

```python
POST /generate
```

Assertions:

- HTTP 200.
- response has `text`.
- text non-empty.
- prefill/decode/router healthy.

Only enable in CI after runner image and plugin availability are confirmed.

## 082. Example: Add NIXL Decode Radix Cache E2E

Goal:

Unskip and harden `TestDisaggregationDecodeRadixCacheNixl`.

Do this only after:

- Unit tests cover full-hit logic.
- Basic NIXL smoke is stable.
- CI environment has NIXL.
- Runtime is measured.
- Flake causes are triaged.

Expected registration suite:

```python
register_cuda_ci(est_time=300, stage="base-c", runner_config="8-gpu-h20")
```

Possible reason for base-c:

- Large model.
- PD launch.
- cache-hit workload.
- GSM8K repeated eval.

Do not move to smaller suite unless model and GPU count are also reduced.

## 083. Debugging CI Discovery

Run:

```bash
python3 test/run_suite.py --hw cpu --suite base-a-test-cpu
```

or:

```bash
python3 test/run_suite.py --hw cuda --suite base-b-test-1-gpu-small
```

Look for:

- No tests found.
- Invalid suite.
- Missing main entry.
- AST parse errors.
- Skipped disabled files.
- Unexpected partition assignment.

If a file is not discovered:

- Is it under `test/registered`?
- Is it `*.py`?
- Is it excluded helper path?
- Does it have top-level registration?
- Does registration use literal values?
- Does it have main entry?

## 084. Debugging E2E Launch Failures

Check:

- Model path.
- Hugging Face token/cache.
- GPU memory.
- `--mem-fraction-static`.
- `--tp`, `--dp`, `--base-gpu-id`.
- Port collisions.
- Router dependency.
- RDMA devices.
- NIXL plugin availability.
- Environment variables.
- `DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH`.

Use health endpoints:

```text
/health
/health_generate
/server_info
```

Add failure output:

```python
self.assertEqual(response.status_code, 200, response.text)
```

## 085. Debugging NIXL Transfer Failures

Map symptom to likely area:

`ImportError: Please install NIXL`

- Missing `nixl` package.

`NIXL backend '<name>' not found`

- Plugin not installed or wrong `SGLANG_DISAGGREGATION_NIXL_BACKEND`.

Memory registration failed:

- Bad pointer.
- Wrong memory kind.
- Backend cannot register target memory.

Transfer init failed:

- Remote agent not added.
- Descriptors invalid.
- Plugin issue.

Transfer `ERR`:

- Transport failure.
- Bad remote memory.
- Node disconnect.

Waiting timeout:

- Decode metadata sent but transfer never completed.
- Prefill did not mark room ready.
- Notification not received.
- Aux path missing.

Full-hit radix cache hangs:

- `aux_nokv` or expected zero chunk count may be broken.

Staging hangs:

- STAGING_REQ/RSP/watermark flow may be broken.
- Final scatter may not submit.

## 086. Debugging Flakiness

Common flake causes:

- Model download latency.
- Server launch timeout.
- GPU memory pressure.
- Port conflicts.
- Long GC pauses.
- RDMA device mismatch.
- NIXL plugin race.
- Heartbeat thresholds too tight.
- Accuracy threshold too tight.
- Background threads swallowing exceptions.

Actions:

- Add targeted unit test for logic.
- Reduce E2E workload.
- Increase timeout only with evidence.
- Make failure path visible.
- Keep health checks after async workloads.
- Use `is_in_ci()` only when local/CI behavior must differ.

## 087. When to Add New Helpers

Add a helper when:

- Three tests repeat the same launch code.
- Constructor is too expensive and a pure parser can be extracted.
- Error-prone notification parsing is embedded in a long method.
- A state transition can be isolated.

Do not add a helper when:

- It hides the behavior under test.
- It introduces cross-test mutable state.
- It creates a new abstraction for one assertion.

## 088. When to Modify Production Code for Testability

Acceptable:

- Extract pure helper from constructor.
- Add small parser function.
- Add explicit error type or message.
- Add state accessor used by tests and diagnostics.

Not acceptable in a test-only change:

- Change runtime semantics.
- Add sleeps.
- Disable checks.
- Add test-only flags unless existing pattern supports it.
- Weaken errors to make tests pass.

## 089. Large Model Test Discipline

Large model tests must justify:

- Why small model is insufficient.
- Why unit coverage is insufficient.
- Why suite has enough GPU memory.
- Why runtime belongs in per-commit or nightly.

Use @docs/basic_usage/deepseek_v3.md for DeepSeek-specific architecture requirements.

Use @docs/basic_usage/qwen3_5.md for Qwen-specific architecture requirements.

Do not use large models for generic HTTP tests.

## 090. Performance Test Discipline

Performance tests should:

- Use stable thresholds.
- Record enough context in failure.
- Avoid running in early suites unless critical.
- Separate correctness from perf when possible.
- Use retry behavior only for known accuracy/perf flake classes.

Do not encode fragile latency thresholds in small shared CI unless runner variance is known.

## 091. Accuracy Threshold Discipline

Accuracy thresholds should:

- Match nearby tests.
- Have enough examples to be meaningful.
- Avoid being too close to expected score.
- Account for repeated-run variance.
- Print metrics for diagnosis.

Bad:

```python
self.assertGreater(metrics["score"], 0.999)
```

unless the task is deterministic and exact.

Good:

```python
self.assertGreater(metrics["score"], 0.62)
```

when nearby PD test uses same model and benchmark.

## 092. Request Timeout Discipline

Always include request timeouts:

```python
requests.get(url, timeout=10)
requests.post(url, json=payload, timeout=60)
```

Timeout choice:

- Health: 10 seconds.
- Simple generate: 30 to 60 seconds.
- Long eval: handled by eval utility.
- Server launch: `DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH`.

## 093. Sleep Discipline

Avoid fixed sleeps.

Prefer:

- `wait_for_http_ready`
- Polling a condition.
- Health endpoint.
- Process poll.

Accept short sleep when:

- Async cleanup is expected.
- The test documents why.
- Sleep is small.

Example:

```python
# Give schedulers a short idle window so post-request crash paths surface.
time.sleep(5)
```

## 094. Global State Discipline

Global state includes:

- Environment variables.
- Imported modules.
- Class attributes on fixtures.
- CUDA visible devices.
- Default backend envs.
- Requests sessions.
- NIXL agent state.

Tests must not leak global state into later files.

Use:

- `patch.dict`.
- `addCleanup`.
- `tearDown`.
- `tearDownClass`.

## 095. Thread Discipline

NIXL manager starts background threads.

Unit tests should avoid starting them.

When testing methods on `NixlKVManager`, prefer:

```python
mgr = object.__new__(NixlKVManager)
```

then set attributes manually.

Do not call `NixlKVManager.__init__` unless the constructor itself is under test and all side effects are mocked.

## 096. Socket Discipline

Disaggregation uses ZMQ sockets and HTTP bootstrap.

Unit tests should not bind real sockets unless testing network utility behavior.

Use fake endpoints and mock `requests`.

Use real sockets only in integration tests.

## 097. Port Discipline

Use `DEFAULT_URL_FOR_TEST` and fixture-derived ports.

Do not hardcode `30000` in CI tests unless following existing fixture pattern and no collision risk exists.

PD fixture offsets:

- load balancer: base port.
- prefill: base port + 100.
- decode: base port + 200.
- bootstrap: base port + 500.

## 098. Backend Independence

A feature is backend-independent when it tests:

- HTTP response shape.
- Request validation.
- Common scheduling logic.
- OpenAI API compatibility.
- Tokenizer behavior.

Do not register multiple hardware backends for backend-independent coverage.

A feature is backend-specific when it tests:

- NIXL transfer behavior.
- Mooncake RDMA behavior.
- AMD AITER behavior.
- NPU memfabric behavior.
- XPU pointer overflow behavior.

Then backend registration or skip is appropriate.

## 099. NIXL vs Mooncake Test Strategy

Mooncake currently has more stable PD coverage in existing tests.

NIXL coverage should grow in layers:

1. CPU unit tests for NIXL wire, notification, status, and descriptor logic.
2. Disabled/gated smoke E2E on NIXL.
3. Stable NIXL smoke in a known NIXL runner.
4. Cache-hit E2E.
5. Accuracy repeated-run E2E.
6. Heterogeneous TP staging E2E if runner supports it.

Do not start at layer 5.

## 100. Minimal NIXL Test Matrix

Recommended minimum coverage:

```text
CPU unit:
  TransferInfo.is_dummy full-hit
  TransferStatus.is_done zero-kv
  notification parser stg/aux/state
  KVArgsRegisterInfo.from_zmq optional fields
  backend params validation
  missing plugin error

CUDA or XPU gated:
  basic PD NIXL generate
  process health after request

Large optional:
  decode radix cache cache-hit
  repeated accuracy
  heterogeneous TP staging
```

## 101. File Mention Convention

When writing agent-facing docs or comments, use `@path` mentions for key files:

- @test/run_suite.py
- @python/sglang/srt/disaggregation/nixl/conn.py
- @test/registered/distributed/test_disaggregation_decode_radix_cache.py

When writing code comments, use plain paths sparingly.

Do not add large doc comments inside test files if the test name and assertions are enough.

## 102. Practical Workflow for Agents

Step 1: Identify behavior.

Step 2: Find nearby tests with `rg`.

```bash
rg "Nixl|nixl|PDDisaggregationServerBase|disaggregation-transfer-backend" test/registered python/sglang/test
```

Step 3: Choose test level.

Step 4: Choose placement.

Step 5: Choose registration.

Step 6: Write test with cleanup.

Step 7: Run direct file.

Step 8: Run suite discovery for target suite.

Step 9: Review output and failure messages.

Step 10: Re-read the new test as a future maintainer.

## 103. Commands for Test Authoring

Find registration examples:

```bash
rg "register_cuda_ci|register_cpu_ci|register_amd_ci" test/registered -n
```

Find disaggregation tests:

```bash
rg "PDDisaggregationServerBase|disaggregation-transfer-backend" test/registered -n
```

Find NIXL code:

```bash
rg "Nixl|nixl|NIXL" python/sglang/srt test/registered -n
```

Run a CPU unit file:

```bash
python3 test/registered/unit/disaggregation/test_disaggregation_wire.py
```

Run suite discovery:

```bash
python3 test/run_suite.py --hw cpu --suite base-a-test-cpu
```

## 104. Example Review of a New NIXL Unit Test

Ask:

- Does it import NIXL package? If yes, can it avoid that?
- Does it call `NixlKVManager.__init__`? If yes, are threads mocked?
- Does it patch env safely?
- Does it rely on dict ordering accidentally?
- Does it assert exact failure behavior?
- Does it run on CPU?
- Does it use `CustomTestCase`?
- Does it register CPU?
- Does it have a main entry?

## 105. Example Review of a New PD E2E Test

Ask:

- Does it call `super().setUpClass()`?
- Does it set model before `launch_all()`?
- Does it set extra args before `launch_all()`?
- Does it use `self.base_url` or `self.lb_url`?
- Does it use request timeouts?
- Does it check process health if crash behavior matters?
- Does it rely on local-only env?
- Does it skip missing backend dependency?
- Does it use the right suite and GPU count?
- Does it avoid duplicate backend registrations?

## 106. Common PR Comments to Preempt

Reviewer: "Why is this not a unit test?"

Answer in code design by making it a unit test if possible.

Reviewer: "Why does this run on AMD too?"

Only register AMD if AMD behavior is under test.

Reviewer: "Why base-c?"

Use base-c only for multi-GPU/large-distributed tests.

Reviewer: "This will flake when NIXL is not installed."

Add dependency skip or disabled registration.

Reviewer: "This leaks server process if setup fails."

Use `CustomTestCase` and defensive cleanup.

Reviewer: "This registration is not discoverable."

Use literal values and current registration shape.

## 107. Writing New Test Files: Final Checklist

Before finishing:

- [ ] Read nearby tests.
- [ ] Read relevant source.
- [ ] Pick smallest test type.
- [ ] Use `CustomTestCase`.
- [ ] Register with literals.
- [ ] Use current CUDA `stage`/`runner_config` where appropriate.
- [ ] Add dependency skip if needed.
- [ ] Add cleanup if launching processes.
- [ ] Use request timeouts.
- [ ] Assert useful details.
- [ ] Run direct test if local dependencies allow.
- [ ] Run suite discovery.
- [ ] Re-read the diff.

## 108. NIXL-Specific Final Checklist

- [ ] Does test avoid real NIXL when testing pure logic?
- [ ] Does real NIXL test skip missing `nixl._api`?
- [ ] Does it validate plugin/backend assumptions?
- [ ] Does it cover aux even when KV pages are empty?
- [ ] Does it cover `decode_prefix_len` if touching radix cache?
- [ ] Does it cover staging metadata if touching heterogeneous TP?
- [ ] Does it avoid starting infinite threads in unit tests?
- [ ] Does it use fake agent for descriptor tests?
- [ ] Does it check `KVPoll.Failed` paths?
- [ ] Does it check worker liveness in E2E?

## 109. Source Map for NIXL Tests

Use this map when deciding where to test.

`TransferInfo.from_zmq`

- Unit test.
- No NIXL dependency needed.

`TransferInfo.is_dummy`

- Unit test.
- Important for decode radix cache.

`KVArgsRegisterInfo.from_zmq`

- Unit test.
- Include optional staging fields.

`TransferStatus.is_done`

- Unit test.
- Include aux/state/PP cases.

`NixlKVManager.__init__`

- Constructor integration with heavy mocking, or E2E.
- Prefer extracting helpers for env parsing.

`register_buffer_to_engine`

- Unit test with fake agent.

`send_kvcache_slice`

- Unit test with fake agent descriptors.
- Integration only if validating real transport.

`send_kvcache_staged`

- Unit test for fallback and size checks.
- GPU test for gather/scatter correctness if needed.

`update_transfer_status`

- Unit test with fake agent notifications.

`NixlKVReceiver.poll`

- Unit test for timeout and completion.

`NixlKVSender.send`

- Unit test for status and chunking with fake manager.

## 110. Source Map for Common Disaggregation Tests

`CommonKVManager.register_to_bootstrap`

- Unit test with mocked `requests.put`.
- Existing: @test/registered/unit/disaggregation/test_register_to_bootstrap.py

`CommonKVManager.try_ensure_parallel_info`

- Unit test with mocked `requests.get`.

`CommonKVManager._resolve_rank_mapping`

- Unit test with fake manager.

`get_mha_kv_ptrs_with_pp`

- Unit test with fake pointers.

`get_mla_kv_ptrs_with_pp`

- Unit test with fake pointers.

`CommonKVBootstrapServer`

- Integration if testing HTTP route behavior.

## 111. When a Test Should Be Nightly

Use nightly when:

- Runtime is too long for PR.
- Needs rare hardware.
- Accuracy is broad and not tied to a small PR surface.
- Test is valuable but too expensive for per-commit.
- Flake risk is acceptable for signal gathering but not gating.

Register:

```python
register_cuda_ci(est_time=600, suite="nightly-8-gpu-h200", nightly=True)
```

Do not hide important correctness regressions in nightly if a unit test can catch them.

## 112. When a Test Should Be Extra CI

Use extra suites when:

- Test is per-commit capable but label-gated due to cost.
- Coverage is useful for risky PRs.
- Maintainers can opt in with `run-ci-extra`.

Example:

```python
register_cuda_ci(est_time=310, stage="extra-b", runner_config="8-gpu-h200")
```

## 113. Slash Command Awareness

From CI workflow skill:

- `/tag-run-ci-label`
- `/tag-run-ci-label extra`
- `/rerun-failed-ci`
- `/tag-and-rerun-ci`
- `/rerun-test <test-file>`
- `/rerun-group <group> [<group> ...]`

Agents usually do not need to edit slash command code for test authoring.

But when debugging, know that `/rerun-test` targets test files and `/rerun-group` expands registered groups.

## 114. Partition Awareness

Large suites are partitioned by LPT based on `est_time`.

If you set `est_time` too low:

- Partition becomes imbalanced.
- CI job can exceed expected runtime.
- Later stages can wait longer.

If you set `est_time` too high:

- It can push unrelated tests into other partitions.
- CI may appear less balanced.

Measure locally when feasible.

## 115. Failure Message Style

Good failure:

```python
self.assertLessEqual(
    accuracy_drop,
    0.03,
    f"Second run accuracy dropped by {accuracy_drop:.4f} "
    f"(first={metrics_first['score']:.4f}, second={metrics_second['score']:.4f})",
)
```

Bad failure:

```python
self.assertTrue(ok)
```

Include:

- Actual value.
- Expected bound.
- Context such as backend, room, URL, or response text.

## 116. Logging in Tests

Printing metrics is acceptable in eval tests:

```python
print(f"Evaluation metrics: {metrics}")
```

Avoid excessive logs in unit tests.

Do not depend on log text unless logging is the behavior under test.

## 117. Imports

Preferred ordering:

```python
import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import requests

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
```

Registration usually follows imports.

Some unit README examples place registration before later imports to stub GPU packages first. Follow nearby style when stubbing is required.

## 118. Handling Existing Inconsistencies

The repo may contain older patterns.

Examples:

- Some tests still use raw `unittest.TestCase`.
- Some docs show older registration snippets.
- Some registration uses `suite=` for CUDA per-commit suites.

For new tests:

- Follow current skill guidance.
- Follow @python/sglang/test/ci/ci_register.py constraints.
- Follow nearby tests only when they are current and relevant.

Do not perform broad cleanup while adding one test.

## 119. Agent Do-Not-Do List

Do not:

- Rewrite CI framework to add one test.
- Move existing tests between suites without reason.
- Unskip a backend class casually.
- Add network installs in tests.
- Add sleeps as synchronization.
- Use global mutable stubs without cleanup.
- Use huge models for generic features.
- Register expensive tests in base-a.
- Ignore `test/run_suite.py` validation.
- Leave a test file undiscoverable.

## 120. Agent Done Definition

A test-writing task is done when:

- Test exists in correct location.
- It targets the requested behavior.
- It uses the smallest sufficient test level.
- It registers correctly.
- It has a main entry.
- It cleans resources.
- It is locally run or skipped with explanation.
- Suite discovery passes or failure is explained.
- NIXL-specific dependency/hardware assumptions are explicit.
- Final response tells user what was changed and what was verified.

## 121. Compact Examples Index

CPU unit:

```python
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
```

CUDA small:

```python
register_cuda_ci(est_time=20, stage="base-b", runner_config="1-gpu-small")
```

CUDA large:

```python
register_cuda_ci(est_time=120, stage="base-b", runner_config="1-gpu-large")
```

Two GPU:

```python
register_cuda_ci(est_time=180, stage="base-b", runner_config="2-gpu-large")
```

Distributed PD:

```python
register_cuda_ci(est_time=300, stage="base-c", runner_config="8-gpu-h20")
```

AMD:

```python
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")
```

Nightly:

```python
register_cuda_ci(est_time=600, suite="nightly-8-gpu-h200", nightly=True)
```

Disabled:

```python
register_cuda_ci(
    est_time=300,
    stage="base-b",
    runner_config="2-gpu-large",
    disabled="requires NIXL-enabled CI runner",
)
```

## 122. Quick NIXL Bug-to-Test Table

Bug: full prompt cache hit hangs.

Test: CPU unit for `aux_nokv` and `TransferStatus.is_done`; E2E cache-hit after.

Bug: plugin missing gives unclear crash.

Test: unit constructor/helper validation for plugin list.

Bug: XPU pointer overflow.

Test: XPU-gated E2E like @test/registered/disaggregation/test_disaggregation_xpu.py; unit descriptor dtype checks.

Bug: staging with underscores in agent name hangs.

Test: unit notification parser with underscore agent.

Bug: prefill node death leaves rooms waiting.

Test: unit `_handle_node_failure`.

Bug: Mamba state wrong under hetero TP.

Test: unit `_send_mamba_state_slice` descriptor offsets; model-specific E2E if needed.

Bug: PP compressed MLA wrong layers.

Test: unit `_mla_slice_ptrs_for_pp`.

Bug: bad backend params accepted.

Test: unit JSON object/string validation.

## 123. Suggested Future NIXL Test Files

Useful additions:

```text
test/registered/unit/disaggregation/test_nixl_transfer_info.py
test/registered/unit/disaggregation/test_nixl_notifications.py
test/registered/unit/disaggregation/test_nixl_transfer_status.py
test/registered/unit/disaggregation/test_nixl_backend_config.py
test/registered/unit/disaggregation/test_nixl_descriptor_building.py
test/registered/disaggregation/test_disaggregation_nixl_basic.py
```

Do not create all at once unless the task asks for broad coverage.

Pick the smallest file that matches the bug.

## 124. Final Agent Reminder

SGLang CI is expensive because it launches real serving stacks on scarce hardware.

Default to CPU unit coverage for logic.

Escalate to server tests only when behavior crosses process or model boundaries.

Escalate to PD E2E only when prefill/decode separation matters.

Escalate to NIXL E2E only when real transport matters.

Keep tests direct, registered, clean, and easy to debug.
