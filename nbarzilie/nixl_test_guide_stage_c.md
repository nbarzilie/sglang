# NIXL Disaggregation Functional Test Guide

This guide assumes the container image already has the required SGLang, CUDA,
NIXL, and test dependencies:

```bash
$MY/sqshs/sglang-nixl-functest.sqsh
```

The test file is:

```bash
test/registered/disaggregation/test_disaggregation_nixl.py
```

It launches one prefill server and one decode server with tensor parallel size
4 each, so the job needs 8 visible GPUs.

## Host Setup

Set the shared path root and export the Hugging Face token before running
`srun`. The token is needed for the Llama test models unless they are already
fully cached.

```bash
export MY=/path/to/your/shared/root
export HF_TOKEN=hf_...
mkdir -p "$MY/logs" "$MY/.cache/huggingface"
```

If your site uses a different Hugging Face token variable, keep `HF_TOKEN`
exported anyway. The Hugging Face libraries consume it directly.

## Repository Remotes

In this checkout the remotes are:

```bash
origin   https://github.com/nbarzilie/sglang.git
upstream https://github.com/sgl-project/sglang.git
```

The branch `nixl-func-tests-public-c` lives on the `nbarzilie` fork. The
commands below first inspect `origin`; if it is not the fork inside the
container, they add or update a remote named `nbarzilie` and fetch from that.

## NIXL Backend Environment

The test probes the configured NIXL backend before launching servers. Defaults:

```bash
export SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS='{"num_threads":"8"}'
```

For another NIXL plugin, override both variables. Examples:

```bash
export SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS='{}'
```

```bash
export SGLANG_DISAGGREGATION_NIXL_BACKEND=GDS_MT
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS='{"thread_count":"8"}'
```

## Run The Test Alone

This command enters the `.sqsh` image, pulls the branch, checks that the key
Python dependencies and configured NIXL backend exist, then runs only the NIXL
disaggregation test.

```bash
srun -A network_research_advdev \
     -p interactive \
     -t 2:00:00 \
     --gpus-per-node=8 \
     --cpus-per-task=32 \
     --mem=0 \
     --container-image=$MY/sqshs/sglang-nixl-functest.sqsh \
     --container-workdir=/workspace/sglang \
     --container-mounts=$MY/.cache/huggingface:/root/.cache/huggingface,$MY/logs:/logs \
     --export=ALL,HF_TOKEN,MY,SGLANG_DISAGGREGATION_NIXL_BACKEND,SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS \
     bash -lc '
set -euxo pipefail

git config --global --add safe.directory /workspace/sglang
git remote -v
if git remote get-url origin | grep -q "github.com/nbarzilie/sglang"; then
  TEST_REMOTE=origin
else
  git remote remove nbarzilie 2>/dev/null || true
  git remote add nbarzilie https://github.com/nbarzilie/sglang.git
  TEST_REMOTE=nbarzilie
fi
git fetch --no-tags "$TEST_REMOTE" nixl-func-tests-public-c
git checkout -B nixl-func-tests-public-c FETCH_HEAD

SGL_KERNEL_VERSION=0.4.4
NIXL_VERSION=1.3.0
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda or \"\")")
ARCH=$(uname -m)
PIP_INSTALL=(python3 -m pip install --break-system-packages --force-reinstall --no-deps)
case "$CUDA_VERSION" in
  12.6*) "${PIP_INSTALL[@]}" "https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sglang_kernel-${SGL_KERNEL_VERSION}+cu124-cp310-abi3-manylinux2014_${ARCH}.whl" ;;
  12.8*|12.9*) "${PIP_INSTALL[@]}" "https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sglang_kernel-${SGL_KERNEL_VERSION}+cu129-cp310-abi3-manylinux2014_${ARCH}.whl" ;;
  13*) "${PIP_INSTALL[@]}" "sglang-kernel==${SGL_KERNEL_VERSION}" ;;
  *) echo "Unsupported or unknown torch CUDA version: ${CUDA_VERSION}" >&2; exit 1 ;;
esac
case "${CUDA_VERSION%%.*}" in
  12) NIXL_BIN_NAME=nixl-cu12 ;;
  13) NIXL_BIN_NAME=nixl-cu13 ;;
  *) echo "Unsupported or unknown torch CUDA version for NIXL: ${CUDA_VERSION}" >&2; exit 1 ;;
esac
"${PIP_INSTALL[@]}" "nixl==${NIXL_VERSION}" "${NIXL_BIN_NAME}==${NIXL_VERSION}"

python3 - <<'"'"'PY'"'"'
import importlib
import json
import os
from importlib.metadata import version
from packaging.version import Version

required_modules = [
    "torch",
    "sglang",
    "pytest",
    "requests",
    "transformers",
    "datasets",
    "nixl._api",
]

for module in required_modules:
    importlib.import_module(module)
    print(f"import ok: {module}")

required_packages = {
    "sglang-kernel": "0.4.4",
}

for package, minimum in required_packages.items():
    installed = version(package)
    print(f"version ok: {package}=={installed}")
    if Version(installed) < Version(minimum):
        raise SystemExit(f"{package}=={installed} is less than required {minimum}")

from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t

backend = os.getenv("SGLANG_DISAGGREGATION_NIXL_BACKEND", "UCX")
params = json.loads(os.getenv("SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS", "{}"))

agent = nixl_agent(
    "sglang_nixl_dependency_probe",
    nixl_agent_config(
        backends=[],
        num_threads=8,
        sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
    ),
)
plugins = agent.get_plugin_list()
print("available NIXL plugins:", plugins)

if backend not in plugins:
    raise SystemExit(f"configured NIXL backend {backend!r} not in available plugins {plugins}")

agent.create_backend(backend, params)
print(f"NIXL backend probe ok: {backend} {params}")
PY

python3 -m pytest -s -v test/registered/disaggregation/test_disaggregation_nixl.py 2>&1 | tee /logs/test_disaggregation_nixl.$SLURM_JOB_ID.log
'
```

Notes:

- Keep the backslash after `--container-mounts=...` as the final character on
  that line. A trailing space after `\` breaks shell continuation.
- The test downloads GSM8K data at runtime if it is not cached.
- The test uses `meta-llama/Llama-3.2-1B-Instruct` for the basic/logprob cases
  and `meta-llama/Llama-3.1-8B-Instruct` for accuracy/failure cases.
- The log is written to `/logs/test_disaggregation_nixl.$SLURM_JOB_ID.log`,
  which maps to `$MY/logs` on the host.

## Run Through The CI Suite

The test registers itself with:

```python
register_cuda_ci(est_time=700, stage="base-c", runner_config="8-gpu-h20")
```

That maps to the per-commit CUDA suite:

```bash
base-c-test-8-gpu-h20
```

To run the same suite selection inside the `.sqsh` container:

```bash
srun -A network_research_advdev \
     -p interactive \
     -t 2:00:00 \
     --gpus-per-node=8 \
     --cpus-per-task=32 \
     --mem=0 \
     --container-image=$MY/sqshs/sglang-nixl-functest.sqsh \
     --container-workdir=/workspace/sglang \
     --container-mounts=$MY/.cache/huggingface:/root/.cache/huggingface,$MY/logs:/logs \
     --export=ALL,HF_TOKEN,MY,SGLANG_DISAGGREGATION_NIXL_BACKEND,SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS \
     bash -lc '
set -euxo pipefail

git config --global --add safe.directory /workspace/sglang
git remote -v
if git remote get-url origin | grep -q "github.com/nbarzilie/sglang"; then
  TEST_REMOTE=origin
else
  git remote remove nbarzilie 2>/dev/null || true
  git remote add nbarzilie https://github.com/nbarzilie/sglang.git
  TEST_REMOTE=nbarzilie
fi
git fetch --no-tags "$TEST_REMOTE" nixl-func-tests-public-c
git checkout -B nixl-func-tests-public-c FETCH_HEAD

SGL_KERNEL_VERSION=0.4.4
NIXL_VERSION=1.3.0
CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda or \"\")")
ARCH=$(uname -m)
PIP_INSTALL=(python3 -m pip install --break-system-packages --force-reinstall --no-deps)
case "$CUDA_VERSION" in
  12.6*) "${PIP_INSTALL[@]}" "https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sglang_kernel-${SGL_KERNEL_VERSION}+cu124-cp310-abi3-manylinux2014_${ARCH}.whl" ;;
  12.8*|12.9*) "${PIP_INSTALL[@]}" "https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sglang_kernel-${SGL_KERNEL_VERSION}+cu129-cp310-abi3-manylinux2014_${ARCH}.whl" ;;
  13*) "${PIP_INSTALL[@]}" "sglang-kernel==${SGL_KERNEL_VERSION}" ;;
  *) echo "Unsupported or unknown torch CUDA version: ${CUDA_VERSION}" >&2; exit 1 ;;
esac
case "${CUDA_VERSION%%.*}" in
  12) NIXL_BIN_NAME=nixl-cu12 ;;
  13) NIXL_BIN_NAME=nixl-cu13 ;;
  *) echo "Unsupported or unknown torch CUDA version for NIXL: ${CUDA_VERSION}" >&2; exit 1 ;;
esac
"${PIP_INSTALL[@]}" "nixl==${NIXL_VERSION}" "${NIXL_BIN_NAME}==${NIXL_VERSION}"

cd test
SGLANG_IS_IN_CI=1 python3 run_suite.py \
  --hw cuda \
  --suite base-c-test-8-gpu-h20 \
  --timeout-per-file 7200 \
  2>&1 | tee /logs/base-c-test-8-gpu-h20.$SLURM_JOB_ID.log
'
```

This runs every registered test in `base-c-test-8-gpu-h20`, not only
`test_disaggregation_nixl.py`. Use the direct `pytest` command above when you
want only the new test file.

## Quick Triage

If the test is skipped or fails before server launch, check:

```bash
python3 - <<'PY'
import json
import os
from nixl._api import nixl_agent, nixl_agent_config, nixl_thread_sync_t

backend = os.getenv("SGLANG_DISAGGREGATION_NIXL_BACKEND", "UCX")
params = json.loads(os.getenv("SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS", "{}"))
agent = nixl_agent(
    "probe",
    nixl_agent_config(
        backends=[],
        num_threads=8,
        sync_mode=nixl_thread_sync_t.NIXL_THREAD_SYNC_STRICT,
    ),
)
print("plugins:", agent.get_plugin_list())
print("backend:", backend)
print("params:", params)
agent.create_backend(backend, params)
print("ok")
PY
```

If server launch fails, confirm:

```bash
nvidia-smi
python3 - <<'PY'
import torch
print(torch.cuda.device_count())
PY
```

Expected GPU count is 8 for the test as currently written.