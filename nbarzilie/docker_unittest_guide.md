# Docker Unit Test Guide

This guide creates an Ubuntu container that clones your SGLang fork branch and runs the NIXL/disaggregation CPU unit tests from `nbarzilie/unit_test.md`.

Branch source:

```text
https://github.com/nbarzilie/sglang.git
feature/nixl-testing-suite
```

The tests are CPU-only. They do not require a GPU, NIXL runtime, model weights, RDMA, or a running SGLang server.

## Files to Create

Create a clean local folder, for example:

```bash
mkdir -p /tmp/sglang-docker-unittest
cd /tmp/sglang-docker-unittest
```

Create this `Dockerfile`:

```dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    build-essential \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
RUN python -m pip install --upgrade pip setuptools wheel

COPY run_unit_tests.sh /usr/local/bin/run_unit_tests.sh
RUN chmod +x /usr/local/bin/run_unit_tests.sh

WORKDIR /work

CMD ["/usr/local/bin/run_unit_tests.sh"]
```

Create this `run_unit_tests.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/nbarzilie/sglang.git}"
BRANCH="${BRANCH:-feature/nixl-testing-suite}"
SRC_DIR="${SRC_DIR:-/work/sglang}"

echo "==> Clone ${REPO_URL} branch ${BRANCH}"
rm -rf "${SRC_DIR}"
git clone --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${SRC_DIR}"
cd "${SRC_DIR}"

echo "==> Use CPU dependency file"
cp python/pyproject_cpu.toml python/pyproject.toml

echo "==> Install CPU PyTorch wheels"
python -m pip install \
  torch==2.9.0 \
  torchvision==0.24.0 \
  torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cpu

echo "==> Install SGLang CPU test dependencies"
python -m pip install -e "python[test]"

export PYTHONPATH="${SRC_DIR}/python:${PYTHONPATH:-}"

echo "==> Python syntax check"
python -m py_compile \
  python/sglang/srt/disaggregation/nixl/conn.py \
  test/registered/unit/disaggregation/test_nixl_transfer_info.py \
  test/registered/unit/disaggregation/test_nixl_transfer_status.py \
  test/registered/unit/disaggregation/test_nixl_notifications.py \
  test/registered/unit/disaggregation/test_nixl_backend_config.py \
  test/registered/unit/disaggregation/test_nixl_descriptor_building.py \
  test/registered/unit/disaggregation/test_nixl_receiver_poll.py \
  test/registered/unit/disaggregation/test_nixl_node_failure.py \
  test/registered/unit/disaggregation/test_nixl_staging.py \
  test/registered/unit/disaggregation/test_nixl_hybrid_state.py \
  test/registered/unit/disaggregation/test_disaggregation_rank_mapping.py

echo "==> Run disaggregation unit tests"
pytest test/registered/unit/disaggregation/ -vv -s
```

Make the script executable:

```bash
chmod +x run_unit_tests.sh
```

## Build the Docker Image

From the folder containing `Dockerfile` and `run_unit_tests.sh`:

```bash
docker build -t sglang-nixl-unit-tests .
```

First build/run can take a long time because it downloads Python test dependencies and CPU PyTorch wheels.

## Run the Tests

```bash
docker run --rm -it sglang-nixl-unit-tests
```

You will see all clone, install, syntax-check, and pytest output directly in the terminal. The container exits with:

- `0` if all tests pass.
- non-zero if install, syntax check, or pytest fails.

To save the full output to a file on your host:

```bash
docker run --rm sglang-nixl-unit-tests 2>&1 | tee unittest_output.log
```

## Run a Different Branch

```bash
docker run --rm -it \
  -e BRANCH=feature/nixl-testing-suite \
  sglang-nixl-unit-tests
```

To test another fork:

```bash
docker run --rm -it \
  -e REPO_URL=https://github.com/nbarzilie/sglang.git \
  -e BRANCH=feature/nixl-testing-suite \
  sglang-nixl-unit-tests
```

## Run One Test File Manually

Start a shell in the container:

```bash
docker run --rm -it --entrypoint bash sglang-nixl-unit-tests
```

Then run:

```bash
/usr/local/bin/run_unit_tests.sh
```

After the script finishes, you can run individual tests inside `/work/sglang`:

```bash
cd /work/sglang
pytest test/registered/unit/disaggregation/test_nixl_transfer_info.py -vv -s
pytest test/registered/unit/disaggregation/test_nixl_backend_config.py -vv -s
```

## Run the CI Suite From Terminal

Use this when you want SGLang's CI runner behavior instead of direct pytest. The CI runner discovers registered tests, prints the selected files, applies per-file timeout handling, and exits non-zero on failure.

### 1. Build the Docker Image

From the folder containing `Dockerfile` and `run_unit_tests.sh`:

```bash
docker build -t sglang-nixl-unit-tests .
```

If you created the files inside this repo, the folder is:

```bash
cd nbarzilie/docker_unittest
docker build -t sglang-nixl-unit-tests .
```

### 2. Start an Interactive Container With a Log Folder

From the same folder:

```bash
mkdir -p ci_logs
docker run --rm -it \
  -v "$PWD/ci_logs:/logs" \
  --entrypoint bash \
  sglang-nixl-unit-tests
```

The `ci_logs` folder is mounted into the container as `/logs`, so `tee /logs/file.log` saves output on your host.

### 3. Clone the Branch Inside the Container

Inside the container:

```bash
export REPO_URL=https://github.com/nbarzilie/sglang.git
export BRANCH=feature/nixl-testing-suite
export SRC_DIR=/work/sglang

rm -rf "$SRC_DIR"
git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$SRC_DIR"
cd "$SRC_DIR"
```

### 4. Install CPU Test Dependencies

Inside the container:

```bash
cp python/pyproject_cpu.toml python/pyproject.toml

python -m pip install \
  torch==2.9.0 \
  torchvision==0.24.0 \
  torchaudio==2.9.0 \
  --index-url https://download.pytorch.org/whl/cpu

python -m pip install -e "python[test]"
export PYTHONPATH="$SRC_DIR/python:${PYTHONPATH:-}"
```

### 5. Show CI Runner Help

Inside the container:

```bash
python3 test/run_suite.py --help
```

Important options:

- `--hw cpu`: run CPU-registered tests.
- `--suite base-a-test-cpu`: run the Base A CPU suite.
- `--continue-on-error`: run all selected files even if one fails.
- `--timeout-per-file 1200`: per-file timeout in seconds. Default is `1200`.
- `--auto-partition-id N --auto-partition-size M`: run one partition of a suite.

### 6. Run the Base A CPU CI Suite

Inside the container:

```bash
python3 test/run_suite.py \
  --hw cpu \
  --suite base-a-test-cpu \
  2>&1 | tee /logs/base-a-test-cpu.log
```

Read the result:

- Exit code `0`: CI suite passed.
- Non-zero exit code: at least one CI file failed or timed out.
- Full output is saved on host at `ci_logs/base-a-test-cpu.log`.

### 7. Run the Suite But Continue After Failures

Use this when you want the full list of failing files in one run:

```bash
python3 test/run_suite.py \
  --hw cpu \
  --suite base-a-test-cpu \
  --continue-on-error \
  2>&1 | tee /logs/base-a-test-cpu-all-failures.log
```

### 8. Run With a Longer Per-File Timeout

Use this if a file times out in a slow Docker environment:

```bash
python3 test/run_suite.py \
  --hw cpu \
  --suite base-a-test-cpu \
  --timeout-per-file 2400 \
  2>&1 | tee /logs/base-a-test-cpu-timeout-2400.log
```

### 9. Run CI Suite Partitions

This mimics CI parallel partitioning. Open two terminals or run them one after another.

Partition 0 of 2:

```bash
python3 test/run_suite.py \
  --hw cpu \
  --suite base-a-test-cpu \
  --auto-partition-id 0 \
  --auto-partition-size 2 \
  2>&1 | tee /logs/base-a-test-cpu-part-0.log
```

Partition 1 of 2:

```bash
python3 test/run_suite.py \
  --hw cpu \
  --suite base-a-test-cpu \
  --auto-partition-id 1 \
  --auto-partition-size 2 \
  2>&1 | tee /logs/base-a-test-cpu-part-1.log
```

All partitions must pass for the suite to pass.

### 10. Run the NIXL Unit Tests Directly After CI Setup

If CI suite output points to a failing disaggregation test, rerun it directly:

```bash
pytest test/registered/unit/disaggregation/test_nixl_transfer_info.py -vv -s
```

Or run the full disaggregation unit-test folder:

```bash
pytest test/registered/unit/disaggregation/ -vv -s
```

Use `test/run_suite.py` when you want CI-style discovery and suite behavior. Use direct `pytest` when you want fast debugging of one file or one folder.

## Expected Test Scope

The default pytest command runs all files under:

```text
test/registered/unit/disaggregation/
```

This includes the NIXL unit-test areas from `nbarzilie/unit_test.md`:

- transfer metadata parsing
- transfer completion status
- notification parsing
- backend config validation
- descriptor construction
- receiver polling
- node failure handling
- staging-buffer control paths
- hybrid state transfer
- disaggregation rank mapping

## Troubleshooting

If clone fails, verify the branch exists and is public:

```bash
git ls-remote --heads https://github.com/nbarzilie/sglang.git feature/nixl-testing-suite
```

If dependency installation fails, rerun with plain output and keep the first package error:

```bash
docker run --rm sglang-nixl-unit-tests
```

If pytest fails, the terminal output shows the failing file, test name, assertion, and traceback. Re-run only that file with:

```bash
pytest path/to/test_file.py -vv -s
```
