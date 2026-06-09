#!/usr/bin/env bash
set -euo pipefail

echo "===== host ====="
hostname
date -Is
uname -a
cat /etc/os-release 2>/dev/null || true

echo
echo "===== git ====="
git rev-parse HEAD 2>/dev/null || true
git status --short 2>/dev/null || true

echo
echo "===== gpu ====="
nvidia-smi -L || true
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total,pci.bus_id --format=csv || true
nvidia-smi topo -m || true

echo
echo "===== nic / ib ====="
ls -l /sys/class/infiniband 2>/dev/null || true
for d in /sys/class/infiniband/*; do
  [ -e "$d" ] || continue
  echo "--- $(basename "$d") ---"
  cat "$d"/fw_ver 2>/dev/null || true
  for p in "$d"/ports/*; do
    [ -e "$p" ] || continue
    echo "port $(basename "$p")"
    cat "$p/state" 2>/dev/null || true
    cat "$p/phys_state" 2>/dev/null || true
    cat "$p/rate" 2>/dev/null || true
  done
done

echo
echo "===== python packages ====="
python3 - <<'PY'
import importlib.metadata as md
import importlib.util
import platform
import subprocess
import sys

print("python", sys.version.replace("\n", " "))
print("platform", platform.platform())

try:
    import sglang
    from sglang.version import __version__

    print("sglang_source", sglang.__file__)
    print("sglang_source_version", __version__)
except Exception as exc:
    print("sglang_source_import_error", repr(exc))

for pkg in [
    "sglang",
    "sglang-kernel",
    "sglang-router",
    "torch",
    "triton",
    "flashinfer-python",
    "flashinfer-cubin",
    "nixl",
    "nixl-cu13",
    "mooncake-transfer-engine",
    "mooncake-transfer-engine-cuda13",
    "transformers",
    "vllm",
    "uvloop",
    "zmq",
    "pyzmq",
    "numpy",
]:
    try:
        print(pkg, md.version(pkg))
    except md.PackageNotFoundError:
        print(pkg, "NOT_INSTALLED")

for mod in ["sglang", "sgl_kernel", "nixl._api", "mooncake.engine"]:
    try:
        spec = importlib.util.find_spec(mod)
    except Exception as exc:
        spec = f"ERROR: {exc!r}"
    print("module", mod, spec)

try:
    print("git", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
except Exception as exc:
    print("git_error", repr(exc))
PY

echo
echo "===== torch / cuda ====="
python3 - <<'PY'
try:
    import torch

    print("torch.__version__", torch.__version__)
    print("torch.version.cuda", torch.version.cuda)
    print("torch.version.hip", torch.version.hip)
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
except Exception as exc:
    print("torch_probe_failed", repr(exc))
PY

echo
echo "===== sglang cli ====="
command -v sglang || true
sglang version 2>/dev/null || true
python3 -m sglang.bench_serving --help >/dev/null 2>&1 && echo "bench_serving OK" || echo "bench_serving FAILED"

echo
echo "===== sglang env ====="
env | grep -E '^(SGLANG|SGL_|CUDA|NCCL|UCX|NIXL|MOONCAKE|HF_|TRANSFORMERS|TRITON|TORCH|PYTHONPATH|LD_LIBRARY_PATH|PATH|MODEL|BACKEND|CASE_NAME|PREFILL_|DECODE_|ROUTER_|IB_DEV|MAX_CONCURRENCY)=' | sort

echo
echo "===== server command args from env ====="
printf 'BACKEND=%s\n' "${BACKEND:-}"
printf 'MODEL=%s\n' "${MODEL:-}"
printf 'CASE_NAME=%s\n' "${CASE_NAME:-}"
printf 'PREFILL_TP_SIZE=%s PREFILL_DP_SIZE=%s PREFILL_BASE_GPU_ID=%s\n' "${PREFILL_TP_SIZE:-}" "${PREFILL_DP_SIZE:-}" "${PREFILL_BASE_GPU_ID:-}"
printf 'DECODE_TP_SIZE=%s DECODE_DP_SIZE=%s DECODE_BASE_GPU_ID=%s\n' "${DECODE_TP_SIZE:-}" "${DECODE_DP_SIZE:-}" "${DECODE_BASE_GPU_ID:-}"
printf 'IB_DEV=%s MAX_CONCURRENCY=%s\n' "${IB_DEV:-}" "${MAX_CONCURRENCY:-}"