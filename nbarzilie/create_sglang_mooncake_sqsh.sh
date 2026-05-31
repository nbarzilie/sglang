#!/usr/bin/env bash
set -u

srun \
  -A network_research_advdev \
  -t 02:00:00 \
  -N 1 \
  -p interactive \
  --gpus-per-node=8 \
  --container-image="$MY/sqshs/sglang_fresh.sqsh" \
  --container-save="$MY/sqshs/sglang_mooncake.sqsh" \
  --container-writable \
  --container-remap-root \
  --container-workdir=/sgl-workspace/sglang \
  --container-mounts="$MY/.cache/huggingface:/root/.cache/huggingface,$MY/logs:/logs" \
  --pty bash -lc '
set +e

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
echo "===== install transfer stacks ====="
python3 -m pip install --upgrade nixl nixl-cu13 --no-deps --break-system-packages
python3 -m pip install --upgrade "cuda-python==13.2.0" --break-system-packages
python3 -m pip uninstall -y mooncake-transfer-engine --break-system-packages
python3 -m pip install --upgrade mooncake-transfer-engine-cuda13 --break-system-packages

echo
echo "===== import checks ====="
python3 - <<'"'"'PY'"'"'
import importlib.metadata as md
import importlib.util
import subprocess
import sys

print("git", subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
print("python", sys.version.replace("\n", " "))
try:
    import sglang
    from sglang.version import __version__ as source_version
    print("sglang_source", sglang.__file__)
    print("sglang_source_version", source_version)
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
    "mooncake-transfer-engine-cuda13",
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
PY

echo
echo "===== gpu / ib summary ====="
nvidia-smi -L
python3 - <<'"'"'PY'"'"'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available(), "device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_capability(i))
PY
ls -l /sys/class/infiniband || true

echo
echo "===== environment markers ====="
env | grep -E "^(PYTHONPATH|LD_LIBRARY_PATH|PATH|SGLANG|SGL_|NIXL|UCX|MOONCAKE)=" | sort

echo
echo "Mooncake image filesystem is ready to inspect. Exit this shell when you want Pyxis to save:"
echo "  $MY/sqshs/sglang_mooncake.sqsh"
exec bash -i
'
