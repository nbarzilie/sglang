# Docker Functional Test Guide

This folder builds an Ubuntu-based SGLang functional-test image and provides scripts for:

1. Regular single-server Qwen3 serving with NIXL installed and import-checked.
2. PD disaggregation with one H100 for prefill and one H100 for decode, using NIXL.
3. Direct execution of `test/registered/disaggregation/test_disaggregation_nixl_basic.py`.
4. A local CPU SGLang health check before moving to the GPU cluster.

NIXL is only used by SGLang as a transfer backend in PD disaggregation mode. The regular Qwen3 script verifies the same container has NIXL installed, but a non-PD SGLang server does not perform NIXL KV transfer.

## Files

```text
nbarzilie/docker_functest/
  Dockerfile
  docker_functest_guide.md
  scripts/
    sglang_functest_common.sh
    check_sglang_cpu_health.sh
    run_qwen3_regular_nixl.sh
    run_qwen3_pd_nixl.sh
    run_disaggregation_nixl_basic_test.sh
```

## 1. Build The Image

Build from a small directory that contains only the Dockerfile and the `scripts/` folder. You do not need the full SGLang repository as the Docker build context because the Dockerfile clones your public branch during image build.

Expected build context:

```text
docker_functest_build/
  Dockerfile
  scripts/
    sglang_functest_common.sh
    check_sglang_cpu_health.sh
    run_qwen3_regular_nixl.sh
    run_qwen3_pd_nixl.sh
    run_disaggregation_nixl_basic_test.sh
```

If you are starting from this checkout, create that small build context with:

```bash
mkdir -p /tmp/docker_functest_build
cp nbarzilie/docker_functest/Dockerfile /tmp/docker_functest_build/
cp -R nbarzilie/docker_functest/scripts /tmp/docker_functest_build/
cd /tmp/docker_functest_build
```

The Dockerfile clones your public SGLang fork branch during image build:

```text
repo:   https://github.com/nbarzilie/sglang.git
branch: feature/nixl-testing-suite
path:   /workspace/sglang
```

This is intentional: the compute node or cluster job should not need to fetch the repository. Build the image once in an environment with internet access, convert/push it, and run the baked image on the cluster.

```bash
docker build \
  -f Dockerfile \
  -t sglang-nixl-functest .
```

The default base image is:

```text
lmsysorg/sglang:dev-cu13
```

Override it if needed:

```bash
docker build \
  --build-arg BASE_IMAGE=lmsysorg/sglang:latest-cu130-runtime \
  --build-arg SGLANG_REPO_URL=https://github.com/nbarzilie/sglang.git \
  --build-arg SGLANG_BRANCH=feature/nixl-testing-suite \
  -f Dockerfile \
  -t sglang-nixl-functest .
```

## 2. Common Runtime Options

Use host networking for the functional scripts because they launch several local services and make loopback health requests.

```bash
COMMON_DOCKER_ARGS=(
  --rm -it
  --network host
  --ipc=host
  --shm-size 64g
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface"
)
```

If the model is gated, pass your Hugging Face token:

```bash
-e HF_TOKEN="$HF_TOKEN"
```

Useful environment variables:

```text
MODEL_PATH                                  default: Qwen/Qwen3-8B
SGLANG_DISAGGREGATION_NIXL_BACKEND          default: UCX
SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS   default: {}
DISAGG_IB_DEVICES                           optional, example: mlx5_0,mlx5_1
LOG_DIR                                     default: /tmp/sglang-functest-logs
KEEP_ALIVE                                  default: 1 for GPU scripts, 0 for CPU health
```

For a quick functional pass, keep the default `Qwen/Qwen3-8B`. For a larger 20-40B Qwen3 model on H100s, set `MODEL_PATH`, for example:

```bash
-e MODEL_PATH=Qwen/Qwen3-32B
```

For regular single-server testing of a 20-40B model on two H100s, also set `TP=2` and expose both GPUs. For PD testing, remember that prefill and decode each load the model on one GPU in this script, so `Qwen/Qwen3-8B` is the safer default; `Qwen/Qwen3-32B` requires enough per-GPU memory headroom for weights plus KV cache.

## 3. CPU Health Check

Run this locally or on a CPU-only node to verify the container can import SGLang, import NIXL, start a CPU SGLang server, pass `/health`, and answer one tiny `/generate` request with dummy weights.

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  -e CPU_MODEL_PATH=Qwen/Qwen3-8B \
  sglang-nixl-functest \
  check_sglang_cpu_health.sh
```

If you want to keep the CPU server running:

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  -e KEEP_ALIVE=1 \
  sglang-nixl-functest \
  check_sglang_cpu_health.sh
```

The CPU script uses:

```text
SGLANG_USE_CPU_ENGINE=1
--device cpu
--load-format dummy
```

## 4. Build, Embed Scripts, And Create SQSH

The Dockerfile clones the public branch into:

```text
/workspace/sglang
```

and installs every functional script into:

```text
/usr/local/bin/check_sglang_cpu_health.sh
/usr/local/bin/run_qwen3_regular_nixl.sh
/usr/local/bin/run_qwen3_pd_nixl.sh
/usr/local/bin/run_disaggregation_nixl_basic_test.sh
```

That means the cluster job does not need to mount or import scripts separately. You only pass the script name as the container command plus environment variables for model, ports, backend, and logs.

Build the Docker image with the branch and scripts embedded:

```bash
mkdir -p /tmp/docker_functest_build
cp nbarzilie/docker_functest/Dockerfile /tmp/docker_functest_build/
cp -R nbarzilie/docker_functest/scripts /tmp/docker_functest_build/
cd /tmp/docker_functest_build

docker build \
  --build-arg SGLANG_REPO_URL=https://github.com/nbarzilie/sglang.git \
  --build-arg SGLANG_BRANCH=feature/nixl-testing-suite \
  -f Dockerfile \
  -t sglang-nixl-functest:latest .
```

Quickly verify the branch and scripts are inside the image:

```bash
docker run --rm sglang-nixl-functest:latest \
  bash -lc 'git -C /workspace/sglang remote -v && git -C /workspace/sglang rev-parse --abbrev-ref HEAD && git -C /workspace/sglang rev-parse --short HEAD && ls -l /usr/local/bin/*qwen3* /usr/local/bin/*nixl* /usr/local/bin/check_sglang_cpu_health.sh'
```

At this point no later cluster command should run `git clone` or mount your local repository. The SQSH contains `/workspace/sglang` from the selected public branch.

If you change any script under `nbarzilie/docker_functest/scripts/` or want a newer branch commit, rebuild the Docker image and recreate the `.sqsh`; the cluster will only see what was baked into that image.

Create a SquashFS image for Enroot/Pyxis-style cluster runs:

```bash
enroot import -o sglang-nixl-functest.sqsh dockerd://sglang-nixl-functest:latest
```

If your cluster cannot read from the local Docker daemon, push to a registry first:

```bash
docker tag sglang-nixl-functest:latest <registry>/sglang-nixl-functest:latest
docker push <registry>/sglang-nixl-functest:latest
enroot import -o sglang-nixl-functest.sqsh docker://<registry>/sglang-nixl-functest:latest
```

Copy the `.sqsh` to shared storage visible from the compute node:

```bash
cp sglang-nixl-functest.sqsh /shared/containers/
```

Example Slurm/Pyxis run for PD NIXL:

```bash
srun \
  --container-image=/shared/containers/sglang-nixl-functest.sqsh \
  --container-workdir=/workspace/sglang \
  --container-mounts=$HOME/.cache/huggingface:/root/.cache/huggingface,/shared/logs:/logs \
  --gres=gpu:2 \
  --ntasks=1 \
  --cpus-per-task=32 \
  --mem=0 \
  bash -lc 'LOG_DIR=/logs MODEL_PATH=Qwen/Qwen3-8B run_qwen3_pd_nixl.sh'
```

Example Slurm/Pyxis run for regular 2-GPU TP with a 32B model:

```bash
srun \
  --container-image=/shared/containers/sglang-nixl-functest.sqsh \
  --container-workdir=/workspace/sglang \
  --container-mounts=$HOME/.cache/huggingface:/root/.cache/huggingface,/shared/logs:/logs \
  --gres=gpu:2 \
  --ntasks=1 \
  --cpus-per-task=32 \
  --mem=0 \
  bash -lc 'LOG_DIR=/logs MODEL_PATH=Qwen/Qwen3-32B TP=2 run_qwen3_regular_nixl.sh'
```

If your cluster uses a different container runtime, keep the same idea: use `sglang-nixl-functest.sqsh` as the image, set the environment variables, and run one of the `/usr/local/bin/*.sh` commands already baked into the image.

## 5. Regular Qwen3 Single-Server Run

Run on one H100:

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  --gpus '"device=0"' \
  -e MODEL_PATH=Qwen/Qwen3-8B \
  -e SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX \
  sglang-nixl-functest \
  run_qwen3_regular_nixl.sh
```

The script:

- Verifies `torch`, `sglang`, `sglang_router`, and `nixl._api` imports.
- Starts `python3 -m sglang.launch_server`.
- Waits for `http://127.0.0.1:30000/health`.
- Sends one `/generate` request.
- Keeps the server running until Ctrl-C unless `KEEP_ALIVE=0`.

Change the port or TP:

```bash
-e REGULAR_PORT=30010
-e TP=1
```

Pass extra SGLang args:

```bash
-e REGULAR_EXTRA_ARGS="--context-length 4096 --mem-fraction-static 0.75"
```

For a regular 32B run over two H100s:

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  --gpus '"device=0,1"' \
  -e MODEL_PATH=Qwen/Qwen3-32B \
  -e TP=2 \
  -e SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX \
  sglang-nixl-functest \
  run_qwen3_regular_nixl.sh
```

## 6. PD Qwen3 With NIXL

Run on one node with two H100s visible as GPUs 0 and 1:

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  --gpus '"device=0,1"' \
  -e MODEL_PATH=Qwen/Qwen3-8B \
  -e SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX \
  sglang-nixl-functest \
  run_qwen3_pd_nixl.sh
```

The script launches:

```text
prefill: GPU 0, http://127.0.0.1:30100
decode:  GPU 1, http://127.0.0.1:30200
router:  http://127.0.0.1:30000
bootstrap port: 30500
```

It then waits for health on prefill, decode, and router, sends one `/generate` request through the router, and keeps all services alive until Ctrl-C unless `KEEP_ALIVE=0`.

If your NIXL/UCX setup requires explicit RDMA devices:

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  --gpus '"device=0,1"' \
  -e MODEL_PATH=Qwen/Qwen3-8B \
  -e DISAGG_IB_DEVICES=mlx5_0,mlx5_1 \
  sglang-nixl-functest \
  run_qwen3_pd_nixl.sh
```

For same-node testing where you want to force UCX away from IB, you can try:

```bash
-e UCX_TLS=sm,self,tcp,cuda_copy,cuda_ipc
```

Port overrides:

```bash
-e PREFILL_PORT=31100
-e DECODE_PORT=31200
-e ROUTER_PORT=31000
-e BOOTSTRAP_PORT=31500
```

Extra args:

```bash
-e PREFILL_EXTRA_ARGS="--context-length 4096"
-e DECODE_EXTRA_ARGS="--context-length 4096"
```

## 7. Run The Registered NIXL Test

Run:

```bash
docker run "${COMMON_DOCKER_ARGS[@]}" \
  --gpus '"device=0,1"' \
  -e SGLANG_DISAGGREGATION_NIXL_BACKEND=UCX \
  sglang-nixl-functest \
  run_disaggregation_nixl_basic_test.sh
```

This directly executes:

```bash
python3 test/registered/disaggregation/test_disaggregation_nixl_basic.py
```

Important: that test currently uses `DEFAULT_SMALL_MODEL_NAME_FOR_TEST`, which is `meta-llama/Llama-3.2-1B-Instruct`. If your environment cannot access that gated model, pass `HF_TOKEN` or change the test/model constant before rebuilding the image.

## 8. Logs

All scripts write service logs under:

```text
/tmp/sglang-functest-logs
```

To persist logs on the host:

```bash
mkdir -p functest_logs
docker run "${COMMON_DOCKER_ARGS[@]}" \
  --gpus '"device=0,1"' \
  -v "$PWD/functest_logs:/logs" \
  -e LOG_DIR=/logs \
  sglang-nixl-functest \
  run_qwen3_pd_nixl.sh
```

Useful files:

```text
regular_qwen3.log
pd_prefill.log
pd_decode.log
pd_router.log
cpu_health_server.log
```

## 9. Troubleshooting

Check imports inside the image:

```bash
docker run --rm -it sglang-nixl-functest python3 - <<'PY'
import torch
import sglang
import sglang_router.launch_router
import nixl._api
print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
print("imports ok")
PY
```

Check GPUs:

```bash
docker run --rm -it --gpus all sglang-nixl-functest nvidia-smi
```

If PD hangs waiting for transfer:

- Confirm `nixl._api` imports in the container.
- Confirm both GPUs are visible with `--gpus '"device=0,1"'`.
- Confirm ports are not already in use.
- Inspect `pd_prefill.log`, `pd_decode.log`, and `pd_router.log`.
- Try setting `UCX_TLS=sm,self,tcp,cuda_copy,cuda_ipc` for same-node testing.
- If using IB/RDMA, set `DISAGG_IB_DEVICES` to active devices from `/sys/class/infiniband`.

If the registered test skips:

```text
NIXL is required for this test.
```

then `nixl._api` did not import in that environment. Rebuild the image and check the Dockerfile `pip install nixl` step.
