# Fresh SGLang + NIXL Hopper Container Guide

This guide builds a clean SGLang image with NIXL for NVIDIA Hopper GPUs such as H100 and H20.

The Dockerfile is self-contained: copy only the Dockerfile to the target setup, build it there, and it will clone SGLang during the image build.

## Files

Use this file name on the target setup:

```bash
nixl.Dockerfile
```

## Dockerfile

```dockerfile
# syntax=docker/dockerfile:1.7
#
# Clean SGLang + NIXL image for NVIDIA Hopper GPUs such as H100 and H20.
#
# Build:
#   DOCKER_BUILDKIT=1 docker build -f nixl.Dockerfile -t sglang:nixl-hopper .
#
# Build a specific SGLang branch, tag, or commit:
#   DOCKER_BUILDKIT=1 docker build -f nixl.Dockerfile -t sglang:nixl-hopper \
#     --build-arg SGLANG_REF=main .
#
# Run:
#   docker run --gpus all --shm-size 32g --ipc=host --network=host --privileged \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     -v /tmp/sglang-hicache:/data/hicache \
#     -e HF_TOKEN=${HF_TOKEN} \
#     sglang:nixl-hopper \
#     python3 -m sglang.launch_server \
#       --model-path meta-llama/Llama-3.1-8B-Instruct \
#       --host 0.0.0.0 \
#       --port 30000 \
#       --enable-hierarchical-cache \
#       --hicache-storage-backend nixl

ARG CUDA_VERSION=13.0.1
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04

ARG CUDA_VERSION
ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git
ARG SGLANG_REF=main
ARG SGLANG_EXTRAS=
ARG PIP_DEFAULT_INDEX

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=9.0 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SGLANG_HICACHE_NIXL_BACKEND_PLUGIN=auto \
    SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR=/data/hicache \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV PATH="/root/.cargo/bin:${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/cuda/nvvm/bin" \
    LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"

RUN --mount=type=cache,target=/var/cache/apt,id=sglang-nixl-apt \
    apt-get update && apt-get install -y --no-install-recommends --allow-change-held-packages \
        ca-certificates \
        curl \
        wget \
        git \
        build-essential \
        cmake \
        pkg-config \
        ninja-build \
        protobuf-compiler \
        protobuf-compiler-grpc \
        python3.12-full \
        python3.12-dev \
        locales \
        libaio1t64 \
        libcurl4 \
        libfabric1 \
        libibverbs1 \
        libibverbs-dev \
        libibumad3 \
        libnccl2 \
        libnccl-dev \
        libnl-3-200 \
        libnl-route-3-200 \
        libnuma1 \
        libnuma-dev \
        libopenmpi-dev \
        librdmacm1 \
        libssl-dev \
        liburing2 \
        rdma-core \
        ibverbs-providers \
        infiniband-diags \
        perftest \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 2 \
    && update-alternatives --set python3 /usr/bin/python3.12 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
    && python3 /tmp/get-pip.py --break-system-packages \
    && rm /tmp/get-pip.py \
    && python3 -m pip config set global.break-system-packages true \
    && if [ -n "${PIP_DEFAULT_INDEX}" ]; then python3 -m pip config set global.index-url "${PIP_DEFAULT_INDEX}"; fi \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8

# setuptools-rust builds sglang.srt.grpc._core from rust/sglang-grpc.
RUN curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --no-modify-path --profile minimal \
    && rustc --version \
    && cargo --version

WORKDIR /sgl-workspace/sglang

RUN git clone --depth=1 --branch "${SGLANG_REF}" "${SGLANG_REPO}" /sgl-workspace/sglang \
    || (git clone "${SGLANG_REPO}" /sgl-workspace/sglang \
        && cd /sgl-workspace/sglang \
        && git checkout "${SGLANG_REF}")

# Install SGLang dependencies from a temporary stub package, then keep the
# cloned source tree intact for the final editable install.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cargo/registry \
    set -eux; \
    case "${CUDA_VERSION}" in \
        13.*) CUINDEX=130 ;; \
        *) echo "docker/nixl.Dockerfile currently targets CUDA 13 for clean H100/H20 builds; got CUDA_VERSION=${CUDA_VERSION}" && exit 1 ;; \
    esac; \
    mkdir -p /tmp/sglang_deps; \
    cp -a python /tmp/sglang_deps/python; \
    cp -a rust /tmp/sglang_deps/rust; \
    cp -a proto /tmp/sglang_deps/proto; \
    cd /tmp/sglang_deps/python; \
    rm -rf sglang *.egg-info; \
    mkdir -p sglang; \
    touch sglang/__init__.py README.md LICENSE; \
    echo '__version__ = "0.0.0"' > sglang/version.py; \
    python3 -m pip install --upgrade pip setuptools wheel; \
    if [ -n "${SGLANG_EXTRAS}" ]; then SGLANG_SPEC=".[${SGLANG_EXTRAS}]"; else SGLANG_SPEC="."; fi; \
    python3 -m pip install --extra-index-url "https://download.pytorch.org/whl/cu${CUINDEX}" "${SGLANG_SPEC}"; \
    rm -rf /tmp/sglang_deps

# Install NIXL. The nixl stub owns the import path; pair it with the CUDA-major
# specific binary wheel so the image does not accidentally ship wrong-CUDA libs.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install nixl nixl-cu13 --no-deps \
    && python3 -m pip install cuda-python==13.2.0

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cargo/registry \
    python3 -m pip install --no-deps -e python \
    && python3 -c "import sglang, torch, nixl._api; print('SGLang/NIXL OK, torch CUDA:', torch.version.cuda)" \
    && mkdir -p "${SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR}"

EXPOSE 30000
WORKDIR /sgl-workspace/sglang
CMD ["/bin/bash"]
```

## Build Docker Image

Build the default image from SGLang `main`:

```bash
DOCKER_BUILDKIT=1 docker build -f nixl.Dockerfile -t sglang:nixl-hopper .
```

Build a specific SGLang tag, branch, or commit:

```bash
DOCKER_BUILDKIT=1 docker build -f nixl.Dockerfile -t sglang:nixl-hopper \
  --build-arg SGLANG_REF=v0.5.10.post1 .
```

Build from a fork:

```bash
DOCKER_BUILDKIT=1 docker build -f nixl.Dockerfile -t sglang:nixl-hopper \
  --build-arg SGLANG_REPO=https://github.com/<user>/sglang.git \
  --build-arg SGLANG_REF=<branch-or-commit> .
```

Optional build args:

```bash
--build-arg CUDA_VERSION=13.0.1
--build-arg SGLANG_EXTRAS=all
--build-arg PIP_DEFAULT_INDEX=https://<internal-pypi>/simple
```

## Run With Docker

Basic server:

```bash
docker run --gpus all \
  --shm-size 32g \
  --ipc=host \
  --network=host \
  --privileged \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v /tmp/sglang-hicache:/data/hicache \
  -e HF_TOKEN="${HF_TOKEN}" \
  sglang:nixl-hopper \
  python3 -m sglang.launch_server \
    --model-path <model> \
    --host 0.0.0.0 \
    --port 30000 \
    --enable-hierarchical-cache \
    --hicache-storage-backend nixl
```

For POSIX NIXL explicitly:

```bash
-e SGLANG_HICACHE_NIXL_BACKEND_PLUGIN=POSIX
```

For GDS/GDS_MT, the host must have the required NVIDIA driver, storage, and mount configuration. Keep `--privileged`, `--ipc=host`, and `--network=host` unless your cluster has a stricter tested recipe.

## Update SGLang Inside A Running Container

The image keeps SGLang at:

```bash
/sgl-workspace/sglang
```

Enter the container:

```bash
docker run -it --gpus all --shm-size 32g --ipc=host --network=host --privileged \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  sglang:nixl-hopper \
  bash
```

Inside the container, update to a branch or tag:

```bash
cd /sgl-workspace/sglang
git fetch origin
git checkout <branch-or-tag-or-commit>
python3 -m pip install --no-deps -e python
python3 -c "import sglang, torch, nixl._api; print('SGLang/NIXL OK, torch CUDA:', torch.version.cuda)"
```

If the target commit is not available because the build used a shallow clone:

```bash
cd /sgl-workspace/sglang
git fetch --unshallow || true
git fetch origin <branch-or-tag-or-commit>
git checkout <branch-or-tag-or-commit>
python3 -m pip install --no-deps -e python
```

If the new SGLang version changed dependencies, install dependencies first:

```bash
cd /sgl-workspace/sglang/python
python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 .
cd /sgl-workspace/sglang
python3 -m pip install --no-deps -e python
```

For repeatable deployment, prefer rebuilding with `--build-arg SGLANG_REF=<commit>` instead of manually updating inside the container.

## Build A SQSH Image With Enroot

If Docker is available where you build:

```bash
DOCKER_BUILDKIT=1 docker build -f nixl.Dockerfile -t sglang:nixl-hopper .
enroot import -o sglang+nixl-hopper.sqsh dockerd://sglang:nixl-hopper
```

Put the resulting `.sqsh` on a shared filesystem visible to the Slurm compute nodes:

```bash
mkdir -p /shared/containers
cp sglang+nixl-hopper.sqsh /shared/containers/
```

If your cluster does not allow Docker but Pyxis can pull a registry image, push the Docker image to a registry first:

```bash
docker tag sglang:nixl-hopper <registry>/<namespace>/sglang:nixl-hopper
docker push <registry>/<namespace>/sglang:nixl-hopper
```

Then create the `.sqsh` through Slurm/Pyxis:

```bash
srun --ntasks=1 \
  --container-image=<registry>/<namespace>/sglang:nixl-hopper \
  --container-save=/shared/containers/sglang+nixl-hopper.sqsh \
  true
```

## Run SQSH With Slurm/Pyxis

Interactive shell:

```bash
srun --partition=<partition> \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --container-image=/shared/containers/sglang+nixl-hopper.sqsh \
  --container-mounts=$HOME/.cache/huggingface:/root/.cache/huggingface,/tmp/sglang-hicache:/data/hicache \
  --container-workdir=/sgl-workspace/sglang \
  --pty bash
```

Launch SGLang:

```bash
srun --partition=<partition> \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=8 \
  --container-image=/shared/containers/sglang+nixl-hopper.sqsh \
  --container-mounts=$HOME/.cache/huggingface:/root/.cache/huggingface,/tmp/sglang-hicache:/data/hicache \
  --container-workdir=/sgl-workspace/sglang \
  bash -lc 'python3 -m sglang.launch_server \
    --model-path <model> \
    --host 0.0.0.0 \
    --port 30000 \
    --tp 8 \
    --enable-hierarchical-cache \
    --hicache-storage-backend nixl'
```

For multi-node serving, add your cluster's normal Slurm and networking flags, such as `--nodes`, `--ntasks-per-node`, RDMA environment variables, and any SGLang distributed launch arguments required by your deployment.

## Update SQSH Using srun Save State

Use this when you already have an `.sqsh` and want to update SGLang inside it, then save a new `.sqsh`.

Do not mount over `/sgl-workspace/sglang` during this job. If you mount over it, the updated source will live on the host mount, not in the saved image.

```bash
srun --partition=<partition> \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --container-image=/shared/containers/sglang+nixl-hopper.sqsh \
  --container-save=/shared/containers/sglang+nixl-hopper-updated.sqsh \
  --container-writable \
  --container-remap-root \
  --container-workdir=/sgl-workspace/sglang \
  bash -lc '
    set -eux
    cd /sgl-workspace/sglang
    git fetch --unshallow || true
    git fetch origin
    git checkout <branch-or-tag-or-commit>
    python3 -m pip install --no-deps -e python
    python3 -c "import sglang, torch, nixl._api; print(\"SGLang/NIXL OK, torch CUDA:\", torch.version.cuda)"
  '
```

If dependencies changed:

```bash
srun --partition=<partition> \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --container-image=/shared/containers/sglang+nixl-hopper.sqsh \
  --container-save=/shared/containers/sglang+nixl-hopper-updated.sqsh \
  --container-writable \
  --container-remap-root \
  --container-workdir=/sgl-workspace/sglang \
  bash -lc '
    set -eux
    cd /sgl-workspace/sglang
    git fetch --unshallow || true
    git fetch origin
    git checkout <branch-or-tag-or-commit>
    cd python
    python3 -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 .
    cd ..
    python3 -m pip install --no-deps -e python
    python3 -c "import sglang, torch, nixl._api; print(\"SGLang/NIXL OK, torch CUDA:\", torch.version.cuda)"
  '
```

Then run future jobs from the updated image:

```bash
srun --partition=<partition> \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=8 \
  --container-image=/shared/containers/sglang+nixl-hopper-updated.sqsh \
  --container-mounts=$HOME/.cache/huggingface:/root/.cache/huggingface,/tmp/sglang-hicache:/data/hicache \
  --container-workdir=/sgl-workspace/sglang \
  bash -lc 'python3 -m sglang.launch_server --model-path <model> --host 0.0.0.0 --port 30000 --tp 8 --enable-hierarchical-cache --hicache-storage-backend nixl'
```

## Quick Validation

Inside Docker, Enroot, or Pyxis:

```bash
python3 -c "import sglang, torch, nixl._api; print('torch CUDA:', torch.version.cuda); print('GPU count:', torch.cuda.device_count())"
```

Check NIXL backend selection while launching:

```bash
export SGLANG_HICACHE_NIXL_BACKEND_PLUGIN=auto
export SGLANG_HICACHE_NIXL_BACKEND_STORAGE_DIR=/data/hicache
```

For a simple file-backed setup, start with `POSIX`:

```bash
export SGLANG_HICACHE_NIXL_BACKEND_PLUGIN=POSIX
```

## Notes

- H100 and H20 are Hopper-class GPUs, so this guide uses `TORCH_CUDA_ARCH_LIST=9.0`.
- The image defaults to CUDA 13.0.1 and installs `nixl` with `nixl-cu13`.
- Use `--ipc=host` and sufficient shared memory for SGLang.
- Use `--network=host` and privileged/RDMA-capable runs when testing NIXL with RDMA or GDS paths.
- For production, pin `SGLANG_REF` to a commit hash instead of `main`.
- Pyxis supports `.sqsh` files through `--container-image` and can export a job filesystem with `--container-save`.
- Enroot can import from registries and local Docker daemon images into SquashFS `.sqsh` images.

## References

- Pyxis usage: https://github-wiki-see.page/m/NVIDIA/pyxis/wiki/Usage
- Enroot Docker image import: https://deepwiki.com/NVIDIA/enroot/4.1-docker-image-import
