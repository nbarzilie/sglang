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
