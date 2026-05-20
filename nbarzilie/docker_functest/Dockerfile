ARG BASE_IMAGE=lmsysorg/sglang:dev-cu13
FROM ${BASE_IMAGE}

ARG SGLANG_REPO_URL=https://github.com/nbarzilie/sglang.git
ARG SGLANG_BRANCH=feature/nixl-testing-suite

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV SGLANG_SOURCE_DIR=/workspace/sglang
ENV PYTHONPATH=/workspace/sglang/python:${PYTHONPATH}
ENV SGLANG_REPO_URL=${SGLANG_REPO_URL}
ENV SGLANG_BRANCH=${SGLANG_BRANCH}

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    iproute2 \
    netcat-openbsd \
    procps \
    ucx-utils \
    libucx0 \
    libibverbs1 \
    libibverbs-dev \
    ibverbs-providers \
    rdma-core \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install --upgrade nixl pytest requests sglang-router

WORKDIR /workspace
RUN rm -rf "${SGLANG_SOURCE_DIR}" \
    && git clone --depth 1 --branch "${SGLANG_BRANCH}" "${SGLANG_REPO_URL}" "${SGLANG_SOURCE_DIR}" \
    && git -C "${SGLANG_SOURCE_DIR}" rev-parse --short HEAD > /workspace/sglang_git_commit.txt \
    && git -C "${SGLANG_SOURCE_DIR}" remote -v > /workspace/sglang_git_remote.txt

COPY nbarzilie/docker_functest/scripts/*.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/*.sh

WORKDIR /workspace/sglang

RUN python3 - <<'PY'
import importlib
import os
import subprocess
import torch

import sglang  # noqa: F401
import nixl._api  # noqa: F401

for module in ["sglang_router.launch_router", "requests", "pytest"]:
    importlib.import_module(module)

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("sglang_source", os.environ["SGLANG_SOURCE_DIR"])
print("sglang_branch", os.environ["SGLANG_BRANCH"])
print("sglang_commit", subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip())
print("sglang, sglang_router, nixl imports OK")
PY

CMD ["/bin/bash"]
