ARG BASE_IMAGE=lmsysorg/sglang:dev-cu13
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV SGLANG_SOURCE_DIR=/workspace/sglang
ENV PYTHONPATH=/workspace/sglang/python:${PYTHONPATH}

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

WORKDIR /workspace/sglang
COPY . /workspace/sglang
COPY nbarzilie/docker_functest/scripts/*.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/*.sh

RUN python3 - <<'PY'
import importlib
import torch

import sglang  # noqa: F401
import nixl._api  # noqa: F401

for module in ["sglang_router.launch_router", "requests", "pytest"]:
    importlib.import_module(module)

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("sglang, sglang_router, nixl imports OK")
PY

CMD ["/bin/bash"]
