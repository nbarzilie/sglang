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
