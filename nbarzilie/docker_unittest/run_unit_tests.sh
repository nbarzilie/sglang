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
