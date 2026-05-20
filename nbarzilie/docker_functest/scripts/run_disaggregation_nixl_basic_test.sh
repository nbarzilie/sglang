#!/usr/bin/env bash
set -euo pipefail

source /usr/local/bin/sglang_functest_common.sh

export SGLANG_DISAGGREGATION_NIXL_BACKEND="${SGLANG_DISAGGREGATION_NIXL_BACKEND:-UCX}"
export SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS="${SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS:-{}}"
export SGLANG_TEST_PD_DISAGG_BACKEND="${SGLANG_TEST_PD_DISAGG_BACKEND:-nixl}"

check_python_deps

cd "${SGLANG_SOURCE_DIR}"

log "Running test/registered/disaggregation/test_disaggregation_nixl_basic.py"
python3 test/registered/disaggregation/test_disaggregation_nixl_basic.py
