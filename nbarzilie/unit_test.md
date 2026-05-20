# NIXL Unit Test Summary and Run Guide

## Summary

This document summarizes the CPU unit tests added for the SGLang NIXL disaggregation backend. These tests focus on pure protocol, parsing, state, descriptor, and failure logic. They do not require a real NIXL installation, GPU, server launch, model weights, or RDMA transport.

Production helper changes were also added in:

```text
python/sglang/srt/disaggregation/nixl/conn.py
```

New helper functions:

- `load_nixl_agent_classes()`
- `parse_nixl_backend_params()`
- `validate_nixl_backend_available()`

These helpers make NIXL import, backend parameter parsing, and plugin validation independently unit-testable.

## Test Files

### `test/registered/unit/disaggregation/test_nixl_transfer_info.py`

Covers NIXL wire metadata parsing.

Key coverage:

- `TransferInfo.from_zmq` parses room, endpoint, port, agent name, KV indices, aux index, response count, state indices, and `decode_prefix_len`.
- Missing optional state and decode-prefix fields default correctly.
- Decode-radix full hits with empty `dst_kv_indices` and `decode_prefix_len > 0` are not treated as dummy transfers.
- Empty KV indices without decode prefix are dummy transfers.
- `KVArgsRegisterInfo.from_zmq` preserves unsigned 64-bit pointers, including addresses with bit 63 set.
- Optional state metadata and staging metadata parse correctly.

### `test/registered/unit/disaggregation/test_nixl_transfer_status.py`

Covers transfer completion state and sender chunk policy.

Key coverage:

- Normal KV transfer completion requires aux, expected KV count, and received chunks.
- Aux-only zero-KV completion succeeds.
- Multi-PP completion requires every expected PP rank.
- State-required transfers wait for state notifications from all expected PP ranks.
- Failure is terminal and makes the status done.
- `NixlKVSender.should_send_kv_chunk(0, last_chunk=True)` returns true for decode-radix full-hit aux-only completion.

### `test/registered/unit/disaggregation/test_nixl_notifications.py`

Covers NIXL notification tag parsing.

Key coverage:

- KV notifications update received chunks and expected chunk counts.
- Staging notifications preserve agent names containing underscores.
- `aux_nokv` notifications mark expected KV count as zero for the PP rank.
- State notifications mark state arrival per PP rank.
- Aux-only full-hit notification can make a transfer complete.

### `test/registered/unit/disaggregation/test_nixl_backend_config.py`

Covers NIXL configuration validation.

Key coverage:

- Backend params accept JSON objects with string keys and values.
- Backend params reject non-object JSON, invalid JSON, and non-string values.
- Missing NIXL import raises actionable install guidance.
- Missing requested backend plugin reports requested backend and available plugins.
- Existing requested backend returns the available plugin list.

### `test/registered/unit/disaggregation/test_nixl_descriptor_building.py`

Covers NIXL memory registration and descriptor construction with fake agents.

Key coverage:

- `register_buffer_to_engine` registers KV and state as `VRAM`, aux as `DRAM`.
- Empty KV registration fails clearly.
- `send_aux` uses `DRAM`, `WRITE`, and the expected notification.
- `send_kvcache` uses `VRAM`, `WRITE`, and unsigned pointer math.
- `send_kvcache_slice` uses `VRAM`, `WRITE`, and unsigned addresses.
- `_send_mamba_state` builds correct VRAM source and destination offsets.
- `maybe_send_extra` dispatches Mamba and SWA paths correctly.
- Non-MLA heterogeneous SWA/NSA state transfer raises an actionable error.

### `test/registered/unit/disaggregation/test_nixl_receiver_poll.py`

Covers decode-side receiver polling.

Key coverage:

- Existing conclude state is returned without polling manager state.
- Pre-transfer status is returned before transfer starts.
- Manager success and failure statuses are terminal.
- Waiting timeout records failure and returns `KVPoll.Failed`.
- Completed successful transfer cleans room tracking and transfer status.
- Completed failed transfer returns `KVPoll.Failed`.

### `test/registered/unit/disaggregation/test_nixl_node_failure.py`

Covers prefill node failure handling.

Key coverage:

- Failed prefill node removes matching connection-pool entries.
- Failed prefill node removes cached prefill info and room tracking.
- Pending rooms from the failed node are marked failed.
- Late failed updates do not resurrect cleared room state.

### `test/registered/unit/disaggregation/test_nixl_staging.py`

Covers NIXL staging-buffer control paths.

Key coverage:

- Staging memory registration uses `VRAM`.
- Empty staging registration fails clearly.
- Staging prefetch is a no-op when staging is disabled or KV buffers are missing.
- Staging prefetch marks rooms as handled when no peer needs staging.
- Deferred staging allocation requeues the chunk.
- Oversized staging allocation raises an actionable error.
- Staging transfer builds the expected bounded notification tag.
- `send_kvcache_staged` posts one bulk `VRAM` `WRITE` and preserves high-bit staging pointers with unsigned request arrays.
- Staging transfer falls back when prefill staging buffer is too small.

### `test/registered/unit/disaggregation/test_nixl_hybrid_state.py`

Covers hybrid state transfer logic.

Key coverage:

- Empty state indices send nothing.
- Mamba homogeneous TP dispatches to `_send_mamba_state`.
- Mamba heterogeneous TP dispatches to `_send_mamba_state_slice`.
- Mamba state slice builds expected descriptor offsets.
- Missing state dimension metadata falls back to normal Mamba state transfer.
- SWA state index mismatch raises.
- Unsupported state types raise an actionable error.

### `test/registered/unit/disaggregation/test_disaggregation_rank_mapping.py`

Covers common PD rank mapping and pointer slicing used by NIXL tests.

Key coverage:

- Same TP maps to matching rank.
- Decode TP larger than prefill TP groups decode ranks.
- Prefill TP larger than decode TP selects multiple prefill ranks.
- CP ranks are filtered unless all-CP transfer is enabled.
- Decode PP size 1 targets all prefill PP ranks.
- Invalid decode CP size raises.
- MHA PP pointer slicing selects matching K/V ranges.
- Compressed MLA layout requires `prefill_end_layer`.
- Compressed MLA KV and state layouts slice expected sections.

## Run Guide

### Prerequisites

Install SGLang test dependencies in the repo environment:

```bash
pip install -e "python[test]"
```

If the source tree is not installed editable, run commands with:

```bash
export PYTHONPATH=python
```

### Run Individual Test Files

```bash
python3 test/registered/unit/disaggregation/test_nixl_transfer_info.py
python3 test/registered/unit/disaggregation/test_nixl_transfer_status.py
python3 test/registered/unit/disaggregation/test_nixl_notifications.py
python3 test/registered/unit/disaggregation/test_nixl_backend_config.py
python3 test/registered/unit/disaggregation/test_nixl_descriptor_building.py
python3 test/registered/unit/disaggregation/test_nixl_receiver_poll.py
python3 test/registered/unit/disaggregation/test_nixl_node_failure.py
python3 test/registered/unit/disaggregation/test_nixl_staging.py
python3 test/registered/unit/disaggregation/test_nixl_hybrid_state.py
python3 test/registered/unit/disaggregation/test_disaggregation_rank_mapping.py
```

### Run All Disaggregation Unit Tests

```bash
pytest test/registered/unit/disaggregation/ -v
```

### Run CPU CI Suite Discovery and Execution

```bash
python3 test/run_suite.py --hw cpu --suite base-a-test-cpu
```

### Fast Syntax Check

This does not execute imports deeply, but it catches syntax errors:

```bash
python3 -m py_compile \
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
```

### Whitespace Check

```bash
git diff --check -- \
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
```
