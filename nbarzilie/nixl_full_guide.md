# NIXL Full Implementation and Integration Guide

## Purpose

This guide is written for an implementation agent that needs to understand, use, and integrate NIXL in LLM inference engines. It combines the repository-level README, the NIXL architecture notes, Python API notes, backend plugin guide, and Python examples into one operational reference. The word "agent" has two meanings in this document:
- **Implementation agent** means the software engineer or AI coding agent reading this guide; **NIXL agent** means the `nixlAgent` / `nixl_agent` object created inside an inference process.

NIXL stands for NVIDIA Inference Xfer Library. NIXL is a point-to-point data transfer library for distributed inference systems. NIXL is designed to move data efficiently between memory and storage types used by inference engines. NIXL is not an inference engine. NIXL is a data plane abstraction for inference engines. NIXL is useful when an inference stack needs high-bandwidth, low-latency movement of tensors, bytes, KV-cache pages, model blocks, request state, or storage-backed data. NIXL hides backend-specific transport and memory registration details behind a common API. NIXL supports a modular backend plugin model. The current repository documentation highlights UCX and GPUDirect Storage as major backend examples. Other storage-oriented and network-oriented plugins may be available depending on the build. NIXL is especially relevant for inference engines that split work across processes, GPUs, nodes, or storage tiers. Common LLM inference uses include:
- Moving KV-cache blocks between prefill and decode workers; Moving activations or state between disaggregated inference stages; Moving model or cache data between GPU memory and CPU memory; Moving tensors between GPU memory and local or remote storage; Building a control/data-plane split where a scheduler coordinates metadata and NIXL moves bytes; Supporting dynamic scale-out and scale-in of inference workers; Reducing engine-specific RDMA, UCX, GDS, and file backend code.

This document is intentionally detailed. It is meant to be read by an agent that has to implement code correctly without repeatedly rediscovering NIXL concepts from the source tree.

## Source Files Read (https://github.com/ai-dynamo/nixl.git)

This guide is based on the requested files:
- `docs/nixl.md`; `docs/python_api.md`; `docs/BackendGuide.md`; `README.md`; `examples/python/remote_storage_example/README.md`; `examples/python/basic_two_peers.py`.

It also uses implementation details from these repository files because the requested Python API doc points to them:
- `src/api/python/_api.py`; `src/api/cpp/nixl.h`; `src/api/cpp/nixl_types.h`; `src/api/cpp/nixl_params.h`; `src/api/cpp/backend/backend_engine.h`; `src/api/cpp/backend/backend_aux.h`; `src/bindings/python/nixl_bindings.cpp`; `examples/python/expanded_two_peers.py`; `examples/python/partial_md_example.py`; `examples/python/query_mem_example.py`; `examples/python/nixl_gds_example.py`; `examples/python/remote_storage_example/nixl_p2p_storage_example.py`; `examples/python/remote_storage_example/nixl_storage_utils/common.py`.

## Executive Summary

NIXL creates a `nixl_agent` per inference process that participates in data movement. Each NIXL agent has a globally meaningful name. Each NIXL agent instantiates one or more transfer backends. Each backend knows how to move data for specific memory/storage types. Application code allocates buffers using its normal allocator. Application code registers those buffers with NIXL. NIXL returns registration descriptors. The application exchanges serialized NIXL metadata out of band. Metadata exchange can happen through direct peer-to-peer socket-like helpers, ETCD, Redis-like external systems, or any custom side channel. After metadata is loaded, an initiator can create a transfer request. Transfer requests are asynchronous. The initiator posts the request and polls status until completion. Optionally, a completion notification can be delivered to the target or peer. NIXL supports both `READ` and `WRITE` operations. For NIXL transfer operations, read/write is expressed from the initiator's point of view:
- `READ`: initiator reads remote/target descriptors into local descriptors; `WRITE`: initiator writes local descriptors into remote/target descriptors.

The application is responsible for correctness around buffer lifetime, buffer reuse, transfer ordering, and memory hazards. NIXL does not provide global ordering across transfer requests. NIXL does not lock application memory ranges. NIXL can select a backend automatically if more than one backend is registered and capable. Application code can also restrict operations to specific backends. The fastest integration path in Python is:
1. `pip install nixl` 2. Create `nixl_agent_config` 3. Create `nixl_agent` 4. Allocate PyTorch tensors 5. Register tensors with `register_memory` 6. Exchange metadata 7. Build transfer descriptor lists with `get_xfer_descs` 8. Create transfer handle with `initialize_xfer` 9. Post with `transfer` 10. Poll with `check_xfer_state` 11. Release handle, remove metadata, deregister memory.

## Supported Platform and Installation

NIXL is supported on Linux. The README states it is tested on Ubuntu 22.04, Ubuntu 24.04, and Fedora. macOS and Windows are not currently supported. For an implementation agent working from macOS, do not assume local runtime validation will work without a Linux container, VM, or remote Linux host. The preferred Python install path is:
```bash
pip install nixl
```

The PyPI package includes the Python API and NIXL libraries. The PyPI package includes CUDA 12 and CUDA 13 backends. At runtime, the correct CUDA backend is selected according to the CUDA version reported by PyTorch. The source build uses Meson and Ninja. Minimal Linux build dependencies include:
- `build-essential`, `cmake`, `pkg-config` on Ubuntu; `gcc-c++`, `cmake`, `pkg-config` on Fedora; Python packages including `meson`, `ninja`, `pybind11`, and `tomlkit`.

Source build:
```bash
meson setup build
ninja -C build
ninja -C build install
```

Debug build:
```bash
meson setup build --buildtype=debug
ninja -C build
```

Common NIXL-specific Meson options:
- `-Dbuild_docs=true`; `-Ducx_path=/path/to/ucx`; `-Dinstall_headers=true`; `-Ddisable_gds_backend=false`; `-Dcudapath_inc=/path/to/cuda/include`; `-Dcudapath_lib=/path/to/cuda/lib`; `-Dstatic_plugins=UCX,POSIX`; `-Denable_plugins=UCX,POSIX`; `-Ddisable_plugins=GDS`.

`enable_plugins` and `disable_plugins` should not be used together. `NIXL_NO_STUBS_FALLBACK` controls whether a stub library is built if the full library build fails. For source Python bindings, the README describes a workflow using `uv`, PyTorch, Meson/Ninja, and the generated meta wheel. For CUDA 12 from source:
```bash
pip install .
meson setup build
ninja -C build install
pip install build/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl
```

For CUDA 13 from source:
```bash
pip install .
./contrib/tomlutil.py --wheel-name nixl-cu13 pyproject.toml
meson setup build
ninja -C build install
pip install build/src/bindings/python/nixl-meta/nixl-*-py3-none-any.whl
```

Basic Python import smoke test:
```bash
python3 -c "import nixl; agent = nixl.nixl_agent('agent1')"
```

When validating examples, start the target process first, then the initiator. Example:
```bash
python3 examples/python/expanded_two_peers.py --mode=target --use_cuda=true --ip=127.0.0.1 --port=4242 &
sleep 5
python3 examples/python/expanded_two_peers.py --mode=initiator --use_cuda=true --ip=127.0.0.1 --port=4242
```

## Core Architecture

NIXL is organized around three major abstractions:
- Memory sections; Transfer backends; Metadata handling.

A memory section is a set of address ranges or storage ranges registered with a NIXL agent. The same conceptual application buffer may be registered with one or more backends. Backends expose a common interface to the NIXL agent. The NIXL agent chooses a backend according to memory types, remote metadata, and available backend capabilities. Metadata handling stores and exchanges the information needed for one-sided transfers. Metadata contains backend connection information and remote identifiers for registered memory. Metadata excludes local-only information that the remote side does not need. The inference engine is expected to provide orchestration. NIXL expects an external conductor or scheduler to allocate buffers, decide which workers communicate, and exchange metadata through a control channel. The control path and data path are deliberately separate. The control path exchanges metadata, descriptors, work requests, and readiness signals. The data path moves bytes through NIXL backends. For LLM inference engines, keep this split explicit:
- Scheduler and worker RPCs are control path; NIXL transfers are data path; KV-cache ownership maps are control path; KV-cache block bytes are data path; Worker membership and liveness are control path; GPU-to-GPU, GPU-to-CPU, or GPU-to-storage payload movement is data path.

## NIXL Agent Model

A NIXL agent is created inside each process that participates in transfers. An inference worker may own one NIXL agent. A multi-GPU process may use one NIXL agent for several GPU devices. An inference stack may create one NIXL agent per worker process, not necessarily per GPU. Agent names should be unique in the service. Choose names that are stable enough for metadata lookup. Good names:
- `prefill-node-3-rank-0`; `decode-worker-12`; `kv-cache-server-a100-07`; `rank-4`.

Bad names:
- `worker`; `target`; `localhost`; Random names not tracked by the scheduler.

The examples use names like `target`, `initiator`, and mode strings for simplicity. Production inference engines should map agent names to scheduler worker IDs. The agent does not allocate user tensor memory. The inference engine allocates memory with PyTorch, CUDA, custom pools, file descriptors, or other mechanisms. The agent registers those allocations. The agent returns descriptor objects that the application must keep while the memory remains registered. When the agent is destroyed, it attempts cleanup of remaining registrations and backend resources. Explicit cleanup is still preferred.

## Memory and Storage Types

NIXL represents memory/storage spaces through segment types. Core segment types in the C++ API are:
- `DRAM_SEG`; `VRAM_SEG`; `BLK_SEG`; `OBJ_SEG`; `FILE_SEG`.

The Python wrapper accepts string names:
- `"DRAM"`; `"VRAM"`; `"BLOCK"`; `"OBJ"`; `"FILE"`; Deprecated aliases: `"cpu"` and `"cuda"`.

Use `"DRAM"` for CPU memory. Use `"VRAM"` for GPU memory. Use `"FILE"` for file descriptors and file-backed storage. Use `"BLOCK"` for block storage. Use `"OBJ"` for object storage. Descriptor fields are interpreted differently by memory type. For DRAM:
- `addr` is a process virtual address; `len` is byte length; `devID` is usually `0` or a region ID; registration metadata string is usually empty or application-defined.

For VRAM:
- `addr` is a GPU virtual address; `len` is byte length; `devID` is the GPU ID; registration metadata string is usually empty.

For FILE:
- `addr` is typically an offset; `len` is byte length or `0` for some query cases; `devID` is a file descriptor; registration metadata string can contain path or mode information, depending on backend needs.

For object storage:
- `addr` is typically an object offset; `len` is byte length or `0`; `devID` may be a key or identifier depending on backend; metadata string can carry an extended key or bucket-like identifier.

For block storage:
- `addr` is typically a block offset; `len` is byte length; `devID` is a volume identifier; metadata string depends on backend.

## Descriptor Lists

NIXL uses descriptor lists for registration and transfer. Registration descriptor list type:
- C++: `nixl_reg_dlist_t`; Python binding: `nixlRegDList`; Python helper: `get_reg_descs`.

Transfer descriptor list type:
- C++: `nixl_xfer_dlist_t`; Python binding: `nixlXferDList`; Python helper: `get_xfer_descs`.

Registration descriptors are four-tuples:
```python
(addr, length, device_id, metadata_string)
```

Transfer descriptors are three-tuples:
```python
(addr, length, device_id)
```

The registration metadata string is backend-specific. For PyTorch tensors, the Python wrapper can construct descriptors automatically. For raw memory or file descriptors, pass tuples and specify `mem_type`. For NumPy arrays, use an `Nx3` C-contiguous array of `uint64` or `int64`. NumPy arrays are useful when constructing many descriptors efficiently. For PyTorch tensor inputs:
- Tensor must be contiguous; CPU tensor maps to `"DRAM"`; CUDA tensor maps to `"VRAM"`; `data_ptr()` becomes `addr`; `numel() * element_size()` becomes length; CPU `get_device()` is normalized to `0`; CUDA `get_device()` becomes GPU device ID.

For list of tensors:
- All tensors must be on the same device type; Each tensor must be contiguous; A descriptor is generated per tensor.

Avoid passing non-contiguous tensor views directly. If an LLM engine uses paged KV-cache tensors, ensure each descriptor corresponds to a contiguous block. If block layout uses strided views, create contiguous views or construct explicit address/length descriptors only when you know the underlying layout is valid. `nixlRegDList.trim()` converts a registration descriptor list into transfer descriptors by dropping metadata. This is useful for transferring the exact regions that were registered. Example pattern:
```python
reg_descs = agent.register_memory(tensor)
xfer_descs = reg_descs.trim()
```

## Transfer Operations

NIXL supports two transfer operations:
- `READ`; `WRITE`.

From the initiator perspective:
- `READ` copies from remote descriptors into local descriptors; `WRITE` copies from local descriptors into remote descriptors.

For local loopback transfers:
- The remote agent name is the local agent name; Storage examples use this for memory-to-file and file-to-memory movement.

For remote transfers:
- The initiator must have loaded remote metadata for the target agent; The remote descriptors must describe buffers registered by the target; The initiator's local descriptors must describe local registered buffers.

Transfer descriptors are paired by position for `initialize_xfer`. When using prepared descriptor lists, descriptor subsets are selected by index. Descriptor sizes should match in a way the backend can perform. For common tensor movement, local and remote descriptors should have corresponding byte lengths. NIXL can merge adjacent descriptors internally unless a deprecated skip flag is used. Transfer handles can be reposted after the previous transfer is complete. Only one active transfer can exist for a given transfer handle at a time. Reposting an active request is an error. No ordering is guaranteed across separate transfer requests. The application must impose ordering by waiting for required transfers to complete before posting dependent transfers.

## Metadata Model

Metadata is central to NIXL. NIXL metadata tells another agent how to access registered buffers through matching backends. Metadata is generated after backend creation and memory registration. Metadata includes:
- Agent name; Backend connection information; Public remote identifiers for registered memory; Backend labels that route metadata to matching backend implementations.

Metadata does not initiate a connection by itself. Loading metadata caches access information. Connections may be established lazily on first transfer. The application can call `make_connection` to proactively connect. Metadata can be exchanged through:
- A custom side channel; Direct peer-to-peer helper methods; Central metadata services such as ETCD; A scheduler-owned metadata store; Any RPC layer that can move binary blobs.

Side-channel metadata exchange:
1. Each agent calls `get_agent_metadata()` 2. Application sends bytes to peer using its own control channel 3. Peer calls `add_remote_agent(metadata)` 4. Optional: peer calls `make_connection(remote_agent)`.

Direct helper exchange:
1. Target listens by enabling the listen thread 2. Initiator calls `fetch_remote_metadata(remote, ip, port)` 3. Initiator calls `send_local_metadata(ip, port)` 4. Each side polls `check_remote_metadata`.

ETCD exchange:
1. Set `NIXL_ETCD_ENDPOINTS` 2. Optionally set `NIXL_ETCD_NAMESPACE` 3. Agent sends local metadata with no peer IP 4. Other agents fetch metadata by agent name 5. Use labels for partial metadata workflows.

Invalidate metadata when a remote agent leaves, fails, or changes registrations. Removing remote metadata disconnects existing backend connections to that agent. After invalidation, transfers to that agent cannot be initiated until metadata is loaded again.

## Configuration Files and Environment Variables

NIXL and plugins can use configuration values from environment variables or TOML config files. When a config option such as `NIXL_OPTION_FOO` is requested, NIXL first checks the environment. If the option exists in the environment, the config file is not used for that option. This is true even if conversion of the environment value fails. Config file selection order:
1. Path from `NIXL_CONFIG_FILE`, if set 2. `$HOME/.nixl.cfg`, if `HOME` is set and the file exists 3. `/etc/nixl.cfg`, if it exists.

If no config file is found, only environment variables are used. If a selected config file fails to read or parse, there is no fallback to the next config file. Integer environment values should use minimal numeric representation. Avoid leading spaces, trailing spaces, unnecessary plus signs, and unnecessary leading zeroes. Unsigned integers can use `0x` hexadecimal. Boolean true values include:
- `y`; `yes`; `on`; `1`; `true`; `enable`.

Boolean false values include:
- `n`; `no`; `off`; `0`; `false`; `disable`.

Boolean matching is case-insensitive. For ETCD:
```bash
export NIXL_ETCD_ENDPOINTS="http://localhost:2379"
export NIXL_ETCD_NAMESPACE="/nixl/agents"
```

For network performance tuning, one example from the remote storage docs is:
```bash
export UCX_MAX_RMA_RAILS=1
```

This may help remote reads from VRAM to DRAM in some UCX rail-selection situations. Do not assume `UCX_MAX_RMA_RAILS=1` is universally optimal. Benchmark on target hardware. For plugin discovery, examples log `NIXL_PLUGIN_DIR` if set. If plugins are missing, confirm plugin search path and build options.

## Python API Import Paths

The public examples use both import styles:
```python
from nixl import nixl_agent, nixl_agent_config
```

and:
```python
from nixl._api import nixl_agent, nixl_agent_config
```

Prefer the public package import if available:
```python
from nixl import nixl_agent, nixl_agent_config
```

Use `_api` imports only when following repository examples or when the public package layout requires it. The logging helper is:
```python
from nixl.logging import get_logger
```

Useful lower-level helpers appear in examples:
```python
import nixl._utils as nixl_utils
```

The helper module is used for raw malloc/free and buffer verification examples. PyTorch tensors are the easiest path for inference-engine integration. Raw pointer descriptors are useful for custom CUDA allocators, memory pools, and file/storage descriptors.

## Python Agent Configuration

Python config class:
```python
nixl_agent_config(
    enable_prog_thread=True,
    enable_listen_thread=False,
    listen_port=DEFAULT_COMM_PORT,
    capture_telemetry=False,
    num_threads=0,
    backends=["UCX"],
    sync_mode=None,
)
```

Important fields:
- `enable_prog_thread`: enable a progress thread if backend supports or needs it; `enable_listen_thread`: enable listener thread for direct metadata exchange; `listen_port`: port used by listener thread; `capture_telemetry`: capture transfer telemetry regardless of environment; `num_threads`: shared worker thread count for supported backends; `backends`: backend names to instantiate when the agent is created; `sync_mode`: one of the NIXL thread synchronization modes.

Default Python backends list is `["UCX"]`. If you want to create an agent without automatically instantiating UCX, pass `backends=[]`. This is useful when probing plugins, using storage-only examples, or creating backends explicitly. Example:
```python
config = nixl_agent_config(
    enable_prog_thread=True,
    enable_listen_thread=True,
    listen_port=5555,
    backends=["UCX"],
)
agent = nixl_agent("target", config)
```

If `enable_listen_thread` is true and `sync_mode` is not specified, Python wrapper selects strict synchronization. If `enable_listen_thread` is false and `sync_mode` is not specified, Python wrapper selects no synchronization. Thread synchronization enum:
- `NIXL_THREAD_SYNC_NONE`; `NIXL_THREAD_SYNC_STRICT`; `NIXL_THREAD_SYNC_RW`; `NIXL_THREAD_SYNC_DEFAULT`.

For multithreaded inference engines, be explicit about synchronization. If multiple threads will call the same `nixl_agent`, prefer a synchronization mode that matches the access pattern. If each worker thread owns its own agent or calls are externally serialized, no synchronization may be acceptable.

## Python Agent Creation

Create an agent:
```python
agent = nixl_agent("agent-name", config)
```

Alternative:
```python
agent = nixl_agent("agent-name", instantiate_all=True)
```

If both config and `instantiate_all=True` are supplied, `instantiate_all` is ignored. On creation, Python wrapper:
- Builds a C++ `nixlAgentConfig`; Creates C++ `nixlAgent`; Discovers available plugins; Instantiates configured backends that are available; Stores backend handles; Stores backend supported memory types and options.

If no plugins are available, agent creation fails. After agent creation, useful discovery methods are:
```python
agent.get_plugin_list()
agent.get_plugin_mem_types("UCX")
agent.get_plugin_params("UCX")
agent.get_backend_mem_types("UCX")
agent.get_backend_params("UCX")
```

Create a backend explicitly:
```python
params = agent.get_plugin_params("POSIX")
agent.create_backend("POSIX", params)
```

Create with custom parameters:
```python
agent.create_backend("UCX", {"num_threads": "4"})
```

Backend names depend on built and installed plugins. Common names in examples:
- `UCX`; `GDS`; `GDS_MT`; `POSIX`; `MOCK_DRAM`; `OBJ`; `UCCL`.

Do not assume a backend exists. Always check `get_plugin_list()` before selecting optional backends. For storage examples, prefer `GDS_MT` if present, then `POSIX`. For network transfers, examples use `UCX`.

## Python Memory Registration

Register a tensor:
```python
reg_descs = agent.register_memory(tensor)
```

Register a list of tensors:
```python
rows = [tensor[i, :] for i in range(tensor.shape[0])]
reg_descs = agent.register_memory(rows)
```

Register raw DRAM tuple descriptors:
```python
reg = [(addr, size, 0, "")]
reg_descs = agent.register_memory(reg, mem_type="DRAM")
```

Register file descriptors:
```python
file_reg = [(0, buf_size, fd, "")]
file_descs = agent.register_memory(file_reg, mem_type="FILE")
```

Restrict registration to specific backends:
```python
agent.register_memory(tensor, backends=["UCX"])
```

When `backends=[]`, NIXL tries all instantiated backends that support the memory type. Registration should happen during worker initialization when possible. For LLM serving, register memory pools, slabs, or page blocks once. Avoid registering and deregistering for every request. Registration may involve backend calls such as memory pinning, memory key creation, or file setup. Registration overhead is often too high for per-token or per-request hot paths. Deregister:
```python
agent.deregister_memory(reg_descs)
```

Deregister from specific backends:
```python
agent.deregister_memory(file_descs, backends=["POSIX"])
```

Keep application buffers alive while registered. Do not free, resize, or reallocate a tensor while NIXL still has a registration for it. Do not use stale descriptors after deregistration. For inference engines with memory pools, tie NIXL registration lifetime to pool lifetime, not request lifetime.

## Python Descriptor Construction

Build transfer descriptors from tensors:
```python
xfer_descs = agent.get_xfer_descs(tensor)
```

Build transfer descriptors from list of contiguous tensors:
```python
rows = [tensor[i, :] for i in range(tensor.shape[0])]
xfer_descs = agent.get_xfer_descs(rows)
```

Build transfer descriptors from raw tuples:
```python
xfer_descs = agent.get_xfer_descs([(addr, size, dev_id)], mem_type="DRAM")
```

Build transfer descriptors from NumPy:
```python
descs_np = np.array(
    [[addr0, len0, dev0], [addr1, len1, dev1]],
    dtype=np.uint64,
)
xfer_descs = agent.get_xfer_descs(descs_np, mem_type="VRAM")
```

Build registration descriptors explicitly:
```python
reg_descs = agent.get_reg_descs([(addr, size, dev_id, "meta")], mem_type="DRAM")
```

Serialize transfer descriptors for control-channel exchange:
```python
payload = agent.get_serialized_descs(xfer_descs)
```

Deserialize descriptors received from another process:
```python
remote_descs = agent.deserialize_descs(payload)
```

The Python wrapper uses `pickle` for descriptor serialization. Do not unpickle descriptor payloads from untrusted sources. In production, only accept descriptor payloads over authenticated scheduler/worker channels. For LLM inference engines, descriptors should often represent KV-cache blocks or groups of blocks. Example descriptor dimensions for paged KV cache:
- One descriptor per KV page; One descriptor per contiguous range of pages; One descriptor per layer block if layout differs by layer; One descriptor per rank-owned shard.

Avoid excessive descriptor counts. If the layout allows, coalesce adjacent pages into larger descriptors. Fewer, larger registrations are usually better than many tiny registrations. Fewer transfer descriptors can reduce validation overhead and backend request overhead. Very large descriptors may reduce scheduling flexibility. Use benchmark data to decide transfer granularity.

## Python Metadata APIs

Get local metadata as bytes:
```python
md = agent.get_agent_metadata()
```

Load remote metadata:
```python
remote_name = agent.add_remote_agent(md)
```

Remove remote metadata:
```python
agent.remove_remote_agent("target")
```

Send local metadata using built-in direct or central mechanisms:
```python
agent.send_local_metadata(ip_addr, port)
```

Fetch remote metadata:
```python
agent.fetch_remote_metadata("target", ip_addr, port)
```

Invalidate local metadata remotely or centrally:
```python
agent.invalidate_local_metadata(ip_addr, port)
```

Check if metadata is loaded:
```python
ready = agent.check_remote_metadata("target")
```

Check partial metadata for a specific descriptor list:
```python
ready = agent.check_remote_metadata("target", target_xfer_descs)
```

Partial metadata:
```python
partial = agent.get_partial_agent_metadata(reg_descs, inc_conn_info=True, backends=["UCX"])
agent.send_partial_agent_metadata(reg_descs, True, ["UCX"], ip_addr, port)
agent.fetch_remote_metadata("target", label="label_1")
```

Partial metadata is useful when the full registration set is large or dynamic. Partial metadata can reduce control-plane payload size. Partial metadata lets a worker expose only specific buffers or pages. With ETCD, labels distinguish metadata blobs. With direct peer exchange, labels are ignored by the peer send/fetch helpers. If `descs` is empty for partial metadata, only backend connection information is included. If `descs` is non-empty, descriptor metadata is included. If `inc_conn_info=True`, connection information for matching backends is included too. If `backends` is non-empty, partial metadata is restricted to those backend handles.

## Python Connection API

Proactively connect to a remote agent:
```python
agent.make_connection("target")
```

Restrict to backends:
```python
agent.make_connection("target", backends=["UCX"])
```

This is optional. Loading remote metadata does not necessarily connect. First transfer can connect lazily. Proactive connection can reduce first-token or first-request transfer latency. Use proactive connection when the scheduler knows stable worker pairings. Avoid aggressive full mesh connection if thousands of workers may never communicate. For prefill/decode disaggregation, consider connecting:
- When a decode worker is assigned to a prefill worker; When a KV-cache route becomes active; During warmup for colocated worker groups.

Disconnect is driven by remote metadata invalidation.

## Python Transfer APIs

Simple combined transfer creation:
```python
handle = agent.initialize_xfer(
    "READ",
    local_descs,
    remote_descs,
    "target",
    b"optional-notification",
)
```

Restrict transfer backend:
```python
handle = agent.initialize_xfer(
    "WRITE",
    local_descs,
    remote_descs,
    "target",
    backends=["UCX"],
)
```

Post transfer:
```python
state = agent.transfer(handle)
```

Possible Python states:
- `"DONE"`; `"PROC"`; `"ERR"`.

Poll:
```python
while True:
    state = agent.check_xfer_state(handle)
    if state == "DONE":
        break
    if state == "ERR":
        raise RuntimeError("NIXL transfer failed")
```

Release:
```python
agent.release_xfer_handle(handle)
```

Or:
```python
handle.release()
```

If release is called while transfer is active, NIXL attempts cancellation. If cancellation cannot complete immediately, release may fail. Do not rely only on Python garbage collection for transfer handles. Explicit release is clearer and safer. Prepared descriptor list flow:
```python
local_side = agent.prep_xfer_dlist("", local_descs)
remote_side = agent.prep_xfer_dlist("target", remote_descs)
handle = agent.make_prepped_xfer(
    "READ",
    local_side,
    [0, 4, 8],
    remote_side,
    [8, 4, 0],
    b"read-complete",
)
```

Prepared descriptor lists are useful when:
- Descriptor blocks are known ahead of time; Many transfers select subsets of the same descriptors; Repeated validation/preparation would be costly; KV-cache block sets are reused across requests or batches.

Release prepared descriptor handles:
```python
agent.release_dlist_handle(local_side)
agent.release_dlist_handle(remote_side)
```

Or:
```python
local_side.release()
remote_side.release()
```

Estimate cost:
```python
duration_us, err_us, method = agent.estimate_xfer_cost(handle)
```

Query selected backend:
```python
backend_name = agent.query_xfer_backend(handle)
```

Transfer telemetry:
```python
telem = agent.get_xfer_telemetry(handle)
print(telem.startTime, telem.postDuration, telem.xferDuration, telem.totalBytes, telem.descCount)
```

Telemetry fields are in microseconds for time durations. Telemetry may require `capture_telemetry=True` or environment-based telemetry configuration. If telemetry is unavailable, the binding may raise a NIXL telemetry-related exception.

## Python Notification APIs

NIXL notifications are backend-supported messages. UCX supports notifications according to the backend guide. Storage-only backends may not support notifications. Send standalone notification:
```python
agent.send_notif("target", b"ready")
```

Restrict backend:
```python
agent.send_notif("target", b"ready", backend="UCX")
```

Get new notifications:
```python
notifs = agent.get_new_notifs()
```

Notification map shape:
```python
{
    "remote_agent_name": [b"message1", b"message2"]
}
```

Accumulate notifications:
```python
notifs = agent.update_notifs()
```

Check for a completion notification and remove it from internal map:
```python
done = agent.check_remote_xfer_done("initiator", b"Done_reading")
```

Prefix matching is default. Substring matching:
```python
done = agent.check_remote_xfer_done("initiator", b"uuid", tag_is_prefix=False)
```

Notifications are useful for:
- Transfer completion signaling to a target; Descriptor exchange in examples; Request messages in storage pipeline example; Lightweight control acknowledgements.

For production inference engines, notifications should be used carefully. They are not a replacement for the scheduler control plane. They can be useful for data-plane-adjacent events, but global request state should remain in the engine scheduler. Notification payloads are bytes. The Python wrapper accepts strings in some paths but bytes are preferred. When embedding serialized descriptors in notification payloads, define a stable framing format. The remote storage example uses:
- First 4 bytes: operation tag; Next 4 bytes: iteration count; Remaining bytes: serialized descriptor list.

For production, use length-prefixed messages or a structured binary protocol.

## Query Memory API

QueryMem asks a specific backend for information about descriptors. Python:
```python
resp = agent.query_memory(file_paths, "POSIX", mem_type="FILE")
```

Return value is a list. Each item is either:
- `None`, if not found or inaccessible; A dictionary of backend-specific information.

The query memory example uses file descriptors represented as:
```python
(0, 0, 0, "/path/to/file")
```

For POSIX file query responses, example fields include:
- `size`; `mode`; `mtime`.

QueryMem is useful for:
- Checking file accessibility before registering storage; Inspecting storage object size; Validating backend-specific reachability; Preflight checks in inference cache offload systems.

Always specify a backend. QueryMem is backend-specific by design.

## C++ API Mental Model

The C++ `nixlAgent` is the core API. The Python `nixl_agent` wraps C++ calls. Important C++ methods:
- `getAvailPlugins`; `getPluginParams`; `getBackendParams`; `createBackend`; `registerMem`; `deregisterMem`; `queryMem`; `makeConnection`; `prepXferDlist`; `makeXferReq`; `createXferReq`; `estimateXferCost`; `postXferReq`; `getXferStatus`; `getXferTelemetry`; `queryXferBackend`; `releaseXferReq`; `releasedDlistH`; `getNotifs`; `genNotif`; `getLocalMD`; `getLocalPartialMD`; `loadRemoteMD`; `invalidateRemoteMD`; `sendLocalMD`; `sendLocalPartialMD`; `fetchRemoteMD`; `invalidateLocalMD`; `checkRemoteMD`.

C++ status values include:
- `NIXL_SUCCESS = 0`; `NIXL_IN_PROG = 1`; `NIXL_ERR_NOT_POSTED`; `NIXL_ERR_INVALID_PARAM`; `NIXL_ERR_BACKEND`; `NIXL_ERR_NOT_FOUND`; `NIXL_ERR_MISMATCH`; `NIXL_ERR_NOT_ALLOWED`; `NIXL_ERR_REPOST_ACTIVE`; `NIXL_ERR_UNKNOWN`; `NIXL_ERR_NOT_SUPPORTED`; `NIXL_ERR_REMOTE_DISCONNECT`; `NIXL_ERR_CANCELED`; `NIXL_ERR_NO_TELEMETRY`.

Python maps many failing C++ statuses into exceptions through pybind. Python convenience methods also map transfer progress statuses into strings. C++ `nixlAgentOptionalArgs` can carry:
- Backend handle hints; Notification payload; Skip descriptor merge flag; Include-connection-info flag; Peer IP address; Port; Metadata label; Custom backend parameter.

Python wrapper exposes these through method arguments rather than direct struct construction.

## Backend Plugin Architecture

NIXL has a northbound API and a southbound API. The northbound API is what inference engines call. The southbound API is what backend plugins implement. Backend plugins are C++ implementations. Plugins can be dynamically loaded or statically built. The plugin manager discovers plugins, checks required symbols, loads them, and creates backend engines. Plugin manager API requirements include:
- `get_plugin_name`; `get_plugin_version`; `create_engine`; `destroy_engine`; `get_backend_mems`; `get_backend_options`.

Backends expose capability indicators:
- `supportsLocal()`; `supportsRemote()`; `supportsNotif()`; `getSupportedMems()`.

Capability indicators determine which southbound methods must be implemented. A network backend should support remote transfers. A network backend should support notifications. A network backend should preferably support local transfers too. A storage backend normally supports local operations. A storage backend may not need notifications. UCX is a network backend example. GDS is a storage backend example. UCX supports local, remote, and notifications. GDS supports local storage access. For storage, NIXL typically does not run an agent on the storage system itself. Instead, the local NIXL agent talks to a local storage client or file descriptor. Thus storage operations often look like loopback transfers to the same NIXL agent.

## Backend Southbound Required Methods

Every backend implements:
- Constructor; Destructor; Capability indicators; `registerMem`; `deregisterMem`; `connect`; `disconnect`; `unloadMD`; `prepXfer`; `postXfer`; `checkXfer`; `releaseReqH`.

Remote-capable backends implement:
- `getPublicData`; `getConnInfo`; `loadRemoteConnInfo`; `loadRemoteMD`.

Local-capable backends implement:
- `loadLocalMD`.

Notification-capable backends implement:
- `getNotifs`; `genNotif`.

Optional backend methods include:
- `queryMem`; `estimateXferCost`; Memory view preparation methods.

Backend registration receives one contiguous descriptor at a time. The backend returns a metadata object pointer. NIXL stores that pointer and passes it back during transfer preparation. Backends should not need to rediscover their own registration metadata. Remote metadata loading creates backend-specific objects from serialized public data. Local metadata loading may return the same pointer or a different target-side metadata pointer. `unloadMD` releases backend metadata objects. `releaseReqH` must be non-blocking from the API perspective. If aborting requires a blocking backend call, implement that asynchronously or through backend progress. If release cannot abort immediately, return error until transfer completion makes release safe. Backend `postXfer` starts the transfer asynchronously. Backend `checkXfer` reports progress and may drive backend progress. Small transfers may complete during `postXfer`. Backends should support descriptor-list parallelism where useful. Backends may optimize across descriptors inside one request. Backends may implement load balancing, resiliency, or parallel transfer internally.

## Backend Selection

When a transfer request is created, NIXL determines which backend can perform it. Selection considers:
- Local descriptor memory type; Remote descriptor memory type; Local backends that support those memory types; Remote metadata indicating backends available on the target; Registration coverage for the descriptor ranges; Optional backend hints provided by the user.

If the user supplies backend hints, selection is limited to those backends. If multiple backends can perform a request, NIXL chooses according to its internal ordering or preference. Use `query_xfer_backend(handle)` to inspect which backend was selected. For deterministic production behavior, specify backend hints when necessary. Examples:
```python
handle = agent.initialize_xfer("READ", local, remote, "target", backends=["UCX"])
```
```python
execute_transfer(agent, mem_descs, file_descs, agent.name, "WRITE", ["GDS_MT"])
```

Avoid relying on automatic backend choice when two backends overlap and have very different performance or semantics.

## Basic Two-Peer Flow

The simplest remote flow has a target and an initiator. Target:
1. Create a NIXL agent with listen thread 2. Allocate target tensor 3. Register target tensor 4. Build target transfer descriptors 5. Wait until initiator metadata is loaded 6. Send target descriptors to initiator 7. Wait for completion notification 8. Deregister memory.

Initiator:
1. Create a NIXL agent 2. Allocate local tensor 3. Register local tensor 4. Fetch target metadata 5. Send local metadata to target 6. Receive target transfer descriptors 7. Wait until target metadata is loaded 8. Build local transfer descriptors 9. Create a `READ` handle 10. Post transfer 11. Poll until `DONE` 12. Verify data 13. Remove remote agent 14. Release handle 15. Invalidate local metadata 16. Deregister memory.

Minimal initiator transfer code:
```python
agent.fetch_remote_metadata("target", ip, port)
agent.send_local_metadata(ip, port)

while not agent.check_remote_metadata("target"):
    pass

local_descs = agent.get_xfer_descs(local_tensor)
remote_descs = agent.deserialize_descs(remote_desc_payload)

handle = agent.initialize_xfer(
    "READ",
    local_descs,
    remote_descs,
    "target",
    b"Done_reading",
)

state = agent.transfer(handle)
if state == "ERR":
    raise RuntimeError("post failed")

while True:
    state = agent.check_xfer_state(handle)
    if state == "DONE":
        break
    if state == "ERR":
        raise RuntimeError("transfer failed")
```

The target uses `get_new_notifs()` to wait for `Done_reading`. The target must not free or reuse the tensor until transfer completion is known. The initiator verifies that local tensor data changed as expected.

## Expanded Two-Peer Flow

The expanded example adds several important production patterns. Pattern one: prepare descriptor lists once.
```python
local_side = agent.prep_xfer_dlist("", initiator_descs)
remote_side = agent.prep_xfer_dlist("target", target_descs)
```

Pattern two: make multiple handles from descriptor indices.
```python
handle = agent.make_prepped_xfer(
    "READ",
    local_side,
    [0, 4, 8],
    remote_side,
    [8, 4, 0],
    b"read-complete",
)
```

Pattern three: post multiple independent requests.
```python
for h in read_handles:
    agent.transfer(h)
```

Pattern four: enforce ordering in application logic.
```python
wait_all(read_handles)
post_all(write_handles)
wait_all(write_handles)
```

Pattern five: repost a completed handle with a new notification.
```python
agent.transfer(existing_handle, b"new-notification")
```

Pattern six: construct dynamic offset descriptors from layout information.
```python
remote_descs = agent.get_xfer_descs(
    [(base_addr + offset, length, remote_dev)],
    mem_type=remote_mem,
)
```

These patterns map directly to LLM inference engines. Known KV pages can be prepared once. Per-request page subsets can be selected by indices. Dynamic offsets can be used for partial blocks or packed request state. Reposting can be used for repeated movement between stable buffer locations. The scheduler must order transfers when later compute depends on earlier data movement.

## Local Storage Flow

Storage transfer can be local loopback. The same agent is both local and remote. Example for GDS:
1. Create an agent with no default backend 2. Check `GDS` plugin exists 3. Create `GDS` backend 4. Allocate DRAM buffers 5. Register DRAM descriptors 6. Open a file descriptor 7. Register FILE descriptors 8. Use `initialize_xfer("WRITE", memory_descs, file_descs, agent.name)` 9. Post and wait 10. Use `initialize_xfer("READ", memory_descs2, file_descs, agent.name)` 11. Post and wait 12. Verify buffer content 13. Release handles 14. Deregister memory and file descriptors 15. Free memory and close file descriptor.

Important detail:
```python
remote_agent = agent.name
```

This indicates a local loopback transfer. For file descriptors:
```python
agent1_file_list = [(0, buf_size, fd, "b")]
agent1_file_descs = agent.register_memory(agent1_file_list, "FILE")
agent1_xfer_files = agent1_file_descs.trim()
```

For LLM engines, storage flow is useful for:
- KV-cache spill to local NVMe; KV-cache restore from local NVMe; Snapshotting request state; Staging model blocks or LoRA adapters; Reading prompt cache content from local files.

Use `GDS` or `GDS_MT` for GPU-direct storage when available and configured correctly. Use `POSIX` fallback when GDS is unavailable or not needed. The remote storage README recommends checking that GDS is not running in compatibility mode if true GPU-direct I/O is expected.

## Remote Storage Flow

The remote storage example implements a client/server storage pattern. The server owns storage files. The client owns GPU memory. The client sends a request to the server with descriptors describing client memory. The server performs the required combination of storage and network transfers. Remote read from storage:
1. Server reads from storage into server memory 2. Server writes server memory over network into client memory.

Remote write to storage:
1. Server reads client memory over network into server memory 2. Server writes server memory into storage.

This is an important pattern: NIXL storage backends operate locally to the server process. Network movement between client and server is a separate NIXL transfer through UCX. Remote storage is built by composing storage and network transfers. The example creates an agent with:
- `GDS_MT` if available; `POSIX` if available; `UCX` for network transfer.

Client setup:
1. Client uses VRAM buffers 2. Client registers memory and local test files 3. Client reads `agents_file` 4. For each server, client sends local metadata and fetches remote metadata 5. Client sends a notification containing operation, iteration count, and serialized memory descriptors 6. Client waits for `COMPLETE`.

Server setup:
1. Server uses DRAM buffers 2. Server registers memory and files 3. Server waits for notifications 4. Server deserializes client descriptors 5. Server pipelines network and storage transfers 6. Server sends `COMPLETE`.

The example uses a two-thread pipeline. For remote reads:
- Start storage read first; Then network write previous/available data to requester; Overlap storage and network where ordering allows.

For remote writes:
- Start network read first; Then storage write previous/available data; Overlap network and storage where ordering allows.

Correctness condition:
- For each individual remote storage operation, storage and network steps must occur in the correct order; Across independent iterations, pipelining can overlap.

For inference engines, this maps to:
- KV-cache tiering servers; Prompt-cache object/file servers; Disaggregated memory-cache nodes; Remote SSD-backed KV cache; Shared cache services feeding decode workers.

## Partial Metadata Pattern

Partial metadata avoids sharing all registered buffers. The partial metadata example registers two target descriptor sets. Only the first set is initially sent. Creating a transfer for the second set fails with not-found. After sending second partial metadata, transfer creation succeeds. This behavior is useful and intentional. Use partial metadata when:
- Workers have large memory pools but only expose selected pages; KV blocks are assigned dynamically; Security or isolation requires exposing only request-owned ranges; Metadata payload size matters; Remote workers should not access all registered buffers.

Basic partial metadata flow:
```python
target_agent.send_partial_agent_metadata(
    target_reg_descs,
    True,
    ["UCX"],
    ip_addr,
    init_port,
)
```

With ETCD:
```python
target_agent.send_partial_agent_metadata(
    target_reg_descs,
    True,
    ["UCX"],
    label="label_1",
)
init_agent.fetch_remote_metadata("target", label="label_1")
```

Readiness check:
```python
while not init_agent.check_remote_metadata("target", target_xfer_descs):
    pass
```

If `initialize_xfer` raises a not-found exception, metadata for some descriptor is absent. Do not handle this by blindly retrying forever. Handle it by:
- Requesting missing metadata from owner; Waiting on scheduler assignment; Failing the request if metadata should already be available.

For paged KV-cache systems, partial metadata can be keyed by:
- request ID; cache block ID range; sequence group; model instance; layer range; decode worker assignment.

ETCD labels can carry structured names, but keep label cardinality manageable.

## LLM Inference Integration Model

When integrating NIXL into an LLM inference engine, define ownership clearly. The inference engine owns:
- Request scheduling; Worker membership; Buffer allocation; KV-cache page allocation; Model parallelism layout; Tensor lifetime; Memory reuse policy; Backpressure; Retry policy; Failure recovery; Metadata distribution policy.

NIXL owns:
- Backend plugin abstraction; Memory registration bookkeeping; Remote metadata loading; Backend selection; Transfer request preparation; Asynchronous transfer posting; Transfer status checks; Backend notifications.

Do not push scheduler responsibilities into NIXL. Do not ask NIXL to decide which worker should own a request. Do not ask NIXL to decide when a KV block can be reused. Use NIXL to execute the byte movement chosen by the scheduler. The integration usually has these components:
- `NixlManager` or `TransferManager` per worker process; Memory registration layer tied to engine memory pools; Metadata publisher/subscriber tied to scheduler control plane; Descriptor builder for engine-specific buffer layouts; Transfer submission path; Transfer completion path; Error and invalidation path; Metrics/telemetry path.

Recommended worker startup:
1. Discover GPU rank, local device ID, node ID, and worker ID 2. Create NIXL agent name from scheduler identity 3. Instantiate required backends 4. Allocate or locate engine memory pools 5. Register long-lived memory pools or slabs 6. Publish metadata after registration 7. Optionally preconnect to known peers 8. Report readiness to scheduler.

Recommended request-time transfer:
1. Scheduler determines source and destination workers 2. Scheduler determines source and destination block IDs 3. Workers build transfer descriptors for selected blocks 4. Destination or initiator obtains remote descriptors and metadata 5. Initiator creates transfer handles 6. Initiator posts transfers 7. Engine overlaps transfer with independent compute when possible 8. Engine polls or integrates completion into event loop 9. Engine marks blocks available only after transfer completion 10. Engine releases or reuses handles as appropriate.

Recommended shutdown:
1. Stop accepting new transfers 2. Wait for or cancel active transfers 3. Notify peers or scheduler of removal 4. Invalidate local metadata 5. Remove remote metadata 6. Deregister memory 7. Destroy NIXL agent 8. Free memory pools.

## KV-Cache Transfer Design

KV cache is often the highest-value NIXL integration target. KV cache may be laid out as:
- One allocation per layer; One allocation per GPU block table; One allocation per page pool; One contiguous arena per worker; Separate K and V pools; Interleaved K/V pages; Sharded by tensor parallel rank.

The NIXL descriptor model can represent all of these if you can compute address, length, and device ID. Recommended approach:
- Register large underlying memory regions once; Build transfer descriptors for KV pages or page ranges; Keep a mapping from engine block IDs to NIXL descriptors; Exchange only descriptors needed by a remote transfer.

Do not register each token's KV slice per request. Do not register and deregister per decode step. Prefer registering the whole KV arena or slab. Use partial metadata when exposing only assigned blocks. For each KV block transfer, define:
- Source agent name; Destination agent name; Source memory type; Destination memory type; Source GPU device ID; Destination GPU device ID; Source address and byte length; Destination address and byte length; Layer coverage; Tensor parallel rank; Request or sequence ID; Completion notification or scheduler event.

For prefill-to-decode transfer:
- Prefill worker usually owns source KV; Decode worker owns destination KV; Either worker can be initiator if it has metadata and descriptors; A common design is decode initiates `READ` from prefill into decode KV pages; Another design is prefill initiates `WRITE` to decode pages after decode exposes descriptors.

Decode-initiated `READ` advantages:
- Decode controls when memory is ready; Decode can pull only needed blocks; Decode sees completion locally.

Prefill-initiated `WRITE` advantages:
- Prefill can push immediately after prefill finishes; Decode can wait for a completion notification; Scheduler can reduce pull logic on decode.

Both are valid. Pick based on existing engine control flow. For disaggregated prefill/decode, avoid double-copy through CPU unless backend or topology requires it. Use VRAM-to-VRAM with UCX/GPUDirect RDMA when possible. If direct path is not available, decide explicitly whether fallback is acceptable.

## Descriptor Granularity for KV Cache

Granularity has performance and scheduling tradeoffs. One descriptor per page:
- More flexible; Easier to select sparse blocks; More descriptor overhead.

One descriptor per contiguous page run:
- Lower descriptor overhead; Requires coalescing; Good for contiguous block allocation.

One descriptor per layer:
- Simple when transferring full layer ranges; Poor for sparse requests; Can over-transfer.

One descriptor per K/V pair:
- Works when K and V are contiguous; Avoids separate requests; Requires layout awareness.

Separate descriptors for K and V:
- More universal; More descriptors; Easier when K and V storage differs.

Recommended default:
- Start with one descriptor per contiguous KV page run per tensor pool; Coalesce adjacent pages before transfer; Benchmark against one descriptor per page.

Descriptor lists should remain stable enough to reuse prepared handles when possible. Prepared handle reuse is easiest if source and destination block positions are stable. If destination pages are allocated dynamically per request, `initialize_xfer` may be simpler.

## Integration With Continuous Batching

Continuous batching means memory and scheduling decisions change every step. NIXL transfers must fit into the engine event loop. Important rules:
- Do not block the whole scheduler while polling one transfer; Poll transfer statuses opportunistically; Integrate transfer completion with request state transitions; Avoid reusing destination KV blocks until transfer completion; Avoid freeing source KV blocks until all dependent transfers complete; Keep active transfer handles associated with request IDs.

Possible state machine:
- `WAITING_FOR_METADATA`; `READY_TO_PREPARE_TRANSFER`; `TRANSFER_POSTED`; `TRANSFER_IN_PROGRESS`; `TRANSFER_DONE`; `READY_FOR_DECODE`; `TRANSFER_FAILED`.

If metadata is missing, request it through control plane. If transfer fails, decide whether to retry, reschedule, or fail request. If the source worker dies, invalidate its metadata and mark dependent requests failed or rescheduled. If the destination worker dies, source should release handles and scheduler should clean ownership records.

## Integration With Tensor Parallelism

Tensor parallelism usually means multiple ranks own shards of each layer's state. Use one NIXL agent per process/rank or per multi-rank process. For each TP rank:
- Register only local rank memory; Use agent names that include TP rank; Transfer only matching shard data.

For KV-cache movement:
- The source TP rank should send to the matching destination TP rank; The scheduler must map source rank to destination rank; Descriptors must use the correct local GPU ID for each process.

Avoid transferring all TP shards through one rank unless the engine intentionally centralizes data. NIXL can move multi-GPU descriptor lists if one agent can access several GPUs and backend supports it. However, topology and process model matter. In many inference engines, per-rank process ownership is clearer.

## Integration With Pipeline Parallelism

Pipeline parallelism splits layers across workers. NIXL can move activations or KV-cache for layer ranges. For activation movement:
- Descriptors often represent output tensors between stages; Transfers are latency sensitive; Prefer prepared descriptors for repeated stage buffers.

For KV movement:
- Only layers owned by source stage are relevant; Destination stage may own the same layer range after migration or cache transfer.

Use metadata labels or descriptor payloads that include layer range. Ensure completion of transfers before scheduling dependent pipeline stage compute.

## Integration With Expert Parallelism and MoE

MoE inference may route tokens to expert workers. NIXL can move token states, expert inputs, or cache data between expert processes. Use NIXL when:
- Payloads are large enough to benefit from RDMA/GPU-direct; You need a common abstraction across CPU/GPU/storage; Existing collectives are not a good fit for point-to-point movement.

Do not replace optimized collective or all-to-all libraries blindly. NIXL is a point-to-point transfer abstraction. For MoE all-to-all, benchmark against purpose-built communication paths.

## Memory Registration Strategy

Registration strategy is critical. Bad strategy:
- Allocate per request; Register per request; Transfer; Deregister per request; Free per request.

Good strategy:
- Allocate long-lived pools; Register pools at worker startup; Transfer subranges; Deregister pools at shutdown.

For PyTorch-managed tensors, long-lived tensors can be registered directly. For engine allocators, expose raw pointer and length descriptors. For CUDA caching allocators, avoid registering memory that may be returned and reused unexpectedly. If using PyTorch tensors:
- Keep Python object references alive; Avoid resizing tensors; Avoid operations that reallocate storage.

If using custom pools:
- Register base allocations; Track suballocations; Build xfer descriptors for subranges; Ensure subrange remains allocated until transfer completion.

For fragmented pages:
- Register the whole arena if possible; Build descriptors per page; Coalesce at transfer time.

For multi-GPU workers:
- Include correct GPU device ID in each descriptor; Do not assume CUDA current device equals descriptor device.

## Metadata Distribution Strategy

Full metadata is easiest. Full metadata may become large for many registered regions. Partial metadata is more scalable. Suggested designs: Small fixed cluster:
- Full metadata exchange at startup; Proactive connections between known peers; Invalidate on shutdown.

Dynamic large cluster:
- Metadata in central store; Fetch on demand; Partial metadata for assigned blocks; TTL or heartbeat-managed invalidation.

Prefill/decode disaggregation:
- Decode fetches prefill metadata when assignment occurs; Prefill or scheduler sends descriptors for completed KV blocks; Decode initiates reads; Metadata invalidated when prefill worker releases blocks or exits.

KV-cache service:
- Cache servers publish connection metadata; Descriptor metadata published per cache segment or page range; Clients fetch only assigned cache metadata.

Scheduler-owned metadata:
- Workers call `get_agent_metadata`; Scheduler stores blob; Scheduler distributes blob to peers; Peers call `add_remote_agent`.

This avoids coupling NIXL to a specific metadata service. ETCD-owned metadata:
- Workers call `send_local_metadata`; Peers call `fetch_remote_metadata`; Environment config points to ETCD endpoints.

This is convenient for cloud-native environments. Custom side-channel metadata:
- Use existing engine RPC; Treat metadata bytes as binary; Authenticate and authorize; Version your message schema.

## Transfer Initiator Choice

The initiator is the agent that creates and posts the transfer request. The initiator must have:
- Local descriptors for its side; Remote descriptors for target side; Remote metadata loaded; Registered local memory.

For a `READ`, initiator receives data. For a `WRITE`, initiator sends data. Choosing initiator:
- Choose the side that naturally waits for completion; Choose the side that has scheduling authority; Choose the side that can access both descriptor sets; Choose the side that should report transfer latency.

Decode-pull design:
- Decode worker initiates `READ`; Prefill worker exposes source descriptors; Decode completion directly gates decode scheduling.

Prefill-push design:
- Prefill worker initiates `WRITE`; Decode worker exposes destination descriptors; Completion notification gates decode scheduling.

Storage-server design:
- Server often initiates network transfers after receiving client descriptors; Client sends request notification; Server composes storage and network operations.

There is no single correct initiator for all inference engines. Pick one model and make it consistent.

## Error Handling

Errors can occur at:
- Agent creation; Backend creation; Memory registration; Metadata fetch/load; Transfer creation; Transfer post; Transfer status check; Notification send/read; Handle release; Deregistration.

Python pybind methods may raise exceptions for C++ errors. Python convenience transfer methods return `"ERR"` for some status paths. Always handle both exceptions and `"ERR"` states. Example:
```python
try:
    handle = agent.initialize_xfer("READ", local, remote, "target")
except Exception as exc:
    handle_transfer_setup_failure(exc)
else:
    state = agent.transfer(handle)
    if state == "ERR":
        handle_post_failure()
```

If `initialize_xfer` fails with not-found:
- Check remote metadata is loaded; Check descriptors are within registered remote ranges; Check partial metadata includes those descriptors; Check backend hints are not excluding the only valid backend; Check memory type matches registration.

If post fails:
- Check handle is valid; Check handle is not already active; Check backend connection has not failed; Check remote agent was not invalidated.

If status becomes `"ERR"`:
- Treat destination buffer as undefined unless backend guarantees otherwise; Do not mark KV blocks ready; Release or attempt to release handle; Notify scheduler; Consider invalidating remote metadata on disconnect errors.

If release fails on active transfer:
- Keep handle tracked; Poll until transfer completes or release succeeds; Avoid leaking active handles during shutdown.

If metadata is invalidated while transfers are active:
- Define engine policy; Prefer draining transfers before planned scale-in; For failure, mark affected transfers failed.

## Dynamic Scaling

Adding a NIXL agent:
1. Create process and NIXL agent 2. Instantiate required backends 3. Register memory 4. Publish metadata 5. Scheduler marks worker available 6. Peers fetch metadata when needed.

Removing a NIXL agent gracefully:
1. Stop assigning new transfers to the worker 2. Wait for active transfers to finish or cancel them 3. Invalidate local metadata 4. Ask peers to remove remote metadata 5. Deregister memory 6. Destroy agent 7. Exit process.

Handling failed agent:
1. Heartbeat detects failure 2. Scheduler marks worker unavailable 3. Peers call `remove_remote_agent` 4. Affected transfer handles are failed or canceled 5. Requests are retried or failed according to engine policy 6. Cache ownership map is repaired.

NIXL metadata caching is built for dynamicity. Loading remote metadata does not force immediate connection. Invalidating metadata disconnects if connected. This lets an inference engine prefetch metadata for likely peers without opening all connections immediately.

## Performance Guidance

Register memory once. Avoid per-request registration. Use larger contiguous registrations. Use transfer descriptors for subranges. Coalesce adjacent transfer descriptors where possible. Use prepared descriptor lists when transferring repeated subsets. Use `initialize_xfer` when descriptor locations are highly dynamic. Use backend hints for deterministic backend choice. Use GPU memory descriptors directly for GPU-to-GPU paths. Avoid staging through CPU unless required. Use GDS/GDS_MT for GPU-direct storage if hardware and driver stack support it. Validate GDS mode and performance outside NIXL too. Use progress thread if backend and workload benefit. Use listen thread for direct metadata helper exchange. Use ETCD or scheduler side channel for larger deployments. Poll efficiently. Do not busy-spin in production unless latency budget requires it and CPU budget allows it. Integrate polling with engine event loop. Batch transfer status checks when possible. Use telemetry to identify transfer sizes, descriptor counts, and durations. Benchmark with realistic KV block sizes. Benchmark at realistic concurrency. Benchmark across actual network topology. Benchmark same-node, cross-node, and storage-tier paths separately. Be careful with rail tuning. `UCX_MAX_RMA_RAILS=1` can help some remote VRAM-to-DRAM cases but may hurt larger transfers. For remote storage, pipeline network and storage when correctness allows. For repeated transfers over same buffers, repost handles after completion. For first-request latency, consider proactive `make_connection`. For large worker pools, avoid full mesh preconnect.

## Safety and Correctness Rules

Do not transfer into a buffer that compute is currently writing. Do not compute from a buffer whose incoming transfer is not complete. Do not free a buffer while a transfer may reference it. Do not deregister memory while transfers reference it. Do not reuse a KV block for another request before transfer completion. Do not assume transfer ordering across handles. Do not post the same handle twice concurrently. Do not use descriptors outside registered ranges. Do not load untrusted pickled descriptors. Do not assume remote metadata means remote process is still alive. Do not assume NIXL notifications replace scheduler acknowledgements. Do not assume storage operations are remote just because storage is physically remote. From NIXL perspective, storage backend access is often local to a client library in the process. Do not assume default backend selection is optimal. Do not assume all plugins are installed. Do not assume macOS or Windows support.

## Teardown Checklist

For each transfer handle:
- Wait for completion or decide to cancel; Call `release_xfer_handle`.

For each prepared descriptor-list handle:
- Call `release_dlist_handle`.

For each remote agent:
- Call `remove_remote_agent`.

For local metadata publication:
- Call `invalidate_local_metadata` if using built-in direct or central metadata mechanisms.

For each registration descriptor list:
- Call `deregister_memory`.

For raw memory:
- Free after deregistration.

For file descriptors:
- Deregister file descriptors before close; Close file descriptors after deregistration.

For PyTorch tensors:
- Let tensors be freed only after deregistration and transfer completion.

For agent:
- Let object destruct after cleanup.

Python destructor has best-effort cleanup, but production code should not depend on it.

## NIXL in a Worker Class

A practical engine wrapper might look like this:
```python
class NixlTransferManager:
    def __init__(self, agent_name, listen_port, backends=("UCX",)):
        self.config = nixl_agent_config(
            enable_prog_thread=True,
            enable_listen_thread=True,
            listen_port=listen_port,
            backends=list(backends),
        )
        self.agent = nixl_agent(agent_name, self.config)
        self.registrations = {}
        self.active = {}

    def register_pool(self, pool_name, tensor):
        reg = self.agent.register_memory(tensor)
        self.registrations[pool_name] = reg
        return reg

    def metadata(self):
        return self.agent.get_agent_metadata()

    def add_peer_metadata(self, metadata):
        return self.agent.add_remote_agent(metadata)

    def read_blocks(self, request_id, local_descs, remote_descs, remote_agent):
        handle = self.agent.initialize_xfer(
            "READ",
            local_descs,
            remote_descs,
            remote_agent,
            request_id.encode(),
        )
        state = self.agent.transfer(handle)
        if state == "ERR":
            self.agent.release_xfer_handle(handle)
            raise RuntimeError("failed to post NIXL transfer")
        self.active[request_id] = handle

    def poll(self):
        done = []
        for request_id, handle in list(self.active.items()):
            state = self.agent.check_xfer_state(handle)
            if state == "DONE":
                self.agent.release_xfer_handle(handle)
                done.append(request_id)
                del self.active[request_id]
            elif state == "ERR":
                self.agent.release_xfer_handle(handle)
                del self.active[request_id]
                raise RuntimeError(f"NIXL transfer failed: {request_id}")
        return done
```

Real production code should avoid raising from `poll` if it would break other active requests. Return structured completion and failure events instead.

## Backend-Aware Storage Manager Pattern

A storage wrapper should instantiate storage and network backends explicitly. Example backend selection:
```python
agent = nixl_agent(name, nixl_agent_config(True, True, port, backends=[]))
plugins = agent.get_plugin_list()

if "GDS_MT" in plugins:
    agent.create_backend("GDS_MT")
elif "GDS" in plugins:
    agent.create_backend("GDS")
elif "POSIX" in plugins:
    agent.create_backend("POSIX")
else:
    raise RuntimeError("no storage backend available")

if "UCX" not in plugins:
    raise RuntimeError("UCX required for remote storage")
agent.create_backend("UCX")
```

Use backend hints on local storage transfers:
```python
handle = agent.initialize_xfer(
    "WRITE",
    memory_descs,
    file_descs,
    agent.name,
    backends=["GDS_MT"],
)
```

Fallback to `POSIX` if GDS is missing or unsuitable. Do not hint `GDS_MT` when using DRAM buffers if your configured backend expects VRAM or specific alignment. Check plugin memory support with `get_backend_mem_types`.

## Testing Strategy

Start with same-host CPU test:
```bash
python3 examples/python/basic_two_peers.py --mode target --ip 127.0.0.1 --port 5555
python3 examples/python/basic_two_peers.py --mode initiator --ip 127.0.0.1 --port 5555
```

Then same-host CUDA test:
```bash
python3 examples/python/basic_two_peers.py --mode target --ip 127.0.0.1 --port 5555 --use_cuda True
python3 examples/python/basic_two_peers.py --mode initiator --ip 127.0.0.1 --port 5555 --use_cuda True
```

Then expanded transfer test:
```bash
python3 examples/python/expanded_two_peers.py --mode target --ip 127.0.0.1 --port 5555 --backend UCX
python3 examples/python/expanded_two_peers.py --mode initiator --ip 127.0.0.1 --port 5555 --backend UCX
```

Then cross-host test using actual IP addresses. Then storage test with GDS/POSIX. Then partial metadata test. Then engine-specific KV-cache transfer test. Engine-specific tests should cover:
- Single block transfer; Multiple block transfer; Sparse block transfer; Coalesced contiguous transfer; Cross-GPU transfer; Cross-node transfer; Transfer cancellation or worker failure; Metadata invalidation; Re-registration after pool recreation; Concurrent transfers; Reposting completed handles; Missing metadata error path; Backend unavailable path; Descriptor outside registration path.

Verification:
- Fill source with deterministic pattern; Fill destination with sentinel; Transfer; Synchronize GPU if needed for verification; Compare exact bytes or tensor values; Validate destination only after transfer completion.

For GPU tensors, ensure compute streams and transfer visibility are handled according to backend and engine requirements. If the backend does not synchronize with a specific CUDA stream, add appropriate engine-side stream/event synchronization. The provided examples use simple blocking polling and PyTorch verification. Production engines must integrate with their CUDA stream model.

## Observability

Use logging around:
- Agent creation; Plugin list; Backend creation; Backend memory types; Registration counts and byte sizes; Metadata publication and fetch; Transfer creation; Transfer post; Transfer completion; Transfer failures; Remote metadata invalidation.

Use telemetry where available:
```python
config = nixl_agent_config(capture_telemetry=True)
```

Transfer telemetry fields:
- `startTime`; `postDuration`; `xferDuration`; `totalBytes`; `descCount`.

Track metrics per backend:
- bytes transferred; transfers posted; transfers completed; transfer failures; transfer latency; post latency; descriptor count; metadata size; metadata fetch latency; active handles; registration count.

For LLM serving, correlate transfer metrics with:
- time to first token; inter-token latency; prefill latency; decode latency; queueing time; KV-cache hit/miss; cache spill/restore time.

## Common Troubleshooting

No plugins available:
- Confirm NIXL installation; Confirm plugin build options; Confirm `NIXL_PLUGIN_DIR` if using nonstandard plugin location; Confirm Linux runtime.

Backend missing:
- Check `agent.get_plugin_list()`; Rebuild with plugin enabled; Install backend dependencies.

UCX transfer fails:
- Confirm UCX plugin exists; Confirm remote metadata was exchanged after registration; Confirm network reachability; Confirm GPU-direct prerequisites if using VRAM; Confirm firewall/ports for listener metadata path.

Metadata not ready:
- Confirm target listen thread is enabled for direct helper path; Confirm correct IP and port; Confirm agent names match; Confirm metadata was sent after memory registration; For ETCD, confirm `NIXL_ETCD_ENDPOINTS`; For partial metadata, confirm descriptor list matches metadata sent.

Transfer creation not found:
- Descriptor outside registered range; Metadata missing for descriptor; Wrong remote agent name; Backend hint excludes valid backend; Memory type mismatch; Remote registration occurred after metadata was generated and metadata was not refreshed.

Transfer stuck in progress:
- Backend progress thread may not be enabled; Application may need to call status checks to progress backend; Network path may be stalled; Remote process may be failed; Transfer may be too large for current timeout expectations.

Notification missing:
- Backend may not support notifications; Remote metadata may be missing; Notification may have been consumed by previous call; Matching tag may be wrong; Application may be polling wrong agent name.

GDS performance poor:
- Confirm GDS plugin is used, not POSIX fallback; Confirm GPUDirect Storage configuration; Confirm no compatibility mode if true direct I/O expected; Confirm file alignment and `O_DIRECT` usage where required; Confirm storage hardware throughput independently.

Python descriptor construction fails:
- Tensor is non-contiguous; List mixes CPU and CUDA tensors; Tuple length wrong; `mem_type` missing for raw tuple descriptors; NumPy array not `Nx3`; NumPy array not C-contiguous; NumPy dtype not `uint64` or `int64`.

## Practical Decision Tables

Use `initialize_xfer` when:
- Descriptor locations are chosen at request time; Simplicity matters more than repeated preparation overhead; Transfers are not repeated over the same descriptor sets.

Use `prep_xfer_dlist` and `make_prepped_xfer` when:
- Descriptor lists are stable; Many transfers select different subsets; You need repeated transfers over fixed block tables; You want to repost handles efficiently.

Use full metadata when:
- Registered memory set is small; Cluster is small; Simplicity matters; All peers can access all registered regions.

Use partial metadata when:
- Registered memory set is large; KV blocks are dynamic; Access should be scoped; Control-plane payload size matters.

Use direct metadata helpers when:
- Running examples; Small deployments; Simple peer-to-peer testing; A listener thread is acceptable.

Use ETCD when:
- Deployment already has ETCD; Peer discovery is dynamic; Centralized metadata is desired; You need labeled partial metadata fetches.

Use scheduler side channel when:
- Engine already has robust RPC; You need authentication and authorization; You want scheduler-controlled metadata lifecycle; You want to avoid extra runtime services.

Use UCX when:
- Moving DRAM/VRAM between processes or nodes; Need remote agent communication; Need notifications.

Use GDS/GDS_MT when:
- Moving between GPU/CPU memory and files through GPUDirect Storage; Storage stack supports it; Performance justifies setup complexity.

Use POSIX when:
- Need standard file operations; GDS unavailable; CPU/file path is acceptable.

## Implementation Checklist for an LLM Engine

Initial code integration:
- Add NIXL dependency; Add config flags for enabling NIXL; Add backend selection config; Add metadata transport choice; Add agent naming scheme; Add per-worker transfer manager; Add memory pool registration; Add descriptor builder for KV-cache blocks; Add metadata publish/fetch; Add transfer submit and poll; Add cleanup.

Correctness integration:
- Track active transfers by request; Pin source and destination blocks until completion; Block decode until required transfers complete; Handle transfer failure; Handle metadata invalidation; Handle worker removal; Handle backend unavailable.

Performance integration:
- Register pools once; Coalesce descriptors; Use prepared descriptor handles where stable; Use backend hints; Add transfer telemetry; Add benchmarks for KV block sizes; Add concurrency tests.

Operational integration:
- Log plugin/backend state; Expose metrics; Add health checks; Add metadata cleanup on shutdown; Add scale-in drain path; Add failure tests.

Security integration:
- Authenticate metadata exchange; Do not accept untrusted pickles; Scope partial metadata if necessary; Avoid exposing all worker memory unnecessarily; Validate descriptor ownership in scheduler.

## Minimal Production-Style Transfer Flow

The following is a concise pattern for decode-pulls-KV-from-prefill. Prefill worker:
```python
prefill_agent = nixl_agent(prefill_name, prefill_config)
kv_reg = prefill_agent.register_memory(kv_pool_tensor)
prefill_metadata = prefill_agent.get_agent_metadata()
scheduler.publish_metadata(prefill_name, prefill_metadata)
```

Decode worker:
```python
decode_agent = nixl_agent(decode_name, decode_config)
decode_reg = decode_agent.register_memory(decode_kv_pool_tensor)

remote_md = scheduler.get_metadata(prefill_name)
decode_agent.add_remote_agent(remote_md)

local_descs = decode_descriptor_builder.destination_descs(request_id, block_ids)
remote_descs = scheduler.get_prefill_descriptors(request_id, block_ids)

handle = decode_agent.initialize_xfer(
    "READ",
    local_descs,
    remote_descs,
    prefill_name,
    request_id.encode(),
    backends=["UCX"],
)

state = decode_agent.transfer(handle)
if state == "ERR":
    raise RuntimeError("failed to post KV transfer")

while True:
    state = decode_agent.check_xfer_state(handle)
    if state == "DONE":
        break
    if state == "ERR":
        raise RuntimeError("KV transfer failed")

decode_agent.release_xfer_handle(handle)
mark_kv_blocks_ready(request_id)
```

In a real engine, replace the busy loop with event-loop polling.

## Version and Compatibility Notes

The guide reflects the repository state read from the local workspace. NIXL APIs may evolve. When updating integration code:
- Check `src/api/python/_api.py`; Check `src/api/cpp/nixl.h`; Check examples under `examples/python`; Check plugin README files for backend-specific behavior.

If a method exists in C++ but not Python, the pybind wrapper may not expose it. If a backend exists in code but not in `get_plugin_list()`, it may not be built or discoverable. If a Python doc references an example file that does not exist, inspect `examples/python/README.md` and the actual files in the directory. In this workspace, `docs/python_api.md` mentions `nixl_api_example.py`, but `rg --files examples/python` did not show that file. Use existing examples as source of truth for runnable workflows.

## Final Rules of Thumb

Think of NIXL as an efficient transfer substrate, not a scheduler. Keep metadata on the control path. Keep payloads on the NIXL data path. Register long-lived memory. Transfer short-lived subranges. Make agent names stable. Make descriptor ownership explicit. Use `READ` when the initiator pulls. Use `WRITE` when the initiator pushes. Use partial metadata for dynamic KV pages. Use prepared descriptor handles for repeated block sets. Use `initialize_xfer` for dynamic one-off offsets. Use notifications as local completion signals, not as the only source of global truth. Always handle missing metadata. Always handle transfer errors. Always release handles. Always deregister memory before freeing it. Benchmark the exact backend, topology, memory type, and block sizes that your inference engine uses.
