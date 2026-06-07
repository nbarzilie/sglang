import json
import os
import uuid

import requests


def _nixl_backend_config(backend, backend_params_json):
    backend_params = json.loads(backend_params_json)
    if not isinstance(backend_params, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in backend_params.items()
    ):
        raise ValueError(
            "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS must be a JSON object "
            "with string keys and string values"
        )

    if backend == "UCX" or backend == "OBJ":
        backend_params.setdefault("num_threads", "8")
    elif backend == "GDS_MT":
        backend_params.setdefault("thread_count", "8")
    elif backend == "UCCL":
        backend_params.setdefault("num_cpus", "8")

    return backend, backend_params


def get_configured_nixl_backend_probe_error(
    backend=None,
    backend_params_json=None,
):
    backend = backend or os.getenv("SGLANG_DISAGGREGATION_NIXL_BACKEND", "UCX")
    backend_params_json = backend_params_json or os.getenv(
        "SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS", "{}"
    )

    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as e:
        return f"NIXL import failed: {e}"

    try:
        backend, backend_params = _nixl_backend_config(backend, backend_params_json)
    except (json.JSONDecodeError, ValueError) as e:
        return str(e)

    try:
        agent_config = nixl_agent_config(backends=[], num_threads=8)
        agent = nixl_agent(f"sglang_nixl_probe_{uuid.uuid4()}", agent_config)
        available_plugins = agent.get_plugin_list()
        if backend not in available_plugins:
            return (
                f"NIXL backend {backend!r} not found. "
                f"Available plugins: {available_plugins}."
            )
        agent.create_backend(backend, backend_params)
    except Exception as e:
        return f"NIXL backend probe failed: {e}"

    return None


def has_configured_nixl_backend():
    return get_configured_nixl_backend_probe_error() is None


def require_configured_nixl_backend():
    error = get_configured_nixl_backend_probe_error()
    if error is not None:
        raise RuntimeError(error)


def configure_nixl_pd_backend(test_cls):
    test_cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
    # NIXL backend/network selection is driven by NIXL environment variables
    # such as SGLANG_DISAGGREGATION_NIXL_BACKEND and backend params, not by the
    # Mooncake-specific --disaggregation-ib-device argument.
    test_cls.rdma_devices = []


def assert_process_healthy(test_case, name, process, url, health_path="/health"):
    test_case.assertIsNotNone(process, f"{name} process was not started")
    test_case.assertIsNone(
        process.poll(),
        f"{name} exited unexpectedly with code {process.returncode}",
    )
    response = requests.get(f"{url}{health_path}", timeout=10)
    test_case.assertEqual(response.status_code, 200, response.text)
