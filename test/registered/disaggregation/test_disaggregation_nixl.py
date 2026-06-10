import json
import os
import unittest
import uuid

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.server_fixtures.disaggregation_utils import (
    assert_process_healthy,
    configure_nixl_pd_backend,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, is_in_ci

register_cuda_ci(est_time=440, stage="base-b", runner_config="2-gpu-large")


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


def _get_configured_nixl_backend_probe_error():
    backend = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND.get()
    backend_params_json = envs.SGLANG_DISAGGREGATION_NIXL_BACKEND_PARAMS.get()

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


def _has_configured_nixl_backend():
    return _get_configured_nixl_backend_probe_error() is None


def _require_configured_nixl_backend():
    error = _get_configured_nixl_backend_probe_error()
    if error is not None:
        raise RuntimeError(error)


_HAS_CONFIGURED_NIXL_BACKEND = is_in_ci() or _has_configured_nixl_backend()


@unittest.skipUnless(
    _HAS_CONFIGURED_NIXL_BACKEND,
    "NIXL with the configured backend is required for this test.",
)
class TestDisaggregationNixlBasic(PDDisaggregationServerBase):
    """Small NIXL PD E2E coverage.

    Mooncake already owns the broad disaggregation functional matrix in
    test_disaggregation_basic.py. This class intentionally mirrors only the
    subset that proves NIXL can launch, transfer KV, serve a request, return
    logprobs, and keep all workers alive.
    """

    @classmethod
    def setUpClass(cls):
        _require_configured_nixl_backend()
        super().setUpClass()
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        configure_nixl_pd_backend(cls)
        cls.launch_all()

    def test_completion_returns_text_and_workers_stay_alive(self):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)

        data = response.json()
        self.assertIn("text", data, f"Unexpected response shape: {data}")
        self.assertGreater(len(data["text"]), 0, "Generated text should not be empty")

        assert_process_healthy(self, "load balancer", self.process_lb, self.lb_url)
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)

    def test_logprob(self):
        response = requests.post(
            self.lb_url + "/generate",
            json={
                "text": "The capital of france is ",
                "sampling_params": {"temperature": 0},
                "return_logprob": True,
                "return_input_logprob": True,
                "logprob_start_len": 0,
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, response.text)

        meta_info = response.json()["meta_info"]
        completion_tokens = meta_info["completion_tokens"]
        input_logprobs = meta_info["input_token_logprobs"]
        output_logprobs = meta_info["output_token_logprobs"]

        self.assertEqual(len(output_logprobs), completion_tokens)
        self.assertGreater(len(input_logprobs), 0)


@unittest.skipUnless(
    _HAS_CONFIGURED_NIXL_BACKEND,
    "NIXL with the configured backend is required for this test.",
)
class TestDisaggregationNixlFailure(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        _require_configured_nixl_backend()
        super().setUpClass()
        os.environ["SGLANG_TEST_DISAGG_FAILURE_PROB"] = "0.05"
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        configure_nixl_pd_backend(cls)
        cls.launch_all()

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("SGLANG_TEST_DISAGG_FAILURE_PROB", None)
        super().tearDownClass()

    def test_injected_transfer_failures_do_not_crash_workers(self):
        failure_count = 0

        for i in range(24):
            try:
                response = requests.post(
                    self.lb_url + "/generate",
                    json={
                        "text": f"Failure injection request {i}: 1 + 1 =",
                        "sampling_params": {"temperature": 0, "max_new_tokens": 4},
                    },
                    timeout=30,
                )
            except requests.RequestException:
                failure_count += 1
                continue

            if response.status_code != 200:
                failure_count += 1
                continue

            try:
                data = response.json()
            except ValueError:
                failure_count += 1
                continue

            if "text" not in data or len(data["text"]) == 0:
                failure_count += 1

        self.assertGreater(
            failure_count,
            0,
            "failure injection did not produce any failed requests or exceptions",
        )

        # Cleanup is part of the correctness check: after injected NIXL
        # transfer failures, workers must still accept abort requests before
        # the final health checks.
        for url in (self.prefill_url, self.decode_url):
            response = requests.post(
                f"{url}/abort_request",
                json={"abort_all": True},
                timeout=10,
            )
            self.assertEqual(response.status_code, 200, response.text)

        assert_process_healthy(self, "load balancer", self.process_lb, self.lb_url)
        assert_process_healthy(self, "prefill", self.process_prefill, self.prefill_url)
        assert_process_healthy(self, "decode", self.process_decode, self.decode_url)


if __name__ == "__main__":
    unittest.main()
