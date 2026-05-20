import unittest

import requests

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_pd_server,
    try_cached_model,
)

register_cuda_ci(est_time=300, stage="base-c", runner_config="8-gpu-h20")


def _has_nixl():
    try:
        import nixl._api  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_has_nixl(), "NIXL is required for this test.")
class TestDisaggregationNixlDPAttention(PDDisaggregationServerBase):
    DP_SIZE = 2
    LOAD_BALANCE_METHOD = "auto"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        envs.SGLANG_ENABLE_JIT_DEEPGEMM.set(False)
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST_MLA)
        cls.transfer_backend = ["--disaggregation-transfer-backend", "nixl"]
        cls.rdma_devices = []
        cls.start_prefill()
        cls.start_decode()
        cls.wait_server_ready(cls.prefill_url + "/health", process=cls.process_prefill)
        cls.wait_server_ready(cls.decode_url + "/health", process=cls.process_decode)
        cls.launch_lb()

    @classmethod
    def start_prefill(cls):
        prefill_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "prefill",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            str(cls.DP_SIZE),
            "--dp",
            str(cls.DP_SIZE),
            "--enable-dp-attention",
            "--load-balance-method",
            cls.LOAD_BALANCE_METHOD,
        ]
        prefill_args += cls.transfer_backend + cls.rdma_devices
        cls.process_prefill = popen_launch_pd_server(
            cls.model,
            cls.prefill_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=prefill_args,
        )

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--tp",
            str(cls.DP_SIZE),
            "--dp",
            str(cls.DP_SIZE),
            "--enable-dp-attention",
            "--base-gpu-id",
            str(cls.DP_SIZE),
            "--load-balance-method",
            cls.LOAD_BALANCE_METHOD,
        ]
        decode_args += cls.transfer_backend + cls.rdma_devices
        cls.process_decode = popen_launch_pd_server(
            cls.model,
            cls.decode_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=decode_args,
        )

    def _assert_process_healthy(self, name, process, url):
        self.assertIsNotNone(process, f"{name} process was not started")
        self.assertIsNone(
            process.poll(),
            f"{name} exited unexpectedly with code {process.returncode}",
        )
        response = requests.get(f"{url}/health", timeout=10)
        self.assertEqual(response.status_code, 200, response.text)

    def test_generate_with_dp_attention(self):
        response = requests.post(
            self.base_url + "/generate",
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

        self._assert_process_healthy("load balancer", self.process_lb, self.lb_url)
        self._assert_process_healthy("prefill", self.process_prefill, self.prefill_url)
        self._assert_process_healthy("decode", self.process_decode, self.decode_url)


if __name__ == "__main__":
    unittest.main()
