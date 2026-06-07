import unittest

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.server_fixtures.disaggregation_utils import (
    assert_process_healthy,
    configure_nixl_pd_backend,
    has_configured_nixl_backend,
    require_configured_nixl_backend,
)
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, is_in_ci

register_cuda_ci(est_time=220, stage="base-b", runner_config="2-gpu-large")


@unittest.skipUnless(
    is_in_ci() or has_configured_nixl_backend(),
    "NIXL with the configured backend is required for this test.",
)
class TestDisaggregationNixlBasic(PDDisaggregationServerBase):
    """Small NIXL PD E2E coverage.

    Mooncake already owns the broad disaggregation functional matrix in
    test_disaggregation_basic.py. This file intentionally mirrors only the
    subset that proves NIXL can launch, transfer KV, serve a request, return
    logprobs, and keep all workers alive.
    """

    @classmethod
    def setUpClass(cls):
        require_configured_nixl_backend()
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


if __name__ == "__main__":
    unittest.main()
