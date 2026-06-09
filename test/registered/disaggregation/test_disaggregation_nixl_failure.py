import os
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
class TestDisaggregationNixlFailure(PDDisaggregationServerBase):
    @classmethod
    def setUpClass(cls):
        require_configured_nixl_backend()
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
        success_count = 0
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

            if "text" in data and len(data["text"]) > 0:
                success_count += 1
            else:
                failure_count += 1

        self.assertGreater(
            success_count,
            0,
            "expected at least one request to complete during NIXL failure injection",
        )
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
