"""Unit tests for NIXL backend configuration validation."""

import builtins
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.disaggregation.nixl.conn import (
    load_nixl_agent_classes,
    parse_nixl_backend_params,
    validate_nixl_backend_available,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNixlBackendConfig(CustomTestCase):
    def test_backend_params_accept_json_object_with_string_values(self):
        self.assertEqual(
            parse_nixl_backend_params('{"num_threads": "4", "rail": "0"}'),
            {"num_threads": "4", "rail": "0"},
        )

    def test_backend_params_reject_non_object_json(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            parse_nixl_backend_params("[]")

    def test_backend_params_reject_non_string_keys_or_values(self):
        with self.assertRaisesRegex(ValueError, "string keys and string values"):
            parse_nixl_backend_params('{"num_threads": 4}')

    def test_backend_params_reject_invalid_json(self):
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            parse_nixl_backend_params("{")

    def test_missing_nixl_import_has_install_guidance(self):
        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "nixl._api":
                raise ImportError("missing nixl")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(ImportError, "Please install NIXL"):
                load_nixl_agent_classes()

    def test_missing_requested_plugin_reports_available_plugins(self):
        agent = MagicMock()
        agent.get_plugin_list.return_value = ["UCX", "GDS_MT"]

        with self.assertRaisesRegex(ValueError, "LIBFABRIC.*UCX.*GDS_MT"):
            validate_nixl_backend_available(agent, "LIBFABRIC")

    def test_requested_plugin_returns_available_plugin_list(self):
        agent = MagicMock()
        agent.get_plugin_list.return_value = ["UCX"]

        self.assertEqual(validate_nixl_backend_available(agent, "UCX"), ["UCX"])


if __name__ == "__main__":
    unittest.main()
