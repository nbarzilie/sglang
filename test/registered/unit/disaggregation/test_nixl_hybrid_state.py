"""Unit tests for NIXL hybrid-state transfer dispatch."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class FakeAgent:
    def __init__(self):
        self.get_xfer_descs_calls = []
        self.initialize_xfer_calls = []
        self.transfer_calls = []

    def get_xfer_descs(self, reqs, mem_type):
        self.get_xfer_descs_calls.append((reqs, mem_type))
        return f"{mem_type}_{len(self.get_xfer_descs_calls)}"

    def initialize_xfer(self, *args):
        self.initialize_xfer_calls.append(args)
        return "handle"

    def transfer(self, handle):
        self.transfer_calls.append(handle)
        return "DONE"


class TestNixlHybridState(CustomTestCase):
    def _make_manager(self, **kwargs):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = kwargs.pop("agent", FakeAgent())
        mgr.attn_tp_size = kwargs.pop("attn_tp_size", 1)
        mgr.is_mla_backend = kwargs.pop("is_mla_backend", False)
        mgr.kv_args = SimpleNamespace(
            engine_rank=kwargs.pop("engine_rank", 0),
            gpu_id=kwargs.pop("gpu_id", 2),
            state_types=kwargs.pop("state_types", []),
            state_data_ptrs=kwargs.pop("state_data_ptrs", []),
            state_item_lens=kwargs.pop("state_item_lens", []),
            state_dim_per_tensor=kwargs.pop("state_dim_per_tensor", []),
            kv_data_ptrs=kwargs.pop("kv_data_ptrs", []),
            kv_item_lens=kwargs.pop("kv_item_lens", []),
            prefill_start_layer=0,
            prefill_end_layer=None,
            mla_compression_ratios=None,
        )
        return mgr

    def test_maybe_send_extra_sends_nothing_for_empty_indices(self):
        mgr = self._make_manager(
            state_types=[StateType.MAMBA],
            state_data_ptrs=[[0x1000]],
            state_item_lens=[[64]],
        )

        self.assertEqual(
            mgr.maybe_send_extra(
                "peer", [[]], [[0x2000]], [[]], 3, "room_state", decode_tp_size=1
            ),
            [],
        )

    def test_mamba_homogeneous_tp_dispatches_to_mamba_transfer(self):
        mgr = self._make_manager(
            state_types=[StateType.MAMBA],
            state_data_ptrs=[[0x1000]],
            state_item_lens=[[64]],
            attn_tp_size=1,
        )
        calls = []
        mgr._send_mamba_state = lambda *args: calls.append(args) or "handle"

        handles = mgr.maybe_send_extra(
            "peer", [[2]], [[0x2000]], [[3]], 4, "room_state", decode_tp_size=1
        )

        self.assertEqual(handles, ["handle"])
        self.assertEqual(calls[0][0], "peer")
        self.assertEqual(calls[0][-1], "room_state_0")

    def test_mamba_heterogeneous_tp_dispatches_to_slice_transfer(self):
        mgr = self._make_manager(
            state_types=[StateType.MAMBA],
            state_data_ptrs=[[0x1000]],
            state_item_lens=[[64]],
            state_dim_per_tensor=[[4]],
            attn_tp_size=2,
        )
        calls = []
        mgr._send_mamba_state_slice = lambda *args: calls.append(args) or "handle"

        handles = mgr.maybe_send_extra(
            "peer",
            [[2]],
            [[0x2000]],
            [[3]],
            4,
            "room_state",
            decode_tp_size=1,
            decode_tp_rank=0,
            dst_state_item_lens=[[128]],
            dst_state_dim_per_tensor=[[8]],
        )

        self.assertEqual(handles, ["handle"])
        self.assertEqual(calls[0][0], "peer")
        self.assertEqual(calls[0][-3:], ("room_state_0", 1, 0))

    @patch("sglang.srt.disaggregation.nixl.conn.logger.warning_once", create=True)
    def test_send_mamba_state_slice_builds_expected_descriptor_offsets(self, _):
        agent = FakeAgent()
        mgr = self._make_manager(
            agent=agent,
            attn_tp_size=2,
            engine_rank=1,
            gpu_id=5,
        )

        handle = mgr._send_mamba_state_slice(
            "peer",
            [2],
            [0x1000],
            [64],
            [4],
            [0x2000],
            [3],
            [128],
            [8],
            6,
            "room_state_0",
            decode_tp_size=1,
            decode_tp_rank=0,
        )

        self.assertEqual(handle, "handle")
        self.assertEqual(agent.get_xfer_descs_calls[0], ([(0x1080, 64, 5)], "VRAM"))
        self.assertEqual(agent.get_xfer_descs_calls[1], ([(0x21C0, 64, 6)], "VRAM"))
        self.assertEqual(agent.initialize_xfer_calls[0][0], "WRITE")
        self.assertEqual(agent.initialize_xfer_calls[0][-1], b"room_state_0")

    @patch("sglang.srt.disaggregation.nixl.conn.logger.warning_once", create=True)
    def test_mamba_slice_falls_back_when_dim_metadata_missing(self, _):
        mgr = self._make_manager(attn_tp_size=2)
        calls = []
        mgr._send_mamba_state = lambda *args: calls.append(args) or "fallback"

        handle = mgr._send_mamba_state_slice(
            "peer",
            [2],
            [0x1000],
            [64],
            [],
            [0x2000],
            [3],
            [],
            [],
            6,
            "room_state_0",
            decode_tp_size=1,
            decode_tp_rank=0,
        )

        self.assertEqual(handle, "fallback")
        self.assertEqual(calls[0][0], "peer")

    def test_swa_state_index_length_mismatch_raises(self):
        mgr = self._make_manager(
            state_types=[StateType.SWA],
            state_data_ptrs=[[0x1000]],
            state_item_lens=[[64]],
            is_mla_backend=True,
        )

        with self.assertRaisesRegex(RuntimeError, "State index length mismatch"):
            mgr.maybe_send_extra(
                "peer",
                [[1, 2]],
                [[0x2000]],
                [[3]],
                4,
                "room_state",
                decode_tp_size=1,
            )

    def test_unknown_state_type_raises_actionable_error(self):
        mgr = self._make_manager(
            state_types=["unknown"],
            state_data_ptrs=[[0x1000]],
            state_item_lens=[[64]],
        )

        with self.assertRaisesRegex(RuntimeError, "does NOT support unknown"):
            mgr.maybe_send_extra(
                "peer",
                [[1]],
                [[0x2000]],
                [[3]],
                4,
                "room_state",
                decode_tp_size=1,
            )


if __name__ == "__main__":
    unittest.main()
