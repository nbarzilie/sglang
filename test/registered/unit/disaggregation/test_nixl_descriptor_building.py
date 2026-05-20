"""Unit tests for NIXL registration and transfer descriptor construction."""

import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.disaggregation.nixl.conn import NixlKVManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class FakeAgent:
    def __init__(self, registration_results=None):
        self.registration_results = list(registration_results or [])
        self.register_memory_calls = []
        self.get_xfer_descs_calls = []
        self.initialize_xfer_calls = []
        self.transfer_calls = []

    def register_memory(self, addrs, mem_type):
        self.register_memory_calls.append((addrs, mem_type))
        if self.registration_results:
            return self.registration_results.pop(0)
        return ["desc"]

    def get_xfer_descs(self, reqs, mem_type):
        self.get_xfer_descs_calls.append((reqs, mem_type))
        return f"{mem_type}_descs_{len(self.get_xfer_descs_calls)}"

    def initialize_xfer(self, *args):
        self.initialize_xfer_calls.append(args)
        return f"handle_{len(self.initialize_xfer_calls)}"

    def transfer(self, handle):
        self.transfer_calls.append(handle)
        return "DONE"


class TestNixlDescriptorBuilding(CustomTestCase):
    def _make_manager(self, agent=None, **kv_overrides):
        mgr = object.__new__(NixlKVManager)
        mgr.agent = agent or FakeAgent()
        mgr.attn_tp_size = kv_overrides.pop("attn_tp_size", 1)
        mgr.is_mla_backend = kv_overrides.pop("is_mla_backend", False)
        defaults = dict(
            engine_rank=0,
            gpu_id=2,
            kv_data_ptrs=[0x1000, 0x2000],
            kv_data_lens=[4096, 4096],
            kv_item_lens=[64],
            aux_data_ptrs=[0x3000],
            aux_data_lens=[512],
            aux_item_lens=[32],
            state_types=[],
            state_data_ptrs=[],
            state_data_lens=[],
            state_item_lens=[],
            state_dim_per_tensor=[],
            page_size=1,
            kv_head_num=1,
            total_kv_head_num=1,
            prefill_start_layer=0,
            prefill_end_layer=None,
            mla_compression_ratios=None,
        )
        defaults.update(kv_overrides)
        mgr.kv_args = SimpleNamespace(**defaults)
        return mgr

    def test_register_buffer_to_engine_registers_kv_aux_and_state_memory(self):
        agent = FakeAgent(registration_results=[["kv"], ["aux"], ["state"]])
        mgr = self._make_manager(
            agent=agent,
            state_data_ptrs=[[0x4000, 0], [0x5000]],
            state_data_lens=[[128, 0], [256]],
        )

        mgr.register_buffer_to_engine()

        self.assertEqual(
            agent.register_memory_calls,
            [
                ([(0x1000, 4096, 2, ""), (0x2000, 4096, 2, "")], "VRAM"),
                ([(0x3000, 512, 0, "")], "DRAM"),
                ([(0x4000, 128, 2, ""), (0x5000, 256, 2, "")], "VRAM"),
            ],
        )
        self.assertEqual(mgr.kv_descs, ["kv"])
        self.assertEqual(mgr.aux_descs, ["aux"])
        self.assertEqual(mgr.state_descs, ["state"])

    def test_register_buffer_to_engine_fails_on_empty_kv_registration(self):
        mgr = self._make_manager(agent=FakeAgent(registration_results=[[]]))

        with self.assertRaisesRegex(Exception, "kv tensors"):
            mgr.register_buffer_to_engine()

    def test_send_aux_uses_dram_descriptors_and_write_notification(self):
        agent = FakeAgent()
        mgr = self._make_manager(agent=agent)

        handle = mgr.send_aux("peer", 2, [0x8000], 5, "12_aux")

        self.assertEqual(handle, "handle_1")
        self.assertEqual(agent.get_xfer_descs_calls[0], ([(0x3040, 32, 0)], "DRAM"))
        self.assertEqual(agent.get_xfer_descs_calls[1], ([(0x80A0, 32, 0)], "DRAM"))
        self.assertEqual(
            agent.initialize_xfer_calls[0],
            ("WRITE", "DRAM_descs_1", "DRAM_descs_2", "peer", b"12_aux"),
        )
        self.assertEqual(agent.transfer_calls, ["handle_1"])

    def test_send_kvcache_generic_uses_vram_and_unsigned_pointer_math(self):
        high_ptr = 0xFFFF_81AB_54E0_1000
        agent = FakeAgent()
        mgr = self._make_manager(
            agent=agent,
            kv_data_ptrs=[high_ptr, high_ptr + 0x1000],
            kv_item_lens=[16],
        )

        handle = mgr.send_kvcache(
            "peer",
            np.array([1, 2], dtype=np.int32),
            [high_ptr + 0x2000, high_ptr + 0x3000],
            np.array([5, 6], dtype=np.int32),
            7,
            "12_kv_0_1_0",
        )

        self.assertEqual(handle, "handle_1")
        src_reqs, src_mem = agent.get_xfer_descs_calls[0]
        dst_reqs, dst_mem = agent.get_xfer_descs_calls[1]
        self.assertEqual(src_mem, "VRAM")
        self.assertEqual(dst_mem, "VRAM")
        self.assertEqual(src_reqs.dtype, np.uint64)
        self.assertEqual(dst_reqs.dtype, np.uint64)
        self.assertEqual(int(src_reqs[0, 0]), high_ptr + 16)
        self.assertEqual(int(src_reqs[0, 1]), 32)
        self.assertEqual(int(src_reqs[0, 2]), 2)
        self.assertEqual(int(dst_reqs[0, 0]), high_ptr + 0x2000 + 5 * 16)
        self.assertEqual(
            agent.initialize_xfer_calls[0][-1],
            b"12_kv_0_1_0",
        )

    def test_send_kvcache_slice_uses_vram_write_and_uint64_addresses(self):
        high_ptr = 0xFFFF_81AB_54E0_1000
        agent = FakeAgent()
        mgr = self._make_manager(
            agent=agent,
            engine_rank=1,
            kv_data_ptrs=[high_ptr, high_ptr + 0x1000],
            kv_item_lens=[64],
            page_size=2,
            kv_head_num=1,
            total_kv_head_num=2,
        )

        handle = mgr.send_kvcache_slice(
            "peer",
            np.array([1], dtype=np.int32),
            [high_ptr + 0x2000, high_ptr + 0x3000],
            np.array([4], dtype=np.int32),
            5,
            "12_kv_0_1_0",
            prefill_tp_size=2,
            decode_tp_size=1,
            decode_tp_rank=0,
            dst_kv_item_len=64,
        )

        self.assertEqual(handle, "handle_1")
        src_reqs, src_mem = agent.get_xfer_descs_calls[0]
        dst_reqs, dst_mem = agent.get_xfer_descs_calls[1]
        self.assertEqual(src_mem, "VRAM")
        self.assertEqual(dst_mem, "VRAM")
        self.assertEqual(src_reqs.dtype, np.uint64)
        self.assertEqual(dst_reqs.dtype, np.uint64)
        self.assertEqual(agent.initialize_xfer_calls[0][0], "WRITE")
        self.assertEqual(agent.initialize_xfer_calls[0][-1], b"12_kv_0_1_0")

    def test_send_mamba_state_uses_vram_descriptor_offsets(self):
        agent = FakeAgent()
        mgr = self._make_manager(agent=agent)

        handle = mgr._send_mamba_state(
            "peer",
            [3],
            [0x1000, 0x2000],
            [32, 64],
            [0x3000, 0x4000],
            [5],
            6,
            "12_state_0_0",
        )

        self.assertEqual(handle, "handle_1")
        self.assertEqual(
            agent.get_xfer_descs_calls[0],
            ([(0x1060, 32, 2), (0x20C0, 64, 2)], "VRAM"),
        )
        self.assertEqual(
            agent.get_xfer_descs_calls[1],
            ([(0x30A0, 32, 6), (0x4140, 64, 6)], "VRAM"),
        )

    def test_maybe_send_extra_dispatches_mamba_and_swa_paths(self):
        mgr = self._make_manager(
            state_types=[StateType.MAMBA, StateType.SWA],
            state_data_ptrs=[[0x1000], [0x2000, 0x3000]],
            state_item_lens=[[32], [64]],
            attn_tp_size=1,
            is_mla_backend=True,
        )
        calls = []
        mgr._send_mamba_state = lambda *args: calls.append(("mamba", args)) or "hm"
        mgr._send_kvcache_generic = lambda **kwargs: calls.append(("swa", kwargs)) or "hs"

        handles = mgr.maybe_send_extra(
            "peer",
            [[1], [2, 3]],
            [[0x4000], [0x5000, 0x6000]],
            [[4], [5, 6]],
            7,
            "12_state_0",
            decode_tp_size=1,
        )

        self.assertEqual(handles, ["hm", "hs"])
        self.assertEqual(calls[0][0], "mamba")
        self.assertEqual(calls[1][0], "swa")
        np.testing.assert_array_equal(
            calls[1][1]["prefill_data_indices"], np.array([2, 3], dtype=np.int32)
        )

    def test_maybe_send_extra_rejects_non_mla_heterogeneous_swa(self):
        mgr = self._make_manager(
            state_types=[StateType.SWA],
            state_data_ptrs=[[0x2000]],
            state_item_lens=[[64]],
            attn_tp_size=2,
            is_mla_backend=False,
        )

        with self.assertRaisesRegex(RuntimeError, "different TP sizes"):
            mgr.maybe_send_extra(
                "peer",
                [[2]],
                [[0x5000]],
                [[5]],
                7,
                "12_state_0",
                decode_tp_size=1,
            )


if __name__ == "__main__":
    unittest.main()
