"""Unit tests for disaggregation rank mapping and PP pointer slicing."""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.common.conn import CommonKVManager, PrefillServerInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class TestDisaggregationRankMapping(CustomTestCase):
    def _make_manager(
        self,
        *,
        decode_tp=2,
        engine_rank=1,
        decode_cp=1,
        cp_rank=0,
        decode_pp=1,
        pp_rank=0,
        all_cp=False,
        is_mla=True,
    ):
        mgr = object.__new__(CommonKVManager)
        mgr.attn_tp_size = decode_tp
        mgr.attn_cp_size = decode_cp
        mgr.attn_cp_rank = cp_rank
        mgr.pp_size = decode_pp
        mgr.pp_rank = pp_rank
        mgr.enable_all_cp_ranks_for_transfer = all_cp
        mgr.is_mla_backend = is_mla
        mgr.kv_args = SimpleNamespace(engine_rank=engine_rank)
        return mgr

    def _make_info(self, *, prefill_tp=2, prefill_cp=1, prefill_pp=1):
        return PrefillServerInfo(
            attn_tp_size=prefill_tp,
            attn_cp_size=prefill_cp,
            dp_size=1,
            pp_size=prefill_pp,
            page_size=1,
            kv_cache_dtype="auto",
            follow_bootstrap_room=True,
        )

    def test_same_tp_maps_to_matching_rank(self):
        mgr = self._make_manager(decode_tp=2, engine_rank=1)
        info = self._make_info(prefill_tp=2)

        mgr._resolve_rank_mapping(info)

        self.assertEqual(info.target_tp_rank, 1)
        self.assertEqual(info.target_tp_ranks, [1])
        self.assertEqual(info.required_dst_info_num, 1)
        self.assertEqual(info.required_prefill_response_num, 1)

    def test_decode_tp_larger_than_prefill_tp_groups_decode_ranks(self):
        mgr = self._make_manager(decode_tp=4, engine_rank=3)
        info = self._make_info(prefill_tp=2)

        mgr._resolve_rank_mapping(info)

        self.assertEqual(info.target_tp_rank, 1)
        self.assertEqual(info.target_tp_ranks, [1])
        self.assertEqual(info.required_dst_info_num, 2)
        self.assertEqual(info.required_prefill_response_num, 1)

    def test_prefill_tp_larger_than_decode_tp_selects_multiple_prefill_ranks(self):
        mgr = self._make_manager(decode_tp=2, engine_rank=1, is_mla=False)
        info = self._make_info(prefill_tp=4)

        mgr._resolve_rank_mapping(info)

        self.assertEqual(info.target_tp_rank, 2)
        self.assertEqual(info.target_tp_ranks, [2, 3])
        self.assertEqual(info.required_dst_info_num, 1)
        self.assertEqual(info.required_prefill_response_num, 2)

    def test_prefill_cp_ranks_are_filtered_unless_all_cp_transfer_enabled(self):
        mgr = self._make_manager(all_cp=False)
        info = self._make_info(prefill_cp=4)

        mgr._resolve_rank_mapping(info)

        self.assertEqual(info.target_cp_ranks, [0])
        self.assertEqual(info.required_prefill_response_num, 1)

        mgr = self._make_manager(all_cp=True)
        info = self._make_info(prefill_cp=4)

        mgr._resolve_rank_mapping(info)

        self.assertEqual(info.target_cp_ranks, [0, 1, 2, 3])
        self.assertEqual(info.required_prefill_response_num, 4)

    def test_decode_pp_one_targets_all_prefill_pp_ranks(self):
        mgr = self._make_manager(decode_pp=1, pp_rank=0)
        info = self._make_info(prefill_pp=4)

        mgr._resolve_rank_mapping(info)

        self.assertEqual(info.target_pp_ranks, [0, 1, 2, 3])
        self.assertEqual(info.required_prefill_response_num, 4)

    def test_invalid_decode_cp_size_raises(self):
        mgr = self._make_manager(decode_cp=2)
        info = self._make_info(prefill_cp=1)

        with self.assertRaises(AssertionError):
            mgr._resolve_rank_mapping(info)


class TestDisaggregationPointerSlicing(CustomTestCase):
    def _make_manager(self, *, start_layer=1, end_layer=3, ratios=None):
        mgr = object.__new__(CommonKVManager)
        mgr.kv_args = SimpleNamespace(
            prefill_start_layer=start_layer,
            prefill_end_layer=end_layer,
            mla_compression_ratios=ratios,
        )
        return mgr

    def test_mha_pointer_slicing_for_decode_full_model_pp_one(self):
        mgr = self._make_manager(start_layer=1)

        src_k, src_v, dst_k, dst_v, num_layers = mgr.get_mha_kv_ptrs_with_pp(
            [10, 11, 20, 21],
            [100, 101, 102, 103, 200, 201, 202, 203],
        )

        self.assertEqual(src_k, [10, 11])
        self.assertEqual(src_v, [20, 21])
        self.assertEqual(dst_k, [101, 102])
        self.assertEqual(dst_v, [201, 202])
        self.assertEqual(num_layers, 2)

    def test_mla_pointer_slicing_requires_prefill_end_layer_for_compressed_layout(self):
        mgr = self._make_manager(start_layer=1, end_layer=None, ratios=[0, 4, 128])

        with self.assertRaises(AssertionError):
            mgr.get_mla_kv_ptrs_with_pp([10, 11], [100, 101, 200, 300])

    def test_compressed_mla_kv_layout_slices_matching_sections(self):
        ratios = [0, 4, 128, 4]
        mgr = self._make_manager(start_layer=1, end_layer=3, ratios=ratios)

        src, dst, num_layers = mgr.get_mla_kv_ptrs_with_pp(
            [10, 11, 12],
            [100, 101, 200, 201, 300],
        )

        self.assertEqual(src, [10, 11, 12])
        self.assertEqual(dst, [100, 200, 300])
        self.assertEqual(num_layers, 3)

    def test_compressed_mla_state_layout_slices_swa_and_compress_sections(self):
        ratios = [0, 4, 128, 4]
        mgr = self._make_manager(start_layer=1, end_layer=3, ratios=ratios)

        src, dst, num_layers = mgr.get_mla_kv_ptrs_with_pp(
            [10, 11, 12, 13],
            [100, 101, 102, 103, 200, 201, 202, 300, 301],
        )

        self.assertEqual(src, [10, 11, 12, 13])
        self.assertEqual(dst, [101, 102, 200, 201, 300])
        self.assertEqual(num_layers, 4)


if __name__ == "__main__":
    unittest.main()
