"""The metadata-builder hook, exercised against a stand-in vLLM."""

from __future__ import annotations

import pytest
import torch

import fake_vllm
from da_vllm.config import DAConfig
from da_vllm.masking.patch import (
    get_patch_state,
    install_patch,
    install_triton_num_stages,
    is_sliding_window_spec,
    uninstall_patch,
)
from da_vllm.masking.shared import clear_shared_mask, install_shared_mask


@pytest.fixture
def vllm(monkeypatch):
    modules = fake_vllm.install(monkeypatch)
    clear_shared_mask()
    uninstall_patch()
    yield modules
    uninstall_patch()
    clear_shared_mask()


def _common(seq_lens, query_lens):
    starts = torch.zeros(len(query_lens) + 1, dtype=torch.int32)
    starts[1:] = torch.cumsum(torch.as_tensor(query_lens, dtype=torch.int32), 0)
    return fake_vllm.CommonAttentionMetadata(
        query_start_loc=starts, seq_lens=torch.as_tensor(seq_lens, dtype=torch.int32)
    )


def _builder(cls, spec, block_size, block_table):
    b = cls(kv_cache_spec=spec, block_size=block_size)
    b.block_tables["block_table"] = block_table
    return b


def test_both_backends_are_patched(vllm):
    state = install_patch(DAConfig(enabled=True), force=True)
    assert set(state.patched_targets) == {
        "FlashAttentionMetadataBuilder",
        "TritonAttentionMetadataBuilder",
    }


def test_cascade_attention_is_disabled(vllm):
    install_patch(DAConfig(enabled=True), force=True)
    bt = torch.arange(1, 9, dtype=torch.int32).reshape(1, 8)
    b = _builder(vllm["vllm.v1.attention.backends.flash_attn"].FlashAttentionMetadataBuilder,
                 fake_vllm.FullAttentionSpec(), 16, bt)
    b.build(4096, _common([128], [1]))
    assert b.seen_prefix_lens == [0]


def test_warmup_before_any_request_is_tolerated(vllm):
    install_patch(DAConfig(enabled=True), force=True)
    bt = torch.arange(1, 9, dtype=torch.int32).reshape(1, 8)
    b = _builder(vllm["vllm.v1.attention.backends.triton_attn"].TritonAttentionMetadataBuilder,
                 fake_vllm.FullAttentionSpec(), 32, bt)
    md = b.build(0, _common([128], [1]))  # no shared mask installed yet
    assert torch.equal(md.block_table, bt)


def test_block_size_prefers_the_full_attention_group(vllm):
    install_patch(DAConfig(enabled=True), force=True)
    bt = torch.arange(1, 9, dtype=torch.int32).reshape(1, 8)
    # vLLM's warmup calls build for every group; the sliding one may come first.
    sliding = _builder(
        vllm["vllm.v1.attention.backends.triton_attn"].TritonAttentionMetadataBuilder,
        fake_vllm.SlidingWindowSpec(), 16, bt,
    )
    sliding.build(0, _common([128], [1]))
    assert get_patch_state().block_size == 16
    full = _builder(
        vllm["vllm.v1.attention.backends.triton_attn"].TritonAttentionMetadataBuilder,
        fake_vllm.FullAttentionSpec(), 32, bt,
    )
    full.build(0, _common([128], [1]))
    state = get_patch_state()
    assert (state.block_size, state.block_size_source) == (32, "full_attention")


def test_sliding_window_group_is_never_compacted(vllm):
    install_patch(DAConfig(enabled=True), force=True)
    store = install_shared_mask(2, 256, "cpu", force=True)
    store.tensor[0].fill_(False)
    store.tensor[0, :16] = True
    store.tensor[0, 112:] = True

    bt = torch.arange(1, 9, dtype=torch.int32).reshape(1, 8)
    b = _builder(
        vllm["vllm.v1.attention.backends.triton_attn"].TritonAttentionMetadataBuilder,
        fake_vllm.SlidingWindowSpec(), 16, bt,
    )
    common = _common([128], [1])
    md = b.build(0, common)
    assert torch.equal(md.block_table, torch.arange(1, 9, dtype=torch.int32).reshape(1, 8))
    assert torch.equal(md.seq_lens, common.seq_lens)


def test_full_attention_group_is_compacted_in_place(vllm):
    install_patch(DAConfig(enabled=True), force=True)
    store = install_shared_mask(2, 256, "cpu", force=True)
    store.tensor[0].fill_(False)
    store.tensor[0, :16] = True  # block 0
    store.tensor[0, 112:] = True  # block 7 (the tail)

    bt = torch.arange(10, 18, dtype=torch.int32).reshape(1, 8)
    original_ptr = bt.data_ptr()
    b = _builder(
        vllm["vllm.v1.attention.backends.flash_attn"].FlashAttentionMetadataBuilder,
        fake_vllm.FullAttentionSpec(), 16, bt,
    )
    md = b.build(0, _common([128], [1]))
    # In place: the runner reassigns block_table per group and CUDA graph
    # replay bakes the pointer.
    assert md.block_table.data_ptr() == original_ptr
    assert bt[0, :2].tolist() == [10, 17]
    assert int(md.seq_lens[0]) == 32


def test_shared_seq_lens_is_routed_through_a_stable_scratch(vllm):
    install_patch(DAConfig(enabled=True), force=True)
    store = install_shared_mask(2, 256, "cpu", force=True)
    store.tensor[0].fill_(False)
    store.tensor[0, :16] = True
    store.tensor[0, 112:] = True

    ptrs = set()
    for _ in range(3):
        bt = torch.arange(10, 18, dtype=torch.int32).reshape(1, 8)
        b = _builder(
            vllm["vllm.v1.attention.backends.flash_attn"].FlashAttentionMetadataBuilder,
            fake_vllm.FullAttentionSpec(), 16, bt,
        )
        common = _common([128], [1])
        shared_before = common.seq_lens.clone()
        md = b.build(0, common)
        # The group-shared tensor is not shrunk in place: doing that corrupted
        # the sliding-window group's kernel on Gemma.
        assert torch.equal(common.seq_lens, shared_before)
        assert md.seq_lens.data_ptr() != common.seq_lens.data_ptr()
        ptrs.add(md.seq_lens.data_ptr())
    assert len(ptrs) == 1  # stable pointer across steps


def test_max_seq_len_is_left_as_a_safe_over_estimate(vllm):
    install_patch(DAConfig(enabled=True), force=True)
    store = install_shared_mask(2, 256, "cpu", force=True)
    store.tensor[0].fill_(False)
    store.tensor[0, :16] = True
    store.tensor[0, 112:] = True
    bt = torch.arange(10, 18, dtype=torch.int32).reshape(1, 8)
    b = _builder(
        vllm["vllm.v1.attention.backends.flash_attn"].FlashAttentionMetadataBuilder,
        fake_vllm.FullAttentionSpec(), 16, bt,
    )
    md = b.build(0, _common([128], [1]))
    assert md.max_seq_len == 128
    assert int(md.seq_lens[0]) < md.max_seq_len


def test_aot_scheduler_metadata_is_not_recomputed(vllm):
    install_patch(DAConfig(enabled=True), force=True)
    install_shared_mask(2, 256, "cpu", force=True)
    bt = torch.arange(10, 18, dtype=torch.int32).reshape(1, 8)
    b = _builder(
        vllm["vllm.v1.attention.backends.flash_attn"].FlashAttentionMetadataBuilder,
        fake_vllm.FullAttentionSpec(), 16, bt,
    )
    md = b.build(0, _common([128], [1]))
    assert md.scheduler_metadata is None


def test_a_broken_remap_never_takes_the_engine_down(vllm, monkeypatch):
    install_patch(DAConfig(enabled=True), force=True)
    install_shared_mask(2, 256, "cpu", force=True)
    monkeypatch.setattr(
        "da_vllm.masking.patch.remap_optimized",
        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    bt = torch.arange(10, 18, dtype=torch.int32).reshape(1, 8)
    b = _builder(
        vllm["vllm.v1.attention.backends.flash_attn"].FlashAttentionMetadataBuilder,
        fake_vllm.FullAttentionSpec(), 16, bt,
    )
    md = b.build(0, _common([128], [1]))  # must not raise
    assert md is not None


def test_uninstall_restores_the_originals(vllm):
    cls = vllm["vllm.v1.attention.backends.flash_attn"].FlashAttentionMetadataBuilder
    original = cls.build
    install_patch(DAConfig(enabled=True), force=True)
    assert cls.build is not original
    uninstall_patch()
    assert cls.build is original


def test_triton_num_stages_is_injected_once(vllm):
    module = vllm["vllm.attention.ops.triton_unified_attention"]
    kernel = module.kernel_unified_attention_3d
    assert install_triton_num_stages(2) is True
    assert install_triton_num_stages(2) is False  # idempotent
    module.kernel_unified_attention_3d[(1, 2)](3, foo="bar")
    assert kernel.calls == [((1, 2), {"foo": "bar", "num_stages": 2})]


def test_sliding_window_detection_falls_back_to_the_attribute():
    class Unknown:
        sliding_window = 512

    class Other:
        sliding_window = None

    assert is_sliding_window_spec(Unknown()) is True
    assert is_sliding_window_spec(Other()) is False
