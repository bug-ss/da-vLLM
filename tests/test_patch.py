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
    is_maskable_spec,
    uninstall_patch,
    uninstall_triton_num_stages,
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
    lens = torch.as_tensor(seq_lens, dtype=torch.int32)
    return fake_vllm.CommonAttentionMetadata(
        query_start_loc=starts,
        seq_lens=lens,
        num_reqs=len(seq_lens),
        block_table_tensor=torch.zeros((len(seq_lens), 1), dtype=torch.int32),
        max_seq_len=int(lens.max()) if len(seq_lens) else 0,
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
    # One builder, several steps -- which is what vLLM does: a builder is
    # created per KV-cache group and reused for the life of the engine.
    bt = torch.arange(10, 18, dtype=torch.int32).reshape(1, 8)
    b = _builder(
        vllm["vllm.v1.attention.backends.flash_attn"].FlashAttentionMetadataBuilder,
        fake_vllm.FullAttentionSpec(), 16, bt,
    )
    for _ in range(3):
        bt.copy_(torch.arange(10, 18, dtype=torch.int32).reshape(1, 8))
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


def test_triton_num_stages_is_injected_on_the_decode_grid_only(vllm):
    """vLLM 0.20.2 has ONE kernel_unified_attention; the decode (split-KV)
    launch is the 3-element grid.  The prefill launch must be left alone."""
    module = vllm["vllm.v1.attention.ops.triton_unified_attention"]
    kernel = module.kernel_unified_attention
    assert install_triton_num_stages(2) is True
    assert install_triton_num_stages(2) is False  # idempotent

    module.kernel_unified_attention[(4, 2, 8)](1, foo="bar")   # decode
    module.kernel_unified_attention[(4, 2)](1, foo="bar")      # prefill
    assert kernel.calls == [
        ((4, 2, 8), {"foo": "bar", "num_stages": 2}),
        ((4, 2), {"foo": "bar"}),
    ]
    assert uninstall_triton_num_stages() is True
    assert module.kernel_unified_attention is kernel


def test_only_a_positively_identified_full_attention_spec_is_masked(vllm):
    """The polarity is what matters.

    vLLM 0.20.2 ships rotating-cache specs that share no attribute:
    SlidingWindowSpec has sliding_window, ChunkedLocalAttentionSpec has
    attention_chunk_size, MambaSpec has neither.  "Is this sliding?" answers
    *no* for a spec it has not heard of, and DA would compact a rotating cache.
    """
    bits = {
        "FullAttentionSpec": fake_vllm.FullAttentionSpec,
        "SlidingWindowSpec": fake_vllm.SlidingWindowSpec,
    }
    assert is_maskable_spec(fake_vllm.FullAttentionSpec(), bits) is True
    assert is_maskable_spec(fake_vllm.SinkFullAttentionSpec(), bits) is True
    assert is_maskable_spec(fake_vllm.SlidingWindowSpec(), bits) is False
    assert is_maskable_spec(fake_vllm.ChunkedLocalAttentionSpec(), bits) is False
    assert is_maskable_spec(fake_vllm.MambaSpec(), bits) is False
    assert is_maskable_spec(None, bits) is False

    # With no classes to check against, refuse anything advertising a bound.
    assert is_maskable_spec(fake_vllm.ChunkedLocalAttentionSpec(), {}) is False
    assert is_maskable_spec(fake_vllm.SlidingWindowSpec(), {}) is False


def test_metadata_capture_records_the_post_remap_state(vllm, tmp_path):
    """Validation check #4's capture half, against the stand-in vLLM."""
    from da_vllm.validation.capture import (
        MetadataCapture,
        kept_fraction_series,
        read_capture,
        write_capture,
    )

    install_patch(DAConfig(enabled=True), force=True)
    store = install_shared_mask(2, 256, "cpu", force=True)
    store.tensor[0].fill_(False)
    store.tensor[0, :16] = True
    store.tensor[0, 112:] = True

    capture = MetadataCapture(limit=4)
    with capture:
        for _ in range(6):  # more than the limit, on purpose
            bt = torch.arange(10, 18, dtype=torch.int32).reshape(1, 8)
            b = _builder(
                vllm["vllm.v1.attention.backends.flash_attn"].FlashAttentionMetadataBuilder,
                fake_vllm.FullAttentionSpec(), 16, bt,
            )
            b.build(0, _common([128], [1]))

    assert len(capture.steps) == 4
    # The remap kept 2 of 8 blocks, so the recorded seq_len is the compacted one.
    assert capture.steps[0].seq_lens == [32]
    assert kept_fraction_series(capture.steps)[0] <= 1.0
    assert capture.steps[0].backend == "FlashAttentionMetadataBuilder"

    path = tmp_path / "capture.jsonl"
    assert write_capture(path, capture.steps) == 4
    assert [s.step for s in read_capture(path)] == [0, 1, 2, 3]


def test_capture_refuses_to_install_before_the_da_patch(vllm):
    from da_vllm.validation.capture import MetadataCapture

    uninstall_patch()
    with pytest.raises(RuntimeError, match="install the DA patch first"):
        MetadataCapture().install()


def test_chunked_local_groups_are_left_alone(vllm):
    """A chunked-local cache rotates too; compacting it dereferences stale slots."""
    install_patch(DAConfig(enabled=True), force=True)
    store = install_shared_mask(2, 256, "cpu", force=True)
    store.tensor[0].fill_(False)
    store.tensor[0, :16] = True
    store.tensor[0, 112:] = True

    bt = torch.arange(1, 9, dtype=torch.int32).reshape(1, 8)
    b = _builder(
        vllm["vllm.v1.attention.backends.triton_attn"].TritonAttentionMetadataBuilder,
        fake_vllm.ChunkedLocalAttentionSpec(), 16, bt,
    )
    common = _common([128], [1])
    md = b.build(0, common)
    assert torch.equal(md.block_table, torch.arange(1, 9, dtype=torch.int32).reshape(1, 8))
    assert torch.equal(md.seq_lens, common.seq_lens)


def test_cuda_graph_capture_is_never_remapped(vllm):
    """Capture runs through build().  A captured graph must not bake a
    compacted table, and Triton's capture path then does seq_lens.fill_(1)
    in place -- which would land in our scratch buffer."""
    install_patch(DAConfig(enabled=True), force=True)
    store = install_shared_mask(2, 256, "cpu", force=True)
    store.tensor[0].fill_(False)
    store.tensor[0, :16] = True
    store.tensor[0, 112:] = True

    bt = torch.arange(10, 18, dtype=torch.int32).reshape(1, 8)
    b = _builder(
        vllm["vllm.v1.attention.backends.triton_attn"].TritonAttentionMetadataBuilder,
        fake_vllm.FullAttentionSpec(), 16, bt,
    )
    common = _common([128], [1])
    md = b.build_for_cudagraph_capture(common)

    # Untouched by DA, and Triton's fill_(1) hit the runner's own tensor.
    assert torch.equal(md.block_table, torch.arange(10, 18, dtype=torch.int32).reshape(1, 8))
    assert md.seq_lens is common.seq_lens
    assert int(common.seq_lens[0]) == 1

    # The flag is cleared, so the next real build does remap.
    state = get_patch_state()
    assert state.in_cudagraph_capture is False
    bt2 = torch.arange(10, 18, dtype=torch.int32).reshape(1, 8)
    b2 = _builder(
        vllm["vllm.v1.attention.backends.triton_attn"].TritonAttentionMetadataBuilder,
        fake_vllm.FullAttentionSpec(), 16, bt2,
    )
    md2 = b2.build(0, _common([128], [1]))
    assert int(md2.seq_lens[0]) == 32


def test_block_table_reuse_is_disabled_while_the_patch_is_installed(vllm):
    """vLLM 0.20.2 caches one metadata build per (spec, builder) and gives
    later KV-cache groups a shallow copy with only the block table swapped.
    That copy would keep the first group's COMPACTED seq_lens next to an
    uncompacted block table."""
    flash = vllm["vllm.v1.attention.backends.flash_attn"].FlashAttentionMetadataBuilder
    assert flash.supports_update_block_table is True
    install_patch(DAConfig(enabled=True), force=True)
    assert flash.supports_update_block_table is False
    uninstall_patch()
    assert flash.supports_update_block_table is True


def test_the_reuse_hazard_is_real_if_reuse_is_left_on(vllm):
    """Demonstrates why the line above exists, by doing what vLLM would do."""
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
    cached = b.build(0, _common([128], [1]))
    assert int(cached.seq_lens[0]) == 32  # compacted to 2 blocks

    other_group_table = torch.arange(100, 108, dtype=torch.int32).reshape(1, 8)
    reused = b.update_block_table(cached, other_group_table, None)
    # Full table, compacted length: the kernel would read blocks 100 and 101.
    assert torch.equal(reused.block_table, other_group_table)
    assert int(reused.seq_lens[0]) == 32


def test_two_full_attention_groups_do_not_share_a_scratch_buffer(vllm):
    """Two maskable groups in one step must not overwrite each other's lengths.

    They are separate builders with separate layer names, so they must get
    separate buffers -- otherwise the second group's compacted lengths land on
    the first group's metadata, next to the first group's block table.
    """
    install_patch(DAConfig(enabled=True), force=True)
    store = install_shared_mask(2, 256, "cpu", force=True)
    store.tensor[0].fill_(False)
    store.tensor[0, :16] = True
    store.tensor[0, 112:] = True

    cls = vllm["vllm.v1.attention.backends.flash_attn"].FlashAttentionMetadataBuilder
    made = []
    for layers, table in (("layers.0", 10), ("layers.1", 100)):
        bt = torch.arange(table, table + 8, dtype=torch.int32).reshape(1, 8)
        b = _builder(cls, fake_vllm.FullAttentionSpec(), 16, bt)
        b.layer_names = (layers,)
        made.append((b, b.build(0, _common([128], [1]))))

    (_, md_a), (_, md_b) = made
    assert md_a.seq_lens.data_ptr() != md_b.seq_lens.data_ptr()
    assert int(md_a.seq_lens[0]) == int(md_b.seq_lens[0]) == 32


def test_a_full_cudagraph_over_decode_is_refused(vllm):
    """The compacted lengths cannot reach a replayed full graph."""
    import enum

    from da_vllm.masking.patch import (
        UnsupportedCudagraphMode,
        assert_cudagraph_mode_supported,
    )

    class Mode(enum.Enum):
        PIECEWISE = 1
        FULL = 2

        def decode_mode(self):
            return self

    class Compilation:
        def __init__(self, mode):
            self.cudagraph_mode = mode

    class Config:
        def __init__(self, mode):
            self.compilation_config = Compilation(mode)

    off = DAConfig(enabled=True)
    assert_cudagraph_mode_supported(Config(Mode.PIECEWISE), off)  # fine
    assert_cudagraph_mode_supported(Config(None), off)  # unknown: don't guess
    with pytest.raises(UnsupportedCudagraphMode, match="PIECEWISE"):
        assert_cudagraph_mode_supported(Config(Mode.FULL), off)
    # An expert who has validated it can override.
    assert_cudagraph_mode_supported(
        Config(Mode.FULL), DAConfig(enabled=True, allow_full_cudagraph=True)
    )


def test_capture_reports_a_real_kept_fraction(vllm):
    """Deriving both numbers from the already-compacted lengths would make the
    ratio 1.0 for every input -- i.e. 'nothing was pruned', which is exactly
    what this harness exists to disprove."""
    from da_vllm.validation.capture import MetadataCapture, kept_fraction_series

    install_patch(DAConfig(enabled=True), force=True)
    store = install_shared_mask(2, 256, "cpu", force=True)
    store.tensor[0].fill_(False)
    store.tensor[0, :16] = True
    store.tensor[0, 112:] = True

    capture = MetadataCapture(limit=2)
    with capture:
        bt = torch.arange(10, 18, dtype=torch.int32).reshape(1, 8)
        b = _builder(
            vllm["vllm.v1.attention.backends.flash_attn"].FlashAttentionMetadataBuilder,
            fake_vllm.FullAttentionSpec(), 16, bt,
        )
        b.build(0, _common([128], [1]))

    step = capture.steps[0]
    assert step.total_blocks == [8]   # before compaction
    assert step.kept_blocks == [2]    # after
    assert kept_fraction_series(capture.steps)[0] == pytest.approx(0.25)
