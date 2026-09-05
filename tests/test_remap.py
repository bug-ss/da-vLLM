"""Bit-identity between the readable and optimized remaps (guide 7).

Do **not** test this by comparing generated tokens: differently shaped
transient tensors change allocator state, last-bit logits differ, and greedy
argmax flips even when the remap is a verified no-op.
"""

from __future__ import annotations

import math
import random

import pytest
import torch

from da_vllm.masking.remap import (
    RemapScratch,
    RemapStats,
    block_allowed,
    remap_optimized,
    remap_readable,
)


def _run_both(mask, block_table, seq_lens, query_lens, block_size):
    a_bt, b_bt = block_table.clone(), block_table.clone()
    n = block_table.shape[0]
    a_sl = torch.zeros(n, dtype=seq_lens.dtype)
    b_sl = torch.zeros(n, dtype=seq_lens.dtype)
    remap_readable(
        mask=mask,
        block_table=a_bt,
        seq_lens=seq_lens,
        out_seq_lens=a_sl,
        query_lens=query_lens,
        block_size=block_size,
    )
    remap_optimized(
        mask=mask,
        block_table=b_bt,
        seq_lens=seq_lens,
        out_seq_lens=b_sl,
        query_lens=query_lens,
        block_size=block_size,
        scratch=RemapScratch(),
    )
    return (a_bt, a_sl), (b_bt, b_sl)


def _assert_identical(a, b, block_size, table_width):
    (a_bt, a_sl), (b_bt, b_sl) = a, b
    assert torch.equal(a_sl, b_sl), "seq_lens differ"
    for r in range(a_bt.shape[0]):
        used = min(math.ceil(int(a_sl[r]) / block_size), table_width)
        assert torch.equal(a_bt[r, :used], b_bt[r, :used]), f"block_table row {r} differs"


@pytest.mark.parametrize("seed", range(40))
def test_optimized_matches_readable_across_a_random_sweep(seed):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    rows = rng.choice([0, 1, 2, 5, 8, 17])
    block_size = rng.choice([1, 2, 16, 32])
    seq_len = rng.randint(1, 4096)
    table_width = rng.randint(1, math.ceil(seq_len / block_size) + 4)
    mask_width = rng.choice([seq_len, max(1, seq_len // 2), seq_len + 128])
    density = rng.choice([0.05, 0.3, 0.7, 1.0])

    mask = torch.rand(max(rows, 1), mask_width) < density
    block_table = torch.randint(1, 100_000, (rows, table_width), dtype=torch.int32)
    seq_lens = torch.randint(1, seq_len + 1, (max(rows, 1),), dtype=torch.int32)[:rows]
    query_lens = torch.where(
        torch.rand(rows) < 0.75,
        torch.ones(rows, dtype=torch.int32),
        torch.randint(2, 64, (rows,), dtype=torch.int32),
    )

    a, b = _run_both(mask, block_table, seq_lens, query_lens, block_size)
    _assert_identical(a, b, block_size, table_width)


def test_zero_rows_is_a_no_op():
    a, b = _run_both(
        torch.ones(1, 64, dtype=torch.bool),
        torch.zeros((0, 4), dtype=torch.int32),
        torch.zeros(0, dtype=torch.int32),
        torch.zeros(0, dtype=torch.int32),
        16,
    )
    assert a[0].numel() == b[0].numel() == 0


def test_sequence_longer_than_the_mask_degrades_to_full_attention():
    block_size = 16
    mask = torch.zeros(1, 32, dtype=torch.bool)  # mask covers only 2 blocks
    mask[0, :16] = True
    block_table = torch.arange(1, 9, dtype=torch.int32).reshape(1, 8)
    seq_lens = torch.tensor([128], dtype=torch.int32)  # 8 blocks
    query_lens = torch.tensor([1], dtype=torch.int32)
    a, b = _run_both(mask, block_table, seq_lens, query_lens, block_size)
    # Block 1 is masked out but blocks 2..7 are past the mask and count as
    # allowed, so exactly one block is dropped -- never a truncated block list.
    assert int(a[1][0]) == 128 - block_size
    _assert_identical(a, b, block_size, 8)


def test_prefill_rows_are_untouched():
    block_size = 16
    mask = torch.zeros(1, 128, dtype=torch.bool)
    mask[0, :16] = True
    block_table = torch.arange(1, 9, dtype=torch.int32).reshape(1, 8)
    seq_lens = torch.tensor([128], dtype=torch.int32)
    for query_len in (2, 37):
        a, b = _run_both(
            mask, block_table, seq_lens, torch.tensor([query_len], dtype=torch.int32), block_size
        )
        assert torch.equal(a[0], block_table)
        assert int(a[1][0]) == 128
        _assert_identical(a, b, block_size, 8)


def test_nothing_pruned_leaves_the_row_alone():
    block_size = 16
    mask = torch.ones(1, 128, dtype=torch.bool)
    block_table = torch.arange(1, 9, dtype=torch.int32).reshape(1, 8)
    seq_lens = torch.tensor([120], dtype=torch.int32)
    a, b = _run_both(mask, block_table, seq_lens, torch.ones(1, dtype=torch.int32), block_size)
    assert torch.equal(a[0], block_table)
    assert int(a[1][0]) == 120
    _assert_identical(a, b, block_size, 8)


def test_compaction_keeps_the_tail_block_and_its_partial_length():
    block_size = 16
    mask = torch.zeros(1, 128, dtype=torch.bool)
    mask[0, :16] = True  # block 0
    mask[0, 96:] = True  # blocks 6 and 7 (7 is the partial tail)
    block_table = torch.arange(10, 18, dtype=torch.int32).reshape(1, 8)
    seq_lens = torch.tensor([120], dtype=torch.int32)  # tail block holds 8 tokens
    a, b = _run_both(mask, block_table, seq_lens, torch.ones(1, dtype=torch.int32), block_size)
    assert a[0][0, :3].tolist() == [10, 16, 17]
    assert int(a[1][0]) == 2 * block_size + 8
    _assert_identical(a, b, block_size, 8)


def test_an_all_masked_row_is_refused_rather_than_zero_length():
    block_size = 16
    mask = torch.zeros(1, 128, dtype=torch.bool)
    block_table = torch.arange(1, 9, dtype=torch.int32).reshape(1, 8)
    seq_lens = torch.tensor([128], dtype=torch.int32)
    a, b = _run_both(mask, block_table, seq_lens, torch.ones(1, dtype=torch.int32), block_size)
    assert int(a[1][0]) == 128
    assert torch.equal(a[0], block_table)
    _assert_identical(a, b, block_size, 8)


def test_shared_seq_lens_is_never_written():
    block_size = 16
    mask = torch.zeros(1, 128, dtype=torch.bool)
    mask[0, :16] = True
    mask[0, 112:] = True
    shared = torch.tensor([128], dtype=torch.int32)
    original = shared.clone()
    out = torch.zeros(1, dtype=torch.int32)
    remap_readable(
        mask=mask,
        block_table=torch.arange(1, 9, dtype=torch.int32).reshape(1, 8),
        seq_lens=shared,
        out_seq_lens=out,
        query_lens=torch.ones(1, dtype=torch.int32),
        block_size=block_size,
    )
    assert torch.equal(shared, original)
    assert int(out[0]) < 128


def test_scratch_pointers_are_stable_across_steps():
    scratch = RemapScratch()
    first = scratch.get("seq_lens", (8,), torch.int32, torch.device("cpu"))
    second = scratch.get("seq_lens", (8,), torch.int32, torch.device("cpu"))
    assert first.data_ptr() == second.data_ptr()


def test_block_allowed_or_reduces_and_treats_the_overhang_as_allowed():
    mask = torch.zeros(1, 32, dtype=torch.bool)
    mask[0, 5] = True
    allowed = block_allowed(mask, 1, 4, 16)
    assert allowed[0].tolist() == [True, False, True, True]


def test_stats_report_the_realized_kept_fraction():
    block_size = 16
    mask = torch.zeros(1, 128, dtype=torch.bool)
    mask[0, :16] = True
    mask[0, 112:] = True
    stats = RemapStats()
    remap_readable(
        mask=mask,
        block_table=torch.arange(1, 9, dtype=torch.int32).reshape(1, 8),
        seq_lens=torch.tensor([128], dtype=torch.int32),
        out_seq_lens=torch.zeros(1, dtype=torch.int32),
        query_lens=torch.ones(1, dtype=torch.int32),
        block_size=block_size,
        stats=stats,
    )
    assert (stats.kept_blocks, stats.total_blocks) == (2, 8)
    assert stats.kept_fraction == 0.25


def test_optimized_also_never_writes_the_shared_seq_lens():
    block_size = 16
    mask = torch.zeros(1, 128, dtype=torch.bool)
    mask[0, :16] = True
    mask[0, 112:] = True
    shared = torch.tensor([128], dtype=torch.int32)
    original = shared.clone()
    out = torch.zeros(1, dtype=torch.int32)
    remap_optimized(
        mask=mask,
        block_table=torch.arange(1, 9, dtype=torch.int32).reshape(1, 8),
        seq_lens=shared,
        out_seq_lens=out,
        query_lens=torch.ones(1, dtype=torch.int32),
        block_size=block_size,
        scratch=RemapScratch(),
    )
    assert torch.equal(shared, original)
    assert int(out[0]) < 128


def test_a_batch_wider_than_the_mask_degrades_to_full_attention():
    block_size = 16
    mask = torch.zeros(1, 128, dtype=torch.bool)  # only one row allocated
    mask[0, :16] = True
    block_table = torch.arange(1, 17, dtype=torch.int32).reshape(2, 8)
    seq_lens = torch.tensor([128, 128], dtype=torch.int32)
    query_lens = torch.ones(2, dtype=torch.int32)
    a, b = _run_both(mask, block_table, seq_lens, query_lens, block_size)
    # Row 0 is pruned; row 1 has no mask row and must be left alone.
    assert int(a[1][0]) < 128 and int(a[1][1]) == 128
    assert torch.equal(a[0][1], block_table[1])
    _assert_identical(a, b, block_size, 8)
