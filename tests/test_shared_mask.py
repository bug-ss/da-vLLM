import random

import pytest
import torch

from da_vllm.masking.shared import (
    SharedMaskStore,
    clear_shared_mask,
    get_shared_mask,
    install_shared_mask,
)
from da_vllm.state_machine import MaskSnapshot, Mode, align_spans


@pytest.fixture(autouse=True)
def _clean():
    clear_shared_mask()
    yield
    clear_shared_mask()


def test_starts_all_true():
    store = SharedMaskStore(4, 64, "cpu")
    assert bool(store.tensor.all())


@pytest.mark.parametrize("block_size", [1, 4, 16, 32])
@pytest.mark.parametrize("seed", range(8))
def test_both_writers_produce_the_same_row(block_size, seed):
    rng = random.Random(seed)
    length = rng.randint(32, 600)
    spans, pos = [], 0
    while pos < length - 2:
        s = pos + rng.randint(0, 25)
        if s >= length:
            break
        e = min(s + rng.randint(1, 40), length)
        spans.append((s, e))
        pos = e + rng.randint(1, 20)
    snapshot = MaskSnapshot(Mode.FOCUS, tuple(spans), length)

    store = SharedMaskStore(2, length + 64, "cpu")
    store.write_snapshot(0, snapshot, block_size=block_size, optimized=True)
    store.write_snapshot(1, snapshot, block_size=block_size, optimized=False)
    assert torch.equal(store.tensor[0], store.tensor[1])

    # And both agree with OR-reducing the dense mask at the same block size.
    dense = torch.tensor(snapshot.dense(length + 64))
    n = ((length + 64) // block_size) * block_size
    expected = dense[:n].unflatten(0, (n // block_size, block_size)).any(1)
    got = store.tensor[0][:n].unflatten(0, (n // block_size, block_size)).any(1)
    assert torch.equal(expected, got)


def test_global_snapshot_resets_the_row_to_all_true():
    store = SharedMaskStore(2, 128, "cpu")
    store.write_snapshot(0, MaskSnapshot(Mode.LOCAL, ((0, 16),), 64), block_size=16)
    assert not bool(store.tensor[0].all())
    store.write_snapshot(0, MaskSnapshot(Mode.GLOBAL, ((0, 64),), 64), block_size=16)
    assert bool(store.tensor[0].all())


def test_the_tail_block_containing_the_current_token_is_always_kept():
    store = SharedMaskStore(1, 128, "cpu")
    snapshot = MaskSnapshot(Mode.LOCAL, ((0, 16),), 100)
    store.write_snapshot(0, snapshot, block_size=32)
    # Position 100 sits in the block starting at 96; that whole block must be
    # visible or the kernel cannot see the current token's own key.
    assert bool(store.tensor[0, 96:128].all())


def test_recycled_rows_do_not_keep_stale_mask_data():
    store = SharedMaskStore(2, 128, "cpu")
    store.write_snapshot(1, MaskSnapshot(Mode.LOCAL, ((0, 16),), 64), block_size=16)
    assert not bool(store.tensor[1].all())
    store.reset_row(1)
    assert bool(store.tensor[1].all())


def test_span_alignment_is_applied_before_writing():
    store = SharedMaskStore(1, 128, "cpu")
    snapshot = MaskSnapshot(Mode.FOCUS, ((17, 19),), 128)
    store.write_snapshot(0, snapshot, block_size=16)
    assert align_spans(snapshot.spans, 16) == ((16, 32),)
    assert bool(store.tensor[0, 16:32].all())
    assert not bool(store.tensor[0, :16].any())


def test_install_is_idempotent_and_clearable():
    a = install_shared_mask(2, 64, "cpu")
    b = install_shared_mask(99, 99, "cpu")
    assert a is b is get_shared_mask()
    clear_shared_mask()
    assert get_shared_mask() is None


def test_reported_size_matches_the_documented_footprint():
    store = SharedMaskStore(8, 1024, "cpu")
    assert store.nbytes() == 8 * 1024
