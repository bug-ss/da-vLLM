"""Block-table remap: turn a boolean KV mask into a compacted block table.

Two implementations of the same function:

* :func:`remap_readable` -- the specification, written the obvious way.  It
  contains three host-device syncs per step (``.any()``, boolean indexing,
  ``.item()``) which serialize the host against the decode stream: about 6x
  slower at long context and batch 1.
* :func:`remap_optimized` -- sync-free.  No early exit, a dense ``scatter_``
  into a sentinel column instead of boolean indexing, ``torch.where`` to
  preserve non-remapped rows exactly, and an aggregation over the R *active*
  rows only.  Scanning the whole ``(max_num_seqs, max_model_len)`` allocation
  costs ~2 ms/step at 262K with vLLM's default 1024 sequences, paid even when
  nothing is pruned; it once showed up as an intercept shift in time-versus-
  bytes fits and made one model look net-negative.

The two must agree bit-for-bit on ``block_table[:, :num_kept_blocks]`` and on
``seq_lens``.  ``tests/test_remap_equivalence.py`` sweeps batch sizes, block
sizes, sequence lengths and prune patterns, including R = 0, narrow block
tables and sequence lengths above ``max_model_len``.

Do **not** use generated-token identity as the equivalence test: differently
shaped transient tensors change allocator state, last-bit logits differ, and
greedy argmax flips even when the remap is a verified no-op.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RemapStats:
    """Only populated when explicitly requested; reading it costs a sync."""

    kept_blocks: int = 0
    total_blocks: int = 0

    @property
    def kept_fraction(self) -> float:
        return self.kept_blocks / self.total_blocks if self.total_blocks else 1.0


@dataclass
class RemapScratch:
    """Shape-keyed tensor cache with **stable pointers**.

    CUDA graph replay bakes in the pointer of every tensor the captured
    launch touched, so a freshly allocated ``seq_lens`` each step would be
    invisible to a replayed graph.  Keying on shape and reusing the buffer
    keeps the pointer stable for the life of the engine (guide 6.4/6.6).
    """

    _cache: dict[tuple[Any, ...], torch.Tensor] = field(default_factory=dict)

    def get(
        self, key: str, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        k = (key, shape, dtype, str(device))
        buf = self._cache.get(k)
        if buf is None:
            buf = torch.empty(shape, dtype=dtype, device=device)
            self._cache[k] = buf
        return buf

    def clear(self) -> None:
        self._cache.clear()


def _ceil_div(a: torch.Tensor, b: int) -> torch.Tensor:
    return torch.div(a + (b - 1), b, rounding_mode="floor")


def block_allowed(
    mask: torch.Tensor, num_rows: int, num_blocks: int, block_size: int
) -> torch.Tensor:
    """OR-reduce the mask into ``(num_rows, num_blocks)`` block granularity.

    Positions past the end of the mask allocation count as allowed: a sequence
    longer than ``max_model_len`` must degrade to full attention, never to a
    truncated block list.
    """
    device = mask.device
    if num_rows == 0 or num_blocks == 0:
        return torch.zeros((num_rows, num_blocks), dtype=torch.bool, device=device)
    allowed = torch.ones((num_rows, num_blocks), dtype=torch.bool, device=device)
    avail_blocks = min(num_blocks, mask.shape[1] // block_size)
    # Rows past the end of the mask allocation also count as allowed: a batch
    # wider than max_num_seqs must degrade to full attention, never to a
    # truncated block list.
    avail_rows = min(num_rows, mask.shape[0])
    if avail_blocks > 0 and avail_rows > 0:
        window = mask[:avail_rows, : avail_blocks * block_size]
        allowed[:avail_rows, :avail_blocks] = window.unflatten(
            1, (avail_blocks, block_size)
        ).any(-1)
    return allowed


def remap_readable(
    *,
    mask: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out_seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    block_size: int,
    stats: RemapStats | None = None,
) -> None:
    """Reference implementation.  Mutates ``block_table`` in place.

    ``out_seq_lens`` is written, never ``seq_lens``: ``seq_lens`` is shared
    across KV-cache groups in the model runner while ``block_table`` is per
    group.  Shrinking the shared tensor in place for the full-attention group
    corrupted the sliding-window group's kernel on Gemma -- near-baseline NLL
    with garbage decoded text (guide 6.4).
    """
    num_rows, table_width = block_table.shape
    out_seq_lens[:num_rows].copy_(seq_lens[:num_rows])
    if num_rows == 0:
        return

    kept_total = 0
    blocks_total = 0
    for r in range(num_rows):
        sl = int(seq_lens[r].item())
        num_blocks = min(-(-sl // block_size), table_width)
        if num_blocks <= 0:
            continue
        allowed = block_allowed(mask[r : r + 1], 1, num_blocks, block_size)[0]
        blocks_total += num_blocks
        # Rows in prefill (query length above 1) and rows with nothing pruned
        # are left untouched.  Chunked prefill therefore needs no handling.
        if int(query_lens[r].item()) != 1 or bool(allowed.all().item()):
            kept_total += num_blocks
            continue
        kept_cols = torch.nonzero(allowed, as_tuple=False).flatten()
        n_kept = int(kept_cols.numel())
        if n_kept == 0:
            # Cannot happen with a real DA mask (the tail block is always
            # kept); refuse to emit a zero-length sequence if it ever does.
            kept_total += num_blocks
            continue
        kept_total += n_kept
        block_table[r, :n_kept] = block_table[r, kept_cols]
        tail_kept = bool(allowed[num_blocks - 1].item())
        tail_valid_len = sl - (num_blocks - 1) * block_size
        out_seq_lens[r] = (n_kept - int(tail_kept)) * block_size + (
            tail_valid_len if tail_kept else 0
        )
    if stats is not None:
        stats.kept_blocks = kept_total
        stats.total_blocks = blocks_total


def remap_optimized(
    *,
    mask: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    out_seq_lens: torch.Tensor,
    query_lens: torch.Tensor,
    block_size: int,
    scratch: RemapScratch,
    stats: RemapStats | None = None,
) -> None:
    """Sync-free equivalent of :func:`remap_readable`.

    Every decision stays on the GPU: ``should_remap`` is a device tensor, there
    is no early exit, and no value is ever read back to the host unless
    ``stats`` is requested (which is why kept-fraction logging must be off
    while timing).
    """
    num_rows, table_width = block_table.shape
    out_seq_lens[:num_rows].copy_(seq_lens[:num_rows])
    if num_rows == 0 or table_width == 0:
        return

    device = block_table.device
    idx = torch.arange(table_width, device=device)
    num_blocks = _ceil_div(seq_lens[:num_rows], block_size).clamp_(max=table_width)
    valid = idx.unsqueeze(0) < num_blocks.unsqueeze(1)
    allowed = block_allowed(mask, num_rows, table_width, block_size)
    kept = valid & allowed

    n_kept = kept.sum(dim=1)
    pruned_any = (valid & ~allowed).any(dim=1)
    should_remap = (query_lens[:num_rows] == 1) & pruned_any & (n_kept > 0)

    # Dense scatter into an (R, table_width + 1) buffer: unkept blocks go to the
    # sentinel column, kept blocks to their compacted position.  The buffer is
    # pre-filled with the original table so columns past the compacted end keep
    # their original values, exactly as the readable version leaves them.
    buf = scratch.get(
        "scatter", (num_rows, table_width + 1), block_table.dtype, device
    )
    buf[:, :table_width].copy_(block_table)
    buf[:, table_width] = 0
    positions = kept.cumsum(dim=1) - 1
    dest = torch.where(kept, positions, torch.full_like(positions, table_width))
    buf.scatter_(1, dest, block_table)
    block_table.copy_(
        torch.where(should_remap.unsqueeze(1), buf[:, :table_width], block_table)
    )

    tail_idx = (num_blocks - 1).clamp(min=0)
    tail_kept = kept.gather(1, tail_idx.unsqueeze(1)).squeeze(1)
    tail_valid_len = seq_lens[:num_rows] - tail_idx * block_size
    new_seq_lens = (n_kept - tail_kept.to(n_kept.dtype)) * block_size + torch.where(
        tail_kept, tail_valid_len.to(n_kept.dtype), torch.zeros_like(n_kept)
    )
    out_seq_lens[:num_rows].copy_(
        torch.where(should_remap, new_seq_lens.to(seq_lens.dtype), seq_lens[:num_rows])
    )
    # ``max_seq_len`` is deliberately left at the uncompacted value: a safe
    # over-estimate, and reading a new maximum back would cost a sync.  Kernels
    # stop at ``seq_lens``.

    if stats is not None:
        stats.kept_blocks = int(kept.sum().item())
        stats.total_blocks = int(valid.sum().item())
