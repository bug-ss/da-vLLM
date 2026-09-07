"""The shared allowed-KV mask: the one piece of state both halves touch.

A module-global boolean tensor of shape ``(max_num_seqs, max_model_len)``,
initialized True.  Row *i* is the allowed-KV mask for batch slot *i*.  The
logits processor writes it; the attention metadata patch reads it.

At ``1024 x 262144`` this is 256 MB, so set ``max_num_seqs`` to your real
concurrency rather than letting vLLM's default stand (guide 6.3).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import torch

from ..state_machine import MaskSnapshot, align_spans

_STORE: "SharedMaskStore | None" = None


class SharedMaskStore:
    def __init__(
        self,
        max_num_seqs: int,
        max_model_len: int,
        device: torch.device | str = "cuda",
        *,
        pin_memory: bool = True,
    ) -> None:
        self.max_num_seqs = int(max_num_seqs)
        self.max_model_len = int(max_model_len)
        self.device = torch.device(device)
        self.pin_memory = pin_memory and self.device.type == "cuda"
        self.tensor = torch.ones(
            (self.max_num_seqs, self.max_model_len), dtype=torch.bool, device=self.device
        )
        self._staging: dict[int, torch.Tensor] = {}
        #: One CUDA event per staging buffer, recorded after its copy. The
        #: copy is asynchronous, so refilling the buffer on the host before it
        #: has drained would race the transfer -- "pinned and on the same
        #: stream" makes the copy ordered against other GPU work, not against
        #: the CPU.
        self._staging_done: dict[int, torch.cuda.Event] = {}

    # -- lifecycle ---------------------------------------------------------

    def reset_row(self, row: int) -> None:
        """All-True.  Called for every active slot with no DA entry, so a slot
        recycled from a DA request never keeps stale mask data (guide 6.2)."""
        self.tensor[row].fill_(True)

    def reset_rows(self, rows: Iterable[int]) -> None:
        for r in rows:
            self.reset_row(r)

    def nbytes(self) -> int:
        return self.tensor.numel() * self.tensor.element_size()

    # -- writers -----------------------------------------------------------

    def write_snapshot(
        self,
        row: int,
        snapshot: MaskSnapshot,
        *,
        block_size: int | None = None,
        optimized: bool = True,
    ) -> None:
        """Write one request's mask into its slot.

        ``block_size`` is the value learned from the full-attention metadata
        builder.  Before the first build call it is None and the mask is
        written token-granular; the remap's OR-reduce rounds outward either
        way, so the two agree.
        """
        if snapshot.is_full:
            self.reset_row(row)
            return
        spans = snapshot.spans
        if block_size:
            spans = align_spans(spans, block_size)
        if optimized:
            self._write_spans_gpu(row, spans, snapshot.boundary, block_size)
        else:
            self._write_dense_pinned(row, spans, snapshot.boundary, block_size)

    def _write_spans_gpu(
        self,
        row: int,
        spans: Sequence[tuple[int, int]],
        boundary: int,
        block_size: int | None,
    ) -> None:
        """A few block-aligned GPU slice fills.

        Replaces staging a dense row through host memory, which produced 8 to
        36 ms tail spikes (guide 7).
        """
        r = self.tensor[row]
        r.fill_(False)
        for s, e in spans:
            s = max(0, min(s, self.max_model_len))
            e = max(0, min(e, self.max_model_len))
            if e > s:
                r[s:e] = True
        tail = min(boundary, self.max_model_len)
        if block_size:
            tail = (tail // block_size) * block_size
        if tail < self.max_model_len:
            r[tail:] = True

    def _write_dense_pinned(
        self,
        row: int,
        spans: Sequence[tuple[int, int]],
        boundary: int,
        block_size: int | None,
    ) -> None:
        """Reference writer: stage a dense row, OR-reduce, copy non-blocking.

        The non-blocking copy is safe only because the buffer is pinned,
        per-slot, and issued on the same stream (guide 6.3).
        """
        buf = self._staging.get(row)
        if buf is None:
            buf = torch.zeros(
                self.max_model_len,
                dtype=torch.bool,
                pin_memory=self.pin_memory,
            )
            self._staging[row] = buf
        else:
            pending = self._staging_done.get(row)
            if pending is not None:
                pending.synchronize()  # last step's copy has drained
        buf.fill_(False)
        for s, e in spans:
            s = max(0, min(s, self.max_model_len))
            e = max(0, min(e, self.max_model_len))
            if e > s:
                buf[s:e] = True
        tail = min(boundary, self.max_model_len)
        if tail < self.max_model_len:
            buf[tail:] = True
        if block_size:
            n = (self.max_model_len // block_size) * block_size
            if n:
                view = buf[:n].unflatten(0, (n // block_size, block_size))
                view.copy_(view.any(dim=1, keepdim=True).expand_as(view))
        self.tensor[row].copy_(buf, non_blocking=self.pin_memory)
        if self.pin_memory:
            event = self._staging_done.get(row)
            if event is None:
                event = self._staging_done[row] = torch.cuda.Event()
            event.record()


def install_shared_mask(
    max_num_seqs: int,
    max_model_len: int,
    device: torch.device | str = "cuda",
    *,
    force: bool = False,
) -> SharedMaskStore:
    global _STORE
    if _STORE is not None and not force:
        return _STORE
    _STORE = SharedMaskStore(max_num_seqs, max_model_len, device)
    return _STORE


def get_shared_mask() -> SharedMaskStore | None:
    return _STORE


def clear_shared_mask() -> None:
    global _STORE
    _STORE = None
