"""Capture real attention-metadata arguments for offline kernel replay.

Validation check #4 (guide 9.4) is "capture real kernel arguments from the
engine, replay offline, shrink the sequence lengths, and check time scales with
the kept fraction".  This is the capture half: it records the shapes and the
block tables the patched metadata builder actually produced, so a replay can
reconstruct a decode-shaped kernel call without keeping the KV cache itself.

Two constraints, both from the guide:

* **Capture in eager mode.**  A Python monkeypatch on a kernel is bypassed
  inside a replayed CUDA graph, so a captured graph tells you nothing.
* **Never leave it on while timing.**  Capture copies tensors to the host every
  step; that is a sync per step, exactly what the sync-free remap exists to
  avoid.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class CapturedStep:
    """One decode step's attention metadata, as the kernel saw it."""

    step: int
    block_size: int
    num_rows: int
    seq_lens: list[int]
    query_lens: list[int]
    kept_blocks: list[int]
    total_blocks: list[int]
    backend: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def kept_fraction(self) -> float:
        total = sum(self.total_blocks)
        return sum(self.kept_blocks) / total if total else 1.0


def write_capture(path: str | Path, steps: Sequence[CapturedStep]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for step in steps:
            fh.write(json.dumps(asdict(step)) + "\n")
    return len(steps)


def read_capture(path: str | Path) -> list[CapturedStep]:
    out: list[CapturedStep] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(CapturedStep(**json.loads(line)))
    return out


class MetadataCapture:
    """Records what the patched builder produced, step by step.

    ``install()`` wraps the already-installed DA hook rather than replacing it,
    so what gets captured is the post-remap state the kernel really reads.
    """

    def __init__(self, limit: int = 64) -> None:
        self.limit = limit
        self.steps: list[CapturedStep] = []
        self._installed: list[tuple[Any, Any]] = []
        self._step = 0
        self._warned = False

    def _warn_no_baseline(self) -> None:
        if not self._warned:
            self._warned = True
            logger.warning(
                "da: capture has no pre-compaction sequence lengths, so the "
                "kept fraction will read as 1.0. Install the capture through "
                "MetadataCapture.install(), which supplies them."
            )

    def record(
        self,
        metadata: Any,
        block_size: int,
        backend: str = "",
        original_seq_lens: Any = None,
    ) -> None:
        """Record one step.

        ``original_seq_lens`` is the length BEFORE compaction.  Without it there
        is nothing to compare against: the metadata's own ``seq_lens`` has
        already been shortened, so kept and total would both be derived from it
        and the ratio would be 1.0 for every input -- which is exactly the
        "nothing was pruned" answer this harness exists to disprove.
        """
        if len(self.steps) >= self.limit:
            return
        block_table = getattr(metadata, "block_table", None)
        seq_lens = getattr(metadata, "seq_lens", None)
        if block_table is None or seq_lens is None:
            return
        import math

        rows = block_table.shape[0]
        width = block_table.shape[1]
        lens = [int(x) for x in seq_lens.tolist()[:rows]]
        kept = [min(width, max(0, math.ceil(n / block_size))) for n in lens]
        if original_seq_lens is None:
            self._warn_no_baseline()
            total = kept
        else:
            before = [int(x) for x in original_seq_lens.tolist()[:rows]]
            total = [min(width, max(0, math.ceil(n / block_size))) for n in before]
        self.steps.append(
            CapturedStep(
                step=self._step,
                block_size=block_size,
                num_rows=int(block_table.shape[0]),
                seq_lens=lens,
                query_lens=[],
                kept_blocks=kept,
                total_blocks=total,
                backend=backend,
                meta={"table_width": int(block_table.shape[1])},
            )
        )
        self._step += 1

    def install(self) -> int:
        """Wrap every builder the DA patch already hooked.  Eager mode only."""
        from ..masking.patch import _import_vllm_bits, get_patch_state

        if get_patch_state() is None:
            raise RuntimeError(
                "install the DA patch first: capture wraps it, so that what is "
                "recorded is the post-remap state the kernel reads"
            )
        bits = _import_vllm_bits()
        count = 0
        for name in ("FlashAttentionMetadataBuilder", "TritonAttentionMetadataBuilder"):
            cls = bits.get(name)
            if cls is None:
                continue
            current = cls.build
            capture = self

            def make(fn, backend):
                def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kw):
                    # Snapshot the lengths BEFORE the DA hook compacts them.
                    before = getattr(common_attn_metadata, "seq_lens", None)
                    before = before.clone() if before is not None else None
                    metadata = fn(
                        self, common_prefix_len, common_attn_metadata,
                        fast_build=fast_build, **kw,
                    )
                    block_size = getattr(self, "block_size", None)
                    if block_size:
                        capture.record(
                            metadata, int(block_size), backend,
                            original_seq_lens=before,
                        )
                    return metadata

                return build

            cls.build = make(current, name)
            self._installed.append((cls, current))
            count += 1
        logger.warning(
            "da: metadata capture installed on %d builder(s). This syncs every "
            "step -- never leave it on while timing.",
            count,
        )
        return count

    def uninstall(self) -> None:
        for cls, original in self._installed:
            cls.build = original
        self._installed.clear()

    def __enter__(self) -> "MetadataCapture":
        self.install()
        return self

    def __exit__(self, *exc) -> None:
        self.uninstall()


def kept_fraction_series(steps: Sequence[CapturedStep]) -> list[float]:
    """The realized kept fraction per captured step.

    Compare this with the token-level metric from
    :func:`da_vllm.metrics.replay.replay`: the block figure must be the larger
    of the two (guide 9.3).
    """
    return [s.kept_fraction for s in steps]
