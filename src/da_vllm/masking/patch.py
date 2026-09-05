"""The vLLM attention-metadata-builder hook.

Monkeypatches ``build(self, common_prefix_len, common_attn_metadata,
fast_build=False)`` on both ``FlashAttentionMetadataBuilder`` and
``TritonAttentionMetadataBuilder`` (vLLM 0.20.2 signatures).  Qwen dense and MoE
models route to FlashAttention; Gemma 4 ships as a multimodal wrapper class that
FlashAttention refuses to bind, so vLLM routes it to the Triton unified
attention kernel.  **Both** builders must be hooked.

The wrapper, in order (guide 6.4):

1. Force ``common_prefix_len = 0``.  Cascade attention assumes all requests
   share the same first N KV positions; after per-request compaction, slot 0 of
   two requests points at different physical blocks.
2. Call the original ``build``.
3. Return immediately for a ``SlidingWindowSpec`` group.  A sliding-window
   cache physically holds only the last ``window`` tokens, so a mask over older
   positions dereferences rotated slots.  Gated DeltaNet layers have no paged
   cache and never call the builder at all.
4. Record the block size for the full-attention group, preferring it over any
   sliding-window value already seen.  vLLM's warmup calls ``build`` for every
   group before the first real step, so a tentative sliding-window value always
   gets overridden.
5. Route ``seq_lens`` through a shape-keyed scratch tensor with a stable
   pointer.  ``seq_lens`` is shared across KV-cache groups; ``block_table`` is
   per group.
6. Mutate ``block_table`` in place -- the runner reassigns it per group and
   CUDA graph replay bakes the pointer.
7. Run the remap with **the calling builder's own** ``self.block_size``, never
   a cached global.  On Gemma-4-31B the sliding group uses 16 and the full
   group 32; compacting 32-token columns with 16-token addressing produces
   garbage block ids (first misdiagnosed as a RoPE issue).

The patch must tolerate vLLM's warmup and profile runs, which call the builder
before any real request exists.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from ..config import DAConfig
from .remap import RemapScratch, RemapStats, remap_optimized, remap_readable
from .shared import get_shared_mask

logger = logging.getLogger(__name__)

_STATE: "PatchState | None" = None


@dataclass
class PatchState:
    """Install-time state.  Env-gated behavior is read once, here, because a
    replayed CUDA graph will not re-read an environment variable."""

    config: DAConfig
    scratch: RemapScratch = field(default_factory=RemapScratch)
    block_size: int | None = None
    block_size_source: str | None = None
    build_calls: int = 0
    remap_calls: int = 0
    last_stats: RemapStats | None = None
    patched_targets: list[str] = field(default_factory=list)
    #: Builder classes whose ``supports_update_block_table`` we turned off.
    disabled_block_table_reuse: list[type] = field(default_factory=list)
    #: True while vLLM is capturing a CUDA graph.  The capture path runs
    #: through ``build`` and Triton's capture override then does
    #: ``seq_lens.fill_(1)`` on whatever tensor the metadata carries -- which
    #: would be our scratch buffer.  Never remap during capture.
    in_cudagraph_capture: bool = False
    #: One-shot warnings, so a structural surprise is loud once, not per step.
    _warned: set[str] = field(default_factory=set)

    def warn_once(self, key: str, message: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning("da: %s", message)

    def note_block_size(self, size: int, *, from_sliding_window: bool) -> None:
        """Learn the block size rather than hardcoding it (guide 3).

        vLLM's hybrid KV-cache manager picks the block size per layer type at
        engine start, and ``cache_config.block_size`` reports the recurrent
        state page size on hybrid models -- do not read it.
        """
        if from_sliding_window:
            if self.block_size is None:
                self.block_size = size
                self.block_size_source = "sliding_window"
            return
        if self.block_size_source != "full_attention" or self.block_size != size:
            logger.info("da: full-attention KV block size = %d", size)
        self.block_size = size
        self.block_size_source = "full_attention"


def get_patch_state() -> PatchState | None:
    return _STATE


def _import_vllm_bits() -> dict[str, Any]:
    """Import the vLLM symbols we patch.  Missing ones are simply absent."""
    out: dict[str, Any] = {}
    try:
        from vllm.v1.attention.backends.flash_attn import (  # type: ignore
            FlashAttentionMetadataBuilder,
        )

        out["FlashAttentionMetadataBuilder"] = FlashAttentionMetadataBuilder
    except Exception as exc:  # pragma: no cover - depends on the install
        logger.info("da: FlashAttention builder unavailable (%s)", exc)
    try:
        from vllm.v1.attention.backends.triton_attn import (  # type: ignore
            TritonAttentionMetadataBuilder,
        )

        out["TritonAttentionMetadataBuilder"] = TritonAttentionMetadataBuilder
    except Exception as exc:  # pragma: no cover
        logger.info("da: Triton builder unavailable (%s)", exc)
    try:
        from vllm.v1.kv_cache_interface import (  # type: ignore
            FullAttentionSpec,
            SlidingWindowSpec,
        )

        out["FullAttentionSpec"] = FullAttentionSpec
        out["SlidingWindowSpec"] = SlidingWindowSpec
    except Exception as exc:  # pragma: no cover
        logger.info("da: kv_cache_interface unavailable (%s)", exc)
    return out


def is_maskable_spec(spec: Any, bits: dict[str, Any] | None = None) -> bool:
    """True only for a cache group we can positively identify as full attention.

    The polarity matters.  vLLM 0.20.2 ships at least three rotating-cache
    specs and they do not share an attribute: ``SlidingWindowSpec`` has
    ``sliding_window``, ``ChunkedLocalAttentionSpec`` has
    ``attention_chunk_size``, ``MambaSpec`` has neither and is not an
    ``AttentionSpec`` at all.  Asking "is this sliding?" therefore answers
    *no* for a spec it has never heard of, and DA would compact a cache whose
    pages rotate -- a mask over older positions dereferences rotated slots.

    So the question is inverted: mask only what is provably
    ``FullAttentionSpec``.  Every gate in DA is biased toward declining.
    """
    bits = bits if bits is not None else {}
    full_cls = bits.get("FullAttentionSpec")
    if full_cls is not None:
        return isinstance(spec, full_cls)
    # No class to check against (vLLM not importable): fall back to refusing
    # anything that advertises a bounded window, and say so.
    if spec is None:
        return False
    return not any(
        getattr(spec, attr, None) is not None
        for attr in ("sliding_window", "attention_chunk_size")
    )


def _query_lens(state: PatchState, common_attn_metadata: Any, num_rows: int, device):
    qsl = getattr(common_attn_metadata, "query_start_loc", None)
    if qsl is None or qsl.numel() < num_rows + 1:
        # No query_start_loc: assume decode.  The remap is a no-op for prefill
        # rows anyway, and getting this wrong only costs a skipped remap.
        state.warn_once("no_qsl", "common_attn_metadata has no usable query_start_loc")
        return torch.ones(num_rows, dtype=torch.int32, device=device)
    return qsl[1 : num_rows + 1] - qsl[:num_rows]


def _apply_remap(state: PatchState, metadata: Any, common_attn_metadata: Any, block_size: int) -> None:
    store = get_shared_mask()
    if store is None:
        # Normal during vLLM's warmup and profile run, which call the builder
        # before any real request exists.  Persisting past warmup means the
        # mask will never be applied in this process -- the silent-no-op class
        # of bug -- so say so once, loudly.
        if state.build_calls > 64:
            state.warn_once(
                "no_store",
                f"no shared mask in pid {os.getpid()} after {state.build_calls} "
                "build calls: attention is unmasked here. With tensor parallel "
                "> 1 the attention workers never construct the logits "
                "processor; the paper's results all used tensor parallel 1 "
                "with one engine per GPU.",
            )
        return

    block_table = getattr(metadata, "block_table", None)
    seq_lens = getattr(metadata, "seq_lens", None)
    if block_table is None or seq_lens is None:
        state.warn_once(
            "no_tensors",
            f"{type(metadata).__name__} has no block_table/seq_lens; DA mask not applied",
        )
        return
    if block_table.dim() != 2:
        state.warn_once("bt_dim", f"unexpected block_table shape {tuple(block_table.shape)}")
        return

    num_rows = min(block_table.shape[0], seq_lens.shape[0], store.max_num_seqs)
    if block_table.shape[0] > store.max_num_seqs:
        state.warn_once(
            "rows_exceed_mask",
            f"block table has {block_table.shape[0]} rows but the shared mask "
            f"only has {store.max_num_seqs}: rows beyond it serve unmasked. "
            "Set max_num_seqs to your real concurrency.",
        )
    if num_rows == 0:
        return

    device = block_table.device
    query_lens = _query_lens(state, common_attn_metadata, num_rows, device)

    # seq_lens is shared across KV-cache groups; block_table is per group.
    # Shrinking the shared tensor in place corrupts the sliding-window group.
    out_seq_lens = state.scratch.get(
        "seq_lens", (num_rows,), seq_lens.dtype, device
    )
    stats = RemapStats() if state.config.log_kept_fraction else None

    view = block_table[:num_rows]
    if state.config.optimized_remap:
        remap_optimized(
            mask=store.tensor,
            block_table=view,
            seq_lens=seq_lens[:num_rows],
            out_seq_lens=out_seq_lens,
            query_lens=query_lens,
            block_size=block_size,
            scratch=state.scratch,
            stats=stats,
        )
    else:
        remap_readable(
            mask=store.tensor,
            block_table=view,
            seq_lens=seq_lens[:num_rows],
            out_seq_lens=out_seq_lens,
            query_lens=query_lens,
            block_size=block_size,
            stats=stats,
        )

    # Hand the kernel the scratch tensor rather than the shared one.  Its
    # pointer is stable across steps, which CUDA graph replay requires.
    if num_rows == seq_lens.shape[0]:
        metadata.seq_lens = out_seq_lens
    else:
        padded = state.scratch.get(
            "seq_lens_padded", tuple(seq_lens.shape), seq_lens.dtype, device
        )
        padded.copy_(seq_lens)
        padded[:num_rows].copy_(out_seq_lens)
        metadata.seq_lens = padded

    # max_seq_len is deliberately left at the uncompacted value: kernels stop
    # at seq_lens, and recomputing the maximum would cost a host sync.  We also
    # do not recompute FlashAttention's AOT scheduler metadata: it falls back
    # to non-AOT scheduling on stale metadata, and Triton never reads it.

    state.remap_calls += 1
    if stats is not None:
        state.last_stats = stats
        logger.info(
            "da: kept %d/%d blocks (%.3f)",
            stats.kept_blocks,
            stats.total_blocks,
            stats.kept_fraction,
        )


def _make_wrapper(original: Callable[..., Any], state: PatchState, bits: dict[str, Any]):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kwargs):
        # 1. Cascade attention assumes a shared prefix across requests; after
        #    per-request compaction that assumption is false.
        common_prefix_len = 0
        metadata = original(
            self, common_prefix_len, common_attn_metadata, fast_build=fast_build, **kwargs
        )
        state.build_calls += 1

        spec = getattr(self, "kv_cache_spec", None)
        block_size = getattr(self, "block_size", None)
        maskable = is_maskable_spec(spec, bits)
        if block_size:
            state.note_block_size(int(block_size), from_sliding_window=not maskable)
        if not maskable:
            return metadata
        if state.in_cudagraph_capture:
            # Capture runs through build(); a captured graph must not bake a
            # compacted table, and Triton's capture path overwrites seq_lens
            # in place afterwards.
            return metadata
        if not block_size:
            state.warn_once("no_bs", "builder has no block_size; DA mask not applied")
            return metadata

        try:
            # 7. The calling builder's own block size, never a cached global.
            _apply_remap(state, metadata, common_attn_metadata, int(block_size))
        except Exception:  # pragma: no cover - never take the engine down
            logger.exception("da: remap failed; serving this step unmasked")
        return metadata

    build.__name__ = getattr(original, "__name__", "build")
    build.__doc__ = original.__doc__
    build._da_original = original  # type: ignore[attr-defined]
    return build


def _make_capture_wrapper(original: Callable[..., Any], state: PatchState):
    """Mark the capture window so ``build`` skips the remap inside it."""

    def build_for_cudagraph_capture(self, common_attn_metadata, *args, **kwargs):
        previous = state.in_cudagraph_capture
        state.in_cudagraph_capture = True
        try:
            return original(self, common_attn_metadata, *args, **kwargs)
        finally:
            state.in_cudagraph_capture = previous

    build_for_cudagraph_capture._da_original = original  # type: ignore[attr-defined]
    return build_for_cudagraph_capture


def _disable_block_table_reuse(cls: type, state: PatchState) -> None:
    """Force the runner to call ``build`` for every KV-cache group.

    vLLM 0.20.2's model runner caches one metadata build per
    ``(kv_cache_spec, builder type)`` and gives later groups a shallow copy
    with only the block table swapped, when the builder sets
    ``supports_update_block_table``.  That copy keeps the *first* group's
    ``seq_lens`` -- which, under DA, is our compacted scratch tensor -- while
    carrying an uncompacted block table.  The kernel would then read the first
    N blocks of the full table: wrong KV, plausible-looking NLL.

    Rebuilding per group costs a little host time and removes the hazard.
    """
    if getattr(cls, "supports_update_block_table", False):
        state.disabled_block_table_reuse.append(cls)
        cls.supports_update_block_table = False
        logger.info(
            "da: disabled block-table reuse on %s so every KV-cache group "
            "rebuilds its metadata",
            cls.__name__,
        )


def install_patch(config: DAConfig, *, force: bool = False) -> PatchState:
    """Install the metadata-builder hook **from inside the engine process**.

    vLLM V1 runs the model in an EngineCore subprocess: a monkeypatch applied in
    the parent process patches nothing.  The reliable places to call this are
    the constructor of a custom logits processor (vLLM instantiates it inside
    EngineCore) and, for tensor parallel > 1, a ``sitecustomize.py`` on
    ``PYTHONPATH`` -- attention workers are separately spawned processes that
    never construct the logits processor.  See
    :func:`da_vllm.masking.sitecustomize_path`.
    """
    global _STATE
    if _STATE is not None and not force:
        return _STATE

    state = PatchState(config=config)
    bits = _import_vllm_bits()
    for name in ("FlashAttentionMetadataBuilder", "TritonAttentionMetadataBuilder"):
        cls = bits.get(name)
        if cls is None:
            continue
        current = cls.build
        if getattr(current, "_da_original", None) is not None and not force:
            state.patched_targets.append(f"{name} (already patched)")
            continue
        original = getattr(current, "_da_original", current)
        cls.build = _make_wrapper(original, state, bits)
        capture = getattr(cls, "build_for_cudagraph_capture", None)
        if capture is not None and getattr(capture, "_da_original", None) is None:
            cls.build_for_cudagraph_capture = _make_capture_wrapper(capture, state)
        _disable_block_table_reuse(cls, state)
        state.patched_targets.append(name)

    if not state.patched_targets:
        logger.warning(
            "da: no attention metadata builder was patched. The mask will do "
            "nothing. This is the silent-no-op failure mode; check that vLLM is "
            "importable in THIS process (pid %d).",
            os.getpid(),
        )
    else:
        logger.info("da: patched %s in pid %d", state.patched_targets, os.getpid())

    if config.triton_decode_num_stages:
        install_triton_num_stages(config.triton_decode_num_stages)

    _STATE = state
    return state


def uninstall_patch() -> None:
    """Restore the original builders.  Used by tests and A/B harnesses."""
    global _STATE
    bits = _import_vllm_bits()
    for name in ("FlashAttentionMetadataBuilder", "TritonAttentionMetadataBuilder"):
        cls = bits.get(name)
        if cls is None:
            continue
        original = getattr(cls.build, "_da_original", None)
        if original is not None:
            cls.build = original
        capture = getattr(cls, "build_for_cudagraph_capture", None)
        capture_original = getattr(capture, "_da_original", None)
        if capture_original is not None:
            cls.build_for_cudagraph_capture = capture_original
    if _STATE is not None:
        for cls in _STATE.disabled_block_table_reuse:
            cls.supports_update_block_table = True
    _STATE = None


class _KernelProxy:
    """Forwards ``kernel[grid](...)`` with extra launch kwargs injected.

    ``__getitem__`` is looked up on the *type*, not the instance, so a plain
    attribute assignment on the JITFunction would not intercept ``kernel[grid]``.
    Hence a proxy object that replaces the module-level name.
    """

    def __init__(self, kernel: Any, *, grid_rank: int | None = None, **launch_kwargs: Any):
        self._kernel = kernel
        self._grid_rank = grid_rank
        self._launch_kwargs = launch_kwargs

    def __getitem__(self, grid):
        launcher = self._kernel[grid]
        applies = self._grid_rank is None or (
            isinstance(grid, tuple) and len(grid) == self._grid_rank
        )

        def run(*args, **kwargs):
            if applies:
                for k, v in self._launch_kwargs.items():
                    kwargs.setdefault(k, v)
            return launcher(*args, **kwargs)

        return run

    def __getattr__(self, item):
        return getattr(self._kernel, item)


#: vLLM 0.20.2 unified the 2D and 3D Triton attention kernels into a single
#: ``kernel_unified_attention``; the decode (split-KV) path is the one launched
#: with a 3-element grid ``(q_blocks, kv_heads, softmax_segments)``.
_TRITON_KERNEL_MODULE = "vllm.v1.attention.ops.triton_unified_attention"
_TRITON_KERNEL_NAME = "kernel_unified_attention"
_TRITON_DECODE_GRID_RANK = 3


def install_triton_num_stages(num_stages: int = 2) -> bool:
    """Pipeline vLLM 0.20.2's Triton decode kernel.

    vLLM 0.20.2 launches the unified attention kernel with no ``num_stages``,
    so on the split-KV decode path its KV loop is unpipelined and latency-bound
    at low batch.  Passing ``num_stages=2`` is bit-exact and worth a few percent
    end to end on Gemma.  Entirely independent of DA -- it is here because it is
    free.

    Only the 3-element (decode) grid is touched; the prefill launch keeps
    whatever Triton picks for it.
    """
    try:
        import importlib

        mod = importlib.import_module(_TRITON_KERNEL_MODULE)
    except Exception as exc:  # pragma: no cover - depends on the install
        logger.info("da: %s unavailable (%s)", _TRITON_KERNEL_MODULE, exc)
        return False
    kernel = getattr(mod, _TRITON_KERNEL_NAME, None)
    if kernel is None:
        logger.warning(
            "da: %s has no %s; the num_stages tweak did nothing. vLLM renamed "
            "or restructured the kernel -- check the version.",
            _TRITON_KERNEL_MODULE,
            _TRITON_KERNEL_NAME,
        )
        return False
    if isinstance(kernel, _KernelProxy):
        return False
    setattr(
        mod,
        _TRITON_KERNEL_NAME,
        _KernelProxy(kernel, grid_rank=_TRITON_DECODE_GRID_RANK, num_stages=num_stages),
    )
    logger.info("da: triton decode kernel launched with num_stages=%d", num_stages)
    return True


def uninstall_triton_num_stages() -> bool:
    try:
        import importlib

        mod = importlib.import_module(_TRITON_KERNEL_MODULE)
    except Exception:  # pragma: no cover
        return False
    kernel = getattr(mod, _TRITON_KERNEL_NAME, None)
    if isinstance(kernel, _KernelProxy):
        setattr(mod, _TRITON_KERNEL_NAME, kernel._kernel)
        return True
    return False
