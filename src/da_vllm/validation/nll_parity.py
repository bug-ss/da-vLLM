"""Three-column NLL parity -- validation check #1 (guide 9.1).

For each prompt, greedy-decode twice through vLLM: once with vanilla attention
and once with DA.  Then teacher-force in HuggingFace three ways and record the
per-token NLL on the completion:

===  ============================================  =====================
col  response                                      mask
===  ============================================  =====================
v    vanilla response                              causal
d    DA response                                   4D DA mask
dv   DA response                                   plain causal
===  ============================================  =====================

Require ``v ~ d < dv``.

**Two columns are not enough.**  When the patch was silently not applied, v and
d agreed perfectly and everything looked fine for weeks.  The ``dv`` column is
what proves the mask changed the computation: the DA response must be *less*
likely under full attention than under the mask it was actually generated with.

Two properties the reference must have or it diverges by an order of magnitude:

* it OR-reduces the KV dimension at the **served block size**, matching the
  engine's block alignment;
* on hybrid models it passes a **per-layer-type** mask -- the DA mask on
  full-attention layers, the sliding mask on sliding-window layers.

Expect a baseline gap of 0.05 to 0.15 nats between vLLM and HF from bf16 drift.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ..metrics.roofline import FULL_ATTENTION_LAYER_TYPES
from ..state_machine import MaskSnapshot

logger = logging.getLogger(__name__)

#: bf16 drift between the two stacks.  A larger gap means something structural.
BASELINE_NLL_TOLERANCE = 0.15


def block_align_mask(mask: torch.Tensor, block_size: int) -> torch.Tensor:
    """OR-reduce the KV dimension in groups of ``block_size``, in place-safe.

    ``mask`` is ``(..., kv_len)`` boolean.  A block is kept if any position in
    it is kept -- exactly what the engine's block table does.
    """
    if block_size <= 1:
        return mask
    kv_len = mask.shape[-1]
    n = (kv_len // block_size) * block_size
    out = mask.clone()
    if n:
        view = out[..., :n].unflatten(-1, (n // block_size, block_size))
        out[..., :n] = view.any(dim=-1, keepdim=True).expand_as(view).flatten(-2)
    if n < kv_len:
        tail = out[..., n:].any(dim=-1, keepdim=True)
        out[..., n:] = tail.expand_as(out[..., n:])
    return out


def da_mask_4d(
    snapshot: MaskSnapshot,
    prompt_len: int,
    total_len: int,
    block_size: int,
    *,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """``(1, 1, total_len, total_len)`` boolean allow-mask for one frozen mode.

    Causal, plus the DA row pattern for every query position at or after the
    first generated token.  Prompt rows keep full causal attention: the prompt
    was prefilled unmasked.
    """
    allow = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)
    causal = torch.tril(torch.ones((total_len, total_len), dtype=torch.bool, device=device))
    row = torch.zeros(total_len, dtype=torch.bool, device=device)
    for s, e in snapshot.spans:
        row[max(0, s) : min(total_len, e)] = True
    row[min(snapshot.boundary, total_len) :] = True
    row = block_align_mask(row.unsqueeze(0), block_size).squeeze(0)
    allow[:prompt_len] = causal[:prompt_len]
    allow[prompt_len:] = causal[prompt_len:] & row.unsqueeze(0)
    return allow.unsqueeze(0).unsqueeze(0)


def sliding_mask_4d(
    total_len: int, window: int, *, device: torch.device | str = "cpu"
) -> torch.Tensor:
    idx = torch.arange(total_len, device=device)
    delta = idx.unsqueeze(1) - idx.unsqueeze(0)
    allow = (delta >= 0) & (delta < window)
    return allow.unsqueeze(0).unsqueeze(0)


def _layer_types(hf_config: Any, num_layers: int) -> list[str]:
    types = getattr(hf_config, "layer_types", None)
    if types:
        return [str(t).lower() for t in types]
    return ["full_attention"] * num_layers


@contextlib.contextmanager
def per_layer_masks(model: Any, full_mask: torch.Tensor, other_mask: torch.Tensor | None):
    """Substitute a different 4D mask per layer type for the duration.

    A single mask for the whole model makes the reference diverge by an order
    of magnitude on a hybrid backbone: the sliding-window layers get a mask over
    positions their cache does not hold.
    """
    layers = model.model.layers
    types = _layer_types(model.config, len(layers))
    originals = []
    for layer, kind in zip(layers, types):
        mask = full_mask if kind in FULL_ATTENTION_LAYER_TYPES else other_mask
        if mask is None:
            originals.append(None)
            continue
        original = layer.forward
        originals.append(original)

        def make(fn, m):
            def forward(*args, **kwargs):
                kwargs["attention_mask"] = m
                return fn(*args, **kwargs)

            return forward

        layer.forward = make(original, mask)
    try:
        yield
    finally:
        for layer, original in zip(layers, originals):
            if original is not None:
                layer.forward = original


def _to_additive(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Boolean allow-mask -> the additive float mask HF attention expects."""
    return torch.zeros_like(mask, dtype=dtype).masked_fill(
        ~mask, torch.finfo(dtype).min
    )


@torch.no_grad()
def teacher_force_nll(
    model: Any,
    input_ids: torch.Tensor,
    completion_start: int,
    *,
    full_mask: torch.Tensor | None = None,
    other_mask: torch.Tensor | None = None,
) -> float:
    """Mean per-token NLL over ``input_ids[completion_start:]``."""
    dtype = next(model.parameters()).dtype
    fm = None if full_mask is None else _to_additive(full_mask, dtype)
    om = None if other_mask is None else _to_additive(other_mask, dtype)
    ctx = per_layer_masks(model, fm, om) if fm is not None else contextlib.nullcontext()
    with ctx:
        out = model(input_ids=input_ids)
    logits = out.logits[0, completion_start - 1 : -1].float()
    targets = input_ids[0, completion_start:]
    return float(
        torch.nn.functional.cross_entropy(logits, targets, reduction="mean").item()
    )


@dataclass(frozen=True)
class ParityResult:
    vanilla_nll: float  # v
    da_masked_nll: float  # d
    da_causal_nll: float  # dv
    tolerance: float = BASELINE_NLL_TOLERANCE

    @property
    def v_matches_d(self) -> bool:
        return abs(self.vanilla_nll - self.da_masked_nll) <= self.tolerance

    @property
    def mask_is_applied(self) -> bool:
        """``d < dv``.  If these are equal, the patch is doing nothing."""
        return self.da_masked_nll < self.da_causal_nll - 1e-4

    @property
    def passed(self) -> bool:
        return self.v_matches_d and self.mask_is_applied

    def explain(self) -> str:
        if self.passed:
            return (
                f"PASS v={self.vanilla_nll:.4f} ~ d={self.da_masked_nll:.4f} < "
                f"dv={self.da_causal_nll:.4f}"
            )
        if not self.mask_is_applied:
            return (
                f"FAIL d={self.da_masked_nll:.4f} is not below dv="
                f"{self.da_causal_nll:.4f}: the DA mask changed nothing. This is "
                "the silent-no-op signature -- check the patch is installed in "
                "the EngineCore process."
            )
        return (
            f"FAIL v={self.vanilla_nll:.4f} and d={self.da_masked_nll:.4f} differ "
            f"by more than {self.tolerance} nats"
        )


def three_column_parity(
    model: Any,
    tokenizer: Any,
    *,
    prompt_ids: Sequence[int],
    vanilla_response_ids: Sequence[int],
    da_response_ids: Sequence[int],
    da_snapshot: MaskSnapshot,
    block_size: int,
    sliding_window: int | None = None,
    device: torch.device | str = "cpu",
) -> ParityResult:
    """Run all three columns and return the verdict."""
    prompt_len = len(prompt_ids)

    v_ids = torch.tensor([list(prompt_ids) + list(vanilla_response_ids)], device=device)
    v = teacher_force_nll(model, v_ids, prompt_len)

    d_ids = torch.tensor([list(prompt_ids) + list(da_response_ids)], device=device)
    total = d_ids.shape[1]
    full_mask = da_mask_4d(da_snapshot, prompt_len, total, block_size, device=device)
    other = (
        sliding_mask_4d(total, sliding_window, device=device)
        if sliding_window
        else None
    )
    d = teacher_force_nll(model, d_ids, prompt_len, full_mask=full_mask, other_mask=other)

    dv = teacher_force_nll(model, d_ids, prompt_len)
    return ParityResult(v, d, dv)
