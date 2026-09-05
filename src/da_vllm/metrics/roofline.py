"""Roofline decode wall-time (guide 11, paper section 5.4 / appendix C).

One decode step on a single B200, bf16, charged at each operation's own
hardware ceiling:

* matmul term:     ``2 * active_params / (peak_flops * MFU)``
* global KV read:  ``attended_tokens * kv_bytes_per_token / (peak_bw * MBU)``
  -- the only term the mask reduces
* local read:      fixed per-step bytes of the sliding-window or recurrent
  layers ``/ (peak_bw * MBU)``

Excluded: prefill, normalization, RoPE, KV writes, and block-alignment
over-read.  Weights are charged as FLOPs, not bytes, and there is no batch
term.  **This is a ceiling at stated utilizations, not a measurement.**  Real
timing behaves differently and the reasons are documented in
:mod:`da_vllm.timing`.

``kv_bytes_per_token`` is derived from the live config or the checkpoint's
tensor shapes -- never from a note.  One 2x error here (Gemma's global head
dim) produced a false "under-delivers at high batch" finding, two theories
built on it, and a "correction" that was itself wrong and had to be retracted.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..models import AttentionGeometry, ModelSpec

# NVIDIA B200, the GPU behind every evaluation and efficiency number.
PEAK_FLOPS_BF16 = 2.25e15
PEAK_HBM_BYTES_PER_S = 8e12
DECODE_MFU = 0.40
ATTENTION_MBU = 0.70

#: Layer-type strings that mean "reads the whole KV cache every step".
FULL_ATTENTION_LAYER_TYPES = frozenset({"full_attention", "global_attention", "full"})


class GeometryError(ValueError):
    """Raised when a config does not state the global-layer geometry explicitly."""


def _get(config: Any, *names: str):
    for name in names:
        if isinstance(config, Mapping):
            if name in config:
                return config[name]
        elif hasattr(config, name):
            return getattr(config, name)
    return None


def count_global_layers(config: Any) -> tuple[int, int]:
    """``(num_layers, num_global_layers)`` from the config's ``layer_types``.

    Counting from ``layer_types`` -- rather than from a sliding-window pattern
    or a ratio -- is the only way that stays right on hybrid models.
    """
    layer_types: Sequence[str] | None = _get(config, "layer_types")
    if not layer_types:
        raise GeometryError(
            "config has no layer_types list; refusing to guess which layers are "
            "full attention"
        )
    n_global = sum(1 for t in layer_types if str(t).lower() in FULL_ATTENTION_LAYER_TYPES)
    if n_global == 0:
        raise GeometryError(f"no full-attention layers found in layer_types={list(layer_types)[:8]}")
    return len(layer_types), n_global


def global_kv_bytes_per_token(config: Any, *, dtype_bytes: int = 2) -> int:
    """bf16 KV bytes read per context token per step, global layers only.

    Reads ``global_head_dim`` / ``num_global_key_value_heads`` when the config
    states them separately (Gemma 4 does) and refuses to fall back to the
    sliding-window layers' values, which are a different size: applying the
    sliding geometry to the global layers doubles the byte count.
    """
    _, n_global = count_global_layers(config)

    head_dim = _get(config, "global_head_dim")
    kv_heads = _get(config, "num_global_key_value_heads")
    layer_types: Sequence[str] = _get(config, "layer_types")
    has_other_geometry = any(
        str(t).lower() not in FULL_ATTENTION_LAYER_TYPES for t in layer_types
    )
    if head_dim is None or kv_heads is None:
        if has_other_geometry and (
            _get(config, "sliding_window") is not None
            or _get(config, "linear_attention") is not None
        ):
            raise GeometryError(
                "this is a hybrid model but the config does not state "
                "global_head_dim / num_global_key_value_heads. Read the global "
                "layers' own head dims explicitly -- the sliding-window values "
                "are a different size."
            )
        head_dim = head_dim if head_dim is not None else _get(config, "head_dim")
        kv_heads = kv_heads if kv_heads is not None else _get(config, "num_key_value_heads")
    if head_dim is None or kv_heads is None:
        raise GeometryError("config states neither global nor default head geometry")
    return int(n_global) * 2 * int(kv_heads) * int(head_dim) * dtype_bytes


def global_kv_bytes_from_shapes(
    shapes: Mapping[str, Sequence[int]],
    global_layer_indices: Sequence[int],
    *,
    key_proj_template: str = "model.layers.{i}.self_attn.k_proj.weight",
    dtype_bytes: int = 2,
) -> int:
    """Derive the same number from the checkpoint's tensor shapes.

    ``k_proj.weight`` has shape ``(kv_heads * head_dim, hidden)``, so its first
    dimension *is* the per-token K size in elements.  K and V are symmetric.
    """
    total = 0
    for i in global_layer_indices:
        name = key_proj_template.format(i=i)
        if name not in shapes:
            raise GeometryError(f"missing {name} in checkpoint shapes")
        total += int(shapes[name][0])
    return total * 2 * dtype_bytes


def geometry_from_config(spec: ModelSpec, config: Any) -> AttentionGeometry:
    """Derive a usable geometry from a live config.

    This is the recommended path.  The registry is a cross-check; anything it
    carries that nobody has verified is marked ``source="placeholder"`` and the
    cost model refuses it.
    """
    n_layers, n_global = count_global_layers(config)
    head_dim = _get(config, "global_head_dim") or _get(config, "head_dim")
    kv_heads = _get(config, "num_global_key_value_heads") or _get(
        config, "num_key_value_heads"
    )
    # Recompute through the strict path so a hybrid config missing the global
    # head dims raises here too, rather than silently borrowing sliding values.
    derived_bytes = global_kv_bytes_per_token(config)
    g = spec.geometry
    geometry = dataclasses.replace(
        g,
        num_layers=n_layers,
        num_global_layers=n_global,
        global_kv_heads=int(kv_heads),
        global_head_dim=int(head_dim),
        source="derived",
    )
    if geometry.global_kv_bytes_per_token != derived_bytes:  # pragma: no cover
        raise GeometryError(
            "internal inconsistency deriving geometry: "
            f"{geometry.global_kv_bytes_per_token} != {derived_bytes}"
        )
    return geometry


def verify_geometry(spec: ModelSpec, config: Any) -> None:
    """Cross-check the registry against the live config.  Raises on mismatch."""
    n_layers, n_global = count_global_layers(config)
    derived = global_kv_bytes_per_token(config)
    g = spec.geometry
    problems = []
    if n_layers != g.num_layers:
        problems.append(f"num_layers {n_layers} != registry {g.num_layers}")
    if n_global != g.num_global_layers:
        problems.append(f"global layers {n_global} != registry {g.num_global_layers}")
    if derived != g.global_kv_bytes_per_token:
        problems.append(
            f"kv bytes/token {derived} != registry {g.global_kv_bytes_per_token}"
        )
    if problems:
        raise GeometryError(
            f"{spec.hub_id}: registry disagrees with the live config: "
            + "; ".join(problems)
            + ". The live config wins -- fix the registry."
        )


@dataclass(frozen=True)
class RooflineBreakdown:
    matmul_s: float
    global_kv_s: float
    local_s: float

    @property
    def total_s(self) -> float:
        return self.matmul_s + self.global_kv_s + self.local_s


def roofline_response(
    *,
    geometry: AttentionGeometry,
    active_params: int,
    attended_tokens: int,
    decode_steps: int,
    peak_flops: float = PEAK_FLOPS_BF16,
    peak_bw: float = PEAK_HBM_BYTES_PER_S,
    mfu: float = DECODE_MFU,
    mbu: float = ATTENTION_MBU,
    allow_placeholder_geometry: bool = False,
) -> RooflineBreakdown:
    """Roofline decode wall-time for one whole response.

    ``attended_tokens`` is the sum over steps (the DA metric); ``decode_steps``
    multiplies the two context-independent terms.

    Refuses an unverified geometry unless you say so explicitly.  Derive the
    real numbers with :func:`geometry_from_config` instead: a 2x byte error
    here produced a false finding, two theories built on it, and a
    "correction" that was itself wrong.
    """
    if not geometry.verified and not allow_placeholder_geometry:
        raise GeometryError(
            "this model's attention geometry is a PLACEHOLDER -- nobody has "
            "verified its layer counts or head dims. Derive it from the live "
            "config with geometry_from_config(spec, hf_config), or pass "
            "allow_placeholder_geometry=True and do not publish the number."
        )
    matmul = decode_steps * (2 * active_params) / (peak_flops * mfu)
    global_kv = attended_tokens * geometry.global_kv_bytes_per_token / (peak_bw * mbu)
    local = decode_steps * geometry.local_bytes_per_step / (peak_bw * mbu)
    return RooflineBreakdown(matmul, global_kv, local)
