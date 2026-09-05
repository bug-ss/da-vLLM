"""Training-side helpers (guide section 13).

**Not part of the paper's results.**  The paper is zero-shot; nothing here was
used to produce a published number.  It is the set of things that held up when
fine-tuning a model on DA traces, kept because each one encodes a specific
failure.

* Render training prompts through :class:`~da_vllm.prompt.PromptRenderer` and
  pass ``tools=`` to the chat template, so the assistant token mask covers the
  synthetic tool turns correctly.  Loss on assistant tokens only.
* Replay the state machine over each training response to build a per-sample 4D
  mask (sink, local window, response, focus spans), then compile it to a
  FlexAttention block mask.  Dense 4D masks do not scale.
* FlexAttention backward is wrong in bf16 when the sequence length is not a
  multiple of 128 with a block mask plus a score modifier (upstream PyTorch
  issue 153799).  Pad to 128.
* Gemma 4's head dim 256 needs about 196 KB of shared memory for Inductor's
  default flex configs; dispatch kernel options by head dim and the device's
  opt-in limit or an A100 crashes.
* Gemma 4 ships as a multimodal wrapper: load the text-only language model for
  training and propagate the model name through.
* Match the training mask's KV granularity to the served block size if you want
  train and serve to agree exactly.  The shipped configs did not, and the
  mismatch is a known approximation.
* Right padding, with the sink at positions 0 to 15, matches vLLM.  Left
  padding bought nothing once sink placement was derived from the attention
  mask.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from .config import DAConfig
from .detect import PromptMap, build_prompt_map
from .models import ModelSpec, resolve
from .prompt import PromptRenderer
from .state_machine import DAStateMachine, Mode, align_spans

logger = logging.getLogger(__name__)

#: FlexAttention backward is wrong in bf16 below this alignment.
FLEX_SEQ_MULTIPLE = 128


class MissingMaskError(RuntimeError):
    """A mask key vanished between the collator and the loss.

    Training-side 4D masks were once dropped by an inherited loss function, and
    an assistant mask fell through a ``not None`` guard.  Both were silent.
    """


@dataclass
class TrainingSample:
    input_ids: list[int]
    #: True where the loss applies: assistant-generated tokens only.
    assistant_mask: list[bool]
    #: One entry per query position: which mode group that row attends under.
    #: Prompt rows are group -1 (plain causal -- the prompt is prefilled
    #: unmasked, exactly as at serving time).
    group_of_query: list[int]
    #: The kept-KV spans of each group, in token space.
    group_spans: tuple[tuple[tuple[int, int], ...], ...]
    #: Position from which every KV is kept unconditionally (the prompt end).
    boundary: int
    prompt_len: int
    pad_len: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def total_len(self) -> int:
        return len(self.input_ids)


def require_mask(batch: dict[str, Any], key: str) -> Any:
    """Fetch a mask key, failing loudly if it is absent or None."""
    if key not in batch or batch[key] is None:
        raise MissingMaskError(
            f"batch is missing {key!r}; a dropped mask trains a model on full "
            f"attention while the config says otherwise. Keys present: "
            f"{sorted(batch)}"
        )
    return batch[key]


def build_sample(
    renderer: PromptRenderer,
    config: DAConfig,
    *,
    context: str,
    question: str,
    response_text: str,
    model: str | ModelSpec | None = None,
) -> TrainingSample:
    """Render one DA trace and replay the state machine over its response.

    Loss lands on the response only: the synthetic tool turns live in the
    prompt, which is exactly why the prompt must be rendered with ``tools=``
    through the same renderer the server uses.

    The replay produces one *group* per contiguous span of constant mode, and a
    per-query-position group assignment.  A single mask for the whole sample
    would be wrong: a ``<local>`` row and a ``<global>`` row of the same trace
    attend to different things.
    """
    spec = resolve(model or renderer.spec)
    prompt = renderer.render_da(context, question)
    prompt_ids = list(renderer.tokenizer.encode(prompt.text, add_special_tokens=False))
    response_ids = list(
        renderer.tokenizer.encode(response_text, add_special_tokens=False)
    )
    pmap: PromptMap = build_prompt_map(
        renderer.tokenizer,
        prompt.text,
        spec.family,
        config,
        prompt_token_ids=prompt_ids,
    )
    prompt_len = len(prompt_ids)

    sm = DAStateMachine(pmap, renderer.tokenizer, config)
    group_spans: list[tuple[tuple[int, int], ...]] = []
    group_of_query: list[int] = [-1] * prompt_len
    index_of: dict[tuple[tuple[int, int], ...], int] = {}
    for t in range(len(response_ids)):
        sm.advance(response_ids[: t + 1])
        spans = sm.snapshot().spans
        idx = index_of.get(spans)
        if idx is None:
            idx = len(group_spans)
            index_of[spans] = idx
            group_spans.append(spans)
        group_of_query.append(idx)

    return TrainingSample(
        input_ids=prompt_ids + response_ids,
        assistant_mask=[False] * prompt_len + [True] * len(response_ids),
        group_of_query=group_of_query,
        group_spans=tuple(group_spans),
        boundary=prompt_len,
        prompt_len=prompt_len,
        meta={
            "num_segments": len(pmap.segments),
            "detection_failure": pmap.failure_reason,
            "focus_attempts": sm.stats.focus_attempts,
            "num_groups": len(group_spans),
        },
    )


def per_step_modes(
    pmap: PromptMap, tokenizer, config: DAConfig, response_ids: Sequence[int]
) -> list[tuple[Mode, tuple[int, ...]]]:
    """Replay the trace and report the mode in force at every response step."""
    sm = DAStateMachine(pmap, tokenizer, config)
    out: list[tuple[Mode, tuple[int, ...]]] = []
    for t in range(len(response_ids)):
        sm.advance(list(response_ids[: t + 1]))
        out.append((sm.mode, sm.focus_ids))
    return out


def pad_to_flex_multiple(
    sample: TrainingSample, pad_token_id: int, multiple: int = FLEX_SEQ_MULTIPLE
) -> TrainingSample:
    """Right-pad to a multiple of 128, sink at positions 0 to 15.

    Right padding matches vLLM's layout, so the sink lands on the same tokens
    train and serve.  Left padding bought nothing once sink placement was
    derived from the attention mask.
    """
    n = sample.total_len
    target = int(math.ceil(n / multiple) * multiple)
    pad = target - n
    if pad == 0:
        return sample
    return TrainingSample(
        input_ids=sample.input_ids + [pad_token_id] * pad,
        assistant_mask=sample.assistant_mask + [False] * pad,
        # Padding rows attend causally and carry no loss; they never join a
        # mode group.
        group_of_query=sample.group_of_query + [-1] * pad,
        group_spans=sample.group_spans,
        boundary=sample.boundary,
        prompt_len=sample.prompt_len,
        pad_len=pad,
        meta=dict(sample.meta),
    )


def mask_mod_from_spans(
    spans: Sequence[tuple[int, int]],
    boundary: int,
    prompt_len: int,
    *,
    kv_block_size: int | None = None,
):
    """A FlexAttention ``mask_mod`` for one frozen DA mode.

    ``kv_block_size`` should be **the served KV block size** if you want train
    and serve to agree exactly; leaving it None trains against a token-granular
    mask, which is a known approximation.
    """
    kept = align_spans(spans, kv_block_size) if kv_block_size else tuple(spans)
    starts = torch.tensor([s for s, _ in kept], dtype=torch.long)
    ends = torch.tensor([e for _, e in kept], dtype=torch.long)

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001 - FlexAttention signature
        causal = kv_idx <= q_idx
        prompt_row = q_idx < prompt_len
        past_boundary = kv_idx >= boundary
        in_span = torch.zeros_like(kv_idx, dtype=torch.bool)
        for s, e in zip(starts.tolist(), ends.tolist()):
            in_span = in_span | ((kv_idx >= s) & (kv_idx < e))
        return causal & (prompt_row | past_boundary | in_span)

    return mask_mod


def allowed_block_table(
    sample: TrainingSample, kv_len: int, kv_block_size: int
) -> torch.Tensor:
    """``(num_groups, num_kv_blocks)`` bool: which KV blocks each group keeps.

    Built at the **served** KV block size so train and serve agree exactly.
    Passing a different granularity is a known approximation, not a free
    choice.
    """
    num_blocks = math.ceil(kv_len / kv_block_size)
    table = torch.zeros((len(sample.group_spans), num_blocks), dtype=torch.bool)
    for g, spans in enumerate(sample.group_spans):
        for s, e in align_spans(spans, kv_block_size):
            table[g, s // kv_block_size : math.ceil(e / kv_block_size)] = True
        table[g, sample.boundary // kv_block_size :] = True
    return table


def mask_mod_from_trace(
    group_of_query: torch.Tensor, allowed: torch.Tensor, kv_block_size: int
):
    """A ``mask_mod`` covering a whole trace, one row pattern per mode group.

    Prompt and padding rows (group -1) stay plain causal: the prompt is
    prefilled unmasked at serving time too.
    """
    groups = group_of_query.long()
    table = allowed

    def mask_mod(b, h, q_idx, kv_idx):  # noqa: ARG001 - FlexAttention signature
        causal = kv_idx <= q_idx
        g = groups[q_idx]
        lookup = table[g.clamp(min=0), kv_idx // kv_block_size]
        return causal & torch.where(g < 0, torch.ones_like(lookup), lookup)

    return mask_mod


def compile_block_mask(mask_mod, q_len: int, kv_len: int, *, device="cuda", batch=1, heads=1):
    """Compile a ``mask_mod`` into a FlexAttention ``BlockMask``.

    Dense 4D masks do not scale past a few thousand tokens; the block mask is
    the only representation that fits a 100K-token trace.
    """
    from torch.nn.attention.flex_attention import create_block_mask  # type: ignore

    if q_len % FLEX_SEQ_MULTIPLE or kv_len % FLEX_SEQ_MULTIPLE:
        raise ValueError(
            f"pad to a multiple of {FLEX_SEQ_MULTIPLE} first: FlexAttention "
            "backward is wrong in bf16 otherwise with a block mask plus a score "
            "modifier (pytorch/pytorch#153799)"
        )
    return create_block_mask(mask_mod, batch, heads, q_len, kv_len, device=device)


#: Shared memory Inductor's default flex configs need at head dim 256.
GEMMA4_FLEX_SHARED_MEMORY_BYTES = 196 * 1024


def flex_kernel_options(head_dim: int, device_shared_memory_bytes: int) -> dict[str, int]:
    """Kernel options dispatched by head dim and the device's opt-in limit.

    Gemma 4's head dim 256 needs about 196 KB; an A100 (164 KB opt-in) crashes
    on the defaults, so the block sizes are cut instead.
    """
    if head_dim < 256:
        return {}
    if device_shared_memory_bytes >= GEMMA4_FLEX_SHARED_MEMORY_BYTES:
        return {"BLOCK_M": 128, "BLOCK_N": 64}
    return {"BLOCK_M": 32, "BLOCK_N": 32, "num_stages": 2, "num_warps": 4}


def load_text_only_model(model: str | ModelSpec, **kwargs):
    """Load the language model, unwrapping Gemma 4's multimodal wrapper.

    The model name is propagated onto the returned module so downstream code
    that keys on it (chat templates, the DA registry) still resolves.
    """
    from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore

    spec = resolve(model)
    config = AutoConfig.from_pretrained(spec.hub_id)
    loaded = AutoModelForCausalLM.from_pretrained(spec.hub_id, **kwargs)
    inner = getattr(loaded, "language_model", None)
    model_obj = inner if inner is not None else loaded
    model_obj.da_model_name = spec.hub_id
    model_obj.da_model_type = getattr(config, "model_type", spec.model_type)
    return model_obj


def chat_template_override(model_type: str, overrides: dict[str, str]) -> str:
    """Look up a chat-template override by ``model_type``, raising on unknown.

    Overrides keyed on Hub-id substrings miss local checkpoints entirely.
    """
    try:
        return overrides[model_type]
    except KeyError:
        raise KeyError(
            f"no chat-template override registered for model_type {model_type!r}; "
            f"registered: {sorted(overrides)}"
        ) from None


def loss_positions(sample: TrainingSample) -> list[int]:
    """Indices the loss applies to.  Empty is an error, not a free batch."""
    positions = [i for i, flag in enumerate(sample.assistant_mask) if flag]
    if not positions:
        raise MissingMaskError(
            "sample has no assistant tokens; the assistant mask was dropped or "
            "the response is empty"
        )
    return positions
