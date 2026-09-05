"""Model registry: exact lookup, no substring routing, no defaults.

Guide 12 lists three failures this module is designed to make impossible:

* "Template-name routing by substring misrouted a format whose name contained
  another format's name.  Use exact registry lookup."
* "Register sampling params per model and fail on an unknown model rather than
  defaulting."
* "Chat-template overrides keyed on Hub-id substrings miss local checkpoints;
  key on ``model_type`` from the config and raise on unknown."

So: two exact-match indices (hub id and ``model_type``), and every lookup
raises :class:`UnknownModelError` instead of falling back.

The attention geometry stored here is a **cross-check only**.  Byte counts used
in the cost model are derived from the live config or the checkpoint's tensor
shapes by :mod:`da_vllm.metrics.roofline`; a 2x error in one of these numbers
(Gemma's global head dim) once produced a false finding that took weeks to
retract, so a derived note is never the source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class UnknownModelError(KeyError):
    """Raised when a model is not in the registry.  Never fall back."""


@dataclass(frozen=True)
class FamilySpec:
    """Everything that differs between chat-template families.

    ``turn_start`` / ``turn_end`` / ``tool_call_open`` are *literals from the
    shipped tokenizer's chat template*.  Guide 4.2: "The tool-call literal
    differs between Qwen 3, 3.5 and 3.6.  Segment detection regexes must be per
    minor version."  The round-trip test (guide 9.6) is what proves the values
    below match the tokenizer you actually serve.
    """

    name: str
    turn_start: str
    turn_end: str
    tool_call_open: str
    #: Anchored at a turn start: matches the turn header and any tool-response
    #: wrapper, up to (but not including) the ``Magic Chunk N`` header.
    tool_response_prefix_regex: str
    #: Anchored at a turn start: matches the assistant/model turn header.
    assistant_turn_prefix_regex: str
    #: Qwen 3.5/3.6 chat templates require ``arguments`` as a dict.  A JSON
    #: string renders as an escaped blob and shifts every later token position.
    tool_arguments_as_json_string: bool = False
    #: Gemma 4 collapses consecutive tool calls into one model turn; Qwen opens
    #: a new turn per call.  The detector needs to know which.
    collapses_consecutive_tool_calls: bool = False
    #: Extra kwargs for ``apply_chat_template``.  Thinking is off for every DA
    #: rollout: models do not follow the protocol inside thinking tags.
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)


QWEN35 = FamilySpec(
    name="qwen3.5",
    turn_start="<|im_start|>",
    turn_end="<|im_end|>",
    tool_call_open="<tool_call>",
    tool_response_prefix_regex=r"<\|im_start\|>(?:user|tool)\n(?:<tool_response>\n)?",
    assistant_turn_prefix_regex=r"<\|im_start\|>assistant\n",
    tool_arguments_as_json_string=False,
    collapses_consecutive_tool_calls=False,
    chat_template_kwargs={"enable_thinking": False},
)

QWEN36 = FamilySpec(
    name="qwen3.6",
    turn_start="<|im_start|>",
    turn_end="<|im_end|>",
    tool_call_open="<tool_call>",
    tool_response_prefix_regex=r"<\|im_start\|>(?:user|tool)\n(?:<tool_response>\n)?",
    assistant_turn_prefix_regex=r"<\|im_start\|>assistant\n",
    tool_arguments_as_json_string=False,
    collapses_consecutive_tool_calls=False,
    chat_template_kwargs={"enable_thinking": False},
)

GEMMA4 = FamilySpec(
    name="gemma4",
    turn_start="<start_of_turn>",
    turn_end="<end_of_turn>",
    tool_call_open="<tool_call>",
    tool_response_prefix_regex=r"<start_of_turn>(?:user|tool)\n(?:<tool_response>\n)?",
    assistant_turn_prefix_regex=r"<start_of_turn>model\n",
    tool_arguments_as_json_string=False,
    collapses_consecutive_tool_calls=True,
    chat_template_kwargs={},
)


@dataclass(frozen=True)
class SamplingSpec:
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float = 0.0
    max_tokens: int = 8192

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "presence_penalty": self.presence_penalty,
            "max_tokens": self.max_tokens,
        }


#: Where a geometry's numbers came from.  Only ``"measured"`` and ``"derived"``
#: may be used in a cost model without an explicit override -- see
#: :func:`da_vllm.metrics.roofline.roofline_response`.
GEOMETRY_SOURCES = ("measured", "derived", "placeholder")


@dataclass(frozen=True)
class AttentionGeometry:
    """Cross-check values.  See module docstring: not the source of truth.

    ``source`` is load-bearing.  ``"measured"`` means the numbers were read off
    the real config or checkpoint and are published; ``"derived"`` means
    :func:`da_vllm.metrics.roofline.geometry_from_config` produced them from a
    live config; ``"placeholder"`` means **nobody has verified them** and the
    cost model refuses to use them.  A 2x error in one of these numbers
    produced a false finding that took weeks to retract, so an unverified entry
    fails loudly instead of quietly being wrong.
    """

    num_layers: int
    num_global_layers: int
    global_kv_heads: int
    global_head_dim: int
    #: "sliding_window" | "gated_deltanet" | "none"
    other_layer_kind: str
    sliding_window: int | None = None
    other_kv_heads: int | None = None
    other_head_dim: int | None = None
    #: Fixed per-step bytes read by the non-global layers, used by the roofline
    #: "local read" term.  Derived, cross-checked, never trusted blind.
    local_bytes_per_step: int = 0
    #: Block size vLLM's hybrid KV-cache manager was *observed* to choose.
    #: Documentation only -- the patch learns the real value from the first
    #: metadata-builder call for the full-attention group (guide 3).
    observed_full_attention_block: int | None = None
    observed_sliding_window_block: int | None = None
    source: str = "placeholder"

    def __post_init__(self) -> None:
        if self.source not in GEOMETRY_SOURCES:
            raise ValueError(
                f"geometry source {self.source!r} must be one of {GEOMETRY_SOURCES}"
            )

    @property
    def verified(self) -> bool:
        return self.source != "placeholder"

    @property
    def global_kv_bytes_per_token(self) -> int:
        """bf16 K and V, summed over the global attention layers only."""
        return (
            self.num_global_layers
            * 2  # K and V
            * self.global_kv_heads
            * self.global_head_dim
            * 2  # bf16
        )


@dataclass(frozen=True)
class ModelSpec:
    hub_id: str
    model_type: str
    family: FamilySpec
    max_model_len: int
    sampling: SamplingSpec
    geometry: AttentionGeometry
    active_params: int
    #: Runaway detector: "no </answer> after this many tokens", set above the
    #: longest legitimate answer observed for this model (guide 10).
    no_answer_token_budget: int = 6000

    @property
    def effective_context_limit(self) -> int:
        """Length filter bound: native limit - 8K generation - 4K DA template."""
        return self.max_model_len - 8192 - 4096


_QWEN_NONTHINKING = SamplingSpec(
    temperature=0.7, top_p=0.8, top_k=20, presence_penalty=1.5, max_tokens=8192
)
_GEMMA4_SAMPLING = SamplingSpec(
    temperature=1.0, top_p=0.95, top_k=64, presence_penalty=0.0, max_tokens=8192
)
#: Judge only: Qwen-3.5-4B with thinking on, model-card thinking sampling.
JUDGE_SAMPLING = SamplingSpec(
    temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5, max_tokens=32768
)


_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        hub_id="google/gemma-4-31B-it",
        model_type="gemma4",
        family=GEMMA4,
        max_model_len=262_144,
        sampling=_GEMMA4_SAMPLING,
        geometry=AttentionGeometry(
            num_layers=60,
            num_global_layers=10,
            global_kv_heads=4,
            global_head_dim=512,
            other_layer_kind="sliding_window",
            sliding_window=1024,
            other_kv_heads=16,
            other_head_dim=256,
            # 50 layers x 1024 window x 16 heads x 256 dim x (K+V) x 2 bytes
            local_bytes_per_step=50 * 1024 * 16 * 256 * 2 * 2,
            observed_full_attention_block=32,
            observed_sliding_window_block=16,
            source="measured",
        ),
        active_params=31_000_000_000,
    ),
    # The four size-scaling models below carry PLACEHOLDER geometry: the
    # published layer counts and head dims cover only the two headline models.
    # `source="placeholder"` (the default) makes the cost model refuse them
    # until you derive the real numbers with
    # `da_vllm.metrics.roofline.geometry_from_config`.
    ModelSpec(
        hub_id="google/gemma-4-12B-it",
        model_type="gemma4_12b",
        family=GEMMA4,
        max_model_len=262_144,
        sampling=_GEMMA4_SAMPLING,
        geometry=AttentionGeometry(
            num_layers=48,
            num_global_layers=8,
            global_kv_heads=4,
            global_head_dim=256,
            other_layer_kind="sliding_window",
            sliding_window=1024,
            other_kv_heads=8,
            other_head_dim=256,
            local_bytes_per_step=40 * 1024 * 8 * 256 * 2 * 2,
        ),
        active_params=12_000_000_000,
    ),
    ModelSpec(
        hub_id="google/gemma-4-E4B-it",
        model_type="gemma4_e4b",
        family=GEMMA4,
        max_model_len=131_072,
        sampling=_GEMMA4_SAMPLING,
        geometry=AttentionGeometry(
            num_layers=35,
            num_global_layers=7,
            global_kv_heads=2,
            global_head_dim=256,
            other_layer_kind="sliding_window",
            sliding_window=1024,
            other_kv_heads=4,
            other_head_dim=256,
            local_bytes_per_step=28 * 1024 * 4 * 256 * 2 * 2,
            observed_full_attention_block=16,
            observed_sliding_window_block=32,
        ),
        active_params=4_000_000_000,
    ),
    ModelSpec(
        hub_id="Qwen/Qwen3.6-27B",
        model_type="qwen3_6",
        family=QWEN36,
        max_model_len=262_144,
        sampling=_QWEN_NONTHINKING,
        geometry=AttentionGeometry(
            num_layers=64,
            num_global_layers=16,
            global_kv_heads=4,
            global_head_dim=256,
            other_layer_kind="gated_deltanet",
            # GDN carries a recurrent state, not a KV cache: measured constant.
            local_bytes_per_step=78 * 1024 * 1024,
            observed_full_attention_block=16,
            source="measured",
        ),
        active_params=27_000_000_000,
    ),
    ModelSpec(
        hub_id="Qwen/Qwen3.5-9B",
        model_type="qwen3_5_9b",
        family=QWEN35,
        max_model_len=262_144,
        sampling=_QWEN_NONTHINKING,
        geometry=AttentionGeometry(
            num_layers=48,
            num_global_layers=12,
            global_kv_heads=4,
            global_head_dim=128,
            other_layer_kind="gated_deltanet",
            local_bytes_per_step=32 * 1024 * 1024,
            observed_full_attention_block=16,
        ),
        active_params=9_000_000_000,
    ),
    ModelSpec(
        hub_id="Qwen/Qwen3.5-4B",
        model_type="qwen3_5_4b",
        family=QWEN35,
        max_model_len=262_144,
        sampling=_QWEN_NONTHINKING,
        geometry=AttentionGeometry(
            num_layers=36,
            num_global_layers=9,
            global_kv_heads=2,
            global_head_dim=128,
            other_layer_kind="gated_deltanet",
            local_bytes_per_step=18 * 1024 * 1024,
            observed_full_attention_block=16,
        ),
        active_params=4_000_000_000,
    ),
)

_BY_HUB_ID: dict[str, ModelSpec] = {s.hub_id: s for s in _SPECS}
_BY_MODEL_TYPE: dict[str, ModelSpec] = {s.model_type: s for s in _SPECS}

#: The judge is fixed for every arm and every model.  Scoring each model with
#: itself as judge reversed the format-versus-mask conclusion (guide 8.4).
JUDGE_MODEL = "Qwen/Qwen3.5-4B"


def get_model(hub_id: str) -> ModelSpec:
    """Exact hub-id lookup.  No substring matching, no default."""
    try:
        return _BY_HUB_ID[hub_id]
    except KeyError:
        raise UnknownModelError(
            f"{hub_id!r} is not registered. Registered hub ids: "
            f"{sorted(_BY_HUB_ID)}. Add a ModelSpec rather than defaulting."
        ) from None


def get_model_by_type(model_type: str) -> ModelSpec:
    """Exact ``config.model_type`` lookup -- works for local checkpoints."""
    try:
        return _BY_MODEL_TYPE[model_type]
    except KeyError:
        raise UnknownModelError(
            f"model_type {model_type!r} is not registered. Registered types: "
            f"{sorted(_BY_MODEL_TYPE)}."
        ) from None


def resolve(model: str | ModelSpec, *, model_type: str | None = None) -> ModelSpec:
    """Resolve by ``model_type`` when given (local checkpoints), else hub id."""
    if isinstance(model, ModelSpec):
        return model
    if model_type is not None:
        return get_model_by_type(model_type)
    return get_model(model)


def register(spec: ModelSpec, *, overwrite: bool = False) -> None:
    """Register a model explicitly.  Refuses to shadow an existing entry."""
    if not overwrite and (spec.hub_id in _BY_HUB_ID or spec.model_type in _BY_MODEL_TYPE):
        raise ConfigConflictError(
            f"{spec.hub_id} / {spec.model_type} already registered; "
            "pass overwrite=True if that is intended"
        )
    _BY_HUB_ID[spec.hub_id] = spec
    _BY_MODEL_TYPE[spec.model_type] = spec


class ConfigConflictError(ValueError):
    pass


def registered_hub_ids() -> list[str]:
    return sorted(_BY_HUB_ID)
