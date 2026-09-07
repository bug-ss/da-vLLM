"""Configuration for Declarative Attention.

Two rules this module exists to enforce, both of them lessons from section 12
of the implementation guide:

* DA is switched on by an **explicit** flag.  Never by "does this request look
  like a DA request" heuristics keyed on prompt or config shape -- such a
  shortcut once matched vanilla traffic too and the mask silently applied to
  the wrong arm.
* Unknown / renamed config keys raise.  A silently-ignored renamed key turned
  masking off in a harness and nobody noticed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Mapping


class ConfigError(ValueError):
    """Raised for unknown keys, bad types or out-of-range config values."""


@dataclass(frozen=True)
class RunawayConfig:
    """Runaway-generation detector (guide section 10).

    Thresholds are the calibrated ones: the longest legitimate low-novelty run
    observed was 2,349 tokens, so the token-level sustain is 2,700.  An earlier
    250-token sustain, calibrated on three sources only, fired on real answers.
    """

    enabled: bool = False
    token_window: int = 80
    token_distinct_ratio: float = 0.30
    token_sustain: int = 2700
    word_window: int = 50
    word_distinct_ratio: float = 0.55
    word_sustain: int = 2000
    # Per-model budget, set above the longest legitimate answer for that model.
    no_answer_token_budget: int = 6000

    def validate(self) -> None:
        if self.token_window < 1 or self.word_window < 1:
            raise ConfigError("runaway windows must be >= 1")
        if not 0.0 < self.token_distinct_ratio <= 1.0:
            raise ConfigError("token_distinct_ratio must be in (0, 1]")
        if not 0.0 < self.word_distinct_ratio <= 1.0:
            raise ConfigError("word_distinct_ratio must be in (0, 1]")
        if self.token_sustain < self.token_window:
            raise ConfigError("token_sustain must be >= token_window")
        if self.word_sustain < self.word_window:
            raise ConfigError("word_sustain must be >= word_window")


@dataclass(frozen=True)
class DAConfig:
    """Everything the DA state machine and the vLLM patch need.

    ``enabled`` gates the *engine-side* machinery.  Individual requests opt in
    separately (see :mod:`da_vllm.masking.logits_processor`), so a single engine
    can serve the DA and DA-no-mask arms only if you also flip this flag -- the
    paper serves them as separate engines with separate compile caches.
    """

    # --- engine-level -----------------------------------------------------
    enabled: bool = False
    max_num_seqs: int = 256
    max_model_len: int = 262_144

    # --- always-attended scaffold (guide 5.3) -----------------------------
    sink_tokens: int = 16
    question_header: str = "# Question"
    local_window_fallback_tokens: int = 1024

    # --- segmentation (guide 4.1) -----------------------------------------
    segment_target_tokens: int = 2048
    segment_max_tokens: int = 2560
    empty_context_placeholder: str = "<empty_context>"

    # --- tag detection (guide 5.1/5.2) ------------------------------------
    detect_scan_chars: int = 500
    tag_tail_slack: int = 8
    max_focus_ids: int = 3

    # --- performance ------------------------------------------------------
    #: Use the sync-free remap.  The readable remap is kept as the reference
    #: implementation the optimized one is validated against (guide 7).
    optimized_remap: bool = True
    #: Write mask rows as block-aligned GPU slice fills instead of staging a
    #: dense row through pinned host memory (guide 6.3 / 7).
    optimized_mask_writer: bool = True
    #: Log the realized kept-block fraction from inside the remap.  This adds a
    #: host-device sync per step: never leave it on while timing (guide 9.3).
    log_kept_fraction: bool = False
    #: Serve anyway when vLLM captures the decode path in a *full* CUDA graph.
    #: Off by default because it is not safe: see the check in
    #: :func:`da_vllm.masking.patch.assert_cudagraph_mode_supported`.
    allow_full_cudagraph: bool = False
    #: Pass num_stages=2 to vLLM 0.20.2's Triton decode kernel.  Bit-exact, a
    #: few percent end to end, and entirely independent of DA (guide 7).
    triton_decode_num_stages: int | None = 2

    runaway: RunawayConfig = field(default_factory=RunawayConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if self.sink_tokens < 0:
            raise ConfigError("sink_tokens must be >= 0")
        if self.segment_max_tokens < self.segment_target_tokens:
            raise ConfigError("segment_max_tokens must be >= segment_target_tokens")
        if self.max_focus_ids < 1:
            raise ConfigError("max_focus_ids must be >= 1")
        if self.max_num_seqs < 1 or self.max_model_len < 1:
            raise ConfigError("max_num_seqs and max_model_len must be >= 1")
        if not self.question_header.strip():
            raise ConfigError("question_header must be a non-empty marker")
        self.runaway.validate()

    # -- strict (de)serialisation -----------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DAConfig":
        """Build a config, raising on any key this dataclass does not define.

        Guide 12: "Config-key renames failed silently and turned masking off in
        a harness.  Validate config keys."
        """
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(
                f"unknown DAConfig keys: {unknown}; known keys: {sorted(known)}"
            )
        kwargs = dict(data)
        runaway = kwargs.pop("runaway", None)
        if runaway is not None:
            if isinstance(runaway, RunawayConfig):
                kwargs["runaway"] = runaway
            else:
                r_known = {f.name for f in dataclasses.fields(RunawayConfig)}
                r_unknown = sorted(set(runaway) - r_known)
                if r_unknown:
                    raise ConfigError(
                        f"unknown RunawayConfig keys: {r_unknown}; "
                        f"known keys: {sorted(r_known)}"
                    )
                kwargs["runaway"] = RunawayConfig(**runaway)
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
