"""Attended-token reconstruction (guide 8.5).

The server does not report attended tokens, so they are recomputed by replaying
the state machine over the returned text.  Per decode step:

* GLOBAL -> every non-pad position so far;
* FOCUS / LOCAL -> the number of True entries in the mask frozen at mode open,
  plus tokens generated since.

Because the mask boundary sits at ``prompt_len`` and everything past it is kept
unconditionally, that reduces to ``kept_prompt_tokens(mode) + t`` at step ``t``.
The vanilla baseline is analytic: step ``t`` attends ``prompt_length + t``.

The replay reproduces the engine's one-step lag by default: the mask applied at
step ``k`` reflects the text through step ``k-2`` (guide 5.5).  Set
``mask_lag_steps=0`` to measure the protocol as declared rather than as served.

The count is token-granular.  The kernel reads at block granularity, so real
bytes are a few percent higher; :attr:`ReplayResult.block_aligned_attended_tokens`
reports that when a block size is supplied.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..config import DAConfig
from ..detect import PromptMap
from ..state_machine import DAStateMachine, MaskSnapshot, Mode, align_spans

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)

#: Responses at or above this many steps ran to the 8192-token cap without
#: terminating.  They inflate the attended-token sum as logged; report both
#: as-logged and with these excluded (guide 8.6).
NON_TERMINATING_STEPS = 8000


@dataclass
class ReplayResult:
    prompt_tokens: int
    decode_steps: int
    attended_tokens: int
    block_aligned_attended_tokens: int | None
    mode_steps: dict[str, int]
    focus_attempts: int
    focus_granted: int
    declines: list[str] = field(default_factory=list)
    transitions: list[tuple[int, str]] = field(default_factory=list)
    answer_text: str | None = None
    mask_lag_steps: int = 2

    @property
    def format_ok(self) -> bool:
        """A response with no parseable answer tag is scored wrong without a
        judge call (guide 8.4)."""
        return self.answer_text is not None

    @property
    def non_terminating(self) -> bool:
        return self.decode_steps >= NON_TERMINATING_STEPS

    @property
    def mean_attended_per_step(self) -> float:
        return self.attended_tokens / self.decode_steps if self.decode_steps else 0.0


def extract_answer(response_text: str) -> str | None:
    matches = ANSWER_RE.findall(response_text)
    return matches[-1].strip() if matches else None


def vanilla_attended_tokens(prompt_tokens: int, decode_steps: int) -> int:
    """Analytic baseline: sum over t of (prompt_tokens + t)."""
    t = decode_steps
    return t * prompt_tokens + t * (t - 1) // 2


def replay(
    prompt_map: PromptMap,
    tokenizer,
    response_text: str,
    config: DAConfig,
    *,
    mask_lag_steps: int = 2,
    block_size: int | None = None,
    response_token_ids: Sequence[int] | None = None,
) -> ReplayResult:
    """Replay one DA response and total its attended tokens."""
    ids = (
        list(response_token_ids)
        if response_token_ids is not None
        else list(tokenizer.encode(response_text, add_special_tokens=False))
    )
    steps = len(ids)
    sm = DAStateMachine(prompt_map, tokenizer, config)

    attended = 0
    aligned_attended = 0
    mode_steps = {m.value: 0 for m in Mode}
    # Grown in place rather than re-sliced: an 8192-step response would
    # otherwise copy tens of millions of list elements.
    visible: list[int] = []
    aligned_cache: dict[MaskSnapshot, int] = {}
    for t in range(steps):
        target = max(0, t - mask_lag_steps)
        while len(visible) < target:
            visible.append(ids[len(visible)])
        sm.advance(visible)
        snap = sm.snapshot()
        mode_steps[snap.mode.value] += 1
        attended += snap.kept_prompt_tokens + t
        if block_size:
            # The mask only changes on a mode transition, so the aligned count
            # is computed once per distinct snapshot.
            # Keyed on the snapshot itself, not id(): a freed snapshot's id
            # can be reused by a different one.
            kept = aligned_cache.get(snap)
            if kept is None:
                kept = sum(
                    min(e, prompt_map.num_prompt_tokens) - s
                    for s, e in align_spans(snap.spans, block_size)
                    if min(e, prompt_map.num_prompt_tokens) > s
                )
                aligned_cache[snap] = kept
            aligned_attended += kept + t

    # Finish the walk so the tag statistics cover the whole response.
    sm.advance(ids)
    return ReplayResult(
        prompt_tokens=prompt_map.num_prompt_tokens,
        decode_steps=steps,
        attended_tokens=attended,
        block_aligned_attended_tokens=aligned_attended if block_size else None,
        mode_steps=mode_steps,
        focus_attempts=sm.stats.focus_attempts,
        focus_granted=sm.stats.focus_granted,
        declines=list(sm.stats.declines),
        transitions=[(s, m.value) for s, m in sm.stats.transitions],
        answer_text=extract_answer(response_text),
        mask_lag_steps=mask_lag_steps,
    )


def replay_vanilla(
    prompt_tokens: int, response_text: str, tokenizer, *, response_token_ids=None
) -> ReplayResult:
    """The vanilla / DA-no-mask arm: full attention at every step."""
    ids = (
        list(response_token_ids)
        if response_token_ids is not None
        else list(tokenizer.encode(response_text, add_special_tokens=False))
    )
    steps = len(ids)
    return ReplayResult(
        prompt_tokens=prompt_tokens,
        decode_steps=steps,
        attended_tokens=vanilla_attended_tokens(prompt_tokens, steps),
        block_aligned_attended_tokens=None,
        mode_steps={Mode.GLOBAL.value: steps, Mode.FOCUS.value: 0, Mode.LOCAL.value: 0},
        focus_attempts=0,
        focus_granted=0,
        answer_text=extract_answer(response_text),
        mask_lag_steps=0,
    )
