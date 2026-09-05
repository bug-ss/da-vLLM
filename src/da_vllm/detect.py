"""Serving-side derivation of the segment map and the always-attended scaffold.

The client never sends the segment map.  The engine derives it from the
rendered prompt, by walking **turn and tool boundaries** -- special tokens a
model cannot emit into a message body -- rather than inline dividers, which
collide with document content (guide 12).

Every gate here is biased toward declining.  A declined focus costs efficiency;
a wrong focus silently corrupts the answer.  Detection failure never raises out
of this module: it returns a :class:`PromptMap` with no segments and a reason
string, and the request runs in GLOBAL for its whole life.  The original
fail-fast design let one bad sample abort a whole run.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Sequence

from .config import DAConfig
from .models import FamilySpec
from .prompt import QUESTION_HEADER

_MAGIC_HEADER_RE = r"Magic Chunk (?P<id>\d+)\n"


@dataclass(frozen=True)
class SegmentSpan:
    """Token span of one call-plus-response pair, wrapper tokens included."""

    index: int
    char_start: int
    char_end: int
    token_start: int
    token_end: int  # exclusive


@dataclass(frozen=True)
class PromptMap:
    """Everything the state machine needs about the prompt region."""

    num_prompt_tokens: int
    segments: tuple[SegmentSpan, ...]
    sink_end: int  # tokens [0, sink_end) are always attended
    local_window_start: int  # tokens [local_window_start, num_prompt_tokens)
    local_window_is_fallback: bool
    failure_reason: str | None = None

    @property
    def focus_available(self) -> bool:
        return bool(self.segments) and self.failure_reason is None

    def by_id(self, segment_id: int) -> SegmentSpan | None:
        # Ids are validated to be exactly 1..N, so this is an index.
        if 1 <= segment_id <= len(self.segments):
            return self.segments[segment_id - 1]
        return None


def token_char_starts(tokenizer, text: str) -> list[int]:
    """Monotone character start offset per prompt token.

    ``add_special_tokens=False`` because the rendered chat string already
    carries its own special tokens; adding a second BOS shifts every position.
    Some fast tokenizers report ``(0, 0)`` for special tokens, which would break
    monotonicity, so degenerate offsets carry the previous end forward.
    """
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    starts: list[int] = []
    prev_end = 0
    for i, (s, e) in enumerate(offsets):
        if e <= s and i > 0:
            starts.append(prev_end)
        else:
            starts.append(s)
            prev_end = max(prev_end, e)
    # Enforce monotonicity outright; bisect below depends on it.
    for i in range(1, len(starts)):
        if starts[i] < starts[i - 1]:
            starts[i] = starts[i - 1]
    return starts


def _char_to_token(starts: Sequence[int], char_offset: int) -> int:
    """Index of the token *containing* ``char_offset`` (rounded down)."""
    i = bisect.bisect_right(starts, char_offset) - 1
    return max(0, i)


def _char_to_token_end(starts: Sequence[int], char_offset: int) -> int:
    """Exclusive token end: the first token starting at or after ``char_offset``."""
    return bisect.bisect_left(starts, char_offset)


@dataclass
class _Turn:
    start: int
    end: int


def _split_turns(text: str, family: FamilySpec) -> list[_Turn]:
    starts = [m.start() for m in re.finditer(re.escape(family.turn_start), text)]
    if not starts:
        return []
    bounds = starts + [len(text)]
    return [_Turn(bounds[i], bounds[i + 1]) for i in range(len(starts))]


def detect_segments(
    prompt_text: str, family: FamilySpec, *, context_region_end: int
) -> tuple[list[tuple[int, int, int]], str | None]:
    """Return ``[(id, char_start, char_end), ...]`` or a failure reason.

    ``char_end`` of segment N is the start of the turn *after* its tool
    response, so the span covers the whole call-plus-response pair including
    wrapper tokens.

    Validation is deliberately strict: turns must alternate
    assistant-call / tool-response, and the detected ids must be exactly
    1..N in order.  A non-greedy XML regex once truncated segments whose bodies
    contained the closing tag, with contiguous ids so nothing flagged it.
    """
    resp_re = re.compile(family.tool_response_prefix_regex + _MAGIC_HEADER_RE)
    asst_re = re.compile(family.assistant_turn_prefix_regex)

    turns = _split_turns(prompt_text, family)
    if not turns:
        return [], "no turn boundaries found in prompt"

    spans: list[tuple[int, int, int]] = []
    expected = 1
    i = 0
    n = len(turns)
    while i < n:
        turn = turns[i]
        if turn.start >= context_region_end:
            break
        m = resp_re.match(prompt_text, turn.start, turn.end)
        if m is None:
            i += 1
            continue
        # A tool response.  Its call must be the turn immediately before it.
        if i == 0:
            return [], "tool response with no preceding assistant turn"
        call = turns[i - 1]
        if asst_re.match(prompt_text, call.start, call.end) is None:
            return [], f"magic chunk {m.group('id')} not preceded by an assistant turn"
        if "get_magic_chunk" not in prompt_text[call.start : call.end]:
            return [], f"magic chunk {m.group('id')} preceded by a non-tool-call turn"
        found = int(m.group("id"))
        if found != expected:
            return [], f"magic chunk ids out of order: expected {expected}, found {found}"
        # The pair ends where the next turn begins (or at the context boundary).
        pair_end = turns[i + 1].start if i + 1 < n else turn.end
        pair_end = min(pair_end, context_region_end)
        spans.append((found, call.start, pair_end))
        expected += 1
        i += 1

    if not spans:
        return [], "no magic chunk tool responses detected"
    ids = [s[0] for s in spans]
    if ids != list(range(1, len(ids) + 1)):
        return [], f"detected ids are not exactly 1..N: {ids[:8]}..."
    return spans, None


def build_prompt_map(
    tokenizer,
    prompt_text: str,
    family: FamilySpec,
    config: DAConfig,
    *,
    prompt_token_ids: Sequence[int] | None = None,
) -> PromptMap:
    """Derive the segment map, the sink and the local window.

    ``prompt_token_ids`` is the engine's own tokenization.  When supplied, a
    length mismatch against our re-tokenization disables focus for the request
    rather than producing spans that are off by a token.
    """
    starts = token_char_starts(tokenizer, prompt_text)
    num_tokens = len(prompt_token_ids) if prompt_token_ids is not None else len(starts)
    sink_end = min(config.sink_tokens, num_tokens)

    reason: str | None = None
    if prompt_token_ids is not None and len(starts) != len(prompt_token_ids):
        reason = (
            f"tokenization mismatch: renderer {len(starts)} tokens vs engine "
            f"{len(prompt_token_ids)}"
        )

    # -- local window ------------------------------------------------------
    # Searched over the prompt region only.  A marker search that ran over the
    # model's own text was one of the silent-no-op bugs (guide 12).
    header_at = prompt_text.rfind(QUESTION_HEADER)
    if header_at >= 0:
        local_start = _char_to_token(starts, header_at)
        local_fallback = False
    else:
        local_start = max(0, num_tokens - config.local_window_fallback_tokens)
        local_fallback = True
    local_start = min(local_start, max(0, num_tokens))

    if reason is not None:
        return PromptMap(num_tokens, (), sink_end, local_start, local_fallback, reason)

    context_region_end = header_at if header_at >= 0 else len(prompt_text)
    raw, reason = detect_segments(
        prompt_text, family, context_region_end=context_region_end
    )
    if reason is not None:
        return PromptMap(num_tokens, (), sink_end, local_start, local_fallback, reason)

    segments: list[SegmentSpan] = []
    prev_end = 0
    for seg_id, cs, ce in raw:
        # Cover every token that overlaps the character span, then clamp the
        # start so a token straddling two segments lands in the earlier one.
        # (With a real tokenizer the turn-start special token begins exactly at
        # ``cs`` and the clamp is a no-op; it matters only for tokenizers that
        # merge the preceding newline into the boundary token.)
        ts = max(_char_to_token(starts, cs), prev_end)
        te = _char_to_token_end(starts, ce)
        te = min(max(te, ts + 1), num_tokens)
        prev_end = te
        segments.append(SegmentSpan(seg_id, cs, ce, ts, te))

    # Spans must be disjoint and ascending; overlapping spans would mean the
    # turn walk went wrong and the mask would keep the wrong region.
    for a, b in zip(segments, segments[1:]):
        if b.token_start < a.token_end:
            return PromptMap(
                num_tokens,
                (),
                sink_end,
                local_start,
                local_fallback,
                f"segment spans overlap at ids {a.index}/{b.index}",
            )

    return PromptMap(num_tokens, tuple(segments), sink_end, local_start, local_fallback, None)
