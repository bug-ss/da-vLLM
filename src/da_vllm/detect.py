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
import logging
import re
from dataclasses import dataclass
from typing import Sequence

from .config import DAConfig
from .models import FamilySpec

logger = logging.getLogger(__name__)

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


def _context_region_end(text: str, family: FamilySpec, header_at: int) -> int:
    """Where the context stops: the start of the turn holding the question.

    Not the header's own character offset.  The last magic chunk runs up to
    this point, and it should stop where the instruction turn begins rather
    than swallowing that turn's opening markers.
    """
    if header_at < 0:
        return len(text)
    starts = [m.start() for m in re.finditer(re.escape(family.turn_start), text)]
    earlier = [x for x in starts if x <= header_at]
    return earlier[-1] if earlier else header_at


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

    Validation is deliberately strict: every tool response must be owned by an
    assistant ``get_magic_chunk`` call turn, and the detected ids must be
    exactly 1..N in order.  A non-greedy XML regex once truncated segments whose
    bodies contained the closing tag, with contiguous ids so nothing flagged it.

    Both turn layouts are accepted, because they are both real: Qwen opens a
    new assistant turn per call, Gemma 4 collapses consecutive calls into one.
    ``FamilySpec.collapses_consecutive_tool_calls`` says which to expect.
    """
    resp_re = re.compile(family.tool_response_prefix_regex + _MAGIC_HEADER_RE)
    asst_re = re.compile(family.assistant_turn_prefix_regex)

    turns = _split_turns(prompt_text, family)
    if not turns:
        return [], "no turn boundaries found in prompt"

    # First pass: find each magic chunk's tool response and the assistant turn
    # that owns it.  Ends are assigned afterwards.
    found: list[tuple[int, int]] = []  # (id, span_start)
    expected = 1
    open_call_turn: _Turn | None = None
    i = 0
    n = len(turns)
    while i < n:
        turn = turns[i]
        if turn.start >= context_region_end:
            break
        is_call = asst_re.match(prompt_text, turn.start, turn.end) is not None and (
            "get_magic_chunk" in prompt_text[turn.start : turn.end]
        )
        if is_call:
            open_call_turn = turn
            i += 1
            continue
        m = resp_re.match(prompt_text, turn.start, turn.end)
        if m is None:
            # Some other turn.  It does NOT end the run: a document body can
            # contain the family's turn literal verbatim (it is inserted into
            # the tool response unescaped), which creates a real turn boundary
            # inside a chunk.  Treat such turns as part of the chunk they fall
            # in rather than as a separator -- see the span-end pass below.
            i += 1
            continue
        if open_call_turn is None:
            return [], f"magic chunk {m.group('id')} has no preceding tool call"
        found_id = int(m.group("id"))
        if found_id != expected:
            return [], f"magic chunk ids out of order: expected {expected}, found {found_id}"
        # Under strict alternation the span starts at the call turn.  In a
        # collapsed run (Gemma) the later responses start at their own turn, so
        # the spans stay disjoint either way.
        span_start = (
            open_call_turn.start
            if not found or found[-1][1] < open_call_turn.start
            else turn.start
        )
        found.append((found_id, span_start))
        expected += 1
        if not family.collapses_consecutive_tool_calls:
            open_call_turn = None
        i += 1

    # Second pass: each chunk runs to the START OF THE NEXT CHUNK, not to the
    # next turn of any kind.  Anything between -- including a turn boundary the
    # document forged -- belongs to the chunk it sits in.  Ending at the next
    # turn instead would silently truncate the chunk and hide the model's own
    # content from a <focus> that named it.
    spans: list[tuple[int, int, int]] = []
    for idx, (found_id, span_start) in enumerate(found):
        span_end = (
            found[idx + 1][1] if idx + 1 < len(found) else context_region_end
        )
        spans.append((found_id, span_start, min(span_end, context_region_end)))

    if not spans:
        return [], "no magic chunk tool responses detected"
    ids = [s[0] for s in spans]
    if ids != list(range(1, len(ids) + 1)):
        return [], f"detected ids are not exactly 1..N: {ids[:8]}..."
    # Coverage must be gap-free: chunk N ends exactly where chunk N+1 begins.
    # A hole means a turn boundary was found inside a chunk and the walk lost
    # part of it -- decline rather than mask a region we cannot account for.
    for a, b in zip(spans, spans[1:]):
        if a[2] != b[1]:
            return [], (
                f"gap between magic chunk {a[0]} and {b[0]} "
                f"({b[1] - a[2]} characters unaccounted for)"
            )
    return spans, None


def _verify_against_engine(
    tokenizer,
    prompt_text: str,
    prompt_token_ids: Sequence[int] | None,
    num_rendered: int,
) -> str | None:
    """Check the text we are about to detect over IS the text the engine has.

    Every segment span is a *token offset* into the engine's prompt.  If the
    text we detect over is not byte-for-byte what the engine tokenized, those
    offsets point at the wrong tokens and the mask keeps the wrong region --
    the model answers from whatever happened to land there, confidently and
    wrongly.

    Comparing token counts is not enough: a different string of the same
    length passes.  So the ids themselves are compared.  It costs one list
    comparison per request, once, and it is the difference between a wrong
    answer and a declined focus.
    """
    if prompt_token_ids is None:
        logger.warning(
            "da: the engine did not supply prompt token ids, so the prompt "
            "text could not be verified. Segment spans are being trusted "
            "unchecked -- if the text differs from what the engine tokenized, "
            "the mask will keep the wrong tokens."
        )
        return None

    engine_ids = list(prompt_token_ids)
    if num_rendered != len(engine_ids):
        return (
            f"tokenization mismatch: detector sees {num_rendered} tokens, engine "
            f"has {len(engine_ids)}. Common cause: the prompt was sent as a "
            "string to /v1/completions, which adds a second BOS by default -- "
            "pass add_special_tokens=false, or send token ids."
        )

    ours = list(tokenizer.encode(prompt_text, add_special_tokens=False))
    if ours != engine_ids:
        first = next(
            (i for i, (a, b) in enumerate(zip(ours, engine_ids)) if a != b),
            0,
        )
        return (
            f"prompt text does not match the engine's tokens (first difference "
            f"at token {first}). The text handed to the detector is not the "
            "text the engine is serving; render once and use the same string "
            "for both."
        )
    return None


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

    reason = _verify_against_engine(tokenizer, prompt_text, prompt_token_ids, len(starts))

    # -- local window ------------------------------------------------------
    # Searched over the prompt region only.  A marker search that ran over the
    # model's own text was one of the silent-no-op bugs (guide 12).
    # The configured marker, not a module constant: the renderer writes the
    # same value, so changing one changes both (guide 4.3).
    header_at = prompt_text.rfind(config.question_header)
    if header_at >= 0:
        local_start = _char_to_token(starts, header_at)
        local_fallback = False
    else:
        local_start = max(0, num_tokens - config.local_window_fallback_tokens)
        local_fallback = True
        logger.warning(
            "da: question header %r not found in the prompt; the local window "
            "fell back to the last %d tokens. Renderer and detector must use "
            "the same DAConfig.question_header.",
            config.question_header,
            config.local_window_fallback_tokens,
        )
    local_start = min(local_start, max(0, num_tokens))

    if reason is not None:
        return PromptMap(num_tokens, (), sink_end, local_start, local_fallback, reason)

    context_region_end = _context_region_end(prompt_text, family, header_at)
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
