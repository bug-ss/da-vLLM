"""The per-request DA state machine (guide 5).

Detection is a **string scan over incrementally decoded text** -- not a
token-id match, not a logits hook.  After each generated token is appended, if
that token's text contains ``>``, four tail-anchored regexes run over the last
few hundred characters.

The mask this produces has a fixed shape:

* tokens ``[0, sink_end)``                   -- the attention sink;
* tokens ``[local_window_start, prompt_len)`` -- question + DA instruction;
* tokens ``[prompt_len, inf)``               -- the response so far *and every
  future token*, unconditionally;
* under FOCUS, the named segment spans.

Because the boundary sits at ``prompt_len`` and everything beyond it is kept
unconditionally, a token generated after the mask was frozen can never fall out
of it -- one of the guide's masking bugs -- and a row only needs rewriting when
the mode changes.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from .config import DAConfig
from .detect import PromptMap


class Mode(str, Enum):
    GLOBAL = "global"
    FOCUS = "focus"
    LOCAL = "local"


DEFAULT_TAG_TAIL_SLACK = 8


@functools.lru_cache(maxsize=8)
def tag_patterns(slack: int = DEFAULT_TAG_TAIL_SLACK) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """The four tail-anchored detection regexes, in application order.

    ``.{0,slack}$`` tolerates the few trailing characters that arrive in the
    same token as the tag's closing ``>``.  ``<global>`` has no regex -- global
    is simply the default state between declared spans.  ``<answer>`` is an
    alias of ``<local>``, so the answer is written without seeing any context
    segment.
    """
    tail = r".{0,%d}$" % int(slack)
    return (
        ("close_focus", re.compile(r"</focus>" + tail, re.S)),
        ("close_local", re.compile(r"</(?:local|answer)>" + tail, re.S)),
        ("open_focus", re.compile(r"<focus[^>]*>" + tail, re.S)),
        ("open_local", re.compile(r"<(?:local|answer)>" + tail, re.S)),
    )


_DEFAULT_PATTERNS = tag_patterns()
CLOSE_FOCUS_RE = _DEFAULT_PATTERNS[0][1]
CLOSE_LOCAL_RE = _DEFAULT_PATTERNS[1][1]
OPEN_FOCUS_RE = _DEFAULT_PATTERNS[2][1]
OPEN_LOCAL_RE = _DEFAULT_PATTERNS[3][1]

_MAGIC_CHUNKS_ATTR_RE = re.compile(r'magic_chunks\s*=\s*"([^"]*)"')
_ID_SPLIT_RE = re.compile(r"[,\s]+")


class DeclineReason(str, Enum):
    """Why a ``<focus>`` was refused.  Every gate is biased toward declining:
    a declined focus costs efficiency, a wrong focus corrupts the answer."""

    NO_SEGMENTS = "no_segments_detected"
    SYNTAX = "syntax_error"
    EMPTY = "empty_id_list"
    UNKNOWN_ID = "unknown_id"
    TOO_MANY = "too_many_ids"


@dataclass(frozen=True)
class MaskSnapshot:
    """A complete description of one request's allowed-KV mask."""

    mode: Mode
    #: Sorted, merged, disjoint token ranges inside ``[0, boundary)``.
    spans: tuple[tuple[int, int], ...]
    #: Every position >= ``boundary`` is attended unconditionally.
    boundary: int
    focus_ids: tuple[int, ...] = ()

    @property
    def kept_prompt_tokens(self) -> int:
        return sum(e - s for s, e in self.spans)

    @property
    def is_full(self) -> bool:
        return self.spans == ((0, self.boundary),) or self.boundary == 0

    def dense(self, length: int | None = None) -> list[bool]:
        """Token-granular mask over ``[0, length)``.  Reference/validation use.

        Produced from the same span list the GPU writer consumes, so the two
        cannot drift (guide 5.3).
        """
        n = self.boundary if length is None else length
        out = [False] * n
        for s, e in self.spans:
            for i in range(max(0, s), min(n, e)):
                out[i] = True
        for i in range(min(n, self.boundary), n):
            out[i] = True
        return out


class _MaskWriter:
    """The single function every kept region goes through (guide 5.3)."""

    def __init__(self) -> None:
        self._spans: list[tuple[int, int]] = []

    def keep(self, start: int, end: int) -> None:
        if end > start:
            self._spans.append((start, end))

    def finish(self) -> tuple[tuple[int, int], ...]:
        if not self._spans:
            return ()
        self._spans.sort()
        merged: list[tuple[int, int]] = [self._spans[0]]
        for s, e in self._spans[1:]:
            ls, le = merged[-1]
            if s <= le:
                merged[-1] = (ls, max(le, e))
            else:
                merged.append((s, e))
        return tuple(merged)


def build_mask(
    prompt_map: PromptMap, mode: Mode, focus_ids: Sequence[int] = ()
) -> MaskSnapshot:
    """Assemble the mask for one mode.  GLOBAL is the all-True mask."""
    n = prompt_map.num_prompt_tokens
    if mode is Mode.GLOBAL:
        return MaskSnapshot(Mode.GLOBAL, ((0, n),) if n else (), n, ())

    w = _MaskWriter()
    w.keep(0, prompt_map.sink_end)
    w.keep(prompt_map.local_window_start, n)
    ids: tuple[int, ...] = ()
    if mode is Mode.FOCUS:
        ids = tuple(focus_ids)
        for sid in ids:
            span = prompt_map.by_id(sid)
            if span is not None:
                w.keep(span.token_start, span.token_end)
    return MaskSnapshot(mode, w.finish(), n, ids)


def align_spans(
    spans: Iterable[tuple[int, int]], block_size: int
) -> tuple[tuple[int, int], ...]:
    """Round each span outward to block boundaries and re-merge.

    A block is kept if any position in it is kept, so the cost is at most one
    extra block per span edge.  Identical to OR-reducing the dense mask in
    groups of ``block_size``; :mod:`tests` asserts that equivalence.
    """
    if block_size < 1:
        raise ValueError("block_size must be >= 1")
    out: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if e <= s:
            continue
        a = (s // block_size) * block_size
        b = -(-e // block_size) * block_size
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return tuple(out)


class IncrementalDetokenizer:
    """Decode one token at a time without splitting multibyte characters.

    A token whose bytes do not yet form a complete character decodes to U+FFFD;
    we hold it back and emit it fused with the next token instead of letting a
    replacement character reach the tag regexes.
    """

    def __init__(self, tokenizer, window: int = 64) -> None:
        self.tokenizer = tokenizer
        self.window = window
        self._ids: list[int] = []
        self._prefix = ""

    def push(self, token_id: int) -> str:
        self._ids.append(token_id)
        text = self.tokenizer.decode(self._ids)
        delta = text[len(self._prefix):] if text.startswith(self._prefix) else text
        if "�" in delta:
            # Incomplete character: keep the prefix where it is so the held
            # characters come out with the next token.
            return ""
        self._prefix = text
        if len(self._ids) > 2 * self.window:
            self._ids = self._ids[-self.window:]
            self._prefix = self.tokenizer.decode(self._ids)
        return delta


@dataclass
class RequestStats:
    focus_attempts: int = 0
    focus_granted: int = 0
    declines: list[str] = field(default_factory=list)
    transitions: list[tuple[int, Mode]] = field(default_factory=list)  # (step, mode)


class DAStateMachine:
    """One per DA-enabled request.

    Construction decodes nothing and allocates nothing large; the caller passes
    an already-built :class:`~da_vllm.detect.PromptMap`.  Requests without DA
    enabled must never reach this class at all -- building the prompt map
    requires decoding the whole prompt, about a second at 131K tokens, on the
    engine's hot path (guide 6.2).
    """

    def __init__(
        self,
        prompt_map: PromptMap,
        tokenizer,
        config: DAConfig,
        *,
        request_id: str = "",
    ) -> None:
        self.prompt_map = prompt_map
        self.config = config
        self.request_id = request_id
        self.mode = Mode.GLOBAL
        self.focus_ids: tuple[int, ...] = ()
        self.stats = RequestStats()
        self._detok = IncrementalDetokenizer(tokenizer)
        self._tail = ""
        self._consumed = 0
        self._patterns = tag_patterns(config.tag_tail_slack)
        self._snapshot = build_mask(prompt_map, Mode.GLOBAL)

    # -- properties --------------------------------------------------------

    @property
    def num_consumed(self) -> int:
        """Generated tokens the state machine has actually seen."""
        return self._consumed

    def snapshot(self) -> MaskSnapshot:
        return self._snapshot

    # -- driving -----------------------------------------------------------

    def advance(self, output_token_ids: Sequence[int]) -> bool:
        """Consume tokens vLLM appended since the last call.

        ``output_token_ids`` must be a **live reference** to vLLM's list, not a
        copy.  vLLM samples on the GPU and copies tokens back asynchronously,
        so a trailing entry may still be the placeholder ``-1``: we stop at the
        first negative id and revisit it next step.  The mask applied at step k
        therefore reflects the text through step k-2.  That is safe because the
        scaffold part of the mask is mode-invariant, so a stale mask never
        drops something the current token needs (guide 5.5).

        Returns True if the mask changed and the GPU row must be rewritten.
        """
        changed = False
        i = self._consumed
        n = len(output_token_ids)
        while i < n:
            tid = output_token_ids[i]
            if tid is None or tid < 0:
                break  # placeholder; revisit next step
            delta = self._detok.push(int(tid))
            i += 1
            if not delta:
                continue
            self._tail = (self._tail + delta)[-2 * self.config.detect_scan_chars:]
            if ">" not in delta:
                continue
            if self._scan(step=i):
                changed = True
        self._consumed = i
        return changed

    # -- tag scanning ------------------------------------------------------

    def _scan(self, step: int) -> bool:
        tail = self._tail[-self.config.detect_scan_chars:]
        hits: list[tuple[int, str, re.Match[str]]] = []
        for kind, pattern in self._patterns:
            m = pattern.search(tail)
            if m is not None:
                hits.append((m.start(), kind, m))
        if not hits:
            return False
        hits.sort(key=lambda h: h[0])

        changed = False
        for _, kind, m in hits:
            if self._transition(kind, m, step):
                changed = True
        return changed

    def _transition(self, kind: str, match: re.Match[str], step: int) -> bool:
        prev_mode, prev_ids = self.mode, self.focus_ids

        if kind == "close_focus":
            if self.mode is not Mode.FOCUS:
                return False
            self._set(Mode.GLOBAL, ())
        elif kind == "close_local":
            if self.mode is not Mode.LOCAL:
                return False
            self._set(Mode.GLOBAL, ())
        elif kind == "open_focus":
            # Focus opens only from GLOBAL.
            if self.mode is not Mode.GLOBAL:
                return False
            self.stats.focus_attempts += 1
            ids, reason = self._parse_focus(match.group(0))
            if reason is not None:
                self.stats.declines.append(reason.value)
                return False
            self.stats.focus_granted += 1
            self._set(Mode.FOCUS, ids)
        elif kind == "open_local":
            if self.mode is not Mode.GLOBAL:
                return False
            self._set(Mode.LOCAL, ())
        else:  # pragma: no cover - internal
            raise AssertionError(kind)

        if (self.mode, self.focus_ids) == (prev_mode, prev_ids):
            return False
        self.stats.transitions.append((step, self.mode))
        return True

    def _set(self, mode: Mode, ids: tuple[int, ...]) -> None:
        self.mode = mode
        self.focus_ids = ids
        self._snapshot = build_mask(self.prompt_map, mode, ids)

    def _parse_focus(
        self, tag_text: str
    ) -> tuple[tuple[int, ...], DeclineReason | None]:
        """Strict parse of ``magic_chunks="..."``.  Any doubt declines."""
        if not self.prompt_map.focus_available:
            return (), DeclineReason.NO_SEGMENTS
        m = _MAGIC_CHUNKS_ATTR_RE.search(tag_text)
        if m is None:
            return (), DeclineReason.SYNTAX
        raw = m.group(1).strip()
        if not raw:
            return (), DeclineReason.EMPTY
        parts = [p for p in _ID_SPLIT_RE.split(raw) if p]
        if not parts:
            return (), DeclineReason.EMPTY
        ids: list[int] = []
        for p in parts:
            if not p.isdigit():
                return (), DeclineReason.SYNTAX
            value = int(p)
            if self.prompt_map.by_id(value) is None:
                return (), DeclineReason.UNKNOWN_ID
            if value not in ids:
                ids.append(value)
        if len(ids) > self.config.max_focus_ids:
            return (), DeclineReason.TOO_MANY
        return tuple(ids), None
