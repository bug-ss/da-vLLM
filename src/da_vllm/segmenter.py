"""Tokenizer-aware semantic segmenter (guide 4.1, paper appendix F).

Targets 2048 tokens per segment with a 2560-token hard cap and descends a
delimiter hierarchy from coarse to fine, splitting only units that exceed the
cap.

Two invariants the implementation exists to guarantee:

1. **Character-offset space only.**  The context is tokenized once with an
   offset mapping; every segment is emitted as a real substring of the input.
   The first segmenter built segments by decoding token-id slices, which split
   multibyte characters and produced U+FFFD in about a fifth of samples.
2. **Lossless partition.**  Concatenating the segments reproduces the input
   exactly (the sole exception is an empty/whitespace-only context, which
   becomes a single placeholder segment so the protocol always has at least one
   addressable chunk).

Segment boundaries are deliberately *not* aligned to KV blocks.  Block
alignment happens later, in the mask (guide 4.1 / 5.4).
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Protocol, Sequence


class OffsetTokenizer(Protocol):
    """The slice of the HF tokenizer API the segmenter needs."""

    def __call__(self, text: str, **kwargs): ...


@dataclass(frozen=True)
class Segment:
    """One magic chunk.  ``index`` is 1-based, matching the prompt."""

    index: int
    start: int  # character offset into the original context
    end: int
    text: str
    num_tokens: int


#: Coarse to fine.  A cut lands *after* the delimiter match, so the delimiter
#: stays attached to the preceding unit and the partition stays lossless.
_DELIMITER_LEVELS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("paragraph", re.compile(r"\n[ \t]*\n[ \t\r\n]*")),
    ("newline", re.compile(r"\n")),
    ("sentence", re.compile(r"(?<=[.!?])\s+")),
    ("clause", re.compile(r"(?<=[;:,])\s+")),
    ("whitespace", re.compile(r"\s+")),
)


class Segmenter:
    def __init__(
        self,
        tokenizer: OffsetTokenizer,
        *,
        target_tokens: int = 2048,
        max_tokens: int = 2560,
        empty_placeholder: str = "<empty_context>",
    ) -> None:
        if max_tokens < target_tokens:
            raise ValueError("max_tokens must be >= target_tokens")
        self.tokenizer = tokenizer
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.empty_placeholder = empty_placeholder

    @classmethod
    def from_config(cls, tokenizer: OffsetTokenizer, config) -> "Segmenter":
        """Build from a :class:`~da_vllm.config.DAConfig` so serving, replay and
        training cannot drift on segment size."""
        return cls(
            tokenizer,
            target_tokens=config.segment_target_tokens,
            max_tokens=config.segment_max_tokens,
            empty_placeholder=config.empty_context_placeholder,
        )

    # -- token counting in offset space ------------------------------------

    def _token_starts(self, text: str) -> list[int]:
        """Character start offset of every token, ascending.

        Tokenized once for the whole context; all later measurements are
        bisections over this list, never re-tokenizations.
        """
        enc = self.tokenizer(
            text, return_offsets_mapping=True, add_special_tokens=False
        )
        offsets = enc["offset_mapping"]
        # Special tokens can come back as (0, 0); drop degenerate spans so they
        # do not distort counts near the start of the text.
        return [s for (s, e) in offsets if e > s]

    @staticmethod
    def _count(starts: Sequence[int], a: int, b: int) -> int:
        """Tokens whose start offset lies in [a, b)."""
        return bisect.bisect_left(starts, b) - bisect.bisect_left(starts, a)

    # -- splitting ----------------------------------------------------------

    @staticmethod
    def _split_at_level(text: str, a: int, b: int, pattern: re.Pattern[str]) -> list[tuple[int, int]] | None:
        """Cut [a, b) after every delimiter match.  None if there is no cut."""
        cuts: list[int] = []
        for m in pattern.finditer(text, a, b):
            end = m.end()
            if a < end < b:
                cuts.append(end)
        if not cuts:
            return None
        units: list[tuple[int, int]] = []
        prev = a
        for c in cuts:
            units.append((prev, c))
            prev = c
        units.append((prev, b))
        return [(s, e) for (s, e) in units if e > s]

    def _split_recursive(
        self, text: str, starts: Sequence[int], a: int, b: int
    ) -> list[tuple[int, int]]:
        """Split [a, b) until every piece is under the cap, or is atomic."""
        out: list[tuple[int, int]] = []
        # Iterative worklist: a 1M-token context would blow the recursion limit.
        stack: list[tuple[int, int, int]] = [(a, b, 0)]
        while stack:
            s, e, level = stack.pop()
            if self._count(starts, s, e) <= self.max_tokens:
                out.append((s, e))
                continue
            units = None
            lvl = level
            while lvl < len(_DELIMITER_LEVELS):
                units = self._split_at_level(text, s, e, _DELIMITER_LEVELS[lvl][1])
                if units and len(units) > 1:
                    break
                units = None
                lvl += 1
            if units is None:
                # A whitespace-free run (base64, minified code) is atomic: it
                # becomes its own over-cap segment rather than being cut
                # mid-word.
                out.append((s, e))
                continue
            # Descend past the level we just used, so a unit that is still over
            # cap is cut at the next finer boundary.
            for us, ue in reversed(units):
                stack.append((us, ue, lvl + 1))
        out.sort()
        return out

    def _pack(
        self, starts: Sequence[int], pieces: Sequence[tuple[int, int]]
    ) -> list[tuple[int, int]]:
        """Greedily pack adjacent pieces toward the target size."""
        packed: list[tuple[int, int]] = []
        cur_start: int | None = None
        cur_end = 0
        cur_tokens = 0
        for s, e in pieces:
            n = self._count(starts, s, e)
            if cur_start is None:
                cur_start, cur_end, cur_tokens = s, e, n
                continue
            if cur_tokens + n <= self.target_tokens:
                cur_end, cur_tokens = e, cur_tokens + n
            else:
                packed.append((cur_start, cur_end))
                cur_start, cur_end, cur_tokens = s, e, n
        if cur_start is not None:
            packed.append((cur_start, cur_end))
        return packed

    # -- public API ---------------------------------------------------------

    def segment(self, context: str) -> list[Segment]:
        """Split ``context`` into magic chunks, numbered 1..N."""
        if not context or not context.strip():
            # Always emit at least one segment.
            text = self.empty_placeholder
            starts = self._token_starts(text)
            return [
                Segment(
                    index=1,
                    start=0,
                    end=len(text),
                    text=text,
                    num_tokens=len(starts),
                )
            ]

        starts = self._token_starts(context)
        pieces = self._split_recursive(context, starts, 0, len(context))
        packed = self._pack(starts, pieces)

        segments: list[Segment] = []
        for i, (s, e) in enumerate(packed, start=1):
            piece = context[s:e]
            segments.append(
                Segment(
                    index=i,
                    start=s,
                    end=e,
                    text=piece,
                    # Exact count for the emitted substring, not the bisection
                    # estimate: the estimate is only used to steer packing.
                    num_tokens=len(self._token_starts(piece)) if piece else 0,
                )
            )
        return segments


def assert_lossless(context: str, segments: Sequence[Segment]) -> None:
    """Round-trip guard used by the tests and the validation harness."""
    joined = "".join(s.text for s in segments)
    if joined != context:
        raise AssertionError(
            "segmentation is not lossless: "
            f"{len(joined)} chars reassembled vs {len(context)} original"
        )
    if "�" in joined and "�" not in context:
        raise AssertionError("segmentation introduced U+FFFD (token-slice decode?)")
