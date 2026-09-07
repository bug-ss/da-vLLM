"""Benchmark suite: 15 long-context sources, and the filtering pipeline.

Rules encoded here (guide 8.1, paper D.1):

* Drop rows without a rubric.
* Filter by context length with **the served model's own tokenizer**.  A token
  count computed with another tokenizer silently overflows the engine cap; the
  datasets' own stored token counts came from a different tokenizer entirely.
* Bounds: at least 4096 tokens, at most ``max_model_len - 8192 - 4096``
  (244K for the 256K models, 116K for Gemma-4-E4B).  Over-length rows are
  **dropped, not truncated**.
* Dedupe contexts by hash of the normalized text.
* Shuffle with seed 42, take 128.
* Use validation splits where a source ships train plus validation.

Two source names that look alike were once swapped in a run, and stale result
directories leaked into an average, so :data:`SOURCES` is the explicit list
every downstream aggregation is computed against.

:data:`SOURCES` also records where each source comes from (dataset path, config
name, split, field names), which is what you need to build the JSONL file
:mod:`da_vllm.eval.pipeline` reads.  Building that file is left to you: the
rubrics and the four sources' synthetic questions are inputs no public split
carries, so a loader here would only cover the easy half.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

SEED = 42
SAMPLES_PER_SOURCE = 128
MIN_CONTEXT_TOKENS = 4096
GENERATION_RESERVE = 8192
TEMPLATE_RESERVE = 4096


@dataclass(frozen=True)
class SourceSpec:
    """One benchmark source.  ``key`` is the only identifier used anywhere."""

    key: str
    group: str  # "single_span" | "multi_span"
    hf_path: str
    hf_name: str | None
    split: str
    question_origin: str  # "original" | "synthetic"
    context_field: str = "context"
    question_field: str = "question"
    answer_field: str = "answer"


SOURCES: tuple[SourceSpec, ...] = (
    # (1) Single-span retrieval / reasoning
    SourceSpec("ruler/niah_single_1", "single_span", "ruler", "niah_single_1", "validation", "original"),
    SourceSpec("ruler/niah_single_2", "single_span", "ruler", "niah_single_2", "validation", "original"),
    SourceSpec("ruler/niah_single_3", "single_span", "ruler", "niah_single_3", "validation", "original"),
    SourceSpec("ruler/niah_multikey_2", "single_span", "ruler", "niah_multikey_2", "validation", "original"),
    SourceSpec("ruler/niah_multikey_3", "single_span", "ruler", "niah_multikey_3", "validation", "original"),
    SourceSpec("lbv1/qmsum", "single_span", "THUDM/LongBench", "qmsum", "test", "synthetic"),
    SourceSpec("lbv2/multidoc_qa", "single_span", "THUDM/LongBench-v2", "multidoc_qa", "train", "synthetic"),
    SourceSpec("loogle/summarization", "single_span", "bigai-nlco/LooGLE", "summarization", "test", "synthetic"),
    SourceSpec("lbv2/code_repo", "single_span", "THUDM/LongBench-v2", "code_repo", "train", "original"),
    SourceSpec("zs/quality", "single_span", "tau/zero_scrolls", "quality", "validation", "original"),
    # (2) Multi-span reasoning
    SourceSpec("lbv2/singledoc_qa", "multi_span", "THUDM/LongBench-v2", "singledoc_qa", "train", "synthetic"),
    SourceSpec("lbv2/dialogue_history", "multi_span", "THUDM/LongBench-v2", "dialogue_history", "train", "original"),
    SourceSpec("loogle/longdep_qa", "multi_span", "bigai-nlco/LooGLE", "longdep_qa", "test", "original"),
    SourceSpec("loogle/shortdep_cloze", "multi_span", "bigai-nlco/LooGLE", "shortdep_cloze", "test", "original"),
    SourceSpec("zs/space_digest", "multi_span", "tau/zero_scrolls", "space_digest", "validation", "original"),
)

SOURCE_KEYS: tuple[str, ...] = tuple(s.key for s in SOURCES)
_BY_KEY = {s.key: s for s in SOURCES}


def get_source(key: str) -> SourceSpec:
    try:
        return _BY_KEY[key]
    except KeyError:
        raise KeyError(
            f"{key!r} is not one of the 15 sources: {list(SOURCE_KEYS)}"
        ) from None


@dataclass
class Example:
    example_id: str
    source: str
    context: str
    question: str
    reference_answer: str
    rubric: str | None = None
    context_tokens: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)


_WS_RE = re.compile(r"\s+")


def normalized_hash(text: str) -> str:
    """Dedupe key: whitespace-normalized, case-folded SHA-256."""
    return hashlib.sha256(_WS_RE.sub(" ", text).strip().casefold().encode()).hexdigest()


def context_limit(max_model_len: int) -> int:
    """``max_model_len - 8192 (generation) - 4096 (DA template)``."""
    return max_model_len - GENERATION_RESERVE - TEMPLATE_RESERVE


def prepare_source(
    examples: Iterable[Example],
    *,
    count_tokens: Callable[[str], int],
    max_model_len: int,
    n: int = SAMPLES_PER_SOURCE,
    seed: int = SEED,
    min_tokens: int = MIN_CONTEXT_TOKENS,
) -> list[Example]:
    """Drop, filter, dedupe, shuffle, take -- in that order.

    ``count_tokens`` must be the served model's tokenizer.  The same draw is
    used for all three arms of a given model; across models it differs only
    where a different tokenizer or context limit changes which examples pass.
    """
    upper = context_limit(max_model_len)
    seen: set[str] = set()
    kept: list[Example] = []
    for ex in examples:
        if not ex.rubric:
            continue
        digest = normalized_hash(ex.context)
        if digest in seen:
            continue
        n_tokens = ex.context_tokens
        if n_tokens is None:
            n_tokens = count_tokens(ex.context)
            ex.context_tokens = n_tokens
        if n_tokens < min_tokens or n_tokens > upper:
            continue  # over-length rows are dropped, never truncated
        seen.add(digest)
        kept.append(ex)
    rng = random.Random(seed)
    rng.shuffle(kept)
    return kept[:n]


def attach_rubrics(
    examples: Sequence[Example], rubrics: dict[str, str]
) -> list[Example]:
    """Attach per-example rubrics keyed by ``example_id``.

    Every example carries a binary rubric whose first bullet states the exact
    correct answer; partial credit is forbidden.  Examples with no rubric are
    dropped by :func:`prepare_source`, never judged without one.
    """
    for ex in examples:
        ex.rubric = rubrics.get(ex.example_id)
    return list(examples)
