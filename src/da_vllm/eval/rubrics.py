"""Rubric generation (guide 8.1, paper appendix F).

Every example carries a binary rubric whose **first bullet states the exact
correct answer**, so the judge can score without ever seeing the context.
Partial credit is forbidden.

The paper generated these with Gemini 3 Flash.  Nothing here is tied to that
model: pass any ``call_model(messages) -> text`` and the generating model's
identity is stored alongside the rubric, because a rubric whose author is
unknown cannot be audited later.

Four sources also use **synthetic questions** written by the same model
(qmsum, multidoc_qa, singledoc_qa, LooGLE summarization).  The paper does not
publish that prompt, so :func:`attach_questions` is the merge point for
questions you generate yourself, and it records their origin the same way.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

from .data import Example, get_source
from .judge import rubric_messages

logger = logging.getLogger(__name__)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)

CallModel = Callable[[list[dict[str, str]]], str]


@dataclass(frozen=True)
class RubricResult:
    example_id: str
    rubric: str | None
    author_model: str
    error: str | None = None


def parse_rubric(raw: str) -> str | None:
    """Parse the ``{"rubric": str}`` object, tolerating code fences."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    match = _JSON_OBJECT_RE.search(text)
    if match is None:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    rubric = data.get("rubric")
    return rubric.strip() if isinstance(rubric, str) and rubric.strip() else None


def generate_rubrics(
    examples: Sequence[Example],
    call_model: CallModel,
    *,
    author_model: str,
) -> list[RubricResult]:
    """Generate one rubric per example.  Failures are reported, never invented.

    An example whose rubric could not be parsed keeps ``rubric=None`` and is
    dropped by :func:`~da_vllm.eval.data.prepare_source` -- it is never judged
    without one.
    """
    results: list[RubricResult] = []
    for example in examples:
        try:
            raw = call_model(
                rubric_messages(example.context, example.question, example.reference_answer)
            )
        except Exception as exc:  # a generation failure is data, not a crash
            logger.warning("rubric generation failed for %s: %s", example.example_id, exc)
            results.append(RubricResult(example.example_id, None, author_model, repr(exc)))
            continue
        rubric = parse_rubric(raw)
        if rubric is None:
            logger.warning("unparseable rubric for %s", example.example_id)
        results.append(
            RubricResult(
                example.example_id,
                rubric,
                author_model,
                None if rubric else "unparseable",
            )
        )
    return results


def attach_generated_rubrics(
    examples: Sequence[Example], results: Iterable[RubricResult]
) -> list[Example]:
    """Attach rubrics and record who wrote them."""
    by_id = {r.example_id: r for r in results}
    for example in examples:
        result = by_id.get(example.example_id)
        if result is None:
            continue
        example.rubric = result.rubric
        example.meta["rubric_author"] = result.author_model
        if result.error:
            example.meta["rubric_error"] = result.error
    return list(examples)


def attach_questions(
    examples: Sequence[Example], questions: dict[str, str], *, author_model: str
) -> list[Example]:
    """Replace questions with synthetic ones, recording their origin.

    Only the four sources the paper marks synthetic should receive these; the
    function refuses to rewrite a question on a source that uses its original
    QA, because silently swapping one for the other is unrecoverable later.
    """
    for example in examples:
        replacement = questions.get(example.example_id)
        if replacement is None:
            continue
        if get_source(example.source).question_origin != "synthetic":
            raise ValueError(
                f"{example.source} uses original QA; refusing to substitute a "
                "synthetic question for it"
            )
        example.question = replacement
        example.meta["question_author"] = author_model
    return list(examples)
