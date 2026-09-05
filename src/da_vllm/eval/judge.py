"""LLM-as-a-judge (guide 8.4, paper appendix F).

One fixed judge for every arm and every model: ``Qwen/Qwen3.5-4B``, thinking
on, model-card thinking sampling, ``max_tokens`` 32768.  Two mistakes this
module is shaped to avoid:

* A thinking judge capped at 4096 tokens was truncated before emitting its
  verdict in about 3.5% of cases, which looked like a prompt problem.  Hence
  the 32768 cap and the explicit :attr:`Verdict.truncated` flag.
* Scoring each model with itself as judge reversed the format-versus-mask
  conclusion.  :func:`judge_prompt` takes no target-model argument at all, and
  every verdict stores the judge identity.

The judge sees the question, the rubric, and only the text inside
``<answer>...</answer>``.  It never sees the context or the DA tags.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..models import JUDGE_MODEL, JUDGE_SAMPLING

JUDGE_SYSTEM_PROMPT = """\
You are an AI assistant specializing in language model evaluation.
You are the judge model, and your goal is to evaluate the response of a target \
model under evaluation. The target model is expected to answer the question \
based on some given context. You will be given a question, the target model's \
response, and a binary evaluation rubric. Your goal is to evaluate the target \
model's response based on the rubric."""

JUDGE_USER_TEMPLATE = """\
# Question
{question}

# Response from target model
<model_response>
{response}
</model_response>

# Evaluation rubric
{rubric}

# Instructions
Evaluate whether the target model's response enclosed in the <model_response> \
tags is correct, based solely on the given evaluation rubric. Do not rely on \
your knowledge of the context or the question.

When applying the rubric, ignore the following surface-level differences (they \
do not by themselves make a response wrong):

1. **Case** ("sigmoid" vs "Sigmoid", "methicillin" vs "Methicillin") - treat \
string matches as case-insensitive unless the rubric explicitly demands a \
specific case.
2. **Minor morphological variation** (singular vs plural, gerund vs noun, e.g. \
"sigmoid activation function" vs "sigmoid activation functions") - accept \
unless the rubric calls out the form explicitly.
3. **Numeric formatting of equivalent values** ("3" vs "Three", "900,000" vs \
"nine hundred thousand", "$17B" vs "$17 billion") - accept as equivalent \
unless the rubric specifies a particular format.
4. **Whitespace, punctuation, and trivial typography** (LaTeX vs unicode, \
hyphens, surrounding quotes).
5. **Wrapper text or paraphrasing** that preserves the substantive content the \
rubric requires (e.g. "The answer is X." or "The UI designer; locating the \
remote when it is lost." both satisfy a rubric that accepts X / "The UI \
designer; finding the remote.").

Do not ignore differences that the rubric explicitly flags as wrong, or that \
change the meaning of the answer (e.g. swapping a different value, naming a \
different entity, hedging that contradicts the source - "over X" when the \
rubric demands the value "stated as X" remains a judgment call: read the \
rubric carefully).

Output your assessment in the following JSON format. Do not include any other \
text in your response.
{{ "correct": boolean }}"""

RUBRIC_SYSTEM_PROMPT = """\
You are an AI assistant designing rigorous rubrics for long-context language \
model evaluation.

INSTRUCTIONS:
You are given a long context, a question about that context, and the reference \
answer. Your task is to design a rubric that lets an external evaluator judge \
whether a model's response to the question is correct, WITHOUT giving the \
evaluator access to the context.

Step 1. Design a rubric to evaluate whether a given response is correct.
1. Assume the evaluator does NOT have access to the context.
2. The first bullet of the rubric MUST state the exact correct answer \
(paraphrasing the reference answer is fine), so the evaluator can judge \
correctness without the context.
3. Allow paraphrases or alternate phrasings of the same specific fact (e.g. \
"Joe Biden" vs "President Biden"). Reject answers that state a different fact, \
an incomplete fact, or a related-but-wrong entity.
4. Identify any distractor information from the context that seems plausible \
but is NOT the correct answer.
5. The rubric only differentiates CORRECT from WRONG - no partial credit, no \
intermediate scores.

Notes on Rubric Formatting:
1. Refer to the given response as "the response" in the rubric.
2. The rubric is a bulleted list with no more than 5 items.
3. Use "CORRECT" and "WRONG" in capital letters.
4. Do not use the word "incorrect".
5. Do not use double negatives.

Step 2. Format Output.
Format your final output strictly as a single valid JSON object. Do not \
include markdown code blocks, backticks, "json" labels, or any \
preamble/postscript text. The output must be a raw string that can be directly \
passed into json.loads() in Python. The object must strictly contain this key:
1. "rubric": str"""

RUBRIC_USER_TEMPLATE = """\
# Context
{context}

# Question
{question}

# Reference answer
{reference_answer}"""

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_OPEN_THINK_RE = re.compile(r"^.*?</think>", re.S)
_JSON_RE = re.compile(r"\{[^{}]*\"correct\"[^{}]*\}", re.S)
_REGEX_FALLBACK = re.compile(r"\"correct\"\s*:\s*(true|false)", re.I)


@dataclass(frozen=True)
class Verdict:
    correct: bool
    #: Stored with every verdict.  Judge identity that is not stored cannot be
    #: reconstructed later, and one run's judge is not another's (guide 12).
    judge_model: str
    parsed_by: str  # "json" | "regex" | "no_answer_tag" | "unparseable"
    truncated: bool = False
    raw: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def judge_messages(question: str, answer_text: str, rubric: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": JUDGE_USER_TEMPLATE.format(
                question=question, response=answer_text, rubric=rubric
            ),
        },
    ]


def rubric_messages(context: str, question: str, reference_answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RUBRIC_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RUBRIC_USER_TEMPLATE.format(
                context=context, question=question, reference_answer=reference_answer
            ),
        },
    ]


def strip_thinking(text: str) -> str:
    """Remove the judge's thinking block, closed or unterminated."""
    text = _THINK_RE.sub("", text)
    if "</think>" in text:
        text = _OPEN_THINK_RE.sub("", text)
    return text.strip()


def parse_verdict(
    raw: str, *, judge_model: str = JUDGE_MODEL, finish_reason: str | None = None
) -> Verdict:
    """Strip thinking, parse JSON, fall back to a regex."""
    truncated = finish_reason == "length"
    body = strip_thinking(raw)
    m = _JSON_RE.search(body)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data.get("correct"), bool):
                return Verdict(data["correct"], judge_model, "json", truncated, raw)
        except json.JSONDecodeError:
            pass
    m = _REGEX_FALLBACK.search(body)
    if m:
        return Verdict(m.group(1).lower() == "true", judge_model, "regex", truncated, raw)
    return Verdict(False, judge_model, "unparseable", truncated, raw)


def score_response(
    question: str,
    answer_text: str | None,
    rubric: str,
    *,
    call_judge,
    judge_model: str = JUDGE_MODEL,
) -> Verdict:
    """Score one response.  ``call_judge(messages) -> (text, finish_reason)``.

    A response with no parseable ``<answer>`` tag is scored wrong **without a
    judge call** (guide 8.4).
    """
    if not answer_text:
        return Verdict(False, judge_model, "no_answer_tag")
    raw, finish_reason = call_judge(judge_messages(question, answer_text, rubric))
    return parse_verdict(raw, judge_model=judge_model, finish_reason=finish_reason)


def judge_sampling_params() -> dict[str, Any]:
    """Thinking on, 32768 tokens.  A 4096 cap truncates ~3.5% of verdicts."""
    params = JUDGE_SAMPLING.to_dict()
    params["max_tokens"] = 32768
    return params
