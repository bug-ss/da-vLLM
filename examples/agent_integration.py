#!/usr/bin/env python3
"""Using DA as the long-context tool inside an agent loop.

The shape that works: your agent owns the conversation, and hands DA a
*document plus one question* whenever it needs something out of a long input.
DA is not a chat backend -- the protocol needs the magic-chunk transcript and a
single question, and thinking mode is off -- so wrap it as a tool rather than
routing the whole agent through it.

    python examples/agent_integration.py --context doc.txt      # needs a GPU
    python examples/agent_integration.py --context doc.txt --dry-run   # no GPU

What this shows:

* one engine, reused across calls, so the model loads once;
* a batch call, because a batch of questions against one document is the case
  DA is best at;
* the accounting an agent should log: attended tokens, focus grants, and the
  declines that tell you the model tried to focus and the server said no.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from da_vllm import DAConfig, DAEngine

QUESTIONS = [
    "In what year was the company founded?",
    "Who is named as the chief executive?",
    "What was the reported revenue for the most recent year?",
]


class LongContextTool:
    """A tool an agent can call.  Owns one engine for its whole lifetime."""

    name = "read_long_document"
    description = (
        "Answer a question about a long document. Returns the answer plus the "
        "number of KV positions the model actually attended to."
    )

    def __init__(self, engine: DAEngine) -> None:
        self.engine = engine
        self.total_attended = 0
        self.total_baseline = 0

    def __call__(self, document: str, question: str) -> dict:
        return self.batch(document, [question])[0]

    def batch(self, document: str, questions: list[str]) -> list[dict]:
        results = self.engine.answer_batch([(document, q) for q in questions])
        payloads = []
        for question, r in zip(questions, results):
            self.total_attended += r.attended_tokens
            self.total_baseline += r.baseline_attended_tokens
            payloads.append(
                {
                    "question": question,
                    # Give the agent the parsed answer, not the DA trace: the
                    # tags are a serving protocol, not something to reason over.
                    "answer": r.answer,
                    "ok": r.format_ok,
                    "attended_tokens": r.attended_tokens,
                    "reduction_pct": round(r.reduction_pct, 1),
                    # Worth logging: a decline means the model asked to focus
                    # and the server refused, so that call ran at full cost.
                    "focus_declines": r.declines,
                    "detection_failure": r.detection_failure,
                }
            )
        return payloads

    @property
    def savings_pct(self) -> float:
        if not self.total_baseline:
            return 0.0
        return (self.total_attended - self.total_baseline) / self.total_baseline * 100


def build_engine(model: str, dry_run: bool) -> DAEngine:
    if not dry_run:
        return DAEngine(model, config=DAConfig(enabled=True))

    # No GPU: the offline tokenizer plus a scripted model, so the plumbing runs.
    from da_vllm.testing import qwen_tokenizer

    tokenizer = qwen_tokenizer()
    scripted = (
        '<global>Magic chunk 2 covers this.</global>'
        '<focus magic_chunks="2">the value</focus>'
        "<local>committing to it</local><answer>the value</answer>"
    )

    def generate_fn(token_id_lists, params):
        ids = tokenizer.encode(scripted, add_special_tokens=False)
        return [(scripted, ids, "stop") for _ in token_id_lists]

    return DAEngine(
        model,
        config=DAConfig(
            enabled=True,
            max_model_len=16384,
            segment_target_tokens=200,
            segment_max_tokens=250,
        ),
        tokenizer=tokenizer,
        generate_fn=generate_fn,
        max_tokens=1024,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--context", help="path to the long document")
    parser.add_argument("--dry-run", action="store_true", help="no GPU, scripted model")
    args = parser.parse_args()

    if args.context:
        document = Path(args.context).read_text(encoding="utf-8")
    else:
        from da_vllm.testing import lorem

        document = lorem(30, seed=1)

    with build_engine(args.model, args.dry_run) as engine:
        tool = LongContextTool(engine)
        # One batch: the agent asks everything it needs about this document at
        # once, so the questions share a batch instead of serializing.
        for payload in tool.batch(document, QUESTIONS):
            print(json.dumps(payload, indent=2))
        print(f"\nattention saved across the session: {tool.savings_pct:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
