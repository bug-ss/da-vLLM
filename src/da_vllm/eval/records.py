"""Raw per-response records -- the only thing published numbers come from.

Guide 12: "Recompute every published number from raw per-response records
against an explicit source list, never from cached summaries."  So the record
carries everything an aggregation could need, including the judge identity and
the prompt fingerprint, and :mod:`da_vllm.eval.score` reads nothing else.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

ARMS = ("vanilla", "da_no_mask", "da")


@dataclass
class ResponseRecord:
    run_id: str
    model: str
    arm: str
    source: str
    example_id: str
    prompt_fingerprint: str
    prompt_tokens: int
    response_text: str
    decode_steps: int
    attended_tokens: int
    finish_reason: str | None = None
    # Judge outcome
    correct: bool | None = None
    judge_model: str | None = None
    judge_parsed_by: str | None = None
    judge_truncated: bool = False
    # Replay detail
    focus_attempts: int = 0
    focus_granted: int = 0
    declines: list[str] = field(default_factory=list)
    mode_steps: dict[str, int] = field(default_factory=dict)
    answer_text: str | None = None
    mask_lag_steps: int = 2
    block_aligned_attended_tokens: int | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}; expected one of {ARMS}")

    @property
    def format_ok(self) -> bool:
        return self.answer_text is not None

    @property
    def non_terminating(self) -> bool:
        from ..metrics.replay import NON_TERMINATING_STEPS

        return self.decode_steps >= NON_TERMINATING_STEPS


def write_records(path: str | Path, records: Iterable[ResponseRecord]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_records(path: str | Path) -> Iterator[ResponseRecord]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield ResponseRecord(**json.loads(line))
