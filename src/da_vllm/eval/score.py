"""Aggregation.  Macro everywhere, and it says so.

* **Accuracy per source** = correct / all responses for that source, format
  failures counted as wrong (they never reach the judge).
* **Headline accuracy** = the unweighted (macro) mean over the 15 sources.
* **Attended tokens** = the mean over responses within a source, then the
  unweighted mean over sources; compared as a percent change against vanilla.

Guide 12: macro and micro averages were once mixed across scripts.  Every
function here is macro over sources and micro within a source, and each returns
the per-source table alongside the headline so the two can never be confused.

Non-terminating responses (0.2 to 1.4% of DA sequences) run to the 8192 cap and
inflate the attended-token sum as logged.  On one mid-size model this was 6% of
responses and put attended tokens above vanilla until excluded, so every
aggregate is reported twice: as-logged, and with ``decode_steps >= 8000``
dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Iterable, Sequence

from .data import SOURCE_KEYS
from .records import ResponseRecord


class MissingSourceError(ValueError):
    """A source in the explicit list has no records.  Never silently skipped."""


@dataclass(frozen=True)
class ArmSummary:
    arm: str
    model: str
    per_source_accuracy: dict[str, float]
    per_source_attended: dict[str, float]
    per_source_steps: dict[str, float]
    n_by_source: dict[str, int]
    excluded_non_terminating: int
    format_correct_rate: float
    focus_attempts_per_response: float

    @property
    def macro_accuracy(self) -> float:
        return fmean(self.per_source_accuracy.values())

    @property
    def macro_attended_tokens(self) -> float:
        return fmean(self.per_source_attended.values())

    @property
    def macro_decode_steps(self) -> float:
        return fmean(self.per_source_steps.values())

    @property
    def num_responses(self) -> int:
        return sum(self.n_by_source.values())


def summarize_arm(
    records: Iterable[ResponseRecord],
    *,
    arm: str,
    model: str,
    sources: Sequence[str] = SOURCE_KEYS,
    exclude_non_terminating: bool = False,
) -> ArmSummary:
    """Aggregate one arm against an **explicit** source list."""
    buckets: dict[str, list[ResponseRecord]] = {k: [] for k in sources}
    unexpected: set[str] = set()
    excluded = 0
    for rec in records:
        if rec.arm != arm or rec.model != model:
            continue
        if rec.source not in buckets:
            unexpected.add(rec.source)
            continue
        if exclude_non_terminating and rec.non_terminating:
            excluded += 1
            continue
        buckets[rec.source].append(rec)

    if unexpected:
        raise MissingSourceError(
            f"records contain sources outside the explicit list: {sorted(unexpected)}"
        )
    empty = [k for k, v in buckets.items() if not v]
    if empty:
        raise MissingSourceError(
            f"no records for {empty} in arm {arm!r} / model {model!r}; a stale "
            "result directory or a swapped source name will otherwise average in"
        )

    acc, att, steps, counts = {}, {}, {}, {}
    all_recs: list[ResponseRecord] = []
    for key, recs in buckets.items():
        # Format failures count as wrong: they are scored without a judge call.
        acc[key] = fmean(1.0 if r.correct else 0.0 for r in recs)
        att[key] = fmean(r.attended_tokens for r in recs)
        steps[key] = fmean(r.decode_steps for r in recs)
        counts[key] = len(recs)
        all_recs.extend(recs)

    return ArmSummary(
        arm=arm,
        model=model,
        per_source_accuracy=acc,
        per_source_attended=att,
        per_source_steps=steps,
        n_by_source=counts,
        excluded_non_terminating=excluded,
        format_correct_rate=fmean(1.0 if r.format_ok else 0.0 for r in all_recs),
        focus_attempts_per_response=fmean(r.focus_attempts for r in all_recs),
    )


@dataclass(frozen=True)
class Comparison:
    baseline: str
    treatment: str
    accuracy_delta_pp: float
    attended_tokens_pct: float
    decode_steps_pct: float
    per_source_attended_pct: dict[str, float] = field(default_factory=dict)


def compare(baseline: ArmSummary, treatment: ArmSummary) -> Comparison:
    """Percent change of ``treatment`` relative to ``baseline``."""

    def pct(new: float, old: float) -> float:
        return (new - old) / old * 100.0 if old else float("nan")

    return Comparison(
        baseline=baseline.arm,
        treatment=treatment.arm,
        accuracy_delta_pp=(treatment.macro_accuracy - baseline.macro_accuracy) * 100.0,
        attended_tokens_pct=pct(
            treatment.macro_attended_tokens, baseline.macro_attended_tokens
        ),
        decode_steps_pct=pct(treatment.macro_decode_steps, baseline.macro_decode_steps),
        per_source_attended_pct={
            k: pct(treatment.per_source_attended[k], baseline.per_source_attended[k])
            for k in baseline.per_source_attended
        },
    )


def report(
    records: Sequence[ResponseRecord],
    *,
    model: str,
    sources: Sequence[str] = SOURCE_KEYS,
) -> dict[str, object]:
    """The full table for one model: three arms, as-logged and filtered."""
    out: dict[str, object] = {"model": model, "sources": list(sources)}
    for label, exclude in (("as_logged", False), ("excl_non_terminating", True)):
        arms = {
            arm: summarize_arm(
                records,
                arm=arm,
                model=model,
                sources=sources,
                exclude_non_terminating=exclude,
            )
            for arm in ("vanilla", "da_no_mask", "da")
        }
        out[label] = {
            "accuracy": {a: s.macro_accuracy * 100 for a, s in arms.items()},
            "attended_tokens": {a: s.macro_attended_tokens for a, s in arms.items()},
            "decode_steps": {a: s.macro_decode_steps for a, s in arms.items()},
            "format_correct_rate": {a: s.format_correct_rate for a, s in arms.items()},
            "focus_attempts_per_response": arms["da"].focus_attempts_per_response,
            "da_vs_vanilla": compare(arms["vanilla"], arms["da"]),
            "da_vs_no_mask": compare(arms["da_no_mask"], arms["da"]),
            "excluded_non_terminating": {
                a: s.excluded_non_terminating for a, s in arms.items()
            },
            "per_source_accuracy": {a: s.per_source_accuracy for a, s in arms.items()},
        }
    return out
