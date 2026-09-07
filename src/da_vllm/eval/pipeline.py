"""The runnable evaluation: prepare -> generate -> replay -> judge -> report.

One arm per invocation is the safe shape (guide 11/12: one engine per arm in
its own process group, so a leaked engine cannot hold VRAM across arms, and a
separate compile cache per configuration).  :func:`run_arm` does one arm;
:func:`run_all` runs them in sequence in this process, closing each engine
before the next.

Examples come from a JSONL file rather than a dataset loader, because the
rubrics and the four sources' synthetic questions are inputs you generate
(see :mod:`da_vllm.eval.rubrics`), not something a public split carries.  One
line per example::

    {"example_id": "...", "source": "ruler/niah_single_1", "context": "...",
     "question": "...", "reference_answer": "...", "rubric": "- CORRECT if ..."}

Everything downstream reads only the raw per-response records this writes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from ..api import DAEngine
from ..config import DAConfig
from ..models import JUDGE_MODEL, ModelSpec, resolve
from .data import SOURCE_KEYS, Example, get_source, prepare_source
from .judge import judge_messages, judge_sampling_params, parse_verdict
from .records import ARMS, ResponseRecord, new_run_id, write_records
from .score import report

logger = logging.getLogger(__name__)


# -- examples ---------------------------------------------------------------


def read_examples(path: str | Path) -> list[Example]:
    """Load examples from JSONL.  Unknown sources fail here, not in the average."""
    out: list[Example] = []
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            try:
                get_source(row["source"])
            except KeyError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from None
            out.append(
                Example(
                    example_id=row["example_id"],
                    source=row["source"],
                    context=row["context"],
                    question=row["question"],
                    reference_answer=row.get("reference_answer", ""),
                    rubric=row.get("rubric"),
                    context_tokens=row.get("context_tokens"),
                    meta=row.get("meta", {}),
                )
            )
    return out


def write_examples(path: str | Path, examples: Iterable[Example]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(
                json.dumps(
                    {
                        "example_id": ex.example_id,
                        "source": ex.source,
                        "context": ex.context,
                        "question": ex.question,
                        "reference_answer": ex.reference_answer,
                        "rubric": ex.rubric,
                        "context_tokens": ex.context_tokens,
                        "meta": ex.meta,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    return n


def prepare(
    examples: Sequence[Example],
    *,
    tokenizer,
    max_model_len: int,
    sources: Sequence[str] = SOURCE_KEYS,
    n: int = 128,
    seed: int = 42,
) -> dict[str, list[Example]]:
    """Apply the filtering pipeline per source, with the served tokenizer."""

    def count_tokens(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    by_source: dict[str, list[Example]] = {key: [] for key in sources}
    for ex in examples:
        if ex.source in by_source:
            by_source[ex.source].append(ex)
    return {
        key: prepare_source(
            rows,
            count_tokens=count_tokens,
            max_model_len=max_model_len,
            n=n,
            seed=seed,
        )
        for key, rows in by_source.items()
    }


# -- generation -------------------------------------------------------------


@dataclass
class RunSpec:
    model: str
    output_dir: Path
    sources: tuple[str, ...] = SOURCE_KEYS
    arms: tuple[str, ...] = ARMS
    #: Cap on examples generated per source.  ``None`` means "everything the
    #: prepared file holds", which is what `da prepare` already sampled to.
    samples_per_source: int | None = None
    max_tokens: int = 8192
    max_num_seqs: int = 256
    batch_size: int = 32
    config_overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def spec(self) -> ModelSpec:
        return resolve(self.model)

    def da_config(self, arm: str) -> DAConfig:
        raw = {
            "enabled": arm == "da",
            "max_num_seqs": self.max_num_seqs,
            "max_model_len": self.spec.max_model_len,
        }
        raw.update(self.config_overrides)
        raw["enabled"] = arm == "da"  # never overridable: the arm decides
        return DAConfig.from_dict(raw)

    def records_path(self, arm: str) -> Path:
        return Path(self.output_dir) / f"records-{arm}.jsonl"


def run_arm(
    spec: RunSpec,
    arm: str,
    examples_by_source: dict[str, list[Example]],
    *,
    engine: DAEngine | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> list[ResponseRecord]:
    """Generate and replay one arm.  Writes ``records-<arm>.jsonl``."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")

    owned = engine is None
    engine = engine or DAEngine(
        spec.model,
        arm=arm,
        config=spec.da_config(arm),
        max_tokens=spec.max_tokens,
    )
    run_id = new_run_id(spec.model, arm)
    records: list[ResponseRecord] = []
    try:
        for source in spec.sources:
            rows = examples_by_source.get(source, [])
            if spec.samples_per_source is not None:
                rows = rows[: spec.samples_per_source]
            for start in range(0, len(rows), spec.batch_size):
                batch = rows[start : start + spec.batch_size]
                answers = engine.answer_batch(
                    [(ex.context, ex.question) for ex in batch]
                )
                for ex, ans in zip(batch, answers):
                    records.append(
                        ResponseRecord(
                            run_id=run_id,
                            model=spec.spec.hub_id,
                            arm=arm,
                            source=source,
                            example_id=ex.example_id,
                            prompt_fingerprint=ans.prompt_fingerprint,
                            prompt_tokens=ans.prompt_tokens,
                            response_text=ans.text,
                            decode_steps=ans.decode_steps,
                            attended_tokens=ans.attended_tokens,
                            block_aligned_attended_tokens=(
                                ans.block_aligned_attended_tokens
                            ),
                            finish_reason=ans.finish_reason,
                            focus_attempts=ans.focus_attempts,
                            focus_granted=ans.focus_granted,
                            declines=ans.declines,
                            mode_steps=ans.mode_steps,
                            answer_text=ans.answer,
                            meta={
                                "num_segments": ans.num_segments,
                                "detection_failure": ans.detection_failure,
                            },
                        )
                    )
                # Written after every batch, not at the end: one over-length
                # prompt raising out of answer_batch would otherwise throw away
                # a whole arm's worth of finished generations.
                write_records(spec.records_path(arm), records)
                if progress is not None:
                    progress(source, min(start + spec.batch_size, len(rows)), len(rows))
    finally:
        if owned:
            engine.close()

    write_records(spec.records_path(arm), records)
    logger.info("da: wrote %d records to %s", len(records), spec.records_path(arm))
    return records


def run_all(
    spec: RunSpec,
    examples_by_source: dict[str, list[Example]],
    *,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, list[ResponseRecord]]:
    """Run every arm in sequence, one engine at a time.

    Prefer one process per arm in production (``da run --arm ...``): an
    orphaned EngineCore reparents to PID 1 and holds VRAM across arms.
    """
    return {
        arm: run_arm(spec, arm, examples_by_source, progress=progress)
        for arm in spec.arms
    }


# -- judging ----------------------------------------------------------------


def shutdown_engine(llm: Any) -> bool:
    """Shut an engine down properly, wherever the method happens to live.

    Returns True if something was called.  An engine that is merely
    dereferenced leaves its EngineCore subprocess to be reaped later, still
    holding VRAM.
    """
    engine = getattr(llm, "llm_engine", None)
    for target in (getattr(engine, "engine_core", None), engine, llm):
        shutdown = getattr(target, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception:  # pragma: no cover
                logger.exception("da: engine shutdown raised; check for leaked VRAM")
            return True
    logger.warning(
        "da: found no shutdown() on the engine; its subprocess may outlive "
        "this process and hold VRAM"
    )
    return False


def judge_with_vllm(
    records: Sequence[ResponseRecord],
    examples: Sequence[Example],
    *,
    judge_model: str = JUDGE_MODEL,
    llm: Any = None,
    batch_size: int = 64,
    sampling_params: Any = None,
) -> list[ResponseRecord]:
    """Score every record with the one fixed judge, in batches.

    Responses with no parseable ``<answer>`` tag never reach the judge.
    """
    by_id = {ex.example_id: ex for ex in examples}
    missing = [r.example_id for r in records if r.example_id not in by_id]
    if missing:
        raise ValueError(f"no example for records {missing[:5]} (and maybe more)")
    for rec in records:
        if not by_id[rec.example_id].rubric:
            raise ValueError(
                f"no rubric for {rec.example_id}: rows without a rubric are "
                "dropped before generation, never judged without one"
            )

    pending = [r for r in records if r.answer_text]
    for rec in records:
        if not rec.answer_text:
            rec.correct = False
            rec.judge_model = judge_model
            rec.judge_parsed_by = "no_answer_tag"

    if not pending:
        return list(records)

    owned = llm is None
    if owned:
        from vllm import LLM  # type: ignore

        llm = LLM(model=judge_model, max_model_len=40960)
    try:
        params = sampling_params
        if params is None:
            try:
                from vllm import SamplingParams  # type: ignore

                params = SamplingParams(n=1, **judge_sampling_params())
            except ImportError:
                # An injected engine may take a plain dict; only a real vLLM
                # judge needs the SamplingParams object.
                params = judge_sampling_params()
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            conversations = [
                judge_messages(
                    by_id[r.example_id].question, r.answer_text, by_id[r.example_id].rubric
                )
                for r in batch
            ]
            outputs = llm.chat(conversations, params)
            for rec, out in zip(batch, outputs):
                completion = out.outputs[0]
                verdict = parse_verdict(
                    completion.text,
                    judge_model=judge_model,
                    finish_reason=getattr(completion, "finish_reason", None),
                )
                rec.correct = verdict.correct
                rec.judge_model = verdict.judge_model
                rec.judge_parsed_by = verdict.parsed_by
                rec.judge_truncated = verdict.truncated
    finally:
        if owned:
            # Same walk DAEngine.close() uses: in vLLM 0.20.2 shutdown lives on
            # the EngineCoreClient, not on LLM or LLMEngine. Probing only
            # llm_engine finds nothing and leaks the judge's subprocess.
            shutdown_engine(llm)

    truncated = sum(1 for r in records if r.judge_truncated)
    if truncated:
        logger.warning(
            "da: %d/%d judge responses hit the token cap before a verdict; a "
            "truncated thinking judge once looked like a prompt problem",
            truncated,
            len(records),
        )
    return list(records)


# -- reporting --------------------------------------------------------------


def finalize(spec: RunSpec, records: Sequence[ResponseRecord]) -> dict[str, Any]:
    """Recompute every number from the raw records and write ``report.json``."""
    out = report(list(records), model=spec.spec.hub_id, sources=list(spec.sources))
    path = Path(spec.output_dir) / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(out, indent=2, default=lambda o: getattr(o, "__dict__", str(o))),
        encoding="utf-8",
    )
    logger.info("da: wrote %s", path)
    return out
