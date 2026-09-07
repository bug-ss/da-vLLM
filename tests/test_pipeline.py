"""The runnable evaluation pipeline, driven with an injected backend."""

from __future__ import annotations

import json

import pytest

from da_vllm.api import DAEngine
from da_vllm.config import DAConfig
from da_vllm.eval.data import SOURCE_KEYS, Example
from da_vllm.eval.pipeline import (
    RunSpec,
    finalize,
    judge_with_vllm,
    prepare,
    read_examples,
    run_arm,
    write_examples,
)
from da_vllm.eval.records import read_records
from da_vllm.eval.rubrics import (
    attach_generated_rubrics,
    attach_questions,
    generate_rubrics,
    parse_rubric,
)
from da_vllm.testing import lorem, qwen_tokenizer

RESPONSE = (
    "<global>chunk 2</global>"
    '<focus magic_chunks="2">the value</focus>'
    "<local>committing</local><answer>the value</answer>"
)
MODEL = "Qwen/Qwen3.6-27B"


def _examples(per_source=2, sources=SOURCE_KEYS, paragraphs=20):
    out = []
    for source in sources:
        for i in range(per_source):
            out.append(
                Example(
                    example_id=f"{source}:{i}",
                    source=source,
                    context=lorem(paragraphs, seed=i) + f"\n\nmarker {source} {i}",
                    question="What is the value?",
                    reference_answer="the value",
                    rubric="- CORRECT if the response states the value",
                )
            )
    return out


def _engine(tokenizer, arm="da", target=200, cap=250, max_model_len=65536):
    config = DAConfig(
        enabled=arm == "da",
        max_model_len=max_model_len,
        max_num_seqs=4,
        segment_target_tokens=target,
        segment_max_tokens=cap,
    )

    def generate_fn(token_id_lists, params):
        ids = tokenizer.encode(RESPONSE, add_special_tokens=False)
        return [(RESPONSE, ids, "stop") for _ in token_id_lists]

    return DAEngine(
        MODEL,
        arm=arm,
        config=config,
        tokenizer=tokenizer,
        generate_fn=generate_fn,
        max_tokens=1024,
    )


# -- examples on disk -------------------------------------------------------


def test_examples_round_trip_through_jsonl(tmp_path):
    path = tmp_path / "examples.jsonl"
    rows = _examples(1, SOURCE_KEYS[:2])
    assert write_examples(path, rows) == 2
    back = read_examples(path)
    assert [e.example_id for e in back] == [e.example_id for e in rows]
    assert back[0].rubric == rows[0].rubric


def test_an_unknown_source_in_the_file_fails_at_load(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "example_id": "x",
                "source": "lbv2/singledoc",  # a near-miss of a real name
                "context": "c",
                "question": "q",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="not one of the 15 sources"):
        read_examples(path)


def test_prepare_uses_the_served_tokenizer_and_the_explicit_source_list():
    tokenizer = qwen_tokenizer()
    prepared = prepare(
        _examples(3, SOURCE_KEYS[:2]),
        tokenizer=tokenizer,
        max_model_len=262_144,
        sources=SOURCE_KEYS[:2],
        n=2,
        seed=42,
    )
    assert set(prepared) == set(SOURCE_KEYS[:2])
    # Contexts here are a few hundred tokens: below the 4096 floor, so nothing
    # survives. That is the filter working, not a bug.
    assert all(len(v) == 0 for v in prepared.values())

    long_rows = [
        Example(f"s{i}", SOURCE_KEYS[0], lorem(400, seed=i), "q", "a", rubric="r")
        for i in range(4)
    ]
    kept = prepare(
        long_rows,
        tokenizer=tokenizer,
        max_model_len=262_144,
        sources=SOURCE_KEYS[:1],
        n=3,
    )
    assert len(kept[SOURCE_KEYS[0]]) == 3


# -- running ----------------------------------------------------------------


def test_run_arm_writes_records_that_score(tmp_path):
    tokenizer = qwen_tokenizer()
    sources = SOURCE_KEYS[:3]
    spec = RunSpec(
        model=MODEL,
        output_dir=tmp_path,
        sources=tuple(sources),
        arms=("da",),
        max_tokens=1024,
        batch_size=2,
    )
    by_source = {s: [] for s in sources}
    for ex in _examples(2, sources):
        by_source[ex.source].append(ex)

    with _engine(tokenizer) as engine:
        records = run_arm(spec, "da", by_source, engine=engine)

    assert len(records) == 6
    assert spec.records_path("da").exists()
    on_disk = list(read_records(spec.records_path("da")))
    assert [r.example_id for r in on_disk] == [r.example_id for r in records]
    assert all(r.attended_tokens < r.prompt_tokens * r.decode_steps for r in records)
    assert all(r.meta["num_segments"] > 1 for r in records)


def test_the_arm_decides_the_mask_not_the_caller(tmp_path):
    spec = RunSpec(model=MODEL, output_dir=tmp_path, config_overrides={"enabled": True})
    assert spec.da_config("vanilla").enabled is False
    assert spec.da_config("da").enabled is True


def test_unknown_arm_is_refused(tmp_path):
    spec = RunSpec(model=MODEL, output_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown arm"):
        run_arm(spec, "danm", {})


def test_run_all_produces_three_comparable_arms(tmp_path):
    tokenizer = qwen_tokenizer()
    sources = SOURCE_KEYS[:2]
    spec = RunSpec(
        model=MODEL,
        output_dir=tmp_path,
        sources=tuple(sources),
        max_tokens=1024,
        batch_size=4,
    )
    by_source = {s: [] for s in sources}
    # A long document on purpose. DA adds ~2.7K tokens of scaffold (the tool
    # declaration, the instruction, the per-chunk turn wrappers) and spends
    # roughly half its steps in global mode, so it only comes out ahead once
    # the context is well clear of that. This is exactly why the evaluation
    # filters out contexts below 4096 tokens.
    for ex in _examples(2, sources, paragraphs=200):
        by_source[ex.source].append(ex)

    # Real segment sizes. With 200-token chunks the per-chunk turn wrappers are
    # ~45% of the prompt and DA cannot win; at the guide's 2048 they are ~8%.
    engines = {
        arm: _engine(tokenizer, arm, target=2048, cap=2560, max_model_len=131072)
        for arm in spec.arms
    }
    all_records = {}
    for arm in spec.arms:
        all_records[arm] = run_arm(spec, arm, by_source, engine=engines[arm])
    for e in engines.values():
        e.close()

    da = sum(r.attended_tokens for r in all_records["da"])
    nomask = sum(r.attended_tokens for r in all_records["da_no_mask"])
    vanilla = sum(r.attended_tokens for r in all_records["vanilla"])
    # The DA-no-mask arm shares DA's prompt, so it reads MORE than vanilla; the
    # mask is what brings it back down. That gap is the whole ablation.
    assert da < vanilla < nomask


def test_finalize_writes_a_report_from_raw_records_only(tmp_path):
    tokenizer = qwen_tokenizer()
    sources = SOURCE_KEYS[:2]
    spec = RunSpec(
        model=MODEL, output_dir=tmp_path, sources=tuple(sources), max_tokens=1024
    )
    by_source = {s: [] for s in sources}
    for ex in _examples(2, sources):
        by_source[ex.source].append(ex)
    records = []
    for arm in spec.arms:
        with _engine(tokenizer, arm) as engine:
            records += run_arm(spec, arm, by_source, engine=engine)
    for r in records:
        r.correct = True
    out = finalize(spec, records)
    assert (tmp_path / "report.json").exists()
    assert set(out["as_logged"]["accuracy"]) == {"vanilla", "da_no_mask", "da"}
    assert out["as_logged"]["accuracy"]["da"] == 100.0


# -- judging ----------------------------------------------------------------


class _StubJudge:
    """Stands in for a vLLM judge engine."""

    def __init__(self, verdict='{"correct": true}'):
        self.verdict = verdict
        self.conversations = []

    def chat(self, conversations, params):
        self.conversations.extend(conversations)

        class _Out:
            def __init__(self, text):
                self.outputs = [type("C", (), {"text": text, "finish_reason": "stop"})()]

        return [_Out(self.verdict) for _ in conversations]


def test_judging_stores_identity_and_never_calls_on_a_format_failure(tmp_path):
    examples = _examples(1, SOURCE_KEYS[:2])
    from da_vllm.eval.records import ResponseRecord

    records = [
        ResponseRecord(
            run_id="r", model=MODEL, arm="da", source=ex.source,
            example_id=ex.example_id, prompt_fingerprint="fp", prompt_tokens=10,
            response_text="t", decode_steps=3, attended_tokens=10,
            answer_text=("the value" if i == 0 else None),
        )
        for i, ex in enumerate(examples)
    ]
    judge = _StubJudge()
    judged = judge_with_vllm(records, examples, llm=judge)
    assert len(judge.conversations) == 1  # the format failure never reached it
    assert judged[0].correct is True and judged[0].judge_parsed_by == "json"
    assert judged[1].correct is False and judged[1].judge_parsed_by == "no_answer_tag"
    assert all(r.judge_model for r in judged)


def test_judging_refuses_a_record_with_no_rubric():
    from da_vllm.eval.records import ResponseRecord

    examples = _examples(1, SOURCE_KEYS[:1])
    examples[0].rubric = None
    records = [
        ResponseRecord(
            run_id="r", model=MODEL, arm="da", source=examples[0].source,
            example_id=examples[0].example_id, prompt_fingerprint="fp",
            prompt_tokens=10, response_text="t", decode_steps=1,
            attended_tokens=1, answer_text="x",
        )
    ]
    with pytest.raises(ValueError, match="no rubric"):
        judge_with_vllm(records, examples, llm=_StubJudge())


# -- rubrics ----------------------------------------------------------------


def test_rubric_parsing_tolerates_code_fences():
    assert parse_rubric('```json\n{"rubric": "- CORRECT if X"}\n```') == "- CORRECT if X"
    assert parse_rubric('{"rubric": ""}') is None
    assert parse_rubric("not json") is None


def test_generation_records_the_author_and_survives_a_failure():
    examples = _examples(1, SOURCE_KEYS[:2])
    calls = []

    def call_model(messages):
        calls.append(messages)
        if len(calls) == 1:
            raise RuntimeError("rate limited")
        return '{"rubric": "- CORRECT if the response states the value"}'

    results = generate_rubrics(examples, call_model, author_model="gemini-3-flash")
    assert results[0].rubric is None and "rate limited" in results[0].error
    assert results[1].rubric.startswith("- CORRECT")
    attach_generated_rubrics(examples, results)
    assert examples[0].rubric is None  # dropped downstream, never judged blind
    assert examples[1].meta["rubric_author"] == "gemini-3-flash"


def test_synthetic_questions_only_replace_synthetic_sources():
    synthetic_source = "lbv1/qmsum"
    original_source = "ruler/niah_single_1"
    rows = [
        Example("a", synthetic_source, "c", "orig", "ref"),
        Example("b", original_source, "c", "orig", "ref"),
    ]
    attach_questions(rows[:1], {"a": "new question"}, author_model="gemini-3-flash")
    assert rows[0].question == "new question"
    assert rows[0].meta["question_author"] == "gemini-3-flash"
    with pytest.raises(ValueError, match="original QA"):
        attach_questions(rows[1:], {"b": "new question"}, author_model="x")


def test_records_survive_a_failure_part_way_through(tmp_path):
    """One over-length prompt must not discard an arm's finished generations."""
    from da_vllm.eval.records import read_records

    tokenizer = qwen_tokenizer()
    sources = SOURCE_KEYS[:1]
    spec = RunSpec(
        model=MODEL, output_dir=tmp_path, sources=tuple(sources),
        max_tokens=1024, batch_size=1,
    )
    rows = _examples(3, sources)
    engine = _engine(tokenizer)
    calls = {"n": 0}
    original = engine.answer_batch

    def explode(items, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise ValueError("prompt is 999999 tokens; never truncated")
        return original(items, **kwargs)

    engine.answer_batch = explode
    with pytest.raises(ValueError):
        run_arm(spec, "da", {sources[0]: rows}, engine=engine)

    on_disk = list(read_records(spec.records_path("da")))
    assert len(on_disk) == 2, "finished records were thrown away"


def test_the_judge_engine_is_actually_shut_down():
    """vLLM 0.20.2 puts shutdown() on the EngineCoreClient, not on LLMEngine."""
    from da_vllm.eval.pipeline import shutdown_engine

    calls = []

    class _Core:
        def shutdown(self):
            calls.append("engine_core")

    class _LLM:
        llm_engine = type("E", (), {"engine_core": _Core()})()

    assert shutdown_engine(_LLM()) is True
    assert calls == ["engine_core"]
    assert shutdown_engine(object()) is False


def test_samples_per_source_actually_caps_the_run(tmp_path):
    tokenizer = qwen_tokenizer()
    sources = SOURCE_KEYS[:1]
    spec = RunSpec(
        model=MODEL, output_dir=tmp_path, sources=tuple(sources),
        max_tokens=1024, batch_size=2, samples_per_source=2,
    )
    with _engine(tokenizer) as engine:
        records = run_arm(spec, "da", {sources[0]: _examples(5, sources)}, engine=engine)
    assert len(records) == 2
