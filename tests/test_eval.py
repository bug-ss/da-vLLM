import pytest

from da_vllm.eval.data import (
    SAMPLES_PER_SOURCE,
    SEED,
    SOURCE_KEYS,
    Example,
    context_limit,
    get_source,
    normalized_hash,
    prepare_source,
)
from da_vllm.eval.judge import (
    JUDGE_SYSTEM_PROMPT,
    RUBRIC_SYSTEM_PROMPT,
    judge_messages,
    judge_sampling_params,
    parse_verdict,
    score_response,
    strip_thinking,
)
from da_vllm.eval.records import ResponseRecord, read_records, write_records
from da_vllm.eval.score import MissingSourceError, compare, report, summarize_arm
from da_vllm.models import JUDGE_MODEL


# -- data -----------------------------------------------------------------


def test_there_are_exactly_fifteen_sources():
    assert len(SOURCE_KEYS) == 15
    assert len(set(SOURCE_KEYS)) == 15
    assert sum(1 for s in SOURCE_KEYS if get_source(s).group == "single_span") == 10


def test_synthetic_question_sources_are_the_four_named_in_the_paper():
    synthetic = {k for k in SOURCE_KEYS if get_source(k).question_origin == "synthetic"}
    assert synthetic == {
        "lbv1/qmsum",
        "lbv2/multidoc_qa",
        "lbv2/singledoc_qa",
        "loogle/summarization",
    }


def test_unknown_source_raises_rather_than_being_skipped():
    with pytest.raises(KeyError):
        get_source("lbv2/singledoc")  # a near-miss of a real name


def test_context_limit_matches_the_paper():
    assert context_limit(262_144) == 244 * 1024
    assert context_limit(131_072) == 116 * 1024


def _examples(n, tokens=10_000, rubric="- CORRECT if x"):
    return [
        Example(f"s:{i}", "src", f"context {i} " * 3, "q", "a", rubric=rubric, context_tokens=tokens)
        for i in range(n)
    ]


def test_rows_without_a_rubric_are_dropped():
    rows = _examples(4) + _examples(2, rubric=None)
    kept = prepare_source(rows, count_tokens=lambda t: 10_000, max_model_len=262_144)
    assert len(kept) == 4


def test_over_length_rows_are_dropped_never_truncated():
    rows = _examples(3, tokens=10_000) + _examples(3, tokens=300_000)
    kept = prepare_source(rows, count_tokens=lambda t: 10_000, max_model_len=262_144)
    assert len(kept) == 3
    assert all(e.context_tokens == 10_000 for e in kept)


def test_short_contexts_are_dropped():
    rows = _examples(3, tokens=100)
    assert prepare_source(rows, count_tokens=lambda t: 100, max_model_len=262_144) == []


def test_token_counts_come_from_the_served_tokenizer_not_the_dataset():
    rows = [Example("s:0", "src", "ctx", "q", "a", rubric="r")]  # no stored count
    calls = []

    def count(text):
        calls.append(text)
        return 5000

    kept = prepare_source(rows, count_tokens=count, max_model_len=262_144)
    assert calls == ["ctx"] and kept[0].context_tokens == 5000


def test_contexts_are_deduped_by_normalized_hash():
    rows = [
        Example("a", "src", "Same   Text\n", "q", "a", rubric="r", context_tokens=9000),
        Example("b", "src", "same text", "q2", "a2", rubric="r", context_tokens=9000),
    ]
    assert normalized_hash(rows[0].context) == normalized_hash(rows[1].context)
    assert len(prepare_source(rows, count_tokens=lambda t: 9000, max_model_len=262_144)) == 1


def test_draw_is_seeded_and_capped():
    rows = _examples(400)
    a = prepare_source(rows, count_tokens=lambda t: 9000, max_model_len=262_144)
    b = prepare_source(_examples(400), count_tokens=lambda t: 9000, max_model_len=262_144)
    assert len(a) == SAMPLES_PER_SOURCE == 128
    assert [e.example_id for e in a] == [e.example_id for e in b]
    c = prepare_source(_examples(400), count_tokens=lambda t: 9000, max_model_len=262_144, seed=SEED + 1)
    assert [e.example_id for e in a] != [e.example_id for e in c]


# -- judge ----------------------------------------------------------------


def test_judge_never_sees_the_context_or_the_da_tags():
    messages = judge_messages("q", "2003", "- CORRECT if 2003")
    blob = " ".join(m["content"] for m in messages)
    assert "<focus" not in blob and "magic chunk" not in blob.lower()
    assert JUDGE_SYSTEM_PROMPT in blob


def test_judge_prompt_lists_the_surface_differences_to_ignore():
    blob = judge_messages("q", "a", "r")[1]["content"]
    for phrase in ("Case", "morphological", "Numeric formatting", "Whitespace", "Wrapper text"):
        assert phrase in blob


def test_rubric_prompt_forbids_partial_credit():
    assert "no partial credit" in RUBRIC_SYSTEM_PROMPT
    assert "first bullet" in RUBRIC_SYSTEM_PROMPT


def test_thinking_block_is_stripped_even_when_unterminated():
    assert strip_thinking("<think>a</think>{}") == "{}"
    assert strip_thinking("rambling</think> {}") == "{}"


@pytest.mark.parametrize(
    "raw,expected,how",
    [
        ('{"correct": true}', True, "json"),
        ('<think>x</think>\n{"correct": false}', False, "json"),
        ('the verdict: "correct": TRUE', True, "regex"),
        ("no verdict at all", False, "unparseable"),
    ],
)
def test_verdict_parsing(raw, expected, how):
    v = parse_verdict(raw)
    assert (v.correct, v.parsed_by) == (expected, how)


def test_truncation_is_flagged_not_silently_scored():
    v = parse_verdict("<think>still thinking", finish_reason="length")
    assert v.truncated and v.parsed_by == "unparseable"


def test_judge_cap_is_large_enough_for_a_thinking_judge():
    assert judge_sampling_params()["max_tokens"] == 32768


def test_missing_answer_tag_is_wrong_without_a_judge_call():
    def boom(_messages):
        raise AssertionError("the judge must not be called")

    v = score_response("q", None, "r", call_judge=boom)
    assert v.correct is False and v.parsed_by == "no_answer_tag"


def test_judge_identity_is_always_stored():
    v = score_response("q", "a", "r", call_judge=lambda m: ('{"correct": true}', "stop"))
    assert v.judge_model == JUDGE_MODEL


# -- records and scoring --------------------------------------------------


def _records(arm, model="M", correct=True, attended=1000, steps=10, sources=SOURCE_KEYS):
    return [
        ResponseRecord(
            run_id="r",
            model=model,
            arm=arm,
            source=src,
            example_id=f"{src}:{i}",
            prompt_fingerprint="fp",
            prompt_tokens=100,
            response_text="x",
            decode_steps=steps,
            attended_tokens=attended,
            correct=correct,
            answer_text="x",
        )
        for src in sources
        for i in range(4)
    ]


def test_records_round_trip_through_jsonl(tmp_path):
    path = tmp_path / "records.jsonl"
    records = _records("da")
    assert write_records(path, records) == len(records)
    assert [r.example_id for r in read_records(path)] == [r.example_id for r in records]


def test_unknown_arm_is_rejected_at_construction():
    with pytest.raises(ValueError):
        ResponseRecord("r", "M", "danm", "s", "e", "fp", 1, "t", 1, 1)


def test_a_missing_source_is_an_error_not_a_smaller_average():
    partial = _records("da", sources=SOURCE_KEYS[:14])
    with pytest.raises(MissingSourceError):
        summarize_arm(partial, arm="da", model="M")


def test_a_source_outside_the_explicit_list_is_an_error():
    extra = _records("da") + _records("da", sources=("ruler/qa_1",))
    with pytest.raises(MissingSourceError, match="outside the explicit list"):
        summarize_arm(extra, arm="da", model="M")


def test_format_failures_count_as_wrong():
    records = _records("da", correct=False)
    for r in records[:30]:
        r.answer_text = None
    summary = summarize_arm(records, arm="da", model="M")
    assert summary.macro_accuracy == 0.0
    assert summary.format_correct_rate == pytest.approx(1 - 30 / len(records))


def test_headline_accuracy_is_a_macro_mean_over_sources():
    records = _records("da")
    # Make one source perfect and one source empty of correct answers.
    for r in records:
        if r.source == SOURCE_KEYS[0]:
            r.correct = False
    summary = summarize_arm(records, arm="da", model="M")
    assert summary.macro_accuracy == pytest.approx(14 / 15)
    # A micro mean over all responses would give the same here only because the
    # sources are balanced; the per-source table is always reported alongside.
    assert summary.per_source_accuracy[SOURCE_KEYS[0]] == 0.0


def test_comparison_reports_percent_change_and_pp_delta():
    v = summarize_arm(_records("vanilla", attended=2000, steps=10), arm="vanilla", model="M")
    d = summarize_arm(
        _records("da", attended=1000, steps=13, correct=False), arm="da", model="M"
    )
    c = compare(v, d)
    assert c.attended_tokens_pct == pytest.approx(-50.0)
    assert c.decode_steps_pct == pytest.approx(30.0)
    assert c.accuracy_delta_pp == pytest.approx(-100.0)


def test_report_gives_both_as_logged_and_filtered_views():
    records = (
        _records("vanilla", attended=2000)
        + _records("da_no_mask", attended=3000)
        + _records("da", attended=1000)
    )
    for r in records:
        if r.arm == "da" and r.example_id.endswith(":0"):
            r.decode_steps = 8192
            r.attended_tokens = 50_000
    out = report(records, model="M")
    assert out["as_logged"]["attended_tokens"]["da"] > out["excl_non_terminating"][
        "attended_tokens"
    ]["da"]
    assert out["excl_non_terminating"]["excluded_non_terminating"]["da"] == 15
