"""The high-level DAEngine, driven through an injected backend (no GPU)."""

from __future__ import annotations

import pytest

from da_vllm import DAConfig, DAEngine

RESPONSE = (
    "<global>The founding year is in magic chunk 2.</global>"
    '<focus magic_chunks="2">Acme was founded in 2003</focus>'
    "<local>The answer is 2003.</local>"
    "<answer>2003</answer>"
)


def _engine(family_case, arm="da", response=RESPONSE, **kwargs):
    hub_id, tok, _ = family_case
    config = DAConfig(
        enabled=(arm == "da"),
        max_model_len=8192,
        max_num_seqs=8,
        segment_target_tokens=120,
        segment_max_tokens=150,
    )

    def generate_fn(token_id_lists, params):
        assert params["max_tokens"] > 0
        ids = tok.encode(response, add_special_tokens=False)
        return [(response, ids, "stop") for _ in token_id_lists]

    if "max_tokens" not in kwargs:
        kwargs["max_tokens"] = 512  # the test window is small
    return DAEngine(
        hub_id, arm=arm, config=config, tokenizer=tok, generate_fn=generate_fn, **kwargs
    )


def test_answer_extracts_the_tag_and_accounts_for_attention(family_case, filler):
    engine = _engine(family_case)
    result = engine.answer(filler, "When was Acme founded?")
    assert result.answer == "2003"
    assert result.format_ok and not result.non_terminating
    assert result.num_segments > 1
    assert result.detection_failure is None
    assert result.focus_attempts == 1 and result.focus_granted == 1
    assert result.attended_tokens < result.baseline_attended_tokens
    assert result.reduction_pct < 0
    assert "focus" in dict(result.mode_trace).values() or any(
        m == "focus" for _, m in result.mode_trace
    )
    assert result.mode_steps["focus"] > 0 and result.mode_steps["local"] > 0
    assert "attended tokens" in result.summary()


def test_the_vanilla_arm_reads_everything(family_case, filler):
    engine = _engine(family_case, arm="vanilla")
    result = engine.answer(filler, "When was Acme founded?")
    assert result.attended_tokens == result.baseline_attended_tokens
    assert result.reduction_pct == 0.0
    assert result.num_segments == 0


def test_the_no_mask_arm_shares_the_da_prompt_but_reads_everything(family_case, filler):
    da = _engine(family_case, arm="da")
    nm = _engine(family_case, arm="da_no_mask")
    a = da.answer(filler, "When?")
    b = nm.answer(filler, "When?")
    assert a.prompt_fingerprint == b.prompt_fingerprint  # identical prompts
    assert b.attended_tokens == b.baseline_attended_tokens  # but no mask
    assert a.attended_tokens < b.attended_tokens


def test_arm_and_config_must_agree(family_case):
    hub_id, tok, _ = family_case
    with pytest.raises(ValueError, match="disagree"):
        DAEngine(hub_id, arm="da", config=DAConfig(enabled=False), tokenizer=tok)


def test_a_batch_is_one_engine_call(family_case, filler):
    calls = []
    hub_id, tok, _ = family_case
    config = DAConfig(enabled=True, max_model_len=8192, segment_target_tokens=120,
                      segment_max_tokens=150)

    def generate_fn(token_id_lists, params):
        calls.append(len(token_id_lists))
        ids = tok.encode(RESPONSE, add_special_tokens=False)
        return [(RESPONSE, ids, "stop") for _ in token_id_lists]

    engine = DAEngine(
        hub_id, config=config, tokenizer=tok, generate_fn=generate_fn, max_tokens=512
    )
    results = engine.answer_batch([(filler, "a?"), (filler, "b?"), (filler, "c?")])
    assert calls == [3]
    assert len(results) == 3
    # Different questions render different prompts.
    assert len({r.prompt_fingerprint for r in results}) == 3


def test_an_over_length_prompt_is_refused_not_truncated(family_case):
    hub_id, tok, _ = family_case
    config = DAConfig(enabled=True, max_model_len=2048, segment_target_tokens=120,
                      segment_max_tokens=150)
    engine = DAEngine(
        hub_id, config=config, tokenizer=tok, generate_fn=lambda a, b: [], max_tokens=1024
    )
    from da_vllm.testing import lorem

    with pytest.raises(ValueError, match="never truncated"):
        engine.answer(lorem(40), "When?")


def test_a_declined_focus_is_reported_not_hidden(family_case, filler):
    engine = _engine(
        family_case, response='<focus magic_chunks="9999">x</focus><answer>?</answer>'
    )
    result = engine.answer(filler, "When?")
    assert result.focus_attempts == 1 and result.focus_granted == 0
    assert result.declines == ["unknown_id"]


def test_a_response_with_no_answer_tag_is_a_format_failure(family_case, filler):
    engine = _engine(family_case, response="<global>thinking and never stopping")
    result = engine.answer(filler, "When?")
    assert not result.format_ok and result.answer is None


def test_the_prompt_map_is_cached_per_prompt(family_case, filler):
    engine = _engine(family_case)
    engine.answer(filler, "When?")
    engine.answer(filler, "When?")
    assert len(engine._prompt_map_cache) == 1


def test_sampling_params_come_from_the_model_card(family_case, filler):
    engine = _engine(family_case, max_tokens=None)
    params = engine.sampling_params()
    assert params["max_tokens"] == 8192  # the paper's generation cap
    if engine.spec.hub_id.startswith("Qwen"):
        assert params["temperature"] == 0.7 and params["presence_penalty"] == 1.5
    else:
        assert params["temperature"] == 1.0 and params["top_k"] == 64


def test_close_is_safe_without_an_engine(family_case):
    engine = _engine(family_case)
    engine.close()
    with _engine(family_case) as e:
        assert e is not None


@pytest.mark.parametrize("paragraphs,expect_win", [(10, False), (200, True)])
def test_da_only_pays_once_the_context_clears_the_scaffold(
    family_case, paragraphs, expect_win
):
    """The scaffold is not free, which is why the eval filters short contexts.

    DA adds the tool declaration, the ~1.7K-token instruction, and a turn
    wrapper per magic chunk, then spends roughly half its steps in global mode
    at the full (larger) prompt length. Below a few thousand context tokens
    that overhead dominates and DA reads *more* than vanilla. The evaluation's
    4096-token floor exists for exactly this reason.
    """
    from da_vllm.testing import lorem

    hub_id, tok, _ = family_case
    document = lorem(paragraphs, seed=2)

    def build(arm):
        config = DAConfig(
            enabled=arm == "da",
            max_model_len=262_144,
            segment_target_tokens=2048,
            segment_max_tokens=2560,
        )

        def generate_fn(token_id_lists, params):
            ids = tok.encode(RESPONSE, add_special_tokens=False)
            return [(RESPONSE, ids, "stop") for _ in token_id_lists]

        return DAEngine(
            hub_id, arm=arm, config=config, tokenizer=tok,
            generate_fn=generate_fn, max_tokens=1024,
        )

    da = build("da").answer(document, "When?")
    nomask = build("da_no_mask").answer(document, "When?")
    vanilla = build("vanilla").answer(document, "When?")

    # The no-mask arm always reads more than vanilla: same prompt, no mask.
    assert nomask.attended_tokens > vanilla.attended_tokens
    # Whether the mask wins back more than the format costs depends on length.
    assert (da.attended_tokens < vanilla.attended_tokens) is expect_win
    # But the mask always beats the same prompt unmasked -- that is the ablation.
    assert da.attended_tokens < nomask.attended_tokens
