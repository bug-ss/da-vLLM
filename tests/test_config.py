import pytest

from da_vllm.config import ConfigError, DAConfig, RunawayConfig


def test_da_is_off_unless_asked():
    assert DAConfig().enabled is False


def test_unknown_key_raises_rather_than_silently_disabling_masking():
    with pytest.raises(ConfigError, match="unknown DAConfig keys"):
        DAConfig.from_dict({"enable": True})
    with pytest.raises(ConfigError, match="unknown RunawayConfig keys"):
        DAConfig.from_dict({"runaway": {"enable": True}})


def test_round_trip_through_dict():
    cfg = DAConfig(enabled=True, sink_tokens=16, runaway=RunawayConfig(enabled=True))
    assert DAConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize(
    "kwargs",
    [
        {"segment_target_tokens": 2048, "segment_max_tokens": 1024},
        {"max_focus_ids": 0},
        {"sink_tokens": -1},
        {"question_header": "   "},
        {"max_num_seqs": 0},
    ],
)
def test_invalid_values_raise(kwargs):
    with pytest.raises(ConfigError):
        DAConfig(**kwargs)


def test_runaway_thresholds_are_the_calibrated_ones():
    r = RunawayConfig()
    # Calibrated so the longest legitimate low-novelty run (2,349 tokens)
    # passes; an earlier 250 threshold fired on real answers.
    assert r.token_sustain == 2700
    assert r.token_distinct_ratio == 0.30
    assert r.word_sustain == 2000
    assert r.enabled is False


def test_segment_sizes_flow_from_the_config_into_the_renderer():
    from da_vllm.prompt import PromptRenderer
    from fakes import qwen_tokenizer

    cfg = DAConfig(enabled=True, segment_target_tokens=64, segment_max_tokens=80)
    renderer = PromptRenderer(qwen_tokenizer(), "Qwen/Qwen3.6-27B", config=cfg)
    assert renderer.segmenter.target_tokens == 64
    assert renderer.segmenter.max_tokens == 80


def test_tag_tail_slack_reaches_the_detection_regexes():
    from da_vllm.state_machine import tag_patterns

    strict = dict(tag_patterns(0))
    loose = dict(tag_patterns(8))
    assert strict["open_local"].search("<local>") is not None
    assert strict["open_local"].search("<local>abc") is None
    assert loose["open_local"].search("<local>abc") is not None
