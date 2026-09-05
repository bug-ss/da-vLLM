import pytest

from da_vllm.detect import build_prompt_map
from da_vllm.metrics.replay import (
    NON_TERMINATING_STEPS,
    extract_answer,
    replay,
    replay_vanilla,
    vanilla_attended_tokens,
)
from da_vllm.metrics.roofline import (
    GeometryError,
    count_global_layers,
    global_kv_bytes_from_shapes,
    global_kv_bytes_per_token,
    roofline_response,
    verify_geometry,
)
from da_vllm.models import get_model

RESPONSE = (
    "<global>The founding year should be in magic chunk 2.</global>"
    '<focus magic_chunks="2">Acme Corp was founded in 2003</focus>'
    "<local>The answer is 2003.</local>"
    "<answer>2003</answer>"
)


@pytest.fixture
def pmap(family_case, config, filler):
    hub_id, tok, renderer = family_case
    prompt = renderer.render_da(filler, "When founded?")
    return tok, build_prompt_map(tok, prompt.text, get_model(hub_id).family, config)


def test_vanilla_baseline_is_analytic():
    assert vanilla_attended_tokens(1000, 3) == 1000 + 1001 + 1002


def test_replay_totals_attended_tokens_below_vanilla(pmap, config):
    tok, m = pmap
    result = replay(m, tok, RESPONSE, config)
    assert result.decode_steps > 0
    assert result.attended_tokens < vanilla_attended_tokens(
        m.num_prompt_tokens, result.decode_steps
    )
    assert result.focus_attempts == 1 and result.focus_granted == 1
    assert result.answer_text == "2003"
    assert result.format_ok


def test_replay_counts_every_step_in_exactly_one_mode(pmap, config):
    tok, m = pmap
    result = replay(m, tok, RESPONSE, config)
    assert sum(result.mode_steps.values()) == result.decode_steps
    assert result.mode_steps["focus"] > 0
    assert result.mode_steps["local"] > 0


def test_mask_lag_only_shifts_a_couple_of_steps(pmap, config):
    tok, m = pmap
    lagged = replay(m, tok, RESPONSE, config, mask_lag_steps=2)
    declared = replay(m, tok, RESPONSE, config, mask_lag_steps=0)
    assert declared.attended_tokens <= lagged.attended_tokens
    # The lag can only cost a couple of steps of extra attention per transition.
    per_step = m.num_prompt_tokens
    assert lagged.attended_tokens - declared.attended_tokens <= 8 * per_step


def test_block_granularity_reads_at_least_as_much_as_token_granularity(pmap, config):
    tok, m = pmap
    result = replay(m, tok, RESPONSE, config, block_size=32)
    assert result.block_aligned_attended_tokens >= result.attended_tokens


def test_a_response_with_no_answer_tag_is_a_format_failure(pmap, config):
    tok, m = pmap
    result = replay(m, tok, "<global>thinking forever", config)
    assert result.answer_text is None
    assert not result.format_ok


def test_non_terminating_flag(pmap, config):
    tok, m = pmap
    result = replay_vanilla(1000, "x", tok, response_token_ids=list(range(NON_TERMINATING_STEPS)))
    assert result.non_terminating
    short = replay_vanilla(1000, "x", tok, response_token_ids=[1, 2, 3])
    assert not short.non_terminating


def test_extract_answer_takes_the_last_tag():
    assert extract_answer("<answer>a</answer> ... <answer>b</answer>") == "b"
    assert extract_answer("no tag") is None


# -- roofline --------------------------------------------------------------

GEMMA_CONFIG = {
    "layer_types": ["full_attention"] * 10 + ["sliding_attention"] * 50,
    "global_head_dim": 512,
    "num_global_key_value_heads": 4,
    "head_dim": 256,
    "num_key_value_heads": 16,
    "sliding_window": 1024,
}


def test_layer_types_is_the_only_source_for_the_layer_count():
    assert count_global_layers(GEMMA_CONFIG) == (60, 10)
    with pytest.raises(GeometryError):
        count_global_layers({"num_hidden_layers": 60})


def test_gemma_global_geometry_is_read_explicitly():
    assert global_kv_bytes_per_token(GEMMA_CONFIG) == 81_920
    verify_geometry(get_model("google/gemma-4-31B-it"), GEMMA_CONFIG)


def test_a_hybrid_config_without_global_head_dims_is_refused():
    partial = {k: v for k, v in GEMMA_CONFIG.items() if not k.startswith("global") and k != "num_global_key_value_heads"}
    with pytest.raises(GeometryError, match="global_head_dim"):
        global_kv_bytes_per_token(partial)


def test_a_wrong_registry_entry_is_caught_by_the_cross_check():
    wrong = dict(GEMMA_CONFIG, global_head_dim=256)  # the 2x error
    with pytest.raises(GeometryError, match="kv bytes/token"):
        verify_geometry(get_model("google/gemma-4-31B-it"), wrong)


def test_checkpoint_shapes_give_the_same_answer():
    shapes = {f"model.layers.{i}.self_attn.k_proj.weight": (4 * 512, 4096) for i in range(10)}
    assert global_kv_bytes_from_shapes(shapes, range(10)) == 81_920


def test_only_the_global_kv_term_moves_with_the_mask():
    spec = get_model("google/gemma-4-31B-it")
    full = roofline_response(
        geometry=spec.geometry,
        active_params=spec.active_params,
        attended_tokens=13_430_000,
        decode_steps=323,
    )
    masked = roofline_response(
        geometry=spec.geometry,
        active_params=spec.active_params,
        attended_tokens=6_450_000,
        decode_steps=323,
    )
    assert masked.matmul_s == full.matmul_s
    assert masked.local_s == full.local_s
    assert masked.global_kv_s < full.global_kv_s


@pytest.mark.parametrize(
    "hub_id,steps_v,attended_v,attended_da,step_growth,expected",
    [
        ("google/gemma-4-31B-it", 323, 13.43e6, 6.45e6, 1.348, 0.71),
        ("Qwen/Qwen3.6-27B", 535, 22.54e6, 15.52e6, 1.313, 0.77),
    ],
)
def test_reproduces_the_papers_projected_speedups(
    hub_id, steps_v, attended_v, attended_da, step_growth, expected
):
    """0.71x on Gemma-4-31B and 0.77x on Qwen-3.6-27B (paper section 5.4).

    Both are recovered at a mean vanilla prompt length near 41-42K tokens,
    which is what the two independently imply -- a useful check that the byte
    counts and the utilization constants are right.
    """
    spec = get_model(hub_id)
    v = roofline_response(
        geometry=spec.geometry,
        active_params=spec.active_params,
        attended_tokens=int(attended_v),
        decode_steps=steps_v,
    )
    da = roofline_response(
        geometry=spec.geometry,
        active_params=spec.active_params,
        attended_tokens=int(attended_da),
        decode_steps=round(steps_v * step_growth),
    )
    assert da.total_s / v.total_s == pytest.approx(expected, abs=0.01)
