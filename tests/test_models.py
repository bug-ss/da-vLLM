import pytest

from da_vllm import models


def test_lookup_is_exact_not_substring():
    with pytest.raises(models.UnknownModelError):
        models.get_model("gemma-4-31B")  # substring of a registered id
    with pytest.raises(models.UnknownModelError):
        models.get_model("google/gemma-4-31b-it")  # wrong case
    assert models.get_model("google/gemma-4-31B-it").model_type == "gemma4"


def test_local_checkpoints_resolve_by_model_type():
    spec = models.resolve("/scratch/my-local-copy", model_type="qwen3_6")
    assert spec.hub_id == "Qwen/Qwen3.6-27B"
    with pytest.raises(models.UnknownModelError):
        models.resolve("/scratch/x", model_type="qwen3_7")


@pytest.mark.parametrize(
    "hub_id,expected",
    [("google/gemma-4-31B-it", 81_920), ("Qwen/Qwen3.6-27B", 65_536)],
)
def test_global_kv_bytes_match_the_published_geometry(hub_id, expected):
    assert models.get_model(hub_id).geometry.global_kv_bytes_per_token == expected


def test_gemma_global_layers_do_not_borrow_the_sliding_geometry():
    g = models.get_model("google/gemma-4-31B-it").geometry
    # The 2x error: 16 KV heads x 256 dim (sliding) instead of 4 x 512 (global).
    sliding_flavoured = g.num_global_layers * 2 * g.other_kv_heads * g.other_head_dim * 2
    assert sliding_flavoured == 2 * g.global_kv_bytes_per_token


def test_effective_context_limit_reserves_generation_and_template():
    assert models.get_model("Qwen/Qwen3.6-27B").effective_context_limit == 244 * 1024
    assert models.get_model("google/gemma-4-E4B-it").effective_context_limit == 116 * 1024


def test_sampling_params_are_the_model_card_values():
    q = models.get_model("Qwen/Qwen3.6-27B").sampling
    assert (q.temperature, q.top_p, q.top_k, q.presence_penalty) == (0.7, 0.8, 20, 1.5)
    g = models.get_model("google/gemma-4-31B-it").sampling
    assert (g.temperature, g.top_p, g.top_k) == (1.0, 0.95, 64)
    assert models.JUDGE_SAMPLING.max_tokens == 32768


def test_thinking_is_off_for_qwen_rollouts():
    fam = models.get_model("Qwen/Qwen3.6-27B").family
    assert fam.chat_template_kwargs.get("enable_thinking") is False
