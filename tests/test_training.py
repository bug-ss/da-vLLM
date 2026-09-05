"""Training-side helpers (guide 13).  Not part of the paper's results."""

from __future__ import annotations

import math

import pytest
import torch

from da_vllm.training import (
    FLEX_SEQ_MULTIPLE,
    MissingMaskError,
    allowed_block_table,
    build_sample,
    chat_template_override,
    compile_block_mask,
    flex_kernel_options,
    loss_positions,
    mask_mod_from_trace,
    pad_to_flex_multiple,
    require_mask,
)

RESPONSE = (
    "<global>chunk 2 has it</global>"
    '<focus magic_chunks="2">founded 2003</focus>'
    "<local>so 2003</local><answer>2003</answer>"
)


@pytest.fixture
def sample(family_case, config, filler):
    _, _, renderer = family_case
    return build_sample(
        renderer, config, context=filler, question="When?", response_text=RESPONSE
    )


def test_loss_covers_the_response_and_never_the_tool_turns(sample):
    assert not any(sample.assistant_mask[: sample.prompt_len])
    assert all(sample.assistant_mask[sample.prompt_len :])
    assert loss_positions(sample)[0] == sample.prompt_len


def test_the_trace_produces_more_than_one_mode_group(sample):
    assert sample.meta["num_groups"] >= 3  # global, focus, local at least
    assert set(sample.group_of_query[: sample.prompt_len]) == {-1}
    assert min(sample.group_of_query[sample.prompt_len :]) >= 0


def test_group_spans_differ_between_modes(sample):
    sizes = {sum(e - s for s, e in spans) for spans in sample.group_spans}
    assert len(sizes) > 1  # a single sample-wide mask would collapse these


def test_padding_is_right_aligned_to_128_and_keeps_the_sink_at_the_front(sample):
    padded = pad_to_flex_multiple(sample, pad_token_id=0)
    assert padded.total_len % FLEX_SEQ_MULTIPLE == 0
    assert padded.input_ids[: sample.total_len] == sample.input_ids
    assert padded.group_of_query[sample.total_len :] == [-1] * padded.pad_len
    assert not any(padded.assistant_mask[sample.total_len :])


def test_padding_is_a_no_op_when_already_aligned():
    from da_vllm.training import TrainingSample

    s = TrainingSample(
        input_ids=[1] * 256,
        assistant_mask=[True] * 256,
        group_of_query=[0] * 256,
        group_spans=(((0, 16),),),
        boundary=128,
        prompt_len=128,
    )
    assert pad_to_flex_multiple(s, 0) is s


def test_compile_refuses_an_unpadded_length():
    with pytest.raises(ValueError, match="153799"):
        compile_block_mask(lambda *a: True, 100, 100, device="cpu")


def test_allowed_table_is_built_at_the_served_block_size(sample):
    kv_len = sample.total_len
    table = allowed_block_table(sample, kv_len, 16)
    assert table.shape == (len(sample.group_spans), math.ceil(kv_len / 16))
    # Everything from the prompt boundary onward is kept in every group.
    assert bool(table[:, sample.boundary // 16 :].all())
    # And no group keeps every block, or the mask would be doing nothing.
    assert not bool(table.all())


def test_mask_mod_reproduces_the_group_pattern(sample):
    kv_block = 16
    kv_len = sample.total_len
    table = allowed_block_table(sample, kv_len, kv_block)
    groups = torch.tensor(sample.group_of_query)
    mod = mask_mod_from_trace(groups, table, kv_block)

    q = torch.tensor([sample.prompt_len + 1])
    kv = torch.arange(kv_len)
    row = mod(None, None, q.expand(kv_len), kv)
    g = int(groups[sample.prompt_len + 1])
    expected = table[g][kv // kv_block] & (kv <= sample.prompt_len + 1)
    assert torch.equal(row, expected)


def test_prompt_rows_stay_plain_causal(sample):
    kv_block = 16
    table = allowed_block_table(sample, sample.total_len, kv_block)
    groups = torch.tensor(sample.group_of_query)
    mod = mask_mod_from_trace(groups, table, kv_block)
    kv = torch.arange(sample.total_len)
    q = torch.full_like(kv, sample.prompt_len - 1)
    row = mod(None, None, q, kv)
    assert torch.equal(row, kv <= sample.prompt_len - 1)


@pytest.mark.parametrize(
    "head_dim,shared,expect_small",
    [(128, 164 * 1024, False), (256, 164 * 1024, True), (256, 228 * 1024, False)],
)
def test_flex_kernel_options_dispatch_on_head_dim_and_device_limit(
    head_dim, shared, expect_small
):
    options = flex_kernel_options(head_dim, shared)
    if head_dim < 256:
        assert options == {}
    elif expect_small:
        assert options["BLOCK_M"] == 32  # an A100 crashes on the defaults
    else:
        assert options["BLOCK_M"] == 128


def test_a_dropped_mask_key_is_loud():
    with pytest.raises(MissingMaskError, match="attention_mask_4d"):
        require_mask({"input_ids": [1]}, "attention_mask_4d")
    with pytest.raises(MissingMaskError):
        require_mask({"attention_mask_4d": None}, "attention_mask_4d")
    assert require_mask({"m": 1}, "m") == 1


def test_a_sample_with_no_assistant_tokens_is_an_error():
    from da_vllm.training import TrainingSample

    s = TrainingSample([1, 2], [False, False], [-1, -1], ((),), 2, 2)
    with pytest.raises(MissingMaskError):
        loss_positions(s)


def test_chat_template_overrides_key_on_model_type_and_raise_on_unknown():
    overrides = {"gemma4": "template-a"}
    assert chat_template_override("gemma4", overrides) == "template-a"
    with pytest.raises(KeyError):
        chat_template_override("gemma4_local_copy", overrides)
