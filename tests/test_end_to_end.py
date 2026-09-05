"""Whole pipeline, one decode loop: render -> detect -> drive -> remap.

This is the test that would have caught the silent no-op.  It does not assert
that some function was called; it asserts that the block table the kernel would
actually read shrinks when the model declares ``<focus>``, grows back on
``</focus>``, and contains exactly the blocks the declaration names.
"""

from __future__ import annotations

import math

import pytest
import torch

from da_vllm.detect import build_prompt_map
from da_vllm.masking.logits_processor import DA_ENABLE_KEY, DA_PROMPT_TEXT_KEY, DADriver
from da_vllm.masking.remap import RemapScratch, remap_optimized, remap_readable
from da_vllm.masking.shared import SharedMaskStore
from da_vllm.metrics.replay import replay
from da_vllm.models import get_model
from da_vllm.state_machine import Mode, align_spans

BLOCK_SIZE = 16

RESPONSE = (
    "<global>The founding year is in magic chunk 2.</global>"
    '<focus magic_chunks="2">founded in 2003</focus>'
    "<local>2003 is the answer.</local>"
    "<answer>2003</answer>"
)


class Params:
    def __init__(self, **extra):
        self.extra_args = extra


@pytest.fixture
def rig(family_case, config, filler):
    hub_id, tok, renderer = family_case
    prompt = renderer.render_da(filler, "When was it founded?")
    prompt_ids = tok.encode(prompt.text, add_special_tokens=False)
    pmap = build_prompt_map(
        tok, prompt.text, get_model(hub_id).family, config, prompt_token_ids=prompt_ids
    )
    assert pmap.focus_available
    store = SharedMaskStore(2, config.max_model_len, "cpu")
    driver = DADriver(config, store, tok, hub_id)
    return hub_id, tok, config, prompt, prompt_ids, pmap, store, driver


def _decode_loop(rig, *, optimized: bool):
    hub_id, tok, config, prompt, prompt_ids, pmap, store, driver = rig
    output: list[int] = []
    params = Params(**{DA_ENABLE_KEY: True, DA_PROMPT_TEXT_KEY: prompt.text})
    driver.add(0, "r0", prompt_ids, output, params)

    # A block table wide enough for the whole sequence, with distinctive ids.
    prompt_len = len(prompt_ids)
    response_ids = tok.encode(RESPONSE, add_special_tokens=False)
    max_blocks = math.ceil((prompt_len + len(response_ids)) / BLOCK_SIZE) + 1
    physical = torch.arange(1000, 1000 + max_blocks, dtype=torch.int32)

    scratch = RemapScratch()
    trace = []
    for step, token in enumerate(response_ids):
        output.append(token)
        # The engine's block size is learned from the builder; feed it in.
        driver.step()
        seq_len = prompt_len + step + 1
        num_blocks = math.ceil(seq_len / BLOCK_SIZE)
        block_table = physical[:num_blocks].clone().unsqueeze(0)
        seq_lens = torch.tensor([seq_len], dtype=torch.int32)
        out_seq_lens = torch.zeros(1, dtype=torch.int32)
        kwargs = dict(
            mask=store.tensor,
            block_table=block_table,
            seq_lens=seq_lens,
            out_seq_lens=out_seq_lens,
            query_lens=torch.ones(1, dtype=torch.int32),
            block_size=BLOCK_SIZE,
        )
        if optimized:
            remap_optimized(**kwargs, scratch=scratch)
        else:
            remap_readable(**kwargs)
        kept = math.ceil(int(out_seq_lens[0]) / BLOCK_SIZE)
        trace.append(
            {
                "step": step,
                "mode": driver.rows[0].state.mode,
                "num_blocks": num_blocks,
                "kept_blocks": kept,
                "kept_ids": block_table[0, :kept].tolist(),
                "seq_len": int(out_seq_lens[0]),
            }
        )
    return trace, pmap, prompt_len


def test_the_block_table_shrinks_and_recovers_with_the_declared_mode(rig):
    trace, pmap, prompt_len = _decode_loop(rig, optimized=True)
    modes = {row["mode"] for row in trace}
    assert modes == {Mode.GLOBAL, Mode.FOCUS, Mode.LOCAL}

    for row in trace:
        if row["mode"] is Mode.GLOBAL:
            assert row["kept_blocks"] == row["num_blocks"], "global must read everything"
        else:
            assert row["kept_blocks"] < row["num_blocks"], "a declared span must prune"

    focus = [r for r in trace if r["mode"] is Mode.FOCUS]
    local = [r for r in trace if r["mode"] is Mode.LOCAL]
    # local sees the scaffold only, focus sees the scaffold plus one segment.
    assert min(r["kept_blocks"] for r in focus) > max(r["kept_blocks"] for r in local)


def test_the_kept_blocks_are_exactly_the_ones_the_declaration_names(rig):
    from da_vllm.state_machine import build_mask

    _, _, _, _, _, pmap, _, _ = rig
    trace, _, prompt_len = _decode_loop(rig, optimized=True)
    row = next(r for r in trace if r["mode"] is Mode.FOCUS)

    aligned = align_spans(build_mask(pmap, Mode.FOCUS, (2,)).spans, BLOCK_SIZE)
    expected = {
        b
        for s, e in aligned
        for b in range(s // BLOCK_SIZE, math.ceil(e / BLOCK_SIZE))
    }
    # Everything from the prompt boundary onward is kept unconditionally.
    expected |= set(prompt_len // BLOCK_SIZE + i for i in range(row["num_blocks"]))
    expected = {b for b in expected if b < row["num_blocks"]}

    got = {int(b) - 1000 for b in row["kept_ids"]}
    assert got == expected
    # The physical block ids stay in ascending order after compaction.
    assert row["kept_ids"] == sorted(row["kept_ids"])


def test_the_tail_block_is_always_present(rig):
    trace, _, prompt_len = _decode_loop(rig, optimized=True)
    for row in trace:
        tail_block_id = 1000 + (prompt_len + row["step"]) // BLOCK_SIZE
        assert tail_block_id in row["kept_ids"], (
            "the block holding the token just written must stay visible or the "
            "kernel cannot see the current token's own key"
        )


def test_readable_and_optimized_drive_identical_decode_loops(family_case, config, filler):
    def build():
        hub_id, tok, renderer = family_case
        prompt = renderer.render_da(filler, "When was it founded?")
        prompt_ids = tok.encode(prompt.text, add_special_tokens=False)
        pmap = build_prompt_map(
            tok, prompt.text, get_model(hub_id).family, config, prompt_token_ids=prompt_ids
        )
        store = SharedMaskStore(2, config.max_model_len, "cpu")
        return hub_id, tok, config, prompt, prompt_ids, pmap, store, DADriver(
            config, store, tok, hub_id
        )

    a, _, _ = _decode_loop(build(), optimized=False)
    b, _, _ = _decode_loop(build(), optimized=True)
    assert [r["kept_ids"] for r in a] == [r["kept_ids"] for r in b]
    assert [r["seq_len"] for r in a] == [r["seq_len"] for r in b]


def test_replayed_attended_tokens_match_what_the_kernel_actually_read(rig):
    hub_id, tok, config, prompt, prompt_ids, pmap, store, driver = rig
    trace, _, prompt_len = _decode_loop(rig, optimized=True)

    result = replay(pmap, tok, RESPONSE, config, mask_lag_steps=0, block_size=BLOCK_SIZE)
    assert result.decode_steps == len(trace)

    # The engine's own accounting: kept blocks x block size, minus the response
    # positions the replay counts separately.
    engine_total = sum(r["kept_blocks"] * BLOCK_SIZE for r in trace)
    replay_total = result.block_aligned_attended_tokens
    # Both are block-granular counts of the same thing. They differ only in
    # how the partial tail block of each step is charged, so they agree to
    # within one block per step.
    assert abs(engine_total - replay_total) <= BLOCK_SIZE * len(trace)
    # And the block-granular count is always the larger of the two: the kernel
    # reads whole blocks.
    assert result.attended_tokens < result.block_aligned_attended_tokens


def test_a_declined_focus_costs_efficiency_but_never_correctness(rig):
    hub_id, tok, config, prompt, prompt_ids, pmap, store, driver = rig
    output: list[int] = []
    driver.add(
        0,
        "r0",
        prompt_ids,
        output,
        Params(**{DA_ENABLE_KEY: True, DA_PROMPT_TEXT_KEY: prompt.text}),
    )
    output.extend(tok.encode('<focus magic_chunks="99999">x', add_special_tokens=False))
    driver.step()
    assert driver.rows[0].state.mode is Mode.GLOBAL
    assert bool(store.tensor[0].all())  # full attention, nothing pruned
