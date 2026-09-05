import pytest
import torch

from da_vllm.state_machine import Mode, MaskSnapshot
from da_vllm.validation.checks import (
    KeptFractionReport,
    StepToggle,
    ValidationError,
    assert_prompt_parity,
    assert_round_trip,
    default_cases,
    kept_fraction_report,
    kernel_scaling,
    round_trip,
)
from da_vllm.validation.nll_parity import (
    ParityResult,
    block_align_mask,
    da_mask_4d,
    sliding_mask_4d,
)


def test_round_trip_passes_on_every_adversarial_context(family_case, config, filler):
    hub_id, tok, renderer = family_case
    results = round_trip(tok, hub_id, default_cases(filler), config, renderer=renderer)
    assert_round_trip(results)
    names = {r.name for r in results}
    assert {"empty", "contains_turn_literal", "contains_magic_chunk_header"} <= names


def test_round_trip_reports_a_broken_family_rather_than_passing(family_case, config, filler):
    hub_id, tok, renderer = family_case
    results = round_trip(tok, hub_id, default_cases(filler), config, renderer=renderer)
    results[0].ok = False
    results[0].failure_reason = "synthetic"
    with pytest.raises(ValidationError):
        assert_round_trip(results)


def test_serve_and_replay_must_render_the_same_prompt(family_case, filler):
    _, _, renderer = family_case
    served = renderer.render_da(filler, "Who?")
    assert_prompt_parity(served, renderer.render_da(filler, "Who?"))
    with pytest.raises(ValidationError, match="different prompts"):
        assert_prompt_parity(served, renderer.render_da(filler, "Who won?"))


def test_kept_fraction_block_level_is_at_least_token_level():
    r = kept_fraction_report(
        attended_tokens=100, block_attended_tokens=104, full_attended_tokens=400
    )
    assert r.agree and r.token_level == 0.25
    assert not KeptFractionReport(0.5, 0.4).agree


def test_kernel_scaling_reports_efficiency_against_the_ideal():
    # A kernel that tracks the kept fraction perfectly.
    report = kernel_scaling(lambda f: f, [1.0, 0.5, 0.25], repeats=1)
    assert report.efficiency == pytest.approx(1.0)
    assert report.tracks_mask
    # A kernel that ignores the mask entirely.
    flat = kernel_scaling(lambda f: 1.0, [1.0, 0.5, 0.25], repeats=1)
    assert not flat.tracks_mask


def test_step_toggle_alternates_within_one_boot():
    t = StepToggle(("readable", "optimized"))
    for i in range(6):
        label = t.next_label()
        t.record(label, 1.0 if label == "readable" else 0.5)
    summary = t.summary()
    assert summary["readable"]["n"] == summary["optimized"]["n"] == 3
    assert summary["optimized"]["median"] < summary["readable"]["median"]


def test_three_column_parity_verdicts():
    assert ParityResult(1.00, 1.02, 1.40).passed
    silent = ParityResult(1.00, 1.00, 1.00)
    assert not silent.passed
    assert "silent-no-op" in silent.explain()
    drifted = ParityResult(1.00, 1.90, 2.50)
    assert not drifted.passed and "differ by more than" in drifted.explain()


def test_reference_mask_or_reduces_at_the_served_block_size():
    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[0, 3] = True
    aligned = block_align_mask(mask, 4)
    assert aligned[0, :4].all() and not aligned[0, 4:16].any()


def test_da_reference_mask_keeps_the_scaffold_and_the_future():
    snapshot = MaskSnapshot(Mode.FOCUS, ((0, 4), (10, 14)), 16)
    m = da_mask_4d(snapshot, prompt_len=16, total_len=20, block_size=4)
    assert m.shape == (1, 1, 20, 20)
    # Prompt rows keep full causal attention: the prompt was prefilled unmasked.
    assert m[0, 0, 15, :16].all()
    # Generated rows see the spans, the boundary onward, and nothing else.
    row = m[0, 0, 18]
    assert row[:4].all() and not row[4:8].any() and row[8:19].all()


def test_sliding_reference_mask_is_a_window():
    m = sliding_mask_4d(6, 3)[0, 0]
    assert m[5].tolist() == [False, False, False, True, True, True]
