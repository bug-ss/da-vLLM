"""Batch reconciliation and per-step driving (guide 6.2)."""

from __future__ import annotations

import pytest
import torch

from da_vllm.masking.logits_processor import DA_ENABLE_KEY, DA_PROMPT_TEXT_KEY, DADriver
from da_vllm.masking.shared import SharedMaskStore
from da_vllm.state_machine import Mode


class Params:
    def __init__(self, **extra):
        self.extra_args = extra or None


@pytest.fixture
def setup(family_case, config, filler):
    hub_id, tok, renderer = family_case
    prompt = renderer.render_da(filler, "Who?")
    ids = tok.encode(prompt.text, add_special_tokens=False)
    store = SharedMaskStore(config.max_num_seqs, config.max_model_len, "cpu")
    driver = DADriver(config, store, tok, hub_id)
    return driver, store, tok, prompt, ids


def _da_params(prompt):
    return Params(**{DA_ENABLE_KEY: True, DA_PROMPT_TEXT_KEY: prompt.text})


def test_non_da_requests_never_build_a_state_machine(setup):
    driver, store, _, prompt, ids = setup
    driver.add(0, "r0", ids, [], Params())
    assert driver.rows == {}
    assert driver.stats.skipped_no_da == 1
    assert bool(store.tensor[0].all())


def test_da_request_builds_a_state_and_starts_global(setup):
    driver, store, _, prompt, ids = setup
    driver.add(0, "r0", ids, [], _da_params(prompt))
    driver.step()
    assert driver.rows[0].state.mode is Mode.GLOBAL
    assert bool(store.tensor[0].all())


def test_focus_writes_a_compacted_row(setup):
    driver, store, tok, prompt, ids = setup
    output: list[int] = []
    driver.add(0, "r0", ids, output, _da_params(prompt))
    driver.step()
    output.extend(tok.encode('<focus magic_chunks="2">', add_special_tokens=False))
    written = driver.step()
    assert written == [0]
    assert driver.rows[0].state.mode is Mode.FOCUS
    kept = int(store.tensor[0, : len(ids)].sum())
    assert 0 < kept < len(ids)


def test_the_output_list_is_read_live_not_copied(setup):
    driver, _, tok, prompt, ids = setup
    output: list[int] = []
    driver.add(0, "r0", ids, output, _da_params(prompt))
    for token in tok.encode("<local>x</local>", add_special_tokens=False):
        output.append(token)
        driver.step()
    assert driver.rows[0].state.num_consumed == len(output)


def test_placeholder_tokens_are_revisited_next_step(setup):
    driver, _, tok, prompt, ids = setup
    tag = tok.encode('<focus magic_chunks="1">', add_special_tokens=False)
    output = list(tag[:-1]) + [-1]
    driver.add(0, "r0", ids, output, _da_params(prompt))
    driver.step()
    assert driver.rows[0].state.mode is Mode.GLOBAL
    output[-1] = tag[-1]
    driver.step()
    assert driver.rows[0].state.mode is Mode.FOCUS


def test_removed_rows_are_reset_to_all_true(setup):
    driver, store, tok, prompt, ids = setup
    output = list(tok.encode("<local>x", add_special_tokens=False))
    driver.add(0, "r0", ids, output, _da_params(prompt))
    driver.step()
    assert not bool(store.tensor[0].all())
    driver.remove(0)
    assert 0 not in driver.rows
    assert bool(store.tensor[0].all())


def test_a_slot_recycled_by_a_non_da_request_is_cleaned(setup):
    driver, store, tok, prompt, ids = setup
    driver.add(0, "r0", ids, list(tok.encode("<local>x", add_special_tokens=False)),
               _da_params(prompt))
    driver.step()
    assert not bool(store.tensor[0].all())
    driver.add(0, "r1", ids, [], Params())  # plain request lands in the same slot
    assert bool(store.tensor[0].all())
    assert 0 not in driver.rows


def test_unidirectional_move_vacates_the_source(setup):
    driver, store, tok, prompt, ids = setup
    driver.add(0, "r0", ids, list(tok.encode("<local>x", add_special_tokens=False)),
               _da_params(prompt))
    driver.step()
    driver.move(0, 3, swap=False)
    assert 0 not in driver.rows and 3 in driver.rows
    assert bool(store.tensor[0].all())
    driver.step()
    assert not bool(store.tensor[3].all())


def test_swap_exchanges_two_rows_and_rewrites_both(setup):
    driver, store, tok, prompt, ids = setup
    focus = list(tok.encode('<focus magic_chunks="1">', add_special_tokens=False))
    local = list(tok.encode("<local>x", add_special_tokens=False))
    driver.add(0, "r0", ids, focus, _da_params(prompt))
    driver.add(1, "r1", ids, local, _da_params(prompt))
    driver.step()
    a_before = store.tensor[0].clone()
    b_before = store.tensor[1].clone()
    assert not torch.equal(a_before, b_before)

    driver.move(0, 1, swap=True)
    assert driver.rows[1].state.mode is Mode.FOCUS
    assert driver.rows[0].state.mode is Mode.LOCAL
    assert set(driver.step()) == {0, 1}
    assert torch.equal(store.tensor[0], b_before)
    assert torch.equal(store.tensor[1], a_before)


def test_swap_with_an_empty_slot_resets_the_vacated_row(setup):
    driver, store, tok, prompt, ids = setup
    driver.add(2, "r0", ids, list(tok.encode("<local>x", add_special_tokens=False)),
               _da_params(prompt))
    driver.step()
    driver.move(2, 5, swap=True)  # slot 5 holds nothing
    assert bool(store.tensor[2].all())
    assert 5 in driver.rows


def test_detection_failure_disables_focus_without_raising(setup, config, family_case):
    driver, store, tok, prompt, ids = setup
    bad_prompt_text = "there are no tool turns in this string"
    params = Params(**{DA_ENABLE_KEY: True, DA_PROMPT_TEXT_KEY: bad_prompt_text})
    driver.add(0, "bad", ids, [], params)
    assert driver.stats.detection_failures
    driver.step()
    focus = tok.encode('<focus magic_chunks="1">', add_special_tokens=False)
    driver.rows[0].output_token_ids = focus
    driver.step()
    assert driver.rows[0].state.mode is Mode.GLOBAL
    assert bool(store.tensor[0].all())


def test_steady_state_writes_nothing_when_the_mode_does_not_change(setup):
    driver, _, tok, prompt, ids = setup
    output: list[int] = []
    driver.add(0, "r0", ids, output, _da_params(prompt))
    driver.step()
    output.extend(tok.encode('<focus magic_chunks="1">', add_special_tokens=False))
    assert driver.step() == [0]
    for token in tok.encode(" some extracted value", add_special_tokens=False):
        output.append(token)
        assert driver.step() == []


def test_the_processor_satisfies_vllms_real_logits_processor_contract(monkeypatch):
    """vLLM loads a custom processor by ``module:qualname`` and then does
    ``issubclass(obj, LogitsProcessor)``.  A dotted path fails to unpack, and a
    non-subclass is rejected -- either way the driver never runs and the mask
    silently does nothing."""
    import importlib

    import fake_vllm
    from da_vllm.serving import DA_LOGITS_PROCESSOR_FQCN

    fake_vllm.install(monkeypatch)
    import da_vllm.masking.logits_processor as module

    importlib.reload(module)
    try:
        loaded = fake_vllm.load_by_fqcn(DA_LOGITS_PROCESSOR_FQCN)
        assert loaded is module.DALogitsProcessor
        assert issubclass(loaded, fake_vllm.LogitsProcessor)
        # The ABC declares four abstract methods; a missing one would make the
        # class un-instantiable at engine start.
        assert not getattr(loaded, "__abstractmethods__", frozenset())
        with pytest.raises(ValueError):
            fake_vllm.load_by_fqcn("da_vllm.masking.logits_processor:DADriver")
    finally:
        importlib.reload(module)


def test_the_fqcn_uses_a_colon_not_a_dot():
    from da_vllm.serving import DA_LOGITS_PROCESSOR_FQCN

    module_path, _, qualname = DA_LOGITS_PROCESSOR_FQCN.partition(":")
    assert qualname == "DALogitsProcessor"
    assert module_path == "da_vllm.masking.logits_processor"
    # vLLM does `logitproc.split(":")` and unpacks into exactly two names.
    assert len(DA_LOGITS_PROCESSOR_FQCN.split(":")) == 2
