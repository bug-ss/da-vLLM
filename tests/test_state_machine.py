import pytest

from da_vllm.detect import build_prompt_map
from da_vllm.models import get_model
from da_vllm.state_machine import (
    DAStateMachine,
    DeclineReason,
    IncrementalDetokenizer,
    Mode,
    align_spans,
    build_mask,
)


@pytest.fixture
def machine(family_case, config, filler):
    hub_id, tok, renderer = family_case
    prompt = renderer.render_da(filler, "Who?")
    pmap = build_prompt_map(tok, prompt.text, get_model(hub_id).family, config)
    return tok, pmap, DAStateMachine(pmap, tok, config)


def _feed(tok, sm, text, *, chunk=1):
    """Append ``text`` to this machine's output stream, one chunk at a time.

    ``advance`` is always handed the whole live output list, exactly as vLLM
    hands it over, so repeated calls accumulate rather than restart.
    """
    stream = getattr(sm, "_test_stream", None)
    if stream is None:
        stream = sm._test_stream = []
    new = tok.encode(text, add_special_tokens=False)
    changes = []
    for i in range(0, len(new), chunk):
        stream.extend(new[i : i + chunk])
        if sm.advance(stream):
            changes.append((len(stream), sm.mode, sm.focus_ids))
    if sm.advance(stream):
        changes.append((len(stream), sm.mode, sm.focus_ids))
    return changes


def test_starts_global_and_returns_to_global_after_every_span(machine):
    tok, _, sm = machine
    assert sm.mode is Mode.GLOBAL
    _feed(tok, sm, '<focus magic_chunks="2">value</focus>')
    assert sm.mode is Mode.GLOBAL
    _feed(tok, sm, "<local>thinking</local>")
    assert sm.mode is Mode.GLOBAL


def test_focus_opens_and_names_the_segment(machine):
    tok, pmap, sm = machine
    _feed(tok, sm, '<global>look at 2</global><focus magic_chunks="2">')
    assert sm.mode is Mode.FOCUS
    assert sm.focus_ids == (2,)
    snap = sm.snapshot()
    span = pmap.by_id(2)
    dense = snap.dense(pmap.num_prompt_tokens)
    assert all(dense[span.token_start : span.token_end])
    assert all(dense[: pmap.sink_end])
    assert all(dense[pmap.local_window_start :])
    other = pmap.by_id(1)
    assert not any(dense[other.token_start : min(other.token_end, pmap.local_window_start)])


def test_answer_is_an_alias_of_local(machine):
    tok, _, sm = machine
    _feed(tok, sm, "<answer>42")
    assert sm.mode is Mode.LOCAL
    assert sm.snapshot().focus_ids == ()


def test_global_tag_causes_no_transition(machine):
    tok, _, sm = machine
    before = sm.snapshot()
    _feed(tok, sm, "<global>navigating</global>")
    assert sm.mode is Mode.GLOBAL
    assert sm.snapshot() == before


def test_focus_opens_only_from_global(machine):
    tok, _, sm = machine
    _feed(tok, sm, "<local>planning")
    assert sm.mode is Mode.LOCAL
    _feed(tok, sm, '<focus magic_chunks="1">')
    assert sm.mode is Mode.LOCAL  # ignored inside local
    assert sm.stats.focus_attempts == 0


@pytest.mark.parametrize(
    "tag,reason",
    [
        ('<focus magic_chunks="">', DeclineReason.EMPTY),
        ('<focus magic_chunks="abc">', DeclineReason.SYNTAX),
        ("<focus>", DeclineReason.SYNTAX),
        ('<focus magic_chunks="99999">', DeclineReason.UNKNOWN_ID),
        ('<focus magic_chunks="1,2,3,4">', DeclineReason.TOO_MANY),
    ],
)
def test_every_gate_is_biased_toward_declining(machine, tag, reason):
    tok, _, sm = machine
    _feed(tok, sm, tag)
    assert sm.mode is Mode.GLOBAL
    assert sm.stats.declines == [reason.value]
    assert sm.snapshot().is_full


def test_multiple_ids_parse_on_commas_and_whitespace(machine):
    tok, _, sm = machine
    _feed(tok, sm, '<focus magic_chunks="1, 2 3">')
    assert sm.mode is Mode.FOCUS
    assert sm.focus_ids == (1, 2, 3)


def test_focus_is_declined_when_no_segments_were_detected(config, family_case, filler):
    hub_id, tok, _ = family_case
    pmap = build_prompt_map(tok, "no turns here", get_model(hub_id).family, config)
    sm = DAStateMachine(pmap, tok, config)
    _feed(tok, sm, '<focus magic_chunks="1">')
    assert sm.mode is Mode.GLOBAL
    assert sm.stats.declines == [DeclineReason.NO_SEGMENTS.value]


def test_placeholder_tokens_stop_the_walk_and_are_revisited(machine):
    tok, _, sm = machine
    ids = tok.encode('<focus magic_chunks="1">', add_special_tokens=False)
    sm.advance(ids[:-1] + [-1])
    assert sm.num_consumed == len(ids) - 1
    assert sm.mode is Mode.GLOBAL
    sm.advance(ids)  # the placeholder resolved
    assert sm.mode is Mode.FOCUS
    assert sm.num_consumed == len(ids)


def test_everything_past_the_boundary_is_attended_unconditionally(machine):
    _, pmap, sm = machine
    snap = build_mask(pmap, Mode.LOCAL)
    dense = snap.dense(pmap.num_prompt_tokens + 500)
    assert all(dense[pmap.num_prompt_tokens :])
    assert snap.boundary == pmap.num_prompt_tokens


def test_local_keeps_only_the_scaffold(machine):
    _, pmap, _ = machine
    local = build_mask(pmap, Mode.LOCAL)
    focus = build_mask(pmap, Mode.FOCUS, (1,))
    assert local.kept_prompt_tokens < focus.kept_prompt_tokens < pmap.num_prompt_tokens
    span = pmap.by_id(1)
    assert focus.kept_prompt_tokens - local.kept_prompt_tokens == (
        span.token_end - span.token_start
    )


def test_spans_and_dense_mask_cannot_drift(machine):
    _, pmap, _ = machine
    snap = build_mask(pmap, Mode.FOCUS, (1, 3))
    dense = snap.dense(pmap.num_prompt_tokens)
    from_spans = [False] * pmap.num_prompt_tokens
    for s, e in snap.spans:
        for i in range(s, e):
            from_spans[i] = True
    assert dense == from_spans


@pytest.mark.parametrize("block_size", [1, 2, 16, 32])
def test_block_alignment_equals_or_reducing_the_dense_mask(machine, block_size):
    _, pmap, _ = machine
    snap = build_mask(pmap, Mode.FOCUS, (2,))
    n = pmap.num_prompt_tokens
    dense = snap.dense(n)
    aligned = align_spans(snap.spans, block_size)
    from_aligned = [False] * n
    for s, e in aligned:
        for i in range(max(0, s), min(n, e)):
            from_aligned[i] = True
    for b in range(0, n // block_size):
        lo, hi = b * block_size, (b + 1) * block_size
        assert any(dense[lo:hi]) == any(from_aligned[lo:hi])
        if any(dense[lo:hi]):
            assert all(from_aligned[lo:hi])


def test_alignment_costs_at_most_one_block_per_edge(machine):
    _, pmap, _ = machine
    snap = build_mask(pmap, Mode.FOCUS, (2,))
    for block_size in (16, 32):
        aligned = align_spans(snap.spans, block_size)
        extra = sum(e - s for s, e in aligned) - snap.kept_prompt_tokens
        assert 0 <= extra <= 2 * block_size * len(snap.spans)


def test_tags_split_across_tokens_still_fire(machine):
    tok, _, sm = machine
    _feed(tok, sm, '<focus magic_chunks="2">', chunk=1)
    assert sm.mode is Mode.FOCUS


def test_close_and_open_arriving_together_apply_in_textual_order(machine):
    tok, _, sm = machine
    _feed(tok, sm, '<focus magic_chunks="1">v')
    assert sm.mode is Mode.FOCUS
    # Both tags land inside one scan: the close must be applied before the open,
    # or the open would be ignored (focus opens only from global).
    _feed(tok, sm, "</focus><local>", chunk=64)
    assert sm.mode is Mode.LOCAL


def test_stats_count_attempts_and_grants(machine):
    tok, _, sm = machine
    _feed(tok, sm, '<focus magic_chunks="1">a</focus>')
    _feed(tok, sm, '<focus magic_chunks="9999">')
    assert sm.stats.focus_attempts == 2
    assert sm.stats.focus_granted == 1
    assert sm.stats.declines == [DeclineReason.UNKNOWN_ID.value]


def test_incremental_detokenizer_holds_back_partial_characters():
    class ByteTok:
        def decode(self, ids):
            return bytes(ids).decode("utf-8", errors="replace")

    d = IncrementalDetokenizer(ByteTok())
    parts = [d.push(b) for b in "é>".encode()]
    assert parts[0] == ""  # first byte of a two-byte character is held
    assert "".join(parts) == "é>"
