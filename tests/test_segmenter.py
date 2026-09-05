import pytest

from da_vllm.segmenter import Segmenter, assert_lossless
from da_vllm.testing import lorem, qwen_tokenizer


@pytest.fixture
def seg():
    return Segmenter(qwen_tokenizer(), target_tokens=200, max_tokens=250)


def test_segments_are_a_lossless_partition(seg):
    text = lorem(60, seed=3)
    segments = seg.segment(text)
    assert len(segments) > 5
    assert_lossless(text, segments)
    assert [s.index for s in segments] == list(range(1, len(segments) + 1))


def test_segments_are_substrings_never_decoded_token_slices(seg):
    # Multibyte-heavy text: a token-id-slice segmenter produced U+FFFD in about
    # a fifth of samples here.
    text = "\n\n".join("日本語のテキストです。" * 40 + " emoji 🚀🚀🚀" for _ in range(20))
    segments = seg.segment(text)
    assert_lossless(text, segments)
    for s in segments:
        assert text[s.start : s.end] == s.text
        assert "�" not in s.text


def test_whitespace_free_run_is_atomic(seg):
    blob = "A" * 50_000
    segments = seg.segment(blob)
    assert len(segments) == 1
    assert segments[0].num_tokens > seg.max_tokens
    assert_lossless(blob, segments)


def test_empty_and_whitespace_contexts_still_yield_one_segment(seg):
    for text in ("", "   ", "\n\n\t"):
        segments = seg.segment(text)
        assert len(segments) == 1
        assert segments[0].text == "<empty_context>"


def test_short_context_is_a_single_segment(seg):
    segments = seg.segment("One short sentence.")
    assert len(segments) == 1


def test_splits_prefer_the_coarsest_boundary_that_fits():
    tok = qwen_tokenizer()
    s = Segmenter(tok, target_tokens=40, max_tokens=50)
    para = " ".join(["word"] * 30)
    text = f"{para}\n\n{para}\n\n{para}"
    segments = s.segment(text)
    assert len(segments) >= 2
    # A paragraph break exists, so no segment should start mid-sentence.
    for seg_ in segments[1:]:
        assert seg_.text.startswith(("word", "\n"))
    assert_lossless(text, segments)


def test_oversized_unit_descends_to_finer_boundaries():
    tok = qwen_tokenizer()
    s = Segmenter(tok, target_tokens=20, max_tokens=25)
    # One paragraph, many sentences: must be cut at sentence ends.
    text = " ".join(f"Sentence number {i} here." for i in range(40))
    segments = s.segment(text)
    assert len(segments) > 1
    assert all(x.num_tokens <= 25 for x in segments)
    assert_lossless(text, segments)


def test_small_pieces_are_packed_toward_the_target():
    tok = qwen_tokenizer()
    s = Segmenter(tok, target_tokens=100, max_tokens=130)
    # Many tiny paragraphs: each is well under the cap, so the only thing that
    # can produce reasonably sized segments is greedy packing.
    text = "\n\n".join("Tiny paragraph." for _ in range(400))
    segments = s.segment(text)
    assert all(x.num_tokens <= 100 for x in segments)
    assert min(x.num_tokens for x in segments[:-1]) > 90
    assert_lossless(text, segments)


def test_no_segment_exceeds_the_hard_cap_unless_atomic():
    tok = qwen_tokenizer()
    s = Segmenter(tok, target_tokens=100, max_tokens=130)
    segments = s.segment(lorem(40, seed=11))
    assert all(x.num_tokens <= 130 for x in segments)


def test_boundaries_are_not_block_aligned():
    # Block alignment belongs to the mask, not the segmenter (guide 4.1).
    tok = qwen_tokenizer()
    s = Segmenter(tok, target_tokens=100, max_tokens=130)
    segments = s.segment(lorem(30, seed=5))
    assert any(x.num_tokens % 16 for x in segments)
