from da_vllm.config import RunawayConfig
from da_vllm.runaway import RunawayDetector


def test_off_by_default_does_nothing():
    d = RunawayDetector(RunawayConfig())
    d.observe(list(range(100)) * 100, "x " * 5000)
    assert not d.triggered


def test_exact_repetition_fires_after_the_sustain_window():
    cfg = RunawayConfig(enabled=True, token_sustain=2700)
    d = RunawayDetector(cfg, no_answer_token_budget=10**9)
    d.observe([7, 8] * 1200)  # 2400 tokens: below the sustain
    assert not d.triggered
    d.observe([7, 8] * 400)
    assert d.triggered_by == "token_distinct_ratio"


def test_a_long_legitimate_low_novelty_run_passes():
    # The longest legitimate low-novelty run observed was 2,349 tokens; the
    # 2,700 sustain is calibrated so it passes.
    cfg = RunawayConfig(enabled=True)
    d = RunawayDetector(cfg, no_answer_token_budget=10**9)
    d.observe([1, 2] * 1174)  # 2,348 tokens
    assert not d.triggered


def test_varied_wording_indecision_is_caught_by_the_word_signal():
    cfg = RunawayConfig(enabled=True, token_distinct_ratio=0.0, word_sustain=2000)
    d = RunawayDetector(cfg, no_answer_token_budget=10**9)
    text = " ".join(["maybe the answer is this or maybe not"] * 400)
    d.observe([], text)
    assert d.triggered_by == "word_distinct_ratio"


def test_no_answer_tag_after_the_budget_fires():
    cfg = RunawayConfig(enabled=True)
    d = RunawayDetector(cfg, no_answer_token_budget=50)
    d.observe(list(range(51)), "reasoning without ever committing")
    assert d.triggered_by == "no_answer_tag"


def test_seeing_the_answer_tag_disarms_the_budget_signal():
    cfg = RunawayConfig(enabled=True)
    d = RunawayDetector(cfg, no_answer_token_budget=10)
    d.observe([1, 2, 3], "done </answer>")
    d.observe(list(range(100)), " trailing")
    assert not d.triggered


def test_a_legitimate_enumeration_with_a_repeating_ngram_does_not_false_fire():
    """The failure mode that ruled out n-gram detection.

    Every item repeats the same three-word stem -- an n-gram detector fires
    immediately -- but the lexical novelty stays high, so neither distinct-ratio
    signal trips.
    """
    cfg = RunawayConfig(enabled=True)
    d = RunawayDetector(cfg, no_answer_token_budget=10**9)
    words, ids = [], []
    for i in range(600):
        words.append(f"The entry lists alpha{i} bravo{i} charlie{i} delta{i} echo{i}.")
        ids.extend([10, 11, 12, 1000 + 5 * i, 1001 + 5 * i, 1002 + 5 * i, 1003 + 5 * i])
    d.observe(ids, " ".join(words))
    assert not d.triggered
