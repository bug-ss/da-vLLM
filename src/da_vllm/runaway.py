"""Runaway-generation detector (guide 10).

A small fraction of DA sequences never terminate: exact repetition (mostly
Qwen) or non-converging indecision with varied wording (mostly Gemma, invisible
to n-gram detectors).  One such sequence holds a batch slot for thousands of
steps and drags throughput down.

Rejected fixes, all of which change non-runaway outputs or false-fire on
legitimate enumerations: a global repetition penalty, ignore-EOS, a lower token
cap, n-gram detection.

What works is the OR of three signals below -- zero false positives on 6,444
stored responses.  The intervention is to overwrite that row's logits to force
EOS, which changes the argmax: a logits processor that does this **must**
return False from ``is_argmax_invariant()`` or vLLM skips ``apply()`` on the
greedy path.

The paper campaign ran with this off; it is off by default here too.
"""

from __future__ import annotations

import re
from collections import deque

from .config import RunawayConfig

_WORD_RE = re.compile(r"\S+")
_ANSWER_CLOSE = "</answer>"


class RunawayDetector:
    """One per request.  Streaming, O(1) per token."""

    def __init__(self, config: RunawayConfig, *, no_answer_token_budget: int | None = None):
        self.config = config
        self.no_answer_token_budget = (
            config.no_answer_token_budget
            if no_answer_token_budget is None
            else no_answer_token_budget
        )
        self._tokens: deque[int] = deque(maxlen=config.token_window)
        self._words: deque[str] = deque(maxlen=config.word_window)
        self._token_run = 0
        self._word_run = 0
        self._num_tokens = 0
        self._num_words = 0
        self._saw_answer_close = False
        self._tail = ""
        self.triggered_by: str | None = None

    @property
    def triggered(self) -> bool:
        return self.triggered_by is not None

    def observe(self, token_ids, text: str = "") -> bool:
        """Feed newly generated tokens and their decoded text."""
        if self.triggered or not self.config.enabled:
            return self.triggered

        for tid in token_ids:
            self._num_tokens += 1
            self._tokens.append(int(tid))
            if len(self._tokens) == self._tokens.maxlen:
                ratio = len(set(self._tokens)) / len(self._tokens)
                self._token_run = self._token_run + 1 if ratio <= self.config.token_distinct_ratio else 0
                if self._token_run >= self.config.token_sustain:
                    self.triggered_by = "token_distinct_ratio"
                    return True

        if text:
            self._tail = (self._tail + text)[-64:]
            if _ANSWER_CLOSE in self._tail:
                self._saw_answer_close = True
            for word in _WORD_RE.findall(text):
                self._num_words += 1
                self._words.append(word.lower())
                if len(self._words) == self._words.maxlen:
                    ratio = len(set(self._words)) / len(self._words)
                    self._word_run = self._word_run + 1 if ratio <= self.config.word_distinct_ratio else 0
                    if self._word_run >= self.config.word_sustain:
                        self.triggered_by = "word_distinct_ratio"
                        return True

        if not self._saw_answer_close and self._num_tokens > self.no_answer_token_budget:
            self.triggered_by = "no_answer_tag"
            return True
        return False
