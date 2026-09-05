"""Offline stand-ins: a deterministic tokenizer with real chat templates.

Everything except the vLLM hook itself -- segmentation, rendering, detection,
the state machine, mask construction, the remap, the replay -- can be exercised
with these, so you can verify an install, dry-run a change, or write a test
without downloading a checkpoint or touching a GPU.

The chat templates are *structurally* faithful to the two families the paper
serves: Qwen opens a new turn per tool call, Gemma collapses consecutive tool
calls into a single model turn.  The detector's strict-alternation logic
depends on exactly that difference.

**Not a substitute for the real tokenizer.**  Token boundaries, special-token
ids and the exact template text all differ.  Run ``da validate --model <model>``
against the tokenizer you actually serve before trusting a segment map.
"""

from __future__ import annotations

import json
import re



_TOKEN_RE = re.compile(r"\s*[A-Za-z]+|\s*\d+|\s*[^\sA-Za-z\d]|\s+")
_MAX_PIECE = 4  # split long runs, so token counts resemble a real BPE


QWEN_TEMPLATE = """
{%- if tools %}
{{- '<|im_start|>system\n' }}
{%- if messages[0].role == 'system' %}{{- messages[0].content + '\n\n' }}{%- endif %}
{{- '# Tools\n\n<tools>\n' }}
{%- for tool in tools %}{{- tool | tojson }}{{- '\n' }}{%- endfor %}
{{- '</tools><|im_end|>\n' }}
{%- set loop_messages = messages[1:] if messages[0].role == 'system' else messages %}
{%- else %}
{%- set loop_messages = messages %}
{%- endif %}
{%- for message in loop_messages %}
{%- if message.role == 'tool' %}
{{- '<|im_start|>user\n<tool_response>\n' + message.content + '\n</tool_response><|im_end|>\n' }}
{%- elif message.role == 'assistant' and message.tool_calls %}
{{- '<|im_start|>assistant\n' + (message.content or '') }}
{%- for tc in message.tool_calls %}
{{- '<tool_call>\n{"name": "' + tc.function.name + '", "arguments": ' + (tc.function.arguments | tojson) + '}\n</tool_call>' }}
{%- endfor %}
{{- '<|im_end|>\n' }}
{%- else %}
{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}
{%- if enable_thinking is defined and not enable_thinking %}{{- '<think>\n\n</think>\n\n' }}{%- endif %}
{%- endif %}
"""

GEMMA_TEMPLATE = """
{%- if messages[0].role == 'system' %}
{{- '<start_of_turn>user\n' + messages[0].content + '\n\n' }}
{%- if tools %}{{- '# Tools\n' }}{%- for tool in tools %}{{- tool | tojson }}{{- '\n' }}{%- endfor %}{%- endif %}
{{- '<end_of_turn>\n' }}
{%- set loop_messages = messages[1:] %}
{%- else %}
{%- set loop_messages = messages %}
{%- endif %}
{%- for message in loop_messages %}
{%- if message.role == 'tool' %}
{{- '<start_of_turn>user\n<tool_response>\n' + message.content + '\n</tool_response><end_of_turn>\n' }}
{%- elif message.role == 'assistant' and message.tool_calls %}
{{- '<start_of_turn>model\n' + (message.content or '') }}
{%- for tc in message.tool_calls %}
{{- '<tool_call>\n{"name": "' + tc.function.name + '", "arguments": ' + (tc.function.arguments | tojson) + '}\n</tool_call>' }}
{%- endfor %}
{{- '<end_of_turn>\n' }}
{%- else %}
{{- '<start_of_turn>' + ('model' if message.role == 'assistant' else 'user') + '\n' + message.content + '<end_of_turn>\n' }}
{%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}{{- '<start_of_turn>model\n' }}{%- endif %}
"""


class SimpleTokenizer:
    """Reversible, offset-exact, vocabulary-on-demand.

    Implements the slice of the HuggingFace tokenizer API this package uses:
    ``__call__(text, return_offsets_mapping=...)``, ``encode``, ``decode``,
    ``convert_ids_to_tokens`` and ``apply_chat_template``.
    """

    def __init__(self, template: str = QWEN_TEMPLATE) -> None:
        try:
            from jinja2 import Environment
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "da_vllm.testing needs jinja2: pip install 'da-vllm[dev]'"
            ) from exc
        self._vocab: dict[str, int] = {}
        self._inv: list[str] = []
        self._env = Environment()
        # transformers installs a plain json.dumps as `tojson`; Jinja's default
        # filter HTML-escapes < > & ' and would not match a real rendering.
        self._env.filters["tojson"] = lambda v, **kw: json.dumps(v, ensure_ascii=False)
        self._template = self._env.from_string(template)
        self.eos_token_id = self._id("<|endoftext|>")

    # -- vocabulary --------------------------------------------------------

    def _id(self, piece: str) -> int:
        i = self._vocab.get(piece)
        if i is None:
            i = len(self._inv)
            self._vocab[piece] = i
            self._inv.append(piece)
        return i

    # -- tokenization ------------------------------------------------------

    def _pieces(self, text: str) -> list[tuple[str, int, int]]:
        out: list[tuple[str, int, int]] = []
        for m in _TOKEN_RE.finditer(text):
            s, e = m.span()
            while s < e:
                cut = min(s + _MAX_PIECE, e)
                out.append((text[s:cut], s, cut))
                s = cut
        return out

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True, **kw):
        pieces = self._pieces(text)
        out = {
            "input_ids": [self._id(p) for p, _, _ in pieces],
            "attention_mask": [1] * len(pieces),
        }
        if return_offsets_mapping:
            out["offset_mapping"] = [(s, e) for _, s, e in pieces]
        return out

    def encode(self, text, add_special_tokens=False, **kw):
        return self(text, add_special_tokens=add_special_tokens)["input_ids"]

    def decode(self, ids, **kw):
        return "".join(self._inv[i] for i in ids)

    def convert_ids_to_tokens(self, ids):
        return [self._inv[i] for i in ids]

    # -- chat template -----------------------------------------------------

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=False, tokenize=False, **kwargs
    ):
        rendered = self._template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        # Jinja's whitespace control in the fake templates leaves stray newlines
        # between control blocks; strip them the way the real templates do.
        rendered = re.sub(r"\n(?=<\|im_start\|>|<start_of_turn>|$)", "\n", rendered)
        rendered = rendered.strip("\n")
        if tokenize:
            return self.encode(rendered)
        return rendered


#: Backwards-compatible alias.
FakeTokenizer = SimpleTokenizer


def qwen_tokenizer() -> SimpleTokenizer:
    """A Qwen-3.5/3.6-shaped tokenizer: one turn per tool call."""
    return SimpleTokenizer(QWEN_TEMPLATE)


def gemma_tokenizer() -> SimpleTokenizer:
    """A Gemma-4-shaped tokenizer: consecutive tool calls collapse."""
    return SimpleTokenizer(GEMMA_TEMPLATE)


def lorem(n_paragraphs: int, seed: int = 0) -> str:
    """Deterministic multi-paragraph filler text."""
    import random

    rng = random.Random(seed)
    words = (
        "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
        "lima mike november oscar papa quebec romeo sierra tango uniform"
    ).split()
    paras = []
    for _ in range(n_paragraphs):
        sentences = []
        for _ in range(rng.randint(3, 7)):
            k = rng.randint(6, 16)
            sentences.append(" ".join(rng.choice(words) for _ in range(k)).capitalize() + ".")
        paras.append(" ".join(sentences))
    return "\n\n".join(paras)
