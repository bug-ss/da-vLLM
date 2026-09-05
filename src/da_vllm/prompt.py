"""The one renderer (guide 4.2/4.3, paper appendix F).

Every path that renders a prompt -- serving, metrics replay, timing, training --
must go through this module.  Serve and replay once differed by the
tool-declaration system block (about 340 tokens on Qwen), so the reported
metrics described a prompt the model never saw.  :meth:`RenderedPrompt.fingerprint`
exists so that divergence is a test failure rather than a silent one.

Non-negotiables encoded here:

* The tool declaration goes through ``apply_chat_template(..., tools=[...])``.
  A hand-written declaration inlined as system text is wrong for every family
  but the one it was copied from.
* ``arguments`` is a dict, not a JSON string (Qwen 3.5/3.6 templates).
* The system turn is prepended on **every** arm, vanilla included.  It exists
  to occupy the 16-token attention sink so no context token can land in it.
* Thinking is off: models do not follow the DA protocol inside thinking tags.
* "Magic Chunk", not "Chunk" -- "Chunk 3" collides with a document's own
  "Section 3" or "Chapter 3".
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Sequence

from .models import ModelSpec, resolve
from .segmenter import Segment, Segmenter

SYSTEM_TEXT = "You are a helpful assistant."

BOOTSTRAP_USER_TEXT = (
    "A question will follow about a long document. First, retrieve the document "
    'using the `get_magic_chunk` tool, one magic chunk at a time, in order from '
    'magic chunk id "1", until every magic chunk has been retrieved.'
)

#: The marker the local window is anchored on.  If you change this header,
#: change :data:`QUESTION_HEADER` with it (guide 4.3).
QUESTION_HEADER = "# Question"

TOOL_DECLARATION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_magic_chunk",
        "description": (
            "Retrieve one magic chunk of the user's document by its 1-indexed id. "
            "The document is split into consecutive magic chunks, which are "
            "arbitrary retrieval units and do not align with the document's own "
            "sections or chapters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": 'The magic chunk id to retrieve (e.g. "1", "2", ...).',
                }
            },
            "required": ["id"],
        },
    },
}

MAGIC_CHUNK_HEADER = "Magic Chunk {n}"

DA_INSTRUCTION_TEMPLATE = """\
# Question
{question}

# Instructions
Answer the question above using only the retrieved document. Each \
get_magic_chunk response above is one magic chunk of the document, in order \
from magic chunk 1. Magic chunks are arbitrary retrieval splits and do not \
align with the document's own sections or chapters. The document is now fully \
retrieved. Do not call get_magic_chunk again.

Reason through the magic chunks using three modes. Use the <answer> tag to \
output your final answer, e.g., <answer>...</answer>.

| Mode | What you can see | Use it when |
| --- | --- | --- |
| `<global>` (default) | all magic chunks | ...you're identifying which magic chunk to focus on next. |
| `<focus magic_chunks="K">` | only magic chunk K (plus values you've already extracted) | ...you're pulling a verbatim value out of magic chunk K. |
| `<local>` | only values you've already extracted (no magic chunks) | ...you're planning over the question, or synthesizing values you've already extracted. |

## Global mode (default)
Use global mode to identify the next magic chunk to examine. Briefly explain \
why the magic chunk is relevant. Do not reason about the answer itself in \
global mode.

## Focus mode
Switch to focus mode to examine a specific magic chunk and extract facts.

Syntax: `<focus magic_chunks="K">VALUE</focus>` (or `<focus magic_chunks="K,M">`, \
`<focus magic_chunks="K,M,N">` for multiple magic chunks).

VALUE is the answer the magic chunk provides: a name, number, date, or short \
noun phrase (typically 1 to 12 words).

When the document was delivered without any magic chunk retrievals (short \
document, inline only), this tag is unavailable.

## Local mode
In local mode you cannot see the document magic chunks, and you cannot re-read \
them. Local mode is for reasoning that builds on facts you have already \
extracted via earlier <focus> blocks, the question, and your prior reasoning.

Syntax: `<local>your reasoning here</local>`

Refer to verbatim values you extracted in earlier focus blocks by name. You do \
not need to re-state them. If you find you need a value you have not yet \
extracted, close the <local> block, open <global>, and identify the magic chunk \
to focus on next.

## Strategy
Use <global>, <focus ...>, and <local> in any order. Repeat any of them as \
needed, then write <answer>.

Three soft requirements:

1. **Use at least one <focus> block.** The point of focus mode is to pull the \
exact value(s) into your attention before answering. Skipping focus and \
guessing from memory is the most common failure mode.
2. **End with a <local> block that names the single value, phrase, or summary \
you will put inside <answer>.** This is the commitment step. Without it, the \
model often retrieves multiple values and forgets which one the question \
actually asks for.
3. **Don't try to recall magic chunk content in <local>.** Once you open \
<local>, the magic chunks are no longer visible to you. Anything you \
"remember" about a magic chunk you didn't focus on is a guess. If you need a \
value you haven't extracted, close </local> and go back to <global> to pick the \
magic chunk to focus on next.

Examples of valid orders:

- `<local>` -> `<global>` -> `<focus>` -> `<local>` -> `<answer>`. Typical \
shape for most questions: plan, locate, extract, conclude.
- `<global>` -> `<focus>` -> `<local>` -> `<answer>`. Short factual / \
needle-in-haystack: the target magic chunk is obvious from the question.
- `<global>` -> `<focus>` -> `<global>` -> `<focus>` -> `<local>` -> \
`<answer>`. Multi-fact, separate magic chunks.
- `<global>` -> `<focus>` -> `<local>` -> `<global>` -> `<focus>` -> `<local>` \
-> `<answer>`. Extract, sanity-check, fetch what's still missing, commit.

## Answer
{answer_spec}"""

VANILLA_INSTRUCTION_TEMPLATE = """\
# Context
{context}

# Question
{question}

# INSTRUCTIONS
Find the answer to the Question based solely on information in the Context \
above. Reason through the context step by step.

## Answer
{answer_spec}"""

#: Shared by all three arms so the judge scores them identically.
ANSWER_SPEC = """\
End your response with the final answer wrapped in <answer>...</answer>. The \
content depends on the question type:

| Question type | Answer format | Example |
| --- | --- | --- |
| Multiple choice | the letter only | `<answer>D</answer>` |
| Cloze (`<mask-N>`) | the missing word(s) | `<answer>Britney</answer>` |
| Short factual | the value | `<answer>March 14, 2024</answer>` |
| Summary | 2 to 3 sentences | `<answer>Congress passed the BUILD Act in 2018, creating the IDFC.</answer>` |

After </answer>, the response is complete."""


@dataclass(frozen=True)
class RenderedPrompt:
    """A prompt string plus everything downstream needs to reason about it."""

    arm: str  # "da" | "vanilla"
    text: str
    messages: list[dict[str, Any]]
    segments: tuple[Segment, ...]
    model_hub_id: str
    tools: tuple[dict[str, Any], ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        """Stable hash of the exact string handed to the engine.

        Guide 9.5: serve and replay must render the same prompt, token for
        token.  Compare fingerprints, not eyeballs.
        """
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def num_segments(self) -> int:
        return len(self.segments)


class PromptRenderer:
    """The single renderer.  Construct once per (model, tokenizer) pair."""

    def __init__(
        self,
        tokenizer,
        model: str | ModelSpec,
        *,
        model_type: str | None = None,
        segmenter: Segmenter | None = None,
        config=None,
        target_tokens: int = 2048,
        max_tokens: int = 2560,
    ) -> None:
        self.tokenizer = tokenizer
        self.spec: ModelSpec = resolve(model, model_type=model_type)
        if segmenter is not None:
            self.segmenter = segmenter
        elif config is not None:
            # Segment sizes come from the one config, so serving, replay and
            # training cannot disagree about where a magic chunk ends.
            self.segmenter = Segmenter.from_config(tokenizer, config)
        else:
            self.segmenter = Segmenter(
                tokenizer, target_tokens=target_tokens, max_tokens=max_tokens
            )

    # -- message construction ---------------------------------------------

    def _tool_call_message(self, segment_id: int) -> dict[str, Any]:
        arguments: Any = {"id": str(segment_id)}
        if self.spec.family.tool_arguments_as_json_string:
            import json

            arguments = json.dumps(arguments)
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{segment_id}",
                    "type": "function",
                    "function": {"name": "get_magic_chunk", "arguments": arguments},
                }
            ],
        }

    def _tool_response_message(self, segment: Segment) -> dict[str, Any]:
        return {
            "role": "tool",
            "name": "get_magic_chunk",
            "tool_call_id": f"call_{segment.index}",
            "content": f"{MAGIC_CHUNK_HEADER.format(n=segment.index)}\n{segment.text}",
        }

    def da_messages(
        self, context: str, question: str, segments: Sequence[Segment] | None = None
    ) -> tuple[list[dict[str, Any]], tuple[Segment, ...]]:
        segs = tuple(segments) if segments is not None else tuple(
            self.segmenter.segment(context)
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_TEXT},
            {"role": "user", "content": BOOTSTRAP_USER_TEXT},
        ]
        for seg in segs:
            messages.append(self._tool_call_message(seg.index))
            messages.append(self._tool_response_message(seg))
        messages.append(
            {
                "role": "user",
                "content": DA_INSTRUCTION_TEMPLATE.format(
                    question=question, answer_spec=ANSWER_SPEC
                ),
            }
        )
        return messages, segs

    def vanilla_messages(self, context: str, question: str) -> list[dict[str, Any]]:
        return [
            # Present on the vanilla path too: context must never land in the
            # attention sink, and the three arms must share prompt scaffolding.
            {"role": "system", "content": SYSTEM_TEXT},
            {
                "role": "user",
                "content": VANILLA_INSTRUCTION_TEMPLATE.format(
                    context=context, question=question, answer_spec=ANSWER_SPEC
                ),
            },
        ]

    # -- rendering ---------------------------------------------------------

    def _apply(self, messages, tools) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tools=list(tools) if tools else None,
            add_generation_prompt=True,
            tokenize=False,
            **self.spec.family.chat_template_kwargs,
        )

    def render_da(
        self, context: str, question: str, segments: Sequence[Segment] | None = None
    ) -> RenderedPrompt:
        messages, segs = self.da_messages(context, question, segments)
        text = self._apply(messages, [TOOL_DECLARATION])
        return RenderedPrompt(
            arm="da",
            text=text,
            messages=messages,
            segments=segs,
            model_hub_id=self.spec.hub_id,
            tools=(TOOL_DECLARATION,),
        )

    def render_vanilla(self, context: str, question: str) -> RenderedPrompt:
        messages = self.vanilla_messages(context, question)
        # No tools on the vanilla arm: the declaration is part of what the
        # DA-no-mask ablation is measuring.
        text = self._apply(messages, None)
        return RenderedPrompt(
            arm="vanilla",
            text=text,
            messages=messages,
            segments=(),
            model_hub_id=self.spec.hub_id,
        )

    def render(self, arm: str, context: str, question: str) -> RenderedPrompt:
        """``arm`` is "da", "da_no_mask" or "vanilla".

        The DA and DA-no-mask arms render *byte-identical* prompts; they differ
        only in whether the engine has the mask patch installed.  That is the
        whole point of the ablation.
        """
        if arm in ("da", "da_no_mask"):
            return self.render_da(context, question)
        if arm == "vanilla":
            return self.render_vanilla(context, question)
        raise ValueError(f"unknown arm {arm!r}; expected da | da_no_mask | vanilla")
