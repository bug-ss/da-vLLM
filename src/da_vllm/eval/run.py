"""Arm runner: render, generate, replay, judge, persist.

Prompts are tokenized here and handed to vLLM as token ids, so the engine
serves exactly the string the renderer produced -- no second
``add_special_tokens`` pass, no duplicated BOS, and the replay works from the
same fingerprint the server saw.

Three arms (guide 8.2):

* ``vanilla``    -- plain prompt, full attention;
* ``da_no_mask`` -- the identical magic-chunk prompt, mask off (a separate
  engine with the patch disabled and its own compile cache);
* ``da``         -- magic-chunk prompt, mask on.

Without the no-mask arm the cost of the prompt format cannot be separated from
the cost of the mask, and the wrong conclusion gets drawn.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from ..config import DAConfig
from ..detect import build_prompt_map
from ..masking.logits_processor import DA_ENABLE_KEY, DA_PROMPT_TEXT_KEY
from ..metrics.replay import replay, replay_vanilla
from ..models import resolve
from ..prompt import PromptRenderer, RenderedPrompt
from ..serving import EngineOptions
from .data import Example
from .judge import Verdict, score_response
from .records import ResponseRecord

logger = logging.getLogger(__name__)


@dataclass
class PreparedRequest:
    example: Example
    prompt: RenderedPrompt
    token_ids: list[int]

    @property
    def prompt_tokens(self) -> int:
        return len(self.token_ids)


def prepare_requests(
    examples: Sequence[Example], renderer: PromptRenderer, arm: str
) -> list[PreparedRequest]:
    out: list[PreparedRequest] = []
    for ex in examples:
        prompt = renderer.render(arm, ex.context, ex.question)
        ids = list(renderer.tokenizer.encode(prompt.text, add_special_tokens=False))
        out.append(PreparedRequest(ex, prompt, ids))
    return out


def sampling_extra_args(arm: str, prompt: RenderedPrompt) -> dict[str, Any]:
    """Per-request DA opt-in.  Explicit flag, never inferred (guide 12)."""
    if arm != "da":
        return {}
    return {DA_ENABLE_KEY: True, DA_PROMPT_TEXT_KEY: prompt.text}


def generate(
    llm: Any,
    requests: Sequence[PreparedRequest],
    options: EngineOptions,
) -> list[Any]:
    """Issue one batch of generations through vLLM."""
    from vllm import SamplingParams, TokensPrompt  # type: ignore

    spec = options.spec
    prompts = [TokensPrompt(prompt_token_ids=r.token_ids) for r in requests]
    params = [
        SamplingParams(
            n=1,
            **spec.sampling.to_dict(),
            extra_args=sampling_extra_args(options.arm, r.prompt) or None,
        )
        for r in requests
    ]
    return llm.generate(prompts, params)


def build_records(
    requests: Sequence[PreparedRequest],
    outputs: Sequence[Any],
    *,
    options: EngineOptions,
    renderer: PromptRenderer,
    config: DAConfig,
    run_id: str,
    mask_lag_steps: int = 2,
    block_size: int | None = None,
) -> list[ResponseRecord]:
    """Replay each response to reconstruct attended tokens (guide 8.5)."""
    records: list[ResponseRecord] = []
    for req, out in zip(requests, outputs):
        completion = out.outputs[0]
        text = completion.text
        token_ids = list(completion.token_ids)
        if options.arm == "da":
            prompt_map = build_prompt_map(
                renderer.tokenizer,
                req.prompt.text,
                options.spec.family,
                config,
                prompt_token_ids=req.token_ids,
            )
            result = replay(
                prompt_map,
                renderer.tokenizer,
                text,
                config,
                mask_lag_steps=mask_lag_steps,
                block_size=block_size,
                response_token_ids=token_ids,
            )
        else:
            result = replay_vanilla(
                req.prompt_tokens,
                text,
                renderer.tokenizer,
                response_token_ids=token_ids,
            )
        records.append(
            ResponseRecord(
                run_id=run_id,
                model=options.model,
                arm=options.arm,
                source=req.example.source,
                example_id=req.example.example_id,
                prompt_fingerprint=req.prompt.fingerprint,
                prompt_tokens=req.prompt_tokens,
                response_text=text,
                decode_steps=result.decode_steps,
                attended_tokens=result.attended_tokens,
                block_aligned_attended_tokens=result.block_aligned_attended_tokens,
                finish_reason=getattr(completion, "finish_reason", None),
                focus_attempts=result.focus_attempts,
                focus_granted=result.focus_granted,
                declines=result.declines,
                mode_steps=result.mode_steps,
                answer_text=result.answer_text,
                mask_lag_steps=result.mask_lag_steps,
                meta={"num_segments": req.prompt.num_segments},
            )
        )
    return records


def judge_records(
    records: Sequence[ResponseRecord],
    examples: Sequence[Example],
    call_judge: Callable[[list[dict[str, str]]], tuple[str, str | None]],
) -> list[ResponseRecord]:
    """Score every record with the one fixed judge, storing its identity."""
    by_id = {ex.example_id: ex for ex in examples}
    for rec in records:
        ex = by_id.get(rec.example_id)
        if ex is None or not ex.rubric:
            raise ValueError(
                f"no rubric for {rec.example_id}: rows without a rubric must be "
                "dropped before generation, never judged without one"
            )
        verdict: Verdict = score_response(
            ex.question, rec.answer_text, ex.rubric, call_judge=call_judge
        )
        rec.correct = verdict.correct
        rec.judge_model = verdict.judge_model
        rec.judge_parsed_by = verdict.parsed_by
        rec.judge_truncated = verdict.truncated
    return list(records)


def new_run_id(model: str, arm: str) -> str:
    return f"{resolve(model).hub_id.replace('/', '_')}-{arm}-{uuid.uuid4().hex[:8]}"
