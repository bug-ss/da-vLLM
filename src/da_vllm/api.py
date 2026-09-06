"""The high-level entry point: ask a question about a long document with DA.

```python
from da_vllm import DAEngine

engine = DAEngine("Qwen/Qwen3.6-27B")          # builds a vLLM engine with the
result = engine.answer(long_document, "When was Acme founded?")   # DA patch on

result.answer            # "2003"
result.attended_tokens   # what the kernel actually read, summed over steps
result.reduction_pct     # against the analytic vanilla baseline
result.mode_trace        # [(step, "focus"), (step, "global"), ...]
```

Everything below the surface is the same machinery the paper's evaluation uses
-- the same renderer, the same state machine, the same replay -- so a number you
get here is the number the evaluation would report.

Three ways to drive it:

* **In-process vLLM** (default): ``DAEngine("Qwen/Qwen3.6-27B")``.
* **A pre-built engine**: ``DAEngine(model, llm=my_llm)`` to share one engine.
* **Any backend**: ``DAEngine(model, generate_fn=...)`` where ``generate_fn``
  takes ``(list_of_token_id_lists, sampling_params_dict)`` and returns
  ``[(text, token_ids, finish_reason), ...]``.  Use this to point DA at a
  remote ``vllm serve`` -- but note the mask only exists where the patch is
  installed, so a remote server must have been started with it (see
  :func:`da_vllm.serving.vllm_serve_command`).  Without the patch you get the
  DA *prompt format* and honest replay accounting, but full attention.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from .config import DAConfig
from .detect import PromptMap, build_prompt_map
from .masking.logits_processor import DA_ENABLE_KEY, DA_PROMPT_TEXT_KEY
from .metrics.replay import ReplayResult, replay, replay_vanilla, vanilla_attended_tokens
from .models import ModelSpec, resolve
from .prompt import PromptRenderer, RenderedPrompt
from .serving import EngineOptions

logger = logging.getLogger(__name__)


def _accepts_prompts(fn: "GenerateFn | None") -> bool:
    """True if ``fn`` wants the rendered prompts as a third argument."""
    if fn is None:
        return False
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins, C callables
        return False
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()):
        return True
    positional = [
        p
        for p in params.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 3

#: ``generate_fn(token_id_lists, sampling_params)`` -> one
#: ``(text, token_ids, finish_reason)`` per prompt.  It may take an optional
#: third argument, ``prompts``, to receive the matching
#: :class:`RenderedPrompt` objects -- a remote backend needs the prompt text to
#: send alongside the ids.
GenerateFn = Callable[..., Sequence[tuple[str, Sequence[int], "str | None"]]]


@dataclass
class DAAnswer:
    """One response, with the accounting that makes DA worth using."""

    text: str
    answer: str | None
    arm: str
    prompt_tokens: int
    decode_steps: int
    attended_tokens: int
    baseline_attended_tokens: int
    num_segments: int
    mode_steps: dict[str, int]
    mode_trace: list[tuple[int, str]]
    focus_attempts: int
    focus_granted: int
    declines: list[str]
    finish_reason: str | None = None
    detection_failure: str | None = None
    block_aligned_attended_tokens: int | None = None
    prompt_fingerprint: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def format_ok(self) -> bool:
        """A response with no parseable ``<answer>`` tag counts as a failure."""
        return self.answer is not None

    @property
    def non_terminating(self) -> bool:
        from .metrics.replay import NON_TERMINATING_STEPS

        return self.decode_steps >= NON_TERMINATING_STEPS

    @property
    def reduction_pct(self) -> float:
        """Percent change in attended tokens against the vanilla baseline."""
        if not self.baseline_attended_tokens:
            return 0.0
        delta = self.attended_tokens - self.baseline_attended_tokens
        return delta / self.baseline_attended_tokens * 100.0

    def summary(self) -> str:
        return (
            f"{self.arm}: {self.decode_steps} steps, "
            f"{self.attended_tokens / 1e6:.2f}M attended tokens "
            f"({self.reduction_pct:+.1f}% vs vanilla), "
            f"{self.focus_granted}/{self.focus_attempts} focus granted"
        )


class DAEngine:
    """A DA-enabled engine plus the renderer and replay that go with it."""

    def __init__(
        self,
        model: str | ModelSpec,
        *,
        arm: str = "da",
        config: DAConfig | None = None,
        tokenizer: Any = None,
        llm: Any = None,
        generate_fn: GenerateFn | None = None,
        engine_options: EngineOptions | None = None,
        max_tokens: int | None = None,
        **engine_kwargs: Any,
    ) -> None:
        self.spec: ModelSpec = resolve(model)
        self.arm = arm
        if config is None:
            config = DAConfig(
                enabled=(arm == "da"),
                max_model_len=self.spec.max_model_len,
            )
        elif config.enabled != (arm == "da"):
            raise ValueError(
                f"arm {arm!r} and DAConfig(enabled={config.enabled}) disagree; "
                "the mask is on for the 'da' arm and off for every other"
            )
        self.config = config
        self.max_tokens = max_tokens

        self.tokenizer = tokenizer if tokenizer is not None else self._load_tokenizer()
        self.renderer = PromptRenderer(self.tokenizer, self.spec, config=config)

        self.options = engine_options or EngineOptions(
            model=self.spec.hub_id,
            arm=arm,
            da_config=config,
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            extra_engine_kwargs=engine_kwargs,
        )
        self._llm = llm
        self._generate_fn = generate_fn
        self._generate_fn_takes_prompts = _accepts_prompts(generate_fn)
        self._prompt_map_cache: dict[str, PromptMap] = {}

    # -- construction helpers ---------------------------------------------

    def _load_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(self.spec.hub_id)

    @property
    def llm(self):
        """The vLLM engine, built on first use."""
        if self._llm is None and self._generate_fn is None:
            from .serving import build_llm

            self._llm = build_llm(self.options)
        return self._llm

    def sampling_params(self, **overrides) -> dict[str, Any]:
        params = self.spec.sampling.to_dict()
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        params.update(overrides)
        return params

    # -- rendering ---------------------------------------------------------

    def render(self, context: str, question: str) -> RenderedPrompt:
        """Render through the one renderer.  Useful on its own for inspection."""
        return self.renderer.render(self.arm, context, question)

    def prompt_map(self, prompt: RenderedPrompt, token_ids: Sequence[int]) -> PromptMap:
        cached = self._prompt_map_cache.get(prompt.fingerprint)
        if cached is None:
            cached = build_prompt_map(
                self.tokenizer,
                prompt.text,
                self.spec.family,
                self.config,
                prompt_token_ids=token_ids,
            )
            self._prompt_map_cache[prompt.fingerprint] = cached
        return cached

    # -- generation --------------------------------------------------------

    def _generate(
        self, token_id_lists: Sequence[Sequence[int]], prompts: Sequence[RenderedPrompt]
    ) -> list[tuple[str, list[int], str | None]]:
        params = self.sampling_params()
        if self._generate_fn is not None:
            out = (
                self._generate_fn(token_id_lists, params, prompts)
                if self._generate_fn_takes_prompts
                else self._generate_fn(token_id_lists, params)
            )
            return [(t, list(ids), fr) for t, ids, fr in out]

        from vllm import SamplingParams  # type: ignore

        try:
            from vllm import TokensPrompt  # type: ignore
        except ImportError:  # pragma: no cover - moved between vLLM versions
            from vllm.inputs import TokensPrompt  # type: ignore

        sampling = []
        for prompt in prompts:
            extra = (
                {DA_ENABLE_KEY: True, DA_PROMPT_TEXT_KEY: prompt.text}
                if self.arm == "da"
                else None
            )
            try:
                sampling.append(SamplingParams(n=1, extra_args=extra, **params))
            except TypeError:  # pragma: no cover - older vLLM has no extra_args
                if extra is not None:
                    raise RuntimeError(
                        "this vLLM build's SamplingParams has no extra_args, so a "
                        "request cannot opt into DA. Upgrade vLLM, or drive the "
                        "engine through generate_fn."
                    ) from None
                sampling.append(SamplingParams(n=1, **params))

        outputs = self.llm.generate(
            [TokensPrompt(prompt_token_ids=list(ids)) for ids in token_id_lists],
            sampling,
        )
        result = []
        for out in outputs:
            completion = out.outputs[0]
            result.append(
                (
                    completion.text,
                    list(completion.token_ids),
                    getattr(completion, "finish_reason", None),
                )
            )
        return result

    # -- the public call ---------------------------------------------------

    def answer(self, context: str, question: str, **kwargs) -> DAAnswer:
        """Answer one question about one document."""
        return self.answer_batch([(context, question)], **kwargs)[0]

    def answer_batch(
        self,
        items: Iterable[tuple[str, str]],
        *,
        mask_lag_steps: int = 2,
        block_size: int | None = None,
    ) -> list[DAAnswer]:
        """Answer a batch.  One engine call, so the batch actually batches."""
        pairs = list(items)
        prompts = [self.render(ctx, q) for ctx, q in pairs]
        token_ids = [
            list(self.tokenizer.encode(p.text, add_special_tokens=False)) for p in prompts
        ]
        reserve = self.sampling_params()["max_tokens"]
        limit = self.config.max_model_len - reserve
        for ids in token_ids:
            if len(ids) > limit:
                raise ValueError(
                    f"prompt is {len(ids)} tokens; only {limit} fit alongside "
                    f"{reserve} reserved for generation in a "
                    f"{self.config.max_model_len}-token window. Segment the "
                    "document further, lower max_tokens, or drop the example -- "
                    "over-length inputs are dropped, never truncated."
                )

        generated = self._generate(token_ids, prompts)

        answers: list[DAAnswer] = []
        for (ctx, question), prompt, ids, (text, out_ids, finish) in zip(
            pairs, prompts, token_ids, generated
        ):
            answers.append(
                self._account(
                    prompt, ids, text, out_ids, finish, mask_lag_steps, block_size
                )
            )
        return answers

    def _account(
        self,
        prompt: RenderedPrompt,
        prompt_ids: Sequence[int],
        text: str,
        out_ids: Sequence[int],
        finish_reason: str | None,
        mask_lag_steps: int,
        block_size: int | None,
    ) -> DAAnswer:
        detection_failure = None
        if self.arm in ("da", "da_no_mask"):
            pmap = self.prompt_map(prompt, prompt_ids)
            detection_failure = pmap.failure_reason
            if self.arm == "da":
                result: ReplayResult = replay(
                    pmap,
                    self.tokenizer,
                    text,
                    self.config,
                    mask_lag_steps=mask_lag_steps,
                    block_size=block_size,
                    response_token_ids=out_ids,
                )
            else:
                # Same prompt, full attention: the ablation that separates the
                # cost of the format from the cost of the mask.
                result = replay_vanilla(
                    len(prompt_ids), text, self.tokenizer, response_token_ids=out_ids
                )
        else:
            result = replay_vanilla(
                len(prompt_ids), text, self.tokenizer, response_token_ids=out_ids
            )

        return DAAnswer(
            text=text,
            answer=result.answer_text,
            arm=self.arm,
            prompt_tokens=len(prompt_ids),
            decode_steps=result.decode_steps,
            attended_tokens=result.attended_tokens,
            baseline_attended_tokens=vanilla_attended_tokens(
                len(prompt_ids), result.decode_steps
            ),
            num_segments=prompt.num_segments,
            mode_steps=result.mode_steps,
            mode_trace=result.transitions,
            focus_attempts=result.focus_attempts,
            focus_granted=result.focus_granted,
            declines=result.declines,
            finish_reason=finish_reason,
            detection_failure=detection_failure,
            block_aligned_attended_tokens=result.block_aligned_attended_tokens,
            prompt_fingerprint=prompt.fingerprint,
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        """Release the engine.  Orphaned EngineCore children hold VRAM.

        In vLLM 0.20.2 neither ``LLM`` nor ``LLMEngine`` has a ``shutdown``:
        the method lives on the ``EngineCoreClient`` at
        ``llm.llm_engine.engine_core``.  Dropping the reference alone leaves
        the EngineCore subprocess to be reaped by ``__del__``, which is exactly
        how it ends up reparented to PID 1 still holding VRAM.
        """
        llm, self._llm = self._llm, None
        if llm is None:
            return
        engine = getattr(llm, "llm_engine", None)
        targets = (
            getattr(engine, "engine_core", None),  # EngineCoreClient (0.20.x)
            engine,
            llm,
        )
        for target in targets:
            shutdown = getattr(target, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:  # pragma: no cover
                    logger.exception(
                        "da: engine shutdown raised; check for leaked VRAM"
                    )
                return
        logger.warning(
            "da: found no shutdown() on the engine; the EngineCore subprocess "
            "may outlive this process and hold VRAM. Run each arm in its own "
            "process group (see da_vllm.serving.engine_process)."
        )

    def __enter__(self) -> "DAEngine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
