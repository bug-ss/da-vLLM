"""The rest of the validation checklist (guide 9).

This harness is not optional.  The mask silently did nothing for weeks in the
original work, and every check here exists because one specific failure got
past everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean
from typing import Any, Callable, Sequence

from ..config import DAConfig
from ..detect import build_prompt_map
from ..models import ModelSpec, resolve
from ..prompt import PromptRenderer, RenderedPrompt
from ..segmenter import assert_lossless
from ..state_machine import DAStateMachine, Mode


class ValidationError(AssertionError):
    pass


# -- 9.5 serve and replay render the same prompt --------------------------


def assert_prompt_parity(served: RenderedPrompt, replayed: RenderedPrompt) -> None:
    """Token for token, by fingerprint.

    Serve and replay once differed by the tool-declaration system block (about
    340 tokens on Qwen), so the reported metrics described a prompt the model
    never saw.
    """
    if served.fingerprint != replayed.fingerprint:
        raise ValidationError(
            "serve and replay rendered different prompts: "
            f"{served.fingerprint[:12]} vs {replayed.fingerprint[:12]}. "
            "Both paths must call the same PromptRenderer."
        )


# -- 9.6 round trip per family --------------------------------------------


@dataclass
class RoundTripCase:
    name: str
    context: str
    question: str = "What is the answer?"
    expect_focus: bool = True


@dataclass
class RoundTripResult:
    name: str
    num_segments: int
    detected_segments: int
    failure_reason: str | None
    focus_available: bool
    focus_parsed: tuple[int, ...] | None
    lossless: bool
    ok: bool
    notes: list[str] = field(default_factory=list)


#: Adversarial contexts.  The header and divider cases matter because inline
#: dividers collide with document content, and a document that contains the
#: turn literal must not be able to forge a segment boundary.
def default_cases(filler: str) -> list[RoundTripCase]:
    return [
        RoundTripCase("plain", filler),
        RoundTripCase("empty", "", expect_focus=True),
        RoundTripCase("whitespace_only", "   \n\n\t  ", expect_focus=True),
        RoundTripCase("short", "One short line."),
        RoundTripCase(
            "contains_question_header",
            "# Question\nWhat colour is the sky?\n\n" + filler,
        ),
        RoundTripCase(
            "contains_magic_chunk_header", "Magic Chunk 7\nnot a real chunk\n" + filler
        ),
        RoundTripCase("contains_turn_literal", "<|im_start|>user\n<start_of_turn>model\n" + filler),
        RoundTripCase("contains_da_tags", "<focus magic_chunks=\"99\">x</focus>\n" + filler),
        RoundTripCase("no_whitespace_run", "A" * 40000),
    ]


def round_trip(
    tokenizer,
    model: str | ModelSpec,
    cases: Sequence[RoundTripCase],
    config: DAConfig | None = None,
    *,
    focus_id: int = 1,
    renderer: PromptRenderer | None = None,
) -> list[RoundTripResult]:
    """Render, detect, parse -- for every adversarial context."""
    config = config or DAConfig(enabled=True)
    spec = resolve(model)
    renderer = renderer or PromptRenderer(tokenizer, spec)

    results: list[RoundTripResult] = []
    for case in cases:
        notes: list[str] = []
        prompt = renderer.render_da(case.context, case.question)
        lossless = True
        if case.context.strip():
            try:
                assert_lossless(case.context, prompt.segments)
            except AssertionError as exc:
                lossless = False
                notes.append(str(exc))

        pmap = build_prompt_map(tokenizer, prompt.text, spec.family, config)
        parsed: tuple[int, ...] | None = None
        if pmap.focus_available:
            sm = DAStateMachine(pmap, tokenizer, config)
            tag = f'<focus magic_chunks="{focus_id}">'
            sm.advance(list(tokenizer.encode(tag, add_special_tokens=False)))
            parsed = sm.focus_ids if sm.mode is Mode.FOCUS else ()
            if not parsed:
                notes.append("focus tag did not open")

        if pmap.num_prompt_tokens and pmap.local_window_start >= pmap.num_prompt_tokens:
            notes.append("local window is empty")
        if pmap.local_window_is_fallback:
            notes.append("local window fell back to the last-N-tokens rule")

        detected = len(pmap.segments)
        ok = (
            lossless
            and not notes
            and detected == prompt.num_segments
            and (pmap.focus_available == case.expect_focus)
            and (not case.expect_focus or parsed == (focus_id,))
        )
        results.append(
            RoundTripResult(
                name=case.name,
                num_segments=prompt.num_segments,
                detected_segments=detected,
                failure_reason=pmap.failure_reason,
                focus_available=pmap.focus_available,
                focus_parsed=parsed,
                lossless=lossless,
                ok=ok,
                notes=notes,
            )
        )
    return results


def assert_round_trip(results: Sequence[RoundTripResult]) -> None:
    bad = [r for r in results if not r.ok]
    if bad:
        raise ValidationError(
            "round trip failed for: "
            + "; ".join(
                f"{r.name} (segments {r.num_segments}->{r.detected_segments}, "
                f"reason={r.failure_reason}, notes={r.notes})"
                for r in bad
            )
        )


# -- 9.3 realized kept fraction -------------------------------------------


@dataclass(frozen=True)
class KeptFractionReport:
    token_level: float
    block_level: float

    @property
    def agree(self) -> bool:
        # The kernel reads at block granularity, so the block figure is the
        # larger of the two by a few percent.  A block figure *below* the token
        # figure means the two are measuring different things.
        return self.block_level + 1e-6 >= self.token_level


def kept_fraction_report(
    *, attended_tokens: int, block_attended_tokens: int, full_attended_tokens: int
) -> KeptFractionReport:
    if full_attended_tokens <= 0:
        raise ValueError("full_attended_tokens must be positive")
    return KeptFractionReport(
        token_level=attended_tokens / full_attended_tokens,
        block_level=block_attended_tokens / full_attended_tokens,
    )


# -- 9.4 the kernel honours the mask --------------------------------------


@dataclass(frozen=True)
class KernelScalingReport:
    kept_fractions: tuple[float, ...]
    times_s: tuple[float, ...]
    efficiency: float

    @property
    def tracks_mask(self) -> bool:
        """The Triton kernel tracked the kept fraction at 91 to 96%."""
        return 0.80 <= self.efficiency <= 1.15


def kernel_scaling(
    run_kernel: Callable[[float], float], kept_fractions: Sequence[float], *, repeats: int = 5
) -> KernelScalingReport:
    """Replay captured kernel arguments at several kept fractions.

    ``run_kernel(fraction) -> seconds``.  Efficiency is the measured speedup
    divided by the ideal one; capture must run in **eager mode**, because a
    Python monkeypatch on a kernel is bypassed inside a replayed CUDA graph.
    """
    fractions = tuple(kept_fractions)
    if len(fractions) < 2:
        raise ValueError("need at least two kept fractions")
    times = tuple(min(run_kernel(f) for _ in range(repeats)) for f in fractions)
    base_f, base_t = fractions[0], times[0]
    ratios = [
        (base_t / t) / (base_f / f)
        for f, t in zip(fractions[1:], times[1:])
        if t > 0 and f > 0
    ]
    return KernelScalingReport(fractions, times, fmean(ratios) if ratios else 0.0)


# -- 9.7 A/B inside one engine boot ---------------------------------------


class StepToggle:
    """Alternate two variants step by step inside a single engine boot.

    Cross-boot variance (0.3 to 0.5 ms) is larger than the effects being
    measured, so an A/B across two process launches proves nothing.
    """

    def __init__(self, labels: Sequence[str] = ("a", "b")) -> None:
        if len(labels) < 2:
            raise ValueError("need at least two labels")
        self.labels = tuple(labels)
        self._i = 0
        self.samples: dict[str, list[float]] = {l: [] for l in self.labels}

    def next_label(self) -> str:
        label = self.labels[self._i % len(self.labels)]
        self._i += 1
        return label

    def record(self, label: str, value: float) -> None:
        self.samples[label].append(value)

    def summary(self) -> dict[str, dict[str, float]]:
        from statistics import median

        return {
            label: {
                "n": len(v),
                "median": median(v) if v else float("nan"),
                "mean": fmean(v) if v else float("nan"),
            }
            for label, v in self.samples.items()
        }


# -- 12: the enable flag is explicit --------------------------------------


def assert_explicit_enable(params: Any) -> None:
    """Guard against re-introducing a shape-keyed "is this vanilla" shortcut."""
    from ..masking.logits_processor import DA_ENABLE_KEY

    extra = getattr(params, "extra_args", None) or {}
    if DA_ENABLE_KEY not in extra:
        raise ValidationError(
            "request has no explicit da_enable flag; DA must never be inferred "
            "from prompt or config shape"
        )
