"""Wall-clock timing protocol (guide 11).

The roofline in :mod:`da_vllm.metrics.roofline` is a ceiling, not a
measurement.  What real timing showed, and why it is not the headline:

* At batch 1 and ~37K tokens the workload is prefill-dominated -- seconds of
  prefill against a fraction of a second of decode -- so DA changes total
  latency by 2 to 3% even when decode-only improves by roughly a quarter.
* At matched concurrency a decode step is paced by the **least-pruned resident
  sequence**.  A real batch almost always contains a global-mode sequence, so
  measured step time drops far less than the byte model predicts.  The
  prediction is reached only when every resident sequence is masked.
* DA emits 1.2 to 1.4x more tokens with a long tail, so end-to-end ms per token
  compares arms at different effective batch sizes.  Compare per-step at fixed
  residency, or use forced equal-length decode.
* Measured MBU on these kernels is below 70%: about 57% for FlashAttention and
  11 to 39% for the Triton path at low batch.

The protocol below is the one that held up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import fmean, median

@dataclass(frozen=True)
class StepSample:
    index: int
    seconds: float
    num_prompt_tokens: int
    num_resident: int


@dataclass
class TimingRun:
    """One arm, one engine boot.

    Steady region only: every sequence resident, zero prompt tokens in the
    step, and steps near any prefill boundary dropped.
    """

    target_residency: int
    warmup_steps: int = 20
    prefill_guard_steps: int = 3
    samples: list[StepSample] = field(default_factory=list)

    def record(
        self, index: int, seconds: float, *, num_prompt_tokens: int, num_resident: int
    ) -> None:
        self.samples.append(StepSample(index, seconds, num_prompt_tokens, num_resident))

    def steady(self) -> list[StepSample]:
        prefill_at = {
            s.index for s in self.samples if s.num_prompt_tokens > 0
        }
        guard = self.prefill_guard_steps

        def near_prefill(i: int) -> bool:
            return any(abs(i - p) <= guard for p in prefill_at)

        return [
            s
            for s in self.samples
            if s.index >= self.warmup_steps
            and s.num_prompt_tokens == 0
            and s.num_resident == self.target_residency
            and not near_prefill(s.index)
        ]

    def summary(self) -> dict[str, float]:
        """Medians, with an untrimmed mean alongside so a tail cannot hide."""
        steady = [s.seconds for s in self.steady()]
        if not steady:
            return {"n": 0}
        return {
            "n": len(steady),
            "median_s": median(steady),
            "mean_s": fmean(steady),
            "max_s": max(steady),
            "min_s": min(steady),
        }


def block_aligned_synthetic_mask(
    num_positions: int, block_size: int, keep_fraction: float, *, stride_blocks: int = 1
) -> list[tuple[int, int]]:
    """Synthetic kept spans defined on **whole blocks**.

    A token-strided mask vanishes after outward rounding to block boundaries
    and silently measures full attention.
    """
    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must be in (0, 1]")
    total_blocks = max(1, -(-num_positions // block_size))
    keep_blocks = max(1, round(total_blocks * keep_fraction))
    step = max(1, total_blocks // keep_blocks) * stride_blocks
    spans: list[tuple[int, int]] = []
    for b in range(0, total_blocks, step):
        if len(spans) >= keep_blocks:
            break
        spans.append((b * block_size, min((b + 1) * block_size, num_positions)))
    # The tail block must always be kept: compacting past the block holding the
    # token just written makes the kernel unable to see the current token's own
    # key, and the model loses coherence within a few steps.
    tail = ((total_blocks - 1) * block_size, num_positions)
    if tail not in spans:
        spans.append(tail)
    return spans


def match_arms_on_kept_fraction(
    arms: dict[str, float], *, tolerance: float = 0.02
) -> bool:
    """Arms are only comparable at the same measured block-level kept fraction."""
    values = list(arms.values())
    return bool(values) and (max(values) - min(values)) <= tolerance


TIMING_CHECKLIST: tuple[str, ...] = (
    "prefix caching off",
    "warmup generate before any measurement",
    "steady region only: all sequences resident, zero prompt tokens in the step",
    "drop steps near any prefill boundary",
    "medians, with an untrimmed mean reported alongside",
    "measure kept fraction and time in separate runs (logging it costs a sync)",
    "match arms on measured block-level kept fraction",
    "define synthetic masks on whole blocks",
    "one engine per arm, in its own process group",
)
