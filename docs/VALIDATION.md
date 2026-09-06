# Checking that DA is actually working

Read this before you believe any number.

In the original work, **the masking silently did nothing for weeks**. The
server ran, the answers looked fine, and the obvious checks all passed. The
only thing that caught it was check 1 below.

That is the failure mode to worry about here. Not a crash — a system that looks
healthy and is quietly doing nothing.

## Check 1 — is the mask changing the answer at all?

`da_vllm.validation.nll_parity.three_column_parity`

The idea: score the same text three ways and see if the numbers line up the way
they should.

Take one document and one question. Generate an answer twice — once normally,
once with DA. Then score both answers with a separate copy of the model, three
ways:

| | which answer | what it could see |
| --- | --- | --- |
| **v** | the normal one | everything |
| **d** | the DA one | only what DA allowed |
| **dv** | the DA one | everything |

You need **v ≈ d, and both below dv**.

Here is why. `v ≈ d` says DA's answer is about as good as the normal one. And
`d < dv` says the DA answer makes *more* sense under the restricted view than
under the full view — which can only be true if the restriction was real.

**Two columns are not enough.** When the mask was broken, v and d matched
perfectly and everything looked fine. The `dv` column is the one that proves
anything happened.

Two things the scoring copy must get right, or the numbers are meaningless:

- Round the mask out to the same block size the server uses. The GPU reads in
  fixed-size blocks, so a mask that doesn't align to them isn't the same mask.
- On models with mixed layer types (Gemma), give each layer type its own mask.
  A single mask for the whole model is wrong by a factor of ten.

Expect the two systems to disagree by 0.05–0.15 anyway, from normal
floating-point drift. `ParityResult.explain()` tells you which condition failed.

## Check 2 — read the answers

Every time. Low scores with garbage text means the cache is corrupted in a
layer the scoring harness cannot see. This is what it looked like when the
sequence-length tensor was being shared between layer groups.

## Check 3 — does the saving you measure match the saving you claim?

Set `DAConfig(log_kept_fraction=True)` and the server logs how many memory
blocks it actually skipped. Compare against what
`da_vllm.metrics.replay` calculated from the text.

The measured one should be slightly *larger*, because the GPU reads whole
blocks and the calculation counts individual tokens.

**Turn this off before timing anything.** It forces the GPU and CPU to
sync every step, which is exactly what the fast path avoids.

## Check 4 — does reading less actually take less time?

```python
from da_vllm.validation import MetadataCapture, kernel_scaling, write_capture

with MetadataCapture(limit=64) as capture:
    llm.generate(prompts, params)
write_capture("capture.jsonl", capture.steps)

report = kernel_scaling(run_kernel, kept_fractions=[1.0, 0.5, 0.25])
assert report.tracks_mask          # 91-96% on the Triton kernel in the paper
```

Record what the GPU was actually asked to do, replay it offline with smaller
inputs, and check the time drops in proportion.

Capture with CUDA graphs off. A Python hook is skipped inside a replayed graph,
so a captured graph tells you nothing.

## Check 5 — does the server see the prompt the app wrote?

`assert_prompt_parity` compares a hash of the two.

These once differed by about 340 tokens — a block of tool definitions present
on one side and not the other. Every reported number described a prompt the
model never saw.

## Check 6 — round trip, per model family

```bash
da validate --model <your-model>
```

Builds a prompt, then has the server-side code read it back, over deliberately
awkward documents:

- one containing the text `# Question`
- one containing a fake `Magic Chunk 7` header
- one containing the model's own turn markers
- one containing DA tags
- empty, whitespace-only, and one long word with no spaces

Each case checks the server found exactly the chunks the app wrote.

**Run this first on any new model.** It is quick and it catches broken chat
templates, which are otherwise invisible.

## Check 7 — A/B inside one server start

`StepToggle` alternates two variants step by step in a single run.

Restarting the server changes timings by 0.3–0.5 ms on its own, which is more
than most of the effects being measured. Comparing across two restarts proves
nothing.

## Check 8 — the fast masking code matches the slow one

`tests/test_remap.py` runs both over random batch sizes, block sizes, document
lengths and mask patterns, and requires **byte-identical** output.

Do not test this by comparing generated text. Differently shaped temporary
arrays shift memory allocation, the last bit of a probability changes, and the
chosen word flips — even when the masking provably did nothing.

## Quick version

```bash
pytest -q                                # everything that needs no GPU
da validate --model <your-model>         # checks 5 and 6, real tokenizer
```

Checks 1, 3 and 4 need a GPU and a running model.
