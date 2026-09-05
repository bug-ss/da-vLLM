# Validation checklist

**This is not optional.** The mask silently did nothing for weeks in the work
this reproduces, and the obvious checks all passed the whole time. Run these
before reporting a number.

## 1. Three-column NLL parity

`da_vllm.validation.nll_parity.three_column_parity`

For each prompt: greedy-decode with vanilla attention and with DA. Then
teacher-force in HuggingFace three ways, recording per-token NLL on the
completion:

| column | response | mask |
| --- | --- | --- |
| `v` | vanilla response | causal |
| `d` | DA response | the 4D DA mask |
| `dv` | DA response | plain causal |

Require **`v ~ d < dv`**.

Two columns are not enough. When the patch was silently not applied, `v` and `d`
agreed perfectly — that is what a no-op looks like. The `dv` column is the only
one that proves the mask changed the computation: the DA response must be *less*
likely under full attention than under the mask it was generated with.

Two properties the HF reference must have, or it diverges by an order of
magnitude:

- it OR-reduces the KV dimension at the **served block size**
  (`block_align_mask`), matching the engine's outward rounding;
- on hybrid models it passes a **per-layer-type** mask (`per_layer_masks`): the
  DA mask on full-attention layers, the sliding mask on sliding-window layers.

Expect a baseline gap of 0.05 to 0.15 nats between vLLM and HF from bf16 drift.
`ParityResult.explain()` names which of the two conditions failed.

## 2. Decoded text inspection

Read the generated text, every time. Low NLL with garbage text is the signature
of corrupted KV in a layer the harness cannot see — the symptom that turned out
to be the shared `seq_lens` being shrunk in place for the full-attention group,
which corrupted the sliding-window group's kernel on Gemma.

## 3. Realized kept fraction

Set `DAConfig(log_kept_fraction=True)` to log the actual kept-over-total block
ratio from inside the remap, and compare it with the token-level metric via
`validation.checks.kept_fraction_report`. The block figure must be the larger of
the two.

**This adds a host-device sync. Never leave it on while timing.**

## 4. The kernel honours the mask

`da_vllm.validation.checks.kernel_scaling`

Capture real kernel arguments from the engine, replay them offline, shrink the
sequence lengths, and check that time scales with the kept fraction. The Triton
kernel tracked the kept fraction at 91 to 96% efficiency. Capture in **eager
mode**: a Python monkeypatch on a kernel is bypassed inside a replayed CUDA
graph.

## 5. Serve and replay render the same prompt

`da_vllm.validation.checks.assert_prompt_parity`

Token for token, by fingerprint. Serve and replay once differed by the
tool-declaration system block — about 340 tokens on Qwen — so the reported
metrics described a prompt the model never saw. Every render path in this
repository goes through `PromptRenderer`; the assertion is what keeps it that
way.

## 6. Round trip, per family

`da_vllm.validation.checks.round_trip` / `da validate --model <model>`

Render, detect, parse — over adversarial contexts:

- a document containing `# Question`;
- a document containing `Magic Chunk 7`;
- a document containing the family's own turn literal;
- a document containing DA tags;
- short, empty, and whitespace-only contexts;
- a whitespace-free run that cannot be split.

Each case asserts that the number of segments rendered equals the number
detected, that ids are exactly 1..N, that the local window did not fall back,
and that a `<focus>` tag parses.

## 7. A/B inside one engine boot

`da_vllm.validation.checks.StepToggle`

Alternate the two variants step by step within a single boot. Cross-boot
variance (0.3 to 0.5 ms) is larger than the effects being measured, so an A/B
across two process launches proves nothing.

## 8. Remap equivalence

`tests/test_remap.py`

Every optimized variant must produce **bit-identical** `block_table[0:seq_len]`
and `seq_lens` against the readable version, across a sweep of batch sizes,
block sizes, sequence lengths and prune patterns, including R = 0, narrow block
tables, and sequence length above `max_model_len`.

Do **not** use generated-token identity as the test: differently shaped
transient tensors change allocator state, last-bit logits differ, and greedy
argmax flips even when the remap is a verified no-op.

## Quick run

```bash
pytest -q                                    # everything that needs no GPU
da validate --model Qwen/Qwen3.6-27B         # checks 5 and 6 against the real tokenizer
```

Checks 1, 3 and 4 need a GPU and a served model.
