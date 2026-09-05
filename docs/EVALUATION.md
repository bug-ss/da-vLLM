# Evaluation

## Data

15 sources, listed explicitly in `da_vllm.eval.data.SOURCES` — every
aggregation is computed against that list, never against whatever happens to be
in a results directory.

**Single-span retrieval / reasoning (10):** RULER niah_single_1/2/3,
niah_multikey_2/3; LongBench v1 qmsum; LongBench v2 multidoc_qa and code_repo;
LooGLE summarization; ZeroSCROLLS quality.

**Multi-span reasoning (5):** LongBench v2 singledoc_qa and dialogue_history;
LooGLE longdep_qa and shortdep_cloze; ZeroSCROLLS space_digest.

Four sources use synthetic questions written by Gemini 3 Flash: qmsum,
multidoc_qa, singledoc_qa, LooGLE summarization.

Every example carries a binary rubric written by Gemini 3 Flash from the
reference answer. The rubric's first bullet states the exact correct answer and
partial credit is forbidden (`eval.judge.RUBRIC_SYSTEM_PROMPT`).

### The filtering pipeline

`eval.data.prepare_source`, in this order:

1. drop rows without a rubric;
2. dedupe contexts by hash of the normalized text;
3. filter by context length — at least 4096 tokens, at most
   `max_model_len - 8192 - 4096` (244K for the 256K models, 116K for
   Gemma-4-E4B). Over-length rows are **dropped, not truncated**;
4. shuffle with seed 42;
5. take 128.

Tokenize with **the served model's own tokenizer** for the length filter. The
datasets' stored token counts came from a different tokenizer, and a count from
the wrong tokenizer silently overflows the engine cap. Use validation splits
where a source ships train plus validation.

Result on the headline models: 1611 examples per arm.

### Running it

`examples/run_eval.sh`, or the four commands it wraps (`da prepare`, `da run`,
`da judge`, `da score`).  Input is a JSONL file, one example per line:

```json
{"example_id": "ruler/niah_single_1:0", "source": "ruler/niah_single_1",
 "context": "...", "question": "...", "reference_answer": "...",
 "rubric": "- CORRECT if the response states ..."}
```

Rubrics are an **input**, not something a public split carries.  Generate them
with `da_vllm.eval.rubrics.generate_rubrics`, which stores the authoring model
next to each rubric -- a rubric whose author is unknown cannot be audited later.
`attach_questions` merges the four sources' synthetic questions and refuses to
rewrite a question on a source that uses its original QA.

Run **one arm per process** (`da run --arm ...`): an orphaned EngineCore
reparents to PID 1 and holds VRAM across arms, and each arm needs its own
compile cache.

## Arms

| Arm | Prompt | Mask |
| --- | --- | --- |
| `vanilla` | plain context inline | full attention |
| `da_no_mask` | the magic-chunk prompt | full attention |
| `da` | the magic-chunk prompt | on |

The DA and DA-no-mask arms render **byte-identical** prompts; they differ only
in whether the engine has the patch installed. Serve the no-mask arm from a
separate engine with a separate compile cache (`EngineOptions` enforces both).

Without the no-mask arm, the cost of the prompt format cannot be separated from
the cost of the mask, and the wrong conclusion gets drawn.

## Generation

Thinking off — models do not follow the DA protocol inside thinking tags.
`max_tokens` 8192, `n = 1`, no seed. Sampling parameters come from each model
card, registered per model; an unknown model raises rather than defaulting:

- Qwen (non-thinking): `T 0.7, top_p 0.8, top_k 20, presence_penalty 1.5`
- Gemma 4: `T 1.0, top_p 0.95, top_k 64`

Serve with `gpu_memory_utilization 0.85`, `max_num_seqs 256`, `max_model_len`
equal to the model's maximum, prefix caching at vLLM's default.

## Judge

`Qwen/Qwen3.5-4B`, thinking **on**, model-card thinking sampling
(`T 1.0, top_p 0.95, top_k 20, presence_penalty 1.5`), `max_tokens` **32768**.

The judge sees the question, the rubric, and only the text inside
`<answer>...</answer>`. Never the context, never the DA tags. Output is
`{"correct": bool}`; the pipeline strips the thinking block, parses JSON, and
falls back to a regex. A response with no parseable answer tag is scored wrong
**without a judge call**.

The judge prompt lists the surface differences to ignore — case,
singular/plural, numeric formatting, whitespace, LaTeX, wrapper text — which was
added after an audit found most false negatives were case and plural mismatches.

Two mistakes to avoid:

- a thinking judge capped at 4096 tokens was truncated before emitting its
  verdict in about 3.5% of cases, which looked like a prompt problem. Hence the
  32768 cap and the explicit `Verdict.truncated` flag;
- scoring each model with itself as judge **reversed** the format-versus-mask
  conclusion. One fixed judge for every arm, and its identity is stored on every
  verdict.

Judge validation: 2,993 responses stratified over source, model and arm,
re-judged by Gemini 3.1 Pro at low thinking under the same rubric and prompt.
Agreement 98.5%, Cohen's kappa 0.94, per-cell Pearson r 0.992, macro accuracy
difference 0.13 points.

## Metrics

**Accuracy per source** = correct / all responses, format failures counted as
wrong. **Headline accuracy** = the unweighted (macro) mean over the 15 sources.
Macro over sources, micro within a source — and `eval.score` returns the
per-source table alongside every headline so the two can never be confused.

**Attended tokens** are reconstructed by replaying the state machine over the
returned text; the server does not report them. Per decode step: in GLOBAL, all
non-pad positions so far; in FOCUS or LOCAL, the True entries of the mask frozen
at mode open, plus tokens generated since. Summed over steps, averaged per
response within a source, then unweighted across sources, and compared as a
percent change against vanilla. The vanilla baseline is analytic: step *t*
attends `prompt_length + t`.

The count is token-granular. The kernel reads at block granularity, so real
bytes are a few percent higher — `ReplayResult.block_aligned_attended_tokens`
reports that when a block size is supplied.

**Decode steps** = number of generated tokens.

### Non-terminating responses

0.2 to 1.4% of DA sequences run to the 8192 cap without terminating and inflate
the attended-token sum as logged. On one mid-size model this was 6% of responses
and put its attended tokens above vanilla until excluded. `eval.score.report`
always returns **both** views: as-logged, and with `decode_steps >= 8000`
dropped.

## DA does not pay on short contexts

The 4096-token floor is not arbitrary.  The DA prompt carries a fixed scaffold
-- the tool declaration, the ~1.7K-token instruction, and a turn wrapper per
magic chunk -- and roughly half the decode steps run in global mode at that
larger prompt length.  Measured against the analytic vanilla baseline with the
2048-token segment target:

| Context | DA vs vanilla attended tokens |
| --- | --- |
| ~1K tokens | worse than vanilla |
| ~7.5K tokens | roughly break-even |
| ~24K tokens | −27% |
| ~47K tokens | −31% |

The DA-no-mask arm reads more than vanilla at **every** length, which is the
point of having it: the format costs, and the mask has to win that back before
it wins anything.

Segment size feeds the same arithmetic.  At the 2048-token target the per-chunk
turn wrappers are about 8% of the prompt; at 200 tokens they are about 45% and
the mask cannot make it back.  `tests/test_api.py` pins both ends of this.

## Numbers to check against

| | Gemma-4-31B | Qwen-3.6-27B |
| --- | --- | --- |
| Accuracy vanilla / no-mask / DA | 87.01 / 87.01 / 85.74 | 85.31 / 84.62 / 82.56 |
| Attended tokens per response, vanilla / no-mask / DA (M) | 13.43 / 22.31 / 6.45 | 22.54 / 29.02 / 15.52 |
| DA vs vanilla attended tokens | −52.0% | −31.1% |
| DA vs no-mask attended tokens | −71.1% | −46.5% |
| Decode steps, DA vs vanilla | +34.8% | +31.3% |
| Focus attempts per response (mean of both models) | 1.48 | |
| Format-correct rate under DA | 0.998 | 0.992 |

Generation is unseeded, so expect small run-to-run movement; attended-token
reduction moves more than accuracy with the sample draw.

The projected decode wall-time ratios — 0.71x on Gemma-4-31B and 0.77x on
Qwen-3.6-27B — are reproduced by `da_vllm.metrics.roofline` from the table
above at a mean vanilla prompt length near 41-42K tokens, which the two models
imply independently. `tests/test_metrics.py` asserts both.

## Where DA does not work

Six sources sit outside what zero-shot prompting covers, in two structural
groups — not weaknesses of the mechanism, and DA still cuts per-step attention
on all six by 39 to 67%:

1. **Evidence destroyed by segmentation** — a global count over all segments
   (`cwe`), or a table the segmenter splits (`structured_data`). Accuracy
   collapses even as per-step attention drops.
2. **Output length grows with the document** — word-frequency enumeration
   (`fwe`), per-segment summarization, full-document ordering, per-example
   in-context learning. Decode steps scale with the document, so attended
   tokens accumulate end to end despite the per-step saving.

`ruler_qa_1` / `ruler_qa_2` are skipped entirely: their hardcoded task prompts
forbid intermediate reasoning ("Only give me the answer and do not output any
other words"), which neutralizes the scaffolding DA relies on.
