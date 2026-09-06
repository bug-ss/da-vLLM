# What each part of this repo does

Plain-language map, in the order things actually happen.

## The one-sentence version

The model writes tags in its own answer saying where it needs to look
(`<global>`, `<focus magic_chunks="3">`, `<local>`). The server reads those tags
as they stream out and hands the attention kernel fewer pages of the KV cache.
Less memory read per token, so decoding gets cheaper.

Everything in `src/da_vllm/` is one of three things: **preparing the prompt**,
**reacting to the tags while the model types**, or **measuring what happened**.

---

## Start here

| File | What it is |
| --- | --- |
| `api.py` | The front door. `DAEngine("model").answer(document, question)` → an answer plus how much attention it saved. If you only read one file, read this one. |
| `examples/offline_dryrun.py` | Runs the whole thing on your laptop with no GPU and no model download. Best way to see what's going on. |
| `cli.py` | The `da` command. Inspect prompts, check your setup, run the evaluation. |
| `config.py` | Every knob, in one dataclass. Unknown keys raise instead of being ignored. |
| `models.py` | Per-model facts: chat-template quirks, sampling settings, attention shapes. Lookup is exact — an unregistered model raises rather than quietly using another model's settings. |

---

## Phase 1 — turning a document into a prompt

This all happens before the model generates anything.

**`segmenter.py` — chop the document into addressable pieces.**
The model can only say "look at chunk 3" if there *are* numbered chunks. This
splits the document into ~2048-token pieces, cutting at the most natural
boundary available (paragraph break, then line break, then sentence, then
clause, then space). It works purely with character positions and never
re-assembles text from token IDs, so it can't corrupt emoji or non-Latin text.
Glue the pieces back together and you get the original document, exactly.

**`prompt.py` — write the prompt.**
Presents each chunk as if the model had already fetched it with a tool call:
"assistant calls `get_magic_chunk(id="3")`", "tool returns `Magic Chunk 3` +
text". Models track tool-call boundaries very reliably because they saw
millions of them in training. Then it appends the question and the instructions
that explain the three tags.

There is exactly **one** renderer, and every path uses it — serving, replay,
evaluation. It stamps a fingerprint (a hash) on each prompt so if two paths ever
drift apart, a test fails instead of the numbers quietly becoming meaningless.

**`detect.py` — find the chunks again, from the server's side.**
The server gets a bare list of token IDs. To honour "look at chunk 3" it has to
know which token positions chunk 3 occupies. This walks the conversation-turn
markers (special tokens a document can't fake) and works out the span of every
chunk. It also finds the "always visible" region: the question and instructions
at the end.

It is deliberately paranoid. If anything looks off — chunk numbers not exactly
1,2,3,…N, a tool response with no matching call, a token count that disagrees
with the engine's — it gives up on focus mode for that request and the request
just runs normally. Never crashes, never guesses.

---

## Phase 2 — reacting while the model types

**`state_machine.py` — watch for tags, decide what stays visible.**
After every generated token, if that token contains a `>`, four small regexes
check whether a tag just closed. `<focus magic_chunks="3">` switches to focus
mode; `</focus>` switches back. `<answer>` counts as `<local>`, so the final
answer is written without the document in view.

It then produces the list of KV positions to keep:

- the first 16 tokens (an "attention sink" — models get incoherent without one,
  which is why the prompt starts with a throwaway system message);
- the question and instructions at the end;
- everything the model has written so far, **and everything it will write** —
  so a token generated later can never fall outside the mask;
- under focus, the named chunk(s).

Parsing a focus tag is biased towards saying no. Bad syntax, unknown chunk
number, more than three chunks → declined, and the request keeps full
attention. A declined focus costs speed; a wrong focus corrupts the answer.

**`masking/shared.py` — the shared scratchpad.**
One big true/false grid on the GPU: one row per concurrent request, one column
per token position. The state machine writes rows; the attention hook reads
them. That's the entire handoff between the two halves.

**`masking/remap.py` — turn true/false into "read fewer pages".**
vLLM stores the KV cache in fixed-size blocks and the kernel reads whole blocks.
So a mask only saves anything if it drops entire blocks. This rewrites the
request's list of blocks to contain only the kept ones, and shortens the
recorded sequence length to match. The kernel then just… reads less. No kernel
changes at all.

Two versions live here: a readable one that's the specification, and a fast one
with no CPU↔GPU stalls. Tests prove they produce byte-identical output.

**`masking/patch.py` — where DA hooks into vLLM.**
Wraps vLLM's attention-metadata builder. Every decode step it runs the remap.
It also has to be careful about several things that will silently corrupt output
if you get them wrong: skip cache types whose pages rotate, don't touch the
sequence-length tensor that's shared between layer groups, use the block size of
the layer group actually calling, and stay out of the way during CUDA graph
capture. Each of those has a comment explaining what breaks.

**`masking/logits_processor.py` — the driver.**
vLLM runs the model in a separate process, so a patch applied from your script
patches nothing. This class is the trick: vLLM builds it *inside* that process,
so its constructor is where the patch gets installed. After that it runs once
per decode step — tracking which request sits in which batch slot, feeding new
tokens to each state machine, and writing any changed mask row.

**`resources/sitecustomize.py`** — same job, for the extra worker processes that
appear when you split a model across multiple GPUs.

**`runaway.py`** — a small safety net for responses that never stop. Off by
default; the paper ran without it.

---

## Phase 3 — measuring what happened

**`metrics/replay.py` — count the savings.**
The server doesn't report how much attention it used, so this replays the state
machine over the finished text and adds it up, step by step. Compares against
"what full attention would have read", which is just arithmetic.

**`metrics/roofline.py` — estimate the time saved.**
Converts attention savings into a wall-clock estimate at the hardware's
theoretical limits. Deliberately refuses to run on models whose attention shape
nobody has verified — a wrong number here caused a retracted finding in the
original work.

---

## Phase 4 — proving it actually works

`validation/` exists because in the original work **the mask did nothing for
weeks and every obvious check passed**.

- **`nll_parity.py`** — the check that catches it. Score the same text three
  ways and require a specific ordering. If two of the three agree perfectly, the
  mask isn't being applied.
- **`checks.py`** — round-trip tests (does the server find the same chunks the
  renderer wrote?) against deliberately nasty documents: ones containing
  `# Question`, ones containing fake chunk headers, empty ones.
- **`capture.py`** — records what the kernel actually received, for offline
  replay.
- **`timing.py`** — how to benchmark this honestly, and the four reasons naive
  benchmarks mislead.

**Run `da validate --model <your-model>` before trusting anything.**

---

## Phase 5 — the benchmark (only if reproducing the paper)

`eval/` is a self-contained harness: 15 benchmark sources, three comparison
arms, one fixed judge model, and scoring that always recomputes from raw
per-response records.

- `data.py` — the source list and the filtering rules (drop short documents,
  drop over-long ones rather than truncating, dedupe, sample 128 with a fixed
  seed).
- `rubrics.py` — generate the grading criteria for each question.
- `pipeline.py` — the actual runner: prepare → generate → judge → score.
- `judge.py` — the grading prompt and answer parsing.
- `score.py` — averaging, done one consistent way and reported alongside the
  per-source breakdown.
- `records.py` — the raw output file format.

Skip all of this if you just want DA in your own application.

---

## The rest

| File | What it is |
| --- | --- |
| `serving.py` | Engine setup: correct flags per arm, a separate compile cache per configuration, process cleanup so a dead engine doesn't hold your GPU. |
| `testing.py` | A fake tokenizer with realistic chat templates. Lets the whole pipeline run with no downloads. |
| `training.py` | Notes and helpers if you ever fine-tune a model on DA traces. Not used by anything else. |

---

## Three ways you might use this

**Just want the speedup in your app** → `api.py`, `config.py`, `models.py`.
Read `docs/USAGE.md`. Ignore `eval/` entirely.

**Want to know how it works** → `examples/offline_dryrun.py`, then
`state_machine.py`, then `masking/remap.py`. `docs/ARCHITECTURE.md` has the
full picture.

**Want to reproduce the paper** → `docs/EVALUATION.md` and
`examples/run_eval.sh`.

---

## Two things to know before you start

**DA costs something on short documents.** The prompt carries a fixed overhead
of roughly 2,700 tokens (tool declaration, instructions, per-chunk wrappers),
and about half the decode steps run in global mode at that larger size. Below a
few thousand tokens of context it reads *more* than plain attention. Break-even
is around 7,500 tokens; by 47,000 it's about 31% cheaper. Send short documents
down a normal path.

**The vLLM hook has never run on a GPU here.** It was written against the real
vLLM 0.20.2 source and matches its contract, and the tests check that contract —
but no kernel has actually executed. `docs/VALIDATION.md` items 1, 3 and 4 are
what turn "matches the contract" into "verified working".
