# da-vLLM — Declarative Attention

An implementation of **Declarative Attention (DA)**: a protocol that elicits a
language model to declare, inside its own chain of thought, where it will
attend — and a vLLM integration that turns those declarations into a compacted
KV block table so the stock attention kernel reads fewer pages.

Three modes, emitted as ordinary text:

| Tag | What the model sees |
| --- | --- |
| `<global>` (default) | every magic chunk |
| `<focus magic_chunks="K">` | only magic chunk K, plus the scaffold |
| `<local>` / `<answer>` | the scaffold and the response so far, no chunks |

A **scaffold** — a 16-token attention sink, the question, the DA instruction,
and the entire response so far — stays attended in every mode. Only the
context's visibility changes.

Reported effect, zero-shot on off-the-shelf models across 15 long-context
sources: attended tokens −52.0% (Gemma-4-31B) and −31.1% (Qwen-3.6-27B), for
accuracy drops of 1.27pp and 2.75pp.

## Install

```bash
pip install -e '.[dev]'                  # library + tests, CPU only
pip install -r requirements-serve.txt    # the pinned torch / vLLM stack + a GPU
```

Everything except the vLLM hook itself runs and is tested without a GPU. The
exact serving stack, including the two dependencies that need special handling,
is in [requirements-serve.txt](requirements-serve.txt) and
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

## Try it with no GPU

```bash
pytest -q                            # 422 tests, no downloads
python examples/offline_dryrun.py
```

The dry run drives the real segmenter, renderer, detector, state machine, mask
writer and block-table remap against a scripted model, and prints the block
table shrinking as each mode is declared:

```
 step     mode  blocks read     of  saving
    0   global          540    540    0.0%
   36    focus          126    542   76.8%
   54    local          111    543   79.6%
```

## Quickstart

```python
from da_vllm import DAEngine

with DAEngine("Qwen/Qwen3.6-27B") as engine:          # patches vLLM's EngineCore
    result = engine.answer(long_document, "When was Acme founded?")

result.answer            # "2003"
result.attended_tokens   # KV positions read, summed over decode steps
result.reduction_pct     # against the analytic vanilla baseline
result.declines          # focus requests the server refused, and why
```

For an agent, wrap it as a tool rather than routing the conversation through it
— see [docs/USAGE.md](docs/USAGE.md) and `examples/agent_integration.py`.

Command line:

```bash
da validate --model Qwen/Qwen3.6-27B     # round-trip render/detect/parse: run this first
da segment  --model Qwen/Qwen3.6-27B --context doc.txt
da render   --model Qwen/Qwen3.6-27B --arm da --context doc.txt --question "When?"
da serve-command --model Qwen/Qwen3.6-27B --arm da    # HTTP serving, with the patch
bash examples/run_eval.sh                             # the full three-arm evaluation
```

## Validate before you believe anything

The mask in the original work silently did nothing for weeks, and two-column
NLL checks agreed perfectly the whole time. Run the checklist in
[docs/VALIDATION.md](docs/VALIDATION.md) before reporting a number. The single
most important check is three-column NLL parity (`v ~ d < dv`) — the `dv`
column is the only one that proves the mask changed the computation.

## Layout

| Path | What it is |
| --- | --- |
| `src/da_vllm/api.py` | `DAEngine` / `DAAnswer` -- the entry point |
| `src/da_vllm/segmenter.py` | tokenizer-aware segmenter, character-offset space only |
| `src/da_vllm/prompt.py` | the one renderer: simulated `get_magic_chunk` transcript |
| `src/da_vllm/detect.py` | serving-side segment map, from turn and tool boundaries |
| `src/da_vllm/state_machine.py` | tag detection, mode transitions, mask construction |
| `src/da_vllm/masking/shared.py` | the shared `(max_num_seqs, max_model_len)` mask |
| `src/da_vllm/masking/remap.py` | block-table remap: readable reference + sync-free version |
| `src/da_vllm/masking/patch.py` | the FlashAttention / Triton metadata-builder hook |
| `src/da_vllm/masking/logits_processor.py` | the driver, running inside EngineCore |
| `src/da_vllm/metrics/` | attended-token replay, roofline decode wall-time |
| `src/da_vllm/eval/` | 15 sources, three arms, rubrics, the fixed judge, macro scoring, the runnable pipeline |
| `src/da_vllm/validation/` | the checklist, including three-column NLL parity |
| `src/da_vllm/timing.py` | the wall-clock protocol that held up |
| `src/da_vllm/training.py` | fine-tuning helpers (not part of the paper's results) |
| `src/da_vllm/testing.py` | offline tokenizer with real chat templates, for dry runs |
| `examples/` | offline dry run, quickstart, agent integration, the eval script |

## Documentation

- [docs/USAGE.md](docs/USAGE.md) — using DA from an agent, serving it over HTTP, running the eval
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how a request flows through the system
- [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) — pinned versions and model-specific facts
- [docs/EVALUATION.md](docs/EVALUATION.md) — data, arms, judge, metrics, numbers to check against
- [docs/VALIDATION.md](docs/VALIDATION.md) — the checklist
- [docs/PITFALLS.md](docs/PITFALLS.md) — every known failure and where this code prevents it

## Tests

```bash
pytest -q          # 422 tests, no GPU required
```

The suite covers the segmenter's losslessness, prompt/detect round trips
against adversarial documents on both chat-template families, every state
machine gate, bit-identity between the two remaps across a randomized sweep,
the metadata-builder hook against a stand-in vLLM, a full simulated decode
loop, the three-arm evaluation pipeline end to end, the CLI, and the roofline
against the paper's projected 0.71x / 0.77x.

## One thing to know before you use it

DA is not free on short inputs. The prompt carries a fixed scaffold — the tool
declaration, a ~1.7K-token instruction, a turn wrapper per magic chunk — and
roughly half the decode steps run in global mode at that larger prompt length.
Below a few thousand context tokens DA reads *more* than vanilla; it breaks even
around 7.5K and reaches −31% by 47K. That is why the evaluation drops contexts
under 4096 tokens, and why you should route short documents down a plain path.
[docs/USAGE.md](docs/USAGE.md) has the numbers.
