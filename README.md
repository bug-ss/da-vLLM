# da-vLLM — Declarative Attention

**The problem.** To write each word of an answer, a language model re-reads the
entire document. Every word. On a long document that is where the time goes.

**The idea.** Let the model say where it needs to look, in its own words, and
have the server listen. The model writes tags as it thinks:

| Tag | Means |
| --- | --- |
| `<global>` | I need to see the whole document |
| `<focus magic_chunks="3">` | I only need chunk 3 right now |
| `<local>` / `<answer>` | I don't need the document at all, just my own notes |

The server reads those tags as they come out and skips the parts of the
document the model said it doesn't need. Nothing is fetched and nothing is
deleted — the document stays loaded, it just stops being re-read.

This is from the paper *Language Models Can Control Their Own Attention*
(arXiv:2609.02737). It works on off-the-shelf models with no fine-tuning.

**What it buys you.** Across 15 long-document benchmarks: 52% less reading on
Gemma-4-31B and 31% less on Qwen-3.6-27B, at a cost of 1.3 and 2.8 points of
accuracy.

## Try it without a GPU

```bash
pip install -e '.[dev]'
pytest -q                            # 453 tests, no downloads
python examples/offline_dryrun.py
```

The dry run uses a fake model but the real segmenter, prompt builder, tag
reader and masking code. It prints how many memory pages the GPU would read at
each step:

```
 step     mode  blocks read     of  saving
    0   global          540    540    0.0%
   36    focus          126    542   76.8%
   54    local          111    543   79.6%
```

## Use it

Everything on one machine:

```python
from da_vllm import DAEngine

with DAEngine("Qwen/Qwen3.6-27B") as engine:
    r = engine.answer(document, "When was Acme founded?")

r.answer            # "2003"
r.reduction_pct     # how much less the GPU read
r.declines          # times the model asked to focus and the server said no
```

App and GPU on different machines:

```python
from da_vllm import DAClient

client = DAClient("http://gpu-box:8000", "google/gemma-4-31B-it")
r = client.answer(document, "When was Acme founded?")
```

See [docs/USAGE.md](docs/USAGE.md) for how to start the server, and for using
DA as a tool inside an agent.

## Install

```bash
pip install -e '.[dev]'                  # library and tests, no GPU
pip install -r requirements-serve.txt    # the pinned vLLM stack, needs a GPU
```

Everything except the vLLM hook runs and is tested without a GPU.

## Read this before you use it

**DA costs something on short documents.** It adds about 2,700 tokens to every
prompt and makes the model write about a third more words. That cost is fixed;
the saving grows with document size. Below a few thousand tokens DA reads
*more* than normal. It breaks even around 7,500 tokens and reaches 31% cheaper
by 47,000. Send short documents down your normal path.

**It does not make answers better.** Accuracy drops slightly. "Focus" describes
where the machine reads from, not the model concentrating harder. If you want
better answers on long documents, this is the wrong lever.

**It has never run on a GPU here.** The vLLM integration was written against
the real vLLM 0.20.2 source and matches it, and the tests check that — but no
kernel has actually executed. Run the checks in
[docs/VALIDATION.md](docs/VALIDATION.md) before trusting any number.

## Docs

| | |
| --- | --- |
| [ORIENTATION.md](docs/ORIENTATION.md) | what every file does, in plain language — **start here** |
| [USAGE.md](docs/USAGE.md) | running it, including across two machines |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | how it works inside |
| [VALIDATION.md](docs/VALIDATION.md) | how to prove it is actually working |
| [EVALUATION.md](docs/EVALUATION.md) | reproducing the paper's numbers |
| [ENVIRONMENT.md](docs/ENVIRONMENT.md) | versions, model facts, GPU setup |
| [PITFALLS.md](docs/PITFALLS.md) | every known way this breaks, and what stops it |

## Layout

| Path | What it is |
| --- | --- |
| `api.py` | `DAEngine` — the main entry point |
| `client.py` | `DAClient` — same thing, over HTTP |
| `segmenter.py` | splits a document into numbered chunks |
| `prompt.py` | builds the prompt (the only place that does) |
| `detect.py` | server side: finds where each chunk sits |
| `state_machine.py` | watches for tags, decides what stays visible |
| `masking/` | the GPU side: the mask, and the vLLM hook |
| `metrics/` | counts what was saved |
| `eval/` | the 15-benchmark harness |
| `validation/` | checks that the masking really happens |
| `timing.py` | how to benchmark this honestly |
| `training.py` | notes for fine-tuning on DA traces |
| `testing.py` | fake tokenizer, so everything runs offline |
| `examples/` | dry run, quickstart, agent integration, remote client, benchmark script |

## Tests

```bash
pytest -q          # 453 tests, no GPU
```

They cover: documents survive being split and rejoined; prompts survive the
round trip on both chat-template families, including documents written to
confuse the parser; every path through the tag reader; the fast and slow
masking code producing byte-identical output; the vLLM hook against a stand-in
built from the real vLLM source; a full simulated decode loop; the HTTP client;
and the cost model reproducing the paper's projections.
