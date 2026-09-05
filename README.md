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
pip install -e '.[dev]'          # library + tests (CPU only, no GPU needed)
pip install -e '.[serve]'        # + the pinned torch / vLLM / transformers stack
```

The pinned serving stack is in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).
Everything except the vLLM hook itself runs and is tested without a GPU.

## Quickstart

```python
from transformers import AutoTokenizer
from da_vllm import DAConfig, PromptRenderer
from da_vllm.serving import EngineOptions, build_llm
from da_vllm.eval.run import prepare_requests, generate, build_records

model = "Qwen/Qwen3.6-27B"
tokenizer = AutoTokenizer.from_pretrained(model)
renderer = PromptRenderer(tokenizer, model)          # the ONE renderer

config = DAConfig(enabled=True, max_num_seqs=256, max_model_len=262_144)
options = EngineOptions(model, arm="da", da_config=config)
llm = build_llm(options)                             # installs the patch in EngineCore
```

Command line:

```bash
da models                                   # the registry
da validate --model Qwen/Qwen3.6-27B        # round-trip render/detect/parse
da render   --model Qwen/Qwen3.6-27B --arm da --context doc.txt --question "When?"
da score    --model Qwen/Qwen3.6-27B --records runs/records.jsonl
da roofline --model google/gemma-4-31B-it --attended-tokens 6450000 --decode-steps 435
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
| `src/da_vllm/segmenter.py` | tokenizer-aware segmenter, character-offset space only |
| `src/da_vllm/prompt.py` | the one renderer: simulated `get_magic_chunk` transcript |
| `src/da_vllm/detect.py` | serving-side segment map, from turn and tool boundaries |
| `src/da_vllm/state_machine.py` | tag detection, mode transitions, mask construction |
| `src/da_vllm/masking/shared.py` | the shared `(max_num_seqs, max_model_len)` mask |
| `src/da_vllm/masking/remap.py` | block-table remap: readable reference + sync-free version |
| `src/da_vllm/masking/patch.py` | the FlashAttention / Triton metadata-builder hook |
| `src/da_vllm/masking/logits_processor.py` | the driver, running inside EngineCore |
| `src/da_vllm/metrics/` | attended-token replay, roofline decode wall-time |
| `src/da_vllm/eval/` | 15 sources, three arms, the fixed judge, macro scoring |
| `src/da_vllm/validation/` | the checklist, including three-column NLL parity |
| `src/da_vllm/timing.py` | the wall-clock protocol that held up |
| `src/da_vllm/training.py` | fine-tuning helpers (not part of the paper's results) |

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how a request flows through the system
- [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) — pinned versions and model-specific facts
- [docs/EVALUATION.md](docs/EVALUATION.md) — data, arms, judge, metrics, numbers to check against
- [docs/VALIDATION.md](docs/VALIDATION.md) — the checklist
- [docs/PITFALLS.md](docs/PITFALLS.md) — every known failure and where this code prevents it

## Tests

```bash
pytest -q          # 360 tests, no GPU required
```

The suite covers the segmenter's losslessness, prompt/detect round trips
against adversarial documents on both chat-template families, every state
machine gate, bit-identity between the two remaps across a randomized sweep,
the metadata-builder hook against a stand-in vLLM, a full simulated decode
loop, and the roofline against the paper's projected 0.71x / 0.77x.
