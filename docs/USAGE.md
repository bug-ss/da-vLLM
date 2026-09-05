# Using DA

## Verify the install without a GPU

```bash
pip install -e '.[dev]'
pytest -q                        # 434 tests, no GPU, no downloads
python examples/offline_dryrun.py
```

The dry run walks the real segmenter, renderer, detector, state machine, mask
writer, block-table remap and accounting against a scripted model, and prints
the block table shrinking as the model declares each mode:

```
 step     mode  blocks read     of  saving
    0   global          540    540    0.0%
   36    focus          126    542   76.8%
   54    local          111    543   79.6%
```

If that runs, everything except the vLLM hook is working.

## One question about one document

```python
from da_vllm import DAEngine

with DAEngine("Qwen/Qwen3.6-27B") as engine:
    result = engine.answer(long_document, "When was Acme founded?")

result.answer             # "2003"
result.attended_tokens    # KV positions read, summed over decode steps
result.reduction_pct      # against the analytic vanilla baseline
result.mode_trace         # [(step, "focus"), (step, "global"), ...]
result.declines           # focus requests the server refused, and why
```

`DAEngine` builds the vLLM engine on first use, which installs the attention
patch inside EngineCore and allocates the shared mask. Everything it reports
comes from the same replay the paper's evaluation uses, so a number here is the
number the evaluation would print.

## Inside an agent

Wrap DA as a **tool**, not as your chat backend. The protocol needs the
magic-chunk transcript and a single question, and thinking mode is off, so
routing a whole agent conversation through it does not work. `examples/agent_integration.py`
is a complete worked example; the shape is:

```python
class LongContextTool:
    def __init__(self, engine): self.engine = engine   # one engine, reused

    def batch(self, document, questions):
        results = self.engine.answer_batch([(document, q) for q in questions])
        return [{"answer": r.answer, "attended_tokens": r.attended_tokens} for r in results]
```

Three things worth doing:

- **Batch questions about the same document.** One engine call, one prefill.
- **Give the agent `result.answer`, not `result.text`.** The DA tags are a
  serving protocol; an agent reasoning over them will imitate them.
- **Log `result.declines` and `result.detection_failure`.** A decline means the
  model asked to focus and the server refused, so that call ran at full cost.
  A detection failure means the request ran in global mode for its whole life.

### DA only pays once the context clears the scaffold

DA adds a fixed cost to every prompt: the tool declaration, the ~1.7K-token
instruction, and a turn wrapper per magic chunk. It then spends roughly half its
decode steps in global mode, at that larger prompt length. Below a few thousand
context tokens that overhead dominates and DA reads **more** than vanilla:

| Context | DA vs vanilla attended tokens |
| --- | --- |
| ~1K tokens | worse than vanilla |
| ~7.5K tokens | roughly break-even |
| ~24K tokens | −27% |
| ~47K tokens | −31% |

This is why the evaluation filters out contexts below 4096 tokens. Send short
documents down a plain path.

Segment size matters for the same reason: at the guide's 2048-token target the
per-chunk turn wrappers are ~8% of the prompt; drop to 200-token chunks and they
become ~45% and DA cannot win. Do not shrink `segment_target_tokens` to get more
addressable chunks.

## Serving over HTTP

The mask only exists where the patch is installed, so an OpenAI-compatible
server has to be started with the logits processor registered:

```bash
eval "$(da serve-command --model Qwen/Qwen3.6-27B --arm da)"
```

Per-request opt-in then travels in `extra_body`:

```json
{"vllm_xargs": {"da_enable": true, "da_prompt_text": "<the rendered prompt>"}}
```

Render that prompt with `PromptRenderer` — the same one the server's detector
assumes — and send it as token ids so the server does not re-tokenize it.

Without the patch you still get the DA prompt format and honest replay
accounting, but full attention. That configuration has a name: it is the
`da_no_mask` arm.

## Running the evaluation

`examples/run_eval.sh` is the whole thing. The steps:

```bash
da validate --model "$MODEL"                                   # 0. never skip
da prepare  --model "$MODEL" --examples in.jsonl --out prepared.jsonl
for ARM in vanilla da_no_mask da; do                           # 2. one process each
  da run   --model "$MODEL" --examples prepared.jsonl --out runs/ --arm "$ARM"
  da judge --records runs/records-$ARM.jsonl --examples prepared.jsonl
done
da score --model "$MODEL" --records runs/records-*.jsonl
```

Input is a JSONL file, one example per line:

```json
{"example_id": "ruler/niah_single_1:0", "source": "ruler/niah_single_1",
 "context": "...", "question": "...", "reference_answer": "...",
 "rubric": "- CORRECT if the response states ..."}
```

Rubrics are inputs, not something a public split carries — generate them with
`da_vllm.eval.rubrics.generate_rubrics`, which stores the authoring model
alongside each one. Four sources also use synthetic questions
(`attach_questions`, which refuses to rewrite a question on a source that uses
its original QA).

## Command reference

| Command | Needs a GPU | What it does |
| --- | --- | --- |
| `da models` | no | the registry, with placeholder geometry flagged |
| `da config` | no | the default `DAConfig` as JSON |
| `da segment` | no | how a document splits into magic chunks |
| `da render` | no | the exact prompt, and its fingerprint |
| `da validate` | no | round-trip render/detect/parse on adversarial contexts |
| `da serve-command` | no | the `vllm serve` argv and env for one arm |
| `da prepare` | no | filter, dedupe and sample examples for one model |
| `da run` | yes | generate and replay one arm (or all three) |
| `da judge` | yes | score records with the one fixed judge |
| `da score` | no | recompute every number from raw records |
| `da roofline` | no | roofline decode wall-time for one response |
