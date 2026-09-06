# Using DA

## What DA does, in one paragraph

Normally, to write each word of an answer, the model re-reads the whole
document. DA lets the model say "for this next bit I only need chunk 3", and
the server then skips the rest of the document while it writes that bit. The
document is never removed and never re-fetched — it just stops being re-read.

## First: check it works, no GPU needed

```bash
pip install -e '.[dev]'
pytest -q                            # 453 tests, no downloads
python examples/offline_dryrun.py
```

The dry run uses a fake model but real everything-else. It prints the number of
memory pages the GPU would read at each step:

```
 step     mode  blocks read     of  saving
    0   global          540    540    0.0%
   36    focus          126    542   76.8%
   54    local          111    543   79.6%
```

If that runs, your install is fine.

## Everything on one machine

```python
from da_vllm import DAEngine

with DAEngine("Qwen/Qwen3.6-27B") as engine:
    r = engine.answer(document, "When was Acme founded?")

r.answer              # "2003"
r.attended_tokens     # how much the GPU read, added up over the whole answer
r.reduction_pct       # compared to reading everything every time
r.declines            # times the model asked to focus and the server said no
r.detection_failure   # set if the server could not find the chunks at all
```

Building the engine also switches DA on inside vLLM. Nothing else to do.

## Model and app on different machines

This is the common case. Your app is on one box, the GPU is on another.

### On the GPU box

**1. Install into the same Python that runs vLLM.** This matters — DA switches
itself on from inside vLLM's own process, so it has to be importable there.

```bash
head -1 $(which vllm)          # shows which python
<that-python> -m pip install da-vllm
<that-python> -c "import da_vllm, vllm; print(da_vllm.__version__, vllm.__version__)"
```

**2. Check the model's chat template before anything else.**

```bash
da validate --model google/gemma-4-31B-it
```

You want `all round-trip cases passed`. This checks that the prompt the app
writes is a prompt the server can read back. Gemma's template has silently
dropped parts of it before. If this fails, nothing else will work.

**3. Start the server with two extra flags.**

```bash
export VLLM_CACHE_ROOT=~/.cache/vllm-da    # a fresh dir; the old cache is stale

vllm serve google/gemma-4-31B-it \
  --logits-processors da_vllm.masking.logits_processor:DALogitsProcessor \
  --additional-config '{"declarative_attention": {"enabled": true}}' \
  --max-num-seqs 256 \
  ...your usual flags...
```

`da serve-command --model <model> --arm da` prints this for you.

Two things to get right:

- **The colon.** vLLM splits that name on `:`. A dot fails to start the server.
- **`--max-num-seqs`.** DA keeps one bit per request slot per token position.
  At 256 that is 17 MB of GPU memory. At vLLM's default of 1024 it is 268 MB
  you are not using.

**4. Check the log says DA is on.**

```
da: patched ['FlashAttentionMetadataBuilder', 'TritonAttentionMetadataBuilder'] in pid 1234
da: driver ready in pid 1234 (mask 17 MB, patched [...])
```

If instead you see `da: no attention metadata builder was patched`, DA is
installed in the wrong Python environment and is doing nothing.

### On the app box

Install the package and the model's **tokenizer only** — no weights, no GPU:

```bash
pip install da-vllm requests
```

Then:

```python
from da_vllm import DAClient

client = DAClient("http://gpu-box:8000", "google/gemma-4-31B-it")

print(client.check())                    # is the server up and serving this model?

r = client.answer(document, "When was Acme founded?")
print(r.answer, r.reduction_pct, r.declines)
```

Same interface as `DAEngine`, same numbers. It just talks over HTTP.

### If you'd rather call the API yourself

You can, but four details are easy to get wrong, and three of them fail
quietly. `DAClient` exists so you don't have to remember them:

| Detail | Why |
| --- | --- |
| Use `/v1/completions`, not `/v1/chat/completions` | Your prompt already went through the chat template. The chat endpoint would apply it a second time. |
| Send token ids, **or** set `"add_special_tokens": false` | `/v1/completions` adds a `<bos>` by default. Your prompt already has one. Two of them shifts every position by one, and DA turns itself off. |
| `"da_enable": 1`, not `true` | vLLM types that field as string/int/float. No boolean. |
| `da_prompt_text` must be the **exact** prompt you sent | This is the text the server searches for chunk boundaries. If it differs from what you sent, the server would mask the wrong part of the document. It checks and refuses — but only if you send it correctly. |

Hand-rolled version:

```python
prompt = renderer.render_da(document, question)
ids = tok.encode(prompt.text, add_special_tokens=False)

requests.post("http://gpu-box:8000/v1/completions", json={
    "model": "google/gemma-4-31B-it",
    "prompt": ids,
    "max_tokens": 8192,
    "temperature": 1.0, "top_p": 0.95, "top_k": 64,
    "return_token_ids": True,
    "vllm_xargs": {"da_enable": 1, "da_prompt_text": prompt.text},
})
```

## Inside an agent

Treat DA as a **tool your agent calls**, not as your chat backend. It needs a
document plus one question, and it does not work with thinking mode on. Routing
a whole conversation through it does not work.

```python
class LongDocumentTool:
    def __init__(self, client):
        self.client = client          # one client, reused

    def ask(self, document, questions):
        results = self.client.answer_batch([(document, q) for q in questions])
        return [{"answer": r.answer, "cost": r.attended_tokens} for r in results]
```

`examples/agent_integration.py` is a full working version, and
`examples/remote_client.py` shows the two-machine setup end to end.

Three habits worth having:

- **Ask all your questions about a document at once.** They share the batch.
- **Give the agent `r.answer`, not `r.text`.** The `<focus>` tags are plumbing.
  An agent that reads them will start imitating them.
- **Log `r.declines` and `r.detection_failure`.** A decline means the model
  asked to focus and the server refused, so that call ran at full cost. A
  detection failure means DA was off for the whole request.

## When DA is not worth it

DA adds about 2,700 tokens to every prompt (the instructions, and a wrapper
around each chunk). It also makes the model write roughly a third more words,
because it narrates its plan. Those costs are fixed. The saving grows with
document size.

| Document size | Result vs normal |
| --- | --- |
| ~1,000 tokens | **worse** — the overhead is bigger than the saving |
| ~7,500 tokens | about even |
| ~24,000 tokens | 27% less reading |
| ~47,000 tokens | 31% less reading |

So: send short documents down your normal path.

Do not shrink the chunk size to get more chunks. At the default 2,048 tokens
per chunk, the wrappers are about 8% of the prompt. At 200 tokens per chunk
they are about 45%, and DA cannot win that back.

And be clear about what you are buying. In the paper's tests, accuracy went
**down** slightly: 87.0% to 85.7% on Gemma, 85.3% to 82.6% on Qwen. DA is a
cost saving you pay for with a little accuracy. It does not make answers better.

## Things that will not work yet

- **More than one GPU per model.** If Gemma is split across GPUs
  (`--tensor-parallel-size 2` or more), DA installs, finds nothing to do, warns,
  and serves normally. Everything in the paper was single-GPU.
- **Models not in the registry.** Run `da models` to see the list. An
  unregistered model raises at startup rather than borrowing another model's
  settings.
- **Thinking mode.** Must be off. Models do not follow the protocol inside
  thinking tags.

## Command reference

| Command | GPU needed | What it does |
| --- | --- | --- |
| `da models` | no | the models it knows about |
| `da config` | no | default settings as JSON |
| `da segment` | no | show how a document splits into chunks |
| `da render` | no | print the exact prompt |
| `da validate` | no | check prompts survive the round trip |
| `da serve-command` | no | print the `vllm serve` command |
| `da prepare` | no | filter and sample benchmark examples |
| `da run` | yes | run one arm of the benchmark |
| `da judge` | yes | grade the answers |
| `da score` | no | compute the final numbers |
| `da roofline` | no | estimate time saved |
