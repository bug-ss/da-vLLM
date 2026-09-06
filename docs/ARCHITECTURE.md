# How it works inside

This is the "what is actually happening" document. For "what should I run", see
[USAGE.md](USAGE.md); for "what is this file", see [ORIENTATION.md](ORIENTATION.md).

## The flow of one request

```
context, question
      |
      v
Segmenter               character-offset space, ~2048-token magic chunks
      |
      v
PromptRenderer          system turn -> bootstrap user turn ->
      |                 (assistant get_magic_chunk call, tool response) x N ->
      |                 DA instruction turn, all via apply_chat_template(tools=)
      v
tokenize  --------------------------------------> vLLM.generate(TokensPrompt,
      |                                              extra_args={da_enable: True})
      v                                                       |
build_prompt_map        turn/tool boundaries -> segment       |  (engine process)
      |                 token spans, sink, local window       v
      v                                            DALogitsProcessor.__init__
DAStateMachine                                         installs the patch
      |                                                installs the shared mask
      |  per decode step                                      |
      |                                                       v
      +--> update_state(batch_update): reconcile rows, advance state machines,
      |                                rewrite changed GPU mask rows
      |                                                       |
      v                                                       v
SharedMaskStore  (max_num_seqs x max_model_len bool) <---- writes
      ^
      | reads
FlashAttention / Triton MetadataBuilder.build  (patched)
      |
      +--> common_prefix_len = 0; skip sliding-window groups; learn block size;
           remap block_table in place; seq_lens through a stable scratch
      |
      v
the stock attention kernel, reading fewer KV pages
```

Afterwards, offline: `metrics.replay` reconstructs attended tokens from the
returned text, `eval.judge` scores the `<answer>` content with one fixed judge,
and `eval.score` aggregates macro over the 15 sources.

## The mask, precisely

A request's mask is a boolean row over KV positions. Three regions are kept in
**every** non-global mode:

1. **The first 16 tokens.** Models need *something* at the very start to stay
   coherent — take it away and long answers collapse into repeating `<local>`
   forever. This is why the prompt opens with a throwaway "You are a helpful
   assistant": those 16 tokens are permanently visible, so they must not be
   document content.
2. **Local window** — from the last occurrence of the question header to the
   end of the prompt, assistant header included. Falls back to the last 1024
   prompt tokens if the marker is missing, and logs a warning when it does. The
   header comes from `DAConfig.question_header`, which the renderer writes and
   the detector reads, so the two cannot diverge; the search is bounded to the
   prompt region, because running it over the model's own output was one of the
   silent bugs.
3. **Everything the model has written, and everything it will write.** The
   cut-off point never moves, so a word written after the mask was set can
   never accidentally fall outside it. It also means the mask only has to be
   rewritten when the mode changes, not every step.

`FOCUS` adds the named segment spans. `LOCAL` adds nothing. `GLOBAL` is the
all-True row.

Every region is written through one function (`_MaskWriter.keep`) that feeds
both the dense mask and the span list, so the GPU writer and the reference mask
cannot drift apart.

### Rounding out to whole blocks

The GPU stores the document in fixed-size blocks (16 or 32 tokens) and reads
whole blocks at a time. So a kept range is widened outwards until it lines up
with block edges — keep a block if any part of it is wanted. That costs at most
one extra block at each end, a few dozen tokens against a 2,048-token chunk.

The block holding the word being written right now is always kept. Drop it and
the model cannot see its own last word, and it falls apart within a few steps.

Nothing is ever thrown away. DA reduces how much is *read*, not how much is
*stored*. It does not free up room for more concurrent requests.

## Tag detection

A string scan over incrementally decoded text — not a token-id match, not a
logits hook. After each generated token, if that token's text contains `>`,
four tail-anchored regexes run over the last 500 characters:

```
open  focus   <focus[^>]*>.{0,8}$        then a strict parse of magic_chunks="..."
close focus   </focus>.{0,8}$
open  local   <(?:local|answer)>.{0,8}$
close local   </(?:local|answer)>.{0,8}$
```

`.{0,8}$` tolerates trailing characters arriving in the same token. `<global>`
has no regex and causes no transition. `<answer>` is an alias of `<local>`, so
the answer is written without seeing any context segment. When several tags land
in one scan they are applied in textual order, so `</focus><local>` closes
before it opens.

Focus opens **only** from GLOBAL. Closing either FOCUS or LOCAL returns to
GLOBAL and drops the mask.

### Two turn layouts

Qwen opens a new assistant turn per tool call; Gemma 4 collapses consecutive
calls into a single model turn. The detector accepts both, driven by
`FamilySpec.collapses_consecutive_tool_calls` rather than by guessing: a tool
response must be owned by an assistant `get_magic_chunk` turn, and under a
collapsing family one such turn can own several consecutive responses. The
spans stay disjoint either way, and any other turn between a call and its
response ends the run.

Focus parsing declines on any doubt — a syntax error, an empty list, an unknown
id, more than three ids, or a prompt with no detected segments — and records the
reason. A declined focus costs efficiency; a wrong focus corrupts the answer.

## Why the mask is always slightly behind

The GPU picks the next word and copies it back to the CPU in the background, so
when DA looks at the output the most recent word is sometimes not there yet
(it shows up as a placeholder, `-1`). DA stops at the first placeholder and
looks again next step.

The effect is that the mask is about two words behind the text. That is safe:
the always-visible parts are the same in every mode, so a slightly stale mask
never hides something the current word needs. The measurement code reproduces
the same lag, so the numbers match what really happened.

## Where the code runs

vLLM V1 runs the model in an EngineCore subprocess: **a monkeypatch applied in
the parent process patches nothing.** The patch is installed from the
constructor of `DALogitsProcessor`, which vLLM instantiates inside EngineCore.
For tensor parallel > 1 see [ENVIRONMENT.md](ENVIRONMENT.md#tensor-parallel--1).

The logits processor is the driver, not a filter: `apply()` is the identity and
`is_argmax_invariant()` returns True. All the work happens in `update_state`,
which reconciles the batch (removed, then added, then moved rows, honouring swap
semantics), advances each state machine over the tokens vLLM appended since the
last step — holding a **live reference** to vLLM's output list, never a copy —
and rewrites the GPU mask row for any request whose mode changed. Requests
without DA enabled skip state construction entirely: building it requires
decoding the whole prompt, about a second at 131K tokens, on the engine's hot
path.

The one exception to `apply()` being the identity is the runaway guard, which
overwrites a row's logits to force EOS. That changes the argmax, so with the
guard on `is_argmax_invariant()` returns False or vLLM skips `apply()` on the
greedy path.

## The remap

For every row `r`, with the calling builder's own block size `b`:

```
allowed[r, j]  = any(mask[r, j*b : (j+1)*b])
num_blocks[r]  = ceil(seq_lens[r] / b)
valid[r, j]    = j < num_blocks[r]
kept[r, j]     = valid[r, j] and allowed[r, j]
remap[r]       = query_len[r] == 1 and any(valid[r] and not allowed[r])
if remap[r]:
    block_table[r, 0:N] = block_table[r, kept columns]
    seq_lens[r] = (N - tail_kept) * b + (tail_valid_len if tail_kept else 0)
```

Rows in prefill (query length above 1) and rows with nothing pruned are
untouched, which is why chunked prefill needs no special handling. Prefix
caching stays correct because the remap rewrites the per-step GPU copy of the
block table, never the block manager's source of truth. FlashAttention's AOT
scheduler metadata is deliberately **not** recomputed: it falls back to non-AOT
scheduling on stale metadata, and Triton never reads it. `max_seq_len` is left at
the uncompacted value — a safe over-estimate, since kernels stop at `seq_lens`.

`remap_readable` is that, literally. `remap_optimized` is the same function with
every host-device sync removed:

- `should_remap` stays on the GPU; there is no early exit;
- boolean indexing becomes a dense `scatter_` into an `(R, max_blocks + 1)`
  buffer that sends unkept blocks to a sentinel column, then `torch.where` to
  preserve non-remapped rows exactly;
- `seq_lens` is computed with `torch.where` and `copy_`;
- only the **R active rows** of the shared mask are aggregated. Scanning the
  whole `(max_num_seqs, max_model_len)` allocation costs about 2 ms per step at
  262K and 7 ms at 1M with vLLM's default 1024 sequences, paid even when nothing
  is pruned; it once showed up as an intercept shift in time-versus-bytes fits
  and made one model look net-negative.

The two are asserted bit-identical across a randomized sweep of batch sizes,
block sizes, sequence lengths and prune patterns, including R = 0, narrow block
tables and sequences longer than `max_model_len`.

## CUDA graphs

Anything the patch touches must keep a stable pointer across replays: `seq_lens`
goes through a shape-keyed scratch cache and `block_table` is mutated in place.
Env-gated behaviour is read once, at install time — a replayed graph will not
re-read an environment variable. Python monkeypatches on kernels are bypassed
inside a replayed graph, so kernel capture for microbenchmarks needs eager mode.


## Verified against vLLM 0.20.2

Everything in this document about vLLM was read off the real
`vllm==0.20.2` wheel, not inferred. The files that matter:

| What | Where in vLLM |
| --- | --- |
| `LogitsProcessor` ABC, `BatchUpdate`, `MoveDirectionality` | `vllm/v1/sample/logits_processor/interface.py` |
| Custom-processor loading | `vllm/v1/sample/logits_processor/__init__.py` |
| `build()` signature, `FlashAttentionMetadata` | `vllm/v1/attention/backends/flash_attn.py` |
| `TritonAttentionMetadata`, capture override | `vllm/v1/attention/backends/triton_attn.py` |
| `CommonAttentionMetadata`, `AttentionMetadataBuilder` | `vllm/v1/attention/backend.py` |
| Per-group metadata construction | `vllm/v1/worker/gpu_model_runner.py` (~2170-2315) |
| KV-cache spec hierarchy | `vllm/v1/kv_cache_interface.py` |
| The Triton attention kernel | `vllm/v1/attention/ops/triton_unified_attention.py` |

Confirmed as the guide describes:

- `build(self, common_prefix_len, common_attn_metadata, fast_build=False)` on
  both builders, and `self.block_size` comes from the builder's own
  `kv_cache_spec` — so using a cached global really would use the wrong one.
- Both metadata classes are plain mutable `@dataclass`es carrying
  `seq_lens`, `block_table`, `query_start_loc`, `max_seq_len` and
  `scheduler_metadata` under exactly those names.
- The model runner builds one `CommonAttentionMetadata` and then does
  `cm = copy(cm_base)` per KV-cache group, replacing **only**
  `block_table_tensor` and `slot_mapping`. `seq_lens` is therefore literally
  the same tensor object across groups — shrinking it in place for the
  full-attention group is exactly the Gemma corruption the guide describes.
- `BatchUpdate` carries `removed`/`added`/`moved` and documents that
  operations are processed in that order, and that `output_tok_ids` "is a
  reference to the request's running output tokens list".
- `SamplingParams.extra_args` exists; over HTTP it arrives as `vllm_xargs`.

Corrected against it — four things this integration had wrong:

1. **A custom logits processor is addressed as `module:qualname`.** The loader
   does `logitproc.split(":")` and unpacks two names, then checks
   `issubclass(obj, LogitsProcessor)`. A dotted path raises before the engine
   finishes starting.
2. **`FlashAttentionMetadataBuilder.supports_update_block_table` is `True`.**
   The runner caches one build per `(kv_cache_spec, builder type)` and hands
   later groups a `copy.copy` with only the block table swapped — which would
   pair the first group's *compacted* `seq_lens` with an *uncompacted* block
   table. The patch turns that reuse off on the builders it hooks and restores
   it on uninstall.
3. **CUDA graph capture runs through `build()`.** Triton's
   `build_for_cudagraph_capture` then does `attn_metadata.seq_lens.fill_(1)`
   in place, which would write into the DA scratch buffer. The patch marks the
   capture window and skips the remap inside it.
4. **The Triton kernels were unified.** There is one
   `kernel_unified_attention` in `vllm/v1/attention/ops/`, not the 2D/3D pair,
   and the decode (split-KV) path is the 3-element-grid launch. `num_stages` is
   indeed never passed, so the guide's tweak still applies — to that launch only.

Also corrected, from the spec hierarchy: asking "is this a sliding-window
spec?" is the wrong polarity. `SlidingWindowSpec` has `sliding_window`,
`ChunkedLocalAttentionSpec` has `attention_chunk_size`, `MambaSpec` has
neither — so the question answers *no* for any rotating cache it has not heard
of. The patch now masks only a spec it can positively identify as
`FullAttentionSpec`.

`tests/fake_vllm.py` mirrors all of the above, so those tests check the
integration against vLLM's actual contract rather than against an assumption.
What they still cannot check is behaviour: no kernel runs, no CUDA graph is
captured, no engine starts. Validation checklist items 1, 3 and 4 remain the
only things that prove the mask does what it claims on real hardware.
