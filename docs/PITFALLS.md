# Every known way this breaks

Each row is something that actually went wrong — in the original work, or here.
The right-hand column is what stops it happening again.

Most of these are not crashes. They are things that look fine and are quietly
wrong, which is why so much of this codebase is checks and refusals.

## Silent no-ops — the worst class

| Failure | Prevention |
| --- | --- |
| Patch installed in the parent process, where vLLM V1 runs nothing | Installed from `DALogitsProcessor.__init__`, which vLLM constructs inside EngineCore (`masking/patch.py::install_patch`). Logs a **warning** naming the pid if no builder was patched, and another if no shared mask exists after 64 build calls |
| Attention workers under TP > 1 never construct the logits processor | `masking.worker_env()` ships `resources/sitecustomize.py` on `PYTHONPATH` with `VLLM_WORKER_MULTIPROC_METHOD=spawn`; `docs/ENVIRONMENT.md` states plainly that the mask itself is not broadcast to TP ranks |
| An "is this vanilla" shortcut keyed on config shape also matched DA traffic | DA is per-request opt-in via an explicit `da_enable` flag; nothing is inferred from prompt or config shape (`masking/logits_processor.py`, `validation/checks.py::assert_explicit_enable`) |
| A renamed config key was ignored and masking turned off | `DAConfig.from_dict` raises on any unknown key, nested config included (`test_config.py`) |
| A config key that exists but nothing reads — the same failure wearing a different hat | `question_header`, `tag_tail_slack` and the segment sizes are each read by the code that acts on them, with a test that changing the value changes the behaviour |
| vLLM rejects a custom logits processor that is not a subclass of its ABC, so the driver never runs | `DALogitsProcessor` resolves vLLM's `LogitsProcessor` as its base at import time, falling back to `object` only where vLLM is absent |
| Addressing the processor with a dotted path: vLLM splits the FQCN on `":"`, so `module.Class` fails to unpack and the engine never starts | `DA_LOGITS_PROCESSOR_FQCN` is `module:qualname`, and a test asserts the split yields exactly two names |
| The runner reusing a cached metadata build across KV-cache groups, pairing group 0's compacted `seq_lens` with group N's uncompacted block table | The patch clears `supports_update_block_table` on the builders it hooks (and restores it on uninstall), so every group rebuilds |
| CUDA graph capture baking a compacted block table, and Triton's capture path stamping `seq_lens.fill_(1)` into the DA scratch buffer | The patch marks the capture window and skips the remap inside it |
| Compacting a rotating cache the "is this sliding?" test has never heard of — `ChunkedLocalAttentionSpec` carries `attention_chunk_size`, not `sliding_window` | Inverted polarity: mask only a spec positively identified as `FullAttentionSpec` |
| A `num_stages` tweak aimed at a kernel that no longer exists, silently doing nothing | Targets `vllm/v1/attention/ops/triton_unified_attention:kernel_unified_attention`, gated on the 3-element decode grid, and warns loudly if the symbol is gone |
| `DAEngine.close()` finding no `shutdown()` and leaving the EngineCore subprocess to be reparented to PID 1 | `close()` walks to `llm.llm_engine.engine_core.shutdown()`, where the method actually lives, and warns if it finds nothing |
| A slot recycled from a DA request keeps a stale mask | Reset on remove, move and overwrite, **plus** a per-step sweep of every slot DA has ever touched that no longer holds a DA request |
| A 4D mask dropped by an inherited loss function; an assistant mask that fell through a `not None` guard | `training.require_mask` / `training.loss_positions` raise loudly instead of training on full attention |
| Two-column NLL checks agreed while the mask did nothing | Three-column parity, `v ~ d < dv`; `ParityResult.explain()` names the no-op signature outright (`validation/nll_parity.py`) |

## Masking and boundaries

| Failure | Prevention |
| --- | --- |
| Tokens generated after the mask was frozen fell out of it | The boundary sits at `prompt_len` and everything past it is kept unconditionally, so the row never needs updating as generation proceeds (`state_machine.build_mask`) |
| Context leaked into the always-attended sink because the prompt started with the user turn | The system turn is prepended on **every** render path, vanilla included, and a test asserts it per arm |
| The marker search ran over the model's own text | The `# Question` search is bounded to the prompt region and anchored with `rfind` (`detect.build_prompt_map`) |
| The sliding-window group was compacted, dereferencing rotated slots | `is_sliding_window_spec` short-circuits the wrapper before any remap; tested against both spec classes and an attribute fallback |
| The tail block was dropped and the model lost coherence | The boundary floors to a block edge, so the block holding the current token is always kept; asserted every step of the end-to-end decode loop |
| Compacting 32-token columns with 16-token addressing (first misdiagnosed as a RoPE issue) | The remap always uses the **calling builder's** `self.block_size`, never a cached global |
| Shrinking the group-shared `seq_lens` in place corrupted the sliding-window kernel — near-baseline NLL, garbage text | `seq_lens` is routed through a shape-keyed scratch with a stable pointer; a test asserts the shared tensor is untouched and the scratch pointer is stable across steps |
| Cascade attention assumes a shared prefix that per-request compaction destroys | `common_prefix_len` forced to 0 in the wrapper |

## Format detection

| Failure | Prevention |
| --- | --- |
| Inline dividers collided with document content; render and detect chose divider variants by different rules | Detection walks **turn and tool boundaries** — special tokens a model cannot emit into a body — and requires ids to be exactly 1..N |
| A non-greedy XML regex truncated segments whose bodies contained the closing tag, with contiguous ids so nothing flagged it | Detection is bounded to the context region and requires strict assistant-call / tool-response alternation; overlapping spans are a hard failure |
| Template-name routing by substring misrouted a format whose name contained another's | `models.get_model` / `get_model_by_type` are exact lookups that raise; `training.chat_template_override` keys on `model_type` |
| The tool-call literal differs between Qwen 3, 3.5 and 3.6 | Per-family regexes live in `FamilySpec`; `da validate` is what proves they match the tokenizer you actually serve |
| A detection failure aborted a whole run | Detection never raises: it returns a reason, focus is disabled for that request, and it runs GLOBAL for life |
| The Gemma 4 chat template dropped tool-call and tool-response messages | `da validate` fails if the segments rendered are not the segments detected; documented as the first thing to check |

## Prompt construction

| Failure | Prevention |
| --- | --- |
| Segments built by decoding token-id slices split multibyte characters (U+FFFD in ~a fifth of samples) | The segmenter works entirely in character-offset space and emits real substrings; `assert_lossless` also fails on introduced U+FFFD |
| Serve and replay differed by the tool-declaration block (~340 tokens on Qwen) | One `PromptRenderer` for every path, and `RenderedPrompt.fingerprint` makes a divergence a test failure |
| A hand-written tool declaration inlined as system text | The declaration goes through `apply_chat_template(tools=[...])` |
| Qwen templates need `arguments` as a dict, not a JSON string | Dict by default, with a per-family override flag; tested |
| Models do not follow the protocol inside thinking tags | Thinking off via per-family `chat_template_kwargs` |
| "Chunk 3" collided with a document's own "Section 3" | "Magic Chunk" everywhere |
| Removing the "1 to 12 words" bound brought back unclosed-focus loops; removing the strategy scaffolding collapsed structured tasks | A test asserts both survive in the rendered instruction |

## Evaluation

| Failure | Prevention |
| --- | --- |
| The dataset's stored token counts came from a different tokenizer and overflowed the engine cap | `prepare_source` takes a `count_tokens` callable — the served tokenizer — and recomputes; tested |
| Two similar source names were swapped, and stale result directories leaked into an average | `SOURCES` is the explicit list; a missing or unexpected source raises `MissingSourceError` rather than shrinking the mean |
| Macro and micro averages mixed across scripts | Everything is macro over sources and micro within, and the per-source table is returned alongside every headline |
| Judge identity was not stored with the verdicts | `Verdict.judge_model` is mandatory and lands on every record |
| A thinking judge capped at 4096 tokens truncated ~3.5% of verdicts | `max_tokens` 32768, and `Verdict.truncated` is set from the finish reason |
| Scoring each model with itself as judge reversed the format-versus-mask conclusion | One fixed judge (`models.JUDGE_MODEL`); `judge_messages` takes no target-model argument |
| Over-length rows truncated instead of dropped | Dropped, with a test |
| Format failures quietly excluded | Counted as wrong, without a judge call |
| Non-terminating responses inflated attended tokens (6% of responses on one model) | `report()` always returns as-logged **and** `decode_steps >= 8000` excluded |
| Numbers recomputed from cached summaries | `eval/score.py` reads only raw per-response records |
| A rubric whose authoring model was not recorded, so it cannot be audited | `generate_rubrics` stores the author on every rubric, and a failed generation is recorded as a failure rather than filled in |
| A synthetic question silently swapped onto a source that uses original QA | `attach_questions` raises unless the source is one of the four the paper marks synthetic |

## Cost model

| Failure | Prevention |
| --- | --- |
| A 2x byte error (Gemma's global head dim) produced a false finding, two theories, and a wrong "correction" | Byte counts derive from the live config (`global_kv_bytes_per_token`) or the checkpoint's tensor shapes (`global_kv_bytes_from_shapes`); a hybrid config missing `global_head_dim` raises rather than borrowing the sliding values; `verify_geometry` cross-checks the registry |
| A registry number nobody verified being published as if it were measured | Geometry carries a `source`; only the two headline models are `"measured"`. The cost model **refuses** a `"placeholder"` geometry, and `da models` flags them. Derive the real values with `geometry_from_config` |
| A fixed per-step remap tax read as a DA property, traced to scanning the whole mask allocation | `remap_optimized` aggregates only the R active rows |
| A macro roofline slope used as a per-kernel attention share | `roofline_response` returns the three terms separately and the docstring says it is a ceiling, not a measurement |
| "DA frees KV capacity" claimed and retracted | Stated in the architecture doc and the state-machine docstring: nothing is evicted, DA reduces bytes read, not bytes stored |
| Stochastic sampling made arms produce different-length outputs, so a length effect looked like a win | `timing.py` documents comparing per-step at fixed residency or forcing equal-length decode |
| Kept-fraction logging left on while timing | `log_kept_fraction` defaults off and both the config and the checklist say it costs a sync |
| Cross-boot variance larger than the effect | `StepToggle` A/Bs step by step inside one boot |
| Token-strided synthetic masks vanished after outward rounding | `block_aligned_synthetic_mask` defines masks on whole blocks |
| Generated-token identity used as a remap equivalence test | Bit-identity on `block_table` and `seq_lens` across a randomized sweep |

## Infrastructure

| Failure | Prevention |
| --- | --- |
| Orphaned EngineCore processes reparented to PID 1 and held VRAM | `serving.engine_process` starts a new session and kills the whole group |
| vLLM's warmup and profile run call the builder before any real request | Tolerated by construction, with a test |
| Auto-sizing `max_num_seqs` from VRAM under-counted hybrid KV capacity 4-6x | Static `max_num_seqs=256`; vLLM's admission control does the limiting |
| `num_gpu_blocks * block_size` read as KV capacity on a hybrid model | `describe_kv_capacity` returns the numbers with the caveat attached |
| `cache_config.block_size` read on a hybrid model (it reports the recurrent page size) | The block size is learned from the first full-attention builder call and prefers it over any sliding value |
| An AOT compile cache built with patches active reused by an unpatched run | A distinct `VLLM_CACHE_ROOT` per (model, arm, engine settings, DA config) |
| No HuggingFace revision pinned | `_load_tokenizer` passes `tokenizer_revision`; the environment doc says to pin |

## Runaway generation

| Failure | Prevention |
| --- | --- |
| A repetition penalty, ignore-EOS, a lower token cap, or n-gram detection — all change non-runaway outputs or false-fire on legitimate enumerations | The OR of three distinct-ratio / no-answer signals (`runaway.py`), with a test for the enumeration false-positive |
| A 250-token sustain calibrated on three sources fired on real answers | 2,700 tokens, calibrated so the longest legitimate low-novelty run (2,349) passes; both are asserted |
| Forcing EOS while claiming argmax invariance, so vLLM skipped `apply()` | `is_argmax_invariant()` returns False whenever the guard is on, and the processor refuses to start if the tokenizer reports no EOS id |

## Training (not part of the paper's results)

| Failure | Prevention |
| --- | --- |
| FlexAttention backward is wrong in bf16 when seqlen is not a multiple of 128 with a block mask plus a score modifier (pytorch#153799) | `pad_to_flex_multiple`; `compile_block_mask` refuses an unpadded length and names the issue |
| Gemma 4's head dim 256 needs ~196 KB of shared memory; Inductor's defaults crash an A100 | `flex_kernel_options` dispatches on head dim and the device's opt-in limit |
| Gemma 4 ships as a multimodal wrapper | `load_text_only_model` unwraps it and propagates the model name |
| Train and serve masks at different KV granularity | `allowed_block_table` takes the served block size; the docstring calls a mismatch a known approximation |
| Dense 4D masks do not scale | The trace compiles to a FlexAttention block mask, one row pattern per mode group |
| An assistant mask that did not cover the synthetic tool turns correctly | The prompt is rendered with `tools=` through the same renderer; loss lands only past `prompt_len`, asserted |
