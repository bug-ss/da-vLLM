# Versions, models and GPU setup

The exact versions the published results came from, plus the model-specific
facts you need if you are adding a model.

| Component | Version |
| --- | --- |
| Python | 3.13 |
| torch | 2.11.0 (CUDA 13 wheels: cudnn 9.19, nccl 2.28.9, cublas 13.1) |
| vLLM | 0.20.2 |
| transformers | 5.5.4 |
| triton | 3.6.0 |
| flashinfer-python | 0.6.8.post1 (pulled by vLLM, not used for attention) |
| flash-linear-attention | 0.5.0 (Qwen 3.5/3.6 Gated DeltaNet layers) |
| causal-conv1d | 1.6.1 (build with no build isolation) |
| deep-gemm | git tag v2.1.1.post3 |
| datasets | 3.6.0 |

Notes:

- There is **no separate `flash-attn` wheel**. The FlashAttention path is
  vLLM's bundled `FLASH_ATTN` backend; the Triton path is vLLM's `TRITON_ATTN`
  backend.
- GPU: NVIDIA B200 for every evaluation and efficiency number. One engine per
  GPU, tensor parallel 1, data parallel by running one engine per GPU.
- bf16 weights and bf16 KV cache throughout.
- One model (Gemma-4-12B) ran on a ported vLLM 0.23.0 stack; equivalence of the
  two stacks was checked on a smaller Gemma model only.
- **Pin HuggingFace revisions.** The original work did not, which makes its
  numbers unreproducible in principle. `AutoTokenizer.from_pretrained(...,
  revision=...)` and the same for the model.
- The Gemma 4 chat template shipped with the tokenizer initially **dropped
  tool-call and tool-response messages**. Before anything else, check the
  rendered prompt actually contains the tool turns:

  ```bash
  da render --model google/gemma-4-31B-it --arm da --context doc.txt \
      --question "When?" | grep -c "Magic Chunk"
  ```

  `da validate --model <model>` does this and more; it fails loudly if the
  segments a prompt renders are not the segments the server detects.

## Model registry

`da_vllm.models` holds the per-model facts. Lookup is exact — by hub id, or by
`config.model_type` for local checkpoints — and an unregistered model raises
rather than defaulting to another family's sampling parameters or chat template.

| Model | Layers | Global attention layers | Other layers | Global KV bytes/token (bf16) |
| --- | --- | --- | --- | --- |
| Gemma-4-31B | 60 | 10, 4 KV heads, head dim 512 | 50 sliding window (1024), 16 KV heads, head dim 256 | 81,920 |
| Qwen-3.6-27B | 64 | 16, 4 KV heads, head dim 256 | 48 Gated DeltaNet (recurrent state, no KV cache) | 65,536 |

**Only these two models' geometry is published.** The four size-scaling models
carry `source="placeholder"` in the registry, `da models` flags them, and the
cost model refuses them outright:

```
$ da roofline --model Qwen/Qwen3.5-4B --attended-tokens 1000 --decode-steps 10
Qwen/Qwen3.5-4B's attention geometry is a PLACEHOLDER: nobody has verified its
layer counts or head dims. Derive it from the live config with
da_vllm.metrics.roofline.geometry_from_config, ...
```

Derive the real values before you publish anything about those models:

```python
from transformers import AutoConfig
from da_vllm.metrics.roofline import geometry_from_config
from da_vllm.models import get_model

spec = get_model("Qwen/Qwen3.5-9B")
geometry = geometry_from_config(spec, AutoConfig.from_pretrained(spec.hub_id))
```

Gemma's global layers use `global_head_dim` and `num_global_key_value_heads` —
**not** the sliding-window layers' values. Applying the sliding geometry to the
global layers doubles the byte count; that exact error produced a false finding
that took weeks to retract. `da_vllm.metrics.roofline.global_kv_bytes_per_token`
refuses to fall back to the sliding values on a hybrid config, and
`verify_geometry` cross-checks the registry against the live config.

## KV block size

vLLM's hybrid KV-cache manager chooses the block size per layer type at engine
start. **Do not read `cache_config.block_size` on hybrid models** — it reports
the recurrent-state page size. Observed values:

| Model | Sliding-window block | Full-attention block |
| --- | --- | --- |
| Qwen3.5-4B / 9B / 27B, Qwen3.6-27B | n/a | 16 |
| Qwen3.5-2B, Qwen3.5-35B-A3B | n/a | 32 |
| Gemma-4-E2B / E4B | 32 | 16 |
| Gemma-4-26B-A4B / 31B | 16 | 32 |

The patch **learns** the block size from the first metadata-builder call for the
full-attention group and prefers it over any sliding-window value already seen
(vLLM's warmup calls `build` for every group before the first real step). The
table above is documentation, not a source of truth: nothing in the code reads
it.

## Attention backends

vLLM decides the backend from the architecture. Qwen dense and MoE models route
to FlashAttention. Gemma 4 ships as a multimodal wrapper class that
FlashAttention refuses to bind, so vLLM routes it to the Triton unified
attention kernel. **Both builders are hooked** — patching only one leaves half
the fleet unmasked and silent.

## Compile caches

vLLM's AOT compile cache key does not include the DA patch state, so a cache
built with patches active is incompatible with an unpatched run of the same
model. `EngineOptions.resolved_cache_root()` derives a distinct
`VLLM_CACHE_ROOT` per (model, arm, engine settings, DA config) so the three arms
can never share one.

## Tensor parallel > 1

vLLM V1 runs the model in an EngineCore subprocess, and with tensor parallel > 1
the attention workers are separately spawned processes that never construct the
logits processor. `da_vllm.masking.worker_env(config)` returns the environment
that fixes this: the shipped `sitecustomize.py` on `PYTHONPATH`, the config in
`DA_VLLM_CONFIG`, and `VLLM_WORKER_MULTIPROC_METHOD=spawn`.

The shared mask tensor itself is **not** broadcast to TP ranks. Every number in
the paper came from tensor parallel 1 with one engine per GPU. Under TP > 1 the
patch installs but finds no mask in the worker, logs that loudly after warmup,
and serves unmasked.
