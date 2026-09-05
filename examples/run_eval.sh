#!/usr/bin/env bash
# The full evaluation, one arm per process (guide 11/12: an orphaned EngineCore
# reparents to PID 1 and holds VRAM across arms, and each arm needs its own
# compile cache).
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.6-27B}"
EXAMPLES="${EXAMPLES:-data/examples.jsonl}"   # one JSON object per line; see
                                              # da_vllm/eval/pipeline.py
OUT="${OUT:-runs/$(echo "$MODEL" | tr '/' '_')}"

# 0. Check the prompt renders and the server detects what the renderer emitted.
#    If this fails, nothing downstream is worth running.
da validate --model "$MODEL"

# 1. Filter, dedupe, sample 128 per source with the SERVED model's tokenizer.
da prepare --model "$MODEL" --examples "$EXAMPLES" --out "$OUT/prepared.jsonl"

# 2. Generate. One arm per process. The no-mask arm is what separates the cost
#    of the prompt format from the cost of the mask -- do not skip it.
for ARM in vanilla da_no_mask da; do
  da run --model "$MODEL" --examples "$OUT/prepared.jsonl" --out "$OUT" --arm "$ARM"
done

# 3. Judge every arm with the one fixed judge. Scoring each model with itself
#    as judge reversed the format-versus-mask conclusion.
for ARM in vanilla da_no_mask da; do
  da judge --records "$OUT/records-$ARM.jsonl" --examples "$OUT/prepared.jsonl"
done

# 4. Recompute every number from the raw records, against the explicit source
#    list. Never from a cached summary.
da score --model "$MODEL" --records "$OUT"/records-*.jsonl | tee "$OUT/report.json"
