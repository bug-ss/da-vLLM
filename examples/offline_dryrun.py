#!/usr/bin/env python3
"""Run the whole DA pipeline with no GPU, no model download, no network.

    python examples/offline_dryrun.py

Uses the offline tokenizer in ``da_vllm.testing`` and a scripted response, so
what you see is the real segmenter, the real renderer, the real detector, the
real state machine, the real mask writer, the real block-table remap and the
real accounting -- only the model is stubbed.

Good for: verifying an install, seeing what the prompt looks like, watching the
block table shrink when the model declares ``<focus>``, and sanity-checking a
change before you spend GPU time on it.
"""

from __future__ import annotations

import math
import textwrap

import torch

from da_vllm import DAConfig, DAEngine
from da_vllm.detect import build_prompt_map
from da_vllm.masking.remap import remap_readable
from da_vllm.masking.shared import SharedMaskStore
from da_vllm.models import get_model
from da_vllm.state_machine import DAStateMachine, Mode
from da_vllm.testing import lorem, qwen_tokenizer

MODEL = "Qwen/Qwen3.6-27B"
BLOCK_SIZE = 16

QUESTION = "In what year was Acme Corp founded?"
RESPONSE = (
    "<global>The company history should be in magic chunk 2.</global>\n"
    '<focus magic_chunks="2">Acme Corp was founded in 2003</focus>\n'
    "<local>The question asks for the founding year: 2003.</local>\n"
    "<answer>2003</answer>"
)


def rule(title: str) -> None:
    print(f"\n\033[1m{title}\033[0m\n" + "-" * len(title))


def main() -> int:
    tokenizer = qwen_tokenizer()
    # Small segments so a short document still produces several magic chunks.
    config = DAConfig(
        enabled=True,
        max_model_len=16384,
        max_num_seqs=4,
        segment_target_tokens=200,
        segment_max_tokens=250,
    )
    document = lorem(30, seed=3)

    def generate_fn(token_id_lists, params):
        ids = tokenizer.encode(RESPONSE, add_special_tokens=False)
        return [(RESPONSE, ids, "stop") for _ in token_id_lists]

    engine = DAEngine(
        MODEL,
        config=config,
        tokenizer=tokenizer,
        generate_fn=generate_fn,
        max_tokens=1024,
    )

    # -- 1. what the model is shown ------------------------------------------
    prompt = engine.render(document, QUESTION)
    rule("1. Prompt")
    print(f"{prompt.num_segments} magic chunks, fingerprint {prompt.fingerprint[:16]}")
    head = prompt.text[:320].replace("\n", "\\n")
    print(f"head: {head}...")

    # -- 2. what the server derives from it ----------------------------------
    prompt_ids = tokenizer.encode(prompt.text, add_special_tokens=False)
    pmap = build_prompt_map(
        tokenizer, prompt.text, get_model(MODEL).family, config,
        prompt_token_ids=prompt_ids,
    )
    rule("2. Segment map (derived server-side from turn and tool boundaries)")
    print(f"prompt tokens      : {pmap.num_prompt_tokens}")
    print(f"attention sink     : [0, {pmap.sink_end})")
    print(f"local window       : [{pmap.local_window_start}, {pmap.num_prompt_tokens})")
    print(f"segments detected  : {len(pmap.segments)} (failure: {pmap.failure_reason})")
    for span in pmap.segments[:3]:
        print(f"  chunk {span.index}: tokens [{span.token_start}, {span.token_end})")
    if len(pmap.segments) > 3:
        print(f"  ... and {len(pmap.segments) - 3} more")

    # -- 3. what the state machine does to the mask --------------------------
    rule("3. Mask per mode")
    for mode, ids in ((Mode.GLOBAL, ()), (Mode.FOCUS, (2,)), (Mode.LOCAL, ())):
        from da_vllm.state_machine import build_mask

        snap = build_mask(pmap, mode, ids)
        pct = 100 * snap.kept_prompt_tokens / pmap.num_prompt_tokens
        print(f"{mode.value:>7}: {snap.kept_prompt_tokens:>6d} prompt tokens kept ({pct:5.1f}%)")

    # -- 4. what the kernel actually reads, step by step ---------------------
    rule("4. Block table, one decode step at a time")
    store = SharedMaskStore(config.max_num_seqs, config.max_model_len, "cpu")
    sm = DAStateMachine(pmap, tokenizer, config)
    response_ids = tokenizer.encode(RESPONSE, add_special_tokens=False)
    physical = torch.arange(
        1000, 1000 + math.ceil((len(prompt_ids) + len(response_ids)) / BLOCK_SIZE) + 1,
        dtype=torch.int32,
    )
    last_mode, rows = None, []
    for step in range(len(response_ids)):
        if sm.advance(response_ids[: step + 1]):
            store.write_snapshot(0, sm.snapshot(), block_size=BLOCK_SIZE)
        seq_len = len(prompt_ids) + step + 1
        n_blocks = math.ceil(seq_len / BLOCK_SIZE)
        table = physical[:n_blocks].clone().unsqueeze(0)
        out = torch.zeros(1, dtype=torch.int32)
        remap_readable(
            mask=store.tensor,
            block_table=table,
            seq_lens=torch.tensor([seq_len], dtype=torch.int32),
            out_seq_lens=out,
            query_lens=torch.ones(1, dtype=torch.int32),
            block_size=BLOCK_SIZE,
        )
        kept = math.ceil(int(out[0]) / BLOCK_SIZE)
        if sm.mode is not last_mode:
            rows.append((step, sm.mode.value, kept, n_blocks))
            last_mode = sm.mode
    print(f"{'step':>5}  {'mode':>7}  {'blocks read':>11}  {'of':>5}  saving")
    for step, mode, kept, total in rows:
        print(f"{step:>5}  {mode:>7}  {kept:>11d}  {total:>5d}  {100 * (1 - kept / total):5.1f}%")

    # -- 5. the accounting the evaluation would report ------------------------
    result = engine.answer(document, QUESTION)
    rule("5. Accounting")
    print(f"answer             : {result.answer!r}")
    print(f"decode steps       : {result.decode_steps}")
    print(f"attended tokens    : {result.attended_tokens:,}")
    print(f"vanilla baseline   : {result.baseline_attended_tokens:,}")
    print(f"reduction          : {result.reduction_pct:+.1f}%")
    print(f"focus granted      : {result.focus_granted}/{result.focus_attempts}")
    print(f"mode steps         : {result.mode_steps}")

    print(
        textwrap.dedent(
            """
            Note: the kernel-level saving above is larger than the response-level
            reduction, because the response spends its first steps in global mode
            and the reduction is averaged over every step.  Both are real; the
            evaluation reports the second.
            """
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
