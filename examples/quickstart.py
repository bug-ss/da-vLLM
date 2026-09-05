#!/usr/bin/env python3
"""Answer a question about a long document with DA, on a real GPU.

    python examples/quickstart.py --model Qwen/Qwen3.6-27B --context doc.txt \
        --question "When was Acme founded?"

Requires the pinned serving stack (``pip install -e '.[serve]'``) and a GPU.
For a no-GPU walk through the same machinery, run ``examples/offline_dryrun.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from da_vllm import DAConfig, DAEngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--context", required=True, help="path to the long document")
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--arm",
        default="da",
        choices=["da", "da_no_mask", "vanilla"],
        help="da_no_mask serves the identical prompt with the mask off",
    )
    parser.add_argument("--max-num-seqs", type=int, default=256)
    args = parser.parse_args()

    context = Path(args.context).read_text(encoding="utf-8")
    config = DAConfig(
        enabled=args.arm == "da",
        max_num_seqs=args.max_num_seqs,
    )

    # The engine installs the attention patch inside vLLM's EngineCore process
    # and allocates the shared mask on first use.
    with DAEngine(args.model, arm=args.arm, config=config) as engine:
        result = engine.answer(context, args.question)

    print(result.text)
    print("\n" + "=" * 60)
    print(f"answer            : {result.answer!r}")
    print(f"magic chunks      : {result.num_segments}")
    print(f"prompt tokens     : {result.prompt_tokens:,}")
    print(f"decode steps      : {result.decode_steps}")
    print(f"attended tokens   : {result.attended_tokens:,}")
    print(f"vanilla baseline  : {result.baseline_attended_tokens:,}")
    print(f"reduction         : {result.reduction_pct:+.1f}%")
    print(f"focus             : {result.focus_granted}/{result.focus_attempts} granted")
    if result.declines:
        print(f"declined because  : {result.declines}")
    if result.detection_failure:
        print(f"DETECTION FAILED  : {result.detection_failure} (ran GLOBAL throughout)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
