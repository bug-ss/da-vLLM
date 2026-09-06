#!/usr/bin/env python3
"""Ask a DA-enabled vLLM server on another machine.

    python examples/remote_client.py \
        --url http://gpu-box:8000 \
        --model google/gemma-4-31B-it \
        --context long_document.txt \
        --question "When was Acme founded?"

Needs only the model's tokenizer locally -- no GPU, no weights.

The server must have been started with DA switched on. Print the command with:

    da serve-command --model google/gemma-4-31B-it --arm da
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from da_vllm import DAClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="e.g. http://gpu-box:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", required=True, help="path to the document")
    parser.add_argument("--question", action="append", required=True,
                        help="repeatable; several questions share a batch")
    parser.add_argument("--api-key")
    parser.add_argument("--served-model-name",
                        help="if the server was started under a different name")
    parser.add_argument("--arm", default="da", choices=["da", "da_no_mask", "vanilla"])
    args = parser.parse_args()

    document = Path(args.context).read_text(encoding="utf-8")
    client = DAClient(
        args.url,
        args.model,
        arm=args.arm,
        api_key=args.api_key,
        served_model_name=args.served_model_name,
    )

    # Is the server up and serving this model?
    print(json.dumps(client.check(), indent=2))
    print()

    results = client.answer_batch([(document, q) for q in args.question])

    total_read = total_normal = 0
    for question, r in zip(args.question, results):
        print(f"Q: {question}")
        print(f"A: {r.answer!r}")
        print(f"   chunks in document : {r.num_segments}")
        print(f"   words written      : {r.decode_steps}")
        print(f"   reading done       : {r.attended_tokens:,}")
        print(f"   reading if normal  : {r.baseline_attended_tokens:,}")
        print(f"   difference         : {r.reduction_pct:+.1f}%")
        if r.declines:
            print(f"   !! focus refused   : {r.declines}")
        if r.detection_failure:
            print(f"   !! DA was OFF      : {r.detection_failure}")
        print()
        total_read += r.attended_tokens
        total_normal += r.baseline_attended_tokens

    if total_normal:
        saved = (total_read - total_normal) / total_normal * 100
        print(f"across all {len(results)} questions: {saved:+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
