"""``da`` -- small operational commands over the DA pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import DAConfig
from .eval.data import SOURCE_KEYS
from .eval.records import read_records
from .eval.score import report
from .metrics.roofline import roofline_response
from .models import get_model, registered_hub_ids


def _tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def cmd_models(args) -> int:
    for hub_id in registered_hub_ids():
        spec = get_model(hub_id)
        print(
            f"{hub_id:28s} type={spec.model_type:12s} ctx={spec.max_model_len:>7d} "
            f"limit={spec.effective_context_limit:>7d} "
            f"kv_bytes/tok={spec.geometry.global_kv_bytes_per_token}"
        )
    return 0


def cmd_render(args) -> int:
    from .prompt import PromptRenderer

    context = Path(args.context).read_text(encoding="utf-8") if args.context else ""
    renderer = PromptRenderer(_tokenizer(args.model), args.model)
    prompt = renderer.render(args.arm, context, args.question)
    if args.fingerprint_only:
        print(prompt.fingerprint)
    else:
        print(prompt.text)
    print(
        f"\n--- {prompt.num_segments} segments, fingerprint {prompt.fingerprint[:16]}",
        file=sys.stderr,
    )
    return 0


def cmd_validate(args) -> int:
    from .prompt import PromptRenderer
    from .validation.checks import assert_round_trip, default_cases, round_trip

    tokenizer = _tokenizer(args.model)
    filler = Path(args.context).read_text(encoding="utf-8") if args.context else (
        "Paragraph one.\n\nParagraph two.\n\n" * 400
    )
    renderer = PromptRenderer(tokenizer, args.model)
    results = round_trip(tokenizer, args.model, default_cases(filler), renderer=renderer)
    for r in results:
        status = "ok " if r.ok else "FAIL"
        print(
            f"{status} {r.name:28s} segments {r.num_segments:>3d}->{r.detected_segments:<3d} "
            f"focus={r.focus_parsed} {r.failure_reason or ''} {' '.join(r.notes)}"
        )
    try:
        assert_round_trip(results)
    except AssertionError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1
    print("\nall round-trip cases passed")
    return 0


def cmd_score(args) -> int:
    records = list(read_records(args.records))
    sources = args.sources.split(",") if args.sources else list(SOURCE_KEYS)
    out = report(records, model=args.model, sources=sources)
    print(json.dumps(out, indent=2, default=lambda o: getattr(o, "__dict__", str(o))))
    return 0


def cmd_roofline(args) -> int:
    spec = get_model(args.model)
    r = roofline_response(
        geometry=spec.geometry,
        active_params=spec.active_params,
        attended_tokens=args.attended_tokens,
        decode_steps=args.decode_steps,
    )
    print(
        json.dumps(
            {
                "model": spec.hub_id,
                "matmul_s": r.matmul_s,
                "global_kv_s": r.global_kv_s,
                "local_s": r.local_s,
                "total_s": r.total_s,
                "note": "ceiling at MFU 0.40 / MBU 0.70, not a measurement",
            },
            indent=2,
        )
    )
    return 0


def cmd_config(args) -> int:
    print(json.dumps(DAConfig(enabled=True).to_dict(), indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="da", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="list the registered models")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("render", help="render a prompt through the one renderer")
    p.add_argument("--model", required=True)
    p.add_argument("--arm", default="da", choices=["da", "da_no_mask", "vanilla"])
    p.add_argument("--question", default="What is the answer?")
    p.add_argument("--context", help="path to a context file")
    p.add_argument("--fingerprint-only", action="store_true")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("validate", help="round-trip render/detect/parse checks")
    p.add_argument("--model", required=True)
    p.add_argument("--context", help="path to a context file used as filler")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("score", help="recompute every number from raw records")
    p.add_argument("--records", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--sources", help="comma-separated explicit source list")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("roofline", help="roofline decode wall-time for one response")
    p.add_argument("--model", required=True)
    p.add_argument("--attended-tokens", type=int, required=True)
    p.add_argument("--decode-steps", type=int, required=True)
    p.set_defaults(func=cmd_roofline)

    p = sub.add_parser("config", help="print the default DA config as JSON")
    p.set_defaults(func=cmd_config)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
