"""``da`` -- operational commands over the DA pipeline.

The commands that need a GPU say so when they fail; everything else
(``models``, ``config``, ``render``, ``segment``, ``validate``, ``prepare``,
``score``, ``roofline``, ``serve-command``) runs on a CPU box with only a
tokenizer.
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from pathlib import Path

from .config import DAConfig
from .eval.data import SOURCE_KEYS
from .eval.records import ARMS, read_records
from .eval.score import report
from .metrics.roofline import roofline_response
from .models import JUDGE_MODEL, get_model, registered_hub_ids


def _tokenizer(model: str, revision: str | None = None):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # a missing tokenizer is a setup problem, not a bug
        raise SystemExit(
            "this command needs a real tokenizer: pip install transformers "
            "(or the pinned stack, pip install -r requirements-serve.txt). "
            "To exercise the pipeline with no downloads, run "
            "examples/offline_dryrun.py instead."
        ) from exc
    return AutoTokenizer.from_pretrained(model, revision=revision)


def _read(path: str | None) -> str:
    return Path(path).read_text(encoding="utf-8") if path else ""


def _dump(obj) -> str:
    return json.dumps(obj, indent=2, default=lambda o: getattr(o, "__dict__", str(o)))


# -- inspection -------------------------------------------------------------


def cmd_models(args) -> int:
    for hub_id in registered_hub_ids():
        spec = get_model(hub_id)
        flag = "" if spec.geometry.verified else "  [geometry: PLACEHOLDER]"
        print(
            f"{hub_id:28s} type={spec.model_type:12s} ctx={spec.max_model_len:>7d} "
            f"limit={spec.effective_context_limit:>7d} "
            f"kv_bytes/tok={spec.geometry.global_kv_bytes_per_token}{flag}"
        )
    print(f"\njudge (fixed for every arm and model): {JUDGE_MODEL}")
    return 0


def cmd_config(args) -> int:
    print(_dump(DAConfig(enabled=True).to_dict()))
    return 0


def cmd_segment(args) -> int:
    from .segmenter import Segmenter, assert_lossless

    text = _read(args.context)
    segmenter = Segmenter(
        _tokenizer(args.model), target_tokens=args.target, max_tokens=args.cap
    )
    segments = segmenter.segment(text)
    for seg in segments:
        head = seg.text[:60].replace("\n", " ")
        print(f"{seg.index:>4d}  {seg.num_tokens:>6d} tok  [{seg.start}:{seg.end}]  {head}")
    if text.strip():
        assert_lossless(text, segments)
        print("\nlossless: segments reassemble the input exactly", file=sys.stderr)
    print(f"{len(segments)} segments", file=sys.stderr)
    return 0


def cmd_render(args) -> int:
    from .prompt import PromptRenderer

    renderer = PromptRenderer(_tokenizer(args.model), args.model)
    prompt = renderer.render(args.arm, _read(args.context), args.question)
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
    filler = _read(args.context) or "Paragraph one.\n\nParagraph two.\n\n" * 400
    renderer = PromptRenderer(tokenizer, args.model)
    results = round_trip(tokenizer, args.model, default_cases(filler), renderer=renderer)
    for r in results:
        status = "ok  " if r.ok else "FAIL"
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


def cmd_serve_command(args) -> int:
    from .serving import EngineOptions, vllm_serve_command

    config = DAConfig(
        enabled=args.arm == "da", max_model_len=get_model(args.model).max_model_len
    )
    options = EngineOptions(args.model, args.arm, config, max_num_seqs=args.max_num_seqs)
    argv, env = vllm_serve_command(options)
    for key in ("VLLM_CACHE_ROOT", "PYTHONPATH", "DA_VLLM_CONFIG",
                "VLLM_WORKER_MULTIPROC_METHOD"):
        if key in env:
            print(f"export {key}={shlex.quote(env[key])}")
    print(shlex.join(argv))
    return 0


# -- evaluation -------------------------------------------------------------


def cmd_prepare(args) -> int:
    from .eval.pipeline import prepare, read_examples, write_examples

    tokenizer = _tokenizer(args.model)
    examples = read_examples(args.examples)
    sources = args.sources.split(",") if args.sources else list(SOURCE_KEYS)
    prepared = prepare(
        examples,
        tokenizer=tokenizer,
        max_model_len=get_model(args.model).max_model_len,
        sources=sources,
        n=args.n,
        seed=args.seed,
    )
    flat = [ex for key in sources for ex in prepared[key]]
    for key in sources:
        print(f"{key:28s} {len(prepared[key]):>4d}")
    written = write_examples(args.out, flat)
    print(f"\nwrote {written} examples to {args.out}", file=sys.stderr)
    empty = [k for k in sources if not prepared[k]]
    if empty:
        print(f"WARNING: no examples survived filtering for {empty}", file=sys.stderr)
        return 1
    return 0


def cmd_run(args) -> int:
    from .eval.pipeline import RunSpec, read_examples, run_all, run_arm

    examples = read_examples(args.examples)
    sources = tuple(args.sources.split(",")) if args.sources else SOURCE_KEYS
    spec = RunSpec(
        model=args.model,
        output_dir=Path(args.out),
        sources=sources,
        arms=(args.arm,) if args.arm else ARMS,
        samples_per_source=args.n,
        max_tokens=args.max_tokens,
        max_num_seqs=args.max_num_seqs,
        batch_size=args.batch_size,
    )
    by_source: dict[str, list] = {key: [] for key in sources}
    for ex in examples:
        if ex.source in by_source:
            by_source[ex.source].append(ex)

    def progress(source, done, total):
        print(f"  {source}: {done}/{total}", file=sys.stderr)

    if args.arm:
        run_arm(spec, args.arm, by_source, progress=progress)
    else:
        run_all(spec, by_source, progress=progress)
    print(f"records in {spec.output_dir}", file=sys.stderr)
    return 0


def cmd_judge(args) -> int:
    from .eval.pipeline import judge_with_vllm, read_examples
    from .eval.records import write_records

    records = list(read_records(args.records))
    examples = read_examples(args.examples)
    judged = judge_with_vllm(records, examples, judge_model=args.judge_model)
    out = args.out or args.records
    write_records(out, judged)
    correct = sum(1 for r in judged if r.correct)
    print(f"{correct}/{len(judged)} correct; wrote {out}", file=sys.stderr)
    return 0


def cmd_score(args) -> int:
    paths = args.records if isinstance(args.records, list) else [args.records]
    records = [rec for path in paths for rec in read_records(path)]
    sources = args.sources.split(",") if args.sources else list(SOURCE_KEYS)
    print(_dump(report(records, model=args.model, sources=sources)))
    return 0


def cmd_roofline(args) -> int:
    spec = get_model(args.model)
    if not spec.geometry.verified and not args.allow_placeholder_geometry:
        print(
            f"{spec.hub_id}'s attention geometry is a PLACEHOLDER: nobody has "
            "verified its layer counts or head dims. Derive it from the live "
            "config with da_vllm.metrics.roofline.geometry_from_config, or pass "
            "--allow-placeholder-geometry and do not publish the number.",
            file=sys.stderr,
        )
        return 1
    r = roofline_response(
        geometry=spec.geometry,
        active_params=spec.active_params,
        attended_tokens=args.attended_tokens,
        decode_steps=args.decode_steps,
        allow_placeholder_geometry=args.allow_placeholder_geometry,
    )
    print(
        _dump(
            {
                "model": spec.hub_id,
                "geometry_source": spec.geometry.source,
                "matmul_s": r.matmul_s,
                "global_kv_s": r.global_kv_s,
                "local_s": r.local_s,
                "total_s": r.total_s,
                "note": "ceiling at MFU 0.40 / MBU 0.70, not a measurement",
            }
        )
    )
    return 0


# -- wiring -----------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="da", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true", help="log at INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="list the registered models")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("config", help="print the default DA config as JSON")
    p.set_defaults(func=cmd_config)

    p = sub.add_parser("segment", help="show how a document splits into magic chunks")
    p.add_argument("--model", required=True)
    p.add_argument("--context", required=True)
    p.add_argument("--target", type=int, default=2048)
    p.add_argument("--cap", type=int, default=2560)
    p.set_defaults(func=cmd_segment)

    p = sub.add_parser("render", help="render a prompt through the one renderer")
    p.add_argument("--model", required=True)
    p.add_argument("--arm", default="da", choices=list(ARMS))
    p.add_argument("--question", default="What is the answer?")
    p.add_argument("--context")
    p.add_argument("--fingerprint-only", action="store_true")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("validate", help="round-trip render/detect/parse checks")
    p.add_argument("--model", required=True)
    p.add_argument("--context", help="a context file to use as filler")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("serve-command", help="print the vllm serve argv for one arm")
    p.add_argument("--model", required=True)
    p.add_argument("--arm", default="da", choices=list(ARMS))
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.set_defaults(func=cmd_serve_command)

    p = sub.add_parser("prepare", help="filter and sample examples for one model")
    p.add_argument("--model", required=True)
    p.add_argument("--examples", required=True, help="input JSONL")
    p.add_argument("--out", required=True, help="output JSONL")
    p.add_argument("--sources", help="comma-separated explicit source list")
    p.add_argument("-n", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("run", help="generate and replay one arm (or all three)")
    p.add_argument("--model", required=True)
    p.add_argument("--examples", required=True, help="prepared JSONL")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--arm", choices=list(ARMS), help="omit to run all three in sequence")
    p.add_argument("--sources")
    p.add_argument("-n", type=int, default=128)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("judge", help="score records with the one fixed judge")
    p.add_argument("--records", required=True)
    p.add_argument("--examples", required=True)
    p.add_argument("--out", help="defaults to overwriting --records")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("score", help="recompute every number from raw records")
    p.add_argument("--records", required=True, nargs="+")
    p.add_argument("--model", required=True)
    p.add_argument("--sources", help="comma-separated explicit source list")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("roofline", help="roofline decode wall-time for one response")
    p.add_argument("--model", required=True)
    p.add_argument("--attended-tokens", type=int, required=True)
    p.add_argument("--decode-steps", type=int, required=True)
    p.add_argument("--allow-placeholder-geometry", action="store_true")
    p.set_defaults(func=cmd_roofline)

    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
