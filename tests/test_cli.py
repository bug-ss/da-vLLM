"""The CLI, exercised offline by swapping in the packaged tokenizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from da_vllm import cli
from da_vllm.eval.data import SOURCE_KEYS
from da_vllm.testing import lorem, qwen_tokenizer

MODEL = "Qwen/Qwen3.6-27B"


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    tok = qwen_tokenizer()
    monkeypatch.setattr(cli, "_tokenizer", lambda model, revision=None: tok)
    return tok


@pytest.fixture
def context_file(tmp_path):
    path = tmp_path / "doc.txt"
    path.write_text(lorem(120, seed=4), encoding="utf-8")
    return path


def test_models_lists_the_registry_and_flags_placeholders(capsys):
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "google/gemma-4-31B-it" in out
    assert "PLACEHOLDER" in out  # the size-scaling models
    assert "Qwen/Qwen3.5-4B" in out


def test_config_prints_valid_json(capsys):
    assert cli.main(["config"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sink_tokens"] == 16 and payload["enabled"] is True


def test_segment_reports_a_lossless_split(context_file, capsys):
    assert cli.main(["segment", "--model", MODEL, "--context", str(context_file),
                     "--target", "2048", "--cap", "2560"]) == 0
    captured = capsys.readouterr()
    assert "lossless" in captured.err
    assert captured.out.strip().splitlines()[0].strip().startswith("1")


def test_render_emits_the_prompt_and_a_fingerprint(context_file, capsys):
    assert cli.main(["render", "--model", MODEL, "--arm", "da",
                     "--context", str(context_file), "--question", "When?"]) == 0
    captured = capsys.readouterr()
    assert "Magic Chunk 1" in captured.out and "# Question" in captured.out
    assert "segments, fingerprint" in captured.err


def test_validate_passes_the_round_trip(context_file, capsys):
    assert cli.main(["validate", "--model", MODEL, "--context", str(context_file)]) == 0
    assert "all round-trip cases passed" in capsys.readouterr().out


def test_roofline_refuses_placeholder_geometry_by_default(capsys):
    assert cli.main(["roofline", "--model", "Qwen/Qwen3.5-4B",
                     "--attended-tokens", "1000", "--decode-steps", "10"]) == 1
    assert "PLACEHOLDER" in capsys.readouterr().err
    assert cli.main(["roofline", "--model", MODEL,
                     "--attended-tokens", "1000", "--decode-steps", "10"]) == 0
    assert json.loads(capsys.readouterr().out)["geometry_source"] == "measured"


def test_serve_command_is_copy_pasteable(capsys):
    assert cli.main(["serve-command", "--model", MODEL, "--arm", "da"]) == 0
    out = capsys.readouterr().out
    assert "vllm serve Qwen/Qwen3.6-27B" in out
    assert "--logits-processors da_vllm.masking.logits_processor:DALogitsProcessor" in out
    assert "export DA_VLLM_CONFIG=" in out
    assert "export VLLM_CACHE_ROOT=" in out
    # A plain (no-mask) arm must not register the processor.
    cli.main(["serve-command", "--model", MODEL, "--arm", "vanilla"])
    assert "--logits-processors" not in capsys.readouterr().out


def _write_examples(path: Path, n_per_source=2, sources=SOURCE_KEYS[:2], paragraphs=200):
    with path.open("w", encoding="utf-8") as fh:
        for source in sources:
            for i in range(n_per_source):
                fh.write(
                    json.dumps(
                        {
                            "example_id": f"{source}:{i}",
                            "source": source,
                            "context": lorem(paragraphs, seed=i),
                            "question": "What is the value?",
                            "reference_answer": "the value",
                            "rubric": "- CORRECT if the response states the value",
                        }
                    )
                    + "\n"
                )


def test_prepare_filters_and_samples(tmp_path, capsys):
    src = tmp_path / "examples.jsonl"
    out = tmp_path / "prepared.jsonl"
    _write_examples(src)
    code = cli.main([
        "prepare", "--model", MODEL, "--examples", str(src), "--out", str(out),
        "--sources", ",".join(SOURCE_KEYS[:2]), "-n", "1",
    ])
    assert code == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 2  # one per source
    assert {r["source"] for r in rows} == set(SOURCE_KEYS[:2])
    assert all(r["context_tokens"] >= 4096 for r in rows)


def test_prepare_reports_a_source_that_filtered_to_nothing(tmp_path, capsys):
    src = tmp_path / "examples.jsonl"
    _write_examples(src, paragraphs=2)  # far below the 4096-token floor
    code = cli.main([
        "prepare", "--model", MODEL, "--examples", str(src),
        "--out", str(tmp_path / "p.jsonl"), "--sources", ",".join(SOURCE_KEYS[:2]),
    ])
    assert code == 1
    assert "no examples survived" in capsys.readouterr().err


def test_score_reads_only_raw_records(tmp_path, capsys):
    from da_vllm.eval.records import ResponseRecord, write_records

    records = [
        ResponseRecord(
            run_id="r", model=MODEL, arm=arm, source=source, example_id=f"{source}:{i}",
            prompt_fingerprint="fp", prompt_tokens=1000, response_text="t",
            decode_steps=10, attended_tokens=(500 if arm == "da" else 1000),
            correct=True, answer_text="x",
        )
        for arm in ("vanilla", "da_no_mask", "da")
        for source in SOURCE_KEYS
        for i in range(2)
    ]
    path = tmp_path / "records.jsonl"
    write_records(path, records)
    assert cli.main(["score", "--model", MODEL, "--records", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["as_logged"]["accuracy"]["da"] == 100.0
    assert payload["as_logged"]["attended_tokens"]["da"] == 500.0


def test_score_refuses_a_partial_source_list(tmp_path):
    from da_vllm.eval.records import ResponseRecord, write_records
    from da_vllm.eval.score import MissingSourceError

    records = [
        ResponseRecord(
            run_id="r", model=MODEL, arm="da", source=SOURCE_KEYS[0],
            example_id="a", prompt_fingerprint="fp", prompt_tokens=1,
            response_text="t", decode_steps=1, attended_tokens=1, correct=True,
        )
    ]
    path = tmp_path / "records.jsonl"
    write_records(path, records)
    with pytest.raises(MissingSourceError):
        cli.main(["score", "--model", MODEL, "--records", str(path)])
