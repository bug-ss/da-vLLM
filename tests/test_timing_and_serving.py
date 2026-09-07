import pytest

from da_vllm.config import DAConfig
from da_vllm.serving import EngineOptions
from da_vllm.timing import (
    TIMING_CHECKLIST,
    TimingRun,
    block_aligned_synthetic_mask,
    match_arms_on_kept_fraction,
)


def test_only_steady_decode_steps_are_measured():
    run = TimingRun(target_residency=4, warmup_steps=3, prefill_guard_steps=1)
    for i in range(12):
        run.record(
            i,
            0.01,
            num_prompt_tokens=64 if i == 7 else 0,
            num_resident=4 if i != 10 else 3,
        )
    kept = {s.index for s in run.steady()}
    assert 0 not in kept and 2 not in kept  # warmup
    assert {6, 7, 8}.isdisjoint(kept)  # prefill and its guard band
    assert 10 not in kept  # residency changed
    assert kept == {3, 4, 5, 9, 11}


def test_summary_reports_a_median_and_an_untrimmed_mean():
    run = TimingRun(target_residency=1, warmup_steps=0, prefill_guard_steps=0)
    for i, t in enumerate([0.01, 0.01, 0.01, 0.5]):
        run.record(i, t, num_prompt_tokens=0, num_resident=1)
    s = run.summary()
    assert s["median_s"] == 0.01
    assert s["mean_s"] > s["median_s"]  # the tail is visible


def test_synthetic_masks_are_defined_on_whole_blocks():
    spans = block_aligned_synthetic_mask(1024, 32, 0.25)
    assert all(s % 32 == 0 for s, _ in spans)
    assert all(e % 32 == 0 or e == 1024 for _, e in spans)
    # The tail block is always kept.
    assert spans[-1][1] == 1024


def test_arms_must_match_on_measured_kept_fraction():
    assert match_arms_on_kept_fraction({"a": 0.50, "b": 0.515})
    assert not match_arms_on_kept_fraction({"a": 0.50, "b": 0.60})


def test_timing_checklist_is_documented():
    assert "prefix caching off" in TIMING_CHECKLIST
    assert any("process group" in item for item in TIMING_CHECKLIST)


def test_each_arm_gets_its_own_compile_cache():
    da = EngineOptions("Qwen/Qwen3.6-27B", "da", DAConfig(enabled=True))
    nomask = EngineOptions("Qwen/Qwen3.6-27B", "da_no_mask", DAConfig(enabled=False))
    vanilla = EngineOptions("Qwen/Qwen3.6-27B", "vanilla", DAConfig(enabled=False))
    roots = {o.resolved_cache_root() for o in (da, nomask, vanilla)}
    assert len(roots) == 3


def test_arm_and_mask_state_cannot_disagree():
    with pytest.raises(ValueError):
        EngineOptions("Qwen/Qwen3.6-27B", "da", DAConfig(enabled=False))
    with pytest.raises(ValueError):
        EngineOptions("Qwen/Qwen3.6-27B", "da_no_mask", DAConfig(enabled=True))


def test_engine_kwargs_are_the_paper_settings():
    o = EngineOptions("Qwen/Qwen3.6-27B", "da", DAConfig(enabled=True))
    kwargs = o.engine_kwargs()
    assert kwargs["gpu_memory_utilization"] == 0.85
    assert kwargs["max_num_seqs"] == 256  # static, never auto-sized from VRAM
    assert kwargs["max_model_len"] == 262_144
    assert kwargs["dtype"] == "bfloat16"
    assert "enable_prefix_caching" not in kwargs  # vLLM's default
    assert kwargs["logits_processors"] == [
        "da_vllm.masking.logits_processor:DALogitsProcessor"
    ]


def test_only_the_da_arm_installs_the_logits_processor():
    o = EngineOptions("Qwen/Qwen3.6-27B", "vanilla", DAConfig(enabled=False))
    assert "logits_processors" not in o.engine_kwargs()
    assert "additional_config" not in o.engine_kwargs()


def test_worker_env_puts_sitecustomize_on_the_path_and_forces_spawn():
    from da_vllm.masking import sitecustomize_dir, worker_env

    env = worker_env(DAConfig(enabled=True), {"PYTHONPATH": "/existing"})
    assert env["PYTHONPATH"].startswith(sitecustomize_dir())
    assert "/existing" in env["PYTHONPATH"]
    assert env["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn"
    assert "DA_VLLM_CONFIG" in env


def test_a_false_boolean_becomes_an_explicit_no_flag():
    """vLLM's default for prefix caching is ON. Omitting the flag when the
    caller asked for False would measure DA against a cached baseline, which
    is item 1 of the timing checklist."""
    from da_vllm.serving import vllm_serve_command

    off = EngineOptions(
        "Qwen/Qwen3.6-27B", "vanilla", DAConfig(enabled=False),
        enable_prefix_caching=False,
    )
    argv, _ = vllm_serve_command(off)
    assert "--no-enable-prefix-caching" in argv

    on = EngineOptions(
        "Qwen/Qwen3.6-27B", "vanilla", DAConfig(enabled=False),
        enable_prefix_caching=True,
    )
    argv, _ = vllm_serve_command(on)
    assert "--enable-prefix-caching" in argv
    assert "--no-enable-prefix-caching" not in argv
