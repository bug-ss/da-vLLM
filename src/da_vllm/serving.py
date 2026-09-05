"""Engine construction and process hygiene.

Three infrastructure lessons are baked in (guide 8.3 / 12):

* **A separate compile cache per configuration.**  vLLM's AOT compile cache key
  does not include the DA patch state, so a cache built with patches active is
  incompatible with an unpatched run of the same model.  Each arm gets its own
  ``VLLM_CACHE_ROOT``.
* **``max_num_seqs`` is set statically.**  Auto-sizing it from VRAM
  under-counted hybrid-model KV capacity by 4 to 6x; let vLLM's admission
  control do the limiting instead.  Relatedly, ``num_gpu_blocks * block_size``
  is *not* the KV capacity on a hybrid model -- probe capacity empirically
  rather than computing it.
* **Engines run in their own process group.**  Orphaned EngineCore processes
  reparent to PID 1 and hold VRAM; killing the group is the only reliable
  cleanup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import subprocess
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .config import DAConfig
from .masking import worker_env
from .models import ModelSpec, resolve

logger = logging.getLogger(__name__)

DEFAULT_GPU_MEMORY_UTILIZATION = 0.85
DEFAULT_MAX_NUM_SEQS = 256

#: How vLLM addresses a custom logits processor: ``module:qualname``.
#: ``_load_logitsprocs_by_fqcns`` does ``logitproc.split(":")``, so a dotted
#: path fails to unpack and the engine never starts.
DA_LOGITS_PROCESSOR_FQCN = "da_vllm.masking.logits_processor:DALogitsProcessor"


@dataclass
class EngineOptions:
    """Serving configuration for one arm."""

    model: str
    arm: str  # "vanilla" | "da_no_mask" | "da"
    da_config: DAConfig
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS
    max_model_len: int | None = None
    tensor_parallel_size: int = 1
    enable_prefix_caching: bool | None = None  # None = vLLM's default
    dtype: str = "bfloat16"
    kv_cache_dtype: str = "auto"
    cache_root: str | None = None
    extra_engine_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.arm not in ("vanilla", "da_no_mask", "da"):
            raise ValueError(f"unknown arm {self.arm!r}")
        # The DA-no-mask arm serves the identical prompt with the patch off.
        # That is the whole ablation: without it, the cost of the prompt format
        # and the cost of the mask cannot be separated.
        if self.arm == "da" and not self.da_config.enabled:
            raise ValueError("arm 'da' requires da_config.enabled=True")
        if self.arm != "da" and self.da_config.enabled:
            raise ValueError(f"arm {self.arm!r} requires da_config.enabled=False")

    @property
    def spec(self) -> ModelSpec:
        return resolve(self.model)

    def resolved_max_model_len(self) -> int:
        return self.max_model_len or self.spec.max_model_len

    def config_digest(self) -> str:
        payload = {
            "model": self.model,
            "arm": self.arm,
            "tp": self.tensor_parallel_size,
            "max_model_len": self.resolved_max_model_len(),
            "max_num_seqs": self.max_num_seqs,
            "dtype": self.dtype,
            "kv_cache_dtype": self.kv_cache_dtype,
            "prefix_caching": self.enable_prefix_caching,
            "da": self.da_config.to_dict(),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

    def resolved_cache_root(self) -> Path:
        base = Path(self.cache_root or os.environ.get("DA_CACHE_BASE", "~/.cache/da-vllm"))
        return (base / f"{self.arm}-{self.config_digest()}").expanduser()

    def engine_env(self) -> dict[str, str]:
        env = worker_env(self.da_config) if self.da_config.enabled else dict(os.environ)
        env["VLLM_CACHE_ROOT"] = str(self.resolved_cache_root())
        return env

    def engine_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "dtype": self.dtype,
            "kv_cache_dtype": self.kv_cache_dtype,
            "max_model_len": self.resolved_max_model_len(),
            # Static, never auto-sized from VRAM.
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "tensor_parallel_size": self.tensor_parallel_size,
        }
        if self.enable_prefix_caching is not None:
            kwargs["enable_prefix_caching"] = self.enable_prefix_caching
        if self.da_config.enabled:
            kwargs["additional_config"] = {
                "declarative_attention": self.da_config.to_dict()
            }
            # vLLM splits a fully-qualified name on ":" -- `module:qualname`,
            # not a dotted path. A dotted string raises "not enough values to
            # unpack" before the engine finishes starting.
            kwargs["logits_processors"] = [DA_LOGITS_PROCESSOR_FQCN]
        kwargs.update(self.extra_engine_kwargs)
        return kwargs


def build_llm(options: EngineOptions):
    """Construct an in-process ``vllm.LLM`` for one arm."""
    os.environ["VLLM_CACHE_ROOT"] = str(options.resolved_cache_root())
    if options.da_config.enabled:
        for key, value in worker_env(options.da_config).items():
            if key in ("PYTHONPATH", "DA_VLLM_CONFIG", "VLLM_WORKER_MULTIPROC_METHOD"):
                os.environ[key] = value
    from vllm import LLM  # type: ignore

    return LLM(**options.engine_kwargs())


def sampling_params_for(model: str | ModelSpec, **overrides):
    """Model-card sampling parameters.  Unknown models raise, never default."""
    spec = resolve(model)
    from vllm import SamplingParams  # type: ignore

    params = spec.sampling.to_dict()
    params.update(overrides)
    # n = 1, no seed: generation is unseeded and small run-to-run movement is
    # expected (guide 8.3/8.6).
    return SamplingParams(n=1, **params)


@contextmanager
def engine_process(argv: Sequence[str], env: dict[str, str] | None = None, *, timeout: float = 60.0):
    """Run an engine in its own process group and kill the whole group.

    Orphaned EngineCore children reparent to PID 1 and hold VRAM across arms
    otherwise.  Always one engine per arm in its own group.
    """
    proc = subprocess.Popen(  # noqa: S603
        list(argv),
        env=env,
        start_new_session=True,  # new process group
    )
    try:
        yield proc
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:  # pragma: no cover
                pass
            deadline = time.monotonic() + timeout
            while proc.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:  # pragma: no cover
                    pass
                proc.wait()


def vllm_serve_command(options: EngineOptions) -> tuple[list[str], dict[str, str]]:
    """The ``vllm serve`` argv and environment for one arm.

    Use this when the model is served over HTTP rather than in process.  The
    DA mask only exists where the patch is installed, so an OpenAI-compatible
    server has to be started with the logits processor registered -- a plain
    ``vllm serve`` gives you the DA prompt format and full attention.

    Per-request opt-in then travels in ``extra_body``::

        {"vllm_xargs": {"da_enable": true, "da_prompt_text": "<the prompt>"}}
    """
    argv = ["vllm", "serve", options.model]
    kwargs = options.engine_kwargs()
    kwargs.pop("model", None)
    logits_processors = kwargs.pop("logits_processors", None)
    additional_config = kwargs.pop("additional_config", None)
    for key, value in kwargs.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv += [flag, str(value)]
    if logits_processors:
        for entry in logits_processors:
            argv += ["--logits-processors", entry]
    if additional_config is not None:
        argv += ["--additional-config", json.dumps(additional_config)]
    return argv, options.engine_env()


def describe_kv_capacity(llm: Any) -> dict[str, Any]:
    """Report what vLLM actually allocated, with the caveat attached.

    ``num_gpu_blocks * block_size`` is **not** the KV capacity on a hybrid
    model: block size differs per layer type and the recurrent-state pages are
    counted too.  Use this for logging; probe real concurrency empirically.
    """
    out: dict[str, Any] = {"caveat": "not a sequence-capacity figure on hybrid models"}
    try:
        cache_config = llm.llm_engine.vllm_config.cache_config
        out["num_gpu_blocks"] = getattr(cache_config, "num_gpu_blocks", None)
        out["reported_block_size"] = getattr(cache_config, "block_size", None)
    except Exception as exc:  # pragma: no cover
        out["error"] = repr(exc)
    return out
