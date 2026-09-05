"""GPU-side machinery: the shared mask, the block-table remap, the vLLM patch."""

from __future__ import annotations

import os
from pathlib import Path

from .patch import get_patch_state, install_patch, install_triton_num_stages, uninstall_patch
from .remap import RemapScratch, RemapStats, remap_optimized, remap_readable
from .shared import (
    SharedMaskStore,
    clear_shared_mask,
    get_shared_mask,
    install_shared_mask,
)

__all__ = [
    "RemapScratch",
    "RemapStats",
    "SharedMaskStore",
    "clear_shared_mask",
    "get_patch_state",
    "get_shared_mask",
    "install_patch",
    "install_shared_mask",
    "install_triton_num_stages",
    "remap_optimized",
    "remap_readable",
    "sitecustomize_dir",
    "uninstall_patch",
    "worker_env",
]


def sitecustomize_dir() -> str:
    """Directory to prepend to ``PYTHONPATH`` for tensor parallel > 1."""
    return str((Path(__file__).resolve().parent.parent / "resources"))


def worker_env(config, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Environment that makes spawned vLLM workers install the patch."""
    import json

    env = dict(base_env if base_env is not None else os.environ)
    path = sitecustomize_dir()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{path}{os.pathsep}{existing}" if existing else path
    env["DA_VLLM_CONFIG"] = json.dumps(config.to_dict())
    # vLLM must spawn, not fork, or the child inherits an already-imported
    # (unpatched) module table.
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env
