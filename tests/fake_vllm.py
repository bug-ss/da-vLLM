"""A minimal stand-in for the vLLM symbols the DA patch monkeypatches.

Just enough surface to exercise the metadata-builder wrapper end to end:
two builder classes with the vLLM 0.20.2 ``build`` signature, the two KV cache
specs, and a module holding a Triton kernel object.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field

import torch


@dataclass
class FullAttentionSpec:
    block_size: int = 32


@dataclass
class SlidingWindowSpec:
    block_size: int = 16
    sliding_window: int = 1024


@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor


@dataclass
class Metadata:
    block_table: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
    scheduler_metadata: object | None = None


@dataclass
class _Builder:
    kv_cache_spec: object
    block_size: int
    block_tables: dict = field(default_factory=dict)
    seen_prefix_lens: list = field(default_factory=list)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        self.seen_prefix_lens.append(common_prefix_len)
        return Metadata(
            block_table=self.block_tables["block_table"],
            seq_lens=common_attn_metadata.seq_lens,
            max_seq_len=int(common_attn_metadata.seq_lens.max()),
        )


class FlashAttentionMetadataBuilder(_Builder):
    pass


class TritonAttentionMetadataBuilder(_Builder):
    pass


class FakeKernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.calls.append((grid, kwargs))
            return None

        return run


def install(monkeypatch) -> dict:
    """Register the fake modules in ``sys.modules`` for one test."""
    root = types.ModuleType("vllm")
    v1 = types.ModuleType("vllm.v1")
    attn = types.ModuleType("vllm.v1.attention")
    backends = types.ModuleType("vllm.v1.attention.backends")
    flash = types.ModuleType("vllm.v1.attention.backends.flash_attn")
    triton = types.ModuleType("vllm.v1.attention.backends.triton_attn")
    kvi = types.ModuleType("vllm.v1.kv_cache_interface")
    ops = types.ModuleType("vllm.attention")
    ops_pkg = types.ModuleType("vllm.attention.ops")
    tua = types.ModuleType("vllm.attention.ops.triton_unified_attention")

    flash.FlashAttentionMetadataBuilder = FlashAttentionMetadataBuilder
    triton.TritonAttentionMetadataBuilder = TritonAttentionMetadataBuilder
    kvi.FullAttentionSpec = FullAttentionSpec
    kvi.SlidingWindowSpec = SlidingWindowSpec
    tua.kernel_unified_attention_3d = FakeKernel()

    modules = {
        "vllm": root,
        "vllm.v1": v1,
        "vllm.v1.attention": attn,
        "vllm.v1.attention.backends": backends,
        "vllm.v1.attention.backends.flash_attn": flash,
        "vllm.v1.attention.backends.triton_attn": triton,
        "vllm.v1.kv_cache_interface": kvi,
        "vllm.attention": ops,
        "vllm.attention.ops": ops_pkg,
        "vllm.attention.ops.triton_unified_attention": tua,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return modules
