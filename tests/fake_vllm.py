"""A stand-in for the vLLM symbols DA patches, mirroring vLLM 0.20.2.

Every signature, field name and behaviour below was read off the real
``vllm==0.20.2`` wheel, not assumed:

* ``vllm/v1/sample/logits_processor/interface.py`` -- the ``LogitsProcessor``
  ABC, ``BatchUpdate``, ``MoveDirectionality``, and ``AddedRequest`` as
  ``(index, params, prompt_tok_ids | None, output_tok_ids)``.
* ``vllm/v1/attention/backends/{flash_attn,triton_attn}.py`` -- the metadata
  dataclasses, ``build(self, common_prefix_len, common_attn_metadata,
  fast_build=False)``, ``supports_update_block_table`` / ``update_block_table``
  on FlashAttention only, and Triton's ``build_for_cudagraph_capture`` doing
  ``seq_lens.fill_(1)`` after ``build``.
* ``vllm/v1/attention/backend.py`` -- ``CommonAttentionMetadata``.
* ``vllm/v1/kv_cache_interface.py`` -- the spec hierarchy, including
  ``ChunkedLocalAttentionSpec`` which carries ``attention_chunk_size`` and no
  ``sliding_window``.
* ``vllm/v1/attention/ops/triton_unified_attention.py`` -- one unified
  ``kernel_unified_attention``, launched with a 3-element grid on the decode
  (split-KV) path.
"""

from __future__ import annotations

import copy
import sys
import types
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto

import torch


# -- vllm.v1.kv_cache_interface --------------------------------------------


@dataclass(frozen=True, kw_only=True)
class KVCacheSpec:
    block_size: int = 16


@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int = 4
    head_size: int = 128


@dataclass(frozen=True, kw_only=True)
class FullAttentionSpec(AttentionSpec):
    block_size: int = 32


@dataclass(frozen=True, kw_only=True)
class SinkFullAttentionSpec(FullAttentionSpec):
    sink_len: int | None = None


@dataclass(frozen=True, kw_only=True)
class SlidingWindowSpec(AttentionSpec):
    block_size: int = 16
    sliding_window: int = 1024


@dataclass(frozen=True, kw_only=True)
class ChunkedLocalAttentionSpec(AttentionSpec):
    """Note: ``attention_chunk_size``, *not* ``sliding_window``."""

    attention_chunk_size: int = 8192


@dataclass(frozen=True, kw_only=True)
class MambaSpec(KVCacheSpec):
    """Not an AttentionSpec at all; has no window attribute of any kind."""


# -- vllm.v1.attention.backend ---------------------------------------------


@dataclass
class CommonAttentionMetadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor
    num_reqs: int
    block_table_tensor: torch.Tensor
    max_seq_len: int = 0
    slot_mapping: torch.Tensor | None = None

    def naive_query_lens(self) -> torch.Tensor:
        return self.query_start_loc[1:] - self.query_start_loc[:-1]


@dataclass
class Metadata:
    """Stands for FlashAttentionMetadata / TritonAttentionMetadata.

    Both are plain (mutable) dataclasses in vLLM, with exactly these names for
    the fields DA touches.
    """

    num_actual_tokens: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor | None = None
    scheduler_metadata: torch.Tensor | None = None


@dataclass
class _Builder:
    kv_cache_spec: object
    block_size: int
    block_tables: dict = field(default_factory=dict)
    seen_prefix_lens: list = field(default_factory=list)
    supports_update_block_table: bool = False

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        self.seen_prefix_lens.append(common_prefix_len)
        return Metadata(
            num_actual_tokens=int(common_attn_metadata.num_reqs),
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=int(common_attn_metadata.seq_lens.max()),
            seq_lens=common_attn_metadata.seq_lens,
            block_table=self.block_tables["block_table"],
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build(
            common_prefix_len=0, common_attn_metadata=common_attn_metadata
        )


class FlashAttentionMetadataBuilder(_Builder):
    # Real: a class attribute set to True in vLLM 0.20.2.
    supports_update_block_table: bool = True

    def update_block_table(self, metadata, blk_table, slot_mapping):
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata


class TritonAttentionMetadataBuilder(_Builder):
    supports_update_block_table: bool = False

    def build_for_cudagraph_capture(self, common_attn_metadata):
        metadata = self.build(0, common_attn_metadata)
        # Real behaviour: full graph capture with real seq_lens is extremely
        # slow, so Triton stamps them to 1 -- in place, on whatever tensor the
        # metadata happens to carry.
        metadata.seq_lens.fill_(1)
        return metadata


# -- vllm.v1.sample.logits_processor ---------------------------------------


class MoveDirectionality(Enum):
    UNIDIRECTIONAL = auto()
    SWAP = auto()


@dataclass(frozen=True)
class BatchUpdate:
    batch_size: int
    removed: list
    added: list
    moved: list


class LogitsProcessor(ABC):
    @classmethod
    def validate_params(cls, sampling_params):
        return None

    @abstractmethod
    def __init__(self, vllm_config, device, is_pin_memory) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply(self, logits): ...

    @abstractmethod
    def is_argmax_invariant(self) -> bool: ...

    @abstractmethod
    def update_state(self, batch_update) -> None: ...


def load_by_fqcn(fqcn: str):
    """Mirrors ``_load_logitsprocs_by_fqcns``: split on ':', then issubclass."""
    import importlib

    module_path, qualname = fqcn.split(":")
    module = importlib.import_module(module_path)
    obj = module
    for attr in qualname.split("."):
        obj = getattr(obj, attr)
    if not isinstance(obj, type):
        raise ValueError("Loaded logit processor must be a type.")
    if not issubclass(obj, LogitsProcessor):
        raise ValueError(f"{obj.__name__} must be a subclass of LogitsProcessor")
    return obj


# -- vllm.v1.attention.ops.triton_unified_attention ------------------------


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
    names = [
        "vllm",
        "vllm.v1",
        "vllm.v1.attention",
        "vllm.v1.attention.backends",
        "vllm.v1.attention.backends.flash_attn",
        "vllm.v1.attention.backends.triton_attn",
        "vllm.v1.attention.ops",
        "vllm.v1.attention.ops.triton_unified_attention",
        "vllm.v1.kv_cache_interface",
        "vllm.v1.sample",
        "vllm.v1.sample.logits_processor",
    ]
    modules = {name: types.ModuleType(name) for name in names}

    modules["vllm.v1.attention.backends.flash_attn"].FlashAttentionMetadataBuilder = (
        FlashAttentionMetadataBuilder
    )
    modules["vllm.v1.attention.backends.triton_attn"].TritonAttentionMetadataBuilder = (
        TritonAttentionMetadataBuilder
    )
    kvi = modules["vllm.v1.kv_cache_interface"]
    kvi.FullAttentionSpec = FullAttentionSpec
    kvi.SlidingWindowSpec = SlidingWindowSpec
    kvi.ChunkedLocalAttentionSpec = ChunkedLocalAttentionSpec
    kvi.SinkFullAttentionSpec = SinkFullAttentionSpec
    kvi.MambaSpec = MambaSpec
    modules["vllm.v1.attention.ops.triton_unified_attention"].kernel_unified_attention = (
        FakeKernel()
    )
    lp = modules["vllm.v1.sample.logits_processor"]
    lp.LogitsProcessor = LogitsProcessor
    lp.BatchUpdate = BatchUpdate
    lp.MoveDirectionality = MoveDirectionality

    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return modules
