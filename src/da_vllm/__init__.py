"""Declarative Attention (DA) for vLLM.

A protocol that elicits a model to declare where it will attend, step by step,
inside its chain of thought -- ``<global>`` (full context), ``<focus
magic_chunks="K">`` (named segments) and ``<local>`` (recent output only) --
and a serving integration that turns those declarations into a compacted KV
block table so the stock attention kernel reads fewer pages.

Start at :class:`~da_vllm.prompt.PromptRenderer` (one renderer for every path),
:class:`~da_vllm.state_machine.DAStateMachine` (tag detection and mask
construction), and :mod:`da_vllm.masking` (the vLLM hook).  Before trusting any
number, run :mod:`da_vllm.validation` -- the mask silently did nothing for
weeks in the work this reproduces.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .api import DAAnswer, DAEngine
from .client import DAClient
from .config import DAConfig, RunawayConfig
from .detect import PromptMap, build_prompt_map
from .models import ModelSpec, get_model, get_model_by_type, registered_hub_ids
from .prompt import PromptRenderer, RenderedPrompt
from .segmenter import Segment, Segmenter
from .state_machine import DAStateMachine, MaskSnapshot, Mode, align_spans

__all__ = [
    "DAAnswer",
    "DAClient",
    "DAConfig",
    "DAEngine",
    "DAStateMachine",
    "MaskSnapshot",
    "Mode",
    "ModelSpec",
    "PromptMap",
    "PromptRenderer",
    "RenderedPrompt",
    "RunawayConfig",
    "Segment",
    "Segmenter",
    "align_spans",
    "build_prompt_map",
    "get_model",
    "get_model_by_type",
    "registered_hub_ids",
]
