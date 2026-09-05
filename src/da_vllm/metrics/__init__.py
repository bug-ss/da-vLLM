"""Metrics: attended-token replay and roofline decode wall-time.

The server does not report attended tokens; they are reconstructed offline by
replaying the state machine over the returned text (:mod:`.replay`).  The
roofline (:mod:`.roofline`) is a ceiling at stated utilizations, not a
measurement -- see :mod:`da_vllm.timing` for what real timing showed.
"""

from .replay import ReplayResult, replay, replay_vanilla, vanilla_attended_tokens
from .roofline import (
    RooflineBreakdown,
    global_kv_bytes_from_shapes,
    global_kv_bytes_per_token,
    roofline_response,
    verify_geometry,
)

__all__ = [
    "ReplayResult",
    "RooflineBreakdown",
    "global_kv_bytes_from_shapes",
    "global_kv_bytes_per_token",
    "replay",
    "replay_vanilla",
    "roofline_response",
    "vanilla_attended_tokens",
    "verify_geometry",
]
