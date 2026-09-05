"""The validation harness.  Not optional -- see ``docs/VALIDATION.md``.

The mask silently did nothing for weeks in the work this reproduces, and the
obvious checks passed the whole time.  Three-column NLL parity
(:mod:`.nll_parity`) is the one that catches it; :mod:`.checks` covers the rest.
"""

from .checks import (
    RoundTripCase,
    RoundTripResult,
    StepToggle,
    ValidationError,
    assert_prompt_parity,
    assert_round_trip,
    default_cases,
    kept_fraction_report,
    kernel_scaling,
    round_trip,
)
from .nll_parity import ParityResult, three_column_parity

__all__ = [
    "ParityResult",
    "RoundTripCase",
    "RoundTripResult",
    "StepToggle",
    "ValidationError",
    "assert_prompt_parity",
    "assert_round_trip",
    "default_cases",
    "kept_fraction_report",
    "kernel_scaling",
    "round_trip",
    "three_column_parity",
]
