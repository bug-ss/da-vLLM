"""Evaluation: 15 sources, three arms, one fixed judge, macro scoring.

Every published number is recomputed from raw per-response records
(:mod:`.records`) against the explicit source list in :data:`.data.SOURCES` --
never from a cached summary.
"""

from .data import SOURCES, SOURCE_KEYS, Example, prepare_source
from .judge import Verdict, judge_messages, parse_verdict, score_response
from .records import ARMS, ResponseRecord, new_run_id, read_records, write_records
from .score import ArmSummary, Comparison, compare, report, summarize_arm

__all__ = [
    "ARMS",
    "ArmSummary",
    "Comparison",
    "Example",
    "ResponseRecord",
    "SOURCES",
    "SOURCE_KEYS",
    "Verdict",
    "compare",
    "new_run_id",
    "judge_messages",
    "parse_verdict",
    "prepare_source",
    "read_records",
    "report",
    "score_response",
    "summarize_arm",
    "write_records",
]
