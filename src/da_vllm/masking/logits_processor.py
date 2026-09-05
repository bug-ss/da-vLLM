"""The logits processor that drives DA from inside EngineCore.

vLLM V1 runs the model in an EngineCore subprocess, so a monkeypatch applied in
the parent process patches nothing.  The simplest reliable place to install the
attention patch is the constructor of a custom logits processor, which vLLM
instantiates inside EngineCore -- which is why the driver lives here and not in
a standalone thread (guide 6.1/6.2).

``apply()`` is the identity and ``is_argmax_invariant()`` returns True, unless
the runaway guard is enabled -- forcing EOS changes the argmax, and vLLM skips
``apply()`` on the greedy path for argmax-invariant processors.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch

from ..config import DAConfig
from ..detect import build_prompt_map
from ..models import ModelSpec, resolve
from ..runaway import RunawayDetector
from ..state_machine import DAStateMachine
from .patch import get_patch_state, install_patch
from .shared import SharedMaskStore, install_shared_mask

logger = logging.getLogger(__name__)

#: Requests opt in explicitly.  A "is this vanilla" shortcut keyed on config
#: shape once also matched DA traffic (guide 12) -- there is no inference here.
DA_ENABLE_KEY = "da_enable"
DA_PROMPT_TEXT_KEY = "da_prompt_text"


@dataclass
class _Entry:
    request_id: str
    state: DAStateMachine
    output_token_ids: Sequence[int]
    detector: RunawayDetector | None = None
    dirty: bool = True
    forced_eos: bool = False


@dataclass
class DriverStats:
    added: int = 0
    removed: int = 0
    moved: int = 0
    rows_written: int = 0
    skipped_no_da: int = 0
    detection_failures: list[str] = field(default_factory=list)


class DADriver:
    """Backend-agnostic core of the logits processor.

    Kept separate from the vLLM class so the reconciliation logic is testable
    without an engine (see ``tests/test_driver.py``).
    """

    def __init__(
        self,
        config: DAConfig,
        store: SharedMaskStore,
        tokenizer: Any,
        model: str | ModelSpec,
        *,
        model_type: str | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.tokenizer = tokenizer
        self.spec = resolve(model, model_type=model_type)
        self.rows: dict[int, _Entry] = {}
        self.stats = DriverStats()

    # -- batch reconciliation ---------------------------------------------

    def remove(self, row: int) -> None:
        if self.rows.pop(row, None) is not None:
            # A slot recycled from a DA request must never keep stale mask data.
            self.store.reset_row(row)
        self.stats.removed += 1

    def add(
        self,
        row: int,
        request_id: str,
        prompt_token_ids: Sequence[int],
        output_token_ids: Sequence[int],
        params: Any,
    ) -> None:
        self.rows.pop(row, None)
        self.store.reset_row(row)
        self.stats.added += 1
        if not self._da_enabled(params):
            # Skip state-machine construction entirely: building it requires
            # decoding the whole prompt, about a second at 131K tokens, on the
            # engine's hot path (guide 6.2).
            self.stats.skipped_no_da += 1
            return

        prompt_text = self._prompt_text(params, prompt_token_ids)
        prompt_map = build_prompt_map(
            self.tokenizer,
            prompt_text,
            self.spec.family,
            self.config,
            prompt_token_ids=prompt_token_ids,
        )
        if prompt_map.failure_reason:
            # Focus is disabled for this request and it runs GLOBAL for life.
            # Never crash the engine on a detection failure: the original
            # fail-fast design let one bad sample abort a whole run.
            logger.warning(
                "da: focus disabled for %s: %s", request_id, prompt_map.failure_reason
            )
            self.stats.detection_failures.append(prompt_map.failure_reason)

        state = DAStateMachine(prompt_map, self.tokenizer, self.config, request_id=request_id)
        detector = (
            RunawayDetector(
                self.config.runaway,
                no_answer_token_budget=self.spec.no_answer_token_budget,
            )
            if self.config.runaway.enabled
            else None
        )
        self.rows[row] = _Entry(
            request_id=request_id,
            state=state,
            # A live reference to vLLM's list, never a copy: the driver reads
            # tokens vLLM appended since the previous step.
            output_token_ids=output_token_ids,
            detector=detector,
        )

    def move(self, src: int, dst: int, swap: bool) -> None:
        """Honour both move directionalities.

        SWAP exchanges the two rows; UNIDIRECTIONAL vacates ``src`` into
        ``dst``.  Either way, both entries are marked dirty so ``step`` rewrites
        their mask rows -- the mask row belongs to the request, not to the slot
        -- and any row left without a DA entry is reset to all-True.
        """
        self.stats.moved += 1
        if src == dst:
            return
        a = self.rows.pop(src, None)
        b = self.rows.pop(dst, None)
        if swap:
            if a is not None:
                a.dirty = True
                self.rows[dst] = a
            if b is not None:
                b.dirty = True
                self.rows[src] = b
        else:
            if a is not None:
                a.dirty = True
                self.rows[dst] = a
        if dst not in self.rows:
            self.store.reset_row(dst)
        if src not in self.rows:
            self.store.reset_row(src)

    # -- per-step work -----------------------------------------------------

    def step(self) -> list[int]:
        """Advance every state machine and rewrite the rows that changed.

        Returns the rows whose mask was rewritten.
        """
        block_size = None
        patch_state = get_patch_state()
        if patch_state is not None:
            block_size = patch_state.block_size

        written: list[int] = []
        for row, entry in self.rows.items():
            before = entry.state.num_consumed
            changed = entry.state.advance(entry.output_token_ids)
            if entry.detector is not None and entry.state.num_consumed > before:
                new_ids = list(entry.output_token_ids[before : entry.state.num_consumed])
                entry.detector.observe(new_ids, "")
            if changed or entry.dirty:
                self.store.write_snapshot(
                    row,
                    entry.state.snapshot(),
                    block_size=block_size,
                    optimized=self.config.optimized_mask_writer,
                )
                entry.dirty = False
                written.append(row)
        self.stats.rows_written += len(written)
        return written

    def runaway_rows(self) -> list[int]:
        return [
            row
            for row, e in self.rows.items()
            if e.detector is not None and e.detector.triggered and not e.forced_eos
        ]

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _extra_args(params: Any) -> dict[str, Any]:
        extra = getattr(params, "extra_args", None)
        return extra if isinstance(extra, dict) else {}

    def _da_enabled(self, params: Any) -> bool:
        return bool(self._extra_args(params).get(DA_ENABLE_KEY, False))

    def _prompt_text(self, params: Any, prompt_token_ids: Sequence[int]) -> str:
        """The rendered prompt, for segment detection.

        Detection itself is always server-side (regex over the tool turns).  The
        client may hand us the exact rendered string to skip a 131K-token
        decode; :func:`~da_vllm.detect.build_prompt_map` still cross-checks it
        against the engine's own token count and declines focus on a mismatch.
        """
        text = self._extra_args(params).get(DA_PROMPT_TEXT_KEY)
        if isinstance(text, str) and text:
            return text
        return self.tokenizer.decode(list(prompt_token_ids))


def _load_tokenizer(model_config: Any):
    from transformers import AutoTokenizer  # imported lazily: engine-side only

    name = getattr(model_config, "tokenizer", None) or getattr(model_config, "model")
    revision = getattr(model_config, "tokenizer_revision", None)
    return AutoTokenizer.from_pretrained(name, revision=revision)


def read_da_config(vllm_config: Any) -> DAConfig:
    """Read the DA config out of vLLM's ``additional_config``.

    Strict: unknown keys raise (guide 12), and DA stays off unless the config
    says otherwise.
    """
    additional = getattr(vllm_config, "additional_config", None) or {}
    raw = dict(additional.get("declarative_attention", {}))
    sched = getattr(vllm_config, "scheduler_config", None)
    model_cfg = getattr(vllm_config, "model_config", None)
    raw.setdefault("max_num_seqs", int(getattr(sched, "max_num_seqs", 256)))
    raw.setdefault("max_model_len", int(getattr(model_cfg, "max_model_len", 262_144)))
    return DAConfig.from_dict(raw)


class DALogitsProcessor:
    """vLLM ``LogitsProcessor``.  Registered via ``logits_processors=[...]``."""

    def __init__(self, vllm_config: Any, device: Any, is_pin_memory: bool) -> None:
        self.config = read_da_config(vllm_config)
        self.device = device
        self.driver: DADriver | None = None
        self._eos_token_id: int | None = None
        if not self.config.enabled:
            logger.info("da: disabled by config; logits processor is inert")
            return

        # Install from inside the engine process (guide 6.1).
        install_patch(self.config)
        store = install_shared_mask(
            self.config.max_num_seqs, self.config.max_model_len, device
        )
        model_config = getattr(vllm_config, "model_config", None)
        tokenizer = _load_tokenizer(model_config)
        model_type = getattr(getattr(model_config, "hf_config", None), "model_type", None)
        self.driver = DADriver(
            self.config,
            store,
            tokenizer,
            getattr(model_config, "model", ""),
            model_type=model_type,
        )
        self._eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if self.config.runaway.enabled and self._eos_token_id is None:
            raise RuntimeError(
                "runaway guard is on but the tokenizer reports no eos_token_id; "
                "confirm the EOS id is in the served stop set before enabling it"
            )
        logger.info(
            "da: driver ready in pid %d (mask %.0f MB, patched %s)",
            os.getpid(),
            store.nbytes() / 1e6,
            (get_patch_state().patched_targets if get_patch_state() else []),
        )

    # -- vLLM interface ----------------------------------------------------

    def is_argmax_invariant(self) -> bool:
        # Forcing EOS changes the argmax; say so or vLLM skips apply() on the
        # greedy path and the guard silently does nothing.
        return not self.config.runaway.enabled

    def update_state(self, batch_update: Any) -> None:
        if self.driver is None:
            return
        if batch_update is not None:
            for removed in getattr(batch_update, "removed", ()):
                self.driver.remove(_index_of(removed))
            for added in getattr(batch_update, "added", ()):
                index, params, prompt_ids, output_ids = _unpack_added(added)
                # vLLM's BatchUpdate carries no request id, only the row; the
                # row is what every log line and mask write is keyed on anyway.
                self.driver.add(
                    index, f"row{index}", prompt_ids, output_ids, params
                )
            for moved in getattr(batch_update, "moved", ()):
                src, dst, swap = _unpack_moved(moved)
                self.driver.move(src, dst, swap)
        self.driver.step()

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.driver is None or not self.config.runaway.enabled:
            return logits  # identity
        rows = self.driver.runaway_rows()
        if not rows:
            return logits
        for row in rows:
            if row >= logits.shape[0]:
                continue
            logits[row].fill_(float("-inf"))
            logits[row, self._eos_token_id] = 0.0
            entry = self.driver.rows.get(row)
            if entry is not None:
                entry.forced_eos = True
                logger.warning(
                    "da: forcing EOS on row %d (%s)",
                    row,
                    entry.detector.triggered_by if entry.detector else "?",
                )
        return logits


def _index_of(removed: Any) -> int:
    return int(removed[0]) if isinstance(removed, tuple) else int(removed)


def _unpack_added(added: Any):
    index, params, *rest = added
    prompt_ids: Sequence[int] = ()
    output_ids: Sequence[int] = ()
    # vLLM has carried both (index, params, prompt_ids, output_ids) and
    # (index, params, output_ids); take the last element as the live output
    # list either way.
    if len(rest) >= 2:
        prompt_ids, output_ids = rest[0], rest[1]
    elif len(rest) == 1:
        output_ids = rest[0]
    return int(index), params, prompt_ids, output_ids


def _unpack_moved(moved: Any):
    src, dst, direction = moved
    name = getattr(direction, "name", str(direction)).upper()
    return int(src), int(dst), "SWAP" in name
