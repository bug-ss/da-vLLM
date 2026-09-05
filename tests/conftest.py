from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from da_vllm.config import DAConfig  # noqa: E402
from da_vllm.prompt import PromptRenderer  # noqa: E402
from da_vllm.segmenter import Segmenter  # noqa: E402
from da_vllm.testing import gemma_tokenizer, lorem, qwen_tokenizer  # noqa: E402

# Small segments so a test context produces many magic chunks quickly.
TEST_TARGET = 120
TEST_CAP = 150


@pytest.fixture
def config() -> DAConfig:
    return DAConfig(enabled=True, max_num_seqs=8, max_model_len=8192)


@pytest.fixture(params=["Qwen/Qwen3.6-27B", "google/gemma-4-31B-it"])
def family_case(request):
    hub_id = request.param
    tok = qwen_tokenizer() if hub_id.startswith("Qwen") else gemma_tokenizer()
    renderer = PromptRenderer(
        tok, hub_id, segmenter=Segmenter(tok, target_tokens=TEST_TARGET, max_tokens=TEST_CAP)
    )
    return hub_id, tok, renderer


@pytest.fixture
def filler() -> str:
    return lorem(8, seed=7)
