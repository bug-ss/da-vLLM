"""The HTTP client, against a stubbed vLLM server."""

from __future__ import annotations

import json

import pytest

from da_vllm import DAConfig
from da_vllm.client import DAClient, DAServerError
from da_vllm.testing import lorem, qwen_tokenizer

MODEL = "Qwen/Qwen3.6-27B"
RESPONSE = (
    "<global>chunk 2</global>"
    '<focus magic_chunks="2">the value</focus>'
    "<local>committing</local><answer>the value</answer>"
)


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _StubServer:
    """Records what the client posted and replies like vLLM would."""

    def __init__(self, tok, status=200, include_ids=True):
        self.tok = tok
        self.status = status
        self.include_ids = include_ids
        self.posts = []
        self.gets = []

    def post(self, url, json=None, headers=None, timeout=None):
        self.posts.append({"url": url, "body": json, "headers": headers})
        if self.status >= 400:
            return _Resp({"error": "boom"}, self.status)
        choice = {"text": RESPONSE, "finish_reason": "stop", "index": 0}
        if self.include_ids:
            choice["token_ids"] = self.tok.encode(RESPONSE, add_special_tokens=False)
        return _Resp({"choices": [choice]})

    def get(self, url, headers=None, timeout=None):
        self.gets.append(url)
        return _Resp({"data": [{"id": MODEL}]})


@pytest.fixture
def client_and_server():
    tok = qwen_tokenizer()
    server = _StubServer(tok)
    config = DAConfig(
        enabled=True, max_model_len=131072,
        segment_target_tokens=2048, segment_max_tokens=2560,
    )
    client = DAClient(
        "http://gpu-server:8000/", MODEL, config=config, tokenizer=tok,
        session=server, max_tokens=1024, api_key="secret",
    )
    return client, server, tok


def test_it_answers_and_accounts_like_the_local_engine(client_and_server):
    client, _, _ = client_and_server
    result = client.answer(lorem(200, seed=1), "What is the value?")
    assert result.answer == "the value"
    assert result.focus_granted == 1
    assert result.attended_tokens < result.baseline_attended_tokens
    assert result.detection_failure is None


def test_it_posts_to_completions_with_token_ids(client_and_server):
    client, server, tok = client_and_server
    client.answer(lorem(200, seed=1), "What is the value?")
    body = server.posts[0]["body"]
    assert server.posts[0]["url"] == "http://gpu-server:8000/v1/completions"
    # Ids, not a string: no server-side tokenize, no second BOS.
    assert isinstance(body["prompt"], list)
    assert all(isinstance(t, int) for t in body["prompt"][:5])
    assert "messages" not in body
    assert body["return_token_ids"] is True


def test_the_da_flag_is_an_integer_and_the_prompt_text_matches_the_ids(client_and_server):
    client, server, tok = client_and_server
    client.answer(lorem(200, seed=1), "What is the value?")
    body = server.posts[0]["body"]
    xargs = body["vllm_xargs"]
    # vllm_xargs is typed str | int | float -- no bool in the union.
    assert xargs["da_enable"] == 1
    assert not isinstance(xargs["da_enable"], bool)
    # The text the server will detect over must be exactly the ids it serves.
    assert tok.encode(xargs["da_prompt_text"], add_special_tokens=False) == body["prompt"]


def test_sampling_comes_from_the_model_card(client_and_server):
    client, server, _ = client_and_server
    client.answer(lorem(200, seed=1), "What is the value?")
    body = server.posts[0]["body"]
    assert body["temperature"] == 0.7 and body["top_p"] == 0.8
    assert body["top_k"] == 20 and body["presence_penalty"] == 1.5
    assert body["max_tokens"] == 1024


def test_non_da_arms_send_no_xargs():
    tok = qwen_tokenizer()
    server = _StubServer(tok)
    client = DAClient(
        "http://gpu:8000", MODEL, arm="vanilla",
        config=DAConfig(enabled=False, max_model_len=131072),
        tokenizer=tok, session=server, max_tokens=1024,
    )
    client.answer(lorem(50, seed=1), "What is the value?")
    assert "vllm_xargs" not in server.posts[0]["body"]


def test_api_key_becomes_a_bearer_header(client_and_server):
    client, server, _ = client_and_server
    client.answer(lorem(200, seed=1), "What is the value?")
    assert server.posts[0]["headers"]["Authorization"] == "Bearer secret"


def test_a_batch_sends_one_request_per_item_with_its_own_prompt(client_and_server):
    client, server, tok = client_and_server
    doc = lorem(200, seed=1)
    results = client.answer_batch([(doc, "a?"), (doc, "b?"), (doc, "c?")])
    assert len(results) == 3 and len(server.posts) == 3
    # Each request carries its OWN prompt text, matching its own ids.
    for post in server.posts:
        body = post["body"]
        sent = body["vllm_xargs"]["da_prompt_text"]
        assert tok.encode(sent, add_special_tokens=False) == body["prompt"]
    assert len({p["body"]["vllm_xargs"]["da_prompt_text"] for p in server.posts}) == 3


def test_a_server_error_is_raised_not_swallowed():
    tok = qwen_tokenizer()
    server = _StubServer(tok, status=500)
    client = DAClient(
        "http://gpu:8000", MODEL,
        config=DAConfig(enabled=True, max_model_len=131072),
        tokenizer=tok, session=server, max_tokens=1024,
    )
    with pytest.raises(DAServerError, match="500"):
        client.answer(lorem(50, seed=1), "q?")


def test_it_copes_with_a_server_that_returns_no_token_ids():
    tok = qwen_tokenizer()
    server = _StubServer(tok, include_ids=False)
    client = DAClient(
        "http://gpu:8000", MODEL,
        config=DAConfig(enabled=True, max_model_len=131072,
                        segment_target_tokens=2048, segment_max_tokens=2560),
        tokenizer=tok, session=server, max_tokens=1024,
    )
    result = client.answer(lorem(200, seed=1), "q?")
    assert result.decode_steps > 0 and result.answer == "the value"


def test_check_reports_what_it_can_see(client_and_server):
    client, server, _ = client_and_server
    report = client.check()
    assert report["model_available"] is True
    assert "da: driver ready" in report["hint"]
    assert server.gets == ["http://gpu-server:8000/v1/models"]


def test_check_flags_a_model_name_mismatch():
    tok = qwen_tokenizer()
    server = _StubServer(tok)
    client = DAClient(
        "http://gpu:8000", MODEL, config=DAConfig(enabled=True, max_model_len=131072),
        tokenizer=tok, session=server, served_model_name="my-gemma-alias",
    )
    report = client.check()
    assert report["model_available"] is False
    assert "served_model_name" in report["hint"]


def test_a_batch_goes_out_in_parallel_but_returns_in_order():
    import threading
    import time

    tok = qwen_tokenizer()
    concurrent, peak, lock = [0], [0], threading.Lock()

    class _SlowServer(_StubServer):
        def post(self, url, json=None, headers=None, timeout=None):
            with lock:
                concurrent[0] += 1
                peak[0] = max(peak[0], concurrent[0])
            try:
                time.sleep(0.05)  # long enough for overlap to be observable
                # Reply with the question echoed, so order can be checked.
                q = json["vllm_xargs"]["da_prompt_text"]
                marker = "Q-A" if "a?" in q else ("Q-B" if "b?" in q else "Q-C")
                text = f'<focus magic_chunks="2">x</focus><answer>{marker}</answer>'
                self.posts.append({"url": url, "body": json, "headers": headers})
                return _Resp({"choices": [{
                    "text": text, "finish_reason": "stop", "index": 0,
                    "token_ids": self.tok.encode(text, add_special_tokens=False),
                }]})
            finally:
                with lock:
                    concurrent[0] -= 1

    server = _SlowServer(tok)
    client = DAClient(
        "http://gpu:8000", MODEL,
        config=DAConfig(enabled=True, max_model_len=131072,
                        segment_target_tokens=2048, segment_max_tokens=2560),
        tokenizer=tok, session=server, max_tokens=1024, max_workers=4,
    )
    doc = lorem(200, seed=1)
    results = client.answer_batch([(doc, "a?"), (doc, "b?"), (doc, "c?")])
    assert [r.answer for r in results] == ["Q-A", "Q-B", "Q-C"]
    assert peak[0] > 1, "requests were sent serially"


def test_max_workers_one_keeps_it_serial(client_and_server):
    client, server, tok = client_and_server
    client.max_workers = 1
    doc = lorem(200, seed=1)
    results = client.answer_batch([(doc, "a?"), (doc, "b?")])
    assert len(results) == 2 and len(server.posts) == 2
