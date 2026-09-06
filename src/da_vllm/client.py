"""Talk to a DA-enabled vLLM server over HTTP.

Use this when your application and the GPU live on different machines.

    from da_vllm import DAClient

    client = DAClient("http://gpu-server:8000", "google/gemma-4-31B-it")
    result = client.answer(document, "When was Acme founded?")

    result.answer            # "2003"
    result.attended_tokens   # what the kernel actually read
    result.reduction_pct     # against reading everything, every step

It gets four easy-to-miss things right, each of which quietly costs you the
whole benefit (or worse) if you hand-roll the request:

1. It posts to ``/v1/completions``, not ``/v1/chat/completions``. The prompt has
   already been through the chat template; letting the server apply it again
   would produce a different prompt from the one the mask was planned for.
2. It sends token ids, so the server does not tokenize a 100K-token string on
   the critical path -- and there is no chance of a second BOS being added.
   (``/v1/completions`` sets ``add_special_tokens`` to true by default, which
   *would* add one.)
3. It sets ``vllm_xargs`` with ``da_enable`` as an **integer**. vLLM types that
   field as string/int/float, with no boolean in the union.
4. It sends the exact prompt string it rendered as ``da_prompt_text``, so the
   server's chunk detection runs over the same text the engine tokenized. A
   mismatch there is the one failure that masks the wrong tokens; the server
   checks and refuses, but only if you send the right string in the first
   place.

Needs ``requests``. Everything else in this package works without it.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

from .api import DAAnswer, DAEngine
from .config import DAConfig
from .models import ModelSpec, resolve

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 600.0


class DAServerError(RuntimeError):
    """The server rejected the request or returned something unusable."""


class DAClient:
    """A remote :class:`~da_vllm.api.DAEngine`.

    Same ``answer`` / ``answer_batch`` interface and the same accounting; the
    only difference is that generation happens over HTTP.
    """

    def __init__(
        self,
        base_url: str,
        model: str | ModelSpec,
        *,
        arm: str = "da",
        config: DAConfig | None = None,
        tokenizer: Any = None,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_tokens: int | None = None,
        served_model_name: str | None = None,
        max_workers: int = 8,
        session: Any = None,
    ) -> None:
        self.spec = resolve(model)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        # What the server calls this model, if it was started under an alias.
        self.served_model_name = served_model_name or self.spec.hub_id
        self._api_key = api_key
        self._session = session
        #: Requests in a batch go out in parallel; the server decides how to
        #: schedule them. Serial requests would mean N times the latency for a
        #: batch of N questions about the same document.
        self.max_workers = max(1, int(max_workers))

        # All the prompt building, replay and accounting is the local engine's;
        # only generation is swapped out.
        self._engine = DAEngine(
            self.spec,
            arm=arm,
            config=config,
            tokenizer=tokenizer,
            generate_fn=self._generate,
            max_tokens=max_tokens,
        )

    # -- properties --------------------------------------------------------

    @property
    def renderer(self):
        return self._engine.renderer

    @property
    def tokenizer(self):
        return self._engine.tokenizer

    @property
    def arm(self) -> str:
        return self._engine.arm

    # -- HTTP --------------------------------------------------------------

    def _http(self):
        if self._session is not None:
            return self._session
        try:
            import requests
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "DAClient needs `requests`: pip install requests"
            ) from exc
        session = requests.Session()
        # requests.Session is thread-safe enough for concurrent posts as long
        # as the pool is big enough for the workers.
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=self.max_workers, pool_maxsize=self.max_workers
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        self._session = session
        return self._session

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _generate(
        self,
        token_id_lists: Sequence[Sequence[int]],
        params: dict,
        prompts: Sequence[Any],
    ) -> list[tuple[str, list[int], str | None]]:
        """The generate_fn DAEngine calls.  One request per prompt.

        ``prompts`` arrives alongside the ids, so the exact rendered string can
        be sent as ``da_prompt_text`` with no second render and no shared state.
        """
        work = [(list(ids), prompt.text) for ids, prompt in zip(token_id_lists, prompts)]
        if len(work) == 1 or self.max_workers == 1:
            return [self._one(ids, text, params) for ids, text in work]

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(work))) as pool:
            # Results must come back in request order, so the accounting lines
            # up with the right question.
            return list(pool.map(lambda item: self._one(item[0], item[1], params), work))

    def _one(
        self, token_ids: list[int], prompt_text: str, params: dict
    ) -> tuple[str, list[int], str | None]:
        body: dict[str, Any] = {
            "model": self.served_model_name,
            "prompt": token_ids,
            "max_tokens": params["max_tokens"],
            "temperature": params["temperature"],
            "top_p": params["top_p"],
            "n": 1,
            # The generated ids come back too, so the replay counts real decode
            # steps instead of re-tokenizing the text and hoping it round-trips.
            "return_token_ids": True,
        }
        # top_k and presence_penalty are top-level vLLM extensions here, not
        # vllm_xargs entries.
        for key in ("top_k", "presence_penalty"):
            if params.get(key):
                body[key] = params[key]

        if self.arm == "da":
            body["vllm_xargs"] = {
                # An integer, not a bool: vllm_xargs is typed str | int | float.
                "da_enable": 1,
                # The exact text the detector must run over.
                "da_prompt_text": prompt_text,
            }

        response = self._http().post(
            f"{self.base_url}/v1/completions",
            json=body,
            headers=self._headers(),
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise DAServerError(
                f"{response.status_code} from {self.base_url}: {response.text[:500]}"
            )
        payload = response.json()
        try:
            choice = payload["choices"][0]
        except (KeyError, IndexError) as exc:
            raise DAServerError(f"unexpected response shape: {payload}") from exc

        text = choice.get("text", "")
        finish = choice.get("finish_reason")
        ids = _token_ids_from_choice(choice)
        if ids is None:
            # No ids returned; re-tokenize so the step count is still right to
            # within tokenizer round-tripping.
            ids = list(self.tokenizer.encode(text, add_special_tokens=False))
        return text, ids, finish

    # -- the public call ---------------------------------------------------

    def answer(self, context: str, question: str, **kwargs) -> DAAnswer:
        return self.answer_batch([(context, question)], **kwargs)[0]

    def answer_batch(
        self, items: Iterable[tuple[str, str]], **kwargs
    ) -> list[DAAnswer]:
        """Answer several questions.

        One HTTP request per question, sent in parallel (up to ``max_workers``)
        so the server can batch them together. Results come back in the order
        you asked.
        """
        pairs = list(items)
        return self._engine.answer_batch(pairs, **kwargs)

    # -- diagnostics -------------------------------------------------------

    def check(self) -> dict[str, Any]:
        """Is the server up, serving this model, and is DA switched on?

        Cannot see inside the server, so it infers DA from a tiny request: if
        masking is active the response comes back normally either way, so this
        reports what it can and tells you where to look for the rest.
        """
        out: dict[str, Any] = {"base_url": self.base_url, "model": self.served_model_name}
        try:
            models = self._http().get(
                f"{self.base_url}/v1/models", headers=self._headers(), timeout=30
            ).json()
            served = [m.get("id") for m in models.get("data", [])]
            out["models_served"] = served
            out["model_available"] = self.served_model_name in served
        except Exception as exc:
            out["error"] = repr(exc)
            return out

        if not out["model_available"]:
            out["hint"] = (
                f"the server is serving {served}; pass served_model_name= if it "
                "was started under a different name"
            )
            return out
        out["hint"] = (
            "DA itself cannot be probed from outside. Check the server log for "
            "'da: driver ready' at startup, and watch answer().declines and "
            "answer().detection_failure on real requests."
        )
        return out


def _token_ids_from_choice(choice: dict) -> list[int] | None:
    ids = choice.get("token_ids")
    if isinstance(ids, list) and ids and isinstance(ids[0], int):
        return ids
    return None
