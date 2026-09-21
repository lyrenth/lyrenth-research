"""Turning sources into an answer with numbered citations.

The model is yours. Anything that speaks the OpenAI-compatible chat
completions protocol works: the large hosted providers, most gateways, and
local servers such as Ollama or llama.cpp. Configure it with three
environment variables (or the matching CLI flags):

    LLM_BASE_URL   e.g. http://localhost:11434/v1 for a local Ollama
    LLM_MODEL      the model name your endpoint expects
    LLM_API_KEY    optional; local servers usually need none
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

from .sources import Source

SYSTEM_PROMPT = (
    "You answer questions using only the numbered sources you are given. "
    "Cite every claim with the number of the source it comes from, like [1] or [2][3]. "
    "If the sources do not contain the answer, say so plainly instead of guessing. "
    "Do not use knowledge that is not in the sources."
)


class ModelError(RuntimeError):
    """The model endpoint was not configured, unreachable, or answered badly."""


@dataclass
class Answer:
    text: str
    sources: List[Source]
    cited: List[int] = field(default_factory=list)
    unknown_citations: List[int] = field(default_factory=list)


def build_context(sources: List[Source]) -> str:
    """Number the sources and keep where and when each was read."""
    parts = []
    for i, s in enumerate(sources, 1):
        header = f"[{i}] {s.title or s.url}\nURL: {s.url}"
        if s.fetched_at:
            header += f"\nRead: {s.fetched_at}"
        if s.trimmed:
            # Said here, not only in a summary line, so a model reading this
            # block alone still knows the page continues past what it has.
            header += "\nNote: this page was longer than the budget. Only its opening is below."
        parts.append(f"{header}\n\n{s.text}")
    return "\n\n---\n\n".join(parts)


def build_messages(question: str, sources: List[Source]) -> list:
    user = f"Question: {question}\n\nSources:\n\n{build_context(sources)}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def check_citations(text: str, n_sources: int):
    """Return (cited, unknown): source numbers used, and numbers that do not exist."""
    found = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})
    cited = [n for n in found if 1 <= n <= n_sources]
    unknown = [n for n in found if n < 1 or n > n_sources]
    return cited, unknown


def call_model(
    messages: list,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: float = 180.0,
) -> str:
    base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
    model = model or os.environ.get("LLM_MODEL", "")
    api_key = api_key if api_key is not None else os.environ.get("LLM_API_KEY", "")
    if not base_url or not model:
        raise ModelError(
            "no model configured: set LLM_BASE_URL and LLM_MODEL "
            "(or pass --model-url and --model), or use --sources-only"
        )
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def post(fields: dict):
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(fields).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # Temperature 0 is what an agent that cites its sources wants: the same
    # pages should give the same answer. Some OpenAI-compatible endpoints
    # refuse the field outright, and the promise on the tin is that this runs
    # against any of them, so a refusal of the field is answered by sending
    # the request again without it rather than by failing the run.
    try:
        try:
            payload = post({"model": model, "messages": messages, "temperature": 0})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            if e.code == 400 and "temperature" in detail.lower():
                payload = post({"model": model, "messages": messages})
            else:
                raise ModelError(f"model endpoint returned HTTP {e.code}: {detail}") from None
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise ModelError(f"model endpoint returned HTTP {e.code}: {detail}") from None
    except urllib.error.URLError as e:
        raise ModelError(f"could not reach the model endpoint: {e.reason}") from None
    try:
        return payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ModelError("model endpoint answered in an unexpected shape") from None


def answer(question: str, sources: List[Source], **model_kwargs) -> Answer:
    if not sources:
        return Answer(text="None of the sources could be read, so there is nothing to answer from.", sources=[])
    text = call_model(build_messages(question, sources), **model_kwargs)
    cited, unknown = check_citations(text, len(sources))
    return Answer(text=text, sources=sources, cited=cited, unknown_citations=unknown)
