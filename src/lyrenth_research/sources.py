"""Reading sources properly: once each, with provenance, inside a budget.

This is the part of a research agent that decides what the model gets to
see. It reads every URL through Lyrenth, which returns each page as a clean
AIDocument with its canonical URL and a measured token count, and then:

- drops a page that was already read under another URL (mobile copies,
  tracking parameters, redirects all resolve to one canonical URL);
- stops adding sources once the token budget would be exceeded, so one
  long page cannot crowd out several short ones;
- records every page it could not read, with the reason, instead of
  quietly leaving a gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from lyrenth import Lyrenth

# The batch endpoint takes up to 20 URLs per call.
BATCH_SIZE = 20


@dataclass
class Source:
    """One page the agent read, with everything a citation needs."""

    url: str
    title: str
    text: str
    fetched_at: str = ""
    tokens: int = 0
    raw_html_tokens: int = 0


@dataclass
class Skipped:
    """A URL that did not make it into the sources, and why."""

    url: str
    reason: str


@dataclass
class Gathered:
    sources: List[Source] = field(default_factory=list)
    skipped: List[Skipped] = field(default_factory=list)
    tokens: int = 0

    @property
    def raw_html_tokens(self) -> int:
        return sum(s.raw_html_tokens for s in self.sources)


def _estimate_tokens(text: str) -> int:
    # Only used when a response carries no economics block. Four characters
    # per token is the usual rough figure for English text.
    return max(1, len(text) // 4)


def _to_source(doc) -> Source:
    raw = doc.raw or {}
    economics = raw.get("economics") or {}
    source = raw.get("source") or {}
    tokens = int(economics.get("output_tokens_approx") or 0) or _estimate_tokens(doc.markdown)
    return Source(
        url=doc.url,
        title=doc.title,
        text=doc.markdown,
        fetched_at=source.get("fetched_at") or "",
        tokens=tokens,
        raw_html_tokens=int(economics.get("raw_html_tokens_approx") or 0),
    )


def _dedupe_input(urls: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for u in urls:
        u = u.strip()
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def gather(
    urls: Iterable[str],
    client: Optional[Lyrenth] = None,
    token_budget: int = 30_000,
    fresh: bool = False,
) -> Gathered:
    """Read `urls` in order and return the sources that fit the budget.

    Order matters: earlier URLs are preferred when the budget runs out, so
    put the sources you trust most first.
    """
    client = client or Lyrenth()
    urls = _dedupe_input(urls)
    out = Gathered()
    seen_canonical = set()

    for start in range(0, len(urls), BATCH_SIZE):
        batch = urls[start : start + BATCH_SIZE]
        for result in client.read_batch(batch, fresh=fresh):
            if not result.ok or result.document is None:
                out.skipped.append(Skipped(result.url, result.error or "could not be read"))
                continue
            src = _to_source(result.document)
            if not src.text.strip():
                out.skipped.append(Skipped(result.url, "page has no readable text"))
                continue
            if src.url in seen_canonical:
                out.skipped.append(Skipped(result.url, f"duplicate of {src.url}"))
                continue
            if out.tokens + src.tokens > token_budget:
                out.skipped.append(
                    Skipped(result.url, f"over budget ({src.tokens:,} tokens would exceed {token_budget:,})")
                )
                continue
            seen_canonical.add(src.url)
            out.sources.append(src)
            out.tokens += src.tokens
    return out
