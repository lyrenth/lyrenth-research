"""Reading sources properly: once each, with provenance, inside a budget.

This is the part of a research agent that decides what the model gets to
see. It reads every URL through Lyrenth, which returns each page as a clean
AIDocument with its canonical URL and a measured token count, and then:

- drops a page that was already read under another URL (mobile copies,
  tracking parameters, redirects all resolve to one canonical URL);
- shares the token budget between the pages, so one long page cannot crowd
  out the others, and trims a page that is still too long rather than
  losing it;
- records every page it could not read, with the reason, instead of
  quietly leaving a gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from lyrenth import Lyrenth

# The batch endpoint takes up to 20 URLs per call.
BATCH_SIZE = 20

# A share too small to say anything is worth nothing, so a page whose share
# falls under this is dropped and named instead of carried as a fragment.
MIN_USEFUL_TOKENS = 1_500


@dataclass
class Source:
    """One page the agent read, with everything a citation needs."""

    url: str
    title: str
    text: str
    fetched_at: str = ""
    tokens: int = 0
    raw_html_tokens: int = 0
    # True when the page was longer than its share of the budget and only
    # its opening survived. Everything that shows a source says so.
    trimmed: bool = False


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


def _allocate(sizes: List[int], budget: int) -> List[int]:
    """Share `budget` between pages, equally, with the leftovers passed on.

    Every page starts with the same allowance. A page shorter than its
    allowance takes only what it needs, and what it leaves is shared again
    between the pages still over theirs, until nothing more can be handed out.

    Reading the pages in order and giving each one whatever was left was the
    old rule, and it meant the first long page could take the whole budget
    and leave a second long page reported as over budget.
    """
    allowances = [0] * len(sizes)
    unsettled = list(range(len(sizes)))
    remaining = budget
    while unsettled:
        share = remaining // len(unsettled)
        settled = [i for i in unsettled if sizes[i] <= share]
        if not settled:
            for i in unsettled:
                allowances[i] = share
            break
        for i in settled:
            allowances[i] = sizes[i]
            remaining -= sizes[i]
            unsettled.remove(i)
    return allowances


def gather(
    urls: Iterable[str],
    client: Optional[Lyrenth] = None,
    token_budget: int = 30_000,
    fresh: bool = False,
) -> Gathered:
    """Read `urls` and return the sources, each with its share of the budget.

    Every page that can be read gets an equal share, and a page shorter than
    its share gives the rest back to the longer ones. Order still decides who
    is carried at all when there are more pages than the budget can hold: the
    later ones are dropped, and each says why.
    """
    client = client or Lyrenth()
    urls = _dedupe_input(urls)
    out = Gathered()
    seen_canonical = set()
    candidates: List[Source] = []

    for start in range(0, len(urls), BATCH_SIZE):
        batch = urls[start : start + BATCH_SIZE]
        for result in client.read_batch(batch, fresh=fresh):
            if not result.ok or result.document is None:
                # The explanation is the sentence a person can act on, the code
                # is only a label. getattr keeps this working against an older
                # installed SDK whose BatchResult has no message field.
                reason = getattr(result, "message", None) or result.error or "could not be read"
                out.skipped.append(Skipped(result.url, reason))
                continue
            src = _to_source(result.document)
            if not src.text.strip():
                out.skipped.append(Skipped(result.url, "page has no readable text"))
                continue
            if src.url in seen_canonical:
                out.skipped.append(Skipped(result.url, f"duplicate of {src.url}"))
                continue
            seen_canonical.add(src.url)
            candidates.append(src)

    no_room = f"no room left in the {token_budget:,} token budget"

    # The budget carries a fixed number of pages at most. Without this, twenty
    # pages against a small budget would leave twenty fragments and no usable
    # source, so the later ones are dropped and named.
    room_for = max(1, token_budget // MIN_USEFUL_TOKENS)
    for src in candidates[room_for:]:
        out.skipped.append(Skipped(src.url, no_room))
    candidates = candidates[:room_for]

    for src, allowance in zip(candidates, _allocate([s.tokens for s in candidates], token_budget)):
        if src.tokens > allowance:
            if allowance < MIN_USEFUL_TOKENS:
                out.skipped.append(Skipped(src.url, no_room))
                continue
            # Four characters to a token is the usual rough figure for English
            # prose, and it only has to be close: the cut is a budget guard,
            # not an exact measure. The opening of an article carries its
            # subject, and a trimmed source is marked everywhere it appears.
            src.text = src.text[: allowance * 4].rstrip()
            src.tokens = allowance
            src.trimmed = True
        out.sources.append(src)
        out.tokens += src.tokens
    return out
