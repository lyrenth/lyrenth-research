"""Command line: lyrenth-research "question" URL [URL ...]"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from lyrenth import Lyrenth, LyrenthError

from . import __version__
from .answer import ModelError, answer, build_context
from .sources import gather


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lyrenth-research",
        description="Read the given URLs through Lyrenth and answer the question with numbered citations.",
    )
    p.add_argument("question", help="the question to answer")
    p.add_argument("urls", nargs="*", help="source URLs, most trusted first")
    p.add_argument("-f", "--file", help="read more URLs from a file, one per line")
    p.add_argument("--budget", type=int, default=30_000, help="token budget for all sources together (default 30000)")
    p.add_argument("--fresh", action="store_true", help="ask Lyrenth for a fresh fetch instead of the stored page")
    p.add_argument("--sources-only", action="store_true", help="read the sources and print the cited context, without calling a model")
    p.add_argument("--json", action="store_true", help="print the result as JSON")
    p.add_argument("--model-url", help="OpenAI-compatible base URL (default: LLM_BASE_URL)")
    p.add_argument("--model", help="model name (default: LLM_MODEL)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def _urls(args) -> list:
    urls = list(args.urls)
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            urls += [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return urls


def _report(gathered, out=None) -> None:
    # Looked up at call time, so a caller that redirects stderr captures it.
    out = out or sys.stderr
    for s in gathered.skipped:
        print(f"skipped  {s.url}: {s.reason}", file=out)
    for i, s in enumerate(gathered.sources, 1):
        # A page can carry no title at all (an RFC served as one <pre>
        # block is the example), and an empty one left a hole in the line.
        label = f"{s.title}  " if s.title else ""
        print(f"read [{i}] {label}{s.tokens:,} tokens  {s.url}", file=out)
    raw = gathered.raw_html_tokens
    line = f"context  {gathered.tokens:,} tokens from {len(gathered.sources)} sources"
    if raw:
        line += f" (raw HTML would be {raw:,})"
    print(line, file=out)


def main(argv=None) -> int:
    # Intermixed, so an option may sit before, between or after the URLs.
    # Plain parse_args refused `"question" --budget N URL URL` on Python
    # 3.9, and an option between two URLs on every version (2026-09-24).
    args = _parser().parse_intermixed_args(argv)
    urls = _urls(args)
    if not urls:
        print("give at least one URL, or a file of URLs with --file", file=sys.stderr)
        return 2

    try:
        gathered = gather(urls, client=Lyrenth(), token_budget=args.budget, fresh=args.fresh)
    except LyrenthError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    _report(gathered)

    if args.sources_only:
        if args.json:
            print(json.dumps({"sources": [asdict(s) for s in gathered.sources],
                              "skipped": [asdict(s) for s in gathered.skipped]}, indent=2))
        else:
            print(build_context(gathered.sources))
        return 0

    try:
        result = answer(args.question, gathered.sources, base_url=args.model_url, model=args.model)
    except ModelError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({
            "question": args.question,
            "answer": result.text,
            "cited": result.cited,
            "unknown_citations": result.unknown_citations,
            "sources": [{"n": i, "title": s.title, "url": s.url, "read": s.fetched_at}
                        for i, s in enumerate(result.sources, 1)],
            "skipped": [asdict(s) for s in gathered.skipped],
        }, indent=2))
        return 0

    print(result.text)
    print()
    print("Sources")
    for i, s in enumerate(result.sources, 1):
        mark = "" if i in result.cited else "  (not cited)"
        label = f"{s.title}  " if s.title else ""
        print(f"  [{i}] {label}{s.url}{mark}")
    if result.unknown_citations:
        nums = ", ".join(f"[{n}]" for n in result.unknown_citations)
        print(f"\nwarning: the answer cites {nums}, which is not one of the sources above", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
