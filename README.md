# lyrenth-research

A research agent that actually reads its sources.

Every AI answer that reads the web pays for reading pages, and Lyrenth
makes that reading several times cheaper. This agent is the easiest way to
see it.

Give it a question and a list of URLs. It reads every page through
[Lyrenth](https://lyrenth.com) as clean text, drops duplicates, stays inside
a token budget, tells you which pages it could not read, and asks your model
for an answer with numbered citations. Then it checks that every citation
points at a source it really read.

```bash
pip install lyrenth-research
export LYRENTH_API_KEY=...            # free key: https://lyrenth.com/signup
export LLM_BASE_URL=http://localhost:11434/v1   # any OpenAI-compatible endpoint
export LLM_MODEL=your-model-name

lyrenth-research "What is the difference between a web crawler and web indexing?" \
  https://en.wikipedia.org/wiki/Web_indexing \
  https://en.wikipedia.org/wiki/Web_crawler
```

## Why

Most research agents get the finding and the writing right and treat the
reading as one line: fetch the page, strip some tags, paste it into the
prompt. That line decides more about the answer than the prompt does. An
agent that reads badly pays for navigation menus and cookie banners, reads
the same article twice under two URLs, cites pages it never really read,
and quietly answers from nothing when a source fails.

This agent does five things on purpose:

1. **Reads clean text, not HTML.** Every page arrives as an AIDocument:
   Markdown content, the canonical URL, when it was read, and how many
   tokens it costs.
2. **Keeps provenance.** Each source keeps its title, canonical URL and
   read time, and all three go into the prompt.
3. **Reads each document once.** A mobile copy, a tracking parameter or an
   old redirect resolves to the same canonical URL and is dropped.
4. **Shares the budget between the pages.** Every page gets an equal share
   of the token budget, a short page hands its leftover to the long ones,
   and a page that is still too long is trimmed and marked rather than
   dropped. Order still decides who is carried when there are more pages
   than the budget can hold, so put the ones you trust most first.
5. **Fails visibly.** Every page that was skipped is listed with the reason,
   and a citation to a source number that does not exist is flagged.

## A real run

Reading step only (`--sources-only`), against the live API on 21 September
2026, unedited. The progress report goes to stderr, the numbered context to
stdout (here, `context.md`).

```text
$ lyrenth-research "What is the difference between a web crawler and web indexing?" \
    https://en.wikipedia.org/wiki/Web_indexing \
    https://en.wikipedia.org/wiki/Web_crawler \
    https://en.wikipedia.org/wiki/Robots.txt \
    https://en.wikipedia.org/wiki/No_such_page_for_this_example \
    https://en.m.wikipedia.org/wiki/Web_indexing \
    --sources-only > context.md
skipped  https://en.wikipedia.org/wiki/No_such_page_for_this_example: en.wikipedia.org does not have a page at this URL (404). Check the URL for typos, or try the site's homepage to see if the path has changed.
skipped  https://en.m.wikipedia.org/wiki/Web_indexing: duplicate of https://en.wikipedia.org/wiki/Web_indexing
read [1] Web indexing - Wikipedia  3,119 tokens  https://en.wikipedia.org/wiki/Web_indexing
read [2] Web crawler - Wikipedia  13,440 tokens  https://en.wikipedia.org/wiki/Web_crawler
read [3] robots.txt - Wikipedia  13,440 tokens  https://en.wikipedia.org/wiki/Robots.txt
context  29,999 tokens from 3 sources (raw HTML would be 193,244)
```

The same question with a model configured, run the same day through an
OpenAI-compatible endpoint. The reading report is identical; this is the
answer, shortened for length and otherwise as printed:

```text
## Web Crawler vs. Web Indexing

**Web crawler** is the software/tool that does the *fetching*. A web crawler (also called a spider or spiderbot) is an internet bot that systematically browses the World Wide Web, typically operated by search engines [2]. It starts with a list of seed URLs, visits them, identifies hyperlinks on the retrieved pages, and adds those to a queue (the "crawl frontier") to visit recursively [2]. In short, its job is to discover and download web pages so they can be processed later [2].

...

### How they relate
- Crawlers **copy pages** for processing by a search engine, and the search engine then **indexes** those downloaded pages so users can search more efficiently [2].
- In effect: crawling = discovering and downloading content; indexing = organizing that content (e.g., via keywords/metadata) into a structure that supports search queries [1][2].

Sources
  [1] Web indexing - Wikipedia  https://en.wikipedia.org/wiki/Web_indexing
  [2] Web crawler - Wikipedia  https://en.wikipedia.org/wiki/Web_crawler
  [3] robots.txt - Wikipedia  https://en.wikipedia.org/wiki/Robots.txt
```

Every claim carries a source number, all three sources were cited, and none
is marked `(not cited)`.

## Two keys, and why

- **`LYRENTH_API_KEY`** reads the pages. The free tier includes 2,000 reads
  a month; one question with five sources uses five.
- **Your model** writes the answer. Anything that speaks the
  OpenAI-compatible chat completions protocol works: the large hosted
  providers, most gateways, and local servers such as Ollama or llama.cpp.
  Set `LLM_BASE_URL` and `LLM_MODEL`, plus `LLM_API_KEY` if your endpoint
  needs one. Nothing is sent anywhere else.

Only need the reading? `--sources-only` prints the numbered, cited context
without calling a model, ready to drop into your own pipeline. Add `--json`
for structured output.

## Options

```text
lyrenth-research QUESTION [URL ...] [options]

  -f, --file FILE     more URLs, one per line (lines starting with # are ignored)
  --budget N          token budget for all sources together (default 30000)
  --fresh             ask for a fresh fetch instead of the stored page
  --sources-only      read and print the context, no model call
  --json              JSON output
  --model-url URL     OpenAI-compatible base URL (default: LLM_BASE_URL)
  --model NAME        model name (default: LLM_MODEL)
```

## As a library

```python
from lyrenth_research import gather, answer

gathered = gather(
    ["https://en.wikipedia.org/wiki/Web_indexing",
     "https://en.wikipedia.org/wiki/Web_crawler"],
    token_budget=30_000,
)
for skipped in gathered.skipped:
    print("skipped", skipped.url, skipped.reason)

result = answer("How do crawlers and indexes relate?", gathered.sources)
print(result.text)
print("cited:", result.cited, "unknown:", result.unknown_citations)
```

`gather` returns the sources it read, the ones it skipped with a reason, and
the tokens used. `answer` returns the model's text, the source numbers it
cited, and any cited numbers that do not exist.

## Where the URLs come from

On purpose, not from here. Your agent may get them from a search provider,
from the user, from links inside pages it already read, or from a curated
list of trusted sites. Finding and reading are separate jobs; keeping them
separate means you can change one without breaking the other.

## Tests

```bash
pip install -e .
python tests/test_research.py
```

No network and no keys needed: reading is tested against a fake client, and
the answer step against a local HTTP server that speaks the chat
completions protocol.

## License

MIT
