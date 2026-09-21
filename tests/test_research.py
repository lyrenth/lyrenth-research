"""Tests for lyrenth-research. No network and no keys needed.

    python tests/test_research.py      # plain, no pytest needed
    pytest tests/                     # if pytest is installed

Reading is tested against a fake Lyrenth client. The answer step is tested
against a real HTTP server on localhost that speaks the OpenAI-compatible
chat completions protocol, so the request the agent sends is checked for real.
"""

import io
import json
import os
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, HTTPServer

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
# Prefer the SDK from this monorepo when it is next door, else the installed one.
sdk = os.path.join(HERE, "..", "..", "lyrenth-python", "src")
if os.path.isdir(sdk):
    sys.path.insert(0, sdk)

from lyrenth import AIDocument, BatchResult  # noqa: E402

import lyrenth_research.cli as cli  # noqa: E402
from lyrenth_research import (  # noqa: E402
    ModelError,
    Source,
    answer,
    build_context,
    check_citations,
    gather,
)


def doc(canonical, title, text, tokens, raw=0, fetched="2026-09-19T12:00:00Z"):
    return AIDocument(
        url=canonical,
        title=title,
        markdown=text,
        raw={
            "source": {"url": canonical, "canonical_url": canonical, "fetched_at": fetched},
            "economics": {"output_tokens_approx": tokens, "raw_html_tokens_approx": raw},
        },
    )


class FakeClient:
    """Answers read_batch from a dict of url -> AIDocument, error string, or
    an (error code, explanation) pair for a failure that carries both."""

    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def read_batch(self, urls, fresh=False, max_tokens=None):
        self.calls.append(list(urls))
        out = []
        for u in urls:
            v = self.pages.get(u, "upstream_not_found")
            if isinstance(v, tuple):
                out.append(BatchResult(url=u, ok=False, error=v[0], message=v[1]))
            elif isinstance(v, str):
                out.append(BatchResult(url=u, ok=False, error=v))
            else:
                out.append(BatchResult(url=u, ok=True, document=v))
        return out


A = "https://example.com/a"
A_MOBILE = "https://m.example.com/a"
B = "https://example.com/b"
C = "https://example.com/c"
MISSING = "https://example.com/missing"
FORBIDDEN = "https://example.com/forbidden"
FORBIDDEN_MESSAGE = (
    "example.com returned 403 to our crawler. This is usually bot protection, "
    "geo / consent gating, or a CDN rule, not necessarily a login wall."
)


def pages():
    return {
        A: doc(A, "Page A", "Alpha text.", 3000, 25000),
        A_MOBILE: doc(A, "Page A", "Alpha text.", 3000, 25000),
        B: doc(B, "Page B", "Beta text.", 20000, 80000),
        C: doc(C, "Page C", "Gamma text.", 16000, 40000),
    }


# ---------------------------------------------------------------- gather

def test_gather_every_branch():
    # A is 3,000 tokens and takes only what it needs. The 27,000 left over is
    # split between B and C, which are both above their share, so all three
    # pages are carried and the two long ones are marked trimmed.
    g = gather([A, B, C, MISSING, A_MOBILE, A], client=FakeClient(pages()), token_budget=30000)
    assert [s.url for s in g.sources] == [A, B, C], g.sources
    assert [s.tokens for s in g.sources] == [3000, 13500, 13500]
    assert [s.trimmed for s in g.sources] == [False, True, True]
    assert g.tokens == 30000
    assert g.raw_html_tokens == 145000
    reasons = {s.url: s.reason for s in g.skipped}
    assert reasons[MISSING] == "upstream_not_found"
    assert reasons[A_MOBILE] == f"duplicate of {A}"
    # The exact same URL given twice is read once, not reported as skipped.
    assert len(g.skipped) == 2


def test_gather_reports_the_explanation_when_the_item_carries_one():
    # The code alone ("upstream_forbidden") tells a reader nothing about what
    # to do next, so the sentence wins when the API sends one.
    g = gather([FORBIDDEN], client=FakeClient({FORBIDDEN: ("upstream_forbidden", FORBIDDEN_MESSAGE)}))
    assert g.sources == []
    assert g.skipped[0].reason == FORBIDDEN_MESSAGE, g.skipped


def test_two_long_pages_both_make_it_in():
    # Giving the first page whatever was left meant a second long page was
    # dropped for want of room, and a question about two subjects came back
    # answered from one of them. Both now get half the budget.
    first = doc(A, "A", "alpha " * 20000, 40000)
    second = doc(B, "B", "beta " * 20000, 35000)
    g = gather([A, B], client=FakeClient({A: first, B: second}), token_budget=20000)
    assert [s.url for s in g.sources] == [A, B], g.skipped
    assert [s.tokens for s in g.sources] == [10000, 10000]
    assert all(s.trimmed for s in g.sources)
    assert g.skipped == []


def test_a_trimmed_page_says_so_in_the_context():
    # A model reading a partial page as if it were whole would cite it for
    # what is not there, so the note travels with the source.
    from lyrenth_research.answer import build_context

    long_doc = doc(A, "Long A", "x " * 20000, 40000)
    g = gather([A], client=FakeClient({A: long_doc}), token_budget=10000)
    assert g.sources[0].trimmed is True
    assert "longer than the budget" in build_context(g.sources)


def test_order_decides_who_is_carried_when_the_budget_is_small():
    # Five pages against a budget with room for two: the first two are read
    # and the three dropped say why.
    urls = [f"https://example.com/p{i}" for i in range(5)]
    small = {u: doc(u, u, "text " * 100, 9000) for u in urls}
    g = gather(urls, client=FakeClient(small), token_budget=4000)
    assert [s.url for s in g.sources] == urls[:2]
    assert [s.reason for s in g.skipped] == ["no room left in the 4,000 token budget"] * 3


def test_gather_batches_by_twenty():
    many = {f"https://example.com/{i}": doc(f"https://example.com/{i}", str(i), "x", 10) for i in range(45)}
    client = FakeClient(many)
    g = gather(list(many), client=client, token_budget=10**6)
    assert [len(c) for c in client.calls] == [20, 20, 5]
    assert len(g.sources) == 45


def test_gather_skips_empty_pages():
    p = {A: doc(A, "Empty", "   \n", 5)}
    g = gather([A], client=FakeClient(p))
    assert g.sources == [] and g.skipped[0].reason == "page has no readable text"


def test_gather_estimates_tokens_without_economics():
    d = AIDocument(url=A, title="A", markdown="x" * 400, raw={"source": {"url": A}})
    g = gather([A], client=FakeClient({A: d}))
    assert g.sources[0].tokens == 100


# ---------------------------------------------------------------- answer helpers

def test_build_context_numbers_and_keeps_provenance():
    ctx = build_context([Source(A, "Page A", "Alpha.", "2026-09-19T12:00:00Z"), Source(B, "", "Beta.")])
    assert ctx.startswith("[1] Page A\nURL: https://example.com/a\nRead: 2026-09-19T12:00:00Z\n\nAlpha.")
    assert "[2] https://example.com/b\nURL: https://example.com/b\n\nBeta." in ctx


def test_check_citations_flags_numbers_that_do_not_exist():
    cited, unknown = check_citations("Alpha [1]. Beta [2][2]. Nonsense [7]. Zero [0].", 2)
    assert cited == [1, 2] and unknown == [0, 7]


def test_answer_without_sources_does_not_call_a_model():
    r = answer("q", [], base_url="http://127.0.0.1:9", model="m")
    assert r.sources == [] and "nothing to answer from" in r.text


def test_missing_model_config_is_a_clear_error():
    saved = {k: os.environ.pop(k, None) for k in ("LLM_BASE_URL", "LLM_MODEL")}
    try:
        answer("q", [Source(A, "A", "Alpha.")])
    except ModelError as e:
        assert "--sources-only" in str(e)
    else:
        raise AssertionError("expected ModelError")
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


# ---------------------------------------------------------------- a real HTTP model endpoint

class _Model(BaseHTTPRequestHandler):
    received = []
    reply = "Alpha says hello [1]. Beta agrees [2]."

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Model.received.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": _Model.reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def _serve():
    srv = HTTPServer(("127.0.0.1", 0), _Model)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"


class _FussyModel(BaseHTTPRequestHandler):
    """An endpoint that refuses the temperature field, as some real ones do."""

    received = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FussyModel.received.append(body)
        if "temperature" in body:
            out = json.dumps({"error": {"message": "`temperature` is deprecated for this model."}}).encode()
            self.send_response(400)
        else:
            out = json.dumps(
                {"choices": [{"message": {"role": "assistant", "content": "Alpha [1]."}}]}
            ).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


def test_an_endpoint_that_refuses_temperature_still_answers():
    # "Anything that speaks the OpenAI-compatible protocol works" is the
    # promise in the README, and one of the large providers answers 400 to
    # the temperature field. Before this, that was a failed run.
    srv = HTTPServer(("127.0.0.1", 0), _FussyModel)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1"
    try:
        _FussyModel.received.clear()
        r = answer("What do they say?", [Source(A, "Page A", "Alpha.")], base_url=url, model="fussy")
        assert r.text == "Alpha [1]."
        assert len(_FussyModel.received) == 2
        assert "temperature" in _FussyModel.received[0]
        assert "temperature" not in _FussyModel.received[1]
    finally:
        srv.shutdown()


def test_answer_calls_an_openai_compatible_endpoint():
    srv, url = _serve()
    try:
        _Model.received.clear()
        _Model.reply = "Alpha says hello [1]. Beta agrees [2]. Also [5]."
        r = answer("What do they say?", [Source(A, "Page A", "Alpha."), Source(B, "Page B", "Beta.")],
                   base_url=url, model="test-model", api_key="secret")
        sent = _Model.received[0]
        assert sent["path"] == "/v1/chat/completions"
        assert sent["auth"] == "Bearer secret"
        assert sent["body"]["model"] == "test-model" and sent["body"]["temperature"] == 0
        user = sent["body"]["messages"][1]["content"]
        assert user.startswith("Question: What do they say?") and "[2] Page B" in user
        assert r.cited == [1, 2] and r.unknown_citations == [5]
    finally:
        srv.shutdown()


def test_cli_end_to_end_with_fake_reader_and_local_model():
    srv, url = _serve()
    real = cli.Lyrenth
    cli.Lyrenth = lambda: FakeClient(pages())
    try:
        _Model.reply = "Alpha is first [1]."
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(["What is first?", A, MISSING, B, "--model-url", url, "--model", "m"])
        assert code == 0, err.getvalue()
        assert "skipped  https://example.com/missing: upstream_not_found" in err.getvalue()
        assert "Alpha is first [1]." in out.getvalue()
        assert "[2] Page B  https://example.com/b  (not cited)" in out.getvalue()

        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(io.StringIO()):
            assert cli.main(["q", A, B, "--sources-only", "--json"]) == 0
        payload = json.loads(out.getvalue())
        assert [s["url"] for s in payload["sources"]] == [A, B]
    finally:
        cli.Lyrenth = real
        srv.shutdown()


def test_cli_needs_a_url():
    with redirect_stderr(io.StringIO()):
        assert cli.main(["question only"]) == 2


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
