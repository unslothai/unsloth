# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Web tool latency: image lookup beside the text search, and URL reads from one batch in flight at once.

Every test that checks an answer compares it with the serial path run in the same process, so the
overlap can only change when things happen, never what the model or the client sees.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import re
import threading
import time

import pytest

from core.inference import search_images
from core.inference import studio_tool_loop as loop_mod
from core.inference import tools
from core.inference.studio_tool_loop import ToolLoopPolicy, ToolLoopRun, stream_with_studio_tools


# ---------------------------------------------------------------------------------------------
# Tool loop harness
# ---------------------------------------------------------------------------------------------

WEB = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "url": {"type": "string"}},
        },
    },
}
PY = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "",
        "parameters": {"type": "object", "properties": {"code": {"type": "string"}}},
    },
}


def _sse(delta = None, finish = None) -> str:
    choice: dict = {"index": 0, "delta": delta or {}}
    if finish is not None:
        choice["finish_reason"] = finish
    return "data: " + json.dumps({"choices": [choice]})


def _batch(calls) -> list[str]:
    deltas = [
        {"index": i, "id": f"c{i}", "function": {"name": name, "arguments": json.dumps(args)}}
        for i, (name, args) in enumerate(calls)
    ]
    return [_sse({"tool_calls": deltas}), _sse(finish = "tool_calls"), "data: [DONE]"]


class Transport:
    heals_text_tool_calls = True

    def __init__(self, calls):
        self.turns = [
            _batch(calls),
            [_sse({"content": "done"}), _sse(finish = "stop"), "data: [DONE]"],
        ]
        self.requests: list[list[dict]] = []

    def stream(self, *, messages, tools, tool_choice, cancel_event):
        self.requests.append([dict(m) for m in messages])
        lines = self.turns.pop(0) if self.turns else ["data: [DONE]"]

        async def gen():
            for line in lines:
                yield line

        return gen()


class Recorder:
    """execute_tool stand-in: records each call's run interval and the peak number running at once."""

    def __init__(
        self,
        delay = 0.4,
        wait_for_cancel = False,
    ):
        self.delay = delay
        self.wait_for_cancel = wait_for_cancel
        self.lock = threading.Lock()
        self.running = 0
        self.peak = 0
        self.spans: dict[str, tuple[float, float]] = {}
        self.calls: list[str] = []
        self.saw_cancel: list[str] = []

    def __call__(self, name, arguments, **kwargs):
        label = arguments.get("url") or arguments.get("code") or arguments.get("query")
        with self.lock:
            self.calls.append(label)
            self.running += 1
            self.peak = max(self.peak, self.running)
        start = time.perf_counter()
        try:
            cancel = kwargs.get("cancel_event")
            if self.wait_for_cancel:
                if cancel.wait(5.0):
                    self.saw_cancel.append(label)
                    return "cancelled"
            else:
                time.sleep(self.delay if name == "web_search" else 0.05)
            return f"RESULT {name} {label}"
        finally:
            with self.lock:
                self.running -= 1
            self.spans[label] = (start, time.perf_counter())


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(loop_mod, "execute_tool", rec)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    return rec


def _run(
    transport,
    *,
    cancel_event = None,
    **policy,
):
    fields = {
        "tools": [WEB, PY],
        "max_calls": 25,
        "timeout": 60,
        "permission_mode": "off",
        "confirm_calls": False,
        "bypass_permissions": False,
        "rag_scope": None,
    }
    fields.update(policy)
    cancel_event = cancel_event or threading.Event()

    async def collect():
        out = []
        async for line in stream_with_studio_tools(
            transport,
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "read"}], session_id = "s1", thread_id = "t1"
            ),
            policy = ToolLoopPolicy(**fields),
            cancel_event = cancel_event,
        ):
            out.append(line)
        return out

    t0 = time.perf_counter()
    lines = asyncio.run(asyncio.wait_for(collect(), timeout = 30))
    elapsed = time.perf_counter() - t0
    # Approval ids are minted fresh per call; keep-alives follow the clock.
    lines = [
        re.sub(r'"approval_id":"[^"]+"', '"approval_id":"<id>"', line)
        for line in lines
        if line != loop_mod._SSE_KEEPALIVE
    ]
    return lines, elapsed


def _serial_and_overlapped(monkeypatch, recorder, calls, **policy):
    """The same batch run serially (concurrency 1, the old loop) and with overlap allowed."""
    monkeypatch.setattr(loop_mod, "_WEB_FETCH_CONCURRENCY", 1, raising = False)
    serial_transport = Transport(calls)
    serial_lines, serial_s = _run(serial_transport, **policy)
    serial_peak, serial_calls = recorder.peak, list(recorder.calls)
    recorder.peak, recorder.calls = 0, []
    monkeypatch.setattr(loop_mod, "_WEB_FETCH_CONCURRENCY", 4, raising = False)
    transport = Transport(calls)
    cancel = threading.Event()
    lines, overlapped_s = _run(transport, cancel_event = cancel, **policy)
    # A batch that finished cleanly never cancels the turn it belongs to.
    assert not cancel.is_set()
    # Same events in the same order, and the same history replayed to the model.
    assert lines == serial_lines
    assert transport.requests == serial_transport.requests
    # Same calls executed; start order among calls running together is not observable.
    assert sorted(recorder.calls) == sorted(serial_calls)
    return serial_peak, serial_s, overlapped_s, lines


def _url(host, path = "p"):
    return ("web_search", {"url": f"https://{host}/{path}"})


def _tool_ends(lines):
    out = []
    for line in lines:
        if line.startswith("data: {"):
            payload = json.loads(line[6:])
            if payload.get("type") == "tool_end":
                out.append((payload["tool_call_id"], payload["result"]))
    return out


def test_independent_url_reads_run_together_and_replay_in_order(monkeypatch, recorder):
    calls = [_url("a.example"), _url("b.example"), _url("c.example")]
    serial_peak, serial_s, overlapped_s, lines = _serial_and_overlapped(
        monkeypatch, recorder, calls
    )
    assert serial_peak == 1
    assert recorder.peak == 3
    # Three 0.4 s reads: about 1.2 s one after another, about 0.4 s together.
    assert overlapped_s < serial_s - 0.5, (serial_s, overlapped_s)
    assert [cid for cid, _ in _tool_ends(lines)] == ["c0", "c1", "c2"]


def test_reads_of_one_host_stay_serial(monkeypatch, recorder):
    calls = [_url("a.example", "1"), _url("a.example", "2"), _url("b.example")]
    _serial_and_overlapped(monkeypatch, recorder, calls)
    a1, a2 = recorder.spans["https://a.example/1"], recorder.spans["https://a.example/2"]
    assert a2[0] >= a1[1]
    # b.example may start beside a.example/2, never beside a.example/1.
    assert recorder.spans["https://b.example/p"][0] >= a1[1]


def test_non_web_call_is_never_overlapped(monkeypatch, recorder):
    calls = [_url("a.example"), ("python", {"code": "print(1)"}), _url("b.example")]
    _serial_and_overlapped(monkeypatch, recorder, calls)
    a, py, b = (
        recorder.spans[k] for k in ("https://a.example/p", "print(1)", "https://b.example/p")
    )
    assert a[1] <= py[0] and py[1] <= b[0]
    assert recorder.peak == 1


def test_web_search_queries_are_not_overlapped(monkeypatch, recorder):
    calls = [_url("a.example"), ("web_search", {"query": "x"}), _url("b.example")]
    _serial_and_overlapped(monkeypatch, recorder, calls)
    assert recorder.peak == 1


def test_reads_that_ask_first_are_never_started_early(monkeypatch, recorder):
    # Auto mode asks before every URL read; each is answered Allow, after the previous one finished.
    answers: list[float] = []

    def allow(slot, approval_id, cancel_event):
        answers.append(time.perf_counter())
        return "allow"

    monkeypatch.setattr(loop_mod, "wait_tool_decision", allow)
    calls = [_url("a.example"), _url("b.example"), _url("c.example")]
    _serial_and_overlapped(monkeypatch, recorder, calls, permission_mode = "auto", confirm_calls = True)
    assert recorder.peak == 1
    starts = sorted(span[0] for span in recorder.spans.values())
    # Each read starts only after its own approval.
    assert all(start >= answer for start, answer in zip(starts, answers[-3:]))


def test_budget_is_reserved_before_reads_start(monkeypatch, recorder):
    calls = [_url("a.example"), _url("b.example"), _url("c.example")]
    _serial_and_overlapped(monkeypatch, recorder, calls, max_calls = 2)
    # The third read is answered by the budget card in both runs and never executes.
    assert sorted(recorder.calls) == ["https://a.example/p", "https://b.example/p"]


def test_repeated_url_keeps_duplicate_handling(monkeypatch, recorder):
    calls = [_url("a.example"), _url("b.example"), _url("a.example")]
    _serial_and_overlapped(monkeypatch, recorder, calls)
    assert sorted(recorder.calls) == ["https://a.example/p", "https://b.example/p"]


def test_cancel_stops_started_reads_promptly(monkeypatch):
    rec = Recorder(wait_for_cancel = True)
    monkeypatch.setattr(loop_mod, "execute_tool", rec)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    cancel = threading.Event()
    threading.Timer(0.3, cancel.set).start()
    calls = [_url("a.example"), _url("b.example"), _url("c.example")]
    lines, elapsed = _run(Transport(calls), cancel_event = cancel)
    assert elapsed < 2.0
    # The started reads saw the cancel too, and only the call the serial loop had reached is recorded.
    assert sorted(rec.saw_cancel) == sorted(rec.calls)
    assert [cid for cid, _ in _tool_ends(lines)] == ["c0"]
    deadline = time.monotonic() + 2
    while rec.running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert rec.running == 0


def test_closing_the_stream_between_calls_stops_started_reads(monkeypatch):
    """Closed at a yield where no tool stream is open (the next card's tool_start), the reads started
    for later calls must stop as a closed running stream does, not run on against a gone turn."""
    lock = threading.Lock()
    running: list = []
    seen: list = []

    def execute(name, arguments, **kwargs):
        url = arguments["url"]
        with lock:
            seen.append(url)
            running.append(url)
        try:
            if "a.example" in url:
                time.sleep(0.2)
            else:
                kwargs["cancel_event"].wait(5.0)
            return f"RESULT {url}"
        finally:
            with lock:
                running.remove(url)

    monkeypatch.setattr(loop_mod, "execute_tool", execute)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    cancel = threading.Event()
    calls = [_url("a.example"), _url("b.example"), _url("c.example")]

    async def close_at_second_card():
        gen = stream_with_studio_tools(
            Transport(calls),
            run = ToolLoopRun(
                messages = [{"role": "user", "content": "read"}], session_id = "s1", thread_id = "t1"
            ),
            policy = ToolLoopPolicy(
                tools = [WEB, PY],
                max_calls = 25,
                timeout = 60,
                permission_mode = "off",
                confirm_calls = False,
                bypass_permissions = False,
                rag_scope = None,
            ),
            cancel_event = cancel,
        )
        starts = 0
        async for line in gen:
            if '"tool_start"' in line:
                starts += 1
                if starts == 2:
                    break
        await gen.aclose()

    asyncio.run(asyncio.wait_for(close_at_second_card(), timeout = 10))
    # The reads for c1 and c2 were started early, so the overlap was exercised.
    assert len(seen) == 3
    deadline = time.monotonic() + 2
    while running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert running == []
    assert cancel.is_set()


# ---------------------------------------------------------------------------------------------
# Real guarded fetcher through the overlapped path
# ---------------------------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        host = (self.headers.get("Host") or "").split(":")[0]
        if host.startswith("loop"):
            # Redirects to itself forever: the hop limit must end it.
            self._redirect(f"http://{self.headers['Host']}/again")
        elif host.startswith("private"):
            self._redirect("http://169.254.169.254/latest/meta-data/")
        else:
            body = (
                b"<html><body><main><h1>Page</h1><p>" + b"word " * 80 + b"</p></main></body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def _redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def local_sites(monkeypatch):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    threading.Thread(target = server.serve_forever, daemon = True).start()
    real = tools._validate_and_resolve_host

    def resolve(host, port):
        # Only the test's own public-looking names reach loopback; everything else is checked for real.
        if host.endswith(".sites.example"):
            return True, "", ["127.0.0.1"]
        return real(host, port)

    monkeypatch.setattr(tools, "_validate_and_resolve_host", resolve)
    monkeypatch.setattr(loop_mod, "build_rag_autoinject", lambda *a, **k: None)
    yield server.server_address[1]
    server.shutdown()
    server.server_close()


def test_guards_hold_on_every_hop_of_overlapped_reads(monkeypatch, local_sites):
    port = local_sites
    calls = [
        ("web_search", {"url": f"http://ok.sites.example:{port}/"}),
        ("web_search", {"url": f"http://private.sites.example:{port}/"}),
        ("web_search", {"url": f"http://loop.sites.example:{port}/"}),
    ]
    seen = []
    real_execute = loop_mod.execute_tool

    def counting(name, arguments, **kwargs):
        seen.append(arguments["url"])
        return real_execute(name, arguments, **kwargs)

    monkeypatch.setattr(loop_mod, "execute_tool", counting)
    monkeypatch.setattr(loop_mod, "_WEB_FETCH_CONCURRENCY", 1, raising = False)
    serial_lines, _ = _run(Transport(calls))
    monkeypatch.setattr(loop_mod, "_WEB_FETCH_CONCURRENCY", 4, raising = False)
    lines, _ = _run(Transport(calls))
    assert lines == serial_lines
    results = dict(_tool_ends(lines))
    assert "Page" in results["c0"]
    # The redirect hop to the metadata address is refused, not followed.
    assert results["c1"].startswith("Blocked")
    assert results["c2"] == "Failed to fetch URL: too many redirects."
    assert len(seen) == 6


# ---------------------------------------------------------------------------------------------
# Image lookup beside the text search
# ---------------------------------------------------------------------------------------------


class _FakeDDGS:
    text_s = 0.3
    image_s = 0.3
    text_error: Exception | None = None
    image_queries: list = []

    def __init__(self, *a, **k):
        pass

    def text(self, query, **k):
        time.sleep(self.text_s)
        if self.text_error is not None:
            raise self.text_error
        return [
            {"title": f"T{i}", "href": f"https://site{i}.example.com/a", "body": f"snippet {i}"}
            for i in range(k.get("max_results") or 5)
        ]

    def images(self, query, **k):
        type(self).image_queries.append(query)
        time.sleep(self.image_s)
        return [
            {
                "title": f"{query} {i}",
                "image": f"https://img{i}.example.org/p.jpg",
                "thumbnail": f"https://img{i}.example.org/t.jpg",
                "url": f"https://src{i}.example.org/page",
            }
            for i in range(k.get("max_results") or 6)
        ]


@pytest.fixture
def fake_ddgs(monkeypatch, tmp_path):
    import ddgs

    monkeypatch.setattr(ddgs, "DDGS", _FakeDDGS)
    monkeypatch.setattr(_FakeDDGS, "text_error", None)
    monkeypatch.setattr(_FakeDDGS, "image_queries", [])
    monkeypatch.setattr(
        tools, "_ddgs_text", lambda client, q, n, backend: client.text(q, max_results = n)
    )
    monkeypatch.setattr(tools, "_resolve_engine_tiers", lambda engines: ["auto"])
    # A failure recorded by another test must not hold this one's lookups back.
    monkeypatch.setattr(tools, "_text_search_failed_at", None, raising = False)
    monkeypatch.setattr(search_images, "_cache_dir", lambda: tmp_path)
    # Registrations made here must not leak into other tests' registry.
    monkeypatch.setattr(search_images, "_registry", {})
    monkeypatch.setattr(search_images, "_inflight", {})
    monkeypatch.setattr(search_images, "_cleared_unservable", set())
    registered: list = []
    real_register = search_images.register_images

    def register(*a, **k):
        entries = real_register(*a, **k)
        registered.append(len(entries))
        return entries

    monkeypatch.setattr(search_images, "register_images", register)
    return registered


def _norm(text: str) -> str:
    # Image ids are random per registration.
    return re.sub(r"\b[0-9a-f]{12}\b", "<id>", text)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"include_images": True},
        {"include_images": True, "image_queries": ["eiffel tower", "louvre"]},
    ],
    ids = ["generic", "named"],
)
def test_image_lookup_overlaps_text_search(monkeypatch, fake_ddgs, kwargs):
    t0 = time.perf_counter()
    overlapped = tools._web_search("paris", timeout = 20, **kwargs)
    overlapped_s = time.perf_counter() - t0
    # No sweep: the lookup runs after the text search, as it always did.
    if hasattr(tools, "_ImageSweep"):
        monkeypatch.setattr(tools._ImageSweep, "start", classmethod(lambda cls, *a: None))
    t0 = time.perf_counter()
    serial = tools._web_search("paris", timeout = 20, **kwargs)
    serial_s = time.perf_counter() - t0
    assert "[[img:" in overlapped or "img" in overlapped
    assert _norm(overlapped) == _norm(serial)
    # 0.3 s text + 0.3 s images one after the other; together about 0.3 s.
    assert overlapped_s < serial_s - 0.2, (serial_s, overlapped_s)


def test_text_only_search_starts_no_image_lookup(fake_ddgs):
    tools._web_search("paris", timeout = 20, include_images = False)
    assert _FakeDDGS.image_queries == []


@pytest.mark.parametrize("named", [False, True], ids = ["generic", "named"])
def test_discarded_image_lookup_registers_nothing(monkeypatch, fake_ddgs, named):
    # A genuine text failure with no fallback: the serial path never reached the images.
    monkeypatch.setattr(_FakeDDGS, "text_error", RuntimeError("engine exploded"))
    monkeypatch.setattr(
        tools, "_wikipedia_search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
    )
    extra = {"image_queries": ["eiffel tower"]} if named else {}
    result = tools._web_search("paris", timeout = 20, include_images = True, **extra)
    assert result.startswith("Search failed") and "img" not in result
    assert fake_ddgs == []


def test_wikipedia_fallback_drops_generic_images_unregistered(monkeypatch, fake_ddgs):
    monkeypatch.setattr(_FakeDDGS, "text_error", RuntimeError("engine exploded"))
    monkeypatch.setattr(
        tools,
        "_wikipedia_search",
        lambda *a, **k: [
            {"title": "Paris", "href": "https://en.wikipedia.org/wiki/Paris", "body": "x"}
        ],
    )
    result = tools._web_search("paris", timeout = 20, include_images = True)
    assert "Wikipedia-only" in result and "[[img:" not in result
    assert fake_ddgs == []


def test_cancel_during_text_search_discards_images(monkeypatch, fake_ddgs):
    cancel = threading.Event()
    threading.Timer(0.1, cancel.set).start()
    result = tools._web_search("paris", timeout = 20, include_images = True, cancel_event = cancel)
    assert result == "Search cancelled."
    assert fake_ddgs == []


def test_failing_text_engines_hold_back_the_early_image_lookup(monkeypatch, fake_ddgs):
    """While text search keeps failing (a rate limited machine), only the first failure pays a discarded
    lookup; the next searches send none until the engines answer again."""
    monkeypatch.setattr(_FakeDDGS, "text_error", RuntimeError("202 Ratelimit"))
    monkeypatch.setattr(
        tools, "_wikipedia_search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
    )
    for named in (False, True, False):
        extra = {"image_queries": ["eiffel tower"]} if named else {}
        result = tools._web_search("paris", timeout = 20, include_images = True, **extra)
        assert result.startswith("Search failed") and "img" not in result
    time.sleep(0.5)
    assert len(_FakeDDGS.image_queries) == 1
    # An empty sweep is an answer, not a failure: it lifts the hold, and so do results.
    monkeypatch.setattr(_FakeDDGS, "text_error", None)
    found = tools._web_search("paris", timeout = 20, include_images = True)
    assert "img" in found
    assert tools._text_search_failed_at is None
    before = len(_FakeDDGS.image_queries)
    t0 = time.perf_counter()
    tools._web_search("paris", timeout = 20, include_images = True)
    # Overlapped again: about one 0.3 s round trip, not two.
    assert time.perf_counter() - t0 < 0.5
    assert len(_FakeDDGS.image_queries) == before + 1
    # The hold expires on its own too.
    monkeypatch.setattr(_FakeDDGS, "text_error", RuntimeError("202 Ratelimit"))
    tools._web_search("paris", timeout = 20, include_images = True)
    monkeypatch.setattr(
        tools, "_text_search_failed_at", time.monotonic() - tools._TEXT_SEARCH_FAILURE_HOLD_S - 1
    )
    before = len(_FakeDDGS.image_queries)
    tools._web_search("paris", timeout = 20, include_images = True)
    time.sleep(0.5)
    assert len(_FakeDDGS.image_queries) == before + 1
