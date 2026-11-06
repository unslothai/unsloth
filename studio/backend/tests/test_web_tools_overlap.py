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
import sys
import threading
import time
from pathlib import Path

import pytest

from core.inference import llama_chat
from core.inference import search_images
from core.inference import studio_tool_loop as loop_mod
from core.inference import tools
from core.inference.studio_tool_loop import ToolLoopPolicy, ToolLoopRun, stream_with_studio_tools

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_llama_cpp_tool_loop import _make_backend, _sse as local_sse  # noqa: E402


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
            content_type = "text/html"
            body = (
                b"<html><body><main><h1>Page</h1><p>" + b"word " * 80 + b"</p></main></body></html>"
            )
            if host.startswith("pdf"):
                content_type, body = "application/pdf", b"%PDF-1.4\n" + b"0" * 64
            if host.startswith("links"):
                links = b"".join(
                    b'<li><a href="/docs/' + b"section-%d/" % i * 12 + b'">part</a></li>'
                    for i in range(12)
                )
                body = (
                    b"<html><body><main><h1>Page</h1><ul>"
                    + links
                    + b"</ul><p>LAST_FACT_MARKER</p></main></body></html>"
                )
            self.send_response(200)
            self.send_header("Content-Type", content_type)
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


class _SlowPages:
    def __init__(self, monkeypatch):
        self.lock = threading.Lock()
        self.running = 0
        self.peak = 0
        self.fetched = []
        real = tools._download_page

        def fetch(url, *args, **kwargs):
            with self.lock:
                self.running += 1
                self.peak = max(self.peak, self.running)
                self.fetched.append(url)
            cancel = kwargs.get("cancel_event")
            deadline = time.monotonic() + 0.4
            while time.monotonic() < deadline and not (cancel and cancel.is_set()):
                time.sleep(0.01)
            with self.lock:
                self.running -= 1
            if cancel and cancel.is_set():
                return "Fetch cancelled."
            return real(url, *args, **kwargs)

        monkeypatch.setattr(tools, "_download_page", fetch)


def _local_turn(
    monkeypatch,
    calls,
    cancel_event = None,
    wait_denies = False,
    **loop_kwargs,
):
    if wait_denies:
        monkeypatch.setattr(llama_chat, "wait_tool_decision", lambda *a, **k: "deny", raising = False)

    frames = [
        local_sse(
            {
                "tool_calls": [
                    {
                        "index": index,
                        "id": call["id"],
                        "type": "function",
                        "function": call["function"],
                    }
                ]
            }
        )
        for index, call in enumerate(calls)
    ]
    finish = 'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n'
    streams = [
        [*frames, finish, "data: [DONE]\n\n"],
        [local_sse({"content": "done"}), "data: [DONE]\n\n"],
    ]
    payloads = []
    backend = _make_backend(monkeypatch, streams, payloads)
    backend.count_chat_tokens = lambda *a, **k: 100
    events = []
    started = time.monotonic()
    for event in backend.generate_chat_completion_with_tools(
        messages = [{"role": "user", "content": "go"}],
        tools = [WEB, PY],
        max_tool_iterations = 2,
        tool_call_timeout = 30,
        cancel_event = cancel_event or threading.Event(),
        **{"permission_mode": "off", **loop_kwargs},
    ):
        if isinstance(event, dict) and event.get("type") in ("tool_start", "tool_end"):
            events.append({key: event.get(key) for key in ("type", "tool_call_id", "result")})
    return events, payloads, time.monotonic() - started


def _local_url(index, host, port):
    return {
        "id": f"c{index}",
        "function": {
            "name": "web_search",
            "arguments": json.dumps({"url": f"http://{host}.sites.example:{port}/p"}),
        },
    }


def _tool_replay(payload):
    return [m for m in payload["messages"] if m.get("role") in ("assistant", "tool")]


def test_local_loop_reads_run_together_and_replay_in_order(monkeypatch, local_sites):
    calls = [_local_url(index, host, local_sites) for index, host in enumerate("abc")]
    slow = _SlowPages(monkeypatch)

    events, payloads, _ = _local_turn(monkeypatch, calls)

    assert slow.peak == 3
    assert [(e["type"], e["tool_call_id"]) for e in events] == [
        (kind, f"c{index}") for index in range(3) for kind in ("tool_start", "tool_end")
    ]
    ends = [e["result"] for e in events if e["type"] == "tool_end"]
    assert all("word word" in result for result in ends)
    tool_messages = [m for m in _tool_replay(payloads[1]) if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["c0", "c1", "c2"]


def test_local_loop_fits_each_started_read_to_its_own_turn(monkeypatch, local_sites):
    calls = [_local_url(index, host, local_sites) for index, host in enumerate("ab")]
    budgets = []
    real = tools.execute_tool

    def execute(name, arguments, **kwargs):
        budgets.append(kwargs.get("result_budget_tokens"))
        kwargs["result_budget_tokens"] = 12
        return real(name, arguments, **kwargs)

    monkeypatch.setattr(tools, "execute_tool", execute)
    _SlowPages(monkeypatch)

    events, _, _ = _local_turn(monkeypatch, calls)

    assert len(budgets) == 2 and all(budget is not None for budget in budgets)
    ends = [e["result"] for e in events if e["type"] == "tool_end"]
    assert len(ends) == 2 and all(len(result) < 400 for result in ends)


def test_local_loop_renders_a_started_read_under_its_own_budget(monkeypatch, local_sites):
    calls = [_local_url(0, "a", local_sites), _local_url(1, "links", local_sites)]
    real = tools.execute_tool

    def execute(name, arguments, **kwargs):
        kwargs["result_budget_tokens"] = 200
        return real(name, arguments, **kwargs)

    monkeypatch.setattr(tools, "execute_tool", execute)
    slow = _SlowPages(monkeypatch)
    monkeypatch.setattr(loop_mod, "_WEB_FETCH_CONCURRENCY", 1, raising = False)
    serial, _, _ = _local_turn(monkeypatch, calls)
    assert slow.peak == 1
    monkeypatch.setattr(loop_mod, "_WEB_FETCH_CONCURRENCY", 4, raising = False)
    overlapped, _, _ = _local_turn(monkeypatch, calls)

    assert slow.peak == 2
    assert overlapped == serial
    assert "LAST_FACT_MARKER" in [e["result"] for e in overlapped if e["type"] == "tool_end"][1]


def test_local_loop_extracts_started_pdfs_one_at_a_time(monkeypatch, local_sites):
    from core.rag import parsers

    calls = [_local_url(0, "pdfa", local_sites), _local_url(1, "pdfb", local_sites)]
    lock = threading.Lock()
    state = {"running": 0, "peak": 0, "calls": 0}

    def parse(data, max_pages = None):
        with lock:
            state["calls"] += 1
            state["running"] += 1
            state["peak"] = max(state["peak"], state["running"])
        time.sleep(0.3)
        with lock:
            state["running"] -= 1
        return [], 1

    monkeypatch.setattr(parsers, "parse_pdf_bytes", parse)
    slow = _SlowPages(monkeypatch)

    _local_turn(monkeypatch, calls)

    assert slow.peak == 2
    assert state["calls"] == 2
    assert state["peak"] == 1


def test_local_loop_pdf_read_does_not_time_out_waiting_for_another(monkeypatch, local_sites):
    from core.rag import parsers

    calls = [_local_url(0, "pdfa", local_sites), _local_url(1, "pdfb", local_sites)]

    def parse(data, max_pages = None):
        time.sleep(1.0)
        return [], 1

    monkeypatch.setattr(parsers, "parse_pdf_bytes", parse)
    # Under two parses, well over one.
    monkeypatch.setattr(tools, "_url_read_timeout", lambda timeout: 1.8)
    slow = _SlowPages(monkeypatch)

    events, _, _ = _local_turn(monkeypatch, calls)

    assert slow.peak == 2
    ends = [e["result"] for e in events if e["type"] == "tool_end"]
    assert ends == ["(PDF contains no extractable text)"] * 2


def test_local_loop_reads_of_one_host_stay_serial(monkeypatch, local_sites):
    calls = [
        _local_url(0, "a", local_sites),
        {
            "id": "c1",
            "function": {
                "name": "web_search",
                "arguments": json.dumps({"url": f"http://a.sites.example:{local_sites}/q"}),
            },
        },
    ]
    slow = _SlowPages(monkeypatch)

    _local_turn(monkeypatch, calls)

    assert slow.peak == 1
    assert len(slow.fetched) == 2


def test_local_loop_read_that_asks_first_is_not_started(monkeypatch, local_sites):
    calls = [_local_url(index, host, local_sites) for index, host in enumerate("ab")]
    slow = _SlowPages(monkeypatch)

    events, _, _ = _local_turn(
        monkeypatch, calls, permission_mode = "ask", confirm_tool_calls = True, wait_denies = True
    )

    assert slow.fetched == []
    assert [e["tool_call_id"] for e in events if e["type"] == "tool_end"] == ["c0", "c1"]


def test_local_loop_never_starts_a_read_after_another_tool(monkeypatch, local_sites):
    calls = [
        _local_url(0, "a", local_sites),
        {"id": "c1", "function": {"name": "python", "arguments": json.dumps({"code": "1"})}},
        _local_url(2, "b", local_sites),
    ]
    slow = _SlowPages(monkeypatch)

    _local_turn(monkeypatch, calls)

    assert slow.peak == 1


def test_local_loop_cancel_stops_started_reads(monkeypatch, local_sites):
    calls = [_local_url(index, host, local_sites) for index, host in enumerate("abc")]
    slow = _SlowPages(monkeypatch)
    cancel = threading.Event()
    threading.Timer(0.1, cancel.set).start()

    _local_turn(monkeypatch, calls, cancel_event = cancel)

    deadline = time.monotonic() + 2
    while slow.running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert slow.running == 0


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
    # Untimed: the first search in a process pays one-off imports the serial arm below would skip.
    tools._web_search("paris", timeout = 20, **kwargs)
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
    real_start = tools._ImageSweep.__dict__["start"].__func__
    early: list = []

    def start(cls, *a):
        sweep = real_start(cls, *a)
        early.append(sweep)
        return sweep

    monkeypatch.setattr(tools._ImageSweep, "start", classmethod(start))
    tools._web_search("paris", timeout = 20, include_images = True)
    # Overlapped again (wall time is not asserted: it moves with host load).
    assert len(early) == 1 and early[0] is not None
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
