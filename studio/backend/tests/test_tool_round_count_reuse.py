# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool-round count reuse: fit-count prefetch and native count memo."""

from __future__ import annotations

import contextlib
import copy
import json
import sys
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import llama_cpp
from core.inference.llama_cpp import LlamaCppBackend

_NATIVE = "/v1/chat/completions/input_tokens"
_REAL_CLIENT = llama_cpp.httpx.Client


class _Server:
    """Fake llama-server count endpoints; records (path, body, thread) for every request."""

    def __init__(
        self,
        monkeypatch,
        *,
        native = True,
    ):
        self.requests: list[tuple[str, str, str]] = []
        self.native = native
        self.fail: dict[str, int] = {}
        self.lock = threading.Lock()
        httpx = llama_cpp.httpx
        client_type = _REAL_CLIENT

        def respond(request):
            path = request.url.path
            body = json.loads(request.content) if request.content else None
            with self.lock:
                self.requests.append(
                    (path, json.dumps(body, sort_keys = True), threading.current_thread().name)
                )
                failing = self.fail.get(path, 0)
                if failing:
                    self.fail[path] = failing - 1
            if failing:
                return httpx.Response(503)
            if path == "/apply-template":
                return httpx.Response(200, json = {"prompt": json.dumps(body, sort_keys = True)})
            if path == "/tokenize":
                return httpx.Response(200, json = {"tokens": list(range(len(body["content"]) // 4))})
            if path == _NATIVE and self.native:
                return httpx.Response(
                    200, json = {"input_tokens": len(json.dumps(body, sort_keys = True)) // 4 + 3}
                )
            return httpx.Response(404)

        monkeypatch.setattr(
            httpx,
            "Client",
            lambda **kw: client_type(transport = httpx.MockTransport(respond), **kw),
        )

    def paths(
        self,
        path = None,
        thread = None,
    ):
        return [
            r
            for r in self.requests
            if (path is None or r[0] == path) and (thread is None or r[2] == thread)
        ]


def _count_backend():
    backend = SimpleNamespace(
        base_url = "http://llama.test",
        is_loaded = True,
        markup_profile = None,
        _auth_headers = None,
        _empty_chat_render_refused = False,
        _process = SimpleNamespace(pid = 4242),
        _model_identifier = "org/model-GGUF",
    )
    backend.count_chat_tokens = MethodType(LlamaCppBackend.count_chat_tokens, backend)
    return backend


_MESSAGES = [{"role": "user", "content": "hello " * 200}]
_TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]


# ── native count memo ──────────────────────────────────────────


def test_an_identical_native_count_is_served_from_memory(monkeypatch):
    server = _Server(monkeypatch)
    backend = _count_backend()
    first = backend.count_chat_tokens(_MESSAGES, None, _TOOLS, strict = True, prefer_native = True)
    second = backend.count_chat_tokens(
        [dict(m) for m in _MESSAGES], None, list(_TOOLS), strict = True, prefer_native = True
    )
    assert first == second > 0
    assert [r[0] for r in server.requests] == [_NATIVE]


def test_a_native_hit_still_honours_should_abort(monkeypatch):
    _Server(monkeypatch)
    backend = _count_backend()
    backend.count_chat_tokens(_MESSAGES, None, _TOOLS, strict = True, prefer_native = True)
    with pytest.raises(llama_cpp.CountAborted):
        backend.count_chat_tokens(
            _MESSAGES, None, _TOOLS, strict = True, prefer_native = True, should_abort = lambda: True
        )


@pytest.mark.parametrize("change", ["restart", "respawn_same_pid", "model", "messages", "tools"])
def test_a_native_count_misses_when_anything_changes(monkeypatch, change):
    server = _Server(monkeypatch)
    backend = _count_backend()
    backend.count_chat_tokens(_MESSAGES, None, _TOOLS, strict = True, prefer_native = True)
    messages, tools = _MESSAGES, _TOOLS
    if change == "restart":
        backend._process = SimpleNamespace(pid = 4343)
    elif change == "respawn_same_pid":
        backend._process = SimpleNamespace(pid = 4242)
    elif change == "model":
        backend._model_identifier = "org/other-GGUF"
    elif change == "messages":
        messages = _MESSAGES + [{"role": "assistant", "content": "x"}]
    elif change == "tools":
        tools = None
    backend.count_chat_tokens(messages, None, tools, strict = True, prefer_native = True)
    assert len(server.paths(_NATIVE)) == 2


def test_a_failed_native_count_is_asked_again(monkeypatch):
    server = _Server(monkeypatch)
    backend = _count_backend()
    server.fail[_NATIVE] = 1
    # Failover: the render path answers this one, and that answer is not kept as a native count.
    assert backend.count_chat_tokens(_MESSAGES, None, _TOOLS, strict = True, prefer_native = True)
    assert [r[0] for r in server.requests] == [_NATIVE, "/apply-template", "/tokenize"]
    backend.count_chat_tokens(_MESSAGES, None, _TOOLS, strict = True, prefer_native = True)
    backend.count_chat_tokens(_MESSAGES, None, _TOOLS, strict = True, prefer_native = True)
    assert len(server.paths(_NATIVE)) == 2
    assert len(server.paths("/apply-template")) == 1


def test_a_server_without_the_native_endpoint_keeps_failing_over(monkeypatch):
    server = _Server(monkeypatch, native = False)
    backend = _count_backend()
    for _ in range(2):
        backend.count_chat_tokens(_MESSAGES, None, _TOOLS, strict = True, prefer_native = True)
    assert len(server.paths(_NATIVE)) == 2
    assert len(server.paths("/apply-template")) == 2


def test_native_counts_never_evict_rendered_counts(monkeypatch):
    _Server(monkeypatch)
    backend = _count_backend()
    backend.count_chat_tokens(_MESSAGES, None, _TOOLS, strict = True)
    for i in range(3 * llama_cpp._COUNT_MEMO_SIZE):
        backend.count_chat_tokens(
            [{"role": "user", "content": f"turn {i}"}], strict = True, prefer_native = True
        )
    assert len(llama_cpp._count_memo(backend)[1]) == 1
    assert len(llama_cpp._count_memo(backend, llama_cpp._NATIVE_COUNT_MEMO)[1]) == 2


# ── fit-count prefetch in the tool loop ───────────────────────


def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _tool_round(i: int) -> list[str]:
    call = {
        "index": 0,
        "id": f"call_{i}",
        "type": "function",
        "function": {"name": "web_search", "arguments": json.dumps({"query": f"q{i}"})},
    }
    return [_sse({"tool_calls": [call]}), "data: [DONE]\n"]


def _loop_backend(monkeypatch, rounds: int):
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = SimpleNamespace(pid = 4242)
    backend._healthy = True
    backend._port = 48847
    backend._api_key = None
    backend._effective_context_length = 8192
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False
    backend._model_identifier = "org/model-GGUF"
    backend._empty_chat_render_refused = False
    streams = [_tool_round(i) for i in range(rounds)] + [
        [_sse({"content": "Done."}), "data: [DONE]\n"]
    ]
    payloads: list[dict] = []

    @contextlib.contextmanager
    def fake_stream_with_retry(
        _client,
        _url,
        payload,
        _cancel,
        headers = None,
        first_token_deadline = None,
    ):
        payloads.append(copy.deepcopy(payload))
        yield type(
            "FakeResponse",
            (),
            {"status_code": 200, "chunks": streams.pop(0), "close": lambda self: None},
        )()

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(
        backend,
        "_iter_text_cancellable",
        lambda response, _c, first_token_deadline = None: iter(response.chunks),
    )
    monkeypatch.setattr(backend, "_maybe_recover_from_mtp_crash", lambda *_a, **_k: False)
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kw: f"result for {arguments.get('query')} " * 200,
    )
    return backend, payloads


def _run_loop(
    monkeypatch,
    *,
    rounds = 2,
    recost = True,
    context_overflow = "truncate_oldest",
    server = None,
):
    backend, payloads = _loop_backend(monkeypatch, rounds)
    recounts = []

    def on_grew(conversation):
        recounts.append(
            backend.count_chat_tokens(
                copy.deepcopy(conversation), None, None, strict = True, prefer_native = True
            )
        )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Search twice."}],
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    },
                }
            ],
            max_tool_iterations = rounds + 1,
            permission_mode = "off",
            bypass_permissions = True,
            context_overflow = context_overflow,
            on_conversation_grew = on_grew if recost else None,
        )
    )
    return events, payloads, recounts


def _renders(server):
    return [body for path, body, _ in server.paths("/apply-template")]


def test_the_prefetch_fills_the_memo_and_the_fit_makes_no_extra_count(monkeypatch):
    server = _Server(monkeypatch)
    _run_loop(monkeypatch, rounds = 2)
    prefetched = server.paths("/apply-template", thread = "fit-count-prefetch")
    assert len(prefetched) == 3
    renders = _renders(server)
    assert len(renders) == len(set(renders)), "a prompt the prefetch counted was rendered again"
    # Each prefetched body is the fit's: nothing else in the round rendered it.
    assert {body for _, body, _ in prefetched} <= set(renders)


def test_the_prefetch_leaves_the_turn_unchanged(monkeypatch):
    server = _Server(monkeypatch)
    events, payloads, recounts = _run_loop(monkeypatch, rounds = 2)
    bodies = sorted(_renders(server))
    # Same turn with the prefetch never started: the fit counts on the loop thread as before.
    real_thread = threading.Thread

    class _NoPrefetch(real_thread):
        def start(self):
            if self.name != "fit-count-prefetch":
                super().start()

        def join(self, timeout = None):
            if self.name != "fit-count-prefetch":
                super().join(timeout)

    monkeypatch.setattr(threading, "Thread", _NoPrefetch)
    server_off = _Server(monkeypatch)
    events_off, payloads_off, recounts_off = _run_loop(monkeypatch, rounds = 2)
    assert not server_off.paths(thread = "fit-count-prefetch")
    assert json.dumps(payloads, sort_keys = True) == json.dumps(payloads_off, sort_keys = True)
    assert json.dumps(events, sort_keys = True, default = repr) == json.dumps(
        events_off, sort_keys = True, default = repr
    )
    assert recounts == recounts_off
    assert bodies == sorted(_renders(server_off))


def test_a_failed_prefetch_falls_back_to_the_fits_own_count(monkeypatch):
    server = _Server(monkeypatch)
    _, payloads_ok, _ = _run_loop(monkeypatch, rounds = 1)
    server = _Server(monkeypatch)
    gate = threading.Event()
    real_fit_count = llama_cpp.LlamaCppBackend.count_chat_tokens

    def count(self, *a, **k):
        if threading.current_thread().name == "fit-count-prefetch":
            gate.set()
            raise RuntimeError("llama-server tokenizer failed")
        return real_fit_count(self, *a, **k)

    monkeypatch.setattr(llama_cpp.LlamaCppBackend, "count_chat_tokens", count)
    _, payloads, _ = _run_loop(monkeypatch, rounds = 1)
    assert gate.is_set()
    assert not server.paths(thread = "fit-count-prefetch")
    # The fit rendered its prompt itself, on the loop thread, and the turn sent the same requests.
    assert server.paths("/apply-template", thread = threading.current_thread().name)
    assert json.dumps(payloads, sort_keys = True) == json.dumps(payloads_ok, sort_keys = True)


@pytest.mark.parametrize("case", ["no_recost", "no_fit"])
def test_no_prefetch_without_a_recost_to_overlap_or_a_fit_to_feed(monkeypatch, case):
    server = _Server(monkeypatch)
    _run_loop(
        monkeypatch,
        rounds = 1,
        recost = case != "no_recost",
        context_overflow = None if case == "no_fit" else "truncate_oldest",
    )
    assert not server.paths(thread = "fit-count-prefetch")
