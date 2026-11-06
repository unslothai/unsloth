# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A tool round near the window recounts the prompt it just fitted; one llama-server render serves both."""

import json
from types import MethodType, SimpleNamespace

import pytest


def _setup(monkeypatch, *, status = 200):
    from core.inference import llama_cpp

    httpx = llama_cpp.httpx
    paths = []

    def respond(request):
        paths.append(request.url.path)
        if request.url.path == "/apply-template":
            if status != 200:
                return httpx.Response(status)
            body = json.loads(request.content)
            return httpx.Response(200, json = {"prompt": json.dumps(body)})
        if request.url.path == "/tokenize":
            text = json.loads(request.content)["content"]
            return httpx.Response(200, json = {"tokens": list(range(len(text) // 4))})
        return httpx.Response(404)

    client_type = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kw: client_type(transport = httpx.MockTransport(respond), **kw)
    )
    backend = SimpleNamespace(
        base_url = "http://llama.test",
        is_loaded = True,
        markup_profile = None,
        _auth_headers = None,
        _empty_chat_render_refused = False,
        _process = SimpleNamespace(pid = 4242),
        _model_identifier = "org/model-GGUF",
    )
    backend.count_chat_tokens = MethodType(llama_cpp.LlamaCppBackend.count_chat_tokens, backend)
    return backend, paths


_MESSAGES = [{"role": "user", "content": "hello " * 200}]
_TOOLS = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]


def test_identical_recount_reuses_the_render(monkeypatch):
    backend, paths = _setup(monkeypatch)
    first = backend.count_chat_tokens(_MESSAGES, None, _TOOLS, strict = True)
    second = backend.count_chat_tokens(
        [dict(m) for m in _MESSAGES], None, list(_TOOLS), strict = True
    )
    assert first == second > 0
    assert paths == ["/apply-template", "/tokenize"]


@pytest.mark.parametrize(
    "change",
    ["messages", "tools", "system", "kwargs", "continue", "options", "process", "model"],
)
def test_anything_the_render_sees_misses(monkeypatch, change):
    backend, paths = _setup(monkeypatch)
    backend.count_chat_tokens(_MESSAGES, None, _TOOLS, strict = True)
    messages, system, tools, kwargs = _MESSAGES, None, _TOOLS, {}
    if change == "messages":
        messages = _MESSAGES + [{"role": "assistant", "content": "x"}]
    elif change == "tools":
        tools = None
    elif change == "system":
        system = "Be brief."
    elif change == "kwargs":
        kwargs["chat_template_kwargs"] = {"enable_thinking": False}
    elif change == "continue":
        messages = _MESSAGES + [{"role": "assistant", "content": "Partial"}]
        backend.count_chat_tokens(messages, None, tools, strict = True)
        kwargs["continue_final_message"] = True
    elif change == "options":
        kwargs["template_options"] = {"tool_choice": "required"}
    elif change == "process":
        backend._process = SimpleNamespace(pid = 4343)
    elif change == "model":
        backend._model_identifier = "org/other-GGUF"
    before = paths.count("/tokenize")
    backend.count_chat_tokens(messages, system, tools, strict = True, **kwargs)
    assert paths.count("/tokenize") == before + 1


def test_a_failed_render_is_not_remembered(monkeypatch):
    backend, paths = _setup(monkeypatch, status = 503)
    with pytest.raises(RuntimeError):
        backend.count_chat_tokens(_MESSAGES, None, _TOOLS, strict = True)
    # The non-strict text fallback is an approximation and must not be served later as a count.
    assert backend.count_chat_tokens(_MESSAGES, None, _TOOLS) > 0
    before = paths.count("/apply-template")
    backend.count_chat_tokens(_MESSAGES, None, _TOOLS)
    assert paths.count("/apply-template") == before + 1


def test_admission_counts_are_not_memoised(monkeypatch):
    backend, paths = _setup(monkeypatch)
    for _ in range(2):
        backend.count_chat_tokens(_MESSAGES, None, _TOOLS, strict = True, prefer_native = True)
    assert paths.count("/v1/chat/completions/input_tokens") == 2


def test_memo_stays_bounded(monkeypatch):
    from core.inference import llama_chat

    backend, _ = _setup(monkeypatch)
    for i in range(3 * llama_chat._COUNT_MEMO_SIZE):
        backend.count_chat_tokens([{"role": "user", "content": f"turn {i}"}], strict = True)
    assert len(llama_chat._count_memo(backend)[1]) == llama_chat._COUNT_MEMO_SIZE
