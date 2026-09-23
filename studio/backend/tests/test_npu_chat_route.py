# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Test NPU chat through the route and proxy against a local HTTP server.

Replay FastFlowLM 1.0.3 stream shapes and record outgoing request bodies.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from fastapi import HTTPException

from core.inference import external_provider as ep_mod
from core.inference import npu_backend as nb
from core.inference.external_provider import _apply_fastflowlm_reasoning_controls
from models.inference import ChatCompletionRequest


def _chunk(
    delta: dict,
    finish = None,
    usage = None,
) -> str:
    body = {
        "id": "chatcmpl-flm",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "qwen3:0.6b",
        "choices": [{"index": 0, "delta": {"role": "assistant", **delta}, "finish_reason": finish}],
    }
    if usage:
        body["usage"] = usage
    return "data: " + json.dumps(body) + "\n\n"


_USAGE = {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}


def _script(prompt: str) -> str:
    if "FAULT" in prompt:
        return (
            _chunk({"content": "par"})
            + '{"error":"qds_device::wait() unexpected command state"}\n'
            + "data: [DONE]\n\n"
        )
    if "TOOL" in prompt:
        call = {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
        }
        return (
            _chunk({"tool_calls": [call]}) + _chunk({}, "tool_calls", _USAGE) + "data: [DONE]\n\n"
        )
    if "FINAL" in prompt:
        # Content on the finishing chunk itself.
        return (
            _chunk({"content": "hello E"})
            + _chunk({"content": "ND hidden"}, "stop", _USAGE)
            + "data: [DONE]\n\n"
        )
    if "THINK" in prompt:
        return (
            _chunk({"reasoning_content": "Let me add."})
            + _chunk({"content": "4"})
            + _chunk({}, "stop", _USAGE)
            + "data: [DONE]\n\n"
        )
    words = ["one", " two", " sev", "en", " eight"]
    return (
        ": ping\n\n"
        + "".join(_chunk({"content": w}) for w in words)
        + _chunk({}, "length", _USAGE)
        + "data: [DONE]\n\n"
    )


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        self.server.recorded.append(
            {"body": body, "auth": self.headers.get("Authorization"), "path": self.path}
        )
        prompt = json.dumps(body.get("messages"))
        data = _script(prompt).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _FakeNpu:
    def __init__(self, upstream) -> None:
        self._upstream = upstream
        self.is_loaded = True

    def upstream(self):
        return self._upstream


@pytest.fixture
def flm(monkeypatch):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.recorded = []
    thread = threading.Thread(target = httpd.serve_forever, daemon = True)
    thread.start()

    def install(**caps):
        upstream = nb.ManagedUpstream(
            provider_type = "lemonade",
            base_url = f"http://127.0.0.1:{httpd.server_address[1]}/v1",
            api_key = "secret-key",
            model = "qwen3-0.6b-FLM",
            public_model = "lemonade:qwen3-0.6b-FLM",
            supports_vision = caps.get("vision", False),
            supports_tools = caps.get("tools", False),
            supports_reasoning = caps.get("reasoning", False),
            context_length = 8192,
        )
        monkeypatch.setattr(nb, "get_npu_backend", lambda: _FakeNpu(upstream))
        return httpd.recorded

    yield install
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout = 10)


def _request():
    from starlette.requests import Request
    async def receive() -> dict:
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "root_path": "",
            "scheme": "http",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8000),
        },
        receive,
    )


def _call(**fields):
    """Run the route; return (status, parsed body or SSE lines)."""
    import routes.inference as ri

    payload = ChatCompletionRequest(
        messages = fields.pop("messages", [{"role": "user", "content": "count"}]), **fields
    )
    out: dict = {}
    loop = asyncio.new_event_loop()
    previous = ep_mod._http_client
    client = httpx.AsyncClient()

    async def go() -> None:
        try:
            response = await ri._npu_chat_completions(payload, _request(), "tester")
            if hasattr(response, "body_iterator"):
                lines = []
                async for piece in response.body_iterator:
                    text = piece.decode() if isinstance(piece, bytes) else piece
                    lines.extend(line for line in text.split("\n") if line.strip())
                out["status"], out["body"] = response.status_code, lines
            else:
                out["status"], out["body"] = response.status_code, json.loads(response.body)
        finally:
            await client.aclose()

    ep_mod._http_client = client
    try:
        loop.run_until_complete(go())
    finally:
        ep_mod._http_client = previous
        loop.close()
    return out["status"], out["body"]


def _data(lines):
    return [
        json.loads(line[5:]) for line in lines if line.startswith("data:") and "[DONE]" not in line
    ]


def test_the_body_fastflowlm_receives(flm):
    recorded = flm(reasoning = True)
    _call(stream = False, temperature = 0.3, min_p = 0.05)
    sent = recorded[-1]
    body = sent["body"]
    assert sent["auth"] == "Bearer secret-key"
    assert body["model"] == "qwen3-0.6b-FLM"
    # Always streamed upstream, whatever the caller asked for.
    assert body["stream"] is True
    # Thinking is explicit: FastFlowLM decides for itself otherwise.
    assert body["think"] is True
    assert "chat_template_kwargs" not in body
    # min_p is parsed into an integer by FastFlowLM 1.0.3, so it is never sent.
    assert "min_p" not in body
    # The sampler values FastFlowLM would otherwise inherit from the previous chat.
    assert {"top_p", "top_k", "repetition_penalty", "temperature"} <= set(body)


@pytest.mark.parametrize("stream", [True, False])
def test_custom_responses_selection_cannot_redirect_managed_npu(flm, stream):
    recorded = flm(reasoning = True)
    status, response = _call(
        stream = stream,
        provider_type = "custom",
        provider_api_type = "responses",
        provider_base_url = "http://127.0.0.1:1/v1",
        external_model = "wrong-model",
    )
    assert status == 200
    sent = recorded[-1]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["auth"] == "Bearer secret-key"
    assert sent["body"]["model"] == "qwen3-0.6b-FLM"
    assert sent["body"]["stream"] is True
    assert sent["body"]["think"] is True
    assert "input" not in sent["body"]
    if stream:
        assert response[-1] == "data: [DONE]"
    else:
        assert response["choices"][0]["message"]["content"] == "one two seven eight"


def test_thinking_off_and_non_reasoning_models(flm):
    recorded = flm(reasoning = True)
    _call(stream = False, enable_thinking = False)
    assert recorded[-1]["body"]["think"] is False
    recorded = flm(reasoning = False)
    _call(stream = False)
    assert recorded[-1]["body"]["think"] is False


def test_non_streaming_reply_is_collected_from_the_stream(flm):
    flm()
    status, body = _call(stream = False)
    assert status == 200
    assert body["object"] == "chat.completion"
    assert body["model"] == "lemonade:qwen3-0.6b-FLM"
    choice = body["choices"][0]
    assert choice["message"]["content"] == "one two seven eight"
    # The stream's own reason: FastFlowLM's non-streaming reply says "stop" for a cutoff.
    assert choice["finish_reason"] == "length"
    assert body["usage"] == _USAGE


def test_streaming_rewrites_the_model_name(flm):
    flm()
    status, lines = _call(stream = True)
    chunks = _data(lines)
    assert status == 200
    assert {chunk["model"] for chunk in chunks} == {"lemonade:qwen3-0.6b-FLM"}
    assert lines[-1] == "data: [DONE]"
    assert (
        "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        == "one two seven eight"
    )


@pytest.mark.parametrize("stream", [True, False])
def test_stop_is_enforced_across_chunks(flm, stream):
    flm()
    status, out = _call(stream = stream, stop = ["seven"])
    if stream:
        chunks = _data(out)
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        assert text == "one two "
        assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
        assert out[-1] == "data: [DONE]"
    else:
        assert out["choices"][0]["message"]["content"] == "one two "
        assert out["choices"][0]["finish_reason"] == "stop"


def test_a_partial_stop_match_is_released(flm):
    flm()
    # "sev" starts "seventy" but the stop never completes, so nothing may be lost.
    _, body = _call(stream = False, stop = ["seventy"])
    assert body["choices"][0]["message"]["content"] == "one two seven eight"


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    "stop, expected",
    [
        (["END"], "hello "),
        # A partial match held back at the end is released in order, not after the last chunk.
        (["END hiddenness"], "hello END hidden"),
    ],
)
def test_stop_on_the_finishing_chunk(flm, stream, stop, expected):
    flm()
    messages = [{"role": "user", "content": "FINAL"}]
    _, out = _call(stream = stream, stop = stop, messages = messages)
    if stream:
        chunks = _data(out)
        text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks)
        finish = [
            c["choices"][0]["finish_reason"] for c in chunks if c["choices"][0]["finish_reason"]
        ]
        assert out[-1] == "data: [DONE]"
    else:
        text = out["choices"][0]["message"]["content"]
        finish = [out["choices"][0]["finish_reason"]]
    assert text == expected
    assert finish == ["stop"]


def test_reasoning_is_kept_apart(flm):
    flm(reasoning = True)
    _, body = _call(stream = False, messages = [{"role": "user", "content": "THINK"}])
    message = body["choices"][0]["message"]
    assert message["reasoning_content"] == "Let me add."
    assert message["content"] == "4"


def test_tool_calls_are_assembled(flm):
    flm(tools = True)
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
    ]
    _, body = _call(stream = False, tools = tools, messages = [{"role": "user", "content": "TOOL"}])
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
        }
    ]


def test_tool_choice_none_withholds_the_tools(flm):
    recorded = flm(tools = True)
    tools = [
        {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}
    ]
    _call(stream = False, tools = tools, tool_choice = "none")
    assert "tools" not in recorded[-1]["body"]


def test_a_bare_error_line_becomes_an_error(flm):
    flm()
    status, body = _call(stream = False, messages = [{"role": "user", "content": "FAULT"}])
    assert status == 502
    assert "qds_device" in body["error"]["message"]
    status, lines = _call(stream = True, messages = [{"role": "user", "content": "FAULT"}])
    errors = [chunk for chunk in _data(lines) if "error" in chunk]
    assert errors and "qds_device" in errors[0]["error"]["message"]


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"response_format": {"type": "json_object"}}, "response_format"),
        ({"seed": 7}, "seed"),
        ({"n": 2}, "n"),
        (
            {
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "f", "parameters": {"type": "object"}},
                    }
                ]
            },
            "tool calling",
        ),
        (
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "what is this"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="},
                            },
                        ],
                    }
                ]
            },
            "does not read images",
        ),
    ],
)
def test_what_fastflowlm_would_drop_is_refused(flm, fields, message):
    recorded = flm()
    with pytest.raises(HTTPException) as caught:
        _call(stream = False, **fields)
    assert caught.value.status_code == 400
    assert message in json.dumps(caught.value.detail)
    assert recorded == []


def test_a_saved_connection_cannot_name_the_managed_type():
    from core.inference.providers import get_connectable_provider_info, list_available_providers

    assert get_connectable_provider_info("lemonade") is None
    assert get_connectable_provider_info("vllm") is not None
    assert "lemonade" not in {
        p["provider_type"] for p in list_available_providers(include_hidden = True)
    }


def test_a_request_cannot_route_to_the_managed_type():
    import routes.inference as ri

    payload = ChatCompletionRequest(
        provider_type = "lemonade",
        provider_base_url = "http://127.0.0.1:1/v1",
        messages = [{"role": "user", "content": "hi"}],
        stream = True,
    )
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(HTTPException) as caught:
            loop.run_until_complete(ri._proxy_to_external_provider(payload, _request(), "t"))
    finally:
        loop.close()
    assert caught.value.status_code == 400
    assert "Unknown provider type" in str(caught.value.detail)


@pytest.mark.parametrize(
    "enable, effort, expected",
    [
        (None, None, {}),
        (True, None, {"think": True}),
        (False, None, {"think": False}),
        (None, "none", {"think": False}),
        (True, "high", {"think": True, "reasoning_effort": "high"}),
        (False, "high", {"think": False}),
    ],
)
def test_fastflowlm_reasoning_controls(enable, effort, expected):
    body: dict = {}
    _apply_fastflowlm_reasoning_controls(body, enable, effort)
    assert body == expected


def test_stop_loading_an_npu_model_does_not_wait_for_the_load(monkeypatch):
    """/load holds the lifecycle gate for the whole /v1/load, so Stop loading must not queue on it."""
    from core.inference.llama_keepwarm import inference_lifecycle_gate
    from models.inference import UnloadRequest
    from routes import inference as routes

    cancelled: list[str] = []

    class _Loading:
        is_loaded = False

        def cancel_load(self, model_id):
            cancelled.append(model_id)
            return True

    monkeypatch.setattr(nb, "peek_npu_backend", lambda: _Loading())

    async def _run():
        async with inference_lifecycle_gate():
            return await asyncio.wait_for(
                routes._unload_model_impl(
                    UnloadRequest(model_path = "lemonade:qwen3-0.6b-FLM"), "owner"
                ),
                timeout = 10,
            )

    response = asyncio.run(_run())
    assert response.status == "unloaded"
    assert cancelled == ["qwen3-0.6b-FLM"]


def test_an_owner_npu_model_is_hidden_from_managed_accounts(monkeypatch):
    """An NPU load publishes the owner as the loader, as a GPU load does."""
    from auth import policy
    from hub.services.models import account_access
    from models.inference import LoadRequest, UnloadRequest
    from routes import inference as routes
    from utils.account_context import AccountContext, run_as

    bob = AccountContext("b" * 32, "bob")
    model = nb.NpuModel(
        id = "qwen3-0.6b-FLM",
        checkpoint = "qwen3:0.6b",
        size_gb = 0.66,
        downloaded = True,
        labels = ("reasoning", "chat"),
        max_context_length = 40960,
    )

    class _Npu:
        is_loaded = False
        loaded_model = None
        loaded_context_length = None
        unloads = 0

        def load(self, model_id, ctx):
            self.is_loaded, self.loaded_model, self.loaded_context_length = True, model, 8192

        def unload(self):
            self.unloads += 1

        def cancel_load(self, model_id):
            return False

    npu = _Npu()
    monkeypatch.setattr(nb, "get_npu_backend", lambda: npu)
    monkeypatch.setattr(nb, "peek_npu_backend", lambda: npu)
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)
    monkeypatch.setattr(account_access, "_resident_accounts", {})
    monkeypatch.setattr(account_access, "_resident_sharers", {})
    monkeypatch.setattr(routes, "_peek_inference_backend", lambda: None)
    monkeypatch.setattr(routes, "release_chat_gpu_claim", lambda: True)

    async def _nothing_loaded(_backend):
        return None

    monkeypatch.setattr(routes, "_unload_llama_before_standard_load", _nothing_loaded)

    response = asyncio.run(
        routes._load_npu_model(
            LoadRequest(model_path = model.model_path),
            current_request_counted = False,
            on_reload_confirmed = None,
            load_cancel_event = None,
        )
    )
    assert response.status == "loaded"
    assert routes._loaded_slot_ident() == model.model_path
    assert not account_access.resident_hidden("chat", routes._loaded_slot_ident())

    def _as_bob():
        return account_access.resident_hidden("chat", routes._loaded_slot_ident())

    assert run_as(bob, _as_bob) is True
    with pytest.raises(HTTPException) as info:
        run_as(
            bob,
            asyncio.run,
            routes._unload_model_impl(UnloadRequest(model_path = model.model_path), "bob"),
        )
    assert info.value.status_code == 404
    assert npu.unloads == 0
