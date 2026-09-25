# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Test NPU chat through the route and proxy against a local HTTP server.

Replay FastFlowLM 1.0.3 stream shapes and record outgoing request bodies.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
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
    if "CUT" in prompt:
        return _chunk({"content": "par"}) + "data: [DONE]\n\n"
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
        if "SLOW" in prompt:
            time.sleep(20)
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
    httpd.daemon_threads = True
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
    previous = ep_mod._http_client, ep_mod._loopback_http_client
    client = httpx.AsyncClient()
    loopback = httpx.AsyncClient(trust_env = False)

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
            await loopback.aclose()

    ep_mod._http_client, ep_mod._loopback_http_client = client, loopback
    try:
        loop.run_until_complete(go())
    finally:
        ep_mod._http_client, ep_mod._loopback_http_client = previous
        loop.close()
    return out["status"], out["body"]


_TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}


def _data(lines):
    return [
        json.loads(line[5:]) for line in lines if line.startswith("data:") and "[DONE]" not in line
    ]


def test_an_environment_proxy_never_sees_npu_chat(flm, monkeypatch):
    seen = []
    proxy = socket.socket()
    proxy.bind(("127.0.0.1", 0))
    proxy.listen(4)
    proxy.settimeout(0.2)

    def serve() -> None:
        while not stop.is_set():
            try:
                conn, _ = proxy.accept()
            except OSError:
                continue
            seen.append(conn.recv(4096))
            conn.close()

    stop = threading.Event()
    thread = threading.Thread(target = serve, daemon = True)
    thread.start()
    # Studio started under a proxy: its shared client picks the proxy up at construction.
    monkeypatch.delenv("NO_PROXY", raising = False)
    monkeypatch.delenv("no_proxy", raising = False)
    monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.getsockname()[1]}")
    try:
        recorded = flm()
        status, body = _call(stream = False)
    finally:
        stop.set()
        thread.join(timeout = 5)
        proxy.close()
    assert seen == []
    assert status == 200 and recorded[-1]["auth"] == "Bearer secret-key"


@pytest.mark.parametrize("stream", [True, False])
def test_a_top_level_image_reaches_a_vision_model(flm, stream):
    import base64
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buffer, format = "PNG")
    recorded = flm(vision = True)
    status, _ = _call(stream = stream, image_base64 = base64.b64encode(buffer.getvalue()).decode())
    assert status == 200
    last = recorded[-1]["body"]["messages"][-1]
    assert isinstance(last["content"], list)
    assert any(part.get("type") == "image_url" for part in last["content"])


def test_the_body_fastflowlm_receives(flm):
    recorded = flm(reasoning = True)
    _call(stream = False, temperature = 0.3, min_p = 0.05)
    sent = recorded[-1]
    body = sent["body"]
    assert sent["auth"] == "Bearer secret-key"
    assert body["model"] == "qwen3-0.6b-FLM"
    assert body["stream"] is True
    assert body["think"] is True
    assert "chat_template_kwargs" not in body
    assert "min_p" not in body
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
    _, body = _call(stream = False, stop = ["seventy"])
    assert body["choices"][0]["message"]["content"] == "one two seven eight"


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.parametrize(
    "stop, expected",
    [
        (["END"], "hello "),
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
        ({"frequency_penalty": 0.5}, "frequency_penalty"),
        ({"tools": [_TOOL], "tool_choice": "required"}, "tool_choice"),
        ({"tools": [_TOOL], "parallel_tool_calls": False}, "parallel_tool_calls"),
        (
            {"tools": [_TOOL], "tool_choice": {"type": "function", "function": {"name": "f"}}},
            "tool_choice",
        ),
        ({"logit_bias": {"42": -100}}, "logit_bias"),
        ({"logprobs": True}, "logprobs"),
        ({"top_logprobs": 3}, "logprobs"),
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

        def loadable_model(self, model_id):
            return model

        def resident(self):
            if not self.is_loaded:
                return None
            return nb.NpuResident(
                model = model,
                context_length = 8192,
                requested_context_length = None,
                base_url = "http://127.0.0.1:1",
                api_key = "key",
            )

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


def test_npu_status_hides_the_owner_runtime_from_managed_accounts(tmp_path, monkeypatch):
    from routes import npu as npu_routes
    from utils.account_context import AccountContext, arun_as

    backend = nb.LemonadeNpuBackend(root = tmp_path)
    monkeypatch.setattr(
        nb, "detect_amd_npu", lambda: {"present": True, "supported": True, "family": "XDNA2"}
    )
    real = backend.status()
    real["loaded_model"] = "lemonade:qwen3-0.6b-FLM"
    monkeypatch.setattr(backend, "status", lambda: real)
    monkeypatch.setattr(npu_routes, "get_npu_backend", lambda: backend)

    assert asyncio.run(npu_routes.npu_status("owner")) == real
    hidden = asyncio.run(arun_as(AccountContext("b" * 32, "bob"), npu_routes.npu_status("bob")))
    assert hidden["supported"] is False and hidden["loaded_model"] is None
    assert hidden.keys() == real.keys()


def test_an_unload_naming_another_npu_model_keeps_the_resident(monkeypatch):
    from models.inference import UnloadRequest
    from routes import inference as routes

    class _Npu:
        is_loaded = True
        loaded_model = nb.NpuModel(
            id = "gemma3-4b-FLM",
            checkpoint = "gemma3:4b",
            size_gb = 4.5,
            downloaded = True,
            labels = ("vision", "chat"),
            max_context_length = 131072,
        )
        unloads = 0

        def cancel_load(self, model_id):
            return False

        def unload(self):
            self.unloads += 1

    npu = _Npu()
    monkeypatch.setattr(nb, "peek_npu_backend", lambda: npu)

    def _unload(model_path):
        return asyncio.run(routes._unload_model_impl(UnloadRequest(model_path = model_path), "owner"))

    _unload("lemonade:qwen3-0.6b-FLM")
    assert npu.unloads == 0
    _unload("lemonade:gemma3-4b-FLM")
    assert npu.unloads == 1


def test_an_npu_load_it_would_refuse_keeps_the_gpu_resident(monkeypatch):
    from models.inference import LoadRequest
    from routes import inference as routes

    class _Npu:
        loaded_model = None
        loaded_context_length = None

        def resident(self):
            return None

        def loadable_model(self, model_id):
            raise nb.NpuError(f"{model_id} is not downloaded yet.")

        def load(self, model_id, ctx):
            raise AssertionError("load() must not run")

    teardowns: list[str] = []

    async def _unload_gpu(_backend):
        teardowns.append("llama")

    monkeypatch.setattr(nb, "get_npu_backend", lambda: _Npu())
    monkeypatch.setattr(routes, "_unload_llama_before_standard_load", _unload_gpu)
    with pytest.raises(HTTPException) as info:
        asyncio.run(
            routes._load_npu_model(
                LoadRequest(model_path = "lemonade:gemma3-4b-FLM"),
                current_request_counted = False,
                on_reload_confirmed = lambda *, cancel: teardowns.append("chats"),
                load_cancel_event = None,
            )
        )
    assert info.value.status_code == 400 and "not downloaded" in info.value.detail
    assert teardowns == []


def test_a_reply_cut_short_is_not_reported_complete(flm):
    flm()
    status, body = _call(stream = False, messages = [{"role": "user", "content": "CUT"}])
    assert status == 502
    assert "before finishing" in body["error"]["message"]


def test_a_streamed_reply_cut_short_ends_with_an_error(flm):
    flm()
    status, lines = _call(stream = True, messages = [{"role": "user", "content": "CUT"}])
    assert status == 200
    assert lines[-1] == "data: [DONE]"
    assert "before finishing" in lines[-2]


@pytest.mark.parametrize(
    "resident_ctx, requested, expected",
    [
        (16384, 0, "loaded"),
        (None, 16384, "loaded"),
        (16384, 16384, "already_loaded"),
        (None, 0, "already_loaded"),
    ],
)
def test_a_same_model_load_reloads_when_the_context_request_changes(
    monkeypatch, resident_ctx, requested, expected
):
    from models.inference import LoadRequest
    from routes import inference as routes

    model = nb.NpuModel(
        id = "qwen3-0.6b-FLM",
        checkpoint = "qwen3:0.6b",
        size_gb = 0.66,
        downloaded = True,
        labels = ("chat",),
        max_context_length = 40960,
    )

    class _Npu:
        requested = resident_ctx

        def resident(self):
            return nb.NpuResident(
                model = model,
                context_length = self.requested or 8192,
                requested_context_length = self.requested,
                base_url = "http://127.0.0.1:1",
                api_key = "key",
            )

        def loadable_model(self, model_id):
            return model

        def load(self, model_id, ctx):
            self.requested = ctx

    async def _nothing_loaded(_backend):
        return None

    monkeypatch.setattr(nb, "get_npu_backend", lambda: _Npu())
    monkeypatch.setattr(routes, "_peek_inference_backend", lambda: None)
    monkeypatch.setattr(routes, "release_chat_gpu_claim", lambda: True)
    monkeypatch.setattr(routes, "_unload_llama_before_standard_load", _nothing_loaded)
    response = asyncio.run(
        routes._load_npu_model(
            LoadRequest(model_path = model.model_path, max_seq_length = requested),
            current_request_counted = False,
            on_reload_confirmed = None,
            load_cancel_event = None,
        )
    )
    assert response.status == expected


@pytest.mark.parametrize("stream", [True, False])
def test_a_stop_sequence_leaves_the_monitor_row_completed(flm, stream):
    from core.inference.api_monitor import api_monitor

    if not api_monitor.enabled:
        pytest.skip("API monitor disabled")
    flm()
    before = api_monitor.active_count()
    _call(stream = stream, stop = ["seven"])
    assert api_monitor.active_count() == before
    assert api_monitor._entries[0].status == "completed"


def test_a_swap_stops_an_npu_reply_still_in_prefill(flm):
    import routes.inference as ri

    flm()
    payload = ChatCompletionRequest(messages = [{"role": "user", "content": "SLOW"}], stream = True)

    async def go():
        previous = ep_mod._http_client
        client = httpx.AsyncClient()
        ep_mod._http_client = client
        try:
            response = await ri._npu_chat_completions(payload, _request(), "tester")

            async def swap_soon() -> None:
                await asyncio.sleep(1.0)
                ri._raise_or_cancel_active_generations(force = True, action = "Loading a model")

            swap = asyncio.create_task(swap_soon())
            lines = []
            async for piece in response.body_iterator:
                text = piece.decode() if isinstance(piece, bytes) else piece
                lines.extend(line for line in text.split("\n") if line.strip())
            await swap
            return lines
        finally:
            ep_mod._http_client = previous
            await client.aclose()

    lines = asyncio.run(asyncio.wait_for(go(), 15))
    assert lines[-1] == "data: [DONE]"
    assert not any('"error"' in line for line in lines)
    assert _data(lines)[-1]["choices"][0]["finish_reason"] == "stop"


def test_a_dropped_progress_stream_does_not_stop_the_download(monkeypatch):
    from routes import npu as npu_routes

    finished = threading.Event()
    release = threading.Event()

    class _Npu:
        def download(self, model_id):
            yield {"event": "progress", "percent": 40}
            assert release.wait(10)
            yield {"event": "complete", "model": model_id, "percent": 100}
            finished.set()

    monkeypatch.setattr(npu_routes, "get_npu_backend", lambda: _Npu())

    async def go():
        response = await npu_routes.download_npu_model("qwen3-0.6b-FLM")
        body = response.body_iterator
        first = await body.__anext__()
        await body.aclose()
        return first

    first = asyncio.run(go())
    assert '"percent": 40' in (first.decode() if isinstance(first, bytes) else first)
    release.set()
    assert finished.wait(10)


def test_a_second_download_request_follows_the_running_pull(monkeypatch):
    from routes import npu as npu_routes

    started: list[str] = []
    release = threading.Event()

    class _Npu:
        def download(self, model_id):
            started.append(model_id)
            assert release.wait(10)
            yield {"event": "complete", "model": model_id, "percent": 100}

    monkeypatch.setattr(npu_routes, "get_npu_backend", lambda: _Npu())
    first = npu_routes._start_download(_Npu(), "gemma3-4b-FLM")
    second = npu_routes._start_download(_Npu(), "gemma3-4b-FLM")
    assert first is second
    release.set()
    assert [event["event"] for event in second.follow()] == ["complete"]
    assert started == ["gemma3-4b-FLM"]


@pytest.mark.parametrize("stream", [True, False])
def test_a_reply_cut_short_is_recorded_failed(flm, stream):
    from core.inference.api_monitor import api_monitor

    if not api_monitor.enabled:
        pytest.skip("API monitor disabled")
    flm()
    _call(stream = stream, messages = [{"role": "user", "content": "CUT"}])
    assert api_monitor._entries[0].status == "error"
    assert "before finishing" in api_monitor._entries[0].error


def test_operator_sampling_pins_reach_fastflowlm(flm, monkeypatch):
    monkeypatch.setenv("UNSLOTH_SAMPLING_TOP_K", "7")
    recorded = flm()
    _call(stream = False)
    assert recorded[-1]["body"]["top_k"] == 7


def _npu_resident(monkeypatch):
    model = nb.NpuModel(
        id = "qwen3-0.6b-FLM",
        checkpoint = "qwen3:0.6b",
        size_gb = 0.66,
        downloaded = True,
        labels = ("chat",),
        max_context_length = 40960,
    )

    class _Npu:
        loaded_model = model

    monkeypatch.setattr(nb, "peek_npu_backend", lambda: _Npu())
    return model


def test_the_npu_resident_answers_to_its_own_names_only(monkeypatch):
    from routes import inference as routes

    model = _npu_resident(monkeypatch)
    for name in (model.model_path, model.id):
        assert routes._loaded_satisfies(name)
        assert routes._loaded_identity_satisfies(name)
    for name in ("lemonade:gemma3-4b-FLM", "unsloth/Qwen3-0.6B-GGUF"):
        assert not routes._loaded_satisfies(name)
        assert not routes._loaded_identity_satisfies(name)


def test_a_request_naming_another_npu_model_is_refused(monkeypatch):
    from routes import inference as routes

    model = _npu_resident(monkeypatch)
    asyncio.run(routes._reject_unservable_model(model.model_path, None))
    with pytest.raises(HTTPException) as caught:
        asyncio.run(routes._reject_unservable_model("lemonade:gemma3-4b-FLM", None))
    assert caught.value.status_code == 404


def test_a_collected_reply_stops_when_the_caller_leaves(flm):
    import routes.inference as ri
    from starlette.requests import Request

    flm()
    left_at = time.monotonic() + 1.0

    async def receive() -> dict:
        # Starlette polls with a zero timeout, so answer by the clock, not after a sleep.
        if time.monotonic() >= left_at:
            return {"type": "http.disconnect"}
        await asyncio.sleep(3600)
        return {"type": "http.disconnect"}

    request = Request(dict(_request().scope), receive)
    payload = ChatCompletionRequest(messages = [{"role": "user", "content": "SLOW"}], stream = False)

    async def go():
        previous = ep_mod._http_client
        client = httpx.AsyncClient()
        ep_mod._http_client = client
        try:
            return await ri._npu_chat_completions(payload, request, "tester")
        finally:
            ep_mod._http_client = previous
            await client.aclose()

    response = asyncio.run(asyncio.wait_for(go(), 15))
    assert response.status_code == 499
