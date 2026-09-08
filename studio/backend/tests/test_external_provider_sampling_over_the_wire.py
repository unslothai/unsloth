# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The same forwarding claims, but over a socket and through the real route.

``test_external_provider_sampling_forwarding.py`` mocks the transport and AST-parses the
route, which proves the source says the right words but cannot fail on the runtime hazard
those words prevent: pydantic v2 records every ``setattr`` in ``model_fields_set``, so a
write before those reads turns an omission into a request for the schema default.

So this drives ``_proxy_to_external_provider`` itself against a loopback ``http.server``.
Stdlib only, so it passes identically on the Linux, macOS and Windows runners.
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient
from models.inference import ChatCompletionRequest


SAMPLING_EXTENSIONS = ("top_k", "min_p", "repetition_penalty", "repeat_penalty")


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args, **kwargs) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        self.server.recorded.append(json.loads(raw.decode()))  # type: ignore[attr-defined]

        sse = (
            'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            "data: [DONE]\n\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(sse)))
        self.end_headers()
        self.wfile.write(sse)


# What a gateway that validates its input accepts; the sampling extensions are absent.
_OPENAI_DOCUMENTED = frozenset(
    {
        "model",
        "messages",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "max_tokens",
        "max_completion_tokens",
        "stop",
        "seed",
        "response_format",
        "tools",
        "tool_choice",
    }
)


class _Server:
    """A live OpenAI-compatible endpoint that keeps every body it was posted."""

    def __init__(self, handler = None) -> None:
        self._handler = handler or _Handler

    def __enter__(self) -> "_Server":
        # Port 0: the OS assigns, so no free-port scan can lose the race on a busy runner.
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler)
        self._httpd.recorded = []  # type: ignore[attr-defined]
        self._thread = threading.Thread(target = self._httpd.serve_forever, daemon = True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout = 10)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    @property
    def bodies(self) -> list[dict]:
        return self._httpd.recorded  # type: ignore[attr-defined]

    def sampling(self, index: int = -1) -> dict:
        body = self.bodies[index]
        return {k: body[k] for k in SAMPLING_EXTENSIONS if k in body}


def _run(coro) -> None:
    """One loop per call, with a matching client, mirroring the sibling test file.

    ``ep_mod._http_client`` is built at import time and its pool binds to whichever loop
    first uses it, so the real client has to be replaced per loop rather than reused.
    """
    loop = asyncio.new_event_loop()
    previous = ep_mod._http_client
    client = httpx.AsyncClient()

    async def wrapper() -> None:
        try:
            await coro()
        finally:
            await client.aclose()

    ep_mod._http_client = client
    try:
        loop.run_until_complete(wrapper())
    finally:
        ep_mod._http_client = previous
        loop.close()


def _client_capture(provider_type: str, **kwargs) -> dict:
    with _Server() as server:
        client = ExternalProviderClient(
            provider_type = provider_type,
            base_url = server.base_url,
            api_key = "",
        )

        async def go() -> None:
            async for _ in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "a-model",
                **kwargs,
            ):
                pass

        _run(go)
        assert server.bodies, "nothing reached the provider"
        return server.sampling()


def _route_capture(**payload_fields) -> dict:
    """POST through `_proxy_to_external_provider`, not around it."""
    import routes.inference as ri
    from starlette.requests import Request

    async def receive() -> dict:
        return {"type": "http.disconnect"}

    with _Server() as server:
        payload = ChatCompletionRequest(
            provider_type = "vllm",
            provider_base_url = server.base_url,
            messages = [{"role": "user", "content": "hi"}],
            model = "a-model",
            stream = True,
            **payload_fields,
        )
        request = Request(
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

        async def go() -> None:
            response = await ri._proxy_to_external_provider(payload, request)
            async for _ in response.body_iterator:
                pass

        _run(go)
        assert server.bodies, "the route never reached the provider"
        return server.sampling()


def test_a_request_that_never_mentioned_them_forwards_nothing():
    # payload.min_p yields 0.01 here, not None; only model_fields_set separates the two.
    assert _route_capture() == {}


def test_the_route_forwards_explicit_values():
    assert _route_capture(top_k = 40, min_p = 0.07, repetition_penalty = 1.15) == {
        "top_k": 40,
        "min_p": 0.07,
        "repetition_penalty": 1.15,
    }


def test_explicit_values_equal_to_the_schema_defaults_are_still_forwarded():
    # 20 / 0.01 / 1.0 ARE the defaults, so a `!= default` shortcut would drop them.
    assert _route_capture(top_k = 20, min_p = 0.01, repetition_penalty = 1.0) == {
        "top_k": 20,
        "min_p": 0.01,
        "repetition_penalty": 1.0,
    }


def test_zero_survives_the_route():
    assert _route_capture(top_k = 0, min_p = 0.0) == {"top_k": 0, "min_p": 0.0}


@pytest.mark.parametrize(
    "field,value",
    [
        ("top_k", 40),
        ("min_p", 0.07),
        ("repetition_penalty", 1.15),
    ],
)
def test_one_field_set_forwards_only_that_field(field, value):
    assert _route_capture(**{field: value}) == {field: value}


def test_writing_to_the_payload_would_make_an_omission_look_explicit():
    payload = ChatCompletionRequest(messages = [{"role": "user", "content": "hi"}])
    assert "min_p" not in payload.model_fields_set
    payload.min_p = payload.min_p  # a no-op write, same value
    assert "min_p" in payload.model_fields_set


@pytest.mark.parametrize("provider_type", ["vllm", "openrouter"])
def test_all_three_reach_a_live_endpoint(provider_type):
    assert _client_capture(
        provider_type,
        top_k = 40,
        min_p = 0.07,
        repetition_penalty = 1.15,
    ) == {"top_k": 40, "min_p": 0.07, "repetition_penalty": 1.15}


def test_llama_server_receives_repeat_penalty_on_the_wire():
    assert _client_capture(
        "llama_cpp",
        top_k = 40,
        min_p = 0.07,
        repetition_penalty = 1.15,
    ) == {"top_k": 40, "min_p": 0.07, "repeat_penalty": 1.15}


def test_ollama_receives_none_of_them_even_from_a_raw_api_caller():
    assert _client_capture("ollama", top_k = 42, min_p = 0.07, repetition_penalty = 1.23) == {}


def test_the_tool_loop_continuation_keeps_the_same_sampling():
    # OAICompatTransport replays **request_kwargs every turn; a tool call must not change it.
    from core.inference.external_tool_transport import OAICompatTransport
    with _Server() as server:
        client = ExternalProviderClient(
            provider_type = "vllm",
            base_url = server.base_url,
            api_key = "",
        )
        transport = OAICompatTransport(
            client,
            model = "a-model",
            stream = True,
            top_k = 40,
            min_p = 0.07,
            repetition_penalty = 1.15,
        )
        cancel_event = threading.Event()
        turns = [
            [{"role": "user", "content": "hi"}],
            [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            ],
        ]

        async def go() -> None:
            for messages in turns:
                async for _ in transport.stream(
                    messages = messages,
                    tools = None,
                    tool_choice = None,
                    cancel_event = cancel_event,
                ):
                    pass

        _run(go)
        assert len(server.bodies) == 2
        assert server.sampling(0) == {"top_k": 40, "min_p": 0.07, "repetition_penalty": 1.15}
        assert server.sampling(1) == server.sampling(0)


def test_a_stale_frontend_bundle_does_not_start_400ing_a_custom_gateway():
    # The pre-PR bundle spread top_k on every custom request and a tab left open across an
    # upgrade still runs it. Without custom's registry guard this same body went 200 -> 400.
    class _Strict(_Handler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode() or "{}")
            self.server.recorded.append(body)  # type: ignore[attr-defined]
            unknown = sorted(set(body) - _OPENAI_DOCUMENTED)
            if unknown:
                payload = json.dumps(
                    {
                        "error": {
                            "message": f"Unrecognized request argument supplied: {', '.join(unknown)}",
                            "type": "invalid_request_error",
                        }
                    }
                ).encode()
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
            else:
                payload = (
                    'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    with _Server(handler = _Strict) as server:
        client = ExternalProviderClient(
            provider_type = "custom",
            base_url = server.base_url,
            api_key = "",
        )
        lines: list[str] = []

        async def go() -> None:
            async for line in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "a-model",
                top_k = 20,
            ):
                lines.append(line)

        _run(go)
        assert server.sampling() == {}, "custom leaked an extension the gateway rejects"
        assert "Unrecognized request argument" not in "".join(lines)
