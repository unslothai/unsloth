# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import sys
from pathlib import Path

from starlette.requests import Request

import routes.inference as ri
from models.inference import ChatCompletionRequest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_external_provider_sampling_over_the_wire import _Handler, _run, _Server  # noqa: E402


_USAGE = {
    "prompt_tokens": 1200,
    "completion_tokens": 7,
    "total_tokens": 1207,
    "prompt_tokens_details": {"cached_tokens": 1024},
}


class _UsageHandler(_Handler):
    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        self.server.recorded.append(json.loads(self.rfile.read(length) or b"{}"))  # type: ignore[attr-defined]
        sse = (
            'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\n'
            'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            f'data: {json.dumps({"choices": [], "usage": _USAGE})}\n\n'
            "data: [DONE]\n\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(sse)))
        self.end_headers()
        self.wfile.write(sse)


def _proxy(provider_type: str, **payload_fields) -> tuple[list[dict], dict]:
    async def receive() -> dict:
        return {"type": "http.disconnect"}

    chunks: list[dict] = []
    with _Server(handler = _UsageHandler) as server:
        payload = ChatCompletionRequest(
            provider_type = provider_type,
            provider_base_url = server.base_url,
            messages = [{"role": "user", "content": "hi"}],
            model = "a-model",
            stream = True,
            **payload_fields,
        )
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": [(b"content-type", b"application/json")],
                "query_string": b"",
            },
            receive,
        )

        async def go() -> None:
            response = await ri._proxy_to_external_provider(payload, request)
            async for part in response.body_iterator:
                text = part.decode() if isinstance(part, bytes) else part
                for line in text.splitlines():
                    body = line[len("data:") :].strip() if line.startswith("data:") else ""
                    if body and body != "[DONE]":
                        chunks.append(json.loads(body))

        _run(go)
        assert server.bodies, "the route never reached the provider"
        return chunks, server.bodies[-1]


def _usage_chunks(chunks: list[dict]) -> list[dict]:
    return [c["usage"] for c in chunks if c.get("choices") == [] and c.get("usage")]


def test_only_an_opted_in_caller_receives_the_usage_chunk():
    opted_in, _ = _proxy("vllm", stream_options = {"include_usage": True})
    plain, _ = _proxy("vllm")
    assert _usage_chunks(opted_in) == [_USAGE]
    assert _usage_chunks(plain) == []
    assert any(c.get("choices") for c in plain)


def test_the_opt_in_does_not_reach_a_custom_endpoint():
    # "custom" is never asked for usage upstream: a strict endpoint 400s on the field.
    _, opted_in = _proxy("custom", stream_options = {"include_usage": True})
    _, plain = _proxy("custom")
    assert opted_in == plain
    assert "stream_options" not in opted_in
