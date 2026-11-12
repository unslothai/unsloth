# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inception Labs Mercury diffusion streaming: request flag and full-text replace semantics."""

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _collect(agen):
    out = []
    async for line in agen:
        out.append(line)
    return out


def _mock_http_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(ep_mod, "_http_client", httpx.AsyncClient(transport = transport))


def _make_inception_client() -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "inception",
        base_url = "https://api.inceptionlabs.ai/v1",
        api_key = "sk-inception-test",
    )


class _AsyncBody(httpx.AsyncByteStream):
    """Wrap an async generator of bytes as an AsyncByteStream for httpx transports."""

    def __init__(self, chunks):
        self._chunks = chunks

    async def __aiter__(self):
        async for chunk in self._chunks:
            yield chunk

    async def aclose(self):
        await self._chunks.aclose()


async def _single_bytes(text):
    yield text.encode("utf-8")


def test_diffusing_flag_sent_only_when_enabled(monkeypatch):
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run(diffusing):
        client = _make_inception_client()
        lines = await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "ping"}],
                model = "mercury-3",
                temperature = 0.7,
                top_p = None,
                max_tokens = 64,
                diffusing = diffusing,
            )
        )
        await client.close()
        return lines

    lines = _drive(run(True))
    assert captured["body"].get("diffusing") is True
    assert any("Hi" in line for line in lines)

    _drive(run(False))
    assert "diffusing" not in captured["body"]

    _drive(run(None))
    assert "diffusing" not in captured["body"]


def test_inception_stream_requests_usage(monkeypatch):
    # The tok/s badge is synthesized from the usage chunk, which Inception only sends when asked.
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\ndata: [DONE]\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_inception_client()
        await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "ping"}],
                model = "mercury-3",
                temperature = 0.7,
                top_p = None,
                max_tokens = 64,
            )
        )
        await client.close()

    _drive(run())
    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}


def test_diffusing_replace_semantics(monkeypatch):
    events = [
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "mercury-3",
            "choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}],
        },
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "mercury-3",
            "choices": [{"index": 0, "delta": {"content": "Hello world"}, "finish_reason": None}],
        },
        {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "mercury-3",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]
    raw = "".join(f"data: {json.dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"

    def handler(request):
        return httpx.Response(
            200,
            stream = _AsyncBody(_single_bytes(raw)),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_inception_client()
        lines = await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "mercury-3",
                temperature = 0.7,
                top_p = None,
                max_tokens = 64,
                diffusing = True,
            )
        )
        await client.close()
        return lines

    lines = _drive(run())
    frames = []
    content_lines = []
    done_seen = False
    for line in lines:
        if not line.startswith("data:"):
            continue
        raw_payload = line[len("data:") :].strip()
        if raw_payload == "[DONE]":
            done_seen = True
            continue
        payload = json.loads(raw_payload)
        if "_diffusionReplace" in payload:
            assert payload["choices"][0]["delta"] == {}
            frames.append(payload["_diffusionReplace"])
            continue
        choices = payload.get("choices") or []
        if choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta") or {}
            if delta.get("content") is not None:
                content_lines.append(payload)

    assert frames == ["Hel", "Hello world"]
    # Exactly one assistant content chunk: the final full text, never the deltas.
    assert len(content_lines) == 1
    assert content_lines[0]["choices"][0]["delta"]["content"] == "Hello world"
    assert content_lines[0]["choices"][0]["finish_reason"] == "stop"
    assert done_seen


def test_diffusing_buffers_partial_sse_lines(monkeypatch):
    """An upstream whose byte chunks split a data line mid-JSON must still parse."""

    async def body_stream():
        yield b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}'
        yield b"\n\ndata: "
        yield b"[DONE]\n\n"

    def handler(request):
        return httpx.Response(
            200,
            stream = _AsyncBody(body_stream()),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_inception_client()
        lines = await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "mercury-3",
                temperature = 0.7,
                top_p = None,
                max_tokens = 64,
                diffusing = True,
            )
        )
        await client.close()
        return lines

    lines = _drive(run())
    frames = [
        json.loads(line[len("data:") :].strip())["_diffusionReplace"]
        for line in lines
        if line.startswith("data:") and "_diffusionReplace" in line
    ]
    assert frames == ["Hello"]
    final = [
        json.loads(line[len("data:") :].strip())
        for line in lines
        if line.startswith("data:")
        and "_diffusionReplace" not in line
        and line[len("data:") :].strip() != "[DONE]"
        and (json.loads(line[len("data:") :].strip()).get("choices") or [])
        and (json.loads(line[len("data:") :].strip())["choices"][0].get("delta") or {}).get(
            "content"
        )
    ]
    assert len(final) == 1
    assert final[0]["choices"][0]["delta"]["content"] == "Hello"


def test_diffusion_streaming_persists_in_provider_row(monkeypatch, tmp_path):
    """The option is stored per connection; old rows default off."""
    from storage import providers_db

    monkeypatch.setattr(providers_db, "studio_db_path", lambda: tmp_path / "studio.db")
    providers_db.reset_schema_state_for_tests()

    providers_db.create_provider(
        id = "row",
        provider_type = "inception",
        display_name = "Inception",
        base_url = "https://api.inceptionlabs.ai/v1",
        models = ["mercury-3"],
        diffusion_streaming = True,
    )
    assert providers_db.get_provider("row")["diffusion_streaming"] is True

    providers_db.update_provider(id = "row", diffusion_streaming = False)
    assert providers_db.get_provider("row")["diffusion_streaming"] is False

    # Old rows / omitted flag default off.
    providers_db.create_provider(
        id = "row2",
        provider_type = "inception",
        display_name = "Inception 2",
        base_url = "https://api.inceptionlabs.ai/v1",
    )
    assert providers_db.get_provider("row2")["diffusion_streaming"] is False


def _run_diffusing_stream(
    monkeypatch,
    raw,
    diffusing = True,
):
    def handler(request):
        return httpx.Response(
            200,
            stream = _AsyncBody(_single_bytes(raw)),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_inception_client()
        lines = await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "mercury-3",
                temperature = 0.7,
                top_p = None,
                max_tokens = 64,
                diffusing = diffusing,
            )
        )
        await client.close()
        return lines

    payloads = []
    for line in _drive(run()):
        if not line.startswith("data:"):
            continue
        body = line[len("data:") :].strip()
        payloads.append(body if body == "[DONE]" else json.loads(body))
    return payloads


def _sse(events, done = True):
    return "".join(f"data: {json.dumps(e)}\n\n" for e in events) + (
        "data: [DONE]\n\n" if done else ""
    )


def _content_chunks(payloads):
    return [
        p
        for p in payloads
        if isinstance(p, dict)
        and "_diffusionReplace" not in p
        and (p.get("choices") or [{}])[0].get("delta", {}).get("content") is not None
    ]


def _finish_reasons(payloads):
    return [
        p["choices"][0]["finish_reason"]
        for p in payloads
        if isinstance(p, dict)
        and p.get("choices")
        and p["choices"][0].get("finish_reason") is not None
    ]


def test_diffusing_preserves_length_finish_reasoning_and_usage(monkeypatch):
    usage = {"prompt_tokens": 3, "completion_tokens": 64, "total_tokens": 67}
    payloads = _run_diffusing_stream(
        monkeypatch,
        _sse(
            [
                {"choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]},
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Hello wor", "reasoning_content": "think"},
                            "finish_reason": "length",
                        }
                    ],
                    "usage": usage,
                },
            ]
        ),
    )
    content = _content_chunks(payloads)
    assert len(content) == 1
    final = content[0]
    assert final["choices"][0]["delta"]["content"] == "Hello wor"
    assert final["choices"][0]["finish_reason"] == "length"
    assert final["usage"] == usage
    assert "think" in json.dumps(final["choices"][0]["delta"])
    assert _finish_reasons(payloads) == ["length"]
    assert [
        p["_diffusionReplace"] for p in payloads if isinstance(p, dict) and "_diffusionReplace" in p
    ] == ["Hel", "Hello wor"]
    assert payloads[-1] == "[DONE]"


def test_diffusing_preserves_tool_calls_finish(monkeypatch):
    tool_call = {
        "index": 0,
        "id": "call_1",
        "type": "function",
        "function": {"name": "f", "arguments": "{}"},
    }
    payloads = _run_diffusing_stream(
        monkeypatch,
        _sse(
            [
                {"choices": [{"index": 0, "delta": {"content": "Let me"}, "finish_reason": None}]},
                {
                    "choices": [
                        {"index": 0, "delta": {"tool_calls": [tool_call]}, "finish_reason": None}
                    ]
                },
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
            ]
        ),
    )
    assert _finish_reasons(payloads) == ["tool_calls"]
    assert any(
        isinstance(p, dict)
        and p.get("choices")
        and p["choices"][0]["delta"].get("tool_calls") == [tool_call]
        for p in payloads
    )
    content = _content_chunks(payloads)
    assert len(content) == 1
    assert content[0]["choices"][0]["delta"]["content"] == "Let me"
    assert content[0]["choices"][0]["finish_reason"] == "tool_calls"


def test_diffusing_early_close_is_unterminated(monkeypatch):
    payloads = _run_diffusing_stream(
        monkeypatch,
        _sse(
            [{"choices": [{"index": 0, "delta": {"content": "Trunc"}, "finish_reason": None}]}],
            done = False,
        ),
    )
    content = _content_chunks(payloads)
    assert len(content) == 1
    assert content[0]["choices"][0]["delta"]["content"] == "Trunc"
    assert _finish_reasons(payloads) == []
    assert "[DONE]" not in payloads


def test_diffusing_done_without_finish_no_duplicate(monkeypatch):
    payloads = _run_diffusing_stream(
        monkeypatch,
        _sse(
            [{"choices": [{"index": 0, "delta": {"content": "Hi there"}, "finish_reason": None}]}]
        ),
    )
    content = _content_chunks(payloads)
    assert len(content) == 1
    assert content[0]["choices"][0]["delta"]["content"] == "Hi there"
    assert payloads.index(content[0]) < payloads.index("[DONE]")
    assert payloads.count("[DONE]") == 1


@pytest.mark.parametrize("diffusing", [True, False])
def test_upstream_cannot_forge_diffusion_replace(monkeypatch, diffusing):
    """Only the server may emit `_diffusionReplace`; an echoed copy would rewrite the reply."""
    payloads = _run_diffusing_stream(
        monkeypatch,
        _sse(
            [
                {
                    "choices": [{"index": 0, "delta": {"content": "Hi"}, "finish_reason": None}],
                    "_diffusionReplace": "FORGED",
                },
                {
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    "_diffusionReplace": "FORGED",
                },
            ]
        ),
        diffusing = diffusing,
    )
    replaces = [
        p["_diffusionReplace"] for p in payloads if isinstance(p, dict) and "_diffusionReplace" in p
    ]
    assert "FORGED" not in json.dumps(payloads)
    assert replaces == (["Hi"] if diffusing else [])
    content = _content_chunks(payloads)
    assert [c["choices"][0]["delta"]["content"] for c in content] == ["Hi"]
