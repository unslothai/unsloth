# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient
from core.inference.external_tool_transport import OAICompatTransport
from models.inference import ChatMessage
from routes.inference import _build_external_messages

_SSE = 'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'


def _send(
    version,
    provider_type = "vllm",
    turns = 1,
    messages = None,
    **kwargs,
):
    """Chat `turns` times against one server; returns the chat bodies and the /version reads."""
    bodies: list[dict] = []
    version_reads: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            version_reads.append(str(request.url))
            if version is None:
                return httpx.Response(404, json = {"detail": "Not Found"})
            return httpx.Response(200, json = {"version": version})
        bodies.append(json.loads(request.content.decode()))
        return httpx.Response(200, content = _SSE, headers = {"content-type": "text/event-stream"})

    mock_client = httpx.AsyncClient(transport = httpx.MockTransport(handler))
    client = ExternalProviderClient(
        provider_type = provider_type, base_url = "http://127.0.0.1:8000/v1", api_key = "k"
    )

    async def run() -> None:
        try:
            for _ in range(turns):
                async for _ in client.stream_chat_completion(
                    messages = messages or [{"role": "user", "content": "hi"}],
                    model = "Qwen/Qwen3.8-27B",
                    **kwargs,
                ):
                    pass
        finally:
            await client.close()
            await mock_client.aclose()

    loop = asyncio.new_event_loop()
    previous = ep_mod._http_client
    ep_mod._http_client = mock_client
    ep_mod._served_versions.clear()
    try:
        loop.run_until_complete(run())
    finally:
        ep_mod._http_client = previous
        ep_mod._served_versions.clear()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
    return bodies, version_reads


@pytest.mark.parametrize("effort", ["minimal", "xhigh", "max"])
@pytest.mark.parametrize("version", ["0.21.0", "0.31.0", "0.31.1.dev12+gabc1234"])
def test_a_current_vllm_receives_the_level_verbatim(effort, version):
    [body], reads = _send(version, reasoning_effort = effort, enable_thinking = True)
    assert body["reasoning_effort"] == effort
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert reads == ["http://127.0.0.1:8000/version"]


@pytest.mark.parametrize("version", ["0.20.2", "0.21.0rc1", "0.11.0", "not-a-version", None])
def test_an_older_or_silent_vllm_leaves_a_wider_level_to_the_template_default(version):
    # Rewriting xhigh to high would 400 on Qwen3.8, whose template accepts low | medium | xhigh.
    for effort in ("minimal", "xhigh", "max"):
        [body], _ = _send(version, reasoning_effort = effort, enable_thinking = True)
        assert "reasoning_effort" not in body
        assert body["chat_template_kwargs"]["enable_thinking"] is True


def test_the_common_levels_never_read_the_version():
    for effort in ("low", "medium", "high", "none"):
        _, reads = _send("0.31.0", reasoning_effort = effort)
        assert reads == []


def test_the_version_is_read_once_per_minute_not_per_turn():
    bodies, reads = _send("0.31.0", turns = 3, reasoning_effort = "xhigh")
    assert [b["reasoning_effort"] for b in bodies] == ["xhigh"] * 3
    assert len(reads) == 1


def test_llama_server_still_clamps_without_asking():
    [body], reads = _send("0.31.0", provider_type = "llama_cpp", reasoning_effort = "xhigh")
    assert body["reasoning_effort"] == "high"
    assert reads == []


def test_preserve_thinking_rides_beside_enable_thinking():
    [body], _ = _send(
        "0.31.0", reasoning_effort = "xhigh", enable_thinking = True, preserve_thinking = True
    )
    assert body["chat_template_kwargs"] == {"preserve_thinking": True, "enable_thinking": True}
    [body], _ = _send("0.31.0", preserve_thinking = False)
    assert body["chat_template_kwargs"] == {"preserve_thinking": False}
    [body], _ = _send("0.31.0")
    assert "chat_template_kwargs" not in body


def test_vllm_reads_a_past_thought_from_reasoning():
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a", "reasoning_content": "prior thought"},
        {"role": "user", "content": "next"},
    ]
    [body], _ = _send("0.31.0", messages = history, preserve_thinking = True)
    assert body["messages"][1] == {
        "role": "assistant",
        "content": "a",
        "reasoning": "prior thought",
    }
    [body], _ = _send("0.31.0", provider_type = "llama_cpp", messages = history)
    assert body["messages"][1]["reasoning_content"] == "prior thought"


@pytest.mark.parametrize("preserve", [True, False, None])
def test_vllm_history_replays_reasoning_only_when_preserved(preserve):
    message = ChatMessage(role = "assistant", content = "answer", reasoning_content = "prior thought")
    [out] = _build_external_messages(
        [message], False, provider_type = "vllm", preserve_thinking = preserve
    )
    assert out.get("reasoning_content") == ("prior thought" if preserve is True else None)


@pytest.mark.parametrize("value", [True, False, None])
def test_vllm_tool_loop_keeps_reasoning_only_when_preserved(value):
    client = ExternalProviderClient(
        provider_type = "vllm", base_url = "http://localhost:8000/v1", api_key = ""
    )
    transport = OAICompatTransport(client, model = "test", preserve_thinking = value)
    assert transport.preserves_reasoning is (value is True)
