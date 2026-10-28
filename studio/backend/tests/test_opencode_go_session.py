# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""OpenCode Go requires a stable session header for each conversation."""

import asyncio

import httpx
import pytest

from core.inference import external_provider as ep


@pytest.mark.parametrize("stream", [True, False])
def test_opencode_go_chat_sends_conversation_session(monkeypatch, stream):
    """Requests from the same conversation carry the same OpenCode session ID."""
    requests = []

    def handler(request):
        """Capture the real outgoing request and supply a minimal completion."""
        requests.append(request)
        if stream:
            return httpx.Response(
                200, headers = {"content-type": "text/event-stream"}, content = "data: [DONE]\n\n"
            )
        return httpx.Response(
            200, json = {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    async def run():
        """Send two chat requests through the provider client."""
        async with httpx.AsyncClient(transport = httpx.MockTransport(handler)) as http:
            monkeypatch.setattr(ep, "_client", lambda: http)
            client = ep.ExternalProviderClient(
                "custom", "https://opencode.ai/inference/go/v1", "key"
            )
            for thread_id in ("conversation-1", "conversation-1", "conversation-2"):
                async for _ in client.stream_chat_completion(
                    messages = [{"role": "user", "content": "hi"}],
                    model = "test-model",
                    stream = stream,
                    thread_id = thread_id,
                ):
                    pass

    asyncio.run(run())
    sessions = [request.headers.get("x-opencode-session") for request in requests]
    assert len(sessions) == 3
    assert sessions[0] and sessions[0] == sessions[1]
    assert sessions[2] and sessions[2] != sessions[0]


@pytest.mark.parametrize("base_url", ["https://example.org/v1", "https://opencode.ai/inference/v1"])
def test_other_providers_do_not_receive_opencode_session(monkeypatch, base_url):
    """OpenCode-specific session identifiers are not sent to other providers."""
    requests = []

    def handler(request):
        """Capture the unrelated provider request."""
        requests.append(request)
        return httpx.Response(
            200, headers = {"content-type": "text/event-stream"}, content = "data: [DONE]\n\n"
        )

    async def run():
        """Exercise the same chat path against an unrelated provider."""
        async with httpx.AsyncClient(transport = httpx.MockTransport(handler)) as http:
            monkeypatch.setattr(ep, "_client", lambda: http)
            client = ep.ExternalProviderClient("custom", base_url, "key")
            async for _ in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "test-model",
                thread_id = "conversation-1",
            ):
                pass

    asyncio.run(run())
    assert len(requests) == 1
    assert "x-opencode-session" not in requests[0].headers


def test_opencode_go_responses_sends_conversation_session(monkeypatch):
    """The Responses API path sends a session header too."""
    requests = []

    def handler(request):
        """Capture a Responses request and return an empty completed response."""
        requests.append(request)
        return httpx.Response(200, json = {"id": "resp_test", "status": "completed", "output": []})

    async def run():
        """Send a Responses request through the actual provider client."""
        async with httpx.AsyncClient(transport = httpx.MockTransport(handler)) as http:
            monkeypatch.setattr(ep, "_client", lambda: http)
            client = ep.ExternalProviderClient(
                "custom", "https://opencode.ai/inference/go/openai/v1", "key", api_type = "responses"
            )
            async for _ in client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}],
                model = "test-model",
                stream = False,
                thread_id = "conversation-1",
            ):
                pass

    asyncio.run(run())
    assert len(requests) == 1
    assert requests[0].headers.get("x-opencode-session")
