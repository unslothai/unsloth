# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/inference/cancel only sets a threading.Event, so the transport has to watch it.

Every provider re-yields through ``stream_chat_completion``, which parks in an
await for the whole of prefill and streaming. Without a watcher a Stop is
invisible until the provider emits again: billed tokens keep arriving and a
model load blocks behind the stalled request.
"""

import asyncio
import threading

from core.inference.external_tool_transport import OAICompatTransport


class _StallingClient:
    """One chunk, then silence: an upstream mid-answer or still in prefill."""

    provider_type = "custom"

    def __init__(self) -> None:
        self.torn_down = False
        self.released = asyncio.Event()

    async def stream_chat_completion(self, **_kwargs):
        try:
            yield 'data: {"choices":[{"delta":{"content":"hi"}}]}'
            await self.released.wait()
            yield "data: [DONE]"
        finally:
            # Where the real client awaits response.aclose().
            self.torn_down = True


def _transport(client):
    return OAICompatTransport(client, model = "local-model")


def test_cancel_closes_a_stalled_upstream_stream():
    async def scenario():
        client = _StallingClient()
        cancel_event = threading.Event()
        seen: list[str] = []

        async def consume():
            async for line in _transport(client).stream(
                messages = [{"role": "user", "content": "hi"}],
                tools = None,
                tool_choice = "auto",
                cancel_event = cancel_event,
            ):
                seen.append(line)

        task = asyncio.ensure_future(consume())
        await asyncio.sleep(0.1)
        assert seen and not client.torn_down, "should still be parked on the provider"

        cancel_event.set()
        await asyncio.wait_for(task, timeout = 5.0)
        assert client.torn_down, "cancel must close the upstream, not await the next chunk"

    asyncio.run(scenario())


def test_closing_the_generator_still_tears_the_upstream_down():
    """The pre-existing GeneratorExit teardown must survive the watcher."""

    async def scenario():
        client = _StallingClient()
        generator = _transport(client).stream(
            messages = [{"role": "user", "content": "hi"}],
            tools = None,
            tool_choice = "auto",
            cancel_event = threading.Event(),
        )
        assert (await generator.__anext__()).startswith("data:")
        await generator.aclose()
        assert client.torn_down

    asyncio.run(scenario())


def test_an_uncancelled_stream_relays_every_line():
    async def scenario():
        client = _StallingClient()
        client.released.set()
        lines = [
            line
            async for line in _transport(client).stream(
                messages = [{"role": "user", "content": "hi"}],
                tools = None,
                tool_choice = "auto",
                cancel_event = threading.Event(),
            )
        ]
        assert lines[-1] == "data: [DONE]"
        assert len(lines) == 2
        assert client.torn_down

    asyncio.run(scenario())


def test_each_transport_turn_is_refitted_before_the_provider_call():
    class RecordingClient:
        provider_type = "custom"

        def __init__(self) -> None:
            self.requests: list[dict] = []

        async def stream_chat_completion(self, **kwargs):
            self.requests.append(kwargs)
            yield "data: [DONE]"

    async def scenario():
        client = RecordingClient()
        fitted_inputs: list[list[dict]] = []

        async def fitter(messages):
            fitted_inputs.append(messages)
            return messages[-2:], 17, 'data: {"choices":[],"context_truncated":{}}'

        transport = OAICompatTransport(
            client,
            model = "local-model",
            max_tokens = 99,
            message_fitter = fitter,
        )
        histories = [
            [{"role": "user", "content": "first"}],
            [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "call"},
                {"role": "tool", "content": "large result"},
            ],
        ]
        relayed = []
        for messages in histories:
            relayed.append(
                [
                    line
                    async for line in transport.stream(
                        messages = messages,
                        tools = None,
                        tool_choice = "auto",
                        cancel_event = threading.Event(),
                    )
                ]
            )

        assert fitted_inputs == histories
        assert client.requests[0]["messages"] == histories[0]
        assert client.requests[1]["messages"] == histories[1][-2:]
        assert [request["max_tokens"] for request in client.requests] == [17, 17]
        assert all("context_truncated" in lines[0] for lines in relayed)

    asyncio.run(scenario())
