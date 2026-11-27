# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

from routes.inference import _enforce_stop_sequences


def _chunk(content = None, finish = None):
    delta = {} if content is None else {"content": content}
    return "data: " + json.dumps(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
    )


def _run(lines, stop):
    closed = []

    async def upstream():
        try:
            for line in lines:
                yield line
        finally:
            closed.append(True)

    async def collect():
        return [line async for line in _enforce_stop_sequences(upstream(), stop, "m")]

    return asyncio.run(collect()), closed


def _text(out):
    text = ""
    for line in out:
        if line.startswith("data: {"):
            for choice in json.loads(line[6:])["choices"]:
                text += choice.get("delta", {}).get("content") or ""
    return text


def test_a_stop_split_across_chunks_cuts_the_reply_and_ends_the_provider_stream():
    out, closed = _run(
        [_chunk("Hello there.\nRi"), _chunk("n: hi"), _chunk(" more"), "data: [DONE]"], ["\nRin:"]
    )
    assert _text(out) == "Hello there."
    finishes = [
        c["finish_reason"]
        for line in out
        if line.startswith("data: {")
        for c in json.loads(line[6:])["choices"]
    ]
    assert finishes[-1] == "stop"
    assert out[-1] == "data: [DONE]"
    assert closed == [True]


def test_a_held_back_tail_that_never_completes_a_stop_is_sent_before_done():
    out, _ = _run([_chunk("ends with \nR"), "data: [DONE]"], ["\nRin:"])
    assert _text(out) == "ends with \nR"
    assert out[-1] == "data: [DONE]"


def test_a_held_back_tail_is_sent_when_the_provider_closes_without_done():
    out, closed = _run([_chunk("ends with \nR")], ["\nRin:"])
    assert _text(out) == "ends with \nR"
    assert closed == [True]


def test_held_back_text_is_sent_before_a_tool_call():
    tool = "data: " + json.dumps(
        {
            "id": "c1",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [{"index": 0, "id": "t1", "function": {"name": "f"}}]},
                    "finish_reason": None,
                }
            ],
        }
    )
    out, _ = _run(
        [_chunk("Hello\nRi"), tool, _chunk(finish = "tool_calls"), "data: [DONE]"], ["\nRin:"]
    )
    events = []
    for line in out:
        if line.startswith("data: {"):
            delta = json.loads(line[6:])["choices"][0]["delta"]
            if delta.get("content"):
                events.append(delta["content"])
            if delta.get("tool_calls"):
                events.append("TOOL")
    assert "".join(events[: events.index("TOOL")]) == "Hello\nRi"


def _tool_call_chunk():
    call = {
        "index": 0,
        "id": "t1",
        "type": "function",
        "function": {"name": "look", "arguments": "{}"},
    }
    return "data: " + json.dumps(
        {
            "id": "c1",
            "choices": [{"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}],
        }
    )


def _tool_event_chunk():
    return "data: " + json.dumps(
        {
            "id": "c1",
            "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
            "_toolEvent": {"type": "x"},
        }
    )


def _text_before_first(out, marker):
    text = ""
    for line in out:
        if marker in line:
            return text
        if line.startswith("data: {"):
            for choice in json.loads(line[6:])["choices"]:
                text += choice.get("delta", {}).get("content") or ""
    raise AssertionError(f"{marker} never sent")


def test_a_speaker_label_inside_provider_thinking_does_not_end_the_reply():
    out, _ = _run(
        [
            _chunk("<think>Consider the transcript:\nAl"),
            _chunk("ex: hello</th"),
            _chunk("ink>Hello there."),
            _chunk(None, "stop"),
            "data: [DONE]",
        ],
        ["\nAlex:"],
    )
    assert _text(out) == "<think>Consider the transcript:\nAlex: hello</think>Hello there."


def test_a_stop_after_thinking_still_cuts_the_reply():
    out, _ = _run(
        [_chunk("<think>plan</think>Sure.\nAlex: no"), _chunk(" more"), "data: [DONE]"], ["\nAlex:"]
    )
    assert _text(out) == "<think>plan</think>Sure."


def test_text_before_a_tool_call_or_tool_event_is_sent_before_it():
    for boundary, marker in (
        (_tool_call_chunk(), '"tool_calls"'),
        (_tool_event_chunk(), '"_toolEvent"'),
    ):
        out, _ = _run(
            [_chunk("Let me look that up."), boundary, _chunk(None, "tool_calls"), "data: [DONE]"],
            ["\nRin:"],
        )
        assert _text_before_first(out, marker) == "Let me look that up."
        assert _text(out) == "Let me look that up."


def test_framed_sse_streams_get_the_same_cut_and_close_the_provider_stream():
    from routes.inference import _sse_frames_with_stops

    closed = []

    async def frames():
        try:
            for frame in [
                ": keepalive\n\n",
                _chunk("Hello.\nAl") + "\n\n",
                _chunk("ex: hi") + "\n\n",
                _chunk(" more") + "\n\n",
            ]:
                yield frame
        finally:
            closed.append(True)

    async def collect():
        return [frame async for frame in _sse_frames_with_stops(frames(), ["\nAlex:"], "m")]

    out = asyncio.run(collect())
    assert all(frame.endswith("\n\n") for frame in out)
    assert out[0] == ": keepalive\n\n"
    assert _text([frame.rstrip("\n") for frame in out]) == "Hello."
    assert out[-1] == "data: [DONE]\n\n"
    assert closed == [True]


def test_a_codex_connection_reply_is_cut_at_a_stop(monkeypatch):
    from types import SimpleNamespace

    import routes.inference as inf
    from core.inference import openai_codex_client as codex_client
    from models.inference import ChatCompletionRequest

    from .test_external_disconnect_watcher_teardown import _codex_provider

    model = _codex_provider(monkeypatch)

    class FakeCodexClient:
        def __init__(self, *_args, **_kwargs):
            pass

        def stream(self, **_kwargs):
            async def _stream():
                for content in ("Hello there.\nAl", "ex: hi", " more"):
                    yield _chunk(content)

            return _stream()

        async def close(self):
            pass

    monkeypatch.setattr(codex_client, "OpenAICodexClient", FakeCodexClient)

    async def _run():
        async def is_disconnected():
            return False

        request = SimpleNamespace(
            headers = {},
            state = SimpleNamespace(skip_api_monitor = True),
            is_disconnected = is_disconnected,
        )
        payload = ChatCompletionRequest(
            messages = [{"role": "user", "content": "hello"}],
            provider_id = "codex-1",
            external_model = model,
            stream = True,
            stop = ["\nAlex:"],
        )
        response = await inf._proxy_to_external_provider(payload, request, current_subject = "t")
        return [chunk async for chunk in response.body_iterator]

    out = [str(chunk).rstrip("\n") for chunk in asyncio.run(_run())]
    assert _text(out) == "Hello there."
    assert out[-1] == "data: [DONE]"
