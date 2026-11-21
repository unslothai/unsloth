# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

from routes.inference import _enforce_stop_sequences


def _chunk(content=None, finish=None):
    delta = {} if content is None else {"content": content}
    return "data: " + json.dumps(
        {"id": "c1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
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
    finishes = [c["finish_reason"] for line in out if line.startswith("data: {") for c in json.loads(line[6:])["choices"]]
    assert finishes[-1] == "stop"
    assert out[-1] == "data: [DONE]"
    assert closed == [True]


def test_a_held_back_tail_that_never_completes_a_stop_is_sent_before_done():
    out, _ = _run([_chunk("ends with \nR"), "data: [DONE]"], ["\nRin:"])
    assert _text(out) == "ends with \nR"
    assert out[-1] == "data: [DONE]"

