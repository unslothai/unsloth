# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json

import httpx
import pytest

from core.inference.http_stream import aiter_sse_lines, closing_response_lines

SEPARATORS = ["\u2028", "\u2029", "\x85"]


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk

    async def aclose(self):
        self.closed = True


def _response(chunks) -> httpx.Response:
    return httpx.Response(200, stream = _Chunks([c.encode() for c in chunks]))


async def _collect(lines):
    return [line async for line in lines]


def _lines(chunks):
    return asyncio.run(_collect(aiter_sse_lines(_response(chunks))))


def _httpx_lines(chunks):
    return asyncio.run(_collect(_response(chunks).aiter_lines()))


def _event(content: str) -> str:
    payload = {"choices": [{"index": 0, "delta": {"content": content}}]}
    return "data: " + json.dumps(payload, ensure_ascii = False)


WIRE = 'data: {"a": 1}\r\n\r\n: keep-alive\n\ndata: x\rdata: y\r\rdata: [DONE]\n\ntail'


@pytest.mark.parametrize("cut", range(len(WIRE) + 1))
def test_matches_httpx_lines_at_every_chunk_boundary(cut):
    chunks = [WIRE[:cut], WIRE[cut:]]
    assert _lines(chunks) == _httpx_lines(chunks)


@pytest.mark.parametrize("separator", SEPARATORS)
@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_unicode_separators_stay_inside_their_line(separator, newline):
    content = f"first{separator}second"
    wire = _event(content) + newline + newline
    for cut in range(len(wire) + 1):
        lines = _lines([wire[:cut], wire[cut:]])
        assert lines == [_event(content), ""], cut
        assert json.loads(lines[0][6:])["choices"][0]["delta"]["content"] == content


def test_a_held_cr_ends_its_line_when_the_next_chunk_arrives():
    seen_before_last = []

    class _Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: 1\r"
            yield b"x"
            seen_before_last.extend(lines)
            yield b"\n"

    lines = []

    async def run():
        async for line in aiter_sse_lines(httpx.Response(200, stream = _Stream())):
            lines.append(line)

    asyncio.run(run())
    assert seen_before_last == ["data: 1"]
    assert lines == ["data: 1", "x"]


def test_closing_response_lines_closes_the_stream():
    stream = _Chunks([_event("a\u2028b").encode() + b"\n\n"])
    response = httpx.Response(200, stream = stream)
    assert asyncio.run(_collect(closing_response_lines(response))) == [_event("a\u2028b"), ""]
    assert stream.closed


def test_llama_event_payload_reads_a_separator_delta():
    from core.inference.llama_cpp import LlamaCppBackend
    event = _event("a\u2028b") + "\n\n"
    assert LlamaCppBackend._sse_event_has_generated_output(event)


def test_openai_monitor_flags_an_upstream_error_holding_a_separator():
    from routes.inference import _monitor_openai_sse_event

    error = {"error": {"message": "bad\u2028input", "code": 500}}
    event = ("data: " + json.dumps(error, ensure_ascii = False) + "\n\n").encode()
    assert _monitor_openai_sse_event(None, event) == "error"
