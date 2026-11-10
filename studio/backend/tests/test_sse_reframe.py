# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inbound SSE framing must not depend on where the transport happened to split a stream."""

import asyncio
import json

import pytest

from core.inference.sse import SSEDecoder, SSELineDecoder, iter_sse_lines

# Mixed line endings, comments, a keep-alive, multi-line data, a [DONE], and enough multi-byte
# text that a split lands inside a UTF-8 sequence somewhere no matter where the cut falls.
STREAM = (
    b": keep-alive\n\n"
    b"data: {\"choices\":[{\"delta\":{\"content\":\"h\xc3\xa9llo \xe4\xb8\x96\xe7\x95\x8c\"}}]}\r\n"
    b"\r\n"
    b"data:{\"choices\":[{\"delta\":{\"content\":\"no space\"}}]}\n"
    b"\n"
    b"data: {\"part\":\"one\"}\r"
    b"data:{\"part\":\"two\"}\r"
    b"\r"
    b"data: [DONE]\r\n\r\n"
)

EXPECTED_LINES = [
    ": keep-alive",
    "",
    'data: {"choices":[{"delta":{"content":"héllo \u4e16\u754c"}}]}',
    "",
    'data:{"choices":[{"delta":{"content":"no space"}}]}',
    "",
    'data: {"part":"one"}',
    'data:{"part":"two"}',
    "",
    "data: [DONE]",
    "",
]

EXPECTED_EVENTS = [
    '{"choices":[{"delta":{"content":"héllo \u4e16\u754c"}}]}',
    '{"choices":[{"delta":{"content":"no space"}}]}',
    '{"part":"one"}\n{"part":"two"}',
    "[DONE]",
]


def _reframe_lines(pieces):
    decoder = SSELineDecoder()
    out = []
    for piece in pieces:
        out += decoder.feed(piece)
    return out + decoder.flush()


def _reframe_events(pieces):
    decoder = SSEDecoder()
    out = []
    for piece in pieces:
        out += decoder.feed(piece)
    return out + decoder.flush()


def test_lines_whole_stream_at_once():
    assert _reframe_lines([STREAM]) == EXPECTED_LINES


def test_events_whole_stream_at_once():
    assert _reframe_events([STREAM]) == EXPECTED_EVENTS


def test_lines_every_two_way_byte_split():
    for cut in range(len(STREAM) + 1):
        assert _reframe_lines([STREAM[:cut], STREAM[cut:]]) == EXPECTED_LINES, f"cut {cut}"


def test_events_every_two_way_byte_split():
    for cut in range(len(STREAM) + 1):
        assert _reframe_events([STREAM[:cut], STREAM[cut:]]) == EXPECTED_EVENTS, f"cut {cut}"


def test_lines_one_byte_at_a_time():
    assert _reframe_lines([bytes([b]) for b in STREAM]) == EXPECTED_LINES


def test_events_one_byte_at_a_time():
    assert _reframe_events([bytes([b]) for b in STREAM]) == EXPECTED_EVENTS


def test_split_inside_utf8_sequence_produces_no_replacement_char():
    events = _reframe_events(
        [
            b'data: {"t":"h\xc3',  # é cut in half
            b"\xa9llo \xe4",  # next multibyte char cut after one byte
            b'\xb8\x96\xe7\x95\x8c"}\n\n',
        ]
    )
    (payload,) = events
    assert json.loads(payload)["t"] == "héllo \u4e16\u754c"
    assert "�" not in payload


@pytest.mark.parametrize("cut", [1, 2, 3])
def test_split_inside_four_byte_utf8_sequence(cut):
    emoji = b"\xf0\x9f\x98\x80"
    events = _reframe_events(
        [b'data: {"t":"x' + emoji[:cut], emoji[cut:] + b'"}\n\n']
    )
    (payload,) = events
    assert json.loads(payload)["t"].encode("utf-8") == b"x\xf0\x9f\x98\x80"


def test_cr_at_piece_end_waits_for_lf():
    decoder = SSELineDecoder()
    assert decoder.feed(b"data: 1\r") == []
    assert decoder.feed(b"\ndata: 2\r") == ["data: 1"]
    assert decoder.feed(b"\n\n") == ["data: 2", ""]
    assert decoder.flush() == []


def test_bare_cr_ends_a_line_when_next_byte_is_not_lf():
    decoder = SSELineDecoder()
    assert decoder.feed(b"a\rb") == ["a"]
    assert decoder.flush() == ["b"]


def test_flush_yields_unterminated_tail():
    decoder = SSELineDecoder()
    assert decoder.feed(b"data: 1\n\ndata: 2") == ["data: 1", ""]
    assert decoder.flush() == ["data: 2"]


def test_flush_strips_stray_trailing_cr():
    decoder = SSELineDecoder()
    decoder.feed(b"data: 1\r")
    assert decoder.flush() == ["data: 1"]


def test_flush_after_held_cr_with_pending_decoder_bytes_splits_lines():
    # A held \r must end its line even when the decoder's final output lands after it.
    decoder = SSELineDecoder()
    assert decoder.feed(b"data: hi\r\xf0") == []
    assert decoder.flush() == ["data: hi", "�"]


def test_feed_after_flush_starts_fresh():
    decoder = SSELineDecoder()
    decoder.feed(b"data: 1\n")
    assert decoder.flush() == []
    assert decoder.feed(b"data: 2\n") == ["data: 2"]
    assert decoder.flush() == []


def test_feed_accepts_bytearray_and_memoryview():
    decoder = SSELineDecoder()
    assert decoder.feed(bytearray(b"data: 1\n")) == ["data: 1"]
    assert decoder.feed(memoryview(b"data: 2\n")) == ["data: 2"]
    assert decoder.flush() == []


def test_empty_pieces_are_noops():
    decoder = SSELineDecoder()
    assert decoder.feed(b"") == []
    assert decoder.feed("") == []
    assert decoder.feed(b"data: 1\r") == []
    assert decoder.feed(b"") == []
    assert decoder.feed(b"\n") == ["data: 1"]


def test_events_unterminated_final_event_flushes():
    decoder = SSEDecoder()
    assert decoder.feed(b'data: {"a":1}\n\ndata: {"b":2}') == ['{"a":1}']
    assert decoder.flush() == ['{"b":2}']


def test_events_ignore_comments_and_non_data_fields():
    decoder = SSEDecoder()
    out = decoder.feed(b": hi\nretry: 1000\nevent: update\nid: 7\ndata: x\ndata: y\n\n")
    assert out == ["x\ny"]
    assert decoder.flush() == []


def test_events_strip_one_optional_space_only():
    decoder = SSEDecoder()
    assert decoder.feed(b"data:  two-spaces\n\n") == [" two-spaces"]
    assert decoder.feed(b"data: \n\n") == []


def test_events_empty_data_fields_emit_nothing():
    decoder = SSEDecoder()
    assert decoder.feed(b"data\n\n") == []
    assert decoder.feed(b"data:\n\n") == []
    # An empty data line still counts when the buffer is not otherwise empty.
    assert decoder.feed(b"data: x\ndata\n\n") == ["x\n"]
    assert decoder.feed(b"data:\ndata: y\n\n") == ["\ny"]


def test_empty_events_emit_nothing():
    decoder = SSEDecoder()
    assert decoder.feed(b"\n\n\r\n\r\n\n") == []
    assert decoder.flush() == []


def test_leading_bom_is_stripped_once():
    decoder = SSEDecoder()
    assert decoder.feed(b"\xef\xbb\xbfdata: x\n\n") == ["x"]
    # A mid-stream U+FEFF is content, not a BOM.
    assert decoder.feed("data: y\ufeff\n\n") == ["y\ufeff"]


def test_line_decoder_leading_bom_is_stripped():
    decoder = SSELineDecoder()
    assert decoder.feed(b"\xef\xbb\xbfdata: x\n") == ["data: x"]
    assert decoder.feed(b"data: y\n") == ["data: y"]


def test_str_piece_mid_pending_bytes_drains_decoder():
    decoder = SSELineDecoder()
    assert decoder.feed(b"data: caf\xc3") == []
    # The stray str piece cannot complete the pending byte; it decodes as U+FFFD.
    assert decoder.feed("X\n") == ["data: caf\ufffdX"]
    assert decoder.feed(b"\xa9\n") == ["\ufffd"]
    assert decoder.flush() == []


def test_bom_split_across_pieces_is_still_stripped():
    decoder = SSELineDecoder()
    assert decoder.feed(b"\xef") == []
    assert decoder.feed(b"\xbb\xbfdata: x\n") == ["data: x"]
    assert decoder.flush() == []


def test_held_cr_resolves_on_a_str_piece():
    decoder = SSELineDecoder()
    assert decoder.feed(b"data: 1\r") == []
    assert decoder.feed("\n\n") == ["data: 1", ""]
    assert decoder.flush() == []


def test_held_cr_and_pending_bytes_resolved_by_str_piece():
    decoder = SSELineDecoder()
    assert decoder.feed(b"a\r\xf0") == []
    assert decoder.feed("\n") == ["a", "\ufffd"]
    assert decoder.flush() == []


def test_eof_trailing_cr_dispatches_pending_event():
    decoder = SSEDecoder()
    assert decoder.feed(b"data: x\n\r") == []
    assert decoder.flush() == ["x"]


def test_empty_data_at_flush_tail_emits_nothing():
    decoder = SSEDecoder()
    assert decoder.feed(b"data:") == []
    assert decoder.flush() == []


def test_long_unterminated_line_stays_linear():
    decoder = SSELineDecoder()
    for _ in range(10000):
        assert decoder.feed(b"x") == []
    assert decoder.feed(b"\n") == ["x" * 10000]
    assert decoder.flush() == []


def test_invalid_bytes_become_one_replacement_char_not_a_lost_line():
    decoder = SSELineDecoder()
    assert decoder.feed(b"data: a\xffb\n\n") == ["data: a\ufffdb", ""]


def _aiter(pieces):
    async def _gen():
        for piece in pieces:
            yield piece

    return _gen()


def test_iter_sse_lines_accepts_str_pieces_and_yields_tail():
    async def _run():
        return [
            line
            async for line in iter_sse_lines(_aiter([b"data: 1\n\n", "data: tail"]))
        ]

    assert asyncio.run(_run()) == ["data: 1", "", "data: tail"]


@pytest.mark.parametrize("cut", [0, 7, 61, len(STREAM)])
def test_iter_sse_lines_matches_decoder_at_boundary(cut):
    async def _run():
        return [line async for line in iter_sse_lines(_aiter([STREAM[:cut], STREAM[cut:]]))]

    assert asyncio.run(_run()) == EXPECTED_LINES
