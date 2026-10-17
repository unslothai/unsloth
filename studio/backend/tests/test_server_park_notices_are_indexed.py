# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server indexes its park notices for a batched request: `: preempted 1`.

One boolean read `preempted 0, preempted 1, resumed 0` as resumed while prompt 1 was still
parked, and the relay mapped only the unindexed spelling, so an indexed notice reached nobody.
"""

from __future__ import annotations

import pytest

from core.inference import llama_preemption as preemption
import routes.inference as inference


def _feed_split(body: bytes, cut: int) -> preemption.ServerParkNotices:
    notices = preemption.ServerParkNotices(lambda: False)
    notices.feed(body[:cut])
    notices.feed(body[cut:])
    return notices


class TestTheGrammar:
    @pytest.mark.parametrize(
        "line, expected",
        [
            (": preempted", ("preempted", 0)),
            (": resumed", ("resumed", 0)),
            (": recomputed", ("recomputed", 0)),
            (": preempted 0", ("preempted", 0)),
            (": preempted 1", ("preempted", 1)),
            (": resumed 12", ("resumed", 12)),
            (": recomputed 3", ("recomputed", 3)),
            (b": preempted 2\r", ("preempted", 2)),
        ],
    )
    def test_notices(self, line, expected):
        assert preemption.parse_server_park_notice(line) == expected

    @pytest.mark.parametrize(
        "line",
        [
            ": preempted 01",
            ": preempted -1",
            ": preempted 1 2",
            ": preempted x",
            ": preemptedx",
            ": preempted  1",
            ": preempt-keepalive",
            "data: : preempted 1",
            'data: {"content": ": preempted 1"}',
            "preempted 1",
            "",
            None,
        ],
    )
    def test_everything_else_is_not_one(self, line):
        assert preemption.parse_server_park_notice(line) is None


class TestTheRelay:
    @pytest.mark.parametrize(
        "line, comment",
        [
            (": preempted 1", inference._OPENAI_PREEMPT_SSE_PAUSED),
            (": resumed 1", inference._OPENAI_PREEMPT_SSE_RESUMED),
            (": recomputed 1", inference._OPENAI_PREEMPT_SSE_RECOMPUTED),
            (b": preempted 0\r\n", inference._OPENAI_PREEMPT_SSE_PAUSED),
            (": preempted", inference._OPENAI_PREEMPT_SSE_PAUSED),
            (": preempt-keepalive", inference._OPENAI_PREEMPT_SSE_KEEPALIVE),
        ],
    )
    def test_indexed_notices_reach_the_client_in_the_form_it_reads(self, line, comment):
        assert inference._server_park_sse(line) == comment

    def test_lookalikes_are_not_relayed(self):
        assert inference._server_park_sse(": preempted 01") is None
        assert inference._server_park_sse(": preempt-keepalive 1") is None
        assert inference._server_park_sse('data: {"x": ": preempted 1"}') is None


class TestTheParkedSet:
    def test_one_prompt_resuming_leaves_the_other_parked(self):
        notices = preemption.ServerParkNotices(lambda: False)
        notices.feed(b": preempted 0\n\n: preempted 1\n\n: resumed 0\n\n")
        assert notices.parked is True
        assert notices.excuses_silence() is True
        notices.feed(b": resumed 1\n\n")
        assert notices.parked is False
        assert notices.excuses_silence() is False

    def test_duplicates_are_idempotent(self):
        notices = preemption.ServerParkNotices(lambda: True)
        notices.feed(b": preempted 1\n\n: preempted 1\n\n: resumed 1\n\n")
        assert notices.parked is False
        notices.feed(b": resumed 1\n\n: resumed 1\n\n")
        assert notices.parked is False
        assert notices.heard_a_notice is True
        assert notices.excuses_silence() is False

    def test_unindexed_is_index_zero(self):
        # A build that leaves index 0 unindexed and indexes the rest.
        notices = preemption.ServerParkNotices(lambda: False)
        notices.feed(b": preempted\n\n: preempted 1\n\n: resumed\n\n")
        assert notices.parked is True
        notices.feed(b": resumed 1\n\n")
        assert notices.parked is False

    def test_a_recompute_does_not_park(self):
        notices = preemption.ServerParkNotices(lambda: False)
        notices.feed(b": preempted 1\n\n: resumed 1\n\n: recomputed 1\n\n")
        assert notices.parked is False
        assert notices.heard_a_notice is True

    def test_crlf(self):
        notices = preemption.ServerParkNotices(lambda: False)
        notices.feed(
            b"data: a\r\n\r\n: preempted 0\r\n\r\n: preempted 1\r\n\r\n: resumed 0\r\n\r\n"
        )
        assert notices.parked is True

    def test_a_payload_quoting_an_indexed_notice_is_not_one(self):
        notices = preemption.ServerParkNotices(lambda: False)
        notices.feed(b'data: {"content":"x\\n: preempted 1\\n"}\n\n')
        notices.feed(b"data: : preempted 1\n\n")
        assert notices.heard_a_notice is False
        assert notices.parked is False

    @pytest.mark.parametrize("newline", [b"\n", b"\r\n", b"\r"])
    def test_every_byte_boundary(self, newline):
        body = newline.join(
            [
                b'data: {"content":"hi"}',
                b"",
                b": preempted 0",
                b"",
                b": preempted 1",
                b"",
                b": preempt-keepalive",
                b"",
                b": resumed 0",
                b"",
                b": recomputed 0",
                b"",
                b"",
            ]
        )
        for cut in range(len(body) + 1):
            notices = _feed_split(body, cut)
            assert notices.parked is True, (newline, cut)
            assert notices.heard_a_notice is True, (newline, cut)

    def test_every_byte_boundary_into_three_reads(self):
        body = b": preempted 1\n\n: preempted 2\n\n: resumed 2\n\n: resumed 1\n\n"
        for a in range(len(body) + 1):
            for b in range(a, len(body) + 1):
                notices = preemption.ServerParkNotices(lambda: False)
                notices.feed(body[:a])
                notices.feed(body[a:b])
                notices.feed(body[b:])
                assert notices.parked is False, (a, b)
                assert notices.heard_a_notice is True, (a, b)

    def test_line_oriented_source(self):
        notices = preemption.ServerParkNotices(lambda: False)
        notices.feed_line(": preempted 0")
        notices.feed_line(": preempted 1")
        notices.feed_line(": resumed 0")
        assert notices.parked is True
        notices.feed_line(b": resumed 1\r")
        assert notices.parked is False
