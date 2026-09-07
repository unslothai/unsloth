# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A durable chat run tells its follower about a pause.

The GUI streams a plain chat (no tools) through a durable run, whose worker reads the
internal SSE stream with ``_SSEDecoder`` and keeps data lines only. The pause comments
(`: preempt-paused`, `: preempt-resumed`) never reached it: measured with four browser
chats and nine server-side parks of 5 to 30 s each, "Paused while another chat finishes"
was shown zero times, while the same comments on the direct stream (tools on) showed it.
The worker now relays them as chunks carrying the frontend's own `_admissionStatus`.
"""

from __future__ import annotations

from core.inference.chat_generation_runs import _admission_status_chunks


def test_the_two_comments_become_status_chunks():
    assert _admission_status_chunks(": preempt-paused\n\n") == [{"_admissionStatus": "paused"}]
    assert _admission_status_chunks(": preempt-resumed\n\n") == [{"_admissionStatus": "resumed"}]


def test_other_comments_and_data_are_ignored():
    assert _admission_status_chunks(": keep-alive\n\n") == []
    assert _admission_status_chunks(": admission-wait\n\n") == []
    assert _admission_status_chunks('data: {"choices": []}\n\n') == []
    assert _admission_status_chunks("") == []


def test_a_piece_carrying_both_keeps_their_order():
    text = ": preempt-paused\n\n: preempt-resumed\r\n\r\n"
    assert [c["_admissionStatus"] for c in _admission_status_chunks(text)] == ["paused", "resumed"]


def test_a_comment_glued_to_data_is_still_seen():
    text = 'data: {"choices":[{"delta":{"content":"x"}}]}\n\n: preempt-paused\n\n'
    assert _admission_status_chunks(text) == [{"_admissionStatus": "paused"}]
