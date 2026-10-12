# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A durable chat run tells its follower about a pause, and about waiting in the queue."""

from __future__ import annotations

from core.inference.chat_generation_runs import _admission_status_chunks
import pytest

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def test_the_two_comments_become_status_chunks():
    assert _admission_status_chunks(": preempt-paused\n\n") == [{"_admissionStatus": "paused"}]
    assert _admission_status_chunks(": preempt-resumed\n\n") == [{"_admissionStatus": "resumed"}]


def test_other_comments_and_data_are_ignored():
    assert _admission_status_chunks(": keep-alive\n\n") == []
    assert _admission_status_chunks(": tool-heartbeat\n\n") == []
    assert _admission_status_chunks('data: {"choices": []}\n\n') == []
    assert _admission_status_chunks("") == []


def test_a_piece_carrying_both_keeps_their_order():
    text = ": preempt-paused\n\n: preempt-resumed\r\n\r\n"
    assert [c["_admissionStatus"] for c in _admission_status_chunks(text)] == ["paused", "resumed"]


def test_the_queue_comments_become_status_chunks():
    # The legacy stream shows "Waiting for a free slot" off these; a durable follower only
    # sees what the supervisor writes as chunks.
    assert _admission_status_chunks(": admission-wait\n\n") == [{"_admissionStatus": "waiting"}]
    assert _admission_status_chunks(": admission-done\n\n") == [{"_admissionStatus": "admitted"}]


def test_a_repeated_wait_is_one_event():
    last: list = []
    relayed = []
    for piece in (": admission-wait\n\n",) * 4 + (": admission-done\n\n", ": admission-wait\n\n"):
        relayed += [c["_admissionStatus"] for c in _admission_status_chunks(piece, last)]
    # Every keepalive repeats the wait; a later requeue is a change and is relayed again.
    assert relayed == ["waiting", "admitted", "waiting"]


def test_a_comment_glued_to_data_is_still_seen():
    text = 'data: {"choices":[{"delta":{"content":"x"}}]}\n\n: preempt-paused\n\n'
    assert _admission_status_chunks(text) == [{"_admissionStatus": "paused"}]
