# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the parent writes about a stopped request, and what the worker reads."""

import multiprocessing as mp
import time
import uuid
from types import SimpleNamespace

import pytest

from core.inference import worker
from core.inference.stop_ledger import _ID_BYTES, _SLOTS, PendingTeardowns, StopLedger

_CTX = mp.get_context("spawn")


def _ids(n):
    return [str(uuid.uuid4()) for _ in range(n)]


def _ledger():
    return StopLedger(_CTX)


def _stopped(ledger):
    """What the worker would read. It is the only way the record is read, so it is the"""
    return ledger.snapshot()[1]


def test_a_request_reads_as_stopped_once_it_is():
    ledger = _ledger()
    mine, theirs = _ids(2)

    assert _stopped(ledger) == set()
    assert ledger.stop(mine)
    assert _stopped(ledger) == {mine}, "a stop names one request and no other"
    assert theirs not in _stopped(ledger)


def test_the_oldest_stop_is_the_one_that_ages_out():
    ledger = _ledger()
    recorded = _ids(_SLOTS + 1)
    for request_id in recorded:
        assert ledger.stop(request_id)

    assert _stopped(ledger) == set(recorded[1:]), "the oldest made way for the newest"


def test_stopping_the_same_request_twice_spends_one_slot():
    ledger = _ledger()
    theirs, mine = _ids(2)
    assert ledger.stop(theirs)
    for _ in range(_SLOTS):
        assert ledger.stop(mine)

    assert _stopped(ledger) == {mine, theirs}, "the one stopped first is still stopped"


def test_a_snapshot_answers_for_every_reply_at_once():
    ledger = _ledger()
    mine, absent = _ids(2)
    theirs = "short"
    ledger.stop(mine)
    ledger.stop(theirs)

    written, stopped = ledger.snapshot()
    assert written == 2
    assert stopped == {mine, theirs}, "the ids as recorded, with nothing padded onto them"
    assert absent not in stopped

