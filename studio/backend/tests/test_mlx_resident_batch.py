# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Replies joining and leaving a batch the MLX worker keeps open."""

import queue as _queue
import threading
from types import SimpleNamespace

import pytest

from core.inference import worker


class _RespQueue:
    def __init__(self):
        self.sent = []

    def put(self, item, *a, **k):
        self.sent.append(item)


class _Session:
    """A batch that hands back whatever the test scripted for each reply."""

    def __init__(self, *, width, record_stats):
        self.width = width
        self.record_stats = record_stats
        self.rows = []
        self.closed = False
        self.admitted = []
        self.withdrawn = []
        self.prefix = ""
        self.script = {}
        self.stats_at_retire = {}
        self.refuse_from = None

    @property
    def rows_in_flight(self):
        return len(self.rows)

    def admit(self, request, handle):
        if self.refuse_from is not None and len(self.admitted) >= self.refuse_from:
            raise RuntimeError("no room")
        self.admitted.append((handle, request))
        self.rows.append(handle)
        return self.prefix

    def step(self):
        for handle in list(self.rows):
            events = self.script.get(handle) or []
            if not events:
                continue
            event = events.pop(0)
            if event is None:
                self.rows.remove(handle)
                # Closing a reply out records what it managed, as the real session
                # does, before the terminal event its caller sees.
                self.record_stats(handle, self.stats_at_retire.get(handle, {}))
            yield handle, event

    def withdraw(self, handles):
        # Only rows it is actually holding, as the real session does: a rollback
        # names every reply the request asked for, admitted or not.
        held = [handle for handle in handles if handle in self.rows]
        self.withdrawn.append(held)
        for handle in held:
            self.rows.remove(handle)
            # Taking a reply back closes it out, and closing it out records what it
            # managed -- as the real session does, on the way to its terminal event.
            self.record_stats(handle, self.stats_at_retire.get(handle, {}))
            yield handle, None

    def close(self):
        self.closed = True


class _Backend:
    def __init__(
        self,
        reason = None,
        script = None,
    ):
        self.reason = reason
        self.script = script or {}
        self.sessions = []
        self.on_open = None

    def resident_unavailable_reason(self, request):
        return self.reason

    def open_resident_text_batch(
        self,
        *,
        width,
        record_stats,
        adapter_state = None,
    ):
        session = _Session(width = width, record_stats = record_stats)
        session.script.update(self.script)
        if self.on_open is not None:
            self.on_open(session)
        self.sessions.append(session)
        return session


def _cmd(request_id = "r1", **extra):
    return {"type": "generate", "request_id": request_id, "messages": [], **extra}


def _batch(backend = None):
    resp = _RespQueue()
    return worker._ResidentBatch(backend or _Backend(), resp), resp




def test_replies_a_command_asked_for_together_are_addressed_by_row():
    batch, resp = _batch()
    batch.admit(_cmd(rows = [{"seed": 1}, {"seed": 2}]), None)
    batch.session.script[("r1", 0)] = ["a", None]
    batch.session.script[("r1", 1)] = ["b", None]
    for _ in range(2):
        batch.step()

    rows = [(m["type"], m.get("row")) for m in resp.sent]
    assert ("token", 0) in rows and ("token", 1) in rows
    assert rows.count(("row_done", 0)) == 1 and rows.count(("row_done", 1)) == 1
    assert [m["type"] for m in resp.sent][-1] == "gen_done"


def test_fanned_replies_report_their_stats_per_row_and_not_again_at_the_end():
    batch, resp = _batch()
    batch.admit(_cmd(rows = [{}, {}]), None)
    batch.session.stats_at_retire[("r1", 0)] = {"n": 1}
    batch.session.stats_at_retire[("r1", 1)] = {"n": 2}
    batch.session.script[("r1", 0)] = [None]
    batch.session.script[("r1", 1)] = [None]
    batch.step()

    done = [m for m in resp.sent if m["type"] == "row_done"]
    assert sorted(m["stats"]["n"] for m in done) == [1, 2]
    assert resp.sent[-1]["stats"] is None




def test_the_batch_opens_at_the_width_its_first_command_asked_for():
    batch, _resp = _batch()
    batch.admit(_cmd(parallel_slots = 6), None)
    assert batch.session.width == 6


class _Idle:
    """A stretch of quiet, measured in reads rather than seconds."""

    def __init__(self, reads = 12):
        self.reads = reads


IDLE = _Idle()

