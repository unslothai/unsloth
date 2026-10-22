# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Removing a failed instrument from the list being iterated makes Python silently skip the next."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.runtime.session import Session  # noqa: E402


class _Recorder:
    def now_ms(self):
        return 0.0

    def emit(self, row):
        pass


class _Ctx:
    def __init__(self):
        self.recorder = _Recorder()
        self.logged = []

    def log(self, msg):
        self.logged.append(msg)


class _Instrument:
    """Records which lifecycle hooks it was actually given."""

    def __init__(
        self,
        name,
        *,
        raises = False,
    ):
        self.name = name
        self.raises = raises
        self.started = []

    def start_cell(self, cell):
        if self.raises:
            raise RuntimeError("instrument exploded")
        self.started.append(cell)

    def end_cell(self, cell):
        if self.raises:
            raise RuntimeError("instrument exploded")
        return {"ok": True}


def _session(instruments):
    s = Session.__new__(Session)
    s.ctx = _Ctx()
    s.instruments = list(instruments)
    return s


def test_an_instrument_after_a_failing_one_still_gets_its_hook():
    frames, heap, keys, rss = (
        _Instrument("frames"),
        _Instrument("heap", raises = True),
        _Instrument("input"),
        _Instrument("rss"),
    )
    s = _session([frames, heap, keys, rss])

    s.each_instrument("start_cell", "c1")

    assert keys.started == ["c1"], "the instrument behind the failing one was skipped"
    assert frames.started == ["c1"]
    assert rss.started == ["c1"]


def test_the_failing_instrument_is_still_dropped():
    """The control: the fix must not cost `_safe` its whole reason for existing."""
    heap, keys = _Instrument("heap", raises = True), _Instrument("input")
    s = _session([heap, keys])

    s.each_instrument("start_cell", "c1")

    assert [i.name for i in s.instruments] == ["input"]
    assert any("heap.start_cell failed" in m for m in s.ctx.logged)


def test_two_failures_in_a_row_drop_both_and_skip_neither():
    a, b, keys = (
        _Instrument("heap", raises = True),
        _Instrument("tracing", raises = True),
        _Instrument("input"),
    )
    s = _session([a, b, keys])

    s.each_instrument("start_cell", "c1")

    assert keys.started == ["c1"]
    assert [i.name for i in s.instruments] == ["input"]
