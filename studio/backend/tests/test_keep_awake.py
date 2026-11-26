# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Windows keep-awake during generation and training jobs (utils/keep_awake.py)."""

from __future__ import annotations

import threading

import pytest

from utils import keep_awake
from utils.keep_awake import (
    ES_CONTINUOUS,
    ES_SYSTEM_REQUIRED,
    keep_system_awake,
    keeps_system_awake,
)

HOLD = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
RELEASE = ES_CONTINUOUS


class _FakeKernel32:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def SetThreadExecutionState(self, flags):  # noqa: N802 - the Win32 name
        self.calls.append((flags, threading.current_thread().name))
        return 1

    @property
    def flags(self) -> list[int]:
        return [f for f, _ in self.calls]


@pytest.fixture
def kernel32(monkeypatch):
    fake = _FakeKernel32()
    monkeypatch.setattr(keep_awake.sys, "platform", "win32")
    monkeypatch.setattr(keep_awake, "_kernel32", lambda: fake)
    assert keep_awake.active_holders() == 0
    yield fake
    assert keep_awake.active_holders() == 0


def test_set_on_start_and_released_on_finish(kernel32):
    with keep_system_awake():
        assert kernel32.flags == [HOLD]
    assert kernel32.flags == [HOLD, RELEASE]
    # both calls come from the one holder thread: the state is per thread
    assert {name for _, name in kernel32.calls} == {"studio-keep-awake"}


def test_released_on_exception(kernel32):
    @keeps_system_awake
    def job():
        assert kernel32.flags == [HOLD]
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match = "boom"):
        job()
    assert kernel32.flags == [HOLD, RELEASE]


def test_released_on_cancel(kernel32):
    cancel = threading.Event()
    started = threading.Event()

    @keeps_system_awake
    def job():
        started.set()
        cancel.wait(5)
        return "cancelled"

    worker = threading.Thread(target = job)
    worker.start()
    assert started.wait(5)
    assert kernel32.flags == [HOLD]
    cancel.set()
    worker.join(5)
    assert kernel32.flags == [HOLD, RELEASE]


def test_overlapping_jobs_are_refcounted(kernel32):
    first_in, second_in = threading.Event(), threading.Event()
    first_out, second_out = threading.Event(), threading.Event()

    def run(entered, leave):
        with keep_system_awake():
            entered.set()
            leave.wait(5)

    a = threading.Thread(target = run, args = (first_in, first_out))
    b = threading.Thread(target = run, args = (second_in, second_out))
    a.start()
    assert first_in.wait(5)
    b.start()
    assert second_in.wait(5)
    assert kernel32.flags == [HOLD]
    assert keep_awake.active_holders() == 2
    first_out.set()
    a.join(5)
    assert kernel32.flags == [HOLD]
    second_out.set()
    b.join(5)
    assert kernel32.flags == [HOLD, RELEASE]
    with keep_system_awake():
        pass
    assert kernel32.flags == [HOLD, RELEASE, HOLD, RELEASE]


def test_nested_holds_are_refcounted(kernel32):
    with keep_system_awake():
        with keep_system_awake():
            pass
        assert kernel32.flags == [HOLD]
    assert kernel32.flags == [HOLD, RELEASE]


def test_failing_kernel32_never_fails_the_job(kernel32, monkeypatch):
    def broken(flags):
        raise OSError("no power api")

    monkeypatch.setattr(kernel32, "SetThreadExecutionState", broken)
    with keep_system_awake():
        result = "ran"
    assert result == "ran"


def test_no_op_off_windows(monkeypatch):
    import ctypes

    monkeypatch.setattr(keep_awake.sys, "platform", "linux")
    monkeypatch.setattr(
        ctypes, "WinDLL", lambda *a, **k: pytest.fail("kernel32 loaded"), raising = False
    )
    assert keep_awake._kernel32() is None
    before = threading.active_count()
    with keep_system_awake():
        assert keep_awake.active_holders() == 1
        assert threading.active_count() == before
    assert keep_awake.active_holders() == 0


def test_job_entry_points_hold_the_system_awake():
    from core.inference.diffusion import DiffusionBackend
    from core.inference.video import VideoBackend
    from core.training import diffusion_training_service, worker

    for fn in (
        DiffusionBackend.generate,
        VideoBackend.generate,
        worker.run_training_process,
        diffusion_training_service._run_diffusion_child,
    ):
        assert _held(fn), fn


def _held(fn):
    """``fn`` or a function it wraps (``functools.wraps`` chains) is a ``keeps_system_awake`` wrapper."""
    while fn is not None:
        if getattr(fn, "keeps_system_awake", False):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False
