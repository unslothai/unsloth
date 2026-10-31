# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""join_when_started absorbs only the not-started-yet window; a real hang must still report alive."""

from __future__ import annotations

import threading
import time

import pytest

from .thread_drain import join_when_started


def test_join_raises_on_its_own_for_a_thread_that_never_started():
    """The premise. Without this the first test below could pass for the wrong reason."""
    never = threading.Thread(target = lambda: None)
    with pytest.raises(RuntimeError, match = "before it is started"):
        never.join(timeout = 0.01)


def test_an_unstarted_thread_does_not_raise_and_is_not_called_drained():
    """Unstarted and finished threads both read is_alive False, so the answer must come from join()."""
    never = threading.Thread(target = lambda: None)
    started = time.monotonic()
    assert join_when_started(never, timeout = 0.05) is False
    # It gives up at the deadline rather than waiting for a thread that will never run.
    assert time.monotonic() - started < 5


def test_a_thread_that_runs_is_waited_for():
    done = threading.Event()

    def work():
        time.sleep(0.05)
        done.set()

    worker = threading.Thread(target = work)
    worker.start()
    assert join_when_started(worker, timeout = 5) is True
    assert done.is_set(), "it returned True while the work had not finished"
    assert not worker.is_alive()


def test_a_thread_still_running_at_the_deadline_is_reported():
    """The half that must NOT be absorbed: callers assert on this to catch a leaked worker."""
    release = threading.Event()
    worker = threading.Thread(target = lambda: release.wait(timeout = 30))
    worker.start()
    try:
        assert join_when_started(worker, timeout = 0.05) is False
    finally:
        release.set()
        worker.join(timeout = 5)


def test_joining_the_current_thread_is_refused_rather_than_retried():
    """Joining the current thread is not a window that closes, so it must raise rather than retry."""
    with pytest.raises(RuntimeError, match = "current thread"):
        join_when_started(threading.current_thread(), timeout = 30)
