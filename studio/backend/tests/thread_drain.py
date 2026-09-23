# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Join a thread found through ``threading.enumerate()`` without racing its start.

``Thread.join()`` raises ``RuntimeError: cannot join thread before it is started`` when the
thread has not reached the point where CPython considers it running. ``threading.enumerate()``
returns threads in exactly that state: ``start()`` puts a thread in ``_limbo`` and it stays
enumerable-but-unjoinable until the interpreter schedules it. The window is short, which is why
a test that drains workers this way passes for months and then fails once on a loaded runner:

    tests/test_diffusion_backend.py:2509: in test_begin_load_rejects_concurrent
        thread.join(timeout = 5)
    RuntimeError: cannot join thread before it is started

Filtering does not close the window. ``Thread.name`` is set at construction, so a name match says
nothing about whether the thread has started. ``Thread.ident`` is assigned in ``_bootstrap_inner``
BEFORE the ``_started`` event that ``join()`` waits on, so an ident check can pass and the very
next line still raise. ``is_alive()`` is False both before a thread starts and after it finishes,
so skipping on it silently drops the worker the drain existed to wait for.

What is reliable is the exception itself: it means "not started YET", and a thread that has been
started always gets there. So retry until the thread joins or the caller's deadline passes.

Where a test owns the worker outright, recording it from inside the thread is better than
enumerating at all, since a thread running its own target is past the window by definition. This
helper is for the drains that legitimately cannot name their thread in advance.
"""

from __future__ import annotations

import threading
import time


def join_when_started(thread: threading.Thread, timeout: float = 5.0) -> bool:
    """Join ``thread``, waiting out the not-started-yet window. True if it finished.

    Never raises for an unstarted thread: that is the condition this exists to absorb. It does
    not report one as drained either. A thread still in `_limbo` at the deadline has not run
    yet and still will, so True is reserved for a thread that actually finished.
    """
    # join() raises RuntimeError for two unrelated reasons, and only one of them is a window
    # that closes. Joining yourself never becomes possible, so retrying it would burn the whole
    # timeout and then report a thread that is trivially still alive.
    if thread is threading.current_thread():
        raise RuntimeError("cannot join current thread")
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        try:
            thread.join(timeout=max(remaining, 0.0))
        except RuntimeError:
            if remaining <= 0:
                # Still not started at the deadline, so it is UNDRAINED, and is_alive() must
                # not be consulted here: it is False before a thread starts exactly as it is
                # after one finishes, so asking it would report a worker that is still
                # scheduled and will run later as one that is already done. Callers restore
                # their monkeypatches on the strength of this answer, which is how that
                # worker would end up running against the next test.
                return False
            time.sleep(0.005)
            continue
        return not thread.is_alive()
