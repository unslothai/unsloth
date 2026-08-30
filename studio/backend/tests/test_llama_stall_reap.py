# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the stall reaper wired to the engine-stats watchdog.

A wedged llama-server slot must cancel the generation that is holding it, and
only that one: the /metrics counters are engine-wide, so a freeze proves nothing
is decoding, but Studio can also hold generations merely queued behind the wedge.
"""

import threading

import pytest

from state import active_generations


@pytest.fixture(autouse = True)
def _clean_registry():
    active_generations.reset_for_tests()
    yield
    active_generations.reset_for_tests()


def _reap(**kw):
    from core.inference.llama_cpp import _reap_stalled_generation

    _reap_stalled_generation(
        running = kw.get("running", 1),
        waiting = kw.get("waiting", 0),
        stalled_s = kw.get("stalled_s", 200.0),
    )


def test_reaps_the_oldest_generation_only():
    first, second = threading.Event(), threading.Event()
    with active_generations.ActiveGeneration(first, thread_id = "t1", run_id = "r1"):
        with active_generations.ActiveGeneration(second, thread_id = "t2", run_id = "r2"):
            _reap()
            assert first.is_set(), "the oldest generation holds the wedged slot"
            assert not second.is_set(), "a generation queued behind it is innocent"


def test_no_active_generation_is_reported_not_reaped():
    # Nothing registered: the held slot is llama-server's to explain. The
    # important property is that this does not raise and does not kill anything.
    _reap()


def test_first_turn_without_ids_is_still_cancellable():
    # A first turn racing persistence has neither thread_id nor run_id. It is the
    # only thing in flight, so cancelling everything cancels exactly it.
    ev = threading.Event()
    with active_generations.ActiveGeneration(ev):
        _reap()
        assert ev.is_set()


def test_reap_runs_from_the_poller_thread_without_raising():
    # The hook is invoked from the stats daemon thread, not the request loop.
    ev = threading.Event()
    errors = []

    def _call():
        try:
            _reap()
        except Exception as exc:  # noqa: BLE001 - the point is to surface it
            errors.append(exc)

    with active_generations.ActiveGeneration(ev, thread_id = "t1", run_id = "r1"):
        t = threading.Thread(target = _call)
        t.start()
        t.join(timeout = 5)
        assert not t.is_alive()
        assert not errors, f"reap raised off the request loop: {errors}"
        assert ev.is_set()
