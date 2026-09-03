# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``_session_in_flight``'s state machine when the guarded body raises.

The guard used to ``return`` from its ``finally``, discarding whatever the tool
raised. Dropping that return is only safe if the cleanup still runs as before:
the refcount, the queued deletes, and the condition other calls wait on. Every
case asserts the full invariant set afterwards, so a leak fails here instead of
hanging some later test. Barriers and bounded joins, no sleep-based timing.
"""

from __future__ import annotations

import faulthandler
import random
import sys
import threading
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import tools  # noqa: E402

DEADLINE = 30.0


def _cleanup_explodes(session_id, delete_files):
    """A cleanup that fails.

    A plain function, not a generator ``.throw()`` one-liner: that would reset
    ``__context__`` and hide the thing one of these tests checks.
    """
    raise OSError("disk gone")


@pytest.fixture(autouse = True)
def clean_lifecycle_state():
    """Start and finish with the module maps empty, whatever the test did."""
    tools._active_sessions.clear()
    tools._pending_removals.clear()
    tools._removing_sessions.clear()
    yield
    tools._active_sessions.clear()
    tools._pending_removals.clear()
    tools._removing_sessions.clear()


def assert_idle(message = ""):
    assert dict(tools._active_sessions) == {}, f"_active_sessions leaked {message}"
    assert dict(tools._pending_removals) == {}, f"_pending_removals leaked {message}"
    assert set(tools._removing_sessions) == set(), f"_removing_sessions leaked {message}"


@pytest.fixture
def removals(monkeypatch):
    """Record every sandbox removal instead of touching the filesystem."""
    seen: list[str] = []

    def _remove(session_id, delete_files):
        seen.append(session_id)

    monkeypatch.setattr(tools, "_remove_session_sandbox_locked", _remove)
    monkeypatch.setattr(tools, "_thread_exists", lambda *a, **k: False)
    return seen


def queue_removal(session_id, *, files = True):
    key = tools._session_key(session_id)
    tools._pending_removals.setdefault(key, {})[session_id] = files


# ── The exception path, with and without a queued delete ──────────


def test_an_exception_leaves_no_lifecycle_state(removals):
    sentinel = ValueError("boom")
    with pytest.raises(ValueError) as caught:
        with tools._session_in_flight("plain"):
            raise sentinel
    assert caught.value is sentinel
    assert removals == []
    assert_idle("after a plain failure")


def test_a_queued_delete_still_runs_once_when_the_body_raises(removals):
    """The cleanup is the whole reason the finally exists. It must still fire."""
    queue_removal("doomed")
    with pytest.raises(ValueError):
        with tools._session_in_flight("doomed"):
            raise ValueError("boom")
    assert removals == ["doomed"], removals
    assert_idle("after a failure with a queued delete")


def test_a_recreated_chat_keeps_its_folder_even_when_the_body_raises(monkeypatch):
    """A chat recreated during the call owns the directory; skip the delete."""
    seen: list[str] = []
    monkeypatch.setattr(
        tools,
        "_remove_session_sandbox_locked",
        lambda s, f: seen.append(s),
    )
    monkeypatch.setattr(tools, "_thread_exists", lambda *a, **k: True)
    queue_removal("recreated")
    with pytest.raises(ValueError):
        with tools._session_in_flight("recreated"):
            raise ValueError("boom")
    assert seen == []
    assert_idle("after a skipped delete")


# ── Nesting and case folding ──────────────────────────────────────


def test_a_nested_guard_deletes_only_at_the_outer_exit(removals):
    queue_removal("nested")
    key = tools._session_key("nested")
    with tools._session_in_flight("nested"):
        with tools._session_in_flight("nested"):
            assert tools._active_sessions[key] == 2
        assert removals == [], "the inner exit deleted a sandbox still in use"
        assert tools._active_sessions[key] == 1
    assert removals == ["nested"]
    assert_idle("after a nested guard")


def test_a_nested_guard_deletes_once_even_when_the_inner_body_raises(removals):
    queue_removal("nested-raise")
    with pytest.raises(ValueError):
        with tools._session_in_flight("nested-raise"):
            with tools._session_in_flight("nested-raise"):
                raise ValueError("boom")
    assert removals == ["nested-raise"], removals
    assert_idle("after a nested failure")


def test_case_variant_ids_share_one_lifecycle_key(removals):
    """One directory on Windows and on a default macOS volume."""
    key = tools._session_key("CasePair")
    assert key == tools._session_key("casepair")
    with pytest.raises(ValueError):
        with tools._session_in_flight("CasePair"):
            with tools._session_in_flight("casepair"):
                assert tools._active_sessions[key] == 2
                raise ValueError("boom")
    assert_idle("after a case-variant failure")


# ── Cleanup that itself fails ─────────────────────────────────────


def test_a_failing_cleanup_still_releases_the_session(monkeypatch):
    """A cleanup error must never strand the chat.

    A key left in ``_removing_sessions`` blocks every later call for it forever.
    """
    monkeypatch.setattr(tools, "_remove_session_sandbox_locked", _cleanup_explodes)
    monkeypatch.setattr(tools, "_thread_exists", lambda *a, **k: False)
    queue_removal("cleanup-fails")
    with pytest.raises(OSError):
        with tools._session_in_flight("cleanup-fails"):
            pass
    assert_idle("after a failing cleanup")


def test_a_failing_cleanup_masks_the_tool_error_but_keeps_it_as_context(monkeypatch):
    """Pins the policy rather than asserting a preference.

    Standard semantics: a ``finally`` exception replaces the one in flight and
    keeps it as ``__context__``. Predates this change (it already happened
    whenever a delete was queued), so it is pinned, not altered.
    """
    monkeypatch.setattr(tools, "_remove_session_sandbox_locked", _cleanup_explodes)
    monkeypatch.setattr(tools, "_thread_exists", lambda *a, **k: False)
    queue_removal("masked")
    tool_error = ValueError("the real tool failure")
    with pytest.raises(OSError) as caught:
        with tools._session_in_flight("masked"):
            raise tool_error
    assert caught.value.__context__ is tool_error
    assert_idle("after a masked tool error")


def test_a_failing_cleanup_wakes_a_waiter_for_the_same_chat(monkeypatch):
    """The condition variable must be notified on the error path too."""
    entered = threading.Event()
    release = threading.Event()

    def _slow_failing_remove(session_id, delete_files):
        entered.set()
        release.wait(DEADLINE)
        raise OSError("disk gone")

    monkeypatch.setattr(tools, "_remove_session_sandbox_locked", _slow_failing_remove)
    monkeypatch.setattr(tools, "_thread_exists", lambda *a, **k: False)
    queue_removal("waited-on")

    def _first():
        try:
            with tools._session_in_flight("waited-on"):
                pass
        except OSError:
            pass

    first = threading.Thread(target = _first, name = "pr9640-first")
    first.start()
    assert entered.wait(DEADLINE), "cleanup never started"

    waiter_in = threading.Event()

    def _second():
        with tools._session_in_flight("waited-on"):
            waiter_in.set()

    second = threading.Thread(target = _second, name = "pr9640-waiter")
    second.start()
    # The waiter must be blocked while the removal is in progress.
    assert not waiter_in.wait(0.5), "a call started inside a folder being deleted"

    release.set()
    assert waiter_in.wait(DEADLINE), "the waiter was never woken after a failed cleanup"
    first.join(DEADLINE)
    second.join(DEADLINE)
    assert not first.is_alive() and not second.is_alive()
    assert_idle("after a woken waiter")


def test_one_failing_delete_does_not_silently_drop_the_others(monkeypatch):
    """Pre-existing behaviour, pinned so a fix is a deliberate choice.

    ``_pending_removals.pop`` takes the whole batch before iterating, so one
    raising entry leaves the rest neither attempted nor queued. Not introduced
    here, but the exception path makes it easy to hit.
    """
    attempted: list[str] = []

    def _remove(session_id, delete_files):
        attempted.append(session_id)
        if session_id == "b":
            raise OSError("disk gone")

    monkeypatch.setattr(tools, "_remove_session_sandbox_locked", _remove)
    monkeypatch.setattr(tools, "_thread_exists", lambda *a, **k: False)
    key = tools._session_key("a")
    # Same lifecycle key, three exact ids queued behind it.
    tools._pending_removals[key] = {"a": True, "b": True, "c": True}
    with pytest.raises(OSError):
        with tools._session_in_flight("a"):
            pass
    assert "b" in attempted
    assert attempted != [
        "a",
        "b",
        "c",
    ], "the batch now completes past a failure -- update this test and say so"
    assert_idle("after a partially failed batch")


# ── Concurrency ───────────────────────────────────────────────────


@pytest.mark.parametrize("seed", list(range(100)))
def test_randomised_schedules_leave_no_lifecycle_state(seed, removals):
    """100 seeds mixing successes, failures, deletes and case variants.

    Joins have a deadline and faulthandler is armed, so a deadlock leaves
    stacks rather than a silent CI timeout.
    """
    rng = random.Random(seed)
    ids = ["alpha", "Alpha", "beta", "BETA", "gamma"]
    errors: list[BaseException] = []
    start = threading.Barrier(8, timeout = DEADLINE)

    def worker(i):
        session = rng.choice(ids)
        fail = rng.random() < 0.5
        cancel = rng.random() < 0.2
        if rng.random() < 0.3:
            queue_removal(session)
        try:
            start.wait()
            with tools._session_in_flight(session):
                if cancel:
                    raise KeyboardInterrupt("stop")
                if fail:
                    raise ValueError(f"boom-{i}")
        except (ValueError, KeyboardInterrupt):
            pass
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    faulthandler.dump_traceback_later(DEADLINE, exit = False)
    try:
        threads = [
            threading.Thread(target = worker, args = (i,), name = f"pr9640-{seed}-{i}") for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(DEADLINE)
        assert not any(t.is_alive() for t in threads), f"deadlock at seed {seed}"
    finally:
        faulthandler.cancel_dump_traceback_later()

    assert errors == [], f"unexpected exception at seed {seed}: {errors}"
    assert_idle(f"at seed {seed}")
