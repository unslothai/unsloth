# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the durable chat generation progress lease.

A durable run sets cancel_on_disconnect=False on purpose, so a closed browser cannot kill
a long generation. Nothing else bounded it: a producer that stopped producing left the run
active forever, which kept the thread "generating", unmounted Send in the UI, and held the
engine slot. The repair, reconcile_orphaned_runs, only ran at process boot, so reloading
the page never cleared it.

These cover the two halves of the fix: streamed output renews a lease on the run row, and a
periodic sweep settles only runs whose lease has expired -- never a slow but advancing one.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import chat_generation_runs as runs_mod  # noqa: E402
from core.inference.chat_generation_runs import ChatGenerationLeaseSweeper  # noqa: E402
from storage import chat_generation_runs_db as runs_db  # noqa: E402
from storage import studio_db  # noqa: E402

_MINUTE_MS = 60_000


class _Capture:
    """Fake structlog logger; the sweeper only ever warns."""

    def __init__(self):
        self.events = []

    def warning(self, event, **kw):
        self.events.append((event, dict(kw)))

    def info(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass

    def error(self, *a, **k):
        pass

    def names(self):
        return [event for event, _kw in self.events]


class _Clock:
    """Fake wall clock for the lease, which is stamped in milliseconds."""

    def __init__(self, start = 1_700_000_000_000):
        self.now = int(start)

    def __call__(self):
        return self.now

    def advance_ms(self, ms):
        self.now += int(ms)


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(runs_db, "now_ms", fake)
    return fake


@pytest.fixture
def capture(monkeypatch):
    cap = _Capture()
    monkeypatch.setattr(runs_mod, "logger", cap)
    return cap


def _seed_run(run_id = "run-1", thread_id = "thread-1"):
    studio_db.upsert_chat_thread(
        {
            "id": thread_id,
            "title": "Chat",
            "modelType": "base",
            "modelId": "local.gguf",
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": f"user-{run_id}",
            "threadId": thread_id,
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
            "createdAt": 2,
        }
    )
    run, _created = runs_db.create_run(
        run_id = run_id,
        owner_subject = "alice",
        thread_id = thread_id,
        user_message_id = f"user-{run_id}",
        assistant_message_id = f"assistant-{run_id}",
        request_payload = {
            "model": "local.gguf",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    return run


def _running_run(run_id = "run-1", thread_id = "thread-1"):
    _seed_run(run_id, thread_id)
    token = runs_db.get_worker_token(run_id)
    assert runs_db.mark_running(run_id, token)
    return token


def _stream(
    run_id,
    token,
    count = 2,
):
    return runs_db.append_events(
        run_id,
        token,
        [("chunk", {"choices": [{"delta": {"content": "hi"}}]}) for _ in range(count)],
    )


def _write_partial_text(message_id, text):
    """Stand in for the client checkpoint that persists partial assistant output."""
    conn = runs_db._connect()
    try:
        conn.execute(
            "UPDATE chat_messages SET content_json=? WHERE id=?",
            (json.dumps([{"type": "text", "text": text}]), message_id),
        )
        conn.commit()
    finally:
        conn.close()


def _sweeper(
    app = None,
    *,
    interval_s = 60.0,
    timeout_s = 600.0,
):
    return ChatGenerationLeaseSweeper(
        app if app is not None else SimpleNamespace(state = SimpleNamespace()),
        interval_s = interval_s,
        timeout_s = timeout_s,
    )


# Lease writes


def test_streamed_chunks_renew_the_lease(clock):
    token = _running_run()
    started_at, started_tokens = runs_db.get_progress("run-1")
    assert started_tokens == 0

    clock.advance_ms(5 * _MINUTE_MS)
    _stream("run-1", token, count = 3)
    progress_at, tokens = runs_db.get_progress("run-1")
    assert progress_at == started_at + 5 * _MINUTE_MS
    assert tokens == 3

    clock.advance_ms(_MINUTE_MS)
    _stream("run-1", token, count = 2)
    assert runs_db.get_progress("run-1") == (progress_at + _MINUTE_MS, 5)


def test_lease_timestamp_never_moves_backwards(clock):
    token = _running_run()
    clock.advance_ms(10 * _MINUTE_MS)
    _stream("run-1", token, count = 1)
    ahead, _tokens = runs_db.get_progress("run-1")

    # NTP step / resume from suspend: the wall clock goes backwards. A regressing
    # lease would age a live run straight into the sweep.
    clock.advance_ms(-9 * _MINUTE_MS)
    _stream("run-1", token, count = 1)
    assert runs_db.get_progress("run-1") == (ahead, 2)


def test_a_settled_run_stops_taking_lease_writes(clock):
    token = _running_run()
    _stream("run-1", token, count = 1)
    runs_db.finish_run("run-1", worker_token = token, status = "completed")
    before = runs_db.get_progress("run-1")
    clock.advance_ms(_MINUTE_MS)
    assert _stream("run-1", token, count = 4) == []
    assert runs_db.get_progress("run-1") == before


# The sweep


@pytest.mark.asyncio
async def test_sweep_reaps_a_run_that_stopped_producing(clock, capture):
    token = _running_run()
    _stream("run-1", token, count = 2)
    _write_partial_text("assistant-run-1", "partial answer")
    clock.advance_ms(11 * _MINUTE_MS)

    assert await _sweeper(timeout_s = 600.0).sweep_once() == ["run-1"]

    run = runs_db.get_run("run-1", "alice")
    assert (run["status"], run["finishReason"]) == ("failed", "interrupted")
    assert run["error"] == runs_mod._LEASE_ERROR
    message = studio_db.get_chat_message("thread-1", "assistant-run-1")
    assert message["metadata"]["generationStatus"] == "failed"
    assert message["metadata"]["incomplete"] == {"reason": "interrupted"}
    # Partial output survives the reap: the point is to release the UI, not to
    # discard what the model already produced.
    assert message["content"] == [{"type": "text", "text": "partial answer"}]
    assert [event["type"] for event in runs_db.list_events("run-1")].count("chunk") == 2
    assert capture.names() == ["chat_generation_run_lease_expired"]
    assert capture.events[0][1]["run_id"] == "run-1"


@pytest.mark.asyncio
async def test_sweep_spares_a_slow_but_progressing_run(clock):
    """5-6 tok/s on a spilled model is normal; it must never be reaped."""
    token = _running_run()
    sweeper = _sweeper(timeout_s = 600.0)
    for _tick in range(30):
        clock.advance_ms(_MINUTE_MS)
        _stream("run-1", token, count = 1)
        assert await sweeper.sweep_once() == []
    assert runs_db.get_run("run-1", "alice")["status"] == "running"


@pytest.mark.asyncio
async def test_sweep_reaps_a_run_wedged_before_its_first_token(clock):
    # No lease write has ever landed, so the age comes from started_at/created_at.
    _running_run()
    clock.advance_ms(11 * _MINUTE_MS)
    assert await _sweeper(timeout_s = 600.0).sweep_once() == ["run-1"]
    assert runs_db.get_run("run-1", "alice")["finishReason"] == "interrupted"


@pytest.mark.asyncio
async def test_sweep_settles_a_stopped_run_as_cancelled(clock):
    token = _running_run()
    _stream("run-1", token, count = 1)
    runs_db.request_cancel("run-1", "alice")
    assert runs_db.get_run("run-1", "alice")["status"] == "cancelling"

    clock.advance_ms(11 * _MINUTE_MS)
    assert await _sweeper(timeout_s = 600.0).sweep_once() == ["run-1"]

    run = runs_db.get_run("run-1", "alice")
    # A Stop the user already asked for must not be reported back as a failure.
    assert (run["status"], run["finishReason"], run["error"]) == ("cancelled", "cancelled", None)
    message = studio_db.get_chat_message("thread-1", "assistant-run-1")
    assert message["metadata"]["incomplete"] == {"reason": "cancelled"}


@pytest.mark.asyncio
async def test_sweep_cancels_the_wedged_producer(clock, capture):
    cancelled = []
    app = SimpleNamespace(
        state = SimpleNamespace(chat_generation_supervisor = SimpleNamespace(cancel = cancelled.append))
    )
    _running_run()
    clock.advance_ms(11 * _MINUTE_MS)
    assert await _sweeper(app, timeout_s = 600.0).sweep_once() == ["run-1"]
    # Settling the row is not enough: a producer parked in the engine still holds
    # its slot and activity reservation until it is cancelled.
    assert cancelled == ["run-1"]


@pytest.mark.asyncio
async def test_a_failing_producer_cancel_does_not_abort_the_sweep(clock, capture):
    def boom(_run_id):
        raise RuntimeError("no such run")

    app = SimpleNamespace(
        state = SimpleNamespace(chat_generation_supervisor = SimpleNamespace(cancel = boom))
    )
    _running_run()
    clock.advance_ms(11 * _MINUTE_MS)
    assert await _sweeper(app, timeout_s = 600.0).sweep_once() == ["run-1"]
    assert runs_db.get_run("run-1", "alice")["status"] == "failed"
    assert "chat_generation_lease_cancel_failed" in capture.names()


@pytest.mark.asyncio
async def test_a_zero_timeout_disables_the_sweep(clock):
    _running_run()
    clock.advance_ms(600 * _MINUTE_MS)
    sweeper = _sweeper(timeout_s = 0.0)
    assert sweeper.enabled is False
    assert await sweeper.sweep_once() == []
    sweeper.start()
    assert sweeper._task is None
    assert runs_db.get_run("run-1", "alice")["status"] == "running"


def test_lease_settings_come_from_the_environment(monkeypatch):
    app = SimpleNamespace(state = SimpleNamespace())
    monkeypatch.setenv("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", "60")
    monkeypatch.setenv("UNSLOTH_STUDIO_CHAT_RUN_LEASE_SWEEP_INTERVAL_S", "5")
    sweeper = ChatGenerationLeaseSweeper(app)
    assert (sweeper._timeout, sweeper._interval) == (60.0, 5.0)

    monkeypatch.setenv("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", "not-a-number")
    assert ChatGenerationLeaseSweeper(app)._timeout == runs_mod._LEASE_TIMEOUT_SECONDS


@pytest.mark.parametrize("raw", ["inf", "-inf", "nan", "Infinity", "NaN"])
def test_a_non_finite_timeout_falls_back_to_the_default(monkeypatch, raw):
    """float() accepts these, and each then breaks the sweep in its own quiet way.

    inf survives every comparison and only fails at int(timeout * 1000) inside the sweep,
    once per pass, forever. nan loses to nothing, so max(0.0, nan) returns 0.0 and the
    sweeper reports itself disabled. Both leave stuck runs unreaped, so neither may parse.
    """
    monkeypatch.setenv("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", raw)
    sweeper = ChatGenerationLeaseSweeper(SimpleNamespace(state = SimpleNamespace()))
    assert sweeper._timeout == runs_mod._LEASE_TIMEOUT_SECONDS
    assert sweeper.enabled is True


@pytest.mark.asyncio
async def test_an_infinite_timeout_still_reaps_a_stuck_run(monkeypatch, clock):
    """The end state the parser protects: int(inf * 1000) would raise on every sweep."""
    _running_run()
    clock.advance_ms(600 * _MINUTE_MS)
    monkeypatch.setenv("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", "inf")
    sweeper = ChatGenerationLeaseSweeper(SimpleNamespace(state = SimpleNamespace()))
    assert await sweeper.sweep_once() == ["run-1"]
    assert runs_db.get_run("run-1", "alice")["finishReason"] == "interrupted"


@pytest.mark.asyncio
async def test_a_nan_timeout_does_not_silently_disable_reaping(monkeypatch, clock):
    _running_run()
    clock.advance_ms(600 * _MINUTE_MS)
    monkeypatch.setenv("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", "nan")
    sweeper = ChatGenerationLeaseSweeper(SimpleNamespace(state = SimpleNamespace()))
    assert sweeper.enabled is True
    assert await sweeper.sweep_once() == ["run-1"]


@pytest.mark.parametrize("raw", ["1e308", "1e306", "1.5e308"])
def test_an_oversized_but_finite_timeout_is_clamped(monkeypatch, raw):
    """Finite is not the same as usable: the multiply to milliseconds overflows first."""
    monkeypatch.setenv("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", raw)
    sweeper = ChatGenerationLeaseSweeper(SimpleNamespace(state = SimpleNamespace()))
    assert sweeper._timeout == runs_mod._MAX_ENV_SECONDS
    # The conversion every consumer performs must survive the applied value.
    assert isinstance(int(sweeper._timeout * 1000), int)


@pytest.mark.asyncio
async def test_an_oversized_timeout_leaves_a_fresh_run_alone_without_raising(monkeypatch, clock):
    """The end state: the sweep completes instead of raising OverflowError every pass."""
    _running_run()
    monkeypatch.setenv("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", "1e308")
    sweeper = ChatGenerationLeaseSweeper(SimpleNamespace(state = SimpleNamespace()))
    assert await sweeper.sweep_once() == []
    assert runs_db.get_run("run-1", "alice")["status"] == "running"


def test_a_non_finite_interval_falls_back_to_the_default(monkeypatch):
    """An inf interval would park the sweep loop until shutdown; nan sleeps forever too."""
    monkeypatch.setenv("UNSLOTH_STUDIO_CHAT_RUN_LEASE_SWEEP_INTERVAL_S", "inf")
    sweeper = ChatGenerationLeaseSweeper(SimpleNamespace(state = SimpleNamespace()))
    assert sweeper._interval == runs_mod._LEASE_SWEEP_INTERVAL_SECONDS


# Boot reconciliation and shutdown


def test_boot_reconcile_still_settles_a_freshly_progressing_run(clock):
    """Unbounded at boot on purpose: every active row is orphaned by the restart."""
    token = _running_run()
    _stream("run-1", token, count = 5)
    assert runs_db.reconcile_orphaned_runs() == 1
    assert runs_db.get_run("run-1", "alice")["finishReason"] == "interrupted"


@pytest.mark.asyncio
async def test_sweeper_runs_on_its_interval_and_stops_promptly(clock):
    sweeps = []
    sweeper = _sweeper(interval_s = 1.0, timeout_s = 600.0)
    sweeper._interval = 0.01

    async def record():
        sweeps.append(1)
        return []

    sweeper.sweep_once = record
    sweeper.start()
    while len(sweeps) < 3:
        await asyncio.sleep(0.01)
    await sweeper.stop()
    settled = len(sweeps)
    await asyncio.sleep(0.05)
    assert len(sweeps) == settled
    assert sweeper._task is None


@pytest.mark.asyncio
async def test_stop_returns_when_a_sweep_will_not_finish(capture, monkeypatch):
    """Same contract as ChatGenerationSupervisor.stop(): bounded, then named."""
    monkeypatch.setattr(runs_mod, "_SHUTDOWN_GRACE_SECONDS", 0.05)
    monkeypatch.setattr(runs_mod, "_SHUTDOWN_CANCEL_SECONDS", 0.05)
    sweeper = _sweeper(interval_s = 1.0, timeout_s = 600.0)
    sweeper._interval = 0.01
    started = asyncio.Event()

    async def wedged():
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # An engine draining its subprocess inside aclose: the cancel lands but
            # unwinding still outlasts the budget. stop() must return regardless.
            await asyncio.sleep(0.3)

    sweeper.sweep_once = wedged
    sweeper.start()
    await asyncio.wait_for(started.wait(), timeout = 2)
    await asyncio.wait_for(sweeper.stop(), timeout = 5)
    assert any("shutdown budget" in name for name in capture.names())
    await asyncio.sleep(0.4)  # let the abandoned sweep unwind before the loop closes


@pytest.mark.asyncio
async def test_supervisor_stop_stops_the_sweeper():
    app = SimpleNamespace(state = SimpleNamespace())
    sweeper = _sweeper(app, interval_s = 1.0, timeout_s = 600.0)
    sweeper._interval = 0.01
    app.state.chat_generation_lease_sweeper = sweeper
    sweeper.start()
    await runs_mod.ChatGenerationSupervisor(app).stop()
    assert sweeper._task is None


def test_start_lease_sweeper_is_idempotent(monkeypatch):
    async def main():
        app = SimpleNamespace(state = SimpleNamespace())
        first = runs_mod.start_lease_sweeper(app)
        assert runs_mod.start_lease_sweeper(app) is first
        assert app.state.chat_generation_lease_sweeper is first
        await first.stop()

    asyncio.run(main())


_ABANDONED_SWEEP_PROGRAM = """
import asyncio, sys, threading
sys.path.insert(0, {backend!r})
from types import SimpleNamespace
from core.inference import chat_generation_runs as m
from storage import chat_generation_runs_db as db

parked = threading.Event()
def _never_returns(*a, **k):
    parked.set()
    threading.Event().wait(600)   # a sweep on SQLite's writer lock
db.reconcile_runs = _never_returns

async def main():
    sweeper = m.ChatGenerationLeaseSweeper(
        SimpleNamespace(state = SimpleNamespace()), interval_s = 0.01, timeout_s = 600.0
    )
    sweeper.start()
    for _ in range(500):
        if parked.is_set():
            break
        await asyncio.sleep(0.01)
    assert parked.is_set(), "the sweep never reached the database"
    await sweeper.stop()

asyncio.run(main())
"""


def test_an_abandoned_sweep_cannot_hold_the_process_open():
    """Asserts the PROCESS exits, not merely that stop() returned.

    asyncio.to_thread runs on non-daemon executor threads that an atexit hook joins, so a
    sweep parked on the writer lock kept the interpreter alive indefinitely after shutdown
    had given up on it. Studio Desktop allows five seconds for a graceful backend exit
    before force-killing, and stops the backend this way before it updates, so that hang
    is user visible. A stop() that returns proves nothing here; only exit does.
    """
    import subprocess

    program = _ABANDONED_SWEEP_PROGRAM.format(backend = str(Path(__file__).resolve().parent.parent))
    try:
        done = subprocess.run(
            [sys.executable, "-c", program], timeout = 60, capture_output = True, text = True
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(
            "the process never exited: an abandoned sweep is still holding it open"
        ) from None
    assert done.returncode == 0, f"stderr: {done.stderr[-2000:]}"


@pytest.mark.asyncio
async def test_stopping_the_sweeper_does_not_spend_the_desktop_exit_budget(monkeypatch):
    """Desktop force-kills after five seconds, so the sweeper's own budget must be small.

    Letting a parked sweep finish buys nothing: the boot reconcile settles every active
    run anyway, so this work is redundant at shutdown.
    """
    import time as _time

    parked = threading.Event()

    def _never_returns(*a, **k):
        parked.set()
        threading.Event().wait(60)

    monkeypatch.setattr(runs_db, "reconcile_runs", _never_returns)
    sweeper = _sweeper(interval_s = 0.01, timeout_s = 600.0)
    sweeper.start()
    for _ in range(500):
        if parked.is_set():
            break
        await asyncio.sleep(0.01)
    assert parked.is_set(), "the sweep never reached the database"
    started = _time.monotonic()
    await sweeper.stop()
    elapsed = _time.monotonic() - started
    assert elapsed < 2.0, f"sweeper.stop() took {elapsed:.2f}s of the graceful-exit budget"
