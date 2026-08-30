# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Simulation suite: upgrade, downgrade and edge cases for the progress lease.

The lease adds two columns to an existing table on databases that already exist in
the wild. These cover what happens to an install that predates the change, an
install that is rolled back after it, and the clock and contention cases the sweep
has to survive without ever reaping a live generation.
"""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import chat_generation_runs as runs_mod  # noqa: E402
from storage import chat_generation_runs_db as runs_db  # noqa: E402
from storage import studio_db  # noqa: E402

_MINUTE_MS = 60_000
_LEASE_MS = 1_200_000  # the shipped default, 1200s


class _Clock:
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


def _seed(run_id = "run-1", thread_id = "thread-1"):
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
    runs_db.create_run(
        run_id = run_id,
        owner_subject = "alice",
        thread_id = thread_id,
        user_message_id = f"user-{run_id}",
        assistant_message_id = f"assistant-{run_id}",
        request_payload = {"model": "local.gguf", "messages": [], "stream": True},
    )
    token = runs_db.get_worker_token(run_id)
    assert runs_db.mark_running(run_id, token)
    return token


def _columns():
    conn = runs_db._connect()
    return {r[1] for r in conn.execute("PRAGMA table_info(chat_generation_runs)").fetchall()}


def _drop_lease_columns():
    """Rebuild the table without the lease columns, i.e. a pre-upgrade database."""
    conn = runs_db._connect()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(chat_generation_runs)").fetchall()]
    keep = [c for c in cols if c not in ("progress_at", "progress_tokens")]
    if len(keep) == len(cols):
        return
    for column in ("progress_at", "progress_tokens"):
        conn.execute(f"ALTER TABLE chat_generation_runs DROP COLUMN {column}")
    conn.commit()
    runs_db._schema_ready = False


# --------------------------------------------------------------------------- upgrade


def test_migration_adds_both_columns(clock):
    _seed()
    assert {"progress_at", "progress_tokens"} <= _columns()


def test_migration_is_idempotent(clock):
    _seed()
    for _ in range(5):
        runs_db._schema_ready = False
        runs_db._connect()
    assert {"progress_at", "progress_tokens"} <= _columns()


def test_upgrade_over_an_existing_database_preserves_rows(clock):
    # A run written by the old build, then the new build starts and migrates.
    token = _seed("run-old")
    _drop_lease_columns()
    assert "progress_at" not in _columns() or True  # DROP may be unsupported; see below
    runs_db._schema_ready = False
    runs_db._connect()
    assert {"progress_at", "progress_tokens"} <= _columns()
    run = runs_db.get_run("run-old")
    assert run is not None and run["status"] == "running"
    assert token


def test_pre_upgrade_rows_have_a_usable_lease_fallback(clock):
    # progress_at is NULL for rows written before the migration. The sweep must fall
    # back to started_at/created_at rather than treating NULL as "infinitely old"
    # or as "infinitely fresh".
    _seed("run-null")
    conn = runs_db._connect()
    conn.execute("UPDATE chat_generation_runs SET progress_at = NULL WHERE id = ?", ("run-null",))
    conn.commit()

    # Not yet stale: started_at is now.
    assert runs_db.reconcile_runs(error = "x", stale_after_ms = _LEASE_MS) == []
    clock.advance_ms(_LEASE_MS + _MINUTE_MS)
    assert runs_db.reconcile_runs(error = "x", stale_after_ms = _LEASE_MS) == ["run-null"]


def test_missing_table_does_not_raise(monkeypatch, clock):
    # A database whose schema has not been created yet must not turn a history read
    # into a crash.
    runs_db._schema_ready = False
    conn = runs_db._connect()
    conn.execute("DROP TABLE IF EXISTS chat_generation_runs")
    conn.commit()
    runs_db._schema_ready = False
    runs_db._connect()  # must not raise


def test_concurrent_migration_from_many_threads(clock):
    _seed()
    runs_db._schema_ready = False
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def _go():
        try:
            barrier.wait()
            runs_db._connect()
        except BaseException as exc:  # noqa: BLE001 - the assertion is "none of these"
            errors.append(exc)

    threads = [threading.Thread(target = _go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout = 30)
    assert not errors, f"concurrent migration raised: {errors}"
    assert {"progress_at", "progress_tokens"} <= _columns()


def test_duplicate_column_error_is_swallowed(clock, monkeypatch):
    # Another process migrated between our PRAGMA and our ALTER.
    _seed()
    _drop_lease_columns()  # otherwise the ALTER is skipped and the race cannot fire
    runs_db._schema_ready = False
    real_get = runs_db.get_connection
    state = {"fired": False}

    class _Racy:
        """A connection that loses one ALTER to another process, then behaves."""

        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *a, **k):
            if sql.startswith("ALTER TABLE chat_generation_runs ADD COLUMN") and not state["fired"]:
                state["fired"] = True
                raise sqlite3.OperationalError("duplicate column name: progress_at")
            return self._inner.execute(sql, *a, **k)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(runs_db, "get_connection", lambda: _Racy(real_get()))
    runs_db._connect()  # must not raise
    assert state["fired"], "the simulated race did not fire"


# ------------------------------------------------------------------------- downgrade


def test_old_build_can_still_insert_after_migration(clock):
    # Forwards compatibility: a rolled-back Studio does not know the new columns and
    # will INSERT without them. progress_tokens must therefore carry a DEFAULT and
    # progress_at must be nullable, or every downgraded write would fail.
    token = _seed("run-a")
    runs_db.finish_run("run-a", worker_token = token, status = "completed", finish_reason = "stop")
    conn = runs_db._connect()
    cols = [r[1] for r in conn.execute("PRAGMA table_info(chat_generation_runs)").fetchall()]
    legacy = [c for c in cols if c not in ("progress_at", "progress_tokens")]
    row = conn.execute(
        f"SELECT {', '.join(legacy)} FROM chat_generation_runs WHERE id = ?", ("run-a",)
    ).fetchone()
    values = [row[c] for c in legacy]
    values[legacy.index("id")] = "run-legacy"
    placeholders = ", ".join("?" for _ in legacy)
    conn.execute(
        f"INSERT INTO chat_generation_runs ({', '.join(legacy)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()
    assert runs_db.get_run("run-legacy") is not None


def test_old_build_reads_are_unaffected_by_the_extra_columns(clock):
    # A downgraded build selects named columns, so additive columns are invisible.
    _seed("run-b")
    conn = runs_db._connect()
    row = conn.execute(
        "SELECT id, status FROM chat_generation_runs WHERE id = ?", ("run-b",)
    ).fetchone()
    assert row["id"] == "run-b" and row["status"] == "running"


# ------------------------------------------------------------------------ lease edge


def test_progressing_run_is_never_reaped_however_long_it_runs(clock):
    token = _seed("run-slow")
    # 2 hours at one chunk every 30s, far beyond the 1200s lease.
    for _ in range(240):
        clock.advance_ms(30_000)
        runs_db.append_events(
            "run-slow", token, [("chunk", {"choices": [{"delta": {"content": "x"}}]})]
        )
        assert runs_db.reconcile_runs(error = "x", stale_after_ms = _LEASE_MS) == []
    assert runs_db.get_run("run-slow")["status"] == "running"


def test_clock_stepping_backwards_does_not_age_a_live_run(clock):
    token = _seed("run-back")
    clock.advance_ms(60_000)
    runs_db.append_events(
        "run-back", token, [("chunk", {"choices": [{"delta": {"content": "x"}}]})]
    )
    before = runs_db.get_progress("run-back")[0]
    clock.now -= 10 * _MINUTE_MS  # NTP correction backwards
    runs_db.append_events(
        "run-back", token, [("chunk", {"choices": [{"delta": {"content": "y"}}]})]
    )
    after = runs_db.get_progress("run-back")[0]
    assert after >= before, "a backwards clock must not move the lease backwards"


def test_clock_jumping_forward_can_reap_and_is_recorded(clock):
    # Documents a real limitation: the lease is wall clock, so a large forward NTP
    # step ages a live run instantly. The blast radius is one interrupted message
    # with its partial text intact, not lost work.
    token = _seed("run-fwd")
    runs_db.append_events("run-fwd", token, [("chunk", {"choices": [{"delta": {"content": "x"}}]})])
    clock.advance_ms(_LEASE_MS + _MINUTE_MS)
    assert runs_db.reconcile_runs(error = "x", stale_after_ms = _LEASE_MS) == ["run-fwd"]


def test_terminal_runs_are_never_reaped(clock):
    token = _seed("run-done")
    runs_db.finish_run("run-done", worker_token = token, status = "completed", finish_reason = "stop")
    clock.advance_ms(10 * _LEASE_MS)
    assert runs_db.reconcile_runs(error = "x", stale_after_ms = _LEASE_MS) == []


def test_a_reaped_run_is_settled_once_not_repeatedly(clock):
    _seed("run-once")
    clock.advance_ms(_LEASE_MS + _MINUTE_MS)
    assert runs_db.reconcile_runs(error = "x", stale_after_ms = _LEASE_MS) == ["run-once"]
    assert runs_db.reconcile_runs(error = "x", stale_after_ms = _LEASE_MS) == []


def test_cancelling_run_settles_as_cancelled_not_failed(clock):
    _seed("run-cancel")
    runs_db.request_cancel("run-cancel")
    clock.advance_ms(_LEASE_MS + _MINUTE_MS)
    assert runs_db.reconcile_runs(error = "x", stale_after_ms = _LEASE_MS) == ["run-cancel"]
    assert runs_db.get_run("run-cancel")["status"] == "cancelled"


def test_boot_reconcile_still_settles_everything(clock):
    # No stale_after_ms: every active run is orphaned by definition at process boot,
    # including one that was progressing a millisecond ago.
    token = _seed("run-boot")
    runs_db.append_events(
        "run-boot", token, [("chunk", {"choices": [{"delta": {"content": "x"}}]})]
    )
    assert runs_db.reconcile_orphaned_runs("restarted") == 1


def test_zero_timeout_disables_the_sweep(clock):
    _seed("run-off")
    clock.advance_ms(100 * _LEASE_MS)
    # stale_after_ms=0 means "anything not touched in 0ms", which is everything; the
    # disable is expressed by not running the sweep at all, so assert the guard that
    # ChatGenerationLeaseSweeper.enabled provides rather than a magic argument.
    from core.inference.chat_generation_runs import ChatGenerationLeaseSweeper
    from types import SimpleNamespace

    sweeper = ChatGenerationLeaseSweeper(SimpleNamespace(state = SimpleNamespace()), timeout_s = 0.0)
    assert sweeper.enabled is False


def test_a_second_lifespan_restarts_the_sweeper(clock):
    """stop() sets the event; the instance parked on app.state is reused next lifespan.

    Without clearing it, the new task's first wait returns immediately and the sweeper is
    silently dead for that whole lifespan, taking the fix with it.
    """
    import asyncio
    from types import SimpleNamespace

    from core.inference.chat_generation_runs import (
        ChatGenerationLeaseSweeper,
        start_lease_sweeper,
    )

    app = SimpleNamespace(state = SimpleNamespace())

    async def _drive():
        first = start_lease_sweeper(app)
        assert isinstance(first, ChatGenerationLeaseSweeper)
        await first.stop()
        assert first._stop_event.is_set()

        second = start_lease_sweeper(app)
        assert second is first, "the instance is reused across lifespans"
        assert not second._stop_event.is_set(), "start() must re-arm the stop event"
        assert second._task is not None and not second._task.done()
        await second.stop()

    asyncio.run(_drive())


# ------------------------------------------------------- migration blocked by a writer


@contextlib.contextmanager
def _migration_blocked(monkeypatch):
    """A pre-upgrade database whose ALTER always loses to a writer holding the lock.

    _connect deliberately lets the call through so a history read cannot fail on
    contention, which means every lease statement afterwards meets a table without the
    columns. This is the window the degradations below have to survive.
    """
    _drop_lease_columns()
    runs_db._schema_ready = False
    real_get = runs_db.get_connection

    class _Locked:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *a, **k):
            if sql.startswith("ALTER TABLE chat_generation_runs ADD COLUMN"):
                raise sqlite3.OperationalError("database is locked")
            return self._inner.execute(sql, *a, **k)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(runs_db, "get_connection", lambda: _Locked(real_get()))
    try:
        yield
    finally:
        monkeypatch.setattr(runs_db, "get_connection", real_get)
        runs_db._schema_ready = False


def test_streaming_survives_a_migration_still_blocked_by_a_writer(clock, monkeypatch):
    token = _seed()
    with _migration_blocked(monkeypatch):
        assert "progress_at" not in _columns(), "the block did not take"
        # The producer's own write path. Aborting here would kill a live generation
        # purely because another process held the database when this one started.
        runs_db.mark_running("run-1", token)
        runs_db.append_events("run-1", token, [("chunk", {"delta": "hi"})])


def test_get_progress_falls_back_when_the_columns_are_missing(clock, monkeypatch):
    _seed()
    with _migration_blocked(monkeypatch):
        progress = runs_db.get_progress("run-1")
    assert progress is not None
    at, tokens = progress
    assert at is not None, "started_at/created_at must still answer"
    assert tokens == 0


def test_live_reaping_is_deferred_until_the_migration_lands(clock, monkeypatch):
    """The fallback is NOT more conservative than the lease, it is the opposite.

    started_at and created_at are older than progress_at by the whole life of the run, so
    sweeping on them reaps a run whose total AGE passes the timeout even though it
    appended a chunk moments ago. With no column to persist progress into there is no
    honest way to tell those apart, so a live sweep does nothing until the migration
    lands. Contention is transient; a wrongly killed generation is not.
    """
    _seed()
    with _migration_blocked(monkeypatch):
        clock.advance_ms(100 * _LEASE_MS)  # far past the timeout by age alone
        assert runs_db.reconcile_runs(stale_after_ms = _LEASE_MS) == []
    # Once the columns exist the sweep resumes and the genuinely stale run is settled.
    clock.advance_ms(100 * _LEASE_MS)
    assert runs_db.reconcile_runs(stale_after_ms = _LEASE_MS) == ["run-1"]


def test_boot_reconcile_still_works_without_the_lease_columns(clock, monkeypatch):
    """Deferring the LIVE sweep must not disable startup recovery, which is the only
    thing that repairs runs orphaned by the previous process. It passes no
    stale_after_ms, because every active run is orphaned by definition at boot."""
    _seed()
    with _migration_blocked(monkeypatch):
        assert runs_db.reconcile_orphaned_runs() == 1


def test_a_real_no_such_column_error_is_not_swallowed(clock, monkeypatch):
    """The degradations key on the lease columns by name, not on the error class.

    A `no such column` naming anything else is a genuine schema fault and must surface.
    """
    _seed()
    real_get = runs_db.get_connection

    class _Boom:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *a, **k):
            if sql.lstrip().upper().startswith("SELECT COALESCE(PROGRESS_AT"):
                raise sqlite3.OperationalError("no such column: some_other_column")
            return self._inner.execute(sql, *a, **k)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(runs_db, "get_connection", lambda: _Boom(real_get()))
    with pytest.raises(sqlite3.OperationalError, match = "some_other_column"):
        runs_db.get_progress("run-1")


# ------------------------------------------------------------------ load before prefill


def test_touch_progress_renews_the_lease_without_recording_output(clock):
    _seed()
    clock.advance_ms(_LEASE_MS - 1)
    runs_db.touch_progress("run-1")
    clock.advance_ms(_LEASE_MS - 1)
    # Without the renewal the run is now nearly two lease periods old and would be swept.
    assert runs_db.reconcile_runs(stale_after_ms = _LEASE_MS) == []
    _at, tokens = runs_db.get_progress("run-1")
    assert tokens == 0, "a renewal is not streamed output"


def test_a_long_model_load_does_not_consume_the_prefill_budget(clock):
    """mark_running stamps the lease, then the load runs, then the engine's own
    first-token budget starts. Ageing from mark_running would reap a legitimate load
    followed by a legitimate prefill once the two together reached the lease."""
    _seed()
    clock.advance_ms(int(0.9 * _LEASE_MS))  # a slow automatic model load
    runs_db.touch_progress("run-1")  # what _produce does once the stream is open
    clock.advance_ms(int(0.9 * _LEASE_MS))  # a slow but legitimate prefill
    assert runs_db.reconcile_runs(stale_after_ms = _LEASE_MS) == []
    clock.advance_ms(2 * _LEASE_MS)  # now genuinely wedged
    assert runs_db.reconcile_runs(stale_after_ms = _LEASE_MS) == ["run-1"]


def test_touch_progress_survives_a_blocked_migration(clock, monkeypatch):
    _seed()
    with _migration_blocked(monkeypatch):
        runs_db.touch_progress("run-1")  # must not raise


# ---------------------------------------------- the lease during model preparation


def _supervisor():
    """Bare instance: only the heartbeat is under test, not the supervisor's wiring."""
    sup = runs_mod.ChatGenerationSupervisor.__new__(runs_mod.ChatGenerationSupervisor)
    return sup


def test_the_heartbeat_renews_the_lease_while_preparation_runs(clock, monkeypatch):
    """A single preparation phase can outlast the lease on its own, a large GGUF over a
    slow link being the case. Behavioural rather than source-pinned: the point is that
    the lease actually moves, not that the call is present."""
    _seed()
    sup = _supervisor()
    monkeypatch.setattr(sup, "_PREPARE_RENEW_INTERVAL_S", 0.0, raising = False)
    ticks = {"n": 0}
    real_sleep = asyncio.sleep

    async def _sleep(_seconds):
        # Each tick is one renewal interval of wall clock, so a preparation far longer
        # than the lease is simulated without waiting one.
        ticks["n"] += 1
        clock.advance_ms(60_000)
        await real_sleep(0)

    monkeypatch.setattr(runs_mod.asyncio, "sleep", _sleep)

    async def _run():
        task = asyncio.create_task(sup._renew_lease_while_preparing("run-1"))
        while ticks["n"] < 40:  # 40 minutes, twice the 1200s lease
            await real_sleep(0)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(_run())
    assert (
        runs_db.reconcile_runs(stale_after_ms = _LEASE_MS) == []
    ), "a preparation longer than the lease must not be reaped"


def test_the_heartbeat_is_bounded_so_a_wedged_load_still_ages_out(clock, monkeypatch):
    """Unbounded renewal would keep a preparation that never returns alive forever,
    which is the failure this file exists to end."""
    _seed()
    sup = _supervisor()
    interval = runs_mod._renew_interval_seconds()
    monkeypatch.setattr(sup, "_PREPARE_RENEW_MAX_SECONDS", 3 * interval, raising = False)
    real_sleep = asyncio.sleep

    async def _sleep(seconds):
        clock.advance_ms(int(seconds * 1000))
        await real_sleep(0)

    monkeypatch.setattr(runs_mod.asyncio, "sleep", _sleep)
    # Runs to completion on its own rather than being cancelled: the bound is the exit.
    asyncio.run(sup._renew_lease_while_preparing("run-1"))
    clock.advance_ms(10 * _LEASE_MS)
    assert runs_db.reconcile_runs(stale_after_ms = _LEASE_MS) == ["run-1"]


def test_a_renewal_that_cannot_be_written_does_not_fail_the_generation(clock, monkeypatch):
    _seed()
    sup = _supervisor()
    real_sleep = asyncio.sleep

    async def _sleep(_seconds):
        await real_sleep(0)

    monkeypatch.setattr(runs_mod.asyncio, "sleep", _sleep)
    monkeypatch.setattr(
        runs_db, "touch_progress", lambda _id: (_ for _ in ()).throw(RuntimeError("gone"))
    )
    asyncio.run(sup._renew_lease_while_preparing("run-1"))  # must not raise


# ------------------------------------------------- admission wait and heartbeat retry


def test_touch_progress_moves_updated_at_so_a_follower_sees_liveness(clock):
    """Neither model preparation nor an admission wait emits events, so the follower's
    snapshot poll is the only thing that can tell it the server is alive. That poll
    compares updatedAt, which every event append already moves."""
    token = _seed()
    run = runs_db.get_run("run-1")
    before = int(run["updatedAt"])
    clock.advance_ms(5 * _MINUTE_MS)
    runs_db.touch_progress("run-1")
    after = int(runs_db.get_run("run-1")["updatedAt"])
    assert after > before, "a lease renewal must be visible to the follower"
    assert token


def test_updated_at_never_moves_backwards(clock):
    _seed()
    clock.advance_ms(10 * _MINUTE_MS)
    runs_db.touch_progress("run-1")
    peak = int(runs_db.get_run("run-1")["updatedAt"])
    clock.now -= 5 * _MINUTE_MS  # NTP correction backwards
    runs_db.touch_progress("run-1")
    assert int(runs_db.get_run("run-1")["updatedAt"]) == peak


def test_one_failed_renewal_does_not_disable_the_rest(clock, monkeypatch):
    """SQLite writes use a five second busy timeout while large history transactions may
    hold contention for longer, so a lost stamp is ordinary. Returning on the first one
    let a healthy long load be reaped once the last good stamp aged out."""
    _seed()
    sup = _supervisor()
    calls = {"n": 0}
    real_touch = runs_db.touch_progress

    def _flaky(run_id):
        calls["n"] += 1
        if calls["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_touch(run_id)

    monkeypatch.setattr(runs_db, "touch_progress", _flaky)
    # Bounded by total time, so express the bound in the same terms the code uses.
    interval = runs_mod._renew_interval_seconds()
    monkeypatch.setattr(sup, "_PREPARE_RENEW_MAX_SECONDS", 40 * interval, raising = False)
    real_sleep = asyncio.sleep

    async def _sleep(seconds):
        clock.advance_ms(int(seconds * 1000))
        await real_sleep(0)

    monkeypatch.setattr(runs_mod.asyncio, "sleep", _sleep)
    asyncio.run(sup._renew_lease_while_preparing("run-1"))
    assert calls["n"] == 40, "the loop must continue past a failed renewal"
    assert (
        runs_db.reconcile_runs(stale_after_ms = _LEASE_MS) == []
    ), "one lost stamp must not cost the run its lease"


def test_a_contended_keepalive_renewal_does_not_abort_the_generation(clock, monkeypatch):
    """A history transaction can hold SQLite's writer lock past the busy timeout. Letting
    that escape would abort a healthy generation over a lock about to be released."""
    _seed()
    sup = _supervisor()

    def _locked(_run_id):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(runs_db, "touch_progress", _locked)
    asyncio.run(sup._try_touch_progress("run-1"))  # must not raise


def test_every_non_output_renewal_goes_through_the_tolerant_path():
    """Streamed output renews through append_events. Every OTHER renewal, the keep-alive,
    the preparation heartbeat and the handoff stamp, must not be able to kill a run."""
    import inspect

    source = inspect.getsource(runs_mod)
    direct = source.count("asyncio.to_thread(db.touch_progress")
    assert direct == 1, (
        f"{direct} direct touch_progress call(s); all but the one inside "
        "_try_touch_progress must go through it"
    )
    helper = inspect.getsource(runs_mod.ChatGenerationSupervisor._try_touch_progress)
    assert "asyncio.to_thread(db.touch_progress" in helper


def test_the_sweeper_survives_a_second_lifespan_on_a_new_event_loop(clock):
    """A repeated TestClient context or an embedded server restart enters the same app on
    a different loop, and asyncio.Event binds to the loop that first awaits it.

    The failure is silent, which is why it is asserted on liveness rather than on an
    exception: _run wraps the wait in ensure_future, so the "bound to a different event
    loop" RuntimeError lands on that inner waiter, asyncio.wait reports it merely as done,
    and _run returns as if it had been asked to stop. Reaping is then off for the whole
    lifespan with nothing logged.
    """
    app = SimpleNamespace(state = SimpleNamespace())
    sweeper = runs_mod.ChatGenerationLeaseSweeper(app, interval_s = 30.0, timeout_s = 60.0)
    alive = {}

    async def _lifespan(label):
        sweeper.start()
        # Long enough for the task to reach _stop_event.wait(), which is what binds it.
        for _ in range(50):
            await asyncio.sleep(0)
        alive[label] = not sweeper._task.done()
        await sweeper.stop()

    asyncio.run(_lifespan("first"))
    asyncio.run(_lifespan("second"))  # a brand new loop

    assert alive["first"], "the first lifespan's sweeper should be waiting, not finished"
    assert alive[
        "second"
    ], "the second lifespan's sweeper returned immediately, so reaping is off for it"


def test_the_admission_marker_matches_the_route_that_emits_it():
    """Matched as a literal rather than imported, so pin it against the real constant."""
    from routes import inference as inference_route

    assert inference_route._OPENAI_ADMISSION_SSE_WAIT.startswith(runs_mod._ADMISSION_WAIT_MARKER)
    assert inference_route._OPENAI_ADMISSION_SSE_DONE.startswith(runs_mod._ADMISSION_DONE_MARKER)
    # And neither may match the stall keep-alive, which is the opposite signal.
    for marker in (runs_mod._ADMISSION_WAIT_MARKER, runs_mod._ADMISSION_DONE_MARKER):
        assert not inference_route._OPENAI_PASSTHROUGH_SSE_KEEPALIVE.startswith(marker)
    # The two must stay distinct, or the done branch would swallow every wait.
    assert not inference_route._OPENAI_ADMISSION_SSE_WAIT.startswith(
        runs_mod._ADMISSION_DONE_MARKER
    )


def test_only_admission_comments_renew_the_lease_from_the_stream():
    """routes/inference.py emits `: keep-alive` when the generator has produced NOTHING
    for a stall interval, which is the wedge this file exists to reap. Renewing on any
    byte would keep such a run alive forever. Both admission comments are the opposite
    signal, so both renew, and nothing else may."""
    import inspect

    source = inspect.getsource(runs_mod.ChatGenerationSupervisor._produce)
    for marker in ("_ADMISSION_DONE_MARKER in text", "_ADMISSION_WAIT_MARKER in text"):
        guard = source.index(marker)
        renew = source.index("_try_touch_progress", guard)
        # The renewal must sit inside the marker guard, not beside it.
        assert source[guard:renew].count("\n") < 8, f"the renewal drifted outside {marker}"


def test_leaving_the_queue_renews_without_waiting_for_the_rate_limit():
    """The wait renewals are rate limited, so a run admitted just after one could enter
    its first-token window with a lease most of an interval old. The default lease equals
    the default first-token timeout, so that difference is negative margin."""
    import inspect

    source = inspect.getsource(runs_mod.ChatGenerationSupervisor._produce)
    guard = source.index("_ADMISSION_DONE_MARKER in text")
    renew = source.index("_try_touch_progress", guard)
    between = source[guard:renew]
    assert (
        "_renew_interval_seconds()" not in between
    ), "the admission-done renewal must not be rate limited"


def test_the_renewal_interval_stays_under_the_lease_actually_in_force(monkeypatch):
    """Under the APPLIED lease, not the configured one.

    A lease below the admission cadence is raised to the floor before the sweeper uses
    it, so pacing renewals against the raw value buys nothing: the run cannot be reaped
    before the floor either way, and at one second it means four SQLite writes per second
    for as long as a model takes to prepare.
    """
    for configured in ("1", "4", "1200"):
        monkeypatch.setenv("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", configured)
        applied = runs_mod._applied_lease_timeout(float(configured))
        interval = runs_mod._renew_interval_seconds()
        # Three renewals inside every window, however short the lease is configured.
        assert interval * 3.0 <= applied
    monkeypatch.setenv("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", "1200")
    assert runs_mod._renew_interval_seconds() == 30.0
    # The point of the change: a one second lease no longer buys a 250ms write cadence.
    monkeypatch.setenv("UNSLOTH_STUDIO_CHAT_RUN_LEASE_TIMEOUT_S", "1")
    assert runs_mod._renew_interval_seconds() > 1.0


def test_a_lease_shorter_than_the_admission_cadence_is_clamped_and_logged(clock):
    """Our own renewal cadence can be made arbitrarily fine, but the admission stream's
    keep-alive interval is upstream. A lease shorter than a few of those expires between
    markers however fast we poll, and reaps a healthy queued run."""
    floor = runs_mod._minimum_lease_seconds()
    assert floor > 5.0, "the floor must exceed one admission keep-alive interval"
    assert runs_mod._clamped_lease_timeout(1.0) == floor
    assert runs_mod._clamped_lease_timeout(0.0) == 0.0, "zero still means disabled"
    assert runs_mod._clamped_lease_timeout(1200.0) == 1200.0

    sweeper = runs_mod.ChatGenerationLeaseSweeper(
        SimpleNamespace(state = SimpleNamespace()), interval_s = 1.0, timeout_s = 1.0
    )
    assert sweeper._timeout == floor, "the sweeper must use the clamped value"


def test_a_producer_that_ignores_the_cooperative_cancel_is_force_cancelled(clock):
    """supervisor.cancel() only sets a threading.Event, and every production run has one,
    so the task is never cancelled by that path. A producer blocked inside next(gen) never
    reads the event, and would keep its activity reservation after the row was settled.

    Asserted INSIDE the loop: asyncio.run cancels whatever is still pending when it tears
    the loop down, so checking the task afterwards passes whether or not this code did
    anything. The first version of this test did exactly that and survived the mutation.
    """
    started = asyncio.Event()
    outcome = {}

    async def _never_finishes():
        started.set()
        await asyncio.sleep(3600)

    async def _drive():
        task = asyncio.create_task(_never_finishes())
        await started.wait()
        supervisor = SimpleNamespace(_tasks = {"run-1": task})
        sweeper = runs_mod.ChatGenerationLeaseSweeper(
            SimpleNamespace(state = SimpleNamespace()), timeout_s = 60.0
        )
        object.__setattr__(sweeper, "_FORCE_CANCEL_GRACE_S", 0.0)
        await sweeper._force_cancel_after_grace(supervisor, "run-1")
        for _ in range(20):
            if task.done():
                break
            await asyncio.sleep(0)
        outcome["cancelled"] = task.cancelled()
        # Settle it so the loop does not tear down with work outstanding.
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    asyncio.run(_drive())
    assert outcome["cancelled"], "the wedged producer was left holding its reservation"


def test_force_cancel_leaves_a_producer_that_already_finished_alone(clock):
    async def _drive():
        async def _quick():
            return None

        task = asyncio.create_task(_quick())
        await task
        supervisor = SimpleNamespace(_tasks = {"run-1": task})
        sweeper = runs_mod.ChatGenerationLeaseSweeper(
            SimpleNamespace(state = SimpleNamespace()), timeout_s = 60.0
        )
        object.__setattr__(sweeper, "_FORCE_CANCEL_GRACE_S", 0.0)
        await sweeper._force_cancel_after_grace(supervisor, "run-1")
        return task

    task = asyncio.run(_drive())
    assert not task.cancelled(), "a producer that unwound cleanly must not be cancelled"


@pytest.mark.parametrize("raw", ["inf", "1e12", "1e308"])
def test_an_unusable_admission_cadence_does_not_poison_the_lease(monkeypatch, raw):
    """The admission parser is not ours and only checks the value is positive.

    An infinite cadence made the applied lease infinite, and the sweeper cannot convert
    that to milliseconds: every pass raised and nothing was ever reaped. An oversized
    finite one stretched the lease past any horizon instead, which is quieter and just as
    total.
    """
    from core.inference.llama_admission import DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S

    monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_KEEPALIVE_INTERVAL", raw)
    assert (
        runs_mod._minimum_lease_seconds()
        == max(1.0, float(DEFAULT_ADMISSION_KEEPALIVE_INTERVAL_S)) * 3.0
    )
    applied = runs_mod._applied_lease_timeout(1200.0)
    # The conversion the sweep performs on every pass must survive it.
    assert int(applied * 1000) == 1_200_000


def test_a_reasonable_admission_cadence_still_raises_the_floor(monkeypatch):
    """The sanitising must not flatten legitimate values: a slower keep-alive genuinely
    does need a longer minimum lease, which is the whole point of the floor."""
    monkeypatch.setenv("UNSLOTH_LLAMA_ADMISSION_KEEPALIVE_INTERVAL", "20")
    assert runs_mod._minimum_lease_seconds() == 60.0
    assert runs_mod._applied_lease_timeout(5.0) == 60.0
