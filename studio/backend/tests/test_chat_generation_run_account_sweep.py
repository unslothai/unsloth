# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression: the lease sweep must not cancel a different account's live run.

Run ids are client chosen and the run rows live in per-account databases, so two accounts
can legitimately hold the same id. The supervisor keys its tasks, cancel events and
reservations by bare run id, so flattening every account's reconciled ids into one list and
calling supervisor.cancel(run_id) lets one account's aged row stop another account's live
generation.
"""

from __future__ import annotations

import asyncio
import secrets
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import chat_generation_runs as runs_mod  # noqa: E402
from core.inference.chat_generation_runs import (  # noqa: E402
    ChatGenerationLeaseSweeper,
    ChatGenerationSupervisor,
)
from storage import chat_generation_runs_db as runs_db  # noqa: E402
from storage import studio_db  # noqa: E402

_MINUTE_MS = 60_000


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


def _running_run(
    owner,
    run_id = "run-1",
    thread_id = "thread-1",
):
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
        owner_subject = owner,
        thread_id = thread_id,
        user_message_id = f"user-{run_id}",
        assistant_message_id = f"assistant-{run_id}",
        request_payload = {
            "model": "local.gguf",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        },
    )
    token = runs_db.get_worker_token(run_id)
    assert runs_db.mark_running(run_id, token)
    return token


@pytest.mark.asyncio
async def test_sweep_does_not_cancel_another_accounts_run_with_the_same_id(
    clock, tmp_path, monkeypatch
):
    from auth import policy, storage
    from state import active_generations
    from utils.account_context import AccountContext, run_as

    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth" / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / "auth" / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    try:
        alice = AccountContext(
            storage.issue_account_setup_code(username = "alice")["account"]["account_id"], "alice"
        )
        bob = AccountContext(
            storage.issue_account_setup_code(username = "bob")["account"]["account_id"], "bob"
        )
        policy.invalidate_account_cache()

        # Alice's row committed but her producer was never registered (supervisor.start
        # never ran, or the process restarted), so nothing live holds the id.
        run_as(alice, lambda: _running_run("alice"))
        clock.advance_ms(11 * _MINUTE_MS)

        app = SimpleNamespace(state = SimpleNamespace())
        supervisor = ChatGenerationSupervisor(app)
        app.state.chat_generation_supervisor = supervisor
        started = asyncio.Event()

        async def _produce(
            run_id,
            cancel_event = None,
            activity = None,
        ):
            started.set()
            await asyncio.sleep(3600)

        monkeypatch.setattr(supervisor, "_produce", _produce)

        # Bob may reuse the client-chosen id: the route only refuses ids held by a LIVE
        # registration, and Alice's stale row has none.
        assert [e for e in active_generations.snapshot() if e["run_id"] == "run-1"] == []

        def _start_bob():
            _running_run("bob")
            supervisor.start("run-1", thread_id = "thread-1", model = "local.gguf")

        run_as(bob, _start_bob)
        for _ in range(100):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set(), "Bob's producer never started"
        bob_event = supervisor._cancel_events["run-1"]
        bob_task = supervisor._tasks["run-1"]
        bob_registration = [e for e in active_generations.snapshot() if e["run_id"] == "run-1"]
        assert [e["account_id"] for e in bob_registration] == [bob.account_id]

        sweeper = ChatGenerationLeaseSweeper(app, interval_s = 60.0, timeout_s = 600.0)
        settled = await sweeper.sweep_once()
        assert settled == ["run-1"]
        assert run_as(alice, lambda: runs_db.get_run("run-1", "alice"))["status"] == "failed"

        assert (
            not bob_event.is_set()
        ), "the sweep cancelled another account's live run that happens to share the id"
        assert not bob_task.done()
        assert run_as(bob, lambda: runs_db.get_run("run-1", "bob"))["status"] == "running"
        bob_task.cancel()
    finally:
        policy.invalidate_account_cache()
