# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Eject authority: resident first, then admitted background load, then the invocation record."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from core.inference.diffusion import DiffusionBackend, _LoadingState
from hub.services.models import account_access

OWNER = "owner"
ALICE = "a" * 32
BOB = "b" * 32


@pytest.fixture
def backend(monkeypatch):
    engine = DiffusionBackend()
    # The eject path's own authorization is under test, not require_resident_control.
    monkeypatch.setattr(account_access, "require_resident_control", lambda *a, **k: None)
    monkeypatch.setattr(engine, "_unload_locked", lambda: setattr(engine, "_state", None))
    return engine


def _resident(engine, repo_id = "org/model"):
    engine._state = SimpleNamespace(repo_id = repo_id)


def _background_load(
    engine,
    account_id,
    repo_id = "org/other",
):
    engine._loading = _LoadingState(repo_id = repo_id, base_repo = repo_id, account_id = account_id)


def _pending_invocation(engine, account_id):
    """What a direct load_pipeline / a caller still inside begin_load preflight leaves behind."""
    engine._load_accounts[object()] = (engine._load_token, account_id)


def test_the_resident_owner_can_eject_while_another_account_queues_a_replacement(backend):
    _resident(backend)
    _background_load(backend, ALICE)
    assert backend.unload(expected_account = BOB)["loaded"] is False
    assert backend._unload_waiters == 0


def test_a_foreign_account_cannot_cancel_a_pending_background_load(backend):
    _background_load(backend, ALICE)
    with pytest.raises(account_access.GpuBusyForAnotherAccountError):
        backend.unload(expected_account = BOB)
    assert backend._unload_waiters == 0 and not backend._cancel_event.is_set()
    assert backend._loading is not None


def test_a_foreign_account_cannot_cancel_a_pending_cpu_load(backend):
    _pending_invocation(backend, ALICE)
    with pytest.raises(account_access.GpuBusyForAnotherAccountError):
        backend.unload(expected_account = BOB)
    assert backend._unload_waiters == 0 and not backend._cancel_event.is_set()


def test_the_account_that_started_the_load_can_still_cancel_it(backend):
    _background_load(backend, ALICE)
    _pending_invocation(backend, ALICE)
    backend.unload(expected_account = ALICE)
    assert backend._cancel_event.is_set() and backend._loading is None
    assert backend._unload_waiters == 0


def test_an_active_foreign_generation_still_refuses_the_eject(backend):
    _resident(backend)
    backend._active_generate_cancel = threading.Event()
    backend._active_generate_account = ALICE
    with pytest.raises(account_access.GpuBusyForAnotherAccountError):
        backend.unload(expected_account = BOB)
    assert not backend._active_generate_cancel.is_set()


def test_a_refused_eject_leaves_no_fence_behind(backend, monkeypatch):
    """Whatever raises, _unload_waiters comes back to zero: a leaked count would make every later
    load wait out a teardown that is not happening."""
    _resident(backend)
    boom = RuntimeError("teardown exploded")

    def explode():
        raise boom

    monkeypatch.setattr(backend, "_unload_locked", explode)
    with pytest.raises(RuntimeError, match = "teardown exploded"):
        backend.unload()
    assert backend._unload_waiters == 0 and backend._teardown_waiters == 0


def test_the_owner_can_eject_an_idle_backend(backend):
    assert backend.unload(expected_account = OWNER)["loaded"] is False


def test_a_preflight_caller_cannot_block_the_owner_of_the_load_in_flight(backend):
    """Bob is inside begin_load validation (recorded, not admitted) while Alice's background load
    runs. The admitted load owns the epoch, so Alice can still cancel her own construction."""
    _background_load(backend, ALICE)
    _pending_invocation(backend, BOB)
    backend.unload(expected_account = ALICE)
    assert backend._cancel_event.is_set() and backend._loading is None


def test_a_preflight_caller_cannot_block_the_resident_owner(backend):
    """Same, with a published resident instead of a load in flight."""
    _resident(backend)
    _pending_invocation(backend, BOB)
    assert backend.unload(expected_account = ALICE)["loaded"] is False


def test_an_eject_arriving_between_the_wait_and_the_lock_is_waited_out(backend, monkeypatch):
    """An eject landing between the wait and the registration lock bumps the epoch too, so a
    re-read adopts the eject's OWN token and the load is admitted into its teardown."""
    from core.inference.diffusion import _account_owned_load

    calls = {"waits": 0}
    real_wait = backend._wait_for_pending_unloads
    started = threading.Event()

    def racing_wait(*args, **kwargs):
        # Counted on ENTRY: the second call is the one that blocks, and the thread that releases the
        # fence has to know it has started before it can let go of it.
        calls["waits"] += 1
        if calls["waits"] >= 2:
            started.set()
        real_wait(*args, **kwargs)
        if calls["waits"] == 1:
            # Exactly the gap: the eject lands after the wait returned and before the caller can
            # take _load_cancel_lock.
            with backend._load_cancel_lock:
                backend._unload_waiters += 1
                backend._load_token += 1

    monkeypatch.setattr(backend, "_wait_for_pending_unloads", racing_wait)

    @_account_owned_load
    def admitted(self, *, _load_token = None):
        return _load_token

    def release():
        started.wait(timeout = 10)
        with backend._load_cancel_lock:
            backend._unload_waiters -= 1
            backend._unload_fence_clear.set()

    releaser = threading.Thread(target = release, daemon = True)
    releaser.start()
    token = admitted(backend)
    releaser.join(timeout = 10)

    assert calls["waits"] == 2, "the load registered without waiting the racing eject out"
    assert token == backend._load_token
    assert backend._load_accounts == {}


def test_a_load_waiting_out_an_eject_sleeps_instead_of_spinning(backend):
    """_teardown_drained is still SET between the eject being accepted and the teardown being
    reserved, so waiting on it there returned instantly and burned a core per waiter."""
    with backend._load_cancel_lock:
        backend._unload_waiters += 1
        backend._unload_fence_clear.clear()
    assert backend._teardown_drained.is_set()

    polls = {"n": 0}
    real_wait = backend._unload_fence_clear.wait

    def counted(timeout = None):
        polls["n"] += 1
        return real_wait(timeout = timeout)

    backend._unload_fence_clear.wait = counted

    def release():
        time.sleep(0.5)
        with backend._load_cancel_lock:
            backend._unload_waiters -= 1
            backend._unload_fence_clear.set()

    releaser = threading.Thread(target = release, daemon = True)
    releaser.start()
    backend._wait_for_pending_unloads()
    releaser.join(timeout = 10)

    # ~5 polls at the 0.1s timeout. A spin does tens of thousands in the same half second.
    assert polls["n"] <= 20, polls


def test_stop_cancels_a_queued_generation_while_an_eject_waits_for_the_lock(backend):
    """Stop must cancel a queued generation while an eject waits for the lock; the fence denies it."""
    queued = threading.Event()
    backend._queued_generate_cancels.add(queued)
    with backend._load_cancel_lock:
        backend._unload_waiters += 1
        backend._unload_fence_clear.clear()
    assert not backend._teardown_waiters and not backend._transition_owns_slot

    assert backend.cancel_generate() is True
    assert queued.is_set()


def test_stop_still_reports_nothing_to_cancel_on_an_idle_backend(backend):
    backend._queued_generate_cancels.add(threading.Event())
    assert backend.cancel_generate() is False


def test_a_cancelled_load_still_reports_the_repos_it_is_reading(backend):
    """The eject drops _loading at once, but that load's thread reads on until it unwinds, holding
    no lock inside _prefetch_files. The delete guard must keep refusing until it is done."""
    backend._loading = _LoadingState(
        repo_id = "org/model-GGUF",
        base_repo = "org/base",
        account_id = ALICE,
        fetch_repo = "mirror/base",
    )
    everything = {"org/model-GGUF", "org/base", "mirror/base"}
    cancelled_token = backend._load_token

    backend.unload(expected_account = ALICE)

    assert backend._loading is None
    assert set(backend.draining_repo_ids()) == everything, "deletable while still being read"
    # Not loading_repo_ids: release_if, keep-warm and the auto-switch read that as ownership.
    assert backend.loading_repo_ids() == ()

    with backend._load_cancel_lock:
        backend._draining_repos.pop(cancelled_token, None)
    assert backend.draining_repo_ids() == ()


def test_a_cancelled_load_still_reports_its_claimed_asset_repos(backend):
    """The encoder download ignores the cancel event, so its repo stays undeletable after an eject."""
    backend._loading = _LoadingState(
        repo_id = "unsloth/Qwen-Image-2.1-GGUF",
        base_repo = "Qwen/Qwen-Image-2.1",
        account_id = ALICE,
        asset_repos = ("unsloth/Qwen-Image-2.1-FP8",),
    )
    backend.unload(expected_account = ALICE)
    assert "unsloth/Qwen-Image-2.1-FP8" in backend.draining_repo_ids()


def test_the_load_thread_releases_its_own_drain(backend, monkeypatch):
    """Nothing else knows when a prefetch returned, and during one there is no record to key on."""
    backend._loading = _LoadingState(
        repo_id = "org/model-GGUF", base_repo = "org/base", account_id = ALICE
    )
    token = backend._load_token
    with backend._load_cancel_lock:
        backend._draining_repos[token] = {"org/model-GGUF", "org/base"}

    monkeypatch.setattr(
        backend, "load_pipeline", lambda **kw: (_ for _ in ()).throw(RuntimeError("cancelled"))
    )
    monkeypatch.setattr("core.inference.diffusion.detect_family_for_pick", lambda *a, **k: None)
    backend._run_load(repo_id = "org/model-GGUF", _load_token = token)

    assert backend.draining_repo_ids() == (), "the thread returned and nothing released its repos"


def test_an_eject_with_no_load_in_flight_holds_nothing(backend):
    """Nothing is reading, so holding ids would only refuse deletes for no reason."""
    _resident(backend)
    backend.unload(expected_account = ALICE)
    assert backend._draining_repos == {} and backend.draining_repo_ids() == ()


def test_a_generation_turned_away_by_the_load_fence_sleeps_on_it(backend):
    """The retry waited on _teardown_drained, still SET in that window, so it spun against the
    very _lock the eject needed."""
    with backend._load_cancel_lock:
        backend._unload_waiters += 1
        backend._unload_fence_clear.clear()
    assert backend._teardown_drained.is_set(), "the window this test is about"

    polls = {"teardown": 0, "fence": 0}
    real_fence = backend._unload_fence_clear.wait
    real_teardown = backend._teardown_drained.wait

    def counted_teardown(timeout = None):
        polls["teardown"] += 1
        return real_teardown(timeout = timeout)

    def counted_fence(timeout = None):
        polls["fence"] += 1
        return real_fence(timeout = timeout)

    backend._teardown_drained.wait = counted_teardown
    backend._unload_fence_clear.wait = counted_fence

    def release():
        time.sleep(0.4)
        with backend._load_cancel_lock:
            backend._unload_waiters -= 1
            backend._unload_fence_clear.set()

    releaser = threading.Thread(target = release, daemon = True)
    releaser.start()
    with backend._generation_slot(threading.Event()):
        pass
    releaser.join(timeout = 10)

    # ~4 sleeps across the 0.4s fence. A spin does tens of thousands, on the wrong event.
    assert polls["fence"] >= 2, polls
    assert polls["teardown"] == 0, "the retry slept on the fence that was already clear"
