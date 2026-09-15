# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Who may eject, and in which order the claims are read.

Three claims can speak for the backend at once: a PUBLISHED resident, an admitted BACKGROUND load,
and a bare INVOCATION record for a caller still in preflight (a CPU Diffusers load claims no GPU and
publishes nothing, so for its whole construction it is only that record). They disagree, so the order
matters:

  * the resident wins, because require_resident_control has already authorized the caller against the
    model that is actually loaded -- a newcomer queueing a replacement over it must not cost the owner
    the right to eject its own model;
  * with no resident, the admitted background load decides;
  * with neither, the invocation record decides, which is what keeps a second account from cancelling
    someone else's pending CPU load.
"""

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
    # The eject path's own authorization is exercised here, not the hub's: require_resident_control
    # answers "may this account touch the RESIDENT", and every case below has already decided that.
    monkeypatch.setattr(account_access, "require_resident_control", lambda *a, **k: None)
    # Stands in for the real teardown, which frees weights this test never allocated. It still drops
    # the resident, so status() afterwards reads what a real eject would leave.
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
    # Alice is merely downloading. Bob owns what is loaded, and require_resident_control said so.
    assert backend.unload(expected_account = BOB)["loaded"] is False
    assert backend._unload_waiters == 0


def test_a_foreign_account_cannot_cancel_a_pending_background_load(backend):
    _background_load(backend, ALICE)
    with pytest.raises(account_access.GpuBusyForAnotherAccountError):
        backend.unload(expected_account = BOB)
    # Refused before the fence moved: the victim's load is untouched.
    assert backend._unload_waiters == 0 and not backend._cancel_event.is_set()
    assert backend._loading is not None


def test_a_foreign_account_cannot_cancel_a_pending_cpu_load(backend):
    # No GPU claim and no published resident: the invocation record is the only owner there is.
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
    """_wait_for_pending_unloads() and the registration lock are two steps. An eject landing in
    between bumps the epoch too, so a naive re-read of _load_token adopts the eject's OWN token,
    passes the cancellation check, and can then win _lock ahead of the eject, publish a pipeline and
    have the teardown destroy it -- all after the request was accepted. The fence has to be read
    under the lock that registers."""
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
        backend._teardown_drained.set()

    releaser = threading.Thread(target = release, daemon = True)
    releaser.start()
    token = admitted(backend)
    releaser.join(timeout = 10)

    assert calls["waits"] == 2, "the load registered without waiting the racing eject out"
    assert token == backend._load_token
    assert backend._load_accounts == {}
