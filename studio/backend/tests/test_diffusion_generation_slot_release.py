# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An admitted generation that fails before its body runs must still hand the generate lock back.

The account slot import used to sit between admission and the try that releases the lock, so an
import failure left the lock held and every later generate, load and unload waited on it forever."""

from __future__ import annotations

import sys
import threading

import pytest

from core.inference.diffusion import DiffusionBackend

pytestmark = pytest.mark.timeout(60)

_SLOT_MODULE = "hub.services.models.account_access"


def test_a_failing_slot_import_releases_the_generate_lock(monkeypatch):
    backend = DiffusionBackend()
    monkeypatch.setitem(sys.modules, _SLOT_MODULE, None)  # the import now raises ImportError
    with pytest.raises(ImportError):
        with backend._generation_slot(threading.Event()):
            pytest.fail("the body must not run when the slot cannot be entered")
    assert not backend._generate_lock.locked()
    assert not backend._generation_owns_slot
    assert backend._active_generate_cancel is None
    assert not backend._queued_generate_cancels


def test_unload_after_a_failed_generation_slot_does_not_wait_forever(monkeypatch):
    backend = DiffusionBackend()
    monkeypatch.setitem(sys.modules, _SLOT_MODULE, None)
    with pytest.raises(ImportError):
        with backend._generation_slot(threading.Event()):
            pass
    monkeypatch.delitem(sys.modules, _SLOT_MODULE)
    done = threading.Event()
    # A daemon thread, so a regression fails this test instead of hanging the session.
    worker = threading.Thread(target = lambda: (backend.unload(), done.set()), daemon = True)
    worker.start()
    assert done.wait(timeout = 30), "unload blocked on a generate lock a failed generation kept"
