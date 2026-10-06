# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A generation that registers after its account was disabled starts cancelled.

Deactivation cancels the generations registered at that instant. A request already past
authentication and waiting on a gate registers afterwards, so the fence set alongside the sweep
has to reach it too, and reactivation lifts it."""

import threading

from auth import storage
from state import active_generations
from studio.backend.tests.test_account_lifecycle import auth_env, headers, matrix  # noqa: F401
from utils.account_context import run_as


def _register(account):
    event = threading.Event()
    with run_as(account, active_generations.ActiveGeneration, event):
        return event.is_set()


def test_late_registration_after_deactivation_is_cancelled(matrix):
    client, _, _ = matrix
    alice = storage.get_account("alice")
    assert _register(alice) is False
    url = f"/api/accounts/{alice.account_id}"
    assert client.patch(url, headers = headers(), json = {"is_active": False}).status_code == 200
    print(f"fenced after deactivation: {_register(alice)}")
    assert _register(alice) is True
    bob = storage.get_account("bob")
    assert _register(bob) is False
    assert client.patch(url, headers = headers(), json = {"is_active": True}).status_code == 200
    assert _register(alice) is False


def test_retirement_fences_late_registration(matrix):
    client, _, _ = matrix
    alice = storage.get_account("alice")
    assert client.delete(f"/api/accounts/{alice.account_id}", headers = headers()).status_code == 204
    assert _register(alice) is True
