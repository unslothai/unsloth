# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Generations registered after deactivation start cancelled; reactivation lifts that fence."""

import threading
from types import SimpleNamespace

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


def test_deactivation_cancels_a_scheduled_run_still_waiting_for_the_model(matrix):
    from core.scheduled_tasks.scheduler import ScheduledTaskScheduler, _ActiveRun

    client, _, _ = matrix
    alice, bob = storage.get_account("alice"), storage.get_account("bob")
    alices_scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    bobs_scheduler = ScheduledTaskScheduler(SimpleNamespace(state = SimpleNamespace()))
    alices_run, bobs_run = threading.Event(), threading.Event()
    alices_scheduler._active = _ActiveRun(alice.account_id, "task-1", "run-1", alices_run)
    bobs_scheduler._active = _ActiveRun(bob.account_id, "task-2", "run-2", bobs_run)

    url = f"/api/accounts/{alice.account_id}"
    assert client.patch(url, headers = headers(), json = {"is_active": False}).status_code == 200
    assert alices_run.is_set()
    assert not bobs_run.is_set()
