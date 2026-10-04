# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Disabling an account cancels its chat loads and fences the ones still in preflight."""

import threading
import uuid

import pytest
from fastapi import HTTPException

from auth import storage
from hub.services.models import account_access as access
from routes import inference
from studio.backend.tests.test_account_lifecycle import auth_env, headers, matrix  # noqa: F401
from utils.account_context import run_as


def _attempt(subject):
    return inference._ScopedLoadAttempt(
        token = uuid.uuid4().hex,
        request_id = None,
        model_path = "org/model",
        subject = subject,
        cancel_event = threading.Event(),
        cancel_complete = threading.Event(),
    )


def test_deactivation_sweeps_loads_and_fences_late_ones(matrix, monkeypatch):
    client, _, _ = matrix
    alice, bob = storage.get_account("alice"), storage.get_account("bob")
    pending, other = _attempt(alice.account_id), _attempt(bob.account_id)
    monkeypatch.setattr(inference, "_pending_load_attempts", {a.token: a for a in (pending, other)})
    monkeypatch.setattr(inference, "_scoped_load_attempts", {})
    monkeypatch.setattr(inference, "_running_load_attempt", None)
    run_as(alice, access.require_live_account)
    url = f"/api/accounts/{alice.account_id}"
    assert client.patch(url, headers = headers(), json = {"is_active": False}).status_code == 200
    assert pending.cancel_event.is_set() and not other.cancel_event.is_set()
    with pytest.raises(HTTPException) as exc:
        run_as(alice, access.require_live_account)
    assert (exc.value.status_code, exc.value.detail) == (403, "Account is disabled")
    run_as(bob, access.require_live_account)
    assert client.patch(url, headers = headers(), json = {"is_active": True}).status_code == 200
    run_as(alice, access.require_live_account)
