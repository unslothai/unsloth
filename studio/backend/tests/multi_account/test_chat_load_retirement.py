# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Retiring an account cancels its chat loads; a retired account never becomes a resident sharer."""

import inspect
import threading
import uuid

import pytest
from fastapi import HTTPException

from auth import policy
from core.training import account_jobs as jobs
from hub.services.models import account_access as access
from routes import accounts, inference
from utils.account_context import AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


def _attempt(subject, request_id = None):
    return inference._ScopedLoadAttempt(
        token = uuid.uuid4().hex,
        request_id = request_id,
        model_path = "org/model",
        subject = subject,
        cancel_event = threading.Event(),
        cancel_complete = threading.Event(),
    )


def test_retirement_cancels_only_the_accounts_loads(monkeypatch):
    pending, scoped, running = (
        _attempt(ALICE.account_id),
        _attempt(ALICE.account_id, "r1"),
        _attempt(ALICE.account_id),
    )
    bob = _attempt(BOB.account_id)
    monkeypatch.setattr(inference, "_pending_load_attempts", {a.token: a for a in (pending, bob)})
    monkeypatch.setattr(inference, "_scoped_load_attempts", {(ALICE.account_id, "r1"): scoped})
    monkeypatch.setattr(inference, "_running_load_attempt", running)
    assert inference.retire_account_loads(ALICE.account_id) == 3
    for attempt in (pending, scoped, running):
        assert attempt.cancel_event.is_set() and attempt.cancel_complete.is_set()
    assert not bob.cancel_event.is_set()
    assert "retire_account_loads(account.account_id)" in inspect.getsource(
        accounts.retire_account_roots
    )


def test_retired_account_neither_publishes_nor_joins_a_resident(monkeypatch):
    monkeypatch.setattr(policy, "installation_has_managed_accounts", lambda: True)
    monkeypatch.setattr(jobs, "_retired", {ALICE.account_id})
    monkeypatch.setattr(access, "_resident_sharers", {})
    monkeypatch.setattr(access, "_resident_accounts", {})
    monkeypatch.setattr(access, "_uncommitted_resident", {})
    for call in (
        lambda: access.publish_resident("chat", "org/model"),
        lambda: access.join_resident("chat"),
    ):
        with pytest.raises(HTTPException) as exc:
            run_as(ALICE, call)
        assert exc.value.status_code == 403
    assert access._resident_sharers == {} and access._uncommitted_resident == {}
    run_as(BOB, access.join_resident, "chat")
    assert access._resident_sharers == {"chat": {BOB.account_id}}
