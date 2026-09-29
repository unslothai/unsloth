# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A login miss spends the same password-hashing work as a wrong password, so an
unauthenticated client cannot time /api/auth/login to learn which names exist."""

import hashlib

import pytest

from auth import hashing, storage
from studio.backend.tests.test_account_lifecycle import auth_env, login, matrix  # noqa: F401


@pytest.fixture
def pbkdf2_calls(monkeypatch):
    calls = []
    real = hashlib.pbkdf2_hmac

    def counting(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(hashing.hashlib, "pbkdf2_hmac", counting)
    return calls


def _rounds_for(client, pbkdf2_calls, username, password):
    del pbkdf2_calls[:]
    assert login(client, username, password).status_code == 401
    return len(pbkdf2_calls)


def test_unknown_inactive_and_pending_names_cost_a_verification(matrix, pbkdf2_calls):
    client, _, _ = matrix
    baseline = _rounds_for(client, pbkdf2_calls, "alice", "wrong-password")
    assert baseline == 1

    assert _rounds_for(client, pbkdf2_calls, "not-an-account", "wrong-password") == baseline

    bob = storage.get_user_record("bob")
    storage.set_account_active(bob["account_id"], False)
    assert _rounds_for(client, pbkdf2_calls, "bob", "bob-password") == baseline

    storage.issue_account_setup_code(username = "carol")
    assert _rounds_for(client, pbkdf2_calls, "carol", "not-the-setup-code") == baseline
