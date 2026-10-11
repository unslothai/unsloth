# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unauthenticated client must not learn account existence from lockout scope."""

from studio.backend.tests.test_account_lifecycle import auth_env, login
from auth import storage


def test_login_throttling_does_not_disclose_account_existence(auth_env):
    client, auth, _ = auth_env
    storage.issue_account_setup_code(username = "alice")
    observed = {}
    for candidate in ("alice", "not-an-account"):
        # Independent attacker sessions, each beginning with empty rate limits.
        auth._LOGIN_BUCKETS.clear()
        auth._LOGIN_IP_BUCKETS.clear()
        for _ in range(auth._LOGIN_MAX_FAILS):
            assert login(client, candidate, "wrong-password").status_code == 401
        # No correct password or privileged request is needed for the oracle.
        response = login(client, "unsloth", "also-wrong")
        observed[candidate] = response.status_code
    print(f"Unauthenticated probe statuses: {observed}")
    assert observed["alice"] == observed["not-an-account"], observed
