# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from auth.oidc_handoff import OIDCSessionHandoffManager


def test_limits_must_be_positive():
    for kwargs in ({"max_age_seconds": 0}, {"max_handoffs": 0}):
        try:
            OIDCSessionHandoffManager(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid handoff limits were accepted")


def test_session_handoff_is_opaque_single_use_and_expires():
    now = [10.0]
    manager = OIDCSessionHandoffManager(max_age_seconds = 60, clock = lambda: now[0])
    code = manager.create(access_token = "access", refresh_token = "refresh", account_id = "a1")

    assert "access" not in code and "refresh" not in code
    handoff = manager.consume(code)
    assert handoff is not None
    assert handoff.access_token == "access"
    assert handoff.refresh_token == "refresh"
    assert handoff.account_id == "a1"
    assert manager.consume(code) is None

    expired = manager.create(access_token = "a2", refresh_token = "r2", account_id = "a2")
    now[0] += 61
    assert manager.consume(expired) is None
