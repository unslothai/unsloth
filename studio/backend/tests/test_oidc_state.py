# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import base64
import hashlib

from auth.oidc_state import OIDCStateManager


def test_attempt_contains_random_state_nonce_and_valid_s256_pkce():
    manager = OIDCStateManager()

    first = manager.create()
    second = manager.create()

    assert first.state != second.state
    assert first.nonce != second.nonce
    assert 43 <= len(first.code_verifier) <= 128
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(first.code_verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    assert first.code_challenge == expected


def test_state_is_single_use_and_returns_nonce_and_verifier():
    manager = OIDCStateManager()
    parameters = manager.create()

    attempt = manager.consume(parameters.state)

    assert attempt is not None
    assert attempt.nonce == parameters.nonce
    assert attempt.code_verifier == parameters.code_verifier
    assert manager.consume(parameters.state) is None


def test_expired_state_is_rejected_and_pruned():
    now = [50.0]
    manager = OIDCStateManager(max_age_seconds = 300, clock = lambda: now[0])
    parameters = manager.create()

    now[0] += 301

    assert manager.consume(parameters.state) is None
    assert len(manager) == 0


def test_state_store_is_bounded_by_evicting_oldest_attempt():
    now = [1.0]
    manager = OIDCStateManager(max_attempts = 2, clock = lambda: now[0])
    oldest = manager.create()
    now[0] += 1
    middle = manager.create()
    now[0] += 1
    newest = manager.create()

    assert len(manager) == 2
    assert manager.consume(oldest.state) is None
    assert manager.consume(middle.state) is not None
    assert manager.consume(newest.state) is not None


def test_empty_or_unknown_state_is_rejected():
    manager = OIDCStateManager()

    assert manager.consume("") is None
    assert manager.consume("unknown") is None
