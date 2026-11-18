# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A spent refresh token rotates again within REFRESH_REUSE_GRACE_SECONDS, so a lost rotation answer
no longer signs every tab out; logout and password changes still end it at once."""

import secrets
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.auth as auth_routes
from auth import policy, storage
from auth.authentication import new_refresh_token_value


@pytest.fixture
def owner(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth" / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    storage.create_initial_user(
        username = storage.DEFAULT_ADMIN_USERNAME,
        password = "human-password-123",
        jwt_secret = secrets.token_urlsafe(64),
        must_change_password = False,
    )
    yield storage.DEFAULT_ADMIN_USERNAME
    policy.invalidate_account_cache()


@pytest.fixture
def client(owner):
    app = FastAPI()
    app.include_router(auth_routes.router, prefix = "/api/auth")
    with TestClient(app) as http:
        yield http


def _issue(username):
    token, expires_at = new_refresh_token_value()
    storage.save_refresh_token(token, username, expires_at)
    return token


def _refresh(http, token):
    return http.post("/api/auth/refresh", json = {"refresh_token": token})


def _age_reuse_rows(seconds):
    conn = storage.get_connection()
    try:
        old = (datetime.now(timezone.utc) - timedelta(seconds = seconds)).isoformat()
        conn.execute("UPDATE refresh_token_reuse SET consumed_at = ?", (old,))
        conn.commit()
    finally:
        conn.close()


def test_a_lost_rotation_answer_does_not_strand_the_session(client, owner):
    spent = _issue(owner)
    first = _refresh(client, spent)
    assert first.status_code == 200
    # The browser never saw `first`, so it retries with the token it still holds.
    retry = _refresh(client, spent)
    assert retry.status_code == 200, retry.text
    pair = retry.json()
    assert pair["refresh_token"] not in (spent, first.json()["refresh_token"])
    # The pair handed out on the retry is a normal, live session.
    assert _refresh(client, pair["refresh_token"]).status_code == 200


def test_two_tabs_racing_on_one_token_both_rotate(client, owner):
    spent = _issue(owner)
    answers = [_refresh(client, spent) for _ in range(2)]
    assert [a.status_code for a in answers] == [200, 200]
    for answer in answers:
        assert _refresh(client, answer.json()["refresh_token"]).status_code == 200


def test_the_window_ends(client, owner):
    spent = _issue(owner)
    assert _refresh(client, spent).status_code == 200
    _age_reuse_rows(storage.REFRESH_REUSE_GRACE_SECONDS + 1)
    assert _refresh(client, spent).status_code == 401


def test_logout_inside_the_window_still_ends_the_session(client, owner):
    spent = _issue(owner)
    assert _refresh(client, spent).status_code == 200
    storage.revoke_user_refresh_tokens(owner)
    assert _refresh(client, spent).status_code == 401


def test_a_password_change_inside_the_window_still_ends_the_session(client, owner):
    spent = _issue(owner)
    assert _refresh(client, spent).status_code == 200
    storage.update_password(owner, "another-password-456", revoke_refresh_tokens = True)
    assert _refresh(client, spent).status_code == 401


def test_an_unknown_token_is_still_rejected(client, owner):
    assert _refresh(client, secrets.token_urlsafe(48)).status_code == 401


def test_a_spent_token_is_gone_from_the_table_older_builds_read(client, owner):
    # Downgrade safety: an older build only reads refresh_tokens, so it must not find the spent token.
    spent = _issue(owner)
    assert _refresh(client, spent).status_code == 200
    assert storage.verify_refresh_token(spent) is None


def test_strict_consume_stays_single_use(owner):
    spent = _issue(owner)
    assert storage.consume_refresh_token(spent) is not None
    assert storage.consume_refresh_token(spent) is None
