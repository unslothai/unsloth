# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The shutdown deadline has to reach the browser: it arms only for an exposed web
UI (`--secure`, external bind), and those launches run detached or tunneled, where
nothing reads stderr."""

import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.bootstrap_timeout import (
    clear_bootstrap_deadline,
    record_bootstrap_deadline,
)
from routes import auth as auth_routes


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(auth_routes.router, prefix = "/api/auth")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse = True)
def _no_leaked_deadline():
    clear_bootstrap_deadline()
    yield
    clear_bootstrap_deadline()


def _status(client):
    response = client.get("/api/auth/status")
    assert response.status_code == 200
    return response.json()


def _password_state(monkeypatch, *, requires_change: bool):
    """Pin what the route reads. ``is_initialized`` goes with it: the route answers
    True for an uninitialized instance without consulting the other call, and a fresh
    checkout (CI) has no users."""
    monkeypatch.setattr(auth_routes.storage, "is_initialized", lambda: True)
    monkeypatch.setattr(
        auth_routes.storage, "requires_password_change", lambda _username: requires_change
    )


class TestTheFieldIsPresent:
    def test_absent_deadline_is_null_not_missing(self, client):
        """A client that always reads the key must not find undefined there."""
        body = _status(client)
        assert "bootstrap_deadline_seconds" in body
        assert body["bootstrap_deadline_seconds"] is None

    def test_an_armed_deadline_is_reported(self, client, monkeypatch):
        _password_state(monkeypatch, requires_change = True)
        record_bootstrap_deadline(3600)
        remaining = _status(client)["bootstrap_deadline_seconds"]
        assert remaining is not None and 3590 <= remaining <= 3600

    def test_it_counts_down_between_calls(self, client, monkeypatch):
        _password_state(monkeypatch, requires_change = True)
        record_bootstrap_deadline(3600)
        first = _status(client)["bootstrap_deadline_seconds"]
        import auth.bootstrap_timeout as bt

        bt._deadline_at = bt._deadline_at - 120
        second = _status(client)["bootstrap_deadline_seconds"]
        assert second < first


class TestItIsNotReportedWhenItCannotFire:
    """A countdown after the password changed would promise a shutdown the handler declines."""

    def test_a_changed_password_reports_no_deadline(self, client, monkeypatch):
        record_bootstrap_deadline(3600)
        _password_state(monkeypatch, requires_change = False)
        body = _status(client)
        assert body["requires_password_change"] is False
        assert body["bootstrap_deadline_seconds"] is None

    def test_the_timer_being_armed_is_not_enough_on_its_own(self, client, monkeypatch):
        """Arming is never undone on a password change, so the timer alone cannot answer this."""
        record_bootstrap_deadline(60)
        _password_state(monkeypatch, requires_change = False)
        assert _status(client)["bootstrap_deadline_seconds"] is None
        _password_state(monkeypatch, requires_change = True)
        assert _status(client)["bootstrap_deadline_seconds"] is not None


class TestTheNumberIsUsable:
    def test_an_expired_deadline_reads_zero_not_negative(self, client, monkeypatch):
        """A negative would print as "shuts down in -12 minutes"."""
        _password_state(monkeypatch, requires_change = True)
        record_bootstrap_deadline(1)
        import auth.bootstrap_timeout as bt

        bt._deadline_at = time.monotonic() - 5
        assert _status(client)["bootstrap_deadline_seconds"] == 0

    def test_the_endpoint_stays_anonymous(self, client, monkeypatch):
        """The deadline is implied by requires_password_change, already returned here."""
        _password_state(monkeypatch, requires_change = True)
        record_bootstrap_deadline(3600)
        response = client.get("/api/auth/status")
        assert response.status_code == 200
        assert response.json()["bootstrap_deadline_seconds"] is not None
