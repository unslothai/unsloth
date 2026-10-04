# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route policy for /api/settings/managed-provider-urls.

Writing it is owner-only and UI-session-only: an sk-unsloth key must not be able
to widen the installation's egress policy. Reading it is deliberately shared, so
a managed account can tell "ask your owner" from "this is not allowed here".
"""

from pathlib import Path
import sys
import types as _types

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.settings as settings
from utils.account_context import OWNER, AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture
def make_client(monkeypatch):
    state: dict = {"allowed": False}

    monkeypatch.setattr(
        settings, "get_managed_private_provider_urls_allowed", lambda: state["allowed"]
    )

    def _set(value):
        if not isinstance(value, bool):
            raise ValueError("Managed private provider URLs must be true or false.")
        state["allowed"] = value
        return value

    monkeypatch.setattr(settings, "set_managed_private_provider_urls_allowed", _set)

    def _build(account, via_api_key = False):
        app = FastAPI()
        app.include_router(settings.router)

        async def subject():
            token = bind_account(account)
            try:
                yield account.username
            finally:
                reset_account(token)

        app.dependency_overrides[settings.get_current_subject] = subject
        app.dependency_overrides[settings.authenticated_via_api_key] = lambda: via_api_key
        return TestClient(app, raise_server_exceptions = False)

    return _build, state


def test_owner_reads_and_writes(make_client):
    build, state = make_client
    with build(OWNER) as client:
        assert client.get("/managed-provider-urls").json()["allowed"] is False
        put = client.put("/managed-provider-urls", json = {"allowed": True})
        assert put.status_code == 200, put.text
        assert put.json()["allowed"] is True
        assert state["allowed"] is True
        assert client.get("/managed-provider-urls").json()["allowed"] is True


def test_response_reports_the_default_and_the_env_override(make_client, monkeypatch):
    from core.inference.providers import _BLOCK_PRIVATE_ENV

    build, _ = make_client
    monkeypatch.delenv(_BLOCK_PRIVATE_ENV, raising = False)
    with build(OWNER) as client:
        body = client.get("/managed-provider-urls").json()
    assert body["default_allowed"] is False
    assert body["locked_by_environment"] is False

    monkeypatch.setenv(_BLOCK_PRIVATE_ENV, "1")
    with build(OWNER) as client:
        assert client.get("/managed-provider-urls").json()["locked_by_environment"] is True


def test_managed_account_may_read(make_client):
    build, state = make_client
    state["allowed"] = True
    with build(ALICE) as client:
        response = client.get("/managed-provider-urls")
    assert response.status_code == 200, response.text
    assert response.json()["allowed"] is True


def test_managed_account_may_not_write(make_client):
    build, state = make_client
    with build(ALICE) as client:
        response = client.put("/managed-provider-urls", json = {"allowed": True})
    assert response.status_code == 403, response.text
    assert state["allowed"] is False


def test_api_key_session_may_not_write(make_client):
    """An sk-unsloth key is owner-authenticated but not a UI session."""
    build, state = make_client
    with build(OWNER, via_api_key = True) as client:
        response = client.put("/managed-provider-urls", json = {"allowed": True})
    assert response.status_code == 403, response.text
    assert state["allowed"] is False


@pytest.mark.parametrize("body", [{"allowed": "maybe"}, {"allowed": 1}, {"allowed": "true"}, {}])
def test_non_boolean_body_is_refused(make_client, body):
    build, state = make_client
    with build(OWNER) as client:
        response = client.put("/managed-provider-urls", json = body)
    assert response.status_code == 422, response.text
    assert state["allowed"] is False
