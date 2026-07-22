# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from models.auth import ChangePasswordRequest  # noqa: E402

# Load routes/auth.py directly so collection does not execute routes/__init__.py,
# which pulls in the heavy training/models/inference routers.
_route_path = _BACKEND_ROOT / "routes" / "auth.py"
_spec = importlib.util.spec_from_file_location("_change_password_route", _route_path)
assert _spec is not None and _spec.loader is not None
auth_routes = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(auth_routes)


@pytest.fixture
def _user(monkeypatch):
    monkeypatch.setattr(
        auth_routes.storage,
        "get_user_and_secret",
        lambda username: ("salt", "hash", "jwt-secret", False),
    )
    monkeypatch.setattr(
        auth_routes.hashing,
        "verify_password",
        lambda password, salt, pwd_hash: password == "bootstrap-pw",
    )


def _change(new_password):
    payload = ChangePasswordRequest(
        current_password = "bootstrap-pw",
        new_password = new_password,
    )
    return asyncio.run(auth_routes.change_password(payload, None, "unsloth"))


def test_rejects_whitespace_only_password(_user):
    with pytest.raises(HTTPException) as excinfo:
        _change(" " * 8)
    assert excinfo.value.status_code == 400
    assert "spaces" in excinfo.value.detail


def test_rejects_tabs_and_spaces_password(_user):
    with pytest.raises(HTTPException) as excinfo:
        _change(" \t \t \t \t ")
    assert excinfo.value.status_code == 400


def test_rejects_password_containing_spaces(_user):
    with pytest.raises(HTTPException) as excinfo:
        _change("correct horse battery")
    assert excinfo.value.status_code == 400
    assert "spaces" in excinfo.value.detail


def test_allows_password_without_spaces(_user, monkeypatch):
    monkeypatch.setattr(auth_routes.storage, "update_password", lambda *args, **kwargs: True)
    monkeypatch.setattr(auth_routes, "create_access_token", lambda subject: "at")
    monkeypatch.setattr(auth_routes, "create_refresh_token", lambda subject: "rt")
    token = _change("correct-horse-battery")
    assert token.access_token == "at"
    assert token.must_change_password is False
