# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seams that join the account domains: admission at the HTTP door, the desktop
marker, schema initialisation for the Hub scan folders, call-time output roots and
account retirement draining the services that hold an account's paths open."""

import asyncio
import inspect
import secrets

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth import authentication, policy, storage
from auth.authentication import get_current_subject
from hub.storage import scan_folders
from routes import inference
from utils import keyless_api_access as keyless
from utils.account_context import (
    OWNER,
    AccountContext,
    bind_account,
    current_account_id,
    reset_account,
    run_as,
)
from utils.models import model_config
from utils.paths import outputs_root

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture
def multi_user(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)


def _isolated_auth_db(
    tmp_path,
    monkeypatch,
    *,
    owner_must_change = False,
):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    keyless._reset_scope_cache()
    storage.create_initial_user(
        "unsloth",
        "owner-password",
        secrets.token_urlsafe(32),
        must_change_password = owner_must_change,
    )


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    _isolated_auth_db(tmp_path, monkeypatch)
    yield
    policy.invalidate_account_cache()
    keyless._reset_scope_cache()


def _client_for(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.include_router(inference.router, prefix = "/v1")
    return TestClient(app)


def _credentials(token):
    return authentication.HTTPAuthorizationCredentials(scheme = "Bearer", credentials = token)


_FULL_ACCESS = "Full access is unavailable while more than one account exists."

_REQUESTS = {
    "/v1/chat/completions": {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    "/v1/responses": {"model": "m", "input": "hi"},
    "/v1/messages": {
        "model": "m",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    },
}


@pytest.mark.parametrize("path", sorted(_REQUESTS))
@pytest.mark.parametrize(
    "flags",
    [
        {"bypass_permissions": True},
        {"permission_mode": "full"},
        {"disable_sandbox": True},
    ],
)
@pytest.mark.parametrize("account", [OWNER, ALICE], ids = ["owner", "alice"])
def test_full_access_is_refused_at_the_door_in_multi_mode(
    multi_user, monkeypatch, path, flags, account
):
    """The refusal arrives as a 400 before any backend is consulted or a stream opens."""

    def never(*_args, **_kwargs):
        raise AssertionError("the request reached the backend")

    monkeypatch.setattr(inference, "get_llama_cpp_backend", never)
    with _client_for(account) as client:
        response = client.post(path, json = {**_REQUESTS[path], **flags})
    assert response.status_code == 400, response.text
    assert response.json()["detail"] == _FULL_ACCESS


@pytest.mark.parametrize("path", sorted(_REQUESTS))
def test_single_account_full_access_passes_the_door(monkeypatch, path):
    """One account: the admission is inert and the handler proceeds as before."""
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)

    def backend():
        raise HTTPException(status_code = 503, detail = "no backend in this test")

    monkeypatch.setattr(inference, "get_llama_cpp_backend", backend)
    monkeypatch.setattr(inference, "produce_openai_chat_completions", lambda *a, **k: backend())
    with _client_for(OWNER) as client:
        response = client.post(path, json = {**_REQUESTS[path], "bypass_permissions": True})
    assert response.status_code != 400 or response.json().get("detail") != _FULL_ACCESS


def test_admission_runs_after_the_docstring_and_reads_every_flag():
    for handler in (
        inference.openai_chat_completions,
        inference.openai_responses,
        inference.anthropic_messages,
    ):
        body = inspect.getsource(handler)
        assert "_admit_tool_access(payload)" in body, handler.__name__
    for handler in (inference.openai_responses, inference.anthropic_messages):
        assert handler.__doc__, handler.__name__
    source = inspect.getsource(inference._admit_tool_access)
    for flag in ("permission_mode", "bypass_permissions", "disable_sandbox"):
        assert flag in source


def test_desktop_marker_only_authenticates_the_owner(auth_db):
    storage.create_initial_user("alice", "alice-password", secrets.token_urlsafe(32))
    owner_secret = storage.get_user_record("unsloth")["jwt_secret"]
    alice_secret = storage.get_user_record("alice")["jwt_secret"]
    owner_token = authentication.create_access_token(
        subject = "unsloth", desktop = True, secret = owner_secret
    )
    alice_token = authentication.create_access_token(
        subject = "alice", desktop = True, secret = alice_secret
    )
    assert authentication.is_desktop_access_token(owner_token)
    assert not authentication.is_desktop_access_token(alice_token)


def test_desktop_marker_does_not_bypass_a_managed_password_change(auth_db):
    storage.create_initial_user(
        "alice",
        "alice-password",
        secrets.token_urlsafe(32),
        must_change_password = True,
    )
    alice_secret = storage.get_user_record("alice")["jwt_secret"]
    token = authentication.create_access_token(subject = "alice", desktop = True, secret = alice_secret)
    with pytest.raises(HTTPException) as refused:
        asyncio.run(authentication.get_current_subject(_credentials(token)))
    assert refused.value.status_code == 403
    assert refused.value.detail == "Password change required"


def test_desktop_marker_still_bypasses_the_owner_password_change(tmp_path, monkeypatch):
    _isolated_auth_db(tmp_path, monkeypatch, owner_must_change = True)
    owner_secret = storage.get_user_record("unsloth")["jwt_secret"]
    token = authentication.create_access_token(subject = "unsloth", desktop = True, secret = owner_secret)
    assert asyncio.run(authentication.get_current_subject(_credentials(token))) == "unsloth"
    policy.invalidate_account_cache()


def test_refresh_drops_the_desktop_marker_for_a_managed_account(auth_db):
    from routes import auth as auth_routes

    storage.create_initial_user(
        "alice",
        "alice-password",
        secrets.token_urlsafe(32),
        must_change_password = True,
    )
    alice_secret = storage.get_user_record("alice")["jwt_secret"]
    refresh = authentication.create_refresh_token(
        subject = "alice", desktop = True, secret = alice_secret
    )
    app = FastAPI()
    app.include_router(auth_routes.router, prefix = "/api/auth")
    with TestClient(app) as client:
        response = client.post("/api/auth/refresh", json = {"refresh_token": refresh})
    assert response.status_code == 200, response.text
    assert response.json()["must_change_password"] is True
    assert not authentication.is_desktop_access_token(response.json()["access_token"])


def test_scan_folders_schema_is_initialised_for_each_account(tmp_path, monkeypatch, multi_user):
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(scan_folders, "_schema_ready", set())
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    folder = tmp_path / "shared-folder"
    folder.mkdir()
    assert run_as(OWNER, scan_folders.list_scan_folders) == []
    run_as(OWNER, scan_folders.add_scan_folder, str(folder))
    # Alice's studio.db is new: the owner's schema pass must not be mistaken for hers.
    assert run_as(ALICE, scan_folders.list_scan_folders) == []
    assert [row["path"] for row in run_as(OWNER, scan_folders.list_scan_folders)] == [
        str(folder.resolve())
    ]
    assert len(scan_folders._schema_ready) == 2


def test_output_scans_resolve_the_acting_account_root(tmp_path, monkeypatch, multi_user):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    for function in (model_config.scan_trained_models, model_config.scan_exported_models):
        parameter = next(iter(inspect.signature(function).parameters.values()))
        assert parameter.default is None, function.__name__
    seen = []
    monkeypatch.setattr(
        model_config, "resolve_output_dir", lambda value: seen.append(value) or tmp_path / "missing"
    )
    run_as(OWNER, model_config.scan_trained_models)
    run_as(ALICE, model_config.scan_trained_models)
    assert seen == [str(run_as(OWNER, outputs_root)), str(run_as(ALICE, outputs_root))]
    assert seen[0] != seen[1]


def test_retirement_drains_jobs_and_sessions_under_the_account(tmp_path, monkeypatch, multi_user):
    from core.inference import mcp_client
    from core.training import account_jobs
    from routes import accounts

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    calls = []
    monkeypatch.setattr(
        account_jobs,
        "retire_account_jobs",
        lambda account: calls.append(("jobs", account.account_id)),
    )
    monkeypatch.setattr(
        mcp_client,
        "close_mcp_sessions",
        lambda *a, **k: calls.append(("mcp", current_account_id())),
    )
    monkeypatch.setattr(
        mcp_client,
        "invalidate_tool_cache",
        lambda *a, **k: calls.append(("tools", current_account_id())),
    )
    accounts.retire_account_roots(ALICE)
    assert calls == [
        ("jobs", ALICE.account_id),
        ("mcp", ALICE.account_id),
        ("tools", ALICE.account_id),
    ]
    with pytest.raises(ValueError):
        accounts.retire_account_roots(OWNER)
