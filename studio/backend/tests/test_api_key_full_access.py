# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Full access (no sandbox) is the Unsloth UI's to grant: an API key or keyless caller is refused unless the operator opts in."""

import asyncio
import secrets

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth import authentication, policy, storage
from auth.authentication import get_current_subject
from routes import inference
from state.tool_policy import require_tool_access
from utils.account_context import OWNER, as_api_caller, bind_account, current_account, reset_account

_OPT_IN = "UNSLOTH_STUDIO_ALLOW_API_FULL_ACCESS"
_FLAGS = [{"bypass_permissions": True}, {"permission_mode": "full"}, {"disable_sandbox": True}]


@pytest.fixture(autouse = True)
def no_opt_in(monkeypatch):
    monkeypatch.delenv(_OPT_IN, raising = False)


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    policy.invalidate_account_cache()
    storage.create_initial_user("unsloth", "owner-password", secrets.token_urlsafe(32))
    yield storage
    policy.invalidate_account_cache()


def _gate_under(credentials):
    """Authenticate like a request, then ask for full access in that same context."""
    from utils.account_context import _current  # noqa: PLC0415

    before = _current.get()

    async def go():
        await authentication._get_current_credential(credentials, allow_password_change = False)
        bound = current_account()
        try:
            require_tool_access(None, bypass_permissions = True)
        except HTTPException as exc:
            return bound, exc
        return bound, None

    try:
        return asyncio.run(go())
    finally:
        _current.set(before)


def _bearer(token):
    return authentication.HTTPAuthorizationCredentials(scheme = "Bearer", credentials = token)


def test_owner_api_key_is_refused_full_access(auth_db):
    raw = storage.create_api_key("unsloth", name = "client")[0]
    bound, refused = _gate_under(_bearer(raw))
    assert bound == OWNER and bound.via_api_key
    assert refused is not None and refused.status_code == 400
    assert _OPT_IN in refused.detail


def test_owner_ui_session_keeps_full_access(auth_db):
    _salt, _hash, secret, _must_change = storage.get_user_and_secret("unsloth")
    token = authentication.create_access_token("unsloth", secret = secret)
    bound, refused = _gate_under(_bearer(token))
    assert bound == OWNER and not bound.via_api_key
    assert refused is None


def test_operator_opt_in_restores_api_key_full_access(auth_db, monkeypatch):
    monkeypatch.setenv(_OPT_IN, "1")
    raw = storage.create_api_key("unsloth", name = "client")[0]
    _bound, refused = _gate_under(_bearer(raw))
    assert refused is None


def test_keyless_caller_counts_as_an_api_caller(auth_db):
    keyless = authentication.HTTPAuthorizationCredentials(
        scheme = authentication.KEYLESS_SCHEME, credentials = ""
    )
    bound, refused = _gate_under(keyless)
    assert bound == OWNER and bound.via_api_key
    assert refused is not None and refused.status_code == 400


@pytest.mark.parametrize("mode", [None, "auto", "ask", "off"])
def test_sandboxed_modes_stay_open_to_api_keys(mode):
    token = bind_account(as_api_caller(OWNER))
    try:
        require_tool_access(mode)
    finally:
        reset_account(token)


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
@pytest.mark.parametrize("flags", _FLAGS)
@pytest.mark.parametrize("api", [True, False], ids = ["api_key", "ui_session"])
def test_the_door_refuses_only_api_key_full_access(monkeypatch, path, flags, api):
    def backend(*_args, **_kwargs):
        raise HTTPException(status_code = 503, detail = "no backend in this test")

    monkeypatch.setattr(inference, "get_llama_cpp_backend", backend)
    monkeypatch.setattr(inference, "produce_openai_chat_completions", backend)
    with _client_for(as_api_caller(OWNER) if api else OWNER) as client:
        response = client.post(path, json = {**_REQUESTS[path], **flags})
    if api:
        assert response.status_code == 400, response.text
        assert _OPT_IN in response.json()["detail"]
    else:
        assert response.status_code == 503, response.text


@pytest.mark.parametrize("flags", _FLAGS[:2])
@pytest.mark.parametrize("api", [True, False], ids = ["api_key", "ui_session"])
def test_durable_chat_runs_refuse_api_key_full_access_before_queueing(flags, api):
    from routes import chat_generation_runs as runs

    payload = runs.CreateChatGenerationRun(
        runId = "r1",
        threadId = "t1",
        userMessageId = "u1",
        assistantMessageId = "a1",
        requestPayload = {"messages": [{"role": "user", "content": "hi"}], **flags},
    )
    token = bind_account(as_api_caller(OWNER) if api else OWNER)
    try:
        if api:
            with pytest.raises(HTTPException) as refused:
                runs._sanitize_request(payload)
            assert refused.value.status_code == 400 and _OPT_IN in refused.value.detail
        else:
            runs._sanitize_request(payload)
    finally:
        reset_account(token)
