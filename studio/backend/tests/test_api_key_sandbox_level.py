# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Sandbox Low (python/terminal without the OS sandbox) is the Unsloth UI's to pick, like full access."""

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from routes import inference
from state.tool_policy import require_tool_access
from utils.account_context import OWNER, as_api_caller, bind_account, reset_account

_OPT_IN = "UNSLOTH_STUDIO_ALLOW_API_FULL_ACCESS"


@pytest.fixture(autouse = True)
def no_opt_in(monkeypatch):
    monkeypatch.delenv(_OPT_IN, raising = False)


def _under(account, *args, **kwargs):
    token = bind_account(account)
    try:
        require_tool_access(*args, **kwargs)
    finally:
        reset_account(token)


@pytest.mark.parametrize("mode", [None, "auto", "ask", "off"])
def test_sandbox_level_low_is_refused_for_api_callers(mode):
    with pytest.raises(HTTPException) as refused:
        _under(as_api_caller(OWNER), mode, sandbox_level = "low")
    assert refused.value.status_code == 400
    assert _OPT_IN in refused.value.detail
    _under(as_api_caller(OWNER), mode, sandbox_level = "high")
    _under(as_api_caller(OWNER), mode)


@pytest.mark.parametrize("level", ["low", "high", None])
def test_owner_ui_session_keeps_sandbox_low(level):
    _under(OWNER, "auto", sandbox_level = level)


def test_operator_opt_in_restores_api_sandbox_low(monkeypatch):
    monkeypatch.setenv(_OPT_IN, "1")
    _under(as_api_caller(OWNER), "auto", sandbox_level = "low")


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
    "/v1/messages": {
        "model": "m",
        "max_tokens": 8,
        "messages": [{"role": "user", "content": "hi"}],
    },
}


@pytest.mark.parametrize("path", sorted(_REQUESTS))
@pytest.mark.parametrize("api", [True, False], ids = ["api_key", "ui_session"])
def test_the_door_refuses_only_api_key_sandbox_low(monkeypatch, path, api):
    def backend(*_args, **_kwargs):
        raise HTTPException(status_code = 503, detail = "no backend in this test")

    monkeypatch.setattr(inference, "get_llama_cpp_backend", backend)
    monkeypatch.setattr(inference, "produce_openai_chat_completions", backend)
    body = {**_REQUESTS[path], "sandbox_level": "low"}
    with _client_for(as_api_caller(OWNER) if api else OWNER) as client:
        response = client.post(path, json = body)
    if api:
        assert response.status_code == 400, response.text
        assert _OPT_IN in response.json()["detail"]
    else:
        assert response.status_code == 503, response.text


@pytest.mark.parametrize("api", [True, False], ids = ["api_key", "ui_session"])
def test_durable_chat_runs_refuse_api_key_sandbox_low_before_queueing(api):
    from routes import chat_generation_runs as runs

    payload = runs.CreateChatGenerationRun(
        runId = "r1",
        threadId = "t1",
        userMessageId = "u1",
        assistantMessageId = "a1",
        requestPayload = {"messages": [{"role": "user", "content": "hi"}], "sandbox_level": "low"},
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
