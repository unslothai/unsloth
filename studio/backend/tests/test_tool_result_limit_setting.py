# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings > Chat tool output limit (#10349, #10135): env first, then the owner's saved value, then 16,000; a small
context window still lowers it."""

from pathlib import Path
import sys

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import routes.settings as settings
from core.inference import studio_tool_loop, tools
from storage import studio_db
from utils import tool_result_limit as limit
from utils.account_context import OWNER, AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")
# 625 lines, 50,000 characters: the reporter's ~50k file.
FILE_50K = ("x" * 79 + "\n") * 625


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", set())
    monkeypatch.delenv(limit.MAX_CHARS_ENV, raising = False)
    monkeypatch.setattr(tools, "_MAX_OUTPUT_CHARS", limit.DEFAULT_MAX_CHARS)
    monkeypatch.setattr(tools, "_loaded_context_tokens", lambda: None)
    token = tools._REQUEST_CONTEXT_TOKENS.set(tools._UNSET_CONTEXT_TOKENS)
    yield
    tools._REQUEST_CONTEXT_TOKENS.reset(token)


def _client(account):
    app = FastAPI()
    app.include_router(settings.router)

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[settings.get_current_subject] = subject
    return TestClient(app, raise_server_exceptions = False)


def test_default_is_unchanged():
    assert limit.saved_max_chars() is None
    assert limit.effective_max_chars() == 16000
    assert tools._tool_result_max_chars() == 16000
    out = tools._truncate(FILE_50K)
    assert "truncated to 16000 chars for the model" in out


def test_a_saved_value_raises_the_cap_and_saving_again_is_a_no_op(tmp_path):
    for _ in range(2):
        assert limit.set_max_chars(64000) == 64000
    assert tools._tool_result_max_chars() == 64000
    assert tools._truncate(FILE_50K, workdir = str(tmp_path)) == FILE_50K
    assert studio_tool_loop._truncate_for_model(FILE_50K) == FILE_50K


def test_a_saved_value_can_lower_the_cap_too():
    limit.set_max_chars(4000)
    assert "truncated to 4000 chars for the model" in tools._truncate(FILE_50K)
    assert len(studio_tool_loop._truncate_for_model(FILE_50K)) < 4100


def test_a_small_window_still_lowers_a_raised_cap():
    limit.set_max_chars(200000)
    tools._REQUEST_CONTEXT_TOKENS.set(8192)
    # 8192 tokens * 4 chars * 0.35 share.
    assert tools._tool_result_char_budget() == 11468


def test_the_hard_cap_never_cuts_below_the_saved_cap(monkeypatch):
    monkeypatch.setattr(tools, "MAX_TOOL_TEXT_CHARS", 10000)
    limit.set_max_chars(50000)
    assert tools._hard_cap_chars() == 54000


@pytest.mark.parametrize("value, expected", [("32000", 32000), ("abc", 16000), ("-5", 16000)])
def test_a_set_environment_variable_decides_as_before(monkeypatch, value, expected):
    limit.set_max_chars(64000)
    monkeypatch.setenv(limit.MAX_CHARS_ENV, value)
    monkeypatch.setattr(tools, "_MAX_OUTPUT_CHARS", tools._env_int(limit.MAX_CHARS_ENV, 16000))
    assert limit.locked_by_environment() is True
    assert limit.effective_max_chars() == expected
    assert tools._tool_result_max_chars() == expected


def test_an_empty_environment_variable_does_not_lock(monkeypatch):
    monkeypatch.setenv(limit.MAX_CHARS_ENV, "")
    limit.set_max_chars(64000)
    assert limit.locked_by_environment() is False
    assert tools._tool_result_max_chars() == 64000


@pytest.mark.parametrize("value", [1999, 200001, True, None, "x", 1.5])
def test_setter_rejects_out_of_range(value):
    with pytest.raises(ValueError):
        limit.set_max_chars(value)


def test_an_unreadable_store_keeps_the_default(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(studio_db, "get_app_setting", boom)
    assert tools._tool_result_max_chars() == 16000


def test_a_managed_account_tool_call_reads_the_owner_value():
    limit.set_max_chars(48000)
    token = bind_account(ALICE)
    try:
        assert tools._tool_result_max_chars() == 48000
    finally:
        reset_account(token)


def test_route_reads_and_saves():
    with _client(OWNER) as client:
        body = client.get("/tool-result-limit").json()
        assert body == {
            "max_chars": 16000,
            "default_chars": 16000,
            "min_chars": 2000,
            "max_allowed_chars": 200000,
            "locked_by_environment": False,
        }
        for _ in range(2):
            response = client.put("/tool-result-limit", json = {"max_chars": 64000})
            assert response.status_code == 200, response.text
            assert response.json()["max_chars"] == 64000
        assert client.get("/tool-result-limit").json()["max_chars"] == 64000
    assert tools._tool_result_max_chars() == 64000


@pytest.mark.parametrize("value", [1999, 200001, "64000", 1.5, True])
def test_route_refuses_out_of_range(value):
    with _client(OWNER) as client:
        assert client.put("/tool-result-limit", json = {"max_chars": value}).status_code == 422
    assert limit.saved_max_chars() is None


def test_route_refuses_a_save_the_environment_decides(monkeypatch):
    monkeypatch.setenv(limit.MAX_CHARS_ENV, "32000")
    with _client(OWNER) as client:
        body = client.get("/tool-result-limit").json()
        assert body["max_chars"] == 32000 and body["locked_by_environment"] is True
        response = client.put("/tool-result-limit", json = {"max_chars": 64000})
    assert response.status_code == 409
    assert limit.MAX_CHARS_ENV in response.json()["detail"]
    assert limit.saved_max_chars() is None
