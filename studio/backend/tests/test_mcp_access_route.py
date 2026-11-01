# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import storage.studio_db as studio_db
from auth import policy, storage
from auth.authentication import create_access_token
from mcp_server import tool_catalog
from routes import settings
from utils.keyless_api_access import set_keyless_api_access
from utils.mcp_access import (
    ENV_FORCE,
    MCP_DISABLED_TOOLS_SETTING_KEY,
    MCP_ENABLED_SETTING_KEY,
    get_mcp_enabled,
)

from .mcp_harness import isolated_auth, owner_auth  # noqa: F401  (fixtures)

UI_ONLY = "Agent access (MCP) can only be changed from the Unsloth UI."
URL = "/api/settings/mcp-access"
pytestmark = pytest.mark.usefixtures("owner_auth")


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(settings.router, prefix = "/api/settings")
    app.state.bind_host = "127.0.0.1"
    app.state.secure = False
    app.state.remote_access_is_colab = False
    app.state.lan_access_is_colab = False
    app.state.lan_access_secure_launch = False
    app.state.cloudflare_url = None
    with TestClient(app, base_url = "http://127.0.0.1:8888", client = ("127.0.0.1", 50000)) as http:
        yield http


def _owner_session():
    return {
        "Authorization": f"Bearer {create_access_token(subject = storage.DEFAULT_ADMIN_USERNAME)}"
    }


def _alice_session():
    storage.issue_account_setup_code(username = "alice")
    record = storage.get_user_record("alice")
    secret = storage.update_account_password(
        "alice",
        "alice-password-123",
        expect_password_hash = record["password_hash"],
        expect_secret = record["jwt_secret"],
    )
    policy.invalidate_account_cache()
    return {"Authorization": f"Bearer {create_access_token(subject = 'alice', secret = secret)}"}


def test_the_owner_session_reads_and_turns_it_on(client):
    got = client.get(URL, headers = _owner_session())
    assert got.status_code == 200, got.text
    assert got.json() == {
        "enabled": False,
        "forced_by_env": False,
        "url": "http://127.0.0.1:8888/mcp/",
        "tools": [{**entry, "enabled": True} for entry in tool_catalog()],
    }

    put = client.put(URL, json = {"enabled": True}, headers = _owner_session())
    assert put.status_code == 200, put.text
    assert put.json()["enabled"] is True
    assert put.json()["url"].endswith("/mcp/")
    assert get_mcp_enabled() is True

    off = client.put(URL, json = {"enabled": False}, headers = _owner_session())
    assert off.json()["enabled"] is False
    assert get_mcp_enabled() is False


@pytest.mark.parametrize("value", ["true", 1, None])
def test_the_payload_must_be_a_strict_bool(client, value):
    response = client.put(URL, json = {"enabled": value}, headers = _owner_session())
    assert response.status_code == 422
    assert get_mcp_enabled() is False


def _owner_api_key():
    raw_key, _row = storage.create_api_key(storage.DEFAULT_ADMIN_USERNAME, name = "agent")
    return {"Authorization": f"Bearer {raw_key}"}


def _keyless(headers):
    def make():
        set_keyless_api_access("full", tools = True)
        return headers

    return make


OWNER_ONLY = "Only the installation owner can do this"


@pytest.mark.parametrize("body", [{"enabled": True}, {"disabled_tools": ["cancel"]}])
@pytest.mark.parametrize("method", ["GET", "PUT"])
@pytest.mark.parametrize(
    "make_headers,detail",
    [
        (_owner_api_key, UI_ONLY),
        (_keyless({}), UI_ONLY),
        (_keyless({"Authorization": "Bearer not-needed"}), UI_ONLY),
        (_alice_session, OWNER_ONLY),
    ],
    ids = ["owner-api-key", "keyless", "keyless-dummy-bearer", "managed-account-session"],
)
def test_anything_but_the_owners_ui_session_is_refused(client, method, make_headers, detail, body):
    response = client.request(method, URL, json = body, headers = make_headers())
    assert response.status_code == 403
    assert response.json()["detail"] == detail
    assert get_mcp_enabled() is False
    assert _stored() == {}


def test_a_put_while_forced_by_env_is_a_conflict(client, monkeypatch):
    monkeypatch.setenv(ENV_FORCE, "1")
    got = client.get(URL, headers = _owner_session())
    assert got.json()["enabled"] is True
    assert got.json()["forced_by_env"] is True

    response = client.put(URL, json = {"enabled": False}, headers = _owner_session())
    assert response.status_code == 409
    assert response.json()["detail"] == "Set by UNSLOTH_STUDIO_ENABLE_MCP."
    assert get_mcp_enabled() is False
    assert _stored() == {}


def _stored():
    return studio_db.get_app_settings([MCP_ENABLED_SETTING_KEY, MCP_DISABLED_TOOLS_SETTING_KEY])


def _off(response):
    return [tool["name"] for tool in response.json()["tools"] if not tool["enabled"]]


def test_the_owner_turns_tools_off_and_back_on(client):
    put = client.put(URL, json = {"disabled_tools": ["voices", "cancel"]}, headers = _owner_session())
    assert put.status_code == 200, put.text
    assert _off(put) == ["voices", "cancel"]
    assert [tool["name"] for tool in put.json()["tools"]] == [e["name"] for e in tool_catalog()]
    assert put.json()["enabled"] is False
    assert _stored() == {MCP_DISABLED_TOOLS_SETTING_KEY: ["cancel", "voices"]}
    assert _off(client.get(URL, headers = _owner_session())) == ["voices", "cancel"]

    both = client.put(URL, json = {"enabled": True, "disabled_tools": []}, headers = _owner_session())
    assert both.status_code == 200, both.text
    assert both.json()["enabled"] is True and _off(both) == []
    assert _stored() == {MCP_ENABLED_SETTING_KEY: True, MCP_DISABLED_TOOLS_SETTING_KEY: []}


def test_two_windows_toggling_different_tools_both_land(client):
    # Each window read the same snapshot; a whole-list PUT from the second would drop the first.
    a = client.put(URL, json = {"tool_states": {"voices": False}}, headers = _owner_session())
    b = client.put(URL, json = {"tool_states": {"cancel": False}}, headers = _owner_session())
    assert a.status_code == b.status_code == 200, (a.text, b.text)
    assert _stored() == {MCP_DISABLED_TOOLS_SETTING_KEY: ["cancel", "voices"]}
    on = client.put(URL, json = {"tool_states": {"voices": True}}, headers = _owner_session())
    assert _off(on) == ["cancel"]


def test_concurrent_tool_states_all_land(client):
    from concurrent.futures import ThreadPoolExecutor

    names = [entry["name"] for entry in tool_catalog()][:8]
    with ThreadPoolExecutor(8) as pool:
        codes = list(
            pool.map(
                lambda name: client.put(
                    URL, json = {"tool_states": {name: False}}, headers = _owner_session()
                ).status_code,
                names,
            )
        )
    assert codes == [200] * len(names)
    assert _stored() == {MCP_DISABLED_TOOLS_SETTING_KEY: sorted(names)}


@pytest.mark.parametrize(
    "body, code",
    [
        ({"tool_states": {"cancel": False}, "disabled_tools": []}, 422),
        ({"tool_states": {"nope": False}}, 422),
        ({"tool_states": {"cancel": "off"}}, 422),
        ({"tool_states": None}, 422),
    ],
    ids = ["both", "unknown", "non-bool", "null"],
)
def test_a_bad_tool_states_body_writes_nothing(client, body, code):
    response = client.put(URL, json = body, headers = _owner_session())
    assert response.status_code == code, response.text
    assert _stored() == {}


def test_tool_states_drops_a_stale_stored_name(client):
    studio_db.upsert_app_settings({MCP_DISABLED_TOOLS_SETTING_KEY: ["gone_tool", "cancel"]})
    put = client.put(URL, json = {"tool_states": {"voices": False}}, headers = _owner_session())
    assert put.status_code == 200, put.text
    assert _stored() == {MCP_DISABLED_TOOLS_SETTING_KEY: ["cancel", "voices"]}


def test_tool_states_recover_an_unreadable_list(client):
    studio_db.upsert_app_settings({MCP_DISABLED_TOOLS_SETTING_KEY: "x"})
    response = client.put(URL, json = {"tool_states": {"cancel": True}}, headers = _owner_session())
    assert response.status_code == 200, response.text
    everything_else = sorted(e["name"] for e in tool_catalog() if e["name"] != "cancel")
    assert _stored() == {MCP_DISABLED_TOOLS_SETTING_KEY: everything_else}


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"disabled_tools": None},
        {"enabled": None, "disabled_tools": ["cancel"]},
        {"disabled_tools": "cancel"},
        {"disabled_tools": ["cancel", 1]},
        {"something": True},
    ],
    ids = ["empty", "null-list", "null-enabled", "string", "non-string", "unknown-field"],
)
def test_a_bad_body_is_a_422_and_writes_nothing(client, body):
    response = client.put(URL, json = body, headers = _owner_session())
    assert response.status_code == 422, response.text
    assert _stored() == {}


def test_an_unknown_tool_is_a_422_and_writes_nothing(client):
    response = client.put(
        URL, json = {"enabled": True, "disabled_tools": ["cancel", "nope"]}, headers = _owner_session()
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Unknown Unsloth Studio MCP tool: nope"
    assert _stored() == {}


@pytest.mark.parametrize("stored", ["x", {}])
def test_an_unreadable_stored_list_shows_every_tool_off(client, stored):
    studio_db.upsert_app_settings({MCP_DISABLED_TOOLS_SETTING_KEY: stored})
    got = client.get(URL, headers = _owner_session())
    assert got.status_code == 200, got.text
    assert _off(got) == [entry["name"] for entry in tool_catalog()]


def test_a_stale_stored_name_is_ignored_then_dropped(client):
    studio_db.upsert_app_settings({MCP_DISABLED_TOOLS_SETTING_KEY: ["gone_tool", "cancel"]})
    assert _off(client.get(URL, headers = _owner_session())) == ["cancel"]
    put = client.put(URL, json = {"disabled_tools": ["chat"]}, headers = _owner_session())
    assert put.status_code == 200, put.text
    assert _stored() == {MCP_DISABLED_TOOLS_SETTING_KEY: ["chat"]}


def test_tools_can_be_chosen_while_forced_by_env(client, monkeypatch):
    monkeypatch.setenv(ENV_FORCE, "1")
    put = client.put(URL, json = {"disabled_tools": ["cancel"]}, headers = _owner_session())
    assert put.status_code == 200, put.text
    assert put.json()["enabled"] is True and put.json()["forced_by_env"] is True
    assert _off(put) == ["cancel"]
    assert _stored() == {MCP_DISABLED_TOOLS_SETTING_KEY: ["cancel"]}

    for body in ({"enabled": False}, {"enabled": True, "disabled_tools": []}):
        response = client.put(URL, json = body, headers = _owner_session())
        assert response.status_code == 409
        assert response.json()["detail"] == "Set by UNSLOTH_STUDIO_ENABLE_MCP."
        assert _stored() == {MCP_DISABLED_TOOLS_SETTING_KEY: ["cancel"]}


def test_an_unreadable_snapshot_sends_no_tool_list(client, monkeypatch):
    from utils import mcp_access

    mcp_access._reset_cache()

    def broken():
        raise RuntimeError("database is locked")

    monkeypatch.setattr(mcp_access, "_read_from_db", broken)
    body = client.get(URL, headers = _owner_session()).json()
    # null, never 23 tools that only look turned off: a toggle built on that view would save it.
    assert body["tools"] is None
    assert body["enabled"] is False
