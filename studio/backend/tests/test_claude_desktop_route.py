# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings → Agents → Claude Desktop: the backend-owned switch, its routes and its lifecycle."""

import json
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

import routes.claude_desktop as route
from core import claude_desktop as session
from utils import claude_desktop as cd
from utils import client_ip
from utils.account_context import AccountContext, OWNER, bind_account, reset_account

USER_CLAUDE = '{\n  "mcpServers": {"fs": {"command": "npx"}}\n}\n'
STUDIO_KEY = "sk-unsloth-studio-claude-1"
ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture
def mac(monkeypatch, tmp_path):
    real_targets = cd.target_paths
    home = tmp_path / "home"
    app = home / "Applications" / "Claude.app"
    desktop = {
        "running": False,
        "opened": [],
        "quit": [],
        "minted": [],
        "revoked": [],
        "app": app,
        "model": cd.gateway_model({"id": "unsloth/gemma-4-E2B-it-GGUF", "context_length": 32768}),
        "quits": True,
    }

    def quit_app(timeout = 30.0):
        desktop["quit"].append(timeout)
        if desktop["quits"]:
            desktop["running"] = False

    def open_app(path):
        desktop["opened"].append(path)
        desktop["running"] = True

    def mint(username):
        desktop["minted"].append(username)
        return STUDIO_KEY, 41

    def revoke(state):
        desktop["revoked"].append(state.get("key_id"))
        return True

    monkeypatch.setattr(cd, "can_launch", lambda platform = None: True)
    monkeypatch.setattr(cd, "app_path", lambda platform = None, home = None: app)
    monkeypatch.setattr(
        cd, "target_paths", lambda platform = None, h = None: real_targets("darwin", home)
    )
    monkeypatch.setattr(cd, "is_running", lambda: desktop["running"])
    monkeypatch.setattr(cd, "quit_app", quit_app)
    monkeypatch.setattr(cd, "open_app", open_app)
    monkeypatch.setattr(cd, "wait_for_exit", lambda timeout = 30.0, poll = 0.2: not desktop["running"])
    monkeypatch.setattr(session, "agents_root", lambda: tmp_path / "studio" / "auth" / "agents")
    monkeypatch.setattr(session, "mint_key", mint)
    monkeypatch.setattr(session, "revoke_key", revoke)
    monkeypatch.setattr(session, "loaded_model", lambda: desktop["model"])
    monkeypatch.setattr(session, "_live", None)
    support = home / "Library" / "Application Support"
    desktop["config"] = support / "Claude" / "claude_desktop_config.json"
    desktop["profile"] = support / "Claude-3p" / "configLibrary" / f"{cd.PROFILE_ID}.json"
    desktop["state"] = tmp_path / "studio" / "auth" / "agents" / "app" / "claude-desktop.json"
    desktop["config"].parent.mkdir(parents = True)
    desktop["config"].write_text(USER_CLAUDE)
    yield desktop
    live = session._live
    if live is not None:
        live.gateway.stop()


def _client(
    monkeypatch,
    account = OWNER,
    via_api_key = False,
    local = True,
):
    monkeypatch.setattr(client_ip, "is_direct_local_request", lambda _request: local)
    app = FastAPI()
    app.include_router(route.router, prefix = "/api/settings/claude-desktop")

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[route.get_current_subject] = subject
    app.dependency_overrides[route.authenticated_via_api_key] = lambda: via_api_key
    return TestClient(app, raise_server_exceptions = False)


URL = "/api/settings/claude-desktop"


# -- status ------------------------------------------------------------------------------------


def test_status_reports_an_installed_idle_claude(mac, monkeypatch):
    body = _client(monkeypatch).get(URL).json()
    assert body == {
        "supported": True,
        "installed": True,
        "app_path": str(mac["app"]),
        "download_url": "https://claude.ai/download",
        "connected": False,
        "held_by": None,
        "held_by_pid": None,
        "running": False,
        "model": "unsloth/gemma-4-E2B-it-GGUF",
        "local": True,
    }


def test_status_off_macos_and_without_the_app(mac, monkeypatch):
    monkeypatch.setattr(cd, "can_launch", lambda platform = None: False)
    body = _client(monkeypatch).get(URL).json()
    assert body["supported"] is False and body["installed"] is False
    monkeypatch.setattr(cd, "can_launch", lambda platform = None: True)
    monkeypatch.setattr(cd, "app_path", lambda platform = None, home = None: None)
    body = _client(monkeypatch).get(URL).json()
    assert body["supported"] is True and body["installed"] is False
    assert body["download_url"] == cd.DOWNLOAD_URL


def test_only_the_owner_in_a_ui_session_reaches_it(mac, monkeypatch):
    assert _client(monkeypatch, account = ALICE).get(URL).status_code == 403
    assert _client(monkeypatch, via_api_key = True).get(URL).status_code == 403
    assert _client(monkeypatch, via_api_key = True).post(f"{URL}/connect", json = {}).status_code == 403


def test_connect_needs_the_computer_running_studio(mac, monkeypatch):
    client = _client(monkeypatch, local = False)
    assert client.get(URL).json()["local"] is False
    response = client.post(f"{URL}/connect", json = {})
    assert response.status_code == 403
    assert "computer running Unsloth Studio" in response.json()["detail"]
    assert mac["minted"] == []


# -- connect / disconnect ----------------------------------------------------------------------


def test_connect_switches_claude_to_a_backend_gateway_and_disconnect_restores(mac, monkeypatch):
    client = _client(monkeypatch)
    response = client.post(f"{URL}/connect", json = {})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "connected" and body["claude"]["connected"] is True
    assert len(mac["minted"]) == 1
    assert mac["opened"] == [mac["app"]]

    profile = json.loads(mac["profile"].read_text())
    assert profile["deploymentDisplayName"] == "Unsloth"
    assert STUDIO_KEY not in mac["profile"].read_text()
    assert cd.probe(profile["inferenceGatewayBaseUrl"])
    state = json.loads(mac["state"].read_text())
    assert state["owner"]["kind"] == "studio" and state["key_id"] == 41
    assert state["base"] == "http://testserver"
    assert session._live.gateway.upstream_key == STUDIO_KEY
    # Connecting twice keeps the one session.
    assert client.post(f"{URL}/connect", json = {}).json()["status"] == "connected"
    assert len(mac["minted"]) == 1

    response = client.post(f"{URL}/disconnect")
    assert response.status_code == 200, response.text
    assert response.json()["claude"]["connected"] is False
    assert mac["quit"] == [session.QUIT_TIMEOUT]
    assert mac["config"].read_text() == USER_CLAUDE
    assert not mac["profile"].exists()
    assert not mac["state"].exists()
    assert mac["revoked"] == [41]
    assert session._live is None
    assert not cd.probe(profile["inferenceGatewayBaseUrl"], timeout = 0.5)


def test_an_open_claude_asks_before_restarting(mac, monkeypatch):
    mac["running"] = True
    client = _client(monkeypatch)
    response = client.post(f"{URL}/connect", json = {})
    assert response.json()["status"] == "needs_restart"
    # Nothing was touched while the dialog is up.
    assert mac["minted"] == [] and mac["quit"] == [] and not mac["state"].exists()
    assert mac["config"].read_text() == USER_CLAUDE

    response = client.post(f"{URL}/connect", json = {"restart": True})
    assert response.json()["status"] == "connected"
    assert len(mac["quit"]) == 1 and mac["opened"] == [mac["app"]]
    assert json.loads(mac["config"].read_text())["deploymentMode"] == "3p"


def test_declining_the_restart_switches_for_the_next_launch(mac, monkeypatch):
    mac["running"] = True
    response = _client(monkeypatch).post(f"{URL}/connect", json = {"restart": False})
    assert response.json()["status"] == "pending_restart"
    assert mac["quit"] == [] and mac["opened"] == []
    assert json.loads(mac["config"].read_text())["deploymentMode"] == "3p"


def test_a_claude_that_will_not_quit_changes_nothing(mac, monkeypatch):
    mac["running"] = True
    mac["quits"] = False
    response = _client(monkeypatch).post(f"{URL}/connect", json = {"restart": True})
    assert response.status_code == 409
    assert "did not quit" in response.json()["detail"]
    assert mac["config"].read_text() == USER_CLAUDE
    assert not mac["state"].exists() and session._live is None
    assert mac["revoked"] == [41]


def test_connect_off_macos_and_without_the_app_are_clear_errors(mac, monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(cd, "app_path", lambda platform = None, home = None: None)
    response = client.post(f"{URL}/connect", json = {})
    assert response.status_code == 409
    assert "https://claude.ai/download" in response.json()["detail"]
    monkeypatch.setattr(cd, "can_launch", lambda platform = None: False)
    response = client.post(f"{URL}/connect", json = {})
    assert response.status_code == 409 and "macOS" in response.json()["detail"]
    assert mac["minted"] == []


def test_a_broken_claude_config_is_refused_untouched(mac, monkeypatch):
    mac["config"].write_text("[not, an, object")
    response = _client(monkeypatch).post(f"{URL}/connect", json = {})
    assert response.status_code == 409
    assert "nothing was changed" in response.json()["detail"]
    assert mac["config"].read_text() == "[not, an, object" and mac["minted"] == []


def test_a_failed_open_is_rolled_back(mac, monkeypatch):
    def broken_open(path):
        raise OSError("open failed")

    monkeypatch.setattr(cd, "open_app", broken_open)
    response = _client(monkeypatch).post(f"{URL}/connect", json = {})
    assert response.status_code == 500
    assert mac["config"].read_text() == USER_CLAUDE
    assert not mac["state"].exists() and not mac["profile"].exists()
    assert mac["revoked"] == [41] and session._live is None


def test_a_terminal_session_holds_claude_until_it_stops(mac, monkeypatch):
    cd.write_state(
        mac["state"], {"files": {"x": {"original": None}}, "owner": {"pid": 4242, "started": 1.0}}
    )
    monkeypatch.setattr(cd, "owner_alive", lambda owner: True)
    client = _client(monkeypatch)
    body = client.get(URL).json()
    assert body["held_by"] == "cli" and body["held_by_pid"] == 4242
    response = client.post(f"{URL}/connect", json = {})
    assert response.status_code == 409 and "PID 4242" in response.json()["detail"]
    response = client.post(f"{URL}/disconnect")
    assert response.status_code == 409 and "Ctrl+C" in response.json()["detail"]


# -- lifecycle ---------------------------------------------------------------------------------


def test_studio_quitting_switches_claude_back_quickly(mac, monkeypatch):
    _client(monkeypatch).post(f"{URL}/connect", json = {})
    session.shutdown()
    assert mac["quit"] == [session.SHUTDOWN_QUIT_TIMEOUT]
    assert mac["config"].read_text() == USER_CLAUDE
    assert not mac["state"].exists() and mac["revoked"] == [41]
    session.shutdown()  # nothing left: a no-op
    assert mac["quit"] == [session.SHUTDOWN_QUIT_TIMEOUT]


def test_a_crashed_studio_is_switched_back_on_the_next_start(mac, monkeypatch):
    _client(monkeypatch).post(f"{URL}/connect", json = {})
    # The process dies: nothing in memory, the record and Claude's files stay switched.
    session._live.gateway.stop()
    monkeypatch.setattr(session, "_live", None)
    state = json.loads(mac["state"].read_text())
    state["owner"] = {"pid": 2**22 + 777, "started": 1.0, "kind": "studio"}
    cd.write_state(mac["state"], state)
    mac["running"] = True

    notes = session.recover()

    assert any("Quit Claude Desktop" in n for n in notes)
    assert mac["config"].read_text() == USER_CLAUDE
    assert not mac["profile"].exists() and not mac["state"].exists()
    assert mac["revoked"] == [41]
    assert session.recover() == []


def test_recovery_leaves_a_live_owner_alone(mac, monkeypatch):
    _client(monkeypatch).post(f"{URL}/connect", json = {})
    live = session._live
    monkeypatch.setattr(session, "_live", None)
    assert session.recover() == []  # this very process still owns it
    assert mac["state"].exists()
    monkeypatch.setattr(session, "_live", live)


def test_revoke_key_finds_the_owner_of_any_switch_key(monkeypatch):
    from auth import storage

    calls = []
    monkeypatch.setattr(storage, "validate_api_key", lambda key: "max" if key == "k" else None)
    monkeypatch.setattr(
        storage, "revoke_api_key", lambda user, key_id: calls.append((user, key_id)) or True
    )
    assert session.revoke_key({"key": "k", "key_id": 9}) is True
    assert calls == [("max", 9)]
    assert session.revoke_key({"key": "gone", "key_id": 9}) is True  # already revoked
    assert session.revoke_key({"key_id": None}) is True
    assert calls == [("max", 9)]


def test_claude_follows_the_first_model_studio_has_loaded(monkeypatch):
    fake = _types.ModuleType("routes.inference")
    loaded = []
    fake._openai_model_objects = lambda: loaded
    monkeypatch.setitem(sys.modules, "routes.inference", fake)
    assert session.loaded_model() is None
    loaded[:] = [{"id": "unsloth/Qwen3.6-GGUF", "context_length": 8192}, {"id": "other"}]
    model = session.loaded_model()
    assert model.model_id == "unsloth/Qwen3.6-GGUF" and model.max_tokens == 8192


def test_main_wires_the_route_and_both_lifecycle_hooks():
    source = (Path(_BACKEND_DIR) / "main.py").read_text()
    assert 'prefix = "/api/settings/claude-desktop"' in source
    startup, shutdown = source.split("    yield\n", 1)
    assert "_claude_desktop.recover()" in startup
    assert "_claude_desktop.shutdown" in shutdown
