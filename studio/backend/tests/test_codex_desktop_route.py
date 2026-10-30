# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Settings → Agents → Codex Desktop: the backend-owned switch, its routes and its lifecycle."""

import json
import os
import plistlib
import sys
import threading
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import routes.codex_desktop as route
from core import codex_desktop as session
from utils import claude_desktop as cd
from utils import client_ip
from utils import codex_desktop as cx
from utils.account_context import AccountContext, OWNER, bind_account, reset_account

USER_CONFIG = (
    'model = "gpt-5.5"\n'
    'model_reasoning_effort = "high"\n'
    "\n"
    "[mcp_servers.fs]\n"
    'command = "npx"\n'
)
STUDIO_KEY = "sk-unsloth-studio-codex-1"
LOADED = "unsloth/gemma-4-E2B-it-GGUF"
ALICE = AccountContext("a" * 32, "alice")
URL = "/api/settings/codex-desktop"


@pytest.fixture
def mac(monkeypatch, tmp_path):
    home = tmp_path / "home"
    app = home / "Applications" / "ChatGPT.app"
    desktop = {
        "running": False,
        "opened": [],
        "quit": [],
        "minted": [],
        "revoked": [],
        "loads": [],
        "load_error": None,
        "app": app,
        "loaded": [{"id": LOADED, "context_length": 32768}],
        "quits": True,
    }

    def quit_app(timeout = 30.0):
        desktop["quit"].append(timeout)
        if desktop["quits"]:
            desktop["running"] = False

    def open_app(path = None):
        desktop["opened"].append(path)
        desktop["running"] = True

    def mint(username):
        desktop["minted"].append(username)
        return STUDIO_KEY, 52

    def revoke(state):
        desktop["revoked"].append(state.get("key_id"))
        return True

    def load(upstream, key, model_id):
        desktop["loads"].append((upstream, key, model_id))
        if desktop["load_error"]:
            raise RuntimeError(desktop["load_error"])
        desktop["loaded"] = [{"id": model_id, "context_length": 8192}]

    monkeypatch.setattr(cd, "user_home", lambda: home)
    monkeypatch.setattr(cx, "can_launch", lambda platform = None: True)
    monkeypatch.setattr(cx, "app_path", lambda home = None: app)
    monkeypatch.setattr(cx, "is_running", lambda app = None: desktop["running"])
    monkeypatch.setattr(cx, "quit_app", quit_app)
    monkeypatch.setattr(cx, "open_app", open_app)
    monkeypatch.setattr(
        cx, "wait_for_exit", lambda timeout = 30.0, poll = 0.2, app = None: not desktop["running"]
    )
    monkeypatch.setattr(cd, "load_model", load)
    monkeypatch.setattr(session, "agents_root", lambda: tmp_path / "studio" / "auth" / "agents")
    monkeypatch.setattr(session, "mint_key", mint)
    monkeypatch.setattr(session, "revoke_key", revoke)
    monkeypatch.setattr(session, "loaded_models", lambda: desktop["loaded"])
    monkeypatch.setattr(session, "_live", None)
    monkeypatch.setattr(session, "_gateway", None)
    desktop["config"] = home / ".codex" / "config.toml"
    desktop["catalog"] = home / ".codex" / cx.CATALOG_NAME
    desktop["state"] = tmp_path / "studio" / "auth" / "agents" / "app" / "codex.json"
    desktop["config"].parent.mkdir(parents = True)
    desktop["config"].write_text(USER_CONFIG)
    os.chmod(desktop["config"], 0o640)
    yield desktop
    session._stop_gateway()


def _client(
    monkeypatch,
    account = OWNER,
    via_api_key = False,
    local = True,
):
    monkeypatch.setattr(client_ip, "is_direct_local_request", lambda _request: local)
    app = FastAPI()
    app.include_router(route.router, prefix = URL)

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[route.get_current_subject] = subject
    app.dependency_overrides[route.authenticated_via_api_key] = lambda: via_api_key
    return TestClient(app, raise_server_exceptions = False)


# -- finding the app ---------------------------------------------------------------------------


def _bundle(path: Path, bundle: str) -> None:
    (path / "Contents").mkdir(parents = True)
    with open(path / "Contents" / "Info.plist", "wb") as handle:
        plistlib.dump({"CFBundleIdentifier": bundle}, handle)


def test_only_the_codex_build_of_chatgpt_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cx, "app_candidates", lambda home = None: [tmp_path / "ChatGPT.app", tmp_path / "Codex.app"]
    )
    _bundle(tmp_path / "ChatGPT.app", "com.openai.chat")
    assert cx.app_path() is None
    _bundle(tmp_path / "Codex.app", cx.BUNDLE_ID)
    assert cx.app_path() == tmp_path / "Codex.app"


def test_candidates_match_ollama_launch_codex_app(tmp_path):
    assert cx.app_candidates(tmp_path) == [
        Path("/Applications/ChatGPT.app"),
        Path("/Applications/Codex.app"),
        tmp_path / "Applications" / "ChatGPT.app",
        tmp_path / "Applications" / "Codex.app",
    ]
    assert cx.can_launch("darwin") and not cx.can_launch("linux")


def test_open_and_quit_use_a_dock_like_environment(monkeypatch, tmp_path):
    calls = []

    class Done:
        returncode = 0

    def run(command, **kwargs):
        calls.append((command, kwargs.get("env")))
        return Done()

    monkeypatch.setenv("HOME", "/tmp/fake-home")
    monkeypatch.setenv("UNSLOTH_STUDIO_AUTH_TOKEN", "secret")
    monkeypatch.setattr(cx.subprocess, "run", run)
    cx.open_app(tmp_path / "ChatGPT.app")
    cx.quit_app()
    assert calls[0][0] == ["/usr/bin/open", "-a", str(tmp_path / "ChatGPT.app"), cx.LAUNCH_URL]
    assert calls[1][0][-1] == f'tell application id "{cx.BUNDLE_ID}" to quit'
    for _, env in calls:
        assert "UNSLOTH_STUDIO_AUTH_TOKEN" not in env
        assert env["HOME"] == str(cd.user_home())


def test_retarget_moves_only_our_model_line():
    text, previous = cx.build(USER_CONFIG, "http://127.0.0.1:8888", "k", "a/m", None)
    state = {"model": "a/m", "previous": previous, "base": "http://127.0.0.1:8888", "key": "k"}
    moved = cx.retarget(text, state, "b/n")
    assert 'model = "b/n"' in moved and 'model = "a/m"' not in moved
    restored, skipped = cx.restore(moved, {**state, "model": "b/n"})
    # Byte-exact comes from the backup in restore_files; the line edits alone may leave a blank line.
    assert restored.rstrip("\n") == USER_CONFIG.rstrip("\n") and skipped == []
    assert cx.retarget(text.replace('model = "a/m"', 'model = "gpt-5.5"'), state, "b/n") is None


# -- status and access -------------------------------------------------------------------------


def test_status_reports_an_installed_idle_codex(mac, monkeypatch):
    body = _client(monkeypatch).get(URL).json()
    assert body == {
        "supported": True,
        "installed": True,
        "app_path": str(mac["app"]),
        "download_url": "https://chatgpt.com/download",
        "connected": False,
        "held_by": None,
        "held_by_pid": None,
        "running": False,
        "model": None,
        "models": [],
        "loaded_model": LOADED,
        "selected_models": [],
        "max_models": 5,
        "local": True,
    }


def test_status_off_macos_and_without_the_app(mac, monkeypatch):
    monkeypatch.setattr(cx, "can_launch", lambda platform = None: False)
    assert _client(monkeypatch).get(URL).json()["supported"] is False
    monkeypatch.setattr(cx, "can_launch", lambda platform = None: True)
    monkeypatch.setattr(cx, "app_path", lambda home = None: None)
    response = _client(monkeypatch).post(f"{URL}/connect", json = {})
    assert response.status_code == 409 and cx.DOWNLOAD_URL in response.json()["detail"]


def test_only_the_owner_in_a_local_ui_session_drives_it(mac, monkeypatch):
    assert _client(monkeypatch, account = ALICE).get(URL).status_code == 403
    assert _client(monkeypatch, via_api_key = True).get(URL).status_code == 403
    response = _client(monkeypatch, local = False).post(f"{URL}/connect", json = {})
    assert response.status_code == 403 and mac["minted"] == []
    assert _client(monkeypatch, local = False).put(f"{URL}/models", json = {}).status_code == 403


# -- connect / disconnect ----------------------------------------------------------------------


def test_connect_points_codex_at_studio_and_disconnect_restores_byte_for_byte(mac, monkeypatch):
    client = _client(monkeypatch)
    response = client.post(f"{URL}/connect", json = {})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "connected" and body["codex"]["connected"] is True
    assert body["codex"]["model"] == LOADED
    assert mac["opened"] == [mac["app"]] and mac["loads"] == []

    config = mac["config"].read_text()
    assert 'model_provider = "unsloth_api"' in config
    assert f'model = "{LOADED}"' in config
    assert f'experimental_bearer_token = "{STUDIO_KEY}"' in config
    # Codex talks to the session's loopback gateway, which forwards to this Studio.
    assert f'base_url = "{session._gateway.base_url}/v1"' in config
    assert session._gateway.base_url.startswith("http://127.0.0.1:")
    assert "[mcp_servers.fs]" in config
    catalog = json.loads(mac["catalog"].read_text())
    assert catalog["models"][0]["slug"] == LOADED
    assert catalog["models"][0]["context_window"] == 32768
    state = json.loads(mac["state"].read_text())
    assert state["owner"]["kind"] == "studio" and state["key_id"] == 52
    assert mac["state"].stat().st_mode & 0o777 == 0o600
    # Twice keeps the one session.
    assert client.post(f"{URL}/connect", json = {}).json()["status"] == "connected"
    assert len(mac["minted"]) == 1

    response = client.post(f"{URL}/disconnect", json = {})
    assert response.json()["status"] == "needs_restart"  # Codex is open: ask first
    response = client.post(f"{URL}/disconnect", json = {"restart": True})
    assert response.status_code == 200, response.text
    assert response.json()["codex"]["connected"] is False
    assert mac["config"].read_text() == USER_CONFIG
    assert mac["config"].stat().st_mode & 0o777 == 0o640
    assert not mac["catalog"].exists() and not mac["state"].exists()
    assert not mac["config"].with_name("config.toml" + cx.BACKUP_SUFFIX).exists()
    assert mac["revoked"] == [52] and session._gateway is None
    # Restarted onto its own setup, as `ollama launch codex-app --restore` does.
    assert len(mac["quit"]) == 1 and mac["opened"] == [mac["app"], mac["app"]]


def test_a_closed_codex_stays_closed_on_disconnect(mac, monkeypatch):
    client = _client(monkeypatch)
    client.post(f"{URL}/connect", json = {})
    mac["running"] = False
    response = client.post(f"{URL}/disconnect")
    assert response.json()["status"] == "disconnected"
    assert mac["quit"] == [] and mac["opened"] == [mac["app"]]
    assert mac["config"].read_text() == USER_CONFIG


def test_disconnect_without_restart_leaves_codex_open(mac, monkeypatch):
    client = _client(monkeypatch)
    client.post(f"{URL}/connect", json = {})
    response = client.post(f"{URL}/disconnect", json = {"restart": False})
    assert response.json()["status"] == "disconnected"
    assert any("Restart Codex" in note for note in response.json()["notes"])
    assert mac["quit"] == [] and mac["config"].read_text() == USER_CONFIG


def test_no_config_yet_is_created_and_removed_again(mac, monkeypatch):
    mac["config"].unlink()
    client = _client(monkeypatch)
    client.post(f"{URL}/connect", json = {})
    assert mac["config"].exists()
    mac["running"] = False
    client.post(f"{URL}/disconnect")
    assert not mac["config"].exists()


def test_an_open_codex_asks_before_restarting(mac, monkeypatch):
    mac["running"] = True
    client = _client(monkeypatch)
    response = client.post(f"{URL}/connect", json = {})
    assert response.json()["status"] == "needs_restart"
    assert mac["minted"] == [] and mac["config"].read_text() == USER_CONFIG
    response = client.post(f"{URL}/connect", json = {"restart": True})
    assert response.json()["status"] == "connected"
    assert len(mac["quit"]) == 1 and mac["opened"] == [mac["app"]]


def test_declining_the_restart_switches_for_the_next_launch(mac, monkeypatch):
    mac["running"] = True
    response = _client(monkeypatch).post(f"{URL}/connect", json = {"restart": False})
    assert response.json()["status"] == "pending_restart"
    assert mac["quit"] == [] and mac["opened"] == []
    assert "unsloth_api" in mac["config"].read_text()


def test_a_codex_that_will_not_quit_changes_nothing(mac, monkeypatch):
    mac["running"] = True
    mac["quits"] = False
    response = _client(monkeypatch).post(f"{URL}/connect", json = {"restart": True})
    assert response.status_code == 409 and "did not quit" in response.json()["detail"]
    assert mac["config"].read_text() == USER_CONFIG
    assert not mac["state"].exists() and session._live is None
    assert mac["revoked"] == [52] and session._gateway is None


def test_no_model_loaded_and_none_picked_is_a_clear_error(mac, monkeypatch):
    mac["loaded"] = []
    response = _client(monkeypatch).post(f"{URL}/connect", json = {})
    assert response.status_code == 409 and "Load a model" in response.json()["detail"]
    assert mac["config"].read_text() == USER_CONFIG and mac["revoked"] == [52]


def test_a_broken_config_is_refused_untouched(mac, monkeypatch):
    mac["config"].write_text("model = \n[[")
    response = _client(monkeypatch).post(f"{URL}/connect", json = {})
    assert response.status_code == 409 and "Nothing was changed" in response.json()["detail"]
    assert mac["minted"] == []


def test_held_by_the_cli_is_reported_and_refused(mac, monkeypatch):
    mac["state"].parent.mkdir(parents = True)
    cx.write_state(
        mac["state"], {"path": str(mac["config"]), "owner": {"pid": os.getpid()}, "key_id": 3}
    )
    client = _client(monkeypatch)
    body = client.get(URL).json()
    assert body["held_by"] == "cli" and body["held_by_pid"] == os.getpid()
    response = client.post(f"{URL}/connect", json = {})
    assert response.status_code == 409 and "unsloth start codex --app" in response.json()["detail"]
    response = client.post(f"{URL}/disconnect")
    assert response.status_code == 409
    assert mac["state"].exists() and mac["minted"] == []


def test_a_crashed_session_is_switched_back_on_the_next_start(mac, monkeypatch):
    client = _client(monkeypatch)
    client.post(f"{URL}/connect", json = {})
    state = json.loads(mac["state"].read_text())
    state["owner"] = {"pid": 999_999_999, "started": 1.0, "kind": "studio"}
    cx.write_state(mac["state"], state)
    monkeypatch.setattr(session, "_live", None)  # the Studio that made it is gone
    assert client.get(URL).json()["held_by"] is None
    session.recover()
    assert mac["config"].read_text() == USER_CONFIG
    assert not mac["state"].exists() and mac["revoked"] == [52]


def test_shutdown_quits_codex_and_switches_back(mac, monkeypatch):
    _client(monkeypatch).post(f"{URL}/connect", json = {})
    session.shutdown()
    assert mac["quit"] == [session.SHUTDOWN_QUIT_TIMEOUT]
    assert mac["config"].read_text() == USER_CONFIG and session._live is None
    assert mac["revoked"] == [52]


# -- picking the model -------------------------------------------------------------------------


def test_a_picked_model_is_loaded_before_codex_is_pointed_at_it(mac, monkeypatch):
    client = _client(monkeypatch)
    response = client.post(f"{URL}/connect", json = {"models": ["unsloth/qwen-GGUF:Q8_0"]})
    assert response.status_code == 200, response.text
    assert mac["loads"] == [("http://testserver", STUDIO_KEY, "unsloth/qwen-GGUF:Q8_0")]
    body = response.json()["codex"]
    assert body["model"] == "unsloth/qwen-GGUF:Q8_0"
    assert body["selected_models"] == ["unsloth/qwen-GGUF:Q8_0"]
    assert 'model = "unsloth/qwen-GGUF:Q8_0"' in mac["config"].read_text()


def test_a_failed_load_changes_nothing(mac, monkeypatch):
    mac["load_error"] = "not enough memory"
    response = _client(monkeypatch).post(f"{URL}/connect", json = {"models": ["unsloth/big"]})
    assert response.status_code == 409 and "not enough memory" in response.json()["detail"]
    assert mac["config"].read_text() == USER_CONFIG and mac["revoked"] == [52]
    assert session._live is None


def test_picking_while_disconnected_is_saved_for_connect(mac, monkeypatch):
    client = _client(monkeypatch)
    response = client.put(f"{URL}/models", json = {"models": ["unsloth/qwen-GGUF"]})
    assert response.json()["status"] == "saved"
    assert response.json()["codex"]["selected_models"] == ["unsloth/qwen-GGUF"]
    assert mac["loads"] == [] and mac["minted"] == []
    client.post(f"{URL}/connect", json = {})
    assert mac["loads"][0][2] == "unsloth/qwen-GGUF"
    # An empty list goes back to following whatever Studio has loaded.
    client.post(f"{URL}/disconnect", json = {"restart": False})
    body = client.put(f"{URL}/models", json = {"models": []}).json()
    assert body["codex"]["selected_models"] == []


def test_changing_the_model_while_connected_restarts_codex(mac, monkeypatch):
    client = _client(monkeypatch)
    client.post(f"{URL}/connect", json = {})
    response = client.put(f"{URL}/models", json = {"models": ["unsloth/qwen-GGUF"]})
    assert response.json()["status"] == "needs_restart" and mac["loads"] == []
    response = client.put(f"{URL}/models", json = {"models": ["unsloth/qwen-GGUF"], "restart": True})
    assert response.json()["status"] == "connected", response.text
    assert response.json()["codex"]["model"] == "unsloth/qwen-GGUF"
    assert len(mac["quit"]) == 1 and len(mac["opened"]) == 2
    config = mac["config"].read_text()
    assert 'model = "unsloth/qwen-GGUF"' in config and LOADED not in config
    assert json.loads(mac["catalog"].read_text())["models"][0]["slug"] == "unsloth/qwen-GGUF"
    assert json.loads(mac["state"].read_text())["model"] == "unsloth/qwen-GGUF"
    assert len(mac["minted"]) == 1
    # Restore still gives back exactly what was there.
    client.post(f"{URL}/disconnect", json = {"restart": False})
    assert mac["config"].read_text() == USER_CONFIG


def test_a_model_line_changed_in_codex_is_left_alone(mac, monkeypatch):
    client = _client(monkeypatch)
    client.post(f"{URL}/connect", json = {})
    mac["running"] = False
    text = mac["config"].read_text().replace(f'model = "{LOADED}"', 'model = "other"')
    mac["config"].write_text(text)
    response = client.put(f"{URL}/models", json = {"models": ["unsloth/qwen-GGUF"]})
    assert response.status_code == 409 and "changed in Codex" in response.json()["detail"]
    assert mac["config"].read_text() == text


def test_models_lists_the_same_catalog_as_claude(mac, monkeypatch):
    import routes.llama_compat as llama_compat

    async def catalog():
        return [{"id": "b", "loaded": False}, {"id": "a", "loaded": True}]

    monkeypatch.setattr(llama_compat, "_chat_catalog", catalog)
    body = _client(monkeypatch).get(f"{URL}/models").json()
    assert [m["id"] for m in body] == ["a", "b"]


def test_start_codex_opens_a_closed_connected_codex(mac, monkeypatch):
    client = _client(monkeypatch)
    client.post(f"{URL}/connect", json = {})
    mac["running"] = False
    response = client.put(f"{URL}/models", json = {"models": ["unsloth/qwen-GGUF"], "restart": True})
    assert response.json()["status"] == "connected"
    assert mac["quit"] == [] and mac["opened"] == [mac["app"], mac["app"]]
    mac["running"] = False
    response = client.put(f"{URL}/models", json = {"models": ["unsloth/qwen-GGUF"], "restart": True})
    assert response.json()["status"] == "connected" and len(mac["opened"]) == 3


# -- several models ----------------------------------------------------------------------------


def test_several_models_become_codex_catalog_and_only_the_default_loads(mac, monkeypatch):
    client = _client(monkeypatch)
    picks = ["unsloth/qwen-GGUF:Q8_0", LOADED, "unsloth/glm-GGUF"]
    response = client.post(f"{URL}/connect", json = {"models": picks})
    assert response.status_code == 200, response.text
    body = response.json()["codex"]
    assert body["model"] == picks[0] and body["models"] == picks
    assert body["selected_models"] == picks
    # The default is loaded up front; the rest load when Codex first asks for them.
    assert [load[2] for load in mac["loads"]] == [picks[0]]
    catalog = json.loads(mac["catalog"].read_text())["models"]
    assert [entry["slug"] for entry in catalog] == picks
    assert [entry["priority"] for entry in catalog] == [99, 100, 101]
    assert all(entry["visibility"] == "list" for entry in catalog)
    assert catalog[0]["context_window"] == 8192 and "context_window" not in catalog[2]
    assert f'model = "{picks[0]}"' in mac["config"].read_text()
    state = json.loads(mac["state"].read_text())
    assert state["models"] == picks and state["follow"] is False
    assert session._gateway.listed("unsloth/glm-GGUF") == "unsloth/glm-GGUF"
    assert session._gateway.listed("gpt-5.5") is None


def test_a_listed_model_picked_in_codex_is_still_restored(mac, monkeypatch):
    client = _client(monkeypatch)
    client.post(f"{URL}/connect", json = {"models": ["unsloth/a", "unsloth/b"]})
    # The user switched to the second model in Codex's own picker; Codex saved it.
    text = mac["config"].read_text().replace('model = "unsloth/a"', 'model = "unsloth/b"')
    mac["config"].write_text(text)
    client.post(f"{URL}/disconnect", json = {"restart": False})
    assert mac["config"].read_text() == USER_CONFIG


def test_a_model_of_the_users_own_survives_restore(mac, monkeypatch):
    client = _client(monkeypatch)
    client.post(f"{URL}/connect", json = {"models": ["unsloth/a", "unsloth/b"]})
    text = mac["config"].read_text().replace('model = "unsloth/a"', 'model = "gpt-5.5-codex"')
    mac["config"].write_text(text)
    response = client.post(f"{URL}/disconnect", json = {"restart": False})
    assert 'model = "gpt-5.5-codex"' in mac["config"].read_text()
    assert any("model" in note for note in response.json()["notes"])


def test_changing_the_list_while_connected_rewrites_catalog_and_default(mac, monkeypatch):
    client = _client(monkeypatch)
    client.post(f"{URL}/connect", json = {"models": ["unsloth/a", "unsloth/b"]})
    mac["running"] = False
    text = mac["config"].read_text().replace('model = "unsloth/a"', 'model = "unsloth/b"')
    mac["config"].write_text(text)
    response = client.put(f"{URL}/models", json = {"models": ["unsloth/c", "unsloth/a"]})
    assert response.json()["status"] == "connected", response.text
    assert response.json()["codex"]["models"] == ["unsloth/c", "unsloth/a"]
    slugs = [e["slug"] for e in json.loads(mac["catalog"].read_text())["models"]]
    assert slugs == ["unsloth/c", "unsloth/a"]
    assert 'model = "unsloth/c"' in mac["config"].read_text()
    assert session._gateway.listed("unsloth/b") is None
    # b was taken off the list, but a model line Codex still holds on it is ours to restore.
    text = mac["config"].read_text().replace('model = "unsloth/c"', 'model = "unsloth/b"')
    mac["config"].write_text(text)
    client.post(f"{URL}/disconnect", json = {"restart": False})
    assert mac["config"].read_text() == USER_CONFIG


def test_the_list_is_capped_and_cleaned(mac, monkeypatch):
    client = _client(monkeypatch)
    six = [f"unsloth/m{i}" for i in range(6)]
    assert client.put(f"{URL}/models", json = {"models": six}).status_code == 422
    body = client.put(f"{URL}/models", json = {"models": [" a ", "a", "", "b"]}).json()
    assert body["codex"]["selected_models"] == ["a", "b"]


def test_the_one_model_pick_of_earlier_builds_is_kept(mac, monkeypatch):
    prefs = mac["state"].parent / session.PREFS_NAME
    prefs.parent.mkdir(parents = True)
    prefs.write_text(json.dumps({"model": "unsloth/qwen-GGUF"}))
    assert _client(monkeypatch).get(URL).json()["selected_models"] == ["unsloth/qwen-GGUF"]
    prefs.write_text(json.dumps({"model": None}))
    assert _client(monkeypatch).get(URL).json()["selected_models"] == []


def test_following_the_loaded_model_never_loads_one_back(mac, monkeypatch):
    _client(monkeypatch).post(f"{URL}/connect", json = {"models": []})
    assert json.loads(mac["state"].read_text())["follow"] is True
    assert session._gateway.listed(LOADED) is None


# -- the gateway --------------------------------------------------------------------------------


class _Upstream:
    """A stand-in for Studio's /v1 that records what reached it and streams a reply."""

    def __init__(self):
        import http.server
        import threading

        seen = self.seen = []
        replies = self.replies = []

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _reply(self, body: bytes, content_type: str) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                seen.append(("GET", self.path, dict(self.headers), b""))
                self._reply(b'{"data": []}', "application/json")

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                seen.append(("POST", self.path, dict(self.headers), body))
                if replies:
                    status, content_type, reply = replies.pop(0)
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(reply)))
                    self.end_headers()
                    self.wfile.write(reply)
                    return
                self._reply(b'data: {"type": "response.completed"}\n\n', "text/event-stream")

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target = self.server.serve_forever, daemon = True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def gateway():
    upstream = _Upstream()
    loaded = {"unsloth/a"}
    loads = []
    failing = {}

    def ensure(model_id):
        loads.append(model_id)
        if model_id in failing:
            raise RuntimeError(failing[model_id])
        loaded.add(model_id)

    gw = cx.Gateway(
        upstream.url,
        STUDIO_KEY,
        lambda: ["unsloth/a", "unsloth/b:Q4_K_M", "unsloth/c"],
        lambda model_id: model_id in loaded,
        ensure,
    ).start()
    yield {"gw": gw, "upstream": upstream, "loads": loads, "failing": failing}
    gw.stop()
    upstream.close()


def _call(
    gw,
    method,
    path,
    payload = None,
    key = STUDIO_KEY,
    headers = None,
):
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", gw.server_address[1], timeout = 10)
    body = json.dumps(payload).encode() if payload is not None else None
    sent = {"Content-Type": "application/json", **(headers or {})}
    if key:
        sent["Authorization"] = f"Bearer {key}"
    conn.request(method, path, body, sent)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    return response.status, data


def test_gateway_loads_a_listed_model_codex_asks_for_then_forwards(gateway):
    gw = gateway["gw"]
    payload = {"model": "unsloth/b:Q4_K_M", "input": "hi", "stream": True}
    status, data = _call(gw, "POST", "/v1/responses", payload)
    assert status == 200 and b"response.completed" in data
    assert gateway["loads"] == ["unsloth/b:Q4_K_M"]
    method, path, headers, body = gateway["upstream"].seen[-1]
    assert (method, path) == ("POST", "/v1/responses")
    assert json.loads(body) == payload  # passed through untouched
    assert headers["Authorization"] == f"Bearer {STUDIO_KEY}"
    # Loaded now: the next turn goes straight through.
    _call(gw, "POST", "/v1/responses", payload)
    assert gateway["loads"] == ["unsloth/b:Q4_K_M"]


def test_gateway_leaves_loaded_and_unlisted_models_alone(gateway):
    gw = gateway["gw"]
    assert _call(gw, "POST", "/v1/responses", {"model": "unsloth/a"})[0] == 200
    assert _call(gw, "POST", "/v1/responses", {"model": "gpt-5.5"})[0] == 200
    assert gateway["loads"] == []
    status, data = _call(gw, "GET", "/v1/models?limit=5")
    assert status == 200 and gateway["upstream"].seen[-1][1] == "/v1/models?limit=5"


def test_gateway_reports_a_failed_load_in_openai_shape(gateway):
    gateway["failing"]["unsloth/c"] = "not enough memory"
    status, data = _call(gateway["gw"], "POST", "/v1/responses", {"model": "unsloth/c"})
    assert status == 409
    assert "not enough memory" in json.loads(data)["error"]["message"]
    assert gateway["upstream"].seen == []


def test_gateway_refuses_other_keys_browsers_and_paths(gateway):
    gw = gateway["gw"]
    assert _call(gw, "POST", "/v1/responses", {"model": "unsloth/c"}, key = "nope")[0] == 401
    assert _call(gw, "GET", "/v1/models", key = None)[0] == 401
    origin = {"Origin": "https://evil.example"}
    assert _call(gw, "POST", "/v1/responses", {"model": "unsloth/c"}, headers = origin)[0] == 403
    assert _call(gw, "POST", "/api/models/load", {"model": "x"})[0] == 404
    assert gateway["loads"] == [] and gateway["upstream"].seen == []
    status, _ = _call(gw, "GET", cx.HEALTH_PATH, key = None)
    assert status == 204


def _slow(gateway, seconds = 0.3):
    gw = gateway["gw"]
    plain, finished = gw.ensure_loaded, threading.Event()

    def slow(model_id):
        threading.Event().wait(seconds)
        try:
            plain(model_id)
        finally:
            finished.set()

    gw.ensure_loaded = slow
    gw.keepalive_seconds = 0.02
    return finished


def test_gateway_keeps_a_streaming_turn_alive_while_the_model_loads(gateway):
    _slow(gateway)
    payload = {"model": "unsloth/c", "input": "hi", "stream": True}
    status, data = _call(gateway["gw"], "POST", "/v1/responses", payload)
    assert status == 200
    assert data.count(cx.KEEPALIVE_COMMENT) >= 3
    assert data.endswith(b'data: {"type": "response.completed"}\n\n')
    assert gateway["loads"] == ["unsloth/c"]


def test_gateway_reports_a_failed_streaming_load_as_response_failed(gateway):
    _slow(gateway, 0.05)
    gateway["failing"]["unsloth/c"] = "not enough memory"
    payload = {"model": "unsloth/c", "input": "hi", "stream": True}
    status, data = _call(gateway["gw"], "POST", "/v1/responses", payload)
    assert status == 200
    event = data.split(b"event: response.failed\ndata: ", 1)[1].split(b"\n\n", 1)[0]
    failed = json.loads(event)
    assert failed["type"] == "response.failed"
    assert "not enough memory" in failed["response"]["error"]["message"]
    assert gateway["upstream"].seen == []


def test_gateway_is_quiet_when_codex_hangs_up_during_a_load(gateway, capfd):
    import socket

    finished = _slow(gateway, 0.3)
    gw = gateway["gw"]
    data = json.dumps({"model": "unsloth/c", "input": "hi", "stream": True}).encode()
    sock = socket.create_connection(("127.0.0.1", gw.server_address[1]), timeout = 5)
    sock.sendall(
        b"POST /v1/responses HTTP/1.1\r\n"
        + f"Host: 127.0.0.1:{gw.server_address[1]}\r\n".encode()
        + f"Authorization: Bearer {STUDIO_KEY}\r\n".encode()
        + b"Content-Type: application/json\r\n"
        + f"Content-Length: {len(data)}\r\n\r\n".encode()
        + data
    )
    assert sock.recv(4096).startswith(b"HTTP/1.1 200")
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
    sock.close()
    assert finished.wait(5)
    threading.Event().wait(0.2)
    assert gateway["upstream"].seen == []
    assert "Traceback" not in capfd.readouterr().err


CODEX_OVERFLOW = (
    "Message too long: 25988 tokens exceeds the 8192-token context window. Try increasing"
)


def _failed_event(message: str) -> bytes:
    failed = {
        "type": "response.failed",
        "response": {
            "status": "failed",
            "error": {"code": "context_length_exceeded", "message": message},
        },
    }
    return b"event: response.failed\ndata: " + json.dumps(failed).encode() + b"\n\n"


def test_gateway_rewords_an_overflow_of_a_window_too_small_for_codex(gateway):
    created = b'event: response.created\ndata: {"type": "response.created"}\n\n'
    gateway["upstream"].replies.append(
        (200, "text/event-stream", created + _failed_event(CODEX_OVERFLOW))
    )
    payload = {"model": "unsloth/a", "input": "hi", "stream": True}
    status, data = _call(gateway["gw"], "POST", "/v1/responses", payload)
    assert status == 200 and data.startswith(created)
    event = json.loads(data.split(b"event: response.failed\ndata: ", 1)[1].split(b"\n\n", 1)[0])
    error = event["response"]["error"]
    # The code stays, so Codex still knows what kind of failure it was.
    assert error["code"] == "context_length_exceeded"
    assert "unsloth/a is loaded with a context of 8,192 tokens" in error["message"]
    assert "Codex needs 25,988 tokens" in error["message"] and "at least 32,768" in error["message"]


def test_gateway_leaves_a_long_codex_session_overflow_alone(gateway):
    long_session = "Message too long: 300000 tokens exceeds the 262144-token context window."
    gateway["upstream"].replies.append((200, "text/event-stream", _failed_event(long_session)))
    payload = {"model": "unsloth/a", "input": "hi", "stream": True}
    _, data = _call(gateway["gw"], "POST", "/v1/responses", payload)
    assert data == _failed_event(long_session)


def test_gateway_rewords_a_plain_400_overflow_for_codex(gateway):
    error = {
        "error": {
            "message": CODEX_OVERFLOW,
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
        }
    }
    gateway["upstream"].replies.append((400, "application/json", json.dumps(error).encode()))
    status, data = _call(
        gateway["gw"], "POST", "/v1/responses", {"model": "unsloth/a", "input": "hi"}
    )
    reply = json.loads(data)
    assert status == 400 and reply["error"]["code"] == "context_length_exceeded"
    assert "at least 32,768" in reply["error"]["message"]
