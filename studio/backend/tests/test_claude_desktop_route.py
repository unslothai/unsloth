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
        "mappings": {"claude-fable-5": ":loaded"},
        "routes": cd.route_names(),
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


# -- the per-slot model mapping ----------------------------------------------------------------

MAPPED = {
    "claude-fable-5": "unsloth/a-GGUF",
    "claude-opus-5": "unsloth/b-GGUF:UD-Q4_K_XL",
    "claude-sonnet-5": "unsloth/a-GGUF",
    "claude-haiku-4-5-20251001": ":loaded",
    "claude-sonnet-4-6": "unsloth/c-GGUF",
}


def test_mappings_are_saved_while_disconnected(mac, monkeypatch, tmp_path):
    client = _client(monkeypatch)
    response = client.put(f"{URL}/mappings", json = {"mappings": {**MAPPED, "bogus": "x"}})
    assert response.status_code == 200, response.text
    assert response.json()["status"] == "saved"
    assert response.json()["claude"]["mappings"] == MAPPED
    saved = tmp_path / "studio" / "auth" / "agents" / "app" / session.MAPPINGS_NAME
    assert json.loads(saved.read_text()) == {"mappings": MAPPED}
    assert saved.stat().st_mode & 0o777 == 0o600
    # Nothing touched Claude.
    assert mac["minted"] == [] and mac["opened"] == [] and mac["quit"] == []
    assert client.get(URL).json()["mappings"] == MAPPED


def test_at_least_one_slot_must_be_mapped(mac, monkeypatch):
    client = _client(monkeypatch)
    response = client.put(f"{URL}/mappings", json = {"mappings": {"claude-opus-5": ""}})
    assert response.status_code == 400
    assert "at least one" in response.json()["detail"]


def test_mappings_need_the_computer_running_studio(mac, monkeypatch):
    response = _client(monkeypatch, local = False).put(f"{URL}/mappings", json = {"mappings": MAPPED})
    assert response.status_code == 403


def test_a_broken_mappings_file_falls_back_to_following_the_loaded_model(mac, tmp_path):
    saved = tmp_path / "studio" / "auth" / "agents" / "app" / session.MAPPINGS_NAME
    saved.parent.mkdir(parents = True, exist_ok = True)
    saved.write_text("{not json")
    assert session.saved_mappings() == cd.DEFAULT_MAPPINGS


def test_the_connected_gateway_follows_the_saved_mapping(mac, monkeypatch):
    loads = []
    monkeypatch.setattr(cd, "load_model", lambda up, key, model_id: loads.append((key, model_id)))
    monkeypatch.setattr(session, "is_loaded", lambda model_id: model_id == "unsloth/a-GGUF")
    client = _client(monkeypatch)
    client.put(f"{URL}/mappings", json = {"mappings": MAPPED})
    assert client.post(f"{URL}/connect", json = {}).json()["status"] == "connected"
    gateway = session._live.gateway
    routes = [(route, model.model_id if model else None) for route, model, _ in gateway.routes()]
    assert routes == [
        ("claude-fable-5", "unsloth/a-GGUF"),
        ("claude-opus-5", "unsloth/b-GGUF:UD-Q4_K_XL"),
        ("claude-sonnet-5", "unsloth/a-GGUF"),
        ("claude-haiku-4-5-20251001", "unsloth/gemma-4-E2B-it-GGUF"),
        ("claude-sonnet-4-6", "unsloth/c-GGUF"),
    ]
    gateway.ready("unsloth/a-GGUF")
    gateway.ready("unsloth/b-GGUF:UD-Q4_K_XL")
    # Loaded with the session's own key, only when it isn't loaded yet.
    assert loads == [(STUDIO_KEY, "unsloth/b-GGUF:UD-Q4_K_XL")]


def test_changing_the_mapping_while_claude_is_open_asks_to_restart(mac, monkeypatch):
    client = _client(monkeypatch)
    client.post(f"{URL}/connect", json = {})
    assert mac["running"] is True
    response = client.put(f"{URL}/mappings", json = {"mappings": MAPPED})
    assert response.json()["status"] == "needs_restart"
    assert response.json()["claude"]["mappings"] == cd.DEFAULT_MAPPINGS  # nothing saved yet

    response = client.put(f"{URL}/mappings", json = {"mappings": MAPPED, "restart": False})
    assert response.json()["status"] == "pending_restart"
    assert [r[0] for r in session._live.gateway.routes()] == list(cd.ROUTE_IDS)
    assert mac["quit"] == []

    changed = {**MAPPED, "claude-opus-5": "unsloth/a-GGUF"}
    response = client.put(f"{URL}/mappings", json = {"mappings": changed, "restart": True})
    assert response.json()["status"] == "connected"
    assert len(mac["quit"]) == 1 and mac["opened"] == [mac["app"], mac["app"]]
    # The same gateway and key carry on: no new session.
    assert len(mac["minted"]) == 1


def test_a_closed_connected_claude_takes_the_mapping_without_asking(mac, monkeypatch):
    client = _client(monkeypatch)
    client.post(f"{URL}/connect", json = {})
    mac["running"] = False
    response = client.put(f"{URL}/mappings", json = {"mappings": MAPPED})
    assert response.json()["status"] == "connected"
    assert mac["quit"] == []


def test_models_lists_studios_chat_catalog_loaded_first(mac, monkeypatch):
    import routes.llama_compat as llama_compat

    async def catalog():
        return [
            {"id": "unsloth/z-GGUF", "loaded": False, "quant": "Q8_0"},
            {"id": "unsloth/a-GGUF", "loaded": False},
            {"id": "unsloth/m-GGUF", "loaded": True},
            {"id": ""},
        ]

    monkeypatch.setattr(llama_compat, "_chat_catalog", catalog)
    body = _client(monkeypatch).get(f"{URL}/models").json()
    assert body == [
        {"id": "unsloth/m-GGUF", "loaded": True, "quant": None},
        {"id": "unsloth/a-GGUF", "loaded": False, "quant": None},
        {"id": "unsloth/z-GGUF", "loaded": False, "quant": "Q8_0"},
    ]


def test_start_claude_opens_a_closed_connected_claude(mac, monkeypatch):
    client = _client(monkeypatch)
    client.post(f"{URL}/connect", json = {})
    mac["running"] = False
    response = client.put(f"{URL}/mappings", json = {"mappings": MAPPED, "restart": True})
    assert response.json()["status"] == "connected"
    assert mac["quit"] == [] and mac["opened"] == [mac["app"], mac["app"]]


# -- shared run settings (Agents → Run settings) ---------------------------------------------


def test_guard_fit_is_linear_between_two_priced_points():
    from routes.desktop_app_models import guard_fitted_context

    gib = 1024**3
    # 8k costs 10 GiB, 262k costs 30 GiB: a 20 GiB budget fits about half way.
    fitted = guard_fitted_context(262_144, 8_192, 10 * gib, 30 * gib, 20 * gib)
    assert fitted == (8_192 + (262_144 - 8_192) // 2) // 1024 * 1024
    # The full context fits: nothing to cap.
    assert guard_fitted_context(262_144, 8_192, 10 * gib, 30 * gib, 31 * gib) is None
    # Not even 8k fits: the loader's floor, never less.
    assert guard_fitted_context(262_144, 8_192, 10 * gib, 30 * gib, 5 * gib) == 8_192
    # No budget (not Apple Silicon): no guard to show.
    assert guard_fitted_context(262_144, 8_192, 10 * gib, 30 * gib, 0) is None


def _model_settings_stubs(
    monkeypatch,
    *,
    override,
    loaded = False,
):
    import routes.desktop_app_models as dam
    import routes.llama_compat as llama_compat
    import utils.openai_auto_switch_settings as overrides
    from models.inference import LoadRequest

    path = "/hub/models--unsloth--Qwen3.8-GGUF/snapshots/abc"

    async def catalog_row(model_id):
        return (
            model_id,
            {"id": model_id, "loaded": loaded, "context_length": 131_072 if loaded else None},
            True,
        )

    async def load_target(model_id):
        request = LoadRequest(model_path = path, gguf_variant = "UD-Q4_K_XL")
        request._override_alias_id = "unsloth/Qwen3.8-GGUF"
        return request, model_id

    async def guard(load_request, trained, fastapi_request, subject):
        return 24 * 1024**3, 98_304

    monkeypatch.setattr(llama_compat, "_catalog_row", catalog_row)
    monkeypatch.setattr(llama_compat, "_resolve_load_target", load_target)
    monkeypatch.setattr(overrides, "resolve_override_for_load", lambda *a, **k: ("k", override))
    monkeypatch.setattr(dam, "_gguf_header_facts", lambda *a: (262_144, True))
    monkeypatch.setattr(dam, "_guard_context", guard)
    return path


def test_model_settings_names_run_settings_key_bounds_and_guard(mac, monkeypatch):
    path = _model_settings_stubs(
        monkeypatch,
        override = {"custom_context_length": 65_536, "speculative_type": "mtp"},
        loaded = True,
    )
    body = (
        _client(monkeypatch)
        .get(f"{URL}/model-settings", params = {"model": "unsloth/Qwen3.8-GGUF:UD-Q4_K_XL"})
        .json()
    )
    # The key Run settings saves a loaded model under, and the one a load reads first.
    assert body["config_id"] == path
    assert body["alias_id"] == "unsloth/Qwen3.8-GGUF"
    assert body["gguf_variant"] == "UD-Q4_K_XL"
    # The values come from the SAME server override a gateway load applies.
    assert body["custom_context_length"] == 65_536
    assert body["speculative_type"] == "mtp"
    assert body["trained_context"] == 262_144
    assert body["mtp_available"] is True
    assert body["guard_context"] == 98_304
    assert body["guard_budget_bytes"] == 24 * 1024**3
    assert body["loaded"] is True and body["loaded_context"] == 131_072


def test_model_settings_without_an_override_is_auto(mac, monkeypatch):
    _model_settings_stubs(monkeypatch, override = {})
    body = (
        _client(monkeypatch)
        .get(f"{URL}/model-settings", params = {"model": "unsloth/Qwen3.8-GGUF"})
        .json()
    )
    assert body["custom_context_length"] is None
    assert body["speculative_type"] is None


def test_model_settings_needs_the_owner_ui_session(mac, monkeypatch):
    _model_settings_stubs(monkeypatch, override = {})
    client = _client(monkeypatch, via_api_key = True)
    assert client.get(f"{URL}/model-settings", params = {"model": "x"}).status_code == 403


def test_guard_prices_the_load_with_its_saved_settings(monkeypatch):
    import asyncio

    import routes.desktop_app_models as dam
    import routes.inference as inference
    from core.inference.llama_cpp import LlamaCppBackend
    from models.inference import LoadRequest

    gib = 1024**3
    asked = []

    async def estimate(request, fastapi_request, subject):
        asked.append((request.n_ctx, request.cache_type_kv, request.speculative_type))
        gpu = 10 * gib + (request.n_ctx - 8_192) * (20 * gib) // (262_144 - 8_192)
        return _types.SimpleNamespace(available = True, gpu_bytes = gpu, total_bytes = gpu)

    monkeypatch.setattr(inference, "estimate_memory", estimate)
    monkeypatch.setattr(
        LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 20 * gib)
    )
    # No vm_stat: the guard falls back to the upstream budget unchanged.
    monkeypatch.setattr(dam, "_vm_stat_bytes", lambda: None)
    load = LoadRequest(
        model_path = "/m", gguf_variant = "Q4", cache_type_kv = "q8_0", speculative_type = "mtp"
    )
    budget, fitted = asyncio.run(dam._guard_context(load, 262_144, None, "owner"))
    assert budget == 20 * gib
    assert 130_000 < fitted < 140_000
    assert asked == [(262_144, "q8_0", "mtp"), (8_192, "q8_0", "mtp")]
    # Off Apple Silicon there is no guard budget, and nothing is priced.
    asked.clear()
    monkeypatch.setattr(
        LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0)
    )
    assert asyncio.run(dam._guard_context(load, 262_144, None, "owner")) == (None, None)
    assert asked == []


# -- the guard budget counts a resident model the load replaces --------------------------------


def _mac_budget_inputs(monkeypatch, *, recommended, available):
    import psutil

    import utils.hardware as hardware_pkg
    import utils.hardware.hardware as hardware

    monkeypatch.setattr(hardware_pkg, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hardware, "_metal_recommended_working_set_bytes", lambda: recommended)
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: _types.SimpleNamespace(total = 512 * 1024**3, available = available),
    )


def test_metal_budget_counts_reclaimable_memory_up_to_the_device_ceiling(
    monkeypatch, real_apple_metal_budget
):
    from core.inference.llama_cpp import LlamaCppBackend

    # Max's 512 GB Mac Studio, measured with GLM-5.3-Flash UD-Q6_K_XL loaded at 1M context:
    # Metal working set 464 GiB, psutil available 188.19 GB, the llama-server's RSS 322.79 GB.
    recommended, available, resident = 498_216_206_336, 188_190_539_776, 322_790_293_504
    _mac_budget_inputs(monkeypatch, recommended = recommended, available = available)
    # Right now: 0.85 x available, the ~149 GiB the guard showed (153.4 GiB at the time).
    now = LlamaCppBackend._apple_metal_memory_budget_bytes()
    assert now == int(available * 0.85)
    assert 148 * 1024**3 < now < 150 * 1024**3
    # With the resident server credited back, the device ceiling binds: 0.85 x 464 GiB.
    credited = LlamaCppBackend._apple_metal_memory_budget_bytes(resident)
    assert credited == int(recommended * 0.85)
    # A small credit only adds itself; a negative one is ignored.
    gib = 1024**3
    assert LlamaCppBackend._apple_metal_memory_budget_bytes(10 * gib) == int(
        (available + 10 * gib) * 0.85
    )
    assert LlamaCppBackend._apple_metal_memory_budget_bytes(-5 * gib) == now


class _FakeLlama:
    def __init__(
        self,
        ident,
        variant,
        mib,
        loaded = True,
    ):
        self.model_identifier = ident
        self.hf_variant = variant
        self.is_loaded = loaded
        self._mib = mib

    def resident_memory_mib(self):
        return self._mib


def _resident_slots(monkeypatch, backends, mode):
    import core.inference.model_slots as model_slots
    import routes.inference as inference
    import utils.multi_model_settings as multi

    slots = list(backends)[1:]
    current = {"slot": None}

    def in_slot(slot, fn):
        current["slot"] = slot
        return fn()

    monkeypatch.setattr(model_slots, "visible", lambda: slots)
    monkeypatch.setattr(model_slots, "in_slot", in_slot)
    monkeypatch.setattr(inference, "get_llama_cpp_backend", lambda: backends[current["slot"]])
    monkeypatch.setattr(multi, "get_multi_model_mode", lambda: mode)


def test_replaced_resident_bytes_credits_only_what_the_load_stops(monkeypatch):
    import routes.desktop_app_models as dam

    mib = 1024 * 1024
    glm = "unsloth/GLM-5.3-Flash-GGUF"
    backends = {
        None: _FakeLlama(glm, "UD-Q6_K_XL", 300_000),
        "slot-1": _FakeLlama("unsloth/Qwen3.8-GGUF", "UD-Q4_K_XL", 20_000),
        "slot-2": _FakeLlama("unsloth/Other-GGUF", "Q8_0", 9_000, loaded = False),
    }
    # Keep multiple loaded on: only the server already serving this model and quant.
    _resident_slots(monkeypatch, backends, "fit")
    assert dam._replaced_resident_bytes(glm, "UD-Q6_K_XL") == 300_000 * mib
    assert dam._replaced_resident_bytes(glm, "ud-q6_k_xl") == 300_000 * mib
    # Another quant of the same repo loads beside it: nothing comes back.
    assert dam._replaced_resident_bytes(glm, "UD-Q4_K_XL") == 0
    assert dam._replaced_resident_bytes("unsloth/New-GGUF", "Q4_K_M") == 0
    # Keep multiple loaded Off: the load replaces every loaded server (not the idle slot).
    _resident_slots(monkeypatch, backends, "off")
    assert dam._replaced_resident_bytes("unsloth/New-GGUF", "Q4_K_M") == 320_000 * mib


def test_replaced_resident_bytes_never_raises(monkeypatch):
    import routes.desktop_app_models as dam

    class Broken(_FakeLlama):
        def resident_memory_mib(self):
            raise RuntimeError("gone")

    _resident_slots(monkeypatch, {None: Broken("m", None, 1)}, "off")
    assert dam._replaced_resident_bytes("m", None) == 0


def test_guard_budget_includes_the_replaced_resident_model(monkeypatch):
    import asyncio

    import routes.desktop_app_models as dam
    import routes.inference as inference
    from core.inference.llama_cpp import LlamaCppBackend
    from models.inference import LoadRequest

    gib = 1024**3
    credits = []

    async def estimate(request, fastapi_request, subject):
        gpu = 300 * gib + (request.n_ctx - 8_192) * (60 * gib) // (1_048_576 - 8_192)
        return _types.SimpleNamespace(available = True, gpu_bytes = gpu, total_bytes = gpu)

    def budget(reclaimable_bytes = 0):
        credits.append(reclaimable_bytes)
        return 150 * gib + int(reclaimable_bytes * 0.85)

    monkeypatch.setattr(inference, "estimate_memory", estimate)
    monkeypatch.setattr(LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(budget))
    monkeypatch.setattr(dam, "_replaced_resident_bytes", lambda path, variant: 300 * gib)
    monkeypatch.setattr(dam, "_vm_stat_bytes", lambda: None)
    load = LoadRequest(model_path = "/glm", gguf_variant = "UD-Q6_K_XL")
    got, fitted = asyncio.run(dam._guard_context(load, 1_048_576, None, "owner"))
    # 150 GiB free + 255 GiB back from the resident copy: the full 1M context fits.
    assert credits == [300 * gib]
    assert got == 405 * gib and fitted is None
    # Nothing resident: the plain right-now budget, which caps the context.
    credits.clear()
    monkeypatch.setattr(dam, "_replaced_resident_bytes", lambda path, variant: 0)
    got, fitted = asyncio.run(dam._guard_context(load, 1_048_576, None, "owner"))
    assert credits == [0] and got == 150 * gib and fitted == 8_192


# -- the guard budget is Metal's working set less what others hold, not psutil's available ----

_GIB = 1024**3
# Max's Mac Studio: 512 GiB unified memory, Metal recommendedMaxWorkingSetSize 464 GiB.
_TOTAL, _WORKING_SET = 512 * _GIB, 464 * _GIB
_MARGIN = int(_WORKING_SET * 0.05)


def _vm(
    wired,
    anonymous,
    file_backed = 0,
    purgeable = 0,
    compressor = 0,
):
    return {
        "Pages wired down": wired,
        "Anonymous pages": anonymous,
        "File-backed pages": file_backed,
        "Pages purgeable": purgeable,
        "Pages occupied by compressor": compressor,
    }


def test_cold_reload_after_an_unload_gets_about_the_whole_working_set():
    from routes.desktop_app_models import metal_guard_budget, unified_memory_held_bytes

    # Right after GLM-5.3-Flash Q6 was unloaded: ~170 GiB of its file pages still sit in the
    # page cache as "active", so psutil's available read 331 GiB and the old guard 281 GiB.
    stats = _vm(wired = 11 * _GIB, anonymous = 36 * _GIB, file_backed = 170 * _GIB, purgeable = _GIB)
    held = unified_memory_held_bytes(stats)
    assert held == 46 * _GIB
    budget = metal_guard_budget(_WORKING_SET, _TOTAL, held)
    assert budget == min(_WORKING_SET, _TOTAL - 46 * _GIB) - _MARGIN
    assert budget > 440 * _GIB > int(331 * _GIB * 0.85)


def test_memory_other_apps_really_hold_still_caps_the_budget():
    from routes.desktop_app_models import (
        guard_fitted_context,
        metal_guard_budget,
        unified_memory_held_bytes,
    )

    # Other apps hold 200 GiB wired (GPU buffers) and 100 GiB anonymous, 8 GiB compressed.
    held = unified_memory_held_bytes(_vm(200 * _GIB, 100 * _GIB, compressor = 8 * _GIB))
    budget = metal_guard_budget(_WORKING_SET, _TOTAL, held)
    assert budget == _TOTAL - 308 * _GIB - _MARGIN
    assert 180 * _GIB < budget < 182 * _GIB
    # A 300 GiB model whose full 1M context needs 360 GiB is cut, and says so.
    fitted = guard_fitted_context(1_048_576, 8_192, 300 * _GIB, 360 * _GIB, budget)
    assert fitted == 8_192


def test_replaced_resident_model_is_still_counted_back():
    from routes.desktop_app_models import metal_guard_budget, unified_memory_held_bytes

    # GLM loaded: its ~298 GiB RSS is Metal-wired. Reloading it frees that first.
    held = unified_memory_held_bytes(_vm(322 * _GIB, 36 * _GIB, purgeable = _GIB))
    resident = 298 * _GIB
    assert metal_guard_budget(_WORKING_SET, _TOTAL, held) == _TOTAL - held - _MARGIN
    warm = metal_guard_budget(_WORKING_SET, _TOTAL, held, resident)
    assert warm == _TOTAL - (held - resident) - _MARGIN
    assert warm > 425 * _GIB
    # A credit larger than what is held never makes the budget exceed Metal's ceiling.
    assert metal_guard_budget(_WORKING_SET, _TOTAL, held, 10**15) == _WORKING_SET - _MARGIN


def test_small_mac_margin_floor_and_unreadable_inputs():
    from routes.desktop_app_models import metal_guard_budget, unified_memory_held_bytes

    # 16 GB Mac, ~10.7 GiB working set: the margin floor is 4 GiB, not 5 %.
    ws = int(10.7 * _GIB)
    assert metal_guard_budget(ws, 16 * _GIB, 3 * _GIB) == ws - 4 * _GIB
    assert metal_guard_budget(ws, 16 * _GIB, 15 * _GIB) == 0
    assert metal_guard_budget(0, 16 * _GIB, 0) == 0
    assert unified_memory_held_bytes(None) is None
    assert unified_memory_held_bytes({"Pages wired down": 1}) is None


def test_vm_stat_is_parsed_in_bytes(monkeypatch):
    import subprocess

    import routes.desktop_app_models as dam

    text = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                                  7194179.\n"
        "Pages wired down:                           21136729.\n"
        "Pages purgeable:                               70026.\n"
        "File-backed pages:                           2727023.\n"
        "Anonymous pages:                             2361779.\n"
        '"Translation faults":                     1492729000.\n'
        "Pages occupied by compressor:                   1891.\n"
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _types.SimpleNamespace(stdout = text, returncode = 0)
    )
    stats = dam._vm_stat_bytes()
    assert stats["Pages wired down"] == 21_136_729 * 16384
    assert stats["Anonymous pages"] == 2_361_779 * 16384
    assert stats["Translation faults"] == 1_492_729_000 * 16384
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _types.SimpleNamespace(stdout = "", returncode = 1)
    )
    assert dam._vm_stat_bytes() is None


def test_guard_budget_off_apple_silicon_is_zero_and_reads_nothing(monkeypatch):
    import routes.desktop_app_models as dam
    from core.inference.llama_cpp import LlamaCppBackend

    def tripwire():
        raise AssertionError("vm_stat read off Apple Silicon")

    monkeypatch.setattr(
        LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda *a: 0)
    )
    monkeypatch.setattr(dam, "_vm_stat_bytes", tripwire)
    assert dam._guard_budget_bytes(0) == 0
    assert dam._guard_budget_bytes(300 * _GIB) == 0


def test_guard_budget_uses_vm_stat_and_falls_back_to_upstream(monkeypatch):
    import psutil

    import routes.desktop_app_models as dam
    import utils.hardware.hardware as hardware
    from core.inference.llama_cpp import LlamaCppBackend

    calls = []

    def upstream(reclaimable_bytes = 0):
        calls.append(reclaimable_bytes)
        return 281 * _GIB

    monkeypatch.setattr(LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(upstream))
    monkeypatch.setattr(hardware, "_metal_recommended_working_set_bytes", lambda: _WORKING_SET)
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: _types.SimpleNamespace(total = _TOTAL, available = 0)
    )
    monkeypatch.setattr(dam, "_vm_stat_bytes", lambda: _vm(11 * _GIB, 36 * _GIB, purgeable = _GIB))
    assert dam._guard_budget_bytes(0) == _WORKING_SET - _MARGIN
    monkeypatch.setattr(dam, "_vm_stat_bytes", lambda: _vm(322 * _GIB, 36 * _GIB))
    assert dam._guard_budget_bytes(298 * _GIB) == _TOTAL - 60 * _GIB - _MARGIN
    assert calls == [0, 298 * _GIB]
    # vm_stat unreadable: the upstream figure, unchanged.
    monkeypatch.setattr(dam, "_vm_stat_bytes", lambda: None)
    assert dam._guard_budget_bytes(0) == 281 * _GIB
