# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Claude Agent inside Studio: the Developer mode toggle, the SDK installer, owner-only access,
hidden connections while off, ignored routing fields, media pickers and scheduled tasks."""

import asyncio
import subprocess
import sys
import threading
import types
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import (
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
)
from core.inference import claude_agent
from core.inference.providers import (
    CLAUDE_AGENT_OFF,
    claude_agent_enabled,
    get_connectable_provider_info,
    list_available_providers,
)
from routes import providers
from storage import credential_secrets, providers_db
from utils import claude_agent_runtime, claude_agent_settings
from utils.account_context import (
    OWNER,
    AccountContext,
    bind_account,
    current_account,
    reset_account,
)

ALICE = AccountContext("a" * 32, "alice")
ENV = claude_agent_settings.CLAUDE_AGENT_ENV
READY = {
    "ready": True,
    "version": "3.1.0",
    "account": "Claude Max",
    "message": None,
    "models": [{"id": "opus", "display_name": "Opus 5.5", "efforts": ["low", "high"]}],
}


@pytest.fixture
def store(monkeypatch):
    per_account: dict[str, dict] = {}
    studio_db = types.ModuleType("storage.studio_db")

    def get_app_setting(key, fallback = None):
        return per_account.get(current_account().account_id, {}).get(key, fallback)

    def upsert_app_settings(settings, **_kw):
        per_account.setdefault(current_account().account_id, {}).update(settings)
        return dict(per_account[current_account().account_id])

    studio_db.get_app_setting = get_app_setting
    studio_db.upsert_app_settings = upsert_app_settings
    monkeypatch.setitem(sys.modules, "storage.studio_db", studio_db)
    monkeypatch.delenv(ENV, raising = False)
    return per_account


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    for module in (credential_secrets, providers_db):
        monkeypatch.setattr(module, "_schema_ready", set())
    monkeypatch.setattr(
        credential_secrets, "get_or_create_credential_encryption_key", lambda: b"k" * 32
    )
    monkeypatch.setattr(providers, "current_credential_write", lambda credential: nullcontext())

    async def status():
        return dict(READY)

    monkeypatch.setattr(providers, "claude_agent_status", status)
    monkeypatch.setenv(ENV, "1")


def _bound(account):
    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    return subject


def providers_client(account):
    app = FastAPI()
    app.dependency_overrides[get_current_subject] = _bound(account)
    app.dependency_overrides[get_current_credential] = lambda: (account.username, None)
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.include_router(providers.router, prefix = "/providers")
    return TestClient(app)


def settings_client(account):
    from routes import settings as settings_route

    app = FastAPI()
    app.dependency_overrides[get_current_subject] = _bound(account)
    app.include_router(settings_route.router, prefix = "/api/settings")
    return TestClient(app)


def _create(client, **extra):
    return client.post(
        "/providers/",
        json = {"provider_type": "claude_agent", "display_name": "Claude Agent", **extra},
    )


# -- 1. Developer mode toggle -----------------------------------------------------------------


def test_the_saved_setting_turns_claude_agent_on(store):
    assert claude_agent_enabled() is False
    with settings_client(OWNER) as client:
        before = client.get("/api/settings/developer-mode").json()
        on = client.put("/api/settings/developer-mode", json = {"claude_agent": True})
    assert before["claude_agent"] is False and before["claude_agent_env_override"] is None
    assert on.status_code == 200, on.text
    assert on.json()["claude_agent"] is True and on.json()["video_cpu"] is False
    assert store[OWNER.account_id][claude_agent_settings.CLAUDE_AGENT_SETTING_KEY] is True
    assert claude_agent_enabled() is True
    assert "claude_agent" in {row["provider_type"] for row in list_available_providers()}


@pytest.mark.parametrize("env, saved, effective", [("1", False, True), ("0", True, False)])
def test_the_env_var_overrides_the_setting_and_says_so(monkeypatch, store, env, saved, effective):
    store[OWNER.account_id] = {claude_agent_settings.CLAUDE_AGENT_SETTING_KEY: saved}
    monkeypatch.setenv(ENV, env)
    with settings_client(OWNER) as client:
        body = client.get("/api/settings/developer-mode").json()
    assert body["claude_agent"] is effective
    assert body["claude_agent_setting"] is saved
    assert body["claude_agent_env_override"] is effective
    assert claude_agent_enabled() is effective


def test_a_managed_account_cannot_turn_it_on(monkeypatch, store):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    with settings_client(ALICE) as client:
        response = client.put("/api/settings/developer-mode", json = {"claude_agent": True})
    assert response.status_code == 403
    assert claude_agent_enabled() is False


def test_toggling_one_switch_leaves_the_other(store):
    with settings_client(OWNER) as client:
        client.put("/api/settings/developer-mode", json = {"video_cpu": True})
        body = client.put("/api/settings/developer-mode", json = {"claude_agent": True}).json()
        empty = client.put("/api/settings/developer-mode", json = {})
    assert body["video_cpu_setting"] is True and body["claude_agent"] is True
    assert empty.status_code == 400


def test_the_off_message_points_at_the_toggle():
    assert "Developer mode" in CLAUDE_AGENT_OFF and ENV in CLAUDE_AGENT_OFF


# -- 2. SDK installer ---------------------------------------------------------------------------


def test_the_installer_targets_studio_python_with_constraints(monkeypatch, tmp_path):
    calls = []
    installed = {"done": False}
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(claude_agent_runtime, "_find_uv", lambda: "/usr/bin/uv")
    monkeypatch.setattr(claude_agent_runtime, "hf_env_offline", lambda: False)
    sdk = object()
    monkeypatch.setattr(
        claude_agent_runtime, "_import_sdk", lambda: sdk if installed["done"] else None
    )
    invalidated = []
    monkeypatch.setattr(
        claude_agent_runtime.importlib, "invalidate_caches", lambda: invalidated.append(1)
    )

    def run(command, **kwargs):
        calls.append((command, kwargs))
        installed["done"] = True

    monkeypatch.setattr(claude_agent_runtime.subprocess, "run", run)
    assert claude_agent_runtime.ensure_claude_agent_sdk() is sdk
    ((command, kwargs),) = calls
    assert command[:5] == ["/usr/bin/uv", "pip", "install", "--python", sys.executable]
    assert "--constraint" in command and command[-1] == "claude-agent-sdk>=0.2.178"
    assert kwargs["check"] is True and 0 < kwargs["timeout"] <= 300
    assert "PIP_CONFIG_FILE" in kwargs["env"]
    assert invalidated, "import caches must be invalidated so no restart is needed"


def test_an_installer_failure_is_reported_without_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(claude_agent_runtime, "_find_uv", lambda: None)
    monkeypatch.setattr(claude_agent_runtime, "hf_env_offline", lambda: False)
    monkeypatch.setattr(claude_agent_runtime, "_import_sdk", lambda: None)

    def run(command, **kwargs):
        raise subprocess.CalledProcessError(
            1,
            command,
            stderr = b"Looking in http://user:secret@proxy.local/simple\nNo matching distribution",
        )

    monkeypatch.setattr(claude_agent_runtime.subprocess, "run", run)
    with pytest.raises(claude_agent_runtime.ClaudeAgentSdkUnavailable) as failure:
        claude_agent_runtime.ensure_claude_agent_sdk()
    text = str(failure.value)
    assert "No matching distribution" in text and "secret" not in text
    assert claude_agent_runtime.MANUAL_COMMAND in text


def test_offline_mode_refuses_to_download(monkeypatch):
    monkeypatch.setattr(claude_agent_runtime, "_import_sdk", lambda: None)
    monkeypatch.setenv("UV_OFFLINE", "1")
    with pytest.raises(claude_agent_runtime.ClaudeAgentSdkUnavailable, match = "offline"):
        claude_agent_runtime.ensure_claude_agent_sdk()


def test_the_install_route_reports_the_status_again(monkeypatch, isolated):
    installs = []
    monkeypatch.setattr(claude_agent_runtime, "ensure_claude_agent_sdk", lambda: installs.append(1))
    with providers_client(OWNER) as client:
        response = client.post("/providers/claude-agent/install-sdk")
    assert response.status_code == 200, response.text
    assert installs == [1] and response.json()["ready"] is True
    assert "sdk_missing" in response.json()


def test_the_install_route_shows_the_error(monkeypatch, isolated):
    def fail():
        raise claude_agent_runtime.ClaudeAgentSdkUnavailable("The install timed out.")

    monkeypatch.setattr(claude_agent_runtime, "ensure_claude_agent_sdk", fail)
    with providers_client(OWNER) as client:
        response = client.post("/providers/claude-agent/install-sdk")
    assert response.status_code == 503
    assert response.json()["detail"] == "The install timed out."


def test_the_install_route_is_owner_only(monkeypatch, isolated):
    monkeypatch.setattr(
        claude_agent_runtime, "ensure_claude_agent_sdk", lambda: pytest.fail("installed")
    )
    with providers_client(ALICE) as client:
        assert client.post("/providers/claude-agent/install-sdk").status_code == 403


# -- 3. managed accounts --------------------------------------------------------------------------


def test_a_managed_account_never_sees_or_creates_claude_agent(isolated):
    with providers_client(ALICE) as client:
        registry = client.get("/providers/registry").json()
        created = _create(client)
        tested = client.post("/providers/test", json = {"provider_type": "claude_agent"})
    assert "claude_agent" not in {row["provider_type"] for row in registry}
    assert created.status_code == 403
    assert tested.status_code in (400, 403)
    token = bind_account(ALICE)
    try:
        assert get_connectable_provider_info("claude_agent") is None
    finally:
        reset_account(token)
    assert get_connectable_provider_info("claude_agent") is not None


def test_a_managed_account_cannot_update_a_claude_agent_row(isolated):
    token = bind_account(ALICE)
    try:
        providers_db.create_provider(
            id = "c" * 16,
            provider_type = "claude_agent",
            display_name = "Claude Agent",
            base_url = "",
            models = ["opus"],
        )
    finally:
        reset_account(token)
    with providers_client(ALICE) as client:
        response = client.put(f"/providers/{'c' * 16}", json = {"display_name": "Mine"})
    assert response.status_code == 403


# -- 4. hidden while off ----------------------------------------------------------------------------


def test_saved_connections_hide_while_off_and_come_back(monkeypatch, isolated):
    with providers_client(OWNER) as client:
        created = _create(client, models = ["opus"])
        assert created.status_code == 201, created.text
        monkeypatch.setenv(ENV, "0")
        hidden = client.get("/providers/").json()
        monkeypatch.setenv(ENV, "1")
        shown = client.get("/providers/").json()
    assert [row["provider_type"] for row in hidden] == []
    assert [row["id"] for row in shown] == [created.json()["id"]]


# -- 5. unused routing fields ------------------------------------------------------------------------


def test_create_and_update_drop_endpoint_key_and_headers(isolated):
    # "not-a-key" would fail decryption with a 400 if it were read.
    with providers_client(OWNER) as client:
        created = _create(
            client,
            base_url = "https://example.com/v1",
            encrypted_api_key = "not-a-key",
            extra_headers = {"X-Secret": "1"},
        )
        assert created.status_code == 201, created.text
        provider_id = created.json()["id"]
        updated = client.put(
            f"/providers/{provider_id}",
            json = {
                "display_name": "Renamed",
                "base_url": "https://example.com/v1",
                "encrypted_api_key": "not-a-key",
                "extra_headers": {"X-Secret": "1"},
            },
        )
    assert updated.status_code == 200, updated.text
    row = providers_db.get_provider(provider_id)
    assert row["display_name"] == "Renamed"
    assert row["base_url"] == ""
    assert not row.get("extra_headers")
    assert credential_secrets.provider_api_key_saved(provider_id) is False


def test_models_route_lists_claude_agent_models_with_display_names(monkeypatch, isolated):
    async def models():
        return [{"id": "opus", "display_name": "Opus 5.5", "context_window": 1_000_000}]

    monkeypatch.setattr(claude_agent, "claude_agent_models", models, raising = False)
    with providers_client(OWNER) as client:
        listed = client.post("/providers/models", json = {"provider_type": "claude_agent"})
    assert listed.status_code == 200, listed.text
    assert listed.json()[0]["display_name"] == "Opus 5.5"
    assert listed.json()[0]["context_length"] == 1_000_000


# -- 6. voice and image pickers -------------------------------------------------------------------------


@pytest.fixture
def claude_agent_row(monkeypatch):
    from routes import inference

    config = {
        "id": "c" * 16,
        "provider_type": "claude_agent",
        "display_name": "Claude Agent",
        "base_url": "",
        "is_enabled": True,
        "models": ["opus"],
    }
    monkeypatch.setattr(inference.providers_db, "get_provider", lambda _id: config)
    return inference


def test_speech_refuses_a_claude_agent_connection(claude_agent_row):
    body = SimpleNamespace(provider_id = "c" * 16, model = "opus", voice = "alloy")
    with pytest.raises(HTTPException) as refused:
        asyncio.run(claude_agent_row._external_tts_speech(body, None))
    assert refused.value.status_code == 400
    assert "chat only" in refused.value.detail


def test_transcription_refuses_a_claude_agent_connection(claude_agent_row):
    with pytest.raises(HTTPException) as refused:
        asyncio.run(
            claude_agent_row._external_stt_transcription(
                provider_id = "c" * 16,
                raw = b"RIFF",
                filename = "a.wav",
                content_type = "audio/wav",
                model = "opus",
                language = None,
                response_format = "json",
                encrypted_api_key = None,
                request = None,
            )
        )
    assert refused.value.status_code == 400
    assert "chat only" in refused.value.detail


# -- 11. scheduled tasks -------------------------------------------------------------------------------


def test_a_scheduled_run_with_claude_agent_off_fails_with_the_reason(monkeypatch):
    from core.scheduled_tasks import scheduler
    from routes import inference

    config = {"id": "c" * 16, "provider_type": "claude_agent", "is_enabled": True}
    monkeypatch.setattr(providers_db, "get_provider", lambda _id: config)
    model = f"external::{'c' * 16}::opus"
    monkeypatch.setattr(inference, "_connection_model_ref", lambda m: ("c" * 16, "opus"))
    monkeypatch.setenv(ENV, "0")
    with pytest.raises(scheduler.RunFailed) as failure:
        scheduler._model_fields(model, may_load = False)
    assert failure.value.message == CLAUDE_AGENT_OFF

    monkeypatch.setenv(ENV, "1")
    fields, provider, load = scheduler._model_fields(model, may_load = False)
    assert fields["provider_id"] == "c" * 16 and fields["stream"] is True
    assert provider is config and load is False


def test_the_scheduler_request_is_admitted_as_the_owner_ui():
    from core.scheduled_tasks import scheduler
    from routes import inference

    request = scheduler._producer_request(None, "run-1", threading.Event(), "UTC")
    # The proxy refuses Claude Agent to API keys and keyless callers; a scheduled run is neither.
    assert inference._request_has_api_key(request) is False
