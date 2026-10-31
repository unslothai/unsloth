# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import asyncio
import http.server
import json
import sqlite3
import threading
from contextlib import nullcontext

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import (
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
)
from core.inference import providers as providers_core
from hub.services.models import account_access as access
from routes import provider_credentials, providers
from storage import credential_secrets, providers_db, studio_db
from utils import managed_provider_url_settings as mpu
from utils.account_context import (
    OWNER,
    AccountContext,
    arun_as,
    bind_account,
    reset_account,
    run_as,
)
from utils.paths import studio_db_path

ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")
UPSTREAM_MODELS = ["unsloth/Qwen3-8B-GGUF", "unsloth/gemma-3-4b-it-GGUF", "other/private-model"]


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.delenv(providers_core._BLOCK_PRIVATE_ENV, raising = False)
    mpu.forget_cached_setting()
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    for module in (credential_secrets, providers_db, studio_db):
        monkeypatch.setattr(module, "_schema_ready", set(), raising = False)
    monkeypatch.setattr(
        credential_secrets, "get_or_create_credential_encryption_key", lambda: b"k" * 32
    )
    monkeypatch.setattr(providers, "current_credential_write", lambda credential: nullcontext())
    monkeypatch.setattr(access, "_allowlists", {})
    providers_core._dns_cache.clear()
    yield
    providers_core._dns_cache.clear()


@pytest.fixture
def upstream():
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("Authorization"))
            body = json.dumps({"data": [{"id": model} for model in UPSTREAM_MODELS]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            seen.append(f"POST {self.path}")
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", seen
    finally:
        server.shutdown()
        server.server_close()


def client_for(account):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[get_current_credential] = lambda: (account.username, None)
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.include_router(providers.router, prefix = "/providers")
    return TestClient(app)


def owner_connection(
    base_url = "https://api.example.com/v1",
    *,
    shared = True,
    models = None,
    available_models = None,
):
    with client_for(OWNER) as client:
        created = client.post(
            "/providers/",
            json = {
                "provider_type": "custom",
                "display_name": "Household Gufo",
                "base_url": base_url,
                "models": models if models is not None else UPSTREAM_MODELS[:2],
                "available_models": available_models or [],
                "shared": shared,
            },
        )
    assert created.status_code == 201, created.text
    provider_id = created.json()["id"]
    run_as(OWNER, credential_secrets.save_provider_api_key, provider_id, "owner-secret")
    return provider_id


def allow_only(monkeypatch, *patterns):
    allowlist = tuple(pattern.lower() for pattern in patterns)
    monkeypatch.setattr(
        access, "model_allowlist", lambda: None if access.is_owner_context() else allowlist
    )


def listed(account):
    with client_for(account) as client:
        response = client.get("/providers/")
    assert response.status_code == 200, response.text
    return response.json()


def test_a_shared_connection_reaches_every_account_with_the_owner_key():
    provider_id = owner_connection()
    for account in (ALICE, BOB):
        [row] = listed(account)
        assert row["id"] == provider_id
        assert row["shared"] is True and row["shared_by_owner"] is True
        assert row["has_api_key"] is True and row["auth_status"] == "connected"
        assert "owner-secret" not in json.dumps(row)
        assert (
            run_as(account, provider_credentials.resolve_provider_api_key_or_400, provider_id, None)
            == "owner-secret"
        )
    [owner_row] = listed(OWNER)
    assert owner_row["shared"] is True and owner_row["shared_by_owner"] is False


def test_an_unshared_connection_stays_private_and_unsharing_revokes():
    private_id = owner_connection(shared = False)
    assert listed(ALICE) == []
    assert run_as(ALICE, credential_secrets.get_provider_api_key, private_id) is None

    shared_id = owner_connection()
    assert [row["id"] for row in listed(ALICE)] == [shared_id]
    with client_for(OWNER) as client:
        assert client.put(f"/providers/{shared_id}", json = {"shared": False}).status_code == 200
    assert listed(ALICE) == []
    assert run_as(ALICE, credential_secrets.get_provider_api_key, shared_id) is None
    assert run_as(ALICE, providers_db.get_provider, shared_id) is None


def test_a_base_url_carrying_a_credential_cannot_be_shared():
    provider_id = owner_connection(shared = False)
    run_as(OWNER, providers_db.update_provider, provider_id, base_url = "https://user:pass@[bad/v1")
    with client_for(OWNER) as client:
        assert client.put(f"/providers/{provider_id}", json = {"shared": True}).status_code == 400
    for base_url in ("https://user:pass@gateway.example.com/v1", "https://gw.example.com/v1?key=s"):
        with client_for(OWNER) as client:
            refused = client.post(
                "/providers/",
                json = {
                    "provider_type": "custom",
                    "display_name": "Gateway",
                    "base_url": base_url,
                    "shared": True,
                },
            )
        assert refused.status_code == 400, base_url
    provider_id = owner_connection()
    with client_for(OWNER) as client:
        moved = client.put(
            f"/providers/{provider_id}", json = {"base_url": "https://gw.example.com/v1?key=s"}
        )
        assert moved.status_code == 400
        unshared = client.put(
            f"/providers/{provider_id}",
            json = {"base_url": "https://gw.example.com/v1?key=s", "shared": False},
        )
        assert unshared.status_code == 200
    assert listed(ALICE) == []


def test_other_accounts_can_use_but_never_change_a_shared_connection():
    provider_id = owner_connection()
    with client_for(ALICE) as client:
        assert (
            client.put(f"/providers/{provider_id}", json = {"display_name": "mine"}).status_code
            == 403
        )
        assert (
            client.put(
                f"/providers/{provider_id}/api-key/migrate", json = {"encrypted_api_key": "x"}
            ).status_code
            == 403
        )
        assert client.delete(f"/providers/{provider_id}").status_code == 403
    [row] = listed(OWNER)
    assert row["display_name"] == "Household Gufo"
    assert run_as(OWNER, credential_secrets.get_provider_api_key, provider_id) == "owner-secret"


def test_only_the_owner_can_share_and_never_a_chatgpt_subscription():
    with client_for(ALICE) as client:
        refused = client.post(
            "/providers/",
            json = {"provider_type": "openai", "display_name": "Mine", "shared": True},
        )
        assert refused.status_code == 403
        own = client.post("/providers/", json = {"provider_type": "openai", "display_name": "Mine"})
        assert own.status_code == 201
        own_id = own.json()["id"]
        assert client.put(f"/providers/{own_id}", json = {"shared": True}).status_code == 403
    assert listed(BOB) == []
    with client_for(OWNER) as client:
        codex = client.post(
            "/providers/",
            json = {"provider_type": "openai_codex", "display_name": "ChatGPT", "shared": True},
        )
    assert codex.status_code == 400


def test_an_accounts_own_connections_come_first_and_keep_their_own_keys():
    shared_id = owner_connection()
    with client_for(ALICE) as client:
        own = client.post("/providers/", json = {"provider_type": "openai", "display_name": "Mine"})
    own_id = own.json()["id"]
    run_as(ALICE, credential_secrets.save_provider_api_key, own_id, "alice-secret")
    assert [row["id"] for row in listed(ALICE)] == [own_id, shared_id]
    assert run_as(ALICE, credential_secrets.get_provider_api_key, own_id) == "alice-secret"
    assert run_as(OWNER, credential_secrets.get_provider_api_key, own_id) is None


def test_model_access_limits_the_models_of_a_shared_connection(monkeypatch):
    provider_id = owner_connection()
    allow_only(monkeypatch, "unsloth/qwen3-*")
    [row] = listed(ALICE)
    assert row["models"] == ["unsloth/Qwen3-8B-GGUF"]
    config = run_as(ALICE, providers_db.get_provider, provider_id)
    run_as(ALICE, access.require_connection_model_allowed, config, "unsloth/Qwen3-8B-GGUF")
    with pytest.raises(HTTPException) as refused:
        run_as(ALICE, access.require_connection_model_allowed, config, "unsloth/gemma-3-4b-it-GGUF")
    assert refused.value.status_code == 403
    owner_config = run_as(OWNER, providers_db.get_provider, provider_id)
    run_as(OWNER, access.require_connection_model_allowed, owner_config, "other/private-model")
    run_as(ALICE, access.require_connection_model_allowed, {"id": "own"}, "other/private-model")
    allow_only(monkeypatch, "google/*")
    assert run_as(ALICE, access.connection_model_allowed, "~google/gemini-pro-latest")
    assert run_as(ALICE, access.connection_model_allowed, "google/gemma-3:latest")


def test_only_the_models_the_owner_enabled_are_shared():
    provider_id = owner_connection(models = UPSTREAM_MODELS[:1])
    config = run_as(ALICE, providers_db.get_provider, provider_id)
    run_as(ALICE, access.require_connection_model_allowed, config, UPSTREAM_MODELS[0])
    with pytest.raises(HTTPException) as refused:
        run_as(ALICE, access.require_connection_model_allowed, config, "other/private-model")
    assert refused.value.status_code == 403


def test_a_shared_row_whose_url_gained_a_credential_elsewhere_is_withdrawn():
    provider_id = owner_connection()
    run_as(
        OWNER, providers_db.update_provider, provider_id, base_url = "https://gw.example.com/v1?key=s"
    )
    assert listed(ALICE) == []
    assert run_as(ALICE, credential_secrets.get_provider_api_key, provider_id) is None


def test_a_shared_connection_with_no_allowed_model_is_hidden(monkeypatch):
    owner_connection()
    allow_only(monkeypatch, "someone/else")
    assert listed(ALICE) == []


def test_listing_models_through_a_shared_lan_connection(monkeypatch, upstream):
    base_url, seen = upstream
    provider_id = owner_connection(base_url)
    mpu.set_managed_private_provider_urls_allowed(True)
    allow_only(monkeypatch, "unsloth/*")
    with client_for(ALICE) as client:
        response = client.post(
            "/providers/models", json = {"provider_type": "custom", "provider_id": provider_id}
        )
    assert response.status_code == 200, response.text
    assert [model["id"] for model in response.json()] == UPSTREAM_MODELS[:2]
    assert seen == ["Bearer owner-secret"]


def test_listing_a_shared_connection_shows_only_the_models_the_owner_enabled(upstream):
    base_url, _ = upstream
    provider_id = owner_connection(
        base_url, models = UPSTREAM_MODELS[:1], available_models = UPSTREAM_MODELS
    )
    mpu.set_managed_private_provider_urls_allowed(True)
    [row] = listed(ALICE)
    assert row["available_models"] == UPSTREAM_MODELS[:1]
    with client_for(ALICE) as client:
        response = client.post(
            "/providers/models", json = {"provider_type": "custom", "provider_id": provider_id}
        )
    assert response.status_code == 200, response.text
    assert [model["id"] for model in response.json()] == UPSTREAM_MODELS[:1]
    with client_for(OWNER) as client:
        response = client.post(
            "/providers/models", json = {"provider_type": "custom", "provider_id": provider_id}
        )
    assert [model["id"] for model in response.json()] == UPSTREAM_MODELS


def test_a_shared_reader_queues_on_the_owner_lock_without_reading_the_store(monkeypatch):
    provider_id = owner_connection()
    config = run_as(ALICE, providers_db.get_provider, provider_id)

    async def guard(**kwargs):
        return provider_credentials.provider_config_guard(provider_id, **kwargs)

    def no_store_read(*args, **kwargs):
        raise AssertionError("provider store read on the event loop")

    async def check():
        owner_lock = await arun_as(OWNER, guard())
        monkeypatch.setattr(providers_db, "get_connection", no_store_read)
        assert await arun_as(ALICE, guard()) is not owner_lock
        reader = await arun_as(ALICE, guard(shared_by_owner = config["shared_by_owner"]))
        assert reader is owner_lock

    asyncio.run(check())


def test_a_shared_lan_connection_still_needs_the_private_address_switch(upstream):
    base_url, seen = upstream
    provider_id = owner_connection(base_url)
    with client_for(ALICE) as client:
        response = client.post(
            "/providers/models", json = {"provider_type": "custom", "provider_id": provider_id}
        )
    assert response.status_code == 400
    assert seen == []


def test_a_store_from_before_sharing_upgrades_with_nothing_shared(tmp_path):
    db = run_as(OWNER, studio_db_path)
    db.parent.mkdir(parents = True, exist_ok = True)
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE llm_providers (id TEXT NOT NULL PRIMARY KEY, provider_type TEXT NOT NULL, "
            "display_name TEXT NOT NULL, base_url TEXT NOT NULL, is_enabled INTEGER NOT NULL DEFAULT 1, "
            "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO llm_providers VALUES ('old', 'openai', 'Old', '', 1, '2026-01-01', '2026-01-01')"
        )
    [row] = run_as(OWNER, providers_db.list_providers)
    assert row["id"] == "old" and row["shared"] is False
    assert listed(ALICE) == []


def test_the_owner_openai_project_containers_stay_off_limits():
    from models.inference import OpenAIContainerRequest
    from routes import inference

    with client_for(OWNER) as client:
        created = client.post(
            "/providers/",
            json = {"provider_type": "openai", "display_name": "OpenAI", "shared": True},
        )
    provider_id = created.json()["id"]
    run_as(OWNER, credential_secrets.save_provider_api_key, provider_id, "owner-secret")
    body = OpenAIContainerRequest(provider_id = provider_id)
    with pytest.raises(HTTPException) as refused:
        run_as(ALICE, inference._resolve_openai_cloud_client, body, allow_saved_key = True)
    assert refused.value.status_code == 403
    assert run_as(OWNER, inference._resolve_openai_cloud_client, body, allow_saved_key = True)


def test_checking_a_shared_connection_probes_no_model_and_quotes_no_upstream_error(
    monkeypatch, upstream
):
    base_url, seen = upstream
    mpu.set_managed_private_provider_urls_allowed(True)
    provider_id = owner_connection(base_url)
    allow_only(monkeypatch, "unsloth/qwen3-*")
    with client_for(ALICE) as client:
        response = client.post(
            "/providers/test",
            json = {
                "provider_type": "custom",
                "provider_id": provider_id,
                "model_id": "unsloth/gemma-3-4b-it-GGUF",
            },
        )
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert len(seen) == 1 and not seen[0].startswith("POST")

    run_as(
        OWNER,
        providers_db.update_provider,
        provider_id,
        base_url = "http://127.0.0.1:9/v1",
    )
    with client_for(ALICE) as client:
        failed = client.post(
            "/providers/test", json = {"provider_type": "custom", "provider_id": provider_id}
        )
    assert failed.json() == {
        "success": False,
        "message": "Connection failed. Ask the owner to check this connection.",
        "models_count": None,
    }
