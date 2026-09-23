# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account must not reach owner-local or LAN provider endpoints.

Managed MCP servers are already restricted to public destinations; provider base
URLs are the same caller-controlled server-side egress, so they get the same rule.
"""

import http.server
import threading
from contextlib import nullcontext

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import (
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
)
from routes import providers
from storage import credential_secrets, providers_db
from utils.account_context import OWNER, AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")

# What an owner-local llama-server would expose and a tenant must never see.
OWNER_LOCAL_MODEL = "owner-local-secret-model"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.delenv("UNSLOTH_STUDIO_BLOCK_PRIVATE_PROVIDER_URLS", raising=False)
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    for module in (credential_secrets, providers_db):
        monkeypatch.setattr(module, "_schema_ready", set())
    monkeypatch.setattr(
        credential_secrets, "get_or_create_credential_encryption_key", lambda: b"k" * 32
    )
    monkeypatch.setattr(providers, "current_credential_write", lambda credential: nullcontext())


@pytest.fixture
def local_provider():
    """A loopback OpenAI-compatible server standing in for the owner's llama-server."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = ('{"data": [{"id": "%s"}]}' % OWNER_LOCAL_MODEL).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
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
    app.include_router(providers.router, prefix="/providers")
    return TestClient(app)


def test_managed_account_cannot_list_models_from_a_loopback_provider(local_provider):
    with client_for(ALICE) as client:
        response = client.post(
            "/providers/models",
            json={"provider_type": "custom", "base_url": local_provider},
        )
    assert response.status_code == 400, response.text
    assert OWNER_LOCAL_MODEL not in response.text


def test_managed_account_cannot_save_a_private_provider_base_url(local_provider):
    with client_for(ALICE) as client:
        for base_url in (local_provider, "http://10.0.0.7:8080/v1", "http://169.254.1.1/v1"):
            created = client.post(
                "/providers/",
                json={
                    "provider_type": "custom",
                    "display_name": "LAN",
                    "base_url": base_url,
                },
            )
            assert created.status_code == 400, f"{base_url}: {created.text}"


@pytest.fixture
def owner_allows_private_urls(monkeypatch):
    """The installation owner has opened private addresses to managed accounts (#11382)."""
    from core.inference import external_provider
    from utils import managed_provider_url_settings

    monkeypatch.setattr(
        managed_provider_url_settings, "get_managed_private_provider_urls_allowed", lambda: True
    )
    # The pinning client is a module-level singleton; drop it so the choice is made fresh.
    monkeypatch.setattr(external_provider, "_managed_clients", {}, raising=False)


def test_managed_account_may_save_a_private_base_url_once_the_owner_allows_it(
    local_provider, owner_allows_private_urls
):
    with client_for(ALICE) as client:
        created = client.post(
            "/providers/",
            json={
                "provider_type": "custom",
                "display_name": "Shared LAN llama-server",
                "base_url": local_provider,
            },
        )
        assert created.status_code == 201, created.text
        assert created.json()["base_url"] == local_provider


def test_managed_account_may_list_models_from_a_private_url_once_the_owner_allows_it(
    local_provider, owner_allows_private_urls
):
    """The save and the send have to agree, or the connection saves and then never works."""
    with client_for(ALICE) as client:
        listed = client.post(
            "/providers/models",
            json={"provider_type": "custom", "base_url": local_provider},
        )
    assert listed.status_code == 200, listed.text
    assert [m["id"] for m in listed.json()] == [OWNER_LOCAL_MODEL]


def test_cloud_metadata_stays_refused_even_when_the_owner_allows_private_urls(
    owner_allows_private_urls,
):
    with client_for(ALICE) as client:
        created = client.post(
            "/providers/",
            json={
                "provider_type": "custom",
                "display_name": "metadata",
                "base_url": "http://169.254.169.254/v1",
            },
        )
    assert created.status_code == 400, created.text
    assert "metadata" in created.text.lower()


def test_owner_keeps_local_providers(local_provider):
    with client_for(OWNER) as client:
        created = client.post(
            "/providers/",
            json={
                "provider_type": "custom",
                "display_name": "Local llama-server",
                "base_url": local_provider,
            },
        )
        assert created.status_code == 201, created.text
        listed = client.post(
            "/providers/models",
            json={"provider_type": "custom", "base_url": local_provider},
        )
        assert listed.status_code == 200, listed.text
        assert [m["id"] for m in listed.json()] == [OWNER_LOCAL_MODEL]
