# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The owner's opt-in for managed-account LAN provider URLs, end to end.

Companion to test_account_provider_network_scope.py, which pins the default: a
managed account may not reach owner-local or LAN endpoints. This pins what the
opt-in changes and, more importantly, what it must NOT change -- the cloud
metadata refusal and the operator's BLOCK_PRIVATE kill switch both survive it.
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
from core.inference import providers as providers_core
from routes import providers
from storage import credential_secrets, providers_db, studio_db
from utils import managed_provider_url_settings as mpu
from utils.account_context import OWNER, AccountContext, bind_account, reset_account

ALICE = AccountContext("a" * 32, "alice")
OWNER_LOCAL_MODEL = "owner-local-secret-model"
BLOCK_PRIVATE_ENV = providers_core._BLOCK_PRIVATE_ENV


@pytest.fixture(autouse = True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.delenv(BLOCK_PRIVATE_ENV, raising = False)
    # Each test is a fresh installation, so nothing may be remembered from the previous one.
    # In production the home is fixed for the life of the process and this cannot arise.
    mpu.forget_cached_setting()
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    for module in (credential_secrets, providers_db, studio_db):
        monkeypatch.setattr(module, "_schema_ready", set(), raising = False)
    monkeypatch.setattr(
        credential_secrets, "get_or_create_credential_encryption_key", lambda: b"k" * 32
    )
    monkeypatch.setattr(providers, "current_credential_write", lambda credential: nullcontext())
    providers_core._dns_cache.clear()
    yield
    providers_core._dns_cache.clear()


@pytest.fixture
def local_provider():
    """A loopback OpenAI-compatible server standing in for the shared llama-server."""

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
    threading.Thread(target = server.serve_forever, daemon = True).start()
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
    app.include_router(providers.router, prefix = "/providers")
    return TestClient(app)


def allow_lan(value: bool):
    mpu.set_managed_private_provider_urls_allowed(value)


# --- the opt-in does what it says ------------------------------------------------


def test_managed_account_lists_models_from_a_loopback_provider_when_allowed(local_provider):
    allow_lan(True)
    with client_for(ALICE) as client:
        response = client.post(
            "/providers/models",
            json = {"provider_type": "custom", "base_url": local_provider},
        )
    assert response.status_code == 200, response.text
    assert [m["id"] for m in response.json()] == [OWNER_LOCAL_MODEL]


def test_managed_account_saves_a_lan_provider_when_allowed(local_provider):
    allow_lan(True)
    with client_for(ALICE) as client:
        for base_url in (local_provider, "http://10.0.0.7:8080/v1", "http://192.168.1.50:11434/v1"):
            created = client.post(
                "/providers/",
                json = {
                    "provider_type": "custom",
                    "display_name": "LAN",
                    "base_url": base_url,
                },
            )
            assert created.status_code == 201, f"{base_url}: {created.text}"


def test_default_still_refuses(local_provider):
    """Regression guard: the switch is off unless the owner turns it on."""
    with client_for(ALICE) as client:
        response = client.post(
            "/providers/models",
            json = {"provider_type": "custom", "base_url": local_provider},
        )
    assert response.status_code == 400, response.text
    assert OWNER_LOCAL_MODEL not in response.text


def test_turning_it_back_off_refuses_again(local_provider):
    allow_lan(True)
    with client_for(ALICE) as client:
        assert (
            client.post(
                "/providers/models",
                json = {"provider_type": "custom", "base_url": local_provider},
            ).status_code
            == 200
        )
    allow_lan(False)
    with client_for(ALICE) as client:
        refused = client.post(
            "/providers/models",
            json = {"provider_type": "custom", "base_url": local_provider},
        )
    assert refused.status_code == 400, refused.text
    assert OWNER_LOCAL_MODEL not in refused.text


def test_owner_is_unaffected_either_way(local_provider):
    for value in (False, True, False):
        allow_lan(value)
        with client_for(OWNER) as client:
            listed = client.post(
                "/providers/models",
                json = {"provider_type": "custom", "base_url": local_provider},
            )
            assert listed.status_code == 200, f"allowed={value}: {listed.text}"


# --- what the opt-in must NOT relax ----------------------------------------------


@pytest.mark.parametrize(
    "metadata_url",
    [
        "http://169.254.169.254/latest/meta-data/",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://[fd00:ec2::254]/latest/meta-data/",
        # Legacy numeric spellings of the same address.
        "http://2852039166/v1",
        "http://0251.0376.0251.0376/v1",
    ],
)
def test_cloud_metadata_is_still_refused_with_the_switch_on(metadata_url):
    allow_lan(True)
    with client_for(ALICE) as client:
        created = client.post(
            "/providers/",
            json = {"provider_type": "custom", "display_name": "md", "base_url": metadata_url},
        )
    assert created.status_code == 400, created.text
    assert "metadata" in created.text.lower(), created.text


def test_block_private_env_overrides_the_switch_for_a_managed_account(monkeypatch, local_provider):
    allow_lan(True)
    monkeypatch.setenv(BLOCK_PRIVATE_ENV, "1")
    with client_for(ALICE) as client:
        created = client.post(
            "/providers/",
            json = {"provider_type": "custom", "display_name": "LAN", "base_url": local_provider},
        )
    assert created.status_code == 400, created.text
    assert BLOCK_PRIVATE_ENV in created.text


def test_block_private_env_overrides_the_switch_for_the_owner(monkeypatch, local_provider):
    allow_lan(True)
    monkeypatch.setenv(BLOCK_PRIVATE_ENV, "1")
    with client_for(OWNER) as client:
        created = client.post(
            "/providers/",
            json = {"provider_type": "custom", "display_name": "LAN", "base_url": local_provider},
        )
    assert created.status_code == 400, created.text


def test_a_row_saved_while_allowed_is_refused_after_the_switch_is_off(local_provider):
    """The saved row keeps its LAN base URL; the choke point in ExternalProviderClient
    re-validates on every outbound use, so turning the switch off takes effect at once."""
    allow_lan(True)
    with client_for(ALICE) as client:
        created = client.post(
            "/providers/",
            json = {"provider_type": "custom", "display_name": "LAN", "base_url": local_provider},
        )
        assert created.status_code == 201, created.text
        provider_id = created.json()["id"]

    allow_lan(False)
    with client_for(ALICE) as client:
        used = client.post(
            "/providers/models",
            json = {"provider_type": "custom", "base_url": local_provider},
        )
        assert used.status_code == 400, used.text
        # The row itself survives; only its use is refused.
        listed = client.get("/providers/")
        assert any(row["id"] == provider_id for row in listed.json()), listed.text
