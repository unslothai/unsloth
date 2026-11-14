# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTP and SQLite coverage for personal model details; only authentication is replaced."""

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import get_current_subject
from routes import settings
from utils.account_context import AccountContext, bind_account, reset_account

MODEL = "download:gguf:org%2FOriginal-GGUF"
ALICE = AccountContext("a" * 32, "alice")
BOB = AccountContext("b" * 32, "bob")


@pytest.fixture
def client(monkeypatch):
    """Mount the actual settings router with bound test identities and isolated account databases."""
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    app = FastAPI()

    async def subject(request: Request):
        """Supply the authenticated identity boundary while retaining real account context propagation."""
        account = BOB if request.headers.get("x-test-account") == "bob" else ALICE
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.include_router(settings.router, prefix = "/api/settings")
    with TestClient(app, raise_server_exceptions = False) as http:
        yield http


def put(
    client,
    model_id = MODEL,
    *,
    account = "alice",
    **annotation,
):
    """Write personal fields through the real HTTP schema and settings route."""
    return client.put(
        "/api/settings/model-annotations",
        headers = {"x-test-account": account},
        json = {"model_id": model_id, "annotation": annotation},
    )


def test_http_save_persists_normalized_details_and_preserves_notes(client):
    """A second HTTP request reads the committed details without changing the inventory key."""
    notes = "  Keep this indentation.\n日本語 notes\n"
    saved = put(client, display_name = " My coder ", labels = ["work", "work", " fast "], notes = notes)
    assert saved.status_code == 200
    expected = {"display_name": "My coder", "labels": ["work", "fast"], "notes": notes}
    assert saved.json() == {"annotation": expected}
    assert client.get("/api/settings/model-annotations").json() == {
        "annotations": {MODEL: expected}
    }


def test_clear_one_row_keeps_other_models(client):
    """Clearing fields removes only that row's annotation, leaving other model details intact."""
    assert put(client, display_name = "Coder").status_code == 200
    other = "download:gguf:org%2FOther"
    assert put(client, other, notes = "Keep me").status_code == 200
    assert put(client).json() == {"annotation": None}
    saved = client.get("/api/settings/model-annotations").json()["annotations"]
    assert MODEL not in saved
    assert saved[other]["notes"] == "Keep me"


def test_http_accounts_cannot_read_or_clear_each_others_notes(client):
    """The same inventory id can hold different details in each account's database."""
    assert put(client, notes = "Alice's note").status_code == 200
    assert client.get(
        "/api/settings/model-annotations", headers = {"x-test-account": "bob"}
    ).json() == {"annotations": {}}
    assert put(client, account = "bob", notes = "Bob's note").status_code == 200
    assert put(client).json() == {"annotation": None}
    bob = client.get("/api/settings/model-annotations", headers = {"x-test-account": "bob"})
    assert bob.json()["annotations"][MODEL]["notes"] == "Bob's note"
    assert client.get("/api/settings/model-annotations").json() == {"annotations": {}}


@pytest.mark.parametrize(
    "annotation",
    [
        {"display_name": "x" * 161},
        {"labels": ["label"] * 21},
        {"labels": ["x" * 65]},
        {"labels": [""]},
        {"notes": "x" * 8001},
    ],
)
def test_invalid_details_do_not_overwrite_a_saved_note(client, annotation):
    """Reject oversized or empty-label input before the persistent map is changed."""
    assert put(client, notes = "Keep me").status_code == 200
    assert put(client, **annotation).status_code == 422
    assert (
        client.get("/api/settings/model-annotations").json()["annotations"][MODEL]["notes"]
        == "Keep me"
    )


def test_blank_inventory_id_is_rejected(client):
    """An update must identify one inventory row instead of creating an empty map key."""
    assert put(client, "   ", notes = "Note").status_code == 422


@pytest.mark.parametrize("operation", ["get_app_setting", "upsert_app_setting_map_entry"])
def test_storage_failures_are_errors_and_preserve_saved_notes(client, monkeypatch, operation):
    """An unavailable database must not look like an empty account or a successful save."""
    assert put(client, notes = "Keep me").status_code == 200

    def unavailable(*args, **kwargs):
        """Fail only the storage boundary so HTTP error handling runs unchanged."""
        raise OSError("Database unavailable")

    with monkeypatch.context() as patch:
        patch.setattr("storage.studio_db." + operation, unavailable)
        response = (
            client.get("/api/settings/model-annotations")
            if operation == "get_app_setting"
            else put(client, notes = "Changed")
        )
        assert response.status_code == 500
    assert (
        client.get("/api/settings/model-annotations").json()["annotations"][MODEL]["notes"]
        == "Keep me"
    )


def test_unauthenticated_settings_request_is_rejected():
    """The actual router still requires authentication when its test identity override is absent."""
    app = FastAPI()
    app.include_router(settings.router, prefix = "/api/settings")
    with TestClient(app) as http:
        assert http.get("/api/settings/model-annotations").status_code == 401
