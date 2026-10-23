# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Owner-only and UI-session-only gates that a managed account or an API key must not get past."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from auth import policy
from auth.authentication import authenticated_via_api_key, get_current_subject
from utils.account_context import (
    AccountContext,
    OWNER,
    bind_account,
    is_owner_context,
    reset_account,
)

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture(autouse = True)
def multi_user(monkeypatch):
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    yield


def _client(
    account,
    build,
    via_api_key = False,
):
    app = FastAPI()

    async def subject():
        token = bind_account(account)
        try:
            yield account.username
        finally:
            reset_account(token)

    app.dependency_overrides[get_current_subject] = subject
    app.dependency_overrides[authenticated_via_api_key] = lambda: via_api_key
    build(app)
    return TestClient(app)


# Conversation import reads the host owner's ~/.claude, ~/.codex and ~/.cursor.


def _import_app(app):
    from routes import external_import
    app.include_router(external_import.router, prefix = "/api/import")


@pytest.fixture
def imports(monkeypatch):
    from routes import external_import

    ran = []

    def run_import(source):
        ran.append(source)
        return type("Summary", (), {})()

    monkeypatch.setattr(external_import, "run_import", run_import)
    return ran


def test_a_managed_account_cannot_import_the_owners_conversations(imports):
    with _client(ALICE, _import_app) as client:
        assert client.post("/api/import/claude").status_code == 403
    assert imports == []


def test_the_owner_still_imports_conversations(imports):
    with _client(OWNER, _import_app) as client:
        assert client.post("/api/import/claude").status_code == 200
    assert len(imports) == 1


# --rpc: llama-server dials the named hosts and streams the model's tensors to them.


@pytest.mark.parametrize("args", [["--rpc", "10.0.0.5:50052"], ["--rpc=10.0.0.5:50052"]])
def test_rpc_is_an_owner_only_flag(args):
    from core.inference.llama_server_args import owner_only_path_flags
    assert owner_only_path_flags(args) == ["--rpc"]


def test_the_owner_can_still_pass_rpc():
    from core.inference.llama_server_args import validate_extra_args
    assert validate_extra_args(["--rpc", "10.0.0.5:50052"]) == ["--rpc", "10.0.0.5:50052"]


def _launch_gate(monkeypatch, stored_by_account):
    """routes.inference._refuse_managed_custom_projector with override rows per account."""
    from routes import inference
    from utils import openai_auto_switch_settings

    def resolve(
        load_id,
        alias_id = None,
        variant = None,
    ):
        # As the real resolver: a managed account without its own row falls back to the owner's.
        row = stored_by_account.get("owner" if is_owner_context() else "managed")
        if not row and not is_owner_context():
            row = stored_by_account.get("owner")
        return (load_id, row) if row else (None, {})

    monkeypatch.setattr(openai_auto_switch_settings, "resolve_override_for_load", resolve)
    monkeypatch.setattr(
        inference,
        "get_llama_cpp_backend",
        lambda: type("Backend", (), {"last_load_intent": None})(),
    )

    def gate(extra_args, identifier = "org/model-GGUF"):
        token = bind_account(ALICE)
        try:
            return inference._refuse_managed_custom_projector(extra_args, identifier)
        finally:
            reset_account(token)

    return gate


def test_a_managed_account_cannot_load_with_rpc(monkeypatch):
    gate = _launch_gate(monkeypatch, {})
    with pytest.raises(HTTPException) as refused:
        gate(["--rpc", "10.0.0.5:50052"])
    assert refused.value.status_code == 403


def test_a_managed_accounts_own_override_is_not_the_owners_choice(monkeypatch):
    # A managed account can save per-model overrides; that row must not count as one the owner chose.
    path_args = ["--chat-template-file", "/etc/hostname"]
    gate = _launch_gate(monkeypatch, {"managed": {"llama_extra_args": path_args}})
    with pytest.raises(HTTPException) as refused:
        gate(path_args)
    assert refused.value.status_code == 403


def test_a_path_the_owner_saved_still_replays_for_a_managed_account(monkeypatch):
    path_args = ["--chat-template-file", "/srv/templates/chat.jinja", "--rpc", "10.0.0.5:50052"]
    gate = _launch_gate(monkeypatch, {"owner": {"llama_extra_args": path_args}})
    assert gate(path_args) is None


# Data Recipes: a saved recipe runs its stdio MCP command later, from the UI.

_STDIO_PAYLOAD = {
    "recipe": {
        "columns": [],
        "mcp_providers": [{"name": "x", "provider_type": "stdio", "command": "touch /tmp/pwned"}],
    }
}
_HTTP_PAYLOAD = {
    "recipe": {
        "columns": [],
        "mcp_providers": [
            {"name": "x", "provider_type": "streamable_http", "endpoint": "https://example.com/mcp"}
        ],
    }
}


@pytest.fixture
def recipe_store(monkeypatch):
    from routes.data_recipe import library

    saved = []
    monkeypatch.setattr(
        library.db, "upsert_recipe", lambda record, base: saved.append(record) or record
    )
    monkeypatch.setattr(
        library.db,
        "import_legacy",
        lambda recipes, executions: saved.extend(recipes) or {"recipes": len(recipes)},
    )
    return saved


def _recipe_app(app):
    from routes.data_recipe import library
    app.include_router(library.router, prefix = "/api/data-recipe")


def _recipe(payload):
    return {"id": "r1", "name": "r", "payload": payload, "createdAt": 1, "updatedAt": 1}


def test_an_api_key_cannot_save_a_recipe_with_a_stdio_mcp_command(recipe_store):
    with _client(OWNER, _recipe_app, via_api_key = True) as client:
        assert (
            client.put("/api/data-recipe/recipes/r1", json = _recipe(_STDIO_PAYLOAD)).status_code
            == 403
        )
        imported = client.post(
            "/api/data-recipe/recipes/import", json = {"recipes": [_recipe(_STDIO_PAYLOAD)]}
        )
        assert imported.status_code == 403
    assert recipe_store == []


def test_recipes_without_stdio_and_ui_sessions_still_save(recipe_store):
    with _client(OWNER, _recipe_app, via_api_key = True) as client:
        assert (
            client.put("/api/data-recipe/recipes/r1", json = _recipe(_HTTP_PAYLOAD)).status_code
            == 200
        )
    with _client(OWNER, _recipe_app, via_api_key = False) as client:
        assert (
            client.put("/api/data-recipe/recipes/r1", json = _recipe(_STDIO_PAYLOAD)).status_code
            == 200
        )
        imported = client.post(
            "/api/data-recipe/recipes/import", json = {"recipes": [_recipe(_STDIO_PAYLOAD)]}
        )
        assert imported.status_code == 200
    assert len(recipe_store) == 3


# MCP servers: an API key cannot touch a stdio row, so it cannot delete one either.


@pytest.fixture
def mcp_rows(monkeypatch):
    from routes import mcp_servers

    rows = {
        "stdio": {"id": "stdio", "url": "npx -y some-server", "use_oauth": False},
        "http": {"id": "http", "url": "https://example.com/mcp", "use_oauth": False},
    }
    deleted = []
    monkeypatch.setattr(mcp_servers.mcp_servers_db, "get_server", lambda sid: rows.get(sid))
    monkeypatch.setattr(mcp_servers.mcp_servers_db, "delete_server", deleted.append)
    monkeypatch.setattr(mcp_servers, "invalidate_tool_cache", lambda sid: None)
    monkeypatch.setattr(mcp_servers, "close_mcp_sessions", lambda *a, **k: None)
    monkeypatch.setattr(mcp_servers, "parse_server_headers", lambda row: {})

    def delete(server_id, via_api_key):
        return asyncio.run(
            mcp_servers.delete_mcp_server(server_id, current_subject = "x", via_api_key = via_api_key)
        )

    return delete, deleted


def test_an_api_key_cannot_delete_a_stdio_mcp_server(mcp_rows):
    delete, deleted = mcp_rows
    with pytest.raises(HTTPException) as refused:
        delete("stdio", True)
    assert refused.value.status_code == 403
    assert deleted == []


def test_mcp_deletes_still_work_for_ui_sessions_and_http_rows(mcp_rows):
    delete, deleted = mcp_rows
    delete("http", True)
    delete("stdio", False)
    assert deleted == ["http", "stdio"]


# Provider capability catalogs are per credential.


def test_capability_cache_does_not_answer_one_key_with_anothers_catalog(monkeypatch):
    from models.providers import ProviderModelsRequest
    from routes import providers as providers_route

    fetched = []

    class Client:
        def __init__(self, **kwargs):
            self.api_key = kwargs["api_key"]

        async def list_models(self):
            fetched.append(self.api_key)
            return [{"id": f"model-for-{self.api_key}"}]

        async def close(self):
            pass

    keys = iter(["sk-alice", "sk-bob"])
    monkeypatch.setattr(providers_route, "ExternalProviderClient", Client)
    monkeypatch.setattr(
        providers_route, "resolve_provider_api_key_or_400", lambda *a, **k: next(keys)
    )
    monkeypatch.setattr(
        providers_route,
        "provider_model_capabilities",
        lambda provider_type, models: [{"id": m["id"]} for m in models],
    )
    providers_route._model_capability_cache.clear()
    try:
        for _ in range(2):
            asyncio.run(
                providers_route.list_provider_model_capabilities(
                    ProviderModelsRequest(provider_type = "openrouter"),
                    _current_subject = "x",
                    via_api_key = False,
                )
            )
    finally:
        providers_route._model_capability_cache.clear()
    assert fetched == ["sk-alice", "sk-bob"]
