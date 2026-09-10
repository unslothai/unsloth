# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's recipe MCP providers obey the chat MCP network boundary: no loopback, no LAN."""

from __future__ import annotations

import sys
import types

import pytest
from fastapi import HTTPException

from auth import policy
from core.inference import mcp_client
from utils.account_context import OWNER, AccountContext, run_as

ALICE = AccountContext("alice-id", "alice")

_PRIVATE_ENDPOINTS = (
    "http://127.0.0.1:9111/mcp",
    "http://localhost:9111/sse",
    "http://10.0.0.5:8000/mcp",
)


@pytest.fixture(autouse = True)
def data_designer_stub(monkeypatch, tmp_path):
    """Stand in for the recipe engine's provider classes, which are an Unsloth-only plugin absent on CI."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)

    class _Provider:
        def __init__(self, **fields):
            self.__dict__.update(fields)

    package = types.ModuleType("data_designer")
    config = types.ModuleType("data_designer.config")
    mcp = types.ModuleType("data_designer.config.mcp")
    mcp.MCPProvider = type("MCPProvider", (_Provider,), {})
    mcp.LocalStdioMCPProvider = type("LocalStdioMCPProvider", (_Provider,), {})
    models = types.ModuleType("data_designer.config.models")
    models.ModelProvider = type("ModelProvider", (_Provider,), {})
    config.mcp = mcp
    config.models = models
    package.config = config
    for name, module in (
        ("data_designer", package),
        ("data_designer.config", config),
        ("data_designer.config.mcp", mcp),
        ("data_designer.config.models", models),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _recipe(endpoint: str) -> dict:
    return {
        "mcp_providers": [
            {"provider_type": "streamable_http", "name": "local", "endpoint": endpoint}
        ]
    }


@pytest.mark.parametrize("endpoint", _PRIVATE_ENDPOINTS)
def test_managed_recipe_mcp_is_refused_where_chat_mcp_refuses(endpoint):
    from core.data_recipe.service import build_mcp_providers

    with pytest.raises(HTTPException) as chat:
        run_as(ALICE, mcp_client.validate_mcp_address, endpoint)
    assert chat.value.status_code == 400
    with pytest.raises(HTTPException) as recipe:
        run_as(ALICE, build_mcp_providers, _recipe(endpoint))
    assert recipe.value.status_code == 403


def test_owner_recipe_mcp_is_unchanged():
    from core.data_recipe.service import build_mcp_providers
    built = run_as(OWNER, build_mcp_providers, _recipe(_PRIVATE_ENDPOINTS[0]))
    assert [provider.endpoint for provider in built] == [_PRIVATE_ENDPOINTS[0]]


def test_single_account_installs_keep_recipe_mcp(monkeypatch):
    from core.data_recipe.service import build_mcp_providers

    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: False)
    built = run_as(OWNER, build_mcp_providers, _recipe(_PRIVATE_ENDPOINTS[0]))
    assert [provider.endpoint for provider in built] == [_PRIVATE_ENDPOINTS[0]]
    with pytest.raises(HTTPException) as refused:
        run_as(ALICE, build_mcp_providers, _recipe(_PRIVATE_ENDPOINTS[0]))
    assert refused.value.status_code == 403


def test_managed_recipe_stdio_provider_is_still_dropped(monkeypatch):
    """The network refusal must not turn the host gate's silent drop of a local command into an error."""
    from core.data_recipe.service import build_mcp_providers

    monkeypatch.setenv("UNSLOTH_STUDIO_ALLOW_STDIO_MCP", "1")
    recipe = {
        "mcp_providers": [
            {"provider_type": "stdio", "name": "fs", "command": "npx", "args": [], "env": {}}
        ]
    }
    assert run_as(ALICE, build_mcp_providers, recipe) == []


def _provider_recipe(endpoint: str) -> dict:
    return {"model_providers": [{"name": "llm", "endpoint": endpoint, "api_key": "k"}]}


@pytest.mark.parametrize(
    "endpoint",
    (
        *(e.replace("http://", "https://") for e in _PRIVATE_ENDPOINTS),
        "https://169.254.169.254/latest",
        "https://[::1]:11434/v1",
    ),
)
def test_managed_recipe_model_provider_must_be_public(endpoint):
    """The engine dials providers itself, so a managed recipe cannot point one at loopback or the LAN."""
    from core.data_recipe.service import build_model_providers

    with pytest.raises(HTTPException) as refused:
        run_as(ALICE, build_model_providers, _provider_recipe(endpoint))
    assert refused.value.status_code == 403
    assert "public" in refused.value.detail


def test_managed_recipe_model_provider_on_a_public_https_address_is_kept():
    from core.data_recipe.service import build_model_providers
    built = run_as(ALICE, build_model_providers, _provider_recipe("https://8.8.8.8/v1"))
    assert [provider.endpoint for provider in built] == ["https://8.8.8.8/v1"]


def test_managed_recipe_model_provider_over_plain_http_is_refused():
    """The engine re-resolves the name when it connects, so only TLS pins the peer; plain HTTP is refused."""
    from core.data_recipe.service import build_model_providers

    with pytest.raises(HTTPException) as refused:
        run_as(ALICE, build_model_providers, _provider_recipe("http://8.8.8.8/v1"))
    assert refused.value.status_code == 403
    assert "HTTPS" in refused.value.detail


def test_owner_recipe_model_provider_endpoints_are_unchanged():
    from core.data_recipe.service import build_model_providers
    built = run_as(OWNER, build_model_providers, _provider_recipe(_PRIVATE_ENDPOINTS[0]))
    assert [provider.endpoint for provider in built] == [_PRIVATE_ENDPOINTS[0]]
