# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""API key / keyless recipes must not send a host env credential to an endpoint they choose."""

import pytest
from fastapi import HTTPException

from core.data_recipe import service
from models.data_recipe import McpToolsListRequest, RecipePayload

SECRET = "OPENAI_API_KEY"


@pytest.fixture(autouse = True)
def host_secret(monkeypatch):
    monkeypatch.setenv(SECRET, "sk-host-secret")


def _model_provider(**fields):
    return {
        "name": "exfil",
        "endpoint": "https://attacker.example/v1",
        "provider_type": "openai",
        **fields,
    }


def _mcp_provider(**fields):
    return {
        "name": "exfil",
        "endpoint": "https://attacker.example/mcp",
        "provider_type": "streamable_http",
        **fields,
    }


def _recipe(**providers):
    return {"columns": [{"name": "c", "column_type": "llm-text"}], **providers}


@pytest.mark.parametrize(
    "recipe",
    [
        _recipe(model_providers = [_model_provider(api_key_env = SECRET)]),
        _recipe(model_providers = [_model_provider(api_key = SECRET)]),
        _recipe(mcp_providers = [_mcp_provider(api_key_env = SECRET)]),
        _recipe(mcp_providers = [_mcp_provider(api_key = SECRET)]),
        _recipe(seed_config = {"source": {"seed_type": "hf", "token": SECRET}}),
    ],
)
def test_env_credentials_are_detected(recipe):
    assert service.recipe_names_host_env_credential(recipe) is True


@pytest.mark.parametrize(
    "recipe",
    [
        _recipe(model_providers = [_model_provider(api_key = "sk-literal-key")]),
        _recipe(model_providers = [_model_provider(api_key_env = "")]),
        _recipe(model_providers = [_model_provider(api_key = "sk-literal-key", api_key_env = SECRET)]),
        _recipe(model_providers = [_model_provider(api_key_env = SECRET, is_local = True)]),
        _recipe(),
    ],
)
def test_own_credentials_and_local_providers_are_not_flagged(recipe):
    assert service.recipe_names_host_env_credential(recipe) is False


@pytest.mark.parametrize(
    "recipe",
    [
        {**_recipe(model_providers = [_model_provider(api_key_env = SECRET)]), "is_local": True},
        _recipe(mcp_providers = [_mcp_provider(api_key_env = SECRET, is_local = True)]),
        _recipe(seed_config = {"source": {"seed_type": "hf", "token": SECRET, "is_local": True}}),
        [_mcp_provider(api_key = SECRET, is_local = True)],
    ],
)
def test_is_local_only_exempts_model_providers_the_routes_rewire(recipe):
    assert service.recipe_names_host_env_credential(recipe) is True


def test_validate_drops_external_auth_from_local_providers():
    import routes.data_recipe.validate as validate_route

    recipe = _recipe(
        model_providers = [
            _model_provider(
                api_key_env = SECRET, api_key = "x", extra_headers = {"a": "b"}, is_local = True
            )
        ]
    )
    validate_route._patch_local_providers(recipe)
    provider = recipe["model_providers"][0]
    assert provider["endpoint"] == "http://127.0.0.1"
    assert provider["api_key"] is None
    assert "api_key_env" not in provider and "extra_headers" not in provider


def _hf_seed(**source):
    return _recipe(
        seed_config = {"source": {"seed_type": "hf", "path": "datasets/a/b/x.csv", **source}}
    )


def test_api_key_hf_seed_on_another_endpoint_must_bring_its_own_token():
    with pytest.raises(HTTPException) as exc:
        service.require_ui_session_for_host_env_credentials(
            _hf_seed(endpoint = "https://attacker.example"), via_api_key = True
        )
    assert exc.value.status_code == 403
    service.require_ui_session_for_host_env_credentials(
        _hf_seed(endpoint = "https://attacker.example", token = "hf_own"), via_api_key = True
    )
    service.require_ui_session_for_host_env_credentials(_hf_seed(), via_api_key = True)
    service.require_ui_session_for_host_env_credentials(
        _hf_seed(endpoint = "https://attacker.example"), via_api_key = False
    )


def test_ui_session_keeps_the_env_lookup():
    recipe = _recipe(model_providers = [_model_provider(api_key_env = SECRET)])
    service.require_ui_session_for_host_env_credentials(recipe, via_api_key = False)


def test_validate_refuses_api_key_before_any_work(monkeypatch):
    import routes.data_recipe.validate as validate_route

    def _never(*args, **kwargs):
        raise AssertionError("a refused request must not reach provider setup")

    monkeypatch.setattr(validate_route, "_patch_local_providers", _never)
    payload = RecipePayload(recipe = _recipe(model_providers = [_model_provider(api_key_env = SECRET)]))
    with pytest.raises(HTTPException) as exc:
        validate_route.validate(payload, via_api_key = True)
    assert exc.value.status_code == 403


def test_create_job_refuses_api_key_before_minting_a_key_or_starting(monkeypatch):
    import routes.data_recipe.jobs as jobs_route

    def _never(*args, **kwargs):
        raise AssertionError("a refused request must not mint a key or start a job")

    monkeypatch.setattr(jobs_route, "_inject_local_providers", _never)
    monkeypatch.setattr(jobs_route, "get_job_manager", _never)
    payload = RecipePayload(recipe = _recipe(model_providers = [_model_provider(api_key = SECRET)]))
    with pytest.raises(HTTPException) as exc:
        jobs_route.create_job(payload, request = None, credential = ("owner", None), via_api_key = True)
    assert exc.value.status_code == 403


def test_mcp_tool_listing_refuses_api_key_before_building_providers(monkeypatch):
    import routes.data_recipe.mcp as mcp_route

    def _never(*args, **kwargs):
        raise AssertionError("a refused request must not build or probe MCP providers")

    monkeypatch.setattr(mcp_route, "build_mcp_providers", _never)
    payload = McpToolsListRequest(mcp_providers = [_mcp_provider(api_key_env = SECRET)])
    with pytest.raises(HTTPException) as exc:
        mcp_route.list_mcp_tools(payload, via_api_key = True)
    assert exc.value.status_code == 403


def test_keyless_callers_count_as_api_callers(monkeypatch):
    import asyncio

    from auth import authentication

    monkeypatch.setattr(authentication, "is_keyless", lambda credentials: True)
    assert asyncio.run(authentication.authenticated_via_api_key(None)) is True
