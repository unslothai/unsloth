# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException

from auth import storage as auth_storage
from models.providers import (
    ProviderCreate,
    ProviderModelsRequest,
    ProviderTestRequest,
    ProviderUpdate,
)


def _load_route_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_routes_dir = Path(__file__).resolve().parents[1] / "routes"
_previous_routes = sys.modules.get("routes")
_routes_package = types.ModuleType("routes")
_routes_package.__path__ = [str(_routes_dir)]
sys.modules["routes"] = _routes_package
try:
    _load_route_module("routes.provider_credentials", _routes_dir / "provider_credentials.py")
    providers_route = _load_route_module(
        "_credential_providers_route", _routes_dir / "providers.py"
    )
    settings_route = _load_route_module("_credential_settings_route", _routes_dir / "settings.py")
finally:
    sys.modules.pop("routes.provider_credentials", None)
    if _previous_routes is None:
        sys.modules.pop("routes", None)
    else:
        sys.modules["routes"] = _previous_routes
from storage import credential_secrets, providers_db


@pytest.fixture(autouse = True)
def isolated_databases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    auth_db = tmp_path / "auth.db"
    studio_db = tmp_path / "studio.db"
    monkeypatch.setattr(auth_storage, "DB_PATH", auth_db)
    monkeypatch.setattr(auth_storage, "_credential_encryption_key_cache", None)
    monkeypatch.setattr(providers_db, "studio_db_path", lambda: studio_db)
    monkeypatch.setattr(credential_secrets, "studio_db_path", lambda: studio_db)
    monkeypatch.setattr(providers_db, "ensure_dir", lambda _path: None)
    monkeypatch.setattr(credential_secrets, "ensure_dir", lambda _path: None)
    monkeypatch.setattr(
        credential_secrets,
        "get_or_create_credential_encryption_key",
        auth_storage.get_or_create_credential_encryption_key,
    )
    providers_db._schema_ready = False
    credential_secrets._schema_ready = False
    yield
    providers_db._schema_ready = False
    credential_secrets._schema_ready = False
    auth_storage._credential_encryption_key_cache = None


def test_provider_create_preserve_replace_clear_and_delete(monkeypatch):
    plaintext_by_envelope = {"first": "sk-first", "second": "sk-second"}
    monkeypatch.setattr(
        providers_route,
        "resolve_provider_api_key_or_400",
        lambda _provider_id, envelope, **_kwargs: plaintext_by_envelope.get(envelope, ""),
    )

    created = asyncio.run(
        providers_route.create_provider_config(
            ProviderCreate(
                provider_type = "openai",
                display_name = "OpenAI",
                encrypted_api_key = "first",
            ),
            _current_subject = "alice",
            via_api_key = False,
        )
    )
    assert created.has_api_key is True
    assert credential_secrets.get_provider_api_key(created.id) == "sk-first"

    metadata_only = asyncio.run(
        providers_route.update_provider_config(
            created.id,
            ProviderUpdate(display_name = "Renamed"),
            _current_subject = "alice",
            via_api_key = False,
        )
    )
    assert metadata_only.display_name == "Renamed"
    assert credential_secrets.get_provider_api_key(created.id) == "sk-first"

    replaced = asyncio.run(
        providers_route.update_provider_config(
            created.id,
            ProviderUpdate(encrypted_api_key = "second"),
            _current_subject = "alice",
            via_api_key = False,
        )
    )
    assert replaced.has_api_key is True
    assert credential_secrets.get_provider_api_key(created.id) == "sk-second"

    cleared = asyncio.run(
        providers_route.update_provider_config(
            created.id,
            ProviderUpdate(clear_api_key = True),
            _current_subject = "alice",
            via_api_key = False,
        )
    )
    assert cleared.has_api_key is False
    assert credential_secrets.get_provider_api_key(created.id) is None

    credential_secrets.save_provider_api_key(created.id, "sk-before-delete")

    asyncio.run(
        providers_route.delete_provider_config(
            created.id, _current_subject = "alice", via_api_key = False
        )
    )
    asyncio.run(
        providers_route.delete_provider_config(
            created.id, _current_subject = "alice", via_api_key = False
        )
    )

    assert credential_secrets.get_provider_api_key(created.id) is None
    assert providers_db.get_provider(created.id) is None


def test_provider_mutations_reject_api_key_authentication():
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            providers_route.create_provider_config(
                ProviderCreate(provider_type = "openai", display_name = "Forbidden"),
                _current_subject = "alice",
                via_api_key = True,
            )
        )
    assert error.value.status_code == 403
    assert providers_db.list_providers() == []


def test_shared_provider_resolver_uses_saved_and_explicit_precedence(monkeypatch):
    credential_secrets.save_provider_api_key("provider-1", "saved")
    assert providers_route.resolve_provider_api_key_or_400("provider-1", None) == "saved"

    assert (
        providers_route.resolve_provider_api_key_or_400("provider-1", None, allow_saved_key = False)
        == ""
    )

    from core.inference import key_exchange

    monkeypatch.setattr(key_exchange, "decrypt_api_key", lambda value: f"explicit:{value}")
    assert (
        providers_route.resolve_provider_api_key_or_400("provider-1", "ciphertext")
        == "explicit:ciphertext"
    )

    monkeypatch.setattr(
        key_exchange,
        "decrypt_api_key",
        lambda _value: (_ for _ in ()).throw(ValueError("secret detail")),
    )
    with pytest.raises(HTTPException) as error:
        providers_route.resolve_provider_api_key_or_400("provider-1", "broken")
    assert error.value.status_code == 400
    assert "secret detail" not in str(error.value.detail)


def test_provider_model_and_connection_routes_use_saved_key(monkeypatch):
    seen_clients: list[tuple[str, str, str]] = []

    class FakeProviderClient:
        def __init__(self, **kwargs):
            seen_clients.append((kwargs["provider_type"], kwargs["base_url"], kwargs["api_key"]))

        async def list_models(self):
            return [{"id": "mistral-large-latest"}]

        async def close(self):
            return None

    monkeypatch.setattr(providers_route, "ExternalProviderClient", FakeProviderClient)

    providers_db.create_provider(
        id = "provider-1",
        provider_type = "mistral",
        display_name = "Mistral",
        base_url = "https://api.mistral.ai/v1",
    )
    credential_secrets.save_provider_api_key("provider-1", "saved-key")

    credential_secrets.save_hf_token("hf-after-restart")
    auth_storage._credential_encryption_key_cache = None
    credential_secrets._schema_ready = False
    assert (
        settings_route.get_hugging_face_token("alice", via_api_key = False).token
        == "hf-after-restart"
    )

    models = asyncio.run(
        providers_route.list_provider_models(
            ProviderModelsRequest(
                provider_type = "custom",
                provider_id = "provider-1",
                base_url = "https://attacker.invalid/v1",
            ),
            _current_subject = "alice",
            via_api_key = False,
        )
    )
    result = asyncio.run(
        providers_route.test_provider(
            ProviderTestRequest(
                provider_type = "custom",
                provider_id = "provider-1",
                base_url = "https://attacker.invalid/v1",
            ),
            _current_subject = "alice",
            via_api_key = False,
        )
    )

    api_key_models = asyncio.run(
        providers_route.list_provider_models(
            ProviderModelsRequest(
                provider_type = "custom",
                provider_id = "provider-1",
            ),
            _current_subject = "alice",
            via_api_key = True,
        )
    )

    assert [model.id for model in models] == ["mistral-large-latest"]
    assert result.success is True

    assert [model.id for model in api_key_models] == ["mistral-large-latest"]
    assert seen_clients == [
        ("mistral", "https://api.mistral.ai/v1", "saved-key"),
        ("mistral", "https://api.mistral.ai/v1", "saved-key"),
        ("mistral", "https://api.mistral.ai/v1", ""),
    ]


def test_hugging_face_routes_are_global_and_idempotent():
    saved = settings_route.update_hugging_face_token(
        settings_route.HuggingFaceTokenPayload(token = " 'hf_alice' "),
        _current_subject = "alice",
        via_api_key = False,
    )
    assert saved.token == "hf_alice"
    assert settings_route.get_hugging_face_token("alice", via_api_key = False).token == "hf_alice"
    assert settings_route.get_hugging_face_token("bob", via_api_key = False).token == "hf_alice"

    assert settings_route.clear_hugging_face_token("alice", via_api_key = False).has_token is False
    assert settings_route.clear_hugging_face_token("alice", via_api_key = False).has_token is False


def test_hugging_face_secret_routes_reject_api_key_authentication():
    with pytest.raises(HTTPException) as get_error:
        settings_route.get_hugging_face_token("alice", via_api_key = True)
    assert get_error.value.status_code == 403

    with pytest.raises(HTTPException) as put_error:
        settings_route.update_hugging_face_token(
            settings_route.HuggingFaceTokenPayload(token = "hf_forbidden"),
            _current_subject = "alice",
            via_api_key = True,
        )
    assert put_error.value.status_code == 403

    with pytest.raises(HTTPException) as delete_error:
        settings_route.clear_hugging_face_token(
            "alice",
            via_api_key = True,
        )
    assert delete_error.value.status_code == 403
