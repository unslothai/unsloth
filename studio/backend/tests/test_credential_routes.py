# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from contextlib import contextmanager

import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from auth import storage as auth_storage
from models.providers import (
    ProviderCreate,
    ProviderCredentialMigration,
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
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    assert created.has_api_key is True
    assert credential_secrets.get_provider_api_key(created.id) == "sk-first"

    metadata_only = asyncio.run(
        providers_route.update_provider_config(
            created.id,
            ProviderUpdate(display_name = "Renamed"),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    assert metadata_only.display_name == "Renamed"
    assert credential_secrets.get_provider_api_key(created.id) == "sk-first"

    replaced = asyncio.run(
        providers_route.update_provider_config(
            created.id,
            ProviderUpdate(encrypted_api_key = "second"),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    assert replaced.has_api_key is True
    assert credential_secrets.get_provider_api_key(created.id) == "sk-second"

    cleared = asyncio.run(
        providers_route.update_provider_config(
            created.id,
            ProviderUpdate(clear_api_key = True),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    assert cleared.has_api_key is False
    assert credential_secrets.get_provider_api_key(created.id) is None

    credential_secrets.save_provider_api_key(created.id, "sk-before-delete")

    asyncio.run(
        providers_route.delete_provider_config(
            created.id, credential = ("alice", None), via_api_key = False
        )
    )
    asyncio.run(
        providers_route.delete_provider_config(
            created.id, credential = ("alice", None), via_api_key = False
        )
    )

    assert credential_secrets.get_provider_api_key(created.id) is None
    assert providers_db.get_provider(created.id) is None


def test_custom_max_output_tokens_create_update_and_clear():
    created = asyncio.run(
        providers_route.create_provider_config(
            ProviderCreate(
                provider_type = "custom",
                display_name = "Custom",
                base_url = "https://example.com/v1",
                models = ["vendor/model"],
                max_output_tokens = 131072,
            ),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    assert created.max_output_tokens == 131072
    assert providers_db.get_provider(created.id)["max_output_tokens"] == 131072

    updated = asyncio.run(
        providers_route.update_provider_config(
            created.id,
            ProviderUpdate(max_output_tokens = 65536),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    assert updated.max_output_tokens == 65536

    preserved = asyncio.run(
        providers_route.update_provider_config(
            created.id,
            ProviderUpdate(display_name = "Renamed Custom"),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    assert preserved.display_name == "Renamed Custom"
    assert preserved.max_output_tokens == 65536

    cleared = asyncio.run(
        providers_route.update_provider_config(
            created.id,
            ProviderUpdate(max_output_tokens = None),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    assert cleared.max_output_tokens is None


@pytest.mark.parametrize("value", [1_048_577, 9_007_199_254_740_991])
def test_custom_max_output_tokens_accepts_safe_integer_values(value):
    payload = ProviderCreate(
        provider_type = "custom",
        display_name = "Custom",
        max_output_tokens = value,
    )
    assert payload.max_output_tokens == value


@pytest.mark.parametrize(
    "value",
    [63, 4096.5, "4096", True, 9_007_199_254_740_992],
)
def test_custom_max_output_tokens_requires_a_safe_integer(value):
    with pytest.raises(ValidationError):
        ProviderCreate(
            provider_type = "custom",
            display_name = "Custom",
            max_output_tokens = value,
        )


def test_known_and_custom_preset_providers_reject_a_non_null_max_output_override():
    for provider_type in ("openai", "vllm", "ollama", "llama_cpp"):
        with pytest.raises(HTTPException) as error:
            asyncio.run(
                providers_route.create_provider_config(
                    ProviderCreate(
                        provider_type = provider_type,
                        display_name = provider_type,
                        base_url = "https://example.com/v1",
                        max_output_tokens = 65536,
                    ),
                    credential = ("alice", None),
                    via_api_key = False,
                )
            )
        assert error.value.status_code == 400


def test_known_and_custom_preset_providers_accept_an_explicit_null_max_output_override():
    """A blank Max Tokens limit field serialises as null, not as an omission.

    The dialog renders that field from the UI provider type, which resolves to
    "custom" for a connection STORED as `openai` once the user has renamed it or
    pointed it at another base URL. Rejecting the null as well as a real value meant
    every unrelated edit of such a connection -- a rename, a model change, a key
    rotation -- failed with an error about a field the user never touched. Clearing
    an override that cannot exist is a no-op, so the null is accepted everywhere and
    only a non-null value is refused.
    """
    for provider_type in ("openai", "vllm", "ollama", "llama_cpp"):
        created = asyncio.run(
            providers_route.create_provider_config(
                ProviderCreate(
                    provider_type = provider_type,
                    display_name = provider_type,
                    base_url = "https://example.com/v1",
                    max_output_tokens = None,
                ),
                credential = ("alice", None),
                via_api_key = False,
            )
        )
        assert created.max_output_tokens is None
        assert providers_db.get_provider(created.id)["max_output_tokens"] is None


def test_known_provider_rejects_max_output_override_update():
    providers_db.create_provider(
        id = "openai-1",
        provider_type = "openai",
        display_name = "OpenAI",
        base_url = "https://api.openai.com/v1",
    )

    with pytest.raises(HTTPException) as error:
        asyncio.run(
            providers_route.update_provider_config(
                "openai-1",
                ProviderUpdate(max_output_tokens = 65536),
                credential = ("alice", None),
                via_api_key = False,
            )
        )
    assert error.value.status_code == 400


def test_known_provider_accepts_a_null_max_output_override_update():
    """The counterpart on the update path: a blank field must not block a rename."""
    providers_db.create_provider(
        id = "openai-1",
        provider_type = "openai",
        display_name = "OpenAI",
        base_url = "https://api.openai.com/v1",
    )

    updated = asyncio.run(
        providers_route.update_provider_config(
            "openai-1",
            ProviderUpdate(display_name = "Renamed", max_output_tokens = None),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    assert updated.display_name == "Renamed"
    assert updated.max_output_tokens is None
    assert providers_db.get_provider("openai-1")["max_output_tokens"] is None


def test_provider_update_validates_before_writes_and_rolls_back_metadata(monkeypatch):
    providers_db.create_provider(
        id = "provider-1",
        provider_type = "openai",
        display_name = "Original",
        base_url = "https://api.openai.com/v1",
    )
    credential_secrets.save_provider_api_key("provider-1", "sk-original")

    def invalid_envelope(_provider_id, _envelope, **_kwargs):
        raise HTTPException(status_code = 400, detail = "invalid envelope")

    monkeypatch.setattr(providers_route, "resolve_provider_api_key_or_400", invalid_envelope)
    with pytest.raises(HTTPException):
        asyncio.run(
            providers_route.update_provider_config(
                "provider-1",
                ProviderUpdate(display_name = "Must not persist", encrypted_api_key = "invalid"),
                credential = ("alice", None),
                via_api_key = False,
            )
        )
    assert providers_db.get_provider("provider-1")["display_name"] == "Original"

    monkeypatch.setattr(
        providers_route,
        "resolve_provider_api_key_or_400",
        lambda _provider_id, _envelope, **_kwargs: "sk-replacement",
    )
    original_save = credential_secrets.save_provider_api_key

    def fail_replacement(provider_id: str, api_key: str):
        if api_key == "sk-replacement":
            raise RuntimeError("simulated credential write failure")
        original_save(provider_id, api_key)

    monkeypatch.setattr(credential_secrets, "save_provider_api_key", fail_replacement)
    with pytest.raises(RuntimeError, match = "credential write failure"):
        asyncio.run(
            providers_route.update_provider_config(
                "provider-1",
                ProviderUpdate(display_name = "Also rolled back", encrypted_api_key = "valid"),
                credential = ("alice", None),
                via_api_key = False,
            )
        )

    assert providers_db.get_provider("provider-1")["display_name"] == "Original"
    assert credential_secrets.get_provider_api_key("provider-1") == "sk-original"


def test_provider_delete_restores_key_when_provider_delete_fails(monkeypatch):
    providers_db.create_provider(
        id = "provider-1",
        provider_type = "openai",
        display_name = "OpenAI",
        base_url = "https://api.openai.com/v1",
    )
    credential_secrets.save_provider_api_key("provider-1", "sk-original")

    def fail_delete(_provider_id: str):
        raise RuntimeError("simulated provider delete failure")

    monkeypatch.setattr(providers_db, "delete_provider", fail_delete)
    with pytest.raises(RuntimeError, match = "provider delete failure"):
        asyncio.run(
            providers_route.delete_provider_config(
                "provider-1", credential = ("alice", None), via_api_key = False
            )
        )

    assert providers_db.get_provider("provider-1") is not None
    assert credential_secrets.get_provider_api_key("provider-1") == "sk-original"


def test_credential_writes_reject_a_rotated_request(monkeypatch):
    @contextmanager
    def reject_stale(_subject, _generation):
        raise auth_storage.CredentialRotated("revoked")
        yield

    monkeypatch.setattr(auth_storage, "credential_generation_guard", reject_stale)
    with pytest.raises(HTTPException) as error:
        settings_route.update_hugging_face_token(
            settings_route.HuggingFaceTokenPayload(token = "hf_stale"),
            credential = ("alice", "stale-generation"),
            via_api_key = False,
        )
    assert error.value.status_code == 401
    assert credential_secrets.get_hf_token() is None


def test_explicit_provider_key_preserves_the_edited_target():
    payload = ProviderModelsRequest(
        provider_id = "provider-1",
        provider_type = "custom",
        encrypted_api_key = "encrypted-replacement",
        base_url = "https://new.example/v1",
    )
    assert providers_route._bind_saved_provider_target(payload) is payload


def test_provider_mutations_reject_api_key_authentication():
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            providers_route.create_provider_config(
                ProviderCreate(provider_type = "openai", display_name = "Forbidden"),
                credential = ("alice", None),
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
    jwt_secret = "current-jwt-secret"
    auth_storage.create_initial_user("alice", "password-123", jwt_secret)
    credential = ("alice", auth_storage.credential_generation(jwt_secret))
    # Exercise a cold encryption-key cache while the real generation guard is active.
    auth_storage._credential_encryption_key_cache = None
    saved = settings_route.update_hugging_face_token(
        settings_route.HuggingFaceTokenPayload(token = " 'hf_alice' "),
        credential = credential,
        via_api_key = False,
    )
    assert saved.token == "hf_alice"
    assert settings_route.get_hugging_face_token("alice", via_api_key = False).token == "hf_alice"
    assert settings_route.get_hugging_face_token("bob", via_api_key = False).token == "hf_alice"

    assert settings_route.clear_hugging_face_token(credential, via_api_key = False).has_token is False
    assert settings_route.clear_hugging_face_token(credential, via_api_key = False).has_token is False


def test_legacy_migration_never_replaces_newer_credentials(monkeypatch):
    credential = ("alice", None)
    settings_route.update_hugging_face_token(
        settings_route.HuggingFaceTokenPayload(token = "hf_newer"),
        credential = credential,
        via_api_key = False,
    )
    migrated_hf = settings_route.migrate_hugging_face_token(
        settings_route.HuggingFaceTokenPayload(token = "hf_legacy"),
        credential = credential,
        via_api_key = False,
    )
    assert migrated_hf.token == "hf_newer"

    providers_db.create_provider(
        id = "provider-1",
        provider_type = "openai",
        display_name = "OpenAI",
        base_url = "https://api.openai.com/v1",
    )
    credential_secrets.save_provider_api_key("provider-1", "sk-newer")
    monkeypatch.setattr(
        providers_route,
        "resolve_provider_api_key_or_400",
        lambda *_args, **_kwargs: "sk-legacy",
    )
    migrated_provider = asyncio.run(
        providers_route.migrate_provider_api_key(
            "provider-1",
            ProviderCredentialMigration(encrypted_api_key = "encrypted-legacy"),
            credential = credential,
            via_api_key = False,
        )
    )
    assert migrated_provider.has_api_key is True
    assert credential_secrets.get_provider_api_key("provider-1") == "sk-newer"


def test_hugging_face_secret_routes_reject_api_key_authentication():
    with pytest.raises(HTTPException) as get_error:
        settings_route.get_hugging_face_token("alice", via_api_key = True)
    assert get_error.value.status_code == 403

    with pytest.raises(HTTPException) as put_error:
        settings_route.update_hugging_face_token(
            settings_route.HuggingFaceTokenPayload(token = "hf_forbidden"),
            credential = ("alice", None),
            via_api_key = True,
        )
    assert put_error.value.status_code == 403

    with pytest.raises(HTTPException) as delete_error:
        settings_route.clear_hugging_face_token(
            ("alice", None),
            via_api_key = True,
        )
    assert delete_error.value.status_code == 403
