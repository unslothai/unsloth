# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from contextlib import contextmanager

import asyncio
import importlib.util
import sys
import threading
import time
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


def test_endpoint_and_saved_key_update_is_atomic_for_independent_readers(monkeypatch):
    """A reader on another connection sees one complete provider bundle.

    The writer is paused after changing the endpoint but before replacing the
    encrypted key.  This is the inverse interleaving that previously exposed the
    new route with the old key.  The reader uses normal storage calls, each with
    its own SQLite connection, so process-local route locks cannot make it pass.
    """
    provider_id = "atomic-provider"
    old_base_url = "http://127.0.0.1:7770/v1"
    new_base_url = "http://127.0.0.1:8880/v1"
    providers_db.create_provider(
        id = provider_id,
        provider_type = "custom",
        display_name = "Atomic TTS",
        base_url = old_base_url,
        models = ["kokoro"],
    )
    credential_secrets.save_provider_api_key(provider_id, "old-secret")
    monkeypatch.setattr(
        providers_route,
        "resolve_provider_api_key_or_400",
        lambda *_args, **_kwargs: "new-secret",
    )

    between_row_and_key = threading.Event()
    finish_key_write = threading.Event()
    original_save = credential_secrets.save_provider_api_key

    def _paused_save(
        saved_provider_id: str,
        api_key: str,
        *,
        connection = None,
    ) -> None:
        assert connection is not None
        between_row_and_key.set()
        assert finish_key_write.wait(timeout = 5)
        original_save(saved_provider_id, api_key, connection = connection)

    monkeypatch.setattr(credential_secrets, "save_provider_api_key", _paused_save)
    failures: list[BaseException] = []

    def _update() -> None:
        try:
            asyncio.run(
                providers_route.update_provider_config(
                    provider_id,
                    ProviderUpdate(
                        base_url = new_base_url,
                        encrypted_api_key = "replacement-envelope",
                    ),
                    credential = ("alice", None),
                    via_api_key = False,
                )
            )
        except BaseException as exc:
            failures.append(exc)

    writer = threading.Thread(target = _update)
    writer.start()
    try:
        assert between_row_and_key.wait(timeout = 5)
        observed_during_write = (
            providers_db.get_provider(provider_id)["base_url"],
            credential_secrets.get_provider_api_key(provider_id),
        )
    finally:
        finish_key_write.set()
        writer.join(timeout = 5)

    assert not writer.is_alive()
    assert failures == []
    observed_after_commit = (
        providers_db.get_provider(provider_id)["base_url"],
        credential_secrets.get_provider_api_key(provider_id),
    )
    assert observed_during_write == (old_base_url, "old-secret")
    assert observed_after_commit == (new_base_url, "new-secret")


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


def test_known_and_custom_preset_providers_accept_a_non_null_max_output_override():
    """The override replaces the frontend's 32,768-token fallback, which every type
    reaches for a model with no documented cap."""
    for provider_type in ("openai", "openrouter", "vllm", "ollama", "llama_cpp"):
        created = asyncio.run(
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
        assert created.max_output_tokens == 65536
        assert providers_db.get_provider(created.id)["max_output_tokens"] == 65536


def test_chatgpt_subscription_rejects_a_non_null_max_output_override():
    """Codex routing, model list and output cap are fixed, so a stored override would
    never be read."""
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            providers_route.create_provider_config(
                ProviderCreate(
                    provider_type = "openai_codex",
                    display_name = "ChatGPT",
                    max_output_tokens = 65536,
                ),
                credential = ("alice", None),
                via_api_key = False,
            )
        )
    assert error.value.status_code == 400
    # on the detail, not the status: a Codex create with no models also 400s on auth
    assert error.value.detail == "ChatGPT subscriptions use a fixed Max Tokens limit."


def test_known_and_custom_preset_providers_accept_an_explicit_null_max_output_override():
    """A blank Max Tokens limit field serialises as null, not as an omission.

    So every provider type has to accept the null, ChatGPT subscriptions included: an
    unrelated edit -- a rename, a model change, a key rotation -- carries the blank field
    along. Only a non-null value on a subscription is refused.
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


def test_known_provider_accepts_max_output_override_update():
    providers_db.create_provider(
        id = "openai-1",
        provider_type = "openai",
        display_name = "OpenAI",
        base_url = "https://api.openai.com/v1",
    )

    updated = asyncio.run(
        providers_route.update_provider_config(
            "openai-1",
            ProviderUpdate(max_output_tokens = 65536),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    assert updated.max_output_tokens == 65536
    assert providers_db.get_provider("openai-1")["max_output_tokens"] == 65536


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

    def fail_replacement(provider_id: str, api_key: str, **kwargs):
        if api_key == "sk-replacement":
            raise RuntimeError("simulated credential write failure")
        original_save(provider_id, api_key, **kwargs)

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
    assert (
        providers_route.resolve_provider_api_key_or_400(
            "provider-1", "ciphertext", prefer_saved_key = True
        )
        == "saved"
    )
    assert (
        providers_route.resolve_provider_api_key_or_400(
            "provider-1",
            "ciphertext",
            allow_saved_key = False,
            prefer_saved_key = True,
        )
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


def test_codex_update_refreshes_the_plan_catalog_before_validating(monkeypatch):
    """A save must not reject a slug the plan lists just because this process forgot it.

    The catalog lives in memory, so a restart between the picker's fetch and this save
    leaves it empty; the update path refreshes it on the same terms as the chat gate.
    """
    from core.inference import openai_codex_client as codex_client

    created = asyncio.run(
        providers_route.create_provider_config(
            ProviderCreate(
                provider_type = "openai_codex",
                display_name = "ChatGPT subscription",
                models = ["gpt-5.4"],
            ),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    listed = "gpt-5.7-nova"
    codex_client.forget_subscription_models(created.id)

    async def _no_catalog(_provider_id):
        return set()

    monkeypatch.setattr(
        providers_route.openai_codex_client, "ensure_subscription_models", _no_catalog
    )
    with pytest.raises(HTTPException) as refused:
        asyncio.run(
            providers_route.update_provider_config(
                created.id,
                ProviderUpdate(models = [listed]),
                credential = ("alice", None),
                via_api_key = False,
            )
        )
    assert refused.value.status_code == 400

    async def _refresh(provider_id):
        codex_client._offered_models[provider_id] = {
            listed: {"id": listed, "display_name": listed, "vision": None, "listed": True}
        }
        return {listed}

    monkeypatch.setattr(providers_route.openai_codex_client, "ensure_subscription_models", _refresh)
    try:
        updated = asyncio.run(
            providers_route.update_provider_config(
                created.id,
                ProviderUpdate(models = [listed]),
                credential = ("alice", None),
                via_api_key = False,
            )
        )
        assert updated.models == [listed]
    finally:
        codex_client.forget_subscription_models(created.id)


def test_codex_update_of_seed_models_never_reaches_upstream(monkeypatch):
    """A curated-only save must not wait on /codex/models when upstream is down."""
    from core.inference import openai_codex_client as codex_client

    created = asyncio.run(
        providers_route.create_provider_config(
            ProviderCreate(
                provider_type = "openai_codex",
                display_name = "ChatGPT subscription",
                models = ["gpt-5.4"],
            ),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    codex_client.forget_subscription_models(created.id)
    calls = []

    async def _refresh(provider_id):
        calls.append(provider_id)
        return set()

    monkeypatch.setattr(providers_route.openai_codex_client, "ensure_subscription_models", _refresh)
    updated = asyncio.run(
        providers_route.update_provider_config(
            created.id,
            ProviderUpdate(models = ["gpt-5.4", "gpt-5.5"]),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    assert updated.models == ["gpt-5.4", "gpt-5.5"]
    assert calls == []


def test_codex_unrelated_edit_survives_an_unreachable_catalog(monkeypatch):
    """Renaming a connection must not need ChatGPT to be reachable.

    The saved selection was proven when it was first accepted, and the picker
    deliberately preserves it through a curated fallback, so refusing the save during an
    outage would strand the row.
    """
    from core.inference import openai_codex_client as codex_client

    listed = "gpt-5.7-nova"
    created = asyncio.run(
        providers_route.create_provider_config(
            ProviderCreate(
                provider_type = "openai_codex",
                display_name = "ChatGPT subscription",
                models = ["gpt-5.4"],
            ),
            credential = ("alice", None),
            via_api_key = False,
        )
    )

    async def _refresh(provider_id):
        codex_client._offered_models[provider_id] = {listed: {"id": listed, "listed": True}}
        return {listed}

    monkeypatch.setattr(providers_route.openai_codex_client, "ensure_subscription_models", _refresh)
    asyncio.run(
        providers_route.update_provider_config(
            created.id,
            ProviderUpdate(models = [listed]),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    codex_client.forget_subscription_models(created.id)

    calls = []

    async def _unreachable(provider_id):
        calls.append(provider_id)
        return set()

    monkeypatch.setattr(
        providers_route.openai_codex_client, "ensure_subscription_models", _unreachable
    )
    try:
        renamed = asyncio.run(
            providers_route.update_provider_config(
                created.id,
                ProviderUpdate(display_name = "Work account", models = [listed]),
                credential = ("alice", None),
                via_api_key = False,
            )
        )
        assert renamed.display_name == "Work account"
        assert renamed.models == [listed]
        # The slug was already on the row, so nothing had to be proven upstream.
        assert calls == []

        # A slug that was never saved here still needs the catalog.
        with pytest.raises(HTTPException) as refused:
            asyncio.run(
                providers_route.update_provider_config(
                    created.id,
                    ProviderUpdate(models = [listed, "gpt-5.9-unheard-of"]),
                    credential = ("alice", None),
                    via_api_key = False,
                )
            )
        assert refused.value.status_code == 400
        assert calls == [created.id]
    finally:
        codex_client.forget_subscription_models(created.id)


def test_codex_save_refuses_a_seed_the_plan_catalog_omits(monkeypatch):
    """Save and chat must judge on the same evidence.

    The inference route treats a known plan catalog as authoritative, so accepting a
    seed it omits here would persist a model that every send then refuses.
    """
    from core.inference import openai_codex_client as codex_client

    created = asyncio.run(
        providers_route.create_provider_config(
            ProviderCreate(
                provider_type = "openai_codex",
                display_name = "ChatGPT subscription",
                models = ["gpt-5.4"],
            ),
            credential = ("alice", None),
            via_api_key = False,
        )
    )

    async def _no_refresh(_provider_id):
        return set()

    monkeypatch.setattr(
        providers_route.openai_codex_client, "ensure_subscription_models", _no_refresh
    )
    codex_client.forget_subscription_models(created.id)
    codex_client._offered_models[created.id] = {"gpt-5.5": {"id": "gpt-5.5", "listed": True}}
    try:
        with pytest.raises(HTTPException) as refused:
            asyncio.run(
                providers_route.update_provider_config(
                    created.id,
                    ProviderUpdate(models = ["gpt-5.4"]),
                    credential = ("alice", None),
                    via_api_key = False,
                )
            )
        assert refused.value.status_code == 400

        kept = asyncio.run(
            providers_route.update_provider_config(
                created.id,
                ProviderUpdate(models = ["gpt-5.5"]),
                credential = ("alice", None),
                via_api_key = False,
            )
        )
        assert kept.models == ["gpt-5.5"]
    finally:
        codex_client.forget_subscription_models(created.id)


def test_codex_save_refuses_a_row_the_account_cannot_vouch_for(monkeypatch):
    """A cold worker must not save account A's slugs under account B.

    The form submits the whole selection with any edit, so an ordinary rename carries the
    saved slugs with it. The inference route already refuses them, so accepting here
    would persist exactly what every send then rejects.
    """
    from core.inference import openai_codex_client as codex_client

    listed = "gpt-5.7-nova"
    created = asyncio.run(
        providers_route.create_provider_config(
            ProviderCreate(
                provider_type = "openai_codex",
                display_name = "ChatGPT subscription",
                models = ["gpt-5.4"],
            ),
            credential = ("alice", None),
            via_api_key = False,
        )
    )

    async def _refresh(provider_id):
        codex_client._offered_models[provider_id] = {listed: {"id": listed, "listed": True}}
        return {listed}

    monkeypatch.setattr(providers_route.openai_codex_client, "ensure_subscription_models", _refresh)
    asyncio.run(
        providers_route.update_provider_config(
            created.id,
            ProviderUpdate(models = [listed]),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    codex_client.forget_subscription_models(created.id)

    async def _no_refresh(_provider_id):
        return set()

    monkeypatch.setattr(
        providers_route.openai_codex_client, "ensure_subscription_models", _no_refresh
    )
    # Cold worker, and the credentials carry no proof for this account.
    monkeypatch.setattr(
        providers_route.openai_codex_auth,
        "load_oauth_bundle",
        lambda _pid: {"account_id": "acct-b"},
    )
    try:
        with pytest.raises(HTTPException) as refused:
            asyncio.run(
                providers_route.update_provider_config(
                    created.id,
                    ProviderUpdate(display_name = "Renamed", models = [listed]),
                    credential = ("alice", None),
                    via_api_key = False,
                )
            )
        assert refused.value.status_code == 400

        monkeypatch.setattr(
            providers_route.openai_codex_auth,
            "load_oauth_bundle",
            lambda _pid: {"account_id": "acct-b", "catalog_account_id": "acct-b"},
        )
        renamed = asyncio.run(
            providers_route.update_provider_config(
                created.id,
                ProviderUpdate(display_name = "Renamed", models = [listed]),
                credential = ("alice", None),
                via_api_key = False,
            )
        )
        assert renamed.display_name == "Renamed"
        assert renamed.models == [listed]
    finally:
        codex_client.forget_subscription_models(created.id)


def test_codex_save_records_the_account_it_validated_against(monkeypatch):
    """Saving the row is what proves it, so that is where the proof is written."""
    from core.inference import openai_codex_client as codex_client

    recorded = []

    async def _remember(provider_id, account_id):
        recorded.append((provider_id, account_id))

    monkeypatch.setattr(
        providers_route.openai_codex_auth,
        "load_oauth_bundle",
        lambda _pid: {"account_id": "acct-1", "catalog_account_id": "acct-1"},
    )
    monkeypatch.setattr(providers_route.openai_codex_auth, "remember_catalog_account", _remember)

    created = asyncio.run(
        providers_route.create_provider_config(
            ProviderCreate(
                provider_type = "openai_codex",
                display_name = "ChatGPT subscription",
                models = ["gpt-5.4"],
            ),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    # Creating does not record: there is no connection behind it yet.
    assert recorded == []

    codex_client.forget_subscription_models(created.id)
    try:
        asyncio.run(
            providers_route.update_provider_config(
                created.id,
                ProviderUpdate(models = ["gpt-5.4", "gpt-5.5"]),
                credential = ("alice", None),
                via_api_key = False,
            )
        )
        assert recorded == [(created.id, "acct-1")]

        # A metadata-only edit carries no selection, so it proves nothing new.
        recorded.clear()
        asyncio.run(
            providers_route.update_provider_config(
                created.id,
                ProviderUpdate(display_name = "Renamed"),
                credential = ("alice", None),
                via_api_key = False,
            )
        )
        assert recorded == []
    finally:
        codex_client.forget_subscription_models(created.id)


def test_codex_save_that_cannot_record_its_proof_keeps_nothing(monkeypatch):
    """A save is the row and the proof together, so half of it must not survive.

    provider_oauth_write_guard is a 30s flock and _token_request's httpx timeout is
    per-phase, not a total budget, so a refresh holding the guard across a stalled token
    request makes remember_catalog_account raise after update_provider has already
    committed. Without a rollback the row keeps models nothing on disk says were ever
    validated, and that outlives the process: with the plan catalog gone after a restart
    and upstream unreachable, saved_models_proven_for answers False, so chat falls back to
    the seed and even an ordinary rename is refused.
    """
    import json

    from core.inference import openai_codex_auth as codex_auth
    from core.inference import openai_codex_client as codex_client

    listed = "gpt-5-codex-max"  # dynamic: carried by the plan, absent from the seed
    created = asyncio.run(
        providers_route.create_provider_config(
            ProviderCreate(
                provider_type = "openai_codex",
                display_name = "ChatGPT subscription",
                models = ["gpt-5.4"],
            ),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    provider_id = created.id

    credential_secrets.get_or_create_credential_encryption_key()
    credential_secrets.upsert_secret(
        credential_secrets.OPENAI_CODEX_OAUTH_KIND,
        provider_id,
        json.dumps(
            {
                "access_token": "at",
                "refresh_token": "rt",
                "expires_at": int(time.time()) + 3600,
                "account_id": "acct-1",
            }
        ),
    )
    codex_client._offered_models[provider_id] = {
        "gpt-5.4": {"id": "gpt-5.4", "listed": True},
        listed: {"id": listed, "listed": True},
    }
    codex_client._catalog_accounts[provider_id] = "acct-1"

    async def _guard_busy(_provider_id, _account_id):
        raise codex_auth.CodexAuthError("ChatGPT credential update is busy. Please retry.")

    monkeypatch.setattr(providers_route.openai_codex_auth, "remember_catalog_account", _guard_busy)

    try:
        with pytest.raises(codex_auth.CodexAuthError):
            asyncio.run(
                providers_route.update_provider_config(
                    provider_id,
                    ProviderUpdate(models = ["gpt-5.4", listed]),
                    credential = ("alice", None),
                    via_api_key = False,
                )
            )

        # The reported failure and the stored row agree: neither half landed.
        row = providers_db.get_provider(provider_id)
        assert row["models"] == ["gpt-5.4"]
        bundle = codex_auth.load_oauth_bundle(provider_id)
        assert bundle is not None and bundle.get("catalog_account_id") is None

        # Restart: the catalog is per process, the row and the bundle are not.
        codex_client.forget_subscription_models(provider_id)
        assert not codex_client.subscription_catalog_known(provider_id)

        # The row carries no slug the seed cannot vouch for, so an unrelated edit still
        # saves with upstream unreachable. Left torn, this is a 400.
        async def _unreachable(_provider_id):
            return set()

        monkeypatch.setattr(
            providers_route.openai_codex_client, "ensure_subscription_models", _unreachable
        )
        recorded = []

        async def _remember(pid, account_id):
            recorded.append((pid, account_id))

        monkeypatch.setattr(
            providers_route.openai_codex_auth, "remember_catalog_account", _remember
        )
        renamed = asyncio.run(
            providers_route.update_provider_config(
                provider_id,
                ProviderUpdate(display_name = "Renamed", models = ["gpt-5.4"]),
                credential = ("alice", None),
                via_api_key = False,
            )
        )
        assert renamed.display_name == "Renamed"
        assert renamed.models == ["gpt-5.4"]
        assert recorded == [(provider_id, "acct-1")]
    finally:
        codex_client.forget_subscription_models(provider_id)


def test_codex_save_records_only_the_account_it_actually_validated(monkeypatch):
    """A rebind between validating and recording must not stamp the new account.

    remember_catalog_account writes only when the bundle still names the account handed
    to it, so passing the one validation used is what makes the record honest.
    """
    from core.inference import openai_codex_client as codex_client

    recorded = []

    async def _remember(provider_id, account_id):
        recorded.append((provider_id, account_id))

    lookups = []

    def _bundle(_pid):
        lookups.append(1)
        # The first read is the one this request judges by; a rebind lands after it, so
        # every later read of the connection names the new account.
        return {"account_id": "acct-a" if len(lookups) == 1 else "acct-b"}

    monkeypatch.setattr(providers_route.openai_codex_auth, "load_oauth_bundle", _bundle)
    monkeypatch.setattr(providers_route.openai_codex_auth, "remember_catalog_account", _remember)

    created = asyncio.run(
        providers_route.create_provider_config(
            ProviderCreate(
                provider_type = "openai_codex",
                display_name = "ChatGPT subscription",
                models = ["gpt-5.4"],
            ),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    codex_client.forget_subscription_models(created.id)
    lookups.clear()
    try:
        asyncio.run(
            providers_route.update_provider_config(
                created.id,
                ProviderUpdate(models = ["gpt-5.4"]),
                credential = ("alice", None),
                via_api_key = False,
            )
        )
        assert recorded == [(created.id, "acct-a")]
    finally:
        codex_client.forget_subscription_models(created.id)


def test_deleting_a_codex_connection_releases_its_plan_catalog():
    """The catalog is per connection and per process, so the delete has to release it.

    Disconnecting the OAuth bundle goes through forget_subscription_models; deleting the
    whole connection took a different path and left the catalog, the account marker and
    the request ticket behind for the life of the process. Provider ids come from uuid4,
    so nothing stale was ever consulted again, but nothing reclaimed it either and a user
    who adds and removes connections grew the maps without bound.
    """
    from core.inference import openai_codex_client as codex_client

    created = asyncio.run(
        providers_route.create_provider_config(
            ProviderCreate(
                provider_type = "openai_codex",
                display_name = "ChatGPT subscription",
                models = ["gpt-5.4"],
            ),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    provider_id = created.id
    models = [{"id": "gpt-5.4", "display_name": "GPT-5.4", "listed": True}]
    codex_client._models_cache[provider_id] = (time.time() + 600, models)
    codex_client._offered_models[provider_id] = {model["id"]: model for model in models}
    codex_client._catalog_accounts[provider_id] = "acct-1"
    codex_client.mark_subscription_catalog_stale(provider_id)
    codex_client._begin_catalog_request(provider_id)
    assert codex_client.subscription_catalog_known(provider_id) is True

    try:
        asyncio.run(
            providers_route.delete_provider_config(
                provider_id, credential = ("alice", None), via_api_key = False
            )
        )
        assert providers_db.get_provider(provider_id) is None
        assert provider_id not in codex_client._models_cache
        assert provider_id not in codex_client._offered_models
        assert provider_id not in codex_client._catalog_accounts
        assert provider_id not in codex_client._stale_catalogs
        assert provider_id not in codex_client._catalog_requests
        assert codex_client.subscription_catalog_known(provider_id) is False
        assert codex_client.offered_subscription_model_ids(provider_id) == set()
    finally:
        codex_client.forget_subscription_models(provider_id)


def test_a_released_connection_leaves_no_ticket_but_still_retires_its_read():
    """forget_subscription_models drops the ticket rather than bumping it.

    The counter is shared by every connection, so a number is never reissued and an
    outstanding read cannot be matched by whatever starts next. Keeping a per-connection
    entry alive purely to hold the high-water mark was the last thing the release path
    could not reclaim.
    """
    from core.inference import openai_codex_client as codex_client

    ticket = codex_client._begin_catalog_request("released-connection")
    codex_client.forget_subscription_models("released-connection")
    assert "released-connection" not in codex_client._catalog_requests
    # The read that was in flight when the connection went away must still decline.
    assert codex_client._catalog_requests.get("released-connection") != ticket
    # A later read anywhere draws a number the retired one cannot be holding.
    assert codex_client._begin_catalog_request("another-connection") > ticket
    assert codex_client._begin_catalog_request("released-connection") != ticket
    codex_client.forget_subscription_models("released-connection")
    codex_client.forget_subscription_models("another-connection")


def test_codex_proof_rollback_leaves_a_concurrent_save_alone(monkeypatch):
    """The undo takes back this request's write, not whatever the row says by then.

    update_provider_config suspends between committing the row and recording the proof:
    remember_catalog_account awaits a 30s file lock, and the failure the rollback exists
    for is exactly the one where that lock was contended. Another write to the row lands
    in between. Restoring the whole pre-request snapshot would put the row back to before
    *both* writes and erase the second one's successful edit.

    The interleaved write goes at providers_db directly rather than through a second
    update_provider_config call. serialize_provider_config holds a per-provider
    asyncio.Lock across the whole handler, so a second call on the same provider cannot
    run while this one is parked -- awaiting one from inside this test deadlocks the
    event loop, since the await that would release the gate sits behind it. What the
    rollback actually defends against is the row moving under the handler, which any
    writer can do, so the row write is both the reachable shape and the one under test.
    """
    import json

    from core.inference import openai_codex_auth as codex_auth
    from core.inference import openai_codex_client as codex_client

    listed = "gpt-5-codex-max"  # dynamic: carried by the plan, absent from the seed
    created = asyncio.run(
        providers_route.create_provider_config(
            ProviderCreate(
                provider_type = "openai_codex",
                display_name = "Original name",
                models = ["gpt-5.4"],
            ),
            credential = ("alice", None),
            via_api_key = False,
        )
    )
    provider_id = created.id

    credential_secrets.get_or_create_credential_encryption_key()
    credential_secrets.upsert_secret(
        credential_secrets.OPENAI_CODEX_OAUTH_KIND,
        provider_id,
        json.dumps(
            {
                "access_token": "at",
                "refresh_token": "rt",
                "expires_at": int(time.time()) + 3600,
                "account_id": "acct-1",
            }
        ),
    )
    codex_client._offered_models[provider_id] = {
        "gpt-5.4": {"id": "gpt-5.4", "listed": True},
        listed: {"id": listed, "listed": True},
    }
    codex_client._catalog_accounts[provider_id] = "acct-1"

    gate = asyncio.Event()
    parked = asyncio.Event()

    async def _blocked_then_busy(_provider_id, _account_id):
        # Stands in for provider_oauth_write_guard's flock acquire: a real suspension
        # point, then the timeout it raises. `parked` is the handshake, so the test
        # resumes on the suspension itself rather than on a count of loop turns.
        parked.set()
        await gate.wait()
        raise codex_auth.CodexAuthError("ChatGPT credential update is busy. Please retry.")

    monkeypatch.setattr(
        providers_route.openai_codex_auth, "remember_catalog_account", _blocked_then_busy
    )

    async def _overlapping_saves():
        adds_a_model = asyncio.ensure_future(
            providers_route.update_provider_config(
                provider_id,
                ProviderUpdate(models = ["gpt-5.4", listed]),
                credential = ("alice", None),
                via_api_key = False,
            )
        )
        await asyncio.wait_for(parked.wait(), timeout = 10)
        # Its row is committed and it is now parked in remember_catalog_account.
        assert providers_db.get_provider(provider_id)["models"] == ["gpt-5.4", listed]

        # Another write renames the row while that one hangs. It touches a column the
        # parked request never wrote, so there is nothing of its own to undo there.
        providers_db.update_provider(
            id = provider_id, display_name = "Renamed while the first save hung"
        )

        gate.set()
        with pytest.raises(codex_auth.CodexAuthError):
            await adds_a_model

    try:
        asyncio.run(_overlapping_saves())

        row = providers_db.get_provider(provider_id)
        # The rename was never in doubt, and nothing it did needs undoing.
        assert row["display_name"] == "Renamed while the first save hung"
        # The unproven model still goes back: that half of the failed save is this
        # request's own, and no one else claimed the column.
        assert row["models"] == ["gpt-5.4"]
        bundle = codex_auth.load_oauth_bundle(provider_id)
        assert bundle is not None and bundle.get("catalog_account_id") is None
    finally:
        codex_client.forget_subscription_models(provider_id)
