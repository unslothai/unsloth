# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""External LLM provider routes: types, saved configs; API keys RSA-encrypted; tests, model listing."""

import asyncio
import hashlib
import ipaddress
import json
import re
import time
import uuid
from typing import Optional
from urllib.parse import urlsplit

import httpx
import structlog
from fastapi import APIRouter, Depends, HTTPException

from auth.authentication import (
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
)

from routes.provider_credentials import (
    current_credential_write,
    require_ui_session,
    resolve_provider_api_key_or_400,
    serialize_provider_config,
)
from core.inference.key_exchange import (
    get_public_key_fingerprint,
    get_public_key_pem,
)
from core.inference.providers import (
    answers_decisions_only,
    get_base_url,
    get_connectable_provider_info,
    get_provider_info,
    list_available_providers,
    validate_provider_base_url,
)
from core.inference.pricing import pricing_snapshot
from core.inference.external_provider import ExternalProviderClient

from core.inference import openai_codex_auth, openai_codex_client
from core.inference.provider_model_capabilities import (
    MODEL_CAPABILITY_PROVIDERS,
    MODELS_DEV_URL,
    provider_model_capabilities,
    trim_models_dev_catalog,
)
from models.providers import (
    ModelCatalogResponse,
    ProviderCreate,
    ProviderCredentialMigration,
    ProviderModelsRequest,
    ProviderModelCapabilityInfo,
    ProviderModelInfo,
    ProviderResponse,
    ProviderRegistryEntry,
    ProviderTestRequest,
    ProviderTestResult,
    ProviderUpdate,
    validate_provider_reasoning_contract,
)
from storage import credential_secrets, providers_db
from hub.services.models import account_access
from utils.paths.storage_roots import cache_root
from utils.utils import safe_curated_detail, log_and_http_error

logger = structlog.get_logger(__name__)


router = APIRouter(dependencies = [Depends(get_current_subject)])

_MAX_RESPONSES_CONNECTIVITY_MODELS = 5
_PROVIDER_CONNECTIVITY_TIMEOUT_SECONDS = 15.0
_RESPONSES_PROBE_BUDGET_SECONDS = 20.0
_API_KEY_REJECTED = "The server rejected the API key."
# Their /models answers without auth, so only a second request can tell a bad key apart.
_KEY_CHECKED_PROVIDERS = frozenset({"openrouter", "huggingface"})
_URL_USERINFO = re.compile(r"(\b[a-z][a-z0-9+.-]*://)[^/\s@'\"]+@", re.IGNORECASE)


def _redact_url_userinfo(text: str) -> str:
    return _URL_USERINFO.sub(r"\1", text)


def _unreachable_detail(base_url: str) -> str:
    try:
        parts = urlsplit(base_url)
        host = parts.hostname or ""
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return "Couldn't reach the server. Is it running?"
    try:
        loopback = host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    shown = f"[{host}]" if ":" in host else host
    detail = f"Couldn't reach {shown}:{port}. Is the server running?"
    if loopback:
        detail += " If Studio runs in Docker, try host.docker.internal."
    return detail


def _known_provider_error(
    exc: Exception,
    base_url: str,
    *,
    listing_models: bool = False,
) -> str | None:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status in (401, 403):
            return _API_KEY_REJECTED
        if listing_models and status in (404, 405):
            return "No /models endpoint at this URL (check that it ends in /v1)."
    elif isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
        return _unreachable_detail(base_url)
    return None


def _provider_error_detail(
    exc: Exception,
    base_url: str,
    *,
    listing_models: bool = False,
) -> str:
    return _known_provider_error(
        exc, base_url, listing_models = listing_models
    ) or _redact_url_userinfo(safe_curated_detail(exc))


def _log_provider_failure(event: str, exc: Exception, **fields) -> None:
    error = _redact_url_userinfo(str(exc))
    # A traceback repeats the raw message, so it is kept only when there was nothing to redact.
    logger.error(event, error = error, exc_info = error == str(exc), **fields)


_SHARED_READ_ONLY = "This connection is shared by the installation owner. Only they can change it."


def _provider_response(row: dict) -> ProviderResponse:
    has_api_key = credential_secrets.provider_api_key_saved(row["id"])
    return ProviderResponse(
        id = row["id"],
        provider_type = row["provider_type"],
        display_name = row["display_name"],
        base_url = row["base_url"],
        api_type = row.get("api_type", "chat_completions"),
        reasoning_config = row.get("reasoning_config"),
        is_enabled = bool(row["is_enabled"]),
        has_api_key = has_api_key,
        auth_kind = ("chatgpt_oauth" if row["provider_type"] == "openai_codex" else "api_key"),
        auth_status = (
            openai_codex_auth.auth_status(row["id"])
            if row["provider_type"] == "openai_codex"
            else ("connected" if has_api_key else "disconnected")
        ),
        models = row.get("models") or [],
        available_models = row.get("available_models") or [],
        max_output_tokens = row.get("max_output_tokens"),
        shared = bool(row.get("shared")),
        shared_by_owner = bool(row.get("shared_by_owner")),
        created_at = row["created_at"],
        updated_at = row["updated_at"],
    )


def _validate_share_request(
    shared: bool | None,
    provider_type: str,
    base_url: str | None = None,
) -> None:
    if not shared:
        return
    if account_access.managed_account():
        raise HTTPException(
            status_code = 403, detail = "Only the installation owner can share connections."
        )
    if provider_type == "openai_codex":
        raise HTTPException(
            status_code = 400, detail = "A ChatGPT subscription connection cannot be shared."
        )
    if providers_db.base_url_carries_credential(base_url):
        # Other accounts are shown the base URL, so a credential inside it would be shared too.
        raise HTTPException(
            status_code = 400,
            detail = "Move the credential in this base URL into the API key field to share it.",
        )


def _own_provider_or_error(provider_id: str) -> dict:
    existing = providers_db.get_own_provider(provider_id)
    if existing is not None:
        return existing
    if providers_db.is_owner_shared_provider(provider_id):
        raise HTTPException(status_code = 403, detail = _SHARED_READ_ONLY)
    raise HTTPException(status_code = 404, detail = "Provider not found")


def _validate_provider_auth_contract(
    info: dict,
    *,
    encrypted_api_key: str | None,
    base_url: str | None,
    models: list[str] | None,
    updating: bool,
    clear_api_key: bool = False,
    provider_id: str | None = None,
    persisted_models: list[str] | None = None,
    validated_account: str | None = None,
) -> None:
    if info.get("auth_kind") != "chatgpt_oauth":
        return
    if encrypted_api_key or clear_api_key:
        raise HTTPException(status_code = 400, detail = "ChatGPT subscriptions do not use API keys.")
    if base_url is not None and (not updating or base_url != info["base_url"]):
        raise HTTPException(status_code = 400, detail = "ChatGPT subscription routing is fixed.")
    if models is None:
        return
    # Same evidence order as the chat route, so a save cannot persist a model every send refuses.
    if provider_id and openai_codex_client.subscription_catalog_known(provider_id):
        allowed = openai_codex_client.offered_subscription_model_ids(provider_id) | {
            slug
            for slug in (persisted_models or [])
            if openai_codex_client.offered_subscription_model(provider_id, slug) is not None
        }
    else:
        allowed = set(info["default_models"])
        proven = not provider_id or openai_codex_client.saved_models_proven_for(
            provider_id, validated_account
        )
        if (
            persisted_models
            and proven
            and not (provider_id and openai_codex_client.subscription_catalog_stale(provider_id))
        ):
            # Already accepted once, so an upstream outage must not block unrelated edits.
            allowed |= set(persisted_models)
    if not models or not set(models).issubset(allowed):
        raise HTTPException(status_code = 400, detail = "Choose only curated Codex models.")


def _validate_max_output_tokens_contract(
    provider_type: str,
    field_was_set: bool,
    value: Optional[int] = None,
) -> None:
    """Rejects a non-null override on ChatGPT subscriptions, where the output cap is fixed."""
    if field_was_set and value is not None and provider_type == "openai_codex":
        raise HTTPException(
            status_code = 400,
            detail = "ChatGPT subscriptions use a fixed Max Tokens limit.",
        )


@router.get("/public-key")
async def get_public_key(current_subject: str = Depends(get_current_subject)):
    """Return the RSA public key PEM for client-side API key encryption.

    ``fingerprint`` is a short SHA256 of the PEM; a mismatch with what the
    frontend captured at encrypt time signals the keypair rotated mid-flight.
    """
    return {
        "public_key": get_public_key_pem(),
        "fingerprint": get_public_key_fingerprint(),
    }


@router.get("/registry", response_model = list[ProviderRegistryEntry])
async def list_registry(
    include_hidden: bool = False,
    include_oauth: bool = False,
    current_subject: str = Depends(get_current_subject),
):
    """List all supported provider types with their default configurations.

    ``include_hidden=true`` also returns the backend-only entries (the
    self-hosted presets), which carry the studio-tools capability the composer
    needs. It is opt-in so that a browser still running a pre-capability bundle,
    which does not know to filter on ``hidden``, keeps seeing exactly the list
    it saw before and cannot render them as duplicate dropdown options.
    OAuth rows need ``include_hidden`` or ``include_oauth``.
    """
    return list_available_providers(include_hidden = include_hidden, include_oauth = include_oauth)


@router.get("/pricing")
async def get_pricing_snapshot(current_subject: str = Depends(get_current_subject)):
    """Static per-MTok pricing table the frontend uses to convert upstream
    usage into per-turn USD cost. See ``core/inference/pricing.py`` for sourcing."""
    return pricing_snapshot()


# FastAPI offloads sync reads; mutations stay on-loop to preserve atomic sequences.
@router.get("/", response_model = list[ProviderResponse])
def list_provider_configs(_current_subject: str = Depends(get_current_subject)):
    """List all saved provider configurations."""
    rows = providers_db.list_providers()
    return [_provider_response(row) for row in rows]


@router.post("/", response_model = ProviderResponse, status_code = 201)
async def create_provider_config(
    payload: ProviderCreate,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """Create a saved provider configuration and optional encrypted API key."""

    require_ui_session(via_api_key)
    info = get_connectable_provider_info(payload.provider_type)
    if info is None:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown provider type: {payload.provider_type}. "
            f"Use GET /api/providers/registry to see available types.",
        )

    try:
        validate_provider_reasoning_contract(
            payload.provider_type, payload.api_type, payload.reasoning_config
        )
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None

    _validate_max_output_tokens_contract(
        payload.provider_type,
        "max_output_tokens" in payload.model_fields_set,
        payload.max_output_tokens,
    )

    _validate_provider_auth_contract(
        info,
        encrypted_api_key = payload.encrypted_api_key,
        base_url = payload.base_url,
        models = payload.models,
        updating = False,
    )

    base_url = payload.base_url or info["base_url"]
    _validate_share_request(payload.shared, payload.provider_type, base_url)
    # An empty base URL stays allowed (custom/vLLM entries carry none until the
    # user fills one in); anything present is checked before a key is decrypted.
    if base_url:
        try:
            base_url = await asyncio.to_thread(validate_provider_base_url, base_url)
        except ValueError as exc:
            raise HTTPException(status_code = 400, detail = str(exc)) from None

    api_key = resolve_provider_api_key_or_400(None, payload.encrypted_api_key)
    provider_id = uuid.uuid4().hex[:16]

    if api_key:
        credential_secrets.get_or_create_credential_encryption_key()
    with current_credential_write(credential):
        providers_db.create_provider(
            id = provider_id,
            provider_type = payload.provider_type,
            display_name = payload.display_name,
            base_url = base_url,
            models = payload.models,
            available_models = payload.available_models,
            max_output_tokens = payload.max_output_tokens,
            api_type = payload.api_type,
            reasoning_config = payload.reasoning_config,
            shared = payload.shared,
        )
        try:
            if api_key:
                credential_secrets.save_provider_api_key(provider_id, api_key)
        except Exception:
            providers_db.delete_provider(provider_id)
            raise

    row = providers_db.get_provider(provider_id)
    return _provider_response(row)


@router.put("/{provider_id}", response_model = ProviderResponse)
@serialize_provider_config
async def update_provider_config(
    provider_id: str,
    payload: ProviderUpdate,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """Update a saved provider configuration."""

    require_ui_session(via_api_key)
    existing = _own_provider_or_error(provider_id)
    _validate_share_request(
        payload.shared if payload.shared is not None else existing.get("shared"),
        existing["provider_type"],
        payload.base_url or existing["base_url"],
    )

    reasoning_config_requested = "reasoning_config" in payload.model_fields_set
    effective_reasoning_config = (
        payload.reasoning_config if reasoning_config_requested else existing.get("reasoning_config")
    )
    try:
        validate_provider_reasoning_contract(
            existing["provider_type"],
            payload.api_type or existing.get("api_type", "chat_completions"),
            effective_reasoning_config,
        )
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None

    existing_info = get_provider_info(existing["provider_type"]) or {}
    max_output_tokens_requested = "max_output_tokens" in payload.model_fields_set
    _validate_max_output_tokens_contract(
        existing["provider_type"],
        max_output_tokens_requested,
        payload.max_output_tokens,
    )
    persisted_models = list(existing.get("models") or [])
    # Single ownership read: a later lookup could name a different account.
    validated_account: str | None = None
    if existing_info.get("auth_kind") == "chatgpt_oauth":
        # Bundle is shared via DB but catalog is per process; another worker may have rebound it.
        current_bundle = openai_codex_auth.load_oauth_bundle(provider_id)
        validated_account = current_bundle.get("account_id") if current_bundle else None
        current_account = validated_account
        if current_account and not openai_codex_client.subscription_catalog_matches_account(
            provider_id, current_account
        ):
            openai_codex_client.forget_subscription_models(provider_id)
            openai_codex_client.mark_subscription_catalog_stale(provider_id)
    if existing_info.get("auth_kind") == "chatgpt_oauth" and payload.models:
        # Only unproven slugs hit the catalog, so saves do not wait on upstream timeouts.
        unproven = (
            set(payload.models) - set(existing_info["default_models"]) - set(persisted_models)
        )
        if unproven:
            try:
                await openai_codex_client.ensure_subscription_models(provider_id)
            except (
                openai_codex_auth.CodexAuthError,
                openai_codex_client.CodexReauthorizationError,
            ) as exc:
                raise HTTPException(status_code = 401, detail = str(exc)) from exc
    _validate_provider_auth_contract(
        existing_info,
        encrypted_api_key = payload.encrypted_api_key,
        base_url = payload.base_url,
        models = payload.models,
        updating = True,
        clear_api_key = payload.clear_api_key,
        provider_id = provider_id,
        persisted_models = persisted_models,
        validated_account = validated_account,
    )

    if payload.clear_api_key and payload.encrypted_api_key:
        raise HTTPException(
            status_code = 400,
            detail = "Cannot replace and clear an API key in the same request",
        )

    metadata_fields = {
        "display_name",
        "base_url",
        "is_enabled",
        "models",
        "available_models",
        "max_output_tokens",
        "api_type",
        "reasoning_config",
        "shared",
    }
    metadata_requested = bool(payload.model_fields_set & metadata_fields)

    # Validate only a changed base URL, or unchanged legacy rows become uneditable.
    base_url = payload.base_url
    if base_url and base_url != existing["base_url"]:
        try:
            base_url = await asyncio.to_thread(validate_provider_base_url, base_url)
        except ValueError as exc:
            raise HTTPException(status_code = 400, detail = str(exc)) from None

    replacement_api_key = None
    if payload.encrypted_api_key:
        credential_secrets.get_or_create_credential_encryption_key()
        replacement_api_key = resolve_provider_api_key_or_400(
            provider_id, payload.encrypted_api_key
        )
        if not replacement_api_key:
            raise HTTPException(status_code = 400, detail = "API key cannot be empty")

    metadata_updates: dict = {}
    if metadata_requested:
        metadata_updates = dict(
            id = provider_id,
            display_name = payload.display_name,
            base_url = base_url,
            is_enabled = payload.is_enabled,
            models = payload.models,
            available_models = payload.available_models,
            api_type = payload.api_type,
            shared = payload.shared,
        )
        if max_output_tokens_requested:
            metadata_updates["max_output_tokens"] = payload.max_output_tokens
        if reasoning_config_requested:
            metadata_updates["reasoning_config"] = (
                payload.reasoning_config.model_dump()
                if payload.reasoning_config is not None
                else None
            )

    # The row snapshot this request found, keyed the way update_provider takes it.
    _restorable = dict(
        display_name = existing["display_name"],
        base_url = existing["base_url"],
        is_enabled = bool(existing["is_enabled"]),
        models = existing.get("models") or [],
        available_models = existing.get("available_models") or [],
        max_output_tokens = existing.get("max_output_tokens"),
        api_type = existing.get("api_type", "chat_completions"),
        reasoning_config = existing.get("reasoning_config"),
        shared = bool(existing.get("shared")),
    )

    def _current_matches(current: dict, field: str, written) -> bool:
        """Does the row still hold what this request wrote into that column?"""
        if field in ("is_enabled", "shared"):
            return bool(current.get(field)) == bool(written)
        return current.get(field) == written

    def _restore_metadata() -> None:
        """Restores only columns this request set and still holds, so a later save is not erased."""
        if not metadata_requested:
            return
        current = providers_db.get_provider(provider_id)
        if current is None:
            return
        undo = {}
        for field, written in metadata_updates.items():
            if field == "id":
                continue
            # Explicit null clears nullable overrides; elsewhere it means "not sent".
            if written is None and field not in {"max_output_tokens", "reasoning_config"}:
                continue
            if not _current_matches(current, field, written):
                continue
            undo[field] = _restorable[field]
        if not undo:
            return
        try:
            providers_db.update_provider(id = provider_id, **undo)
        except Exception:
            logger.exception("provider.update_metadata_rollback_failed", provider_id = provider_id)

    with current_credential_write(credential):
        credential_requested = replacement_api_key is not None or payload.clear_api_key
        if metadata_requested and credential_requested:
            # Commit metadata and key together so no process routes to the new endpoint with the old key.
            with providers_db.provider_bundle_transaction() as connection:
                providers_db.update_provider(**metadata_updates, connection = connection)
                if replacement_api_key is not None:
                    credential_secrets.save_provider_api_key(
                        provider_id,
                        replacement_api_key,
                        connection = connection,
                    )
                else:
                    credential_secrets.delete_provider_api_key(
                        provider_id,
                        connection = connection,
                    )
        else:
            if metadata_requested:
                providers_db.update_provider(**metadata_updates)
            if replacement_api_key is not None:
                credential_secrets.save_provider_api_key(provider_id, replacement_api_key)
            elif payload.clear_api_key:
                credential_secrets.delete_provider_api_key(provider_id)

    if not metadata_requested and not payload.encrypted_api_key and not payload.clear_api_key:
        raise HTTPException(status_code = 400, detail = "No fields to update")

    row = providers_db.get_provider(provider_id)
    if existing_info.get("auth_kind") == "chatgpt_oauth" and payload.models is not None:
        # Recorded after the row commit and as part of the save; a failure here undoes the row too.
        if validated_account:
            try:
                await openai_codex_auth.remember_catalog_account(provider_id, validated_account)
            except Exception:
                _restore_metadata()
                raise
    return _provider_response(row)


@router.put("/{provider_id}/api-key/migrate", response_model = ProviderResponse)
@serialize_provider_config
async def migrate_provider_api_key(
    provider_id: str,
    payload: ProviderCredentialMigration,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """Insert a browser legacy key only when this provider has no saved key."""
    require_ui_session(via_api_key)
    _own_provider_or_error(provider_id)
    api_key = resolve_provider_api_key_or_400(
        None, payload.encrypted_api_key, allow_saved_key = False
    )
    if not api_key:
        raise HTTPException(status_code = 400, detail = "API key cannot be empty")
    credential_secrets.get_or_create_credential_encryption_key()
    with current_credential_write(credential):
        credential_secrets.save_provider_api_key_if_absent(provider_id, api_key)
    return _provider_response(providers_db.get_provider(provider_id))


@router.delete("/{provider_id}", status_code = 204)
@serialize_provider_config
async def delete_provider_config(
    provider_id: str,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """Idempotently delete a saved provider and its installation credential."""
    if providers_db.get_own_provider(provider_id) is None:
        if providers_db.is_owner_shared_provider(provider_id):
            raise HTTPException(status_code = 403, detail = _SHARED_READ_ONLY)
        if account_access.managed_account():
            raise HTTPException(status_code = 404, detail = "Provider config not found")
    require_ui_session(via_api_key)
    await openai_codex_auth.cancel_provider_flows(provider_id)
    credential_secrets.get_or_create_credential_encryption_key()

    async with openai_codex_auth.provider_oauth_write_guard(provider_id):
        with current_credential_write(credential):
            existing_api_key = credential_secrets.get_provider_api_key(provider_id)
            existing_oauth = credential_secrets.get_secret(
                credential_secrets.OPENAI_CODEX_OAUTH_KIND, provider_id
            )

            existing_oauth_flow = credential_secrets.get_secret(
                credential_secrets.OPENAI_CODEX_OAUTH_FLOW_KIND, provider_id
            )
            credential_secrets.delete_provider_api_key(provider_id)
            credential_secrets.delete_secret(
                credential_secrets.OPENAI_CODEX_OAUTH_KIND, provider_id
            )
            credential_secrets.delete_secret(
                credential_secrets.OPENAI_CODEX_OAUTH_FLOW_KIND, provider_id
            )
            try:
                providers_db.delete_provider(provider_id)
            except Exception:
                try:
                    if existing_api_key:
                        credential_secrets.save_provider_api_key(provider_id, existing_api_key)
                    if existing_oauth:
                        credential_secrets.upsert_secret(
                            credential_secrets.OPENAI_CODEX_OAUTH_KIND,
                            provider_id,
                            existing_oauth,
                        )

                    if existing_oauth_flow:
                        credential_secrets.upsert_secret(
                            credential_secrets.OPENAI_CODEX_OAUTH_FLOW_KIND,
                            provider_id,
                            existing_oauth_flow,
                        )
                except Exception:
                    logger.exception(
                        "provider.delete_credential_rollback_failed", provider_id = provider_id
                    )
                raise
            # Release the per-process catalog only after the delete commits; a rollback still needs it.
            openai_codex_client.forget_subscription_models(provider_id)


def _is_shared_saved_target(payload) -> bool:
    return bool(
        payload.provider_id
        and not payload.encrypted_api_key
        and providers_db.is_owner_shared_provider(payload.provider_id)
    )


def _shared_target_models(payload) -> set[str] | None:
    if not _is_shared_saved_target(payload):
        return None
    return set((providers_db.get_provider(payload.provider_id) or {}).get("models") or [])


def _bind_saved_provider_target(payload):
    """Use the saved provider's endpoint whenever its saved credential may be used."""
    if not payload.provider_id or payload.encrypted_api_key:
        return payload
    config = providers_db.get_provider(payload.provider_id)
    if config is None:
        raise HTTPException(
            status_code = 404,
            detail = f"Provider config not found: {payload.provider_id}",
        )
    if not config["is_enabled"]:
        raise HTTPException(
            status_code = 400,
            detail = f"Provider '{config['display_name']}' is disabled.",
        )
    update = {
        "provider_type": config["provider_type"],
        "api_type": config.get("api_type", "chat_completions"),
    }
    # Pinning the saved endpoint keeps a saved credential off other hosts; a keyless row has none to protect, so an
    # edited base URL is used as sent.
    if (
        not payload.base_url
        or config["provider_type"] == "openai_codex"
        or credential_secrets.provider_api_key_saved(config["id"])
    ):
        update["base_url"] = config["base_url"]
    return payload.model_copy(update = update)


def _unlisted_model_message(model_id: str, models: list) -> str | None:
    ids = [m["id"] for m in models if isinstance(m, dict) and isinstance(m.get("id"), str)]
    if not ids or model_id in ids:
        return None
    message = f"Connected, but model '{model_id}' is not listed by this server."
    near = next((i for i in ids if i.lower() == model_id.lower()), None)
    if near:
        return f"{message} Did you mean '{near}'?"
    shown = ", ".join(ids[:3]) + (", ..." if len(ids) > 3 else "")
    return f"{message} Available: {shown}"


async def _test_custom_provider_connectivity(
    client,
    model_id: str,
    api_type: str = "chat_completions",
    base_url: str = "",
) -> ProviderTestResult:
    """Probe a custom OpenAI-compatible endpoint without assuming /chat/completions. TTS-only gateways such as
    Kokoro expose ``/models`` and ``/audio/speech`` but not ``/chat/completions``. Try those first, then fall
    back to a chat probe."""
    model_id = (model_id or "").strip()
    models_error: Exception | None = None
    models = None
    try:
        models = await client.list_models()
    except Exception as exc:
        models_error = exc

    if models is not None and api_type != "responses":
        return ProviderTestResult(
            success = True,
            message = (model_id and _unlisted_model_message(model_id, models))
            or f"Connected successfully. Found {len(models)} model(s).",
            models_count = len(models),
        )

    responses_model_ids: list[str] = []
    if model_id and api_type == "responses":
        responses_model_ids = [model_id]
    elif api_type == "responses" and models is not None:
        for model in models:
            candidate = model.get("id") if isinstance(model, dict) else None
            candidate = candidate.strip() if isinstance(candidate, str) else ""
            if candidate and candidate not in responses_model_ids:
                responses_model_ids.append(candidate)
            if len(responses_model_ids) >= _MAX_RESPONSES_CONNECTIVITY_MODELS:
                break
        if not responses_model_ids:
            return ProviderTestResult(
                success = False,
                message = (
                    "Connection failed: /models responded, but no model ID was available "
                    "to test the Responses endpoint."
                ),
                models_count = len(models),
            )
        model_id = responses_model_ids[0]

    if not model_id:
        if models is not None:
            return ProviderTestResult(
                success = True,
                message = f"Connected successfully. Found {len(models)} model(s).",
                models_count = len(models),
            )
        return ProviderTestResult(
            success = False,
            message = (
                "Connection failed: could not reach /models and no model ID was "
                f"provided to test further. {_provider_error_detail(models_error, base_url)}"
            ),
            models_count = None,
        )

    if api_type == "responses":
        last_failure = "Responses endpoint returned no completion."
        # Several unanswered models would otherwise keep the dialog waiting for over a minute.
        deadline = asyncio.get_running_loop().time() + _RESPONSES_PROBE_BUDGET_SECONDS
        for response_model_id in responses_model_ids:
            probe_timeout = min(
                _PROVIDER_CONNECTIVITY_TIMEOUT_SECONDS,
                deadline - asyncio.get_running_loop().time(),
            )
            if probe_timeout <= 0:
                break
            try:
                received_response = False
                model_failure: str | None = None
                response_stream = client.stream_chat_completion(
                    messages = [{"role": "user", "content": "ping"}],
                    model = response_model_id,
                    temperature = None,
                    top_p = None,
                    max_tokens = 16,
                )

                async def consume_response_stream():
                    nonlocal received_response, model_failure
                    async for line in response_stream:
                        if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                            event = json.loads(line[6:])
                            received_response = received_response or bool(event.get("choices"))
                            if event.get("error"):
                                model_failure = event["error"].get(
                                    "message", "Responses request failed"
                                )

                try:
                    await asyncio.wait_for(consume_response_stream(), timeout = probe_timeout)
                finally:
                    await response_stream.aclose()
                if received_response and model_failure is None:
                    return ProviderTestResult(
                        success = True,
                        message = "Connected successfully. Responses endpoint responded.",
                    )
                last_failure = model_failure or "Responses endpoint returned no completion."
            except asyncio.TimeoutError:
                last_failure = f"Responses endpoint timed out after {probe_timeout:.0f} seconds."
            except Exception as exc:
                last_failure = _provider_error_detail(exc, base_url)
        return ProviderTestResult(
            success = False,
            message = f"Connection failed: {last_failure}",
        )

    try:
        await client.create_speech(
            text = ".",
            model = model_id,
            voice = "alloy",
            response_format = "wav",
        )
        return ProviderTestResult(
            success = True,
            message = "Connected successfully. Audio speech endpoint responded.",
            models_count = None,
        )
    except Exception:
        pass

    try:
        await client.chat_completion(
            messages = [{"role": "user", "content": "ping"}],
            model = model_id,
            temperature = 0.0,
            top_p = None,
            max_tokens = 1,
        )
        return ProviderTestResult(
            success = True,
            message = "Connected successfully. Chat completions endpoint responded.",
            models_count = None,
        )
    except Exception as exc:
        return ProviderTestResult(
            success = False,
            message = f"Connection failed: {_provider_error_detail(exc, base_url)}",
            models_count = None,
        )


async def _test_decision_connectivity(client, model_id: str) -> ProviderTestResult:
    if not model_id:
        return ProviderTestResult(
            success = False, message = "Connection failed: add a model ID to test with."
        )
    try:
        result = await client.create_decision(
            model_id,
            "The ticket says the checkout page is down.",
            {"outage": {"type": "noul", "instructions": "Is something broken?"}},
        )
        answer = result["answers"]["outage"]
        speaks_system_one = answer["type"] == "noul" and isinstance(answer["noul"], (int, float))
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code not in (404, 405):
            return ProviderTestResult(
                success = False, message = f"Connection failed: {safe_curated_detail(exc)}"
            )
        speaks_system_one = False
    except httpx.HTTPError as exc:
        return ProviderTestResult(
            success = False, message = f"Connection failed: {safe_curated_detail(exc)}"
        )
    except (ValueError, TypeError, KeyError):
        speaks_system_one = False
    if not speaks_system_one:
        return ProviderTestResult(
            success = False,
            message = "Connection failed: this endpoint does not speak the System One API.",
        )
    return ProviderTestResult(
        success = True, message = "Connected successfully. It answered a decision."
    )


@router.post("/test", response_model = ProviderTestResult)
async def test_provider(
    payload: ProviderTestRequest,
    _current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """
    Test connectivity to an external provider.

    Makes a lightweight GET /models call to verify the API key works. Custom
    endpoints try /models first, then /audio/speech or /chat/completions when a
    model ID is available. An explicit encrypted key takes precedence over the
    saved provider key.
    """

    shared = _is_shared_saved_target(payload)
    payload = _bind_saved_provider_target(payload)
    info = get_connectable_provider_info(payload.provider_type)
    if info is None:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown provider type: {payload.provider_type}",
        )

    api_key = resolve_provider_api_key_or_400(
        payload.provider_id,
        payload.encrypted_api_key,
        allow_saved_key = not via_api_key,
    )

    base_url = payload.base_url or info["base_url"]
    if payload.provider_type == "custom":
        if not base_url:
            return ProviderTestResult(
                success = False,
                message = "Connection failed: Base URL is required for custom providers.",
                models_count = None,
            )
    try:
        base_url = await asyncio.to_thread(validate_provider_base_url, base_url)
    except ValueError as exc:
        return ProviderTestResult(
            success = False,
            message = f"Connection failed: {exc}",
            models_count = None,
        )

    client = await asyncio.to_thread(
        ExternalProviderClient,
        provider_type = payload.provider_type,
        base_url = base_url,
        api_key = api_key,
        api_type = payload.api_type,
        timeout = _PROVIDER_CONNECTIVITY_TIMEOUT_SECONDS,
    )

    if shared:
        # Others learn only reachability: no model probe, no upstream error text (it can quote the URL).
        try:
            if answers_decisions_only(payload.provider_type, payload.api_type):
                await client.list_decision_models()
            else:
                await client.list_models()
        except Exception:
            return ProviderTestResult(
                success = False,
                message = "Connection failed. Ask the owner to check this connection.",
                models_count = None,
            )
        finally:
            await client.close()
        return ProviderTestResult(
            success = True, message = "Connected successfully.", models_count = None
        )

    try:
        if answers_decisions_only(payload.provider_type, payload.api_type):
            return await _test_decision_connectivity(client, (payload.model_id or "").strip())
        if payload.provider_type == "custom":
            return await _test_custom_provider_connectivity(
                client, payload.model_id or "", payload.api_type, base_url
            )
        if info.get("model_list_mode") == "curated":
            await client.verify_models_endpoint_lightweight()
            if payload.provider_type in _KEY_CHECKED_PROVIDERS and await client.api_key_rejected():
                return ProviderTestResult(
                    success = False, message = f"Connection failed: {_API_KEY_REJECTED}"
                )
            return ProviderTestResult(
                success = True,
                message = (
                    "Connected successfully. Full model list is not fetched for this provider — "
                    "use suggestions and manual model IDs in the dialog."
                ),
                models_count = None,
            )
        models = await client.list_models()
        if payload.provider_type in _KEY_CHECKED_PROVIDERS and await client.api_key_rejected():
            return ProviderTestResult(
                success = False, message = f"Connection failed: {_API_KEY_REJECTED}"
            )
        return ProviderTestResult(
            success = True,
            message = f"Connected successfully. Found {len(models)} model(s).",
            models_count = len(models),
        )
    except Exception as exc:
        _log_provider_failure("providers.test_failed", exc, provider_type = payload.provider_type)
        return ProviderTestResult(
            success = False,
            message = (
                f"Connection failed: {_provider_error_detail(exc, base_url, listing_models = True)}"
            ),
            models_count = None,
        )
    finally:
        await client.close()


_MODEL_CAPABILITY_CACHE_TTL_SECONDS = 3600.0
_model_capability_cache: dict[str, tuple[float, list[dict]]] = {}


_MODEL_CATALOG_TTL_SECONDS = 24 * 3600.0
_model_catalog_cache: dict | None = None


def _model_catalog_cache_path():
    return cache_root() / "model_catalog.json"


def _read_model_catalog_file() -> dict | None:
    try:
        data = json.loads(_model_catalog_cache_path().read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    if (
        isinstance(data, dict)
        and isinstance(data.get("providers"), dict)
        and isinstance(data.get("fetched_at"), (int, float))
    ):
        return data
    return None


def _write_model_catalog_file(data: dict) -> None:
    try:
        path = _model_catalog_cache_path()
        path.parent.mkdir(parents = True, exist_ok = True)
        path.write_text(json.dumps(data), encoding = "utf-8")
    except OSError as exc:
        logger.warning("providers.model_catalog_cache_write_failed", error = str(exc))


def cached_catalog_model_inputs(namespace: str, model: str) -> list[str] | None:
    global _model_catalog_cache
    if _model_catalog_cache is None:
        _model_catalog_cache = _read_model_catalog_file()
    bucket = ((_model_catalog_cache or {}).get("providers") or {}).get(namespace) or {}
    entry = bucket.get(model.strip().lower())
    inputs = entry.get("input") if isinstance(entry, dict) else None
    return inputs if isinstance(inputs, list) else None


async def _fetch_models_dev_catalog() -> dict:
    from core.inference.external_provider import _client

    response = await _client().get(MODELS_DEV_URL, timeout = 20.0)
    response.raise_for_status()
    return {"fetched_at": time.time(), "providers": trim_models_dev_catalog(response.json())}


@router.get("/model-catalog", response_model = ModelCatalogResponse)
async def get_model_catalog(_current_subject: str = Depends(get_current_subject)):
    global _model_catalog_cache
    cached = _model_catalog_cache or _read_model_catalog_file()
    if cached is not None and time.time() - cached["fetched_at"] < _MODEL_CATALOG_TTL_SECONDS:
        _model_catalog_cache = cached
        return cached
    try:
        fresh = await _fetch_models_dev_catalog()
    except Exception as exc:
        logger.warning("providers.model_catalog_refresh_failed", error = str(exc))
        if cached is not None:
            _model_catalog_cache = cached
            return cached
        raise HTTPException(
            status_code = 503, detail = "The model catalog is unavailable offline."
        ) from None
    _model_catalog_cache = fresh
    _write_model_catalog_file(fresh)
    return fresh


@router.post("/model-capabilities", response_model = list[ProviderModelCapabilityInfo])
async def list_provider_model_capabilities(
    payload: ProviderModelsRequest,
    _current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    shared_models = _shared_target_models(payload)
    capabilities = await _list_model_capabilities(payload, via_api_key)
    if shared_models is not None:
        capabilities = [entry for entry in capabilities if entry.get("id", "") in shared_models]
    return capabilities


async def _list_model_capabilities(payload: ProviderModelsRequest, via_api_key: bool):
    payload = _bind_saved_provider_target(payload)
    info = get_connectable_provider_info(payload.provider_type)
    if info is None:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown provider type: {payload.provider_type}",
        )
    if payload.provider_type not in MODEL_CAPABILITY_PROVIDERS:
        return []

    api_key = resolve_provider_api_key_or_400(
        payload.provider_id,
        payload.encrypted_api_key,
        allow_saved_key = not via_api_key,
    )
    base_url = payload.base_url or info["base_url"]
    try:
        base_url = await asyncio.to_thread(validate_provider_base_url, base_url)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None

    # Per credential: one key's model list is not another caller's.
    key_digest = hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()[:16]
    cache_key = f"{payload.provider_type}\n{base_url}\n{key_digest}"
    cached = _model_capability_cache.get(cache_key)
    if cached is not None and time.monotonic() - cached[0] < _MODEL_CAPABILITY_CACHE_TTL_SECONDS:
        return cached[1]

    client = await asyncio.to_thread(
        ExternalProviderClient,
        provider_type = payload.provider_type,
        base_url = base_url,
        api_key = api_key,
        api_type = payload.api_type,
        timeout = 15.0,
    )
    try:
        models = await client.list_models()
    except Exception as exc:
        raise log_and_http_error(
            exc,
            502,
            f"Failed to list model capabilities from {payload.provider_type}.",
            event = "providers.list_model_capabilities_failed",
            log = logger,
        )
    finally:
        await client.close()
    capabilities = provider_model_capabilities(payload.provider_type, models)
    if capabilities:
        now = time.monotonic()
        for key in [
            key
            for key, (stored_at, _) in _model_capability_cache.items()
            if now - stored_at >= _MODEL_CAPABILITY_CACHE_TTL_SECONDS
        ]:
            del _model_capability_cache[key]
        _model_capability_cache[cache_key] = (now, capabilities)
    return capabilities


def _model_capability_names(model: dict) -> Optional[list[str]]:
    # Another server's shape under this key must not fail the listing's validation.
    values = model.get("capabilities")
    if not isinstance(values, list):
        return None
    return [name for name in values if isinstance(name, str) and name]


@router.post("/models", response_model = list[ProviderModelInfo])
async def list_provider_models(
    payload: ProviderModelsRequest,
    _current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """
    List models available from an external provider.

    An explicit encrypted key takes precedence over the saved provider key.
    """

    shared_models = _shared_target_models(payload)
    payload = _bind_saved_provider_target(payload)
    info = get_connectable_provider_info(payload.provider_type)
    if info is None:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown provider type: {payload.provider_type}",
        )

    api_key = resolve_provider_api_key_or_400(
        payload.provider_id,
        payload.encrypted_api_key,
        allow_saved_key = not via_api_key,
    )

    if info.get("model_list_mode") == "curated":
        return [
            ProviderModelInfo(
                id = m,
                display_name = m,
                context_length = None,
                owned_by = None,
            )
            for m in info.get("default_models", [])
            if shared_models is None or m in shared_models
        ]

    base_url = payload.base_url or info["base_url"]
    try:
        base_url = await asyncio.to_thread(validate_provider_base_url, base_url)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None

    client = await asyncio.to_thread(
        ExternalProviderClient,
        provider_type = payload.provider_type,
        base_url = base_url,
        api_key = api_key,
        api_type = payload.api_type,
        timeout = 15.0,
    )

    if answers_decisions_only(payload.provider_type, payload.api_type):
        try:
            ids = await client.list_decision_models()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                raise HTTPException(
                    status_code = 502, detail = f"The server refused the API key (HTTP {status})."
                ) from None
            if status not in (404, 405):
                raise HTTPException(
                    status_code = 502,
                    detail = f"The server answered HTTP {status} when asked for its models.",
                ) from None
            ids = []
        except httpx.HTTPError:
            raise HTTPException(
                status_code = 502, detail = "Couldn't reach the server. Check the base URL."
            ) from None
        except ValueError:
            ids = []
        return [
            ProviderModelInfo(id = m, display_name = m, context_length = None, owned_by = None)
            for m in ids
            if shared_models is None or m in shared_models
        ]

    try:
        models = await client.list_models()
        # Model-id filters only apply on the vendor's own host; proxies and Azure use other names.
        _NATIVE_HOSTS = {
            "gemini": ("generativelanguage.googleapis.com",),
            "openai": ("api.openai.com",),
        }
        apply_registry_model_filters = True
        native_hosts = _NATIVE_HOSTS.get(payload.provider_type)
        if native_hosts is not None:
            try:
                from urllib.parse import urlparse as _urlparse
                _host = (_urlparse(base_url).hostname or "").lower()
            except Exception:
                _host = ""
            apply_registry_model_filters = _host in native_hosts

        if apply_registry_model_filters:
            allow_prefixes = info.get("model_id_allow_prefixes")
            if allow_prefixes is not None:
                prefix_tuple = tuple(str(p) for p in allow_prefixes if str(p))
                if prefix_tuple:
                    models = [m for m in models if m.get("id", "").startswith(prefix_tuple)]
            allowlist = info.get("model_id_allowlist")
            if allowlist is not None:
                models = [m for m in models if allowlist.match(m.get("id", ""))]
            deny_exact = info.get("model_id_deny_exact")
            if deny_exact is not None:
                deny_ids = {str(m) for m in deny_exact if str(m)}
                if deny_ids:
                    models = [m for m in models if m.get("id", "") not in deny_ids]
            denylist = info.get("model_id_denylist")
            if denylist is not None:
                models = [m for m in models if not denylist.search(m.get("id", ""))]
        # Optional cap after filtering to keep large catalogs picker-sized. Unsorted, so "first N matches"; pair with
        # default_models for flagships.
        if shared_models is not None:
            models = [m for m in models if m.get("id", "") in shared_models]
        limit = info.get("model_id_limit")
        if isinstance(limit, int) and limit > 0:
            models = models[:limit]
        return [
            ProviderModelInfo(
                id = m.get("id", ""),
                display_name = m.get("id", ""),
                context_length = m.get("context_length") or m.get("context_window"),
                owned_by = m.get("owned_by"),
                capabilities = _model_capability_names(m),
            )
            for m in models
        ]
    except Exception as exc:
        _log_provider_failure(
            "providers.list_models_failed", exc, provider_type = payload.provider_type
        )
        detail = _known_provider_error(exc, base_url, listing_models = True)
        raise HTTPException(
            status_code = 502,
            detail = detail or f"Failed to list models from {payload.provider_type}.",
        ) from None
    finally:
        await client.close()
