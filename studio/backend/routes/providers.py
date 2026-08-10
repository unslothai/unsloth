# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
API routes for external LLM provider management.

Endpoints:
  - Discover available provider types (registry)
  - CRUD for saved provider configurations and owner-scoped API keys
  - Fetch the RSA public key for API key encryption
  - Test provider connectivity
  - List models from a provider
"""

import uuid
import structlog
from fastapi import APIRouter, Depends, HTTPException

from auth.authentication import get_current_credential, get_current_subject

from routes.provider_credentials import (
    current_credential_write,
    resolve_provider_api_key_or_400,
)
from core.inference.key_exchange import (
    get_public_key_fingerprint,
    get_public_key_pem,
)
from core.inference.providers import (
    get_base_url,
    get_provider_info,
    list_available_providers,
)
from core.inference.pricing import pricing_snapshot
from core.inference.external_provider import ExternalProviderClient
from models.providers import (
    ProviderCreate,
    ProviderModelsRequest,
    ProviderModelInfo,
    ProviderResponse,
    ProviderRegistryEntry,
    ProviderTestRequest,
    ProviderTestResult,
    ProviderUpdate,
)
from storage import credential_secrets, providers_db
from utils.utils import safe_curated_detail, log_and_http_error

logger = structlog.get_logger(__name__)

router = APIRouter()


def _provider_response(row: dict, current_subject: str) -> ProviderResponse:
    return ProviderResponse(
        id = row["id"],
        provider_type = row["provider_type"],
        display_name = row["display_name"],
        base_url = row["base_url"],
        is_enabled = bool(row["is_enabled"]),
        has_api_key = credential_secrets.has_secret(
            current_subject,
            credential_secrets.PROVIDER_API_KEY_KIND,
            row["id"],
        ),
        models = row.get("models") or [],
        available_models = row.get("available_models") or [],
        created_at = row["created_at"],
        updated_at = row["updated_at"],
    )


# ── Public key for API key encryption ─────────────────────────────


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


# ── Provider registry (static) ───────────────────────────────────


@router.get("/registry", response_model = list[ProviderRegistryEntry])
async def list_registry(current_subject: str = Depends(get_current_subject)):
    """List all supported provider types with their default configurations."""
    return list_available_providers()


# ── Per-MTok pricing snapshot for client-side cost display ──────────


@router.get("/pricing")
async def get_pricing_snapshot(current_subject: str = Depends(get_current_subject)):
    """Static per-MTok pricing table the frontend uses to convert upstream
    usage into per-turn USD cost. See ``core/inference/pricing.py`` for sourcing."""
    return pricing_snapshot()


# ── Provider config CRUD ──────────────────────────────────────────


@router.get("/", response_model = list[ProviderResponse])
async def list_provider_configs(current_subject: str = Depends(get_current_subject)):
    """List all saved provider configurations."""
    rows = providers_db.list_providers()
    return [_provider_response(row, current_subject) for row in rows]


@router.post("/", response_model = ProviderResponse, status_code = 201)
async def create_provider_config(
    payload: ProviderCreate, credential: tuple = Depends(get_current_credential)
):
    current_subject = credential[0]
    """Create a saved provider configuration and optional encrypted API key."""
    info = get_provider_info(payload.provider_type)
    if info is None:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown provider type: {payload.provider_type}. "
            f"Use GET /api/providers/registry to see available types.",
        )

    api_key = resolve_provider_api_key_or_400(current_subject, None, payload.encrypted_api_key)
    provider_id = uuid.uuid4().hex[:16]
    base_url = payload.base_url or info["base_url"]

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
        )
        try:
            if api_key:
                credential_secrets.save_provider_api_key(current_subject, provider_id, api_key)
        except Exception:
            providers_db.delete_provider(provider_id)
            raise

    row = providers_db.get_provider(provider_id)
    return _provider_response(row, current_subject)


@router.put("/{provider_id}", response_model = ProviderResponse)
async def update_provider_config(
    provider_id: str,
    payload: ProviderUpdate,
    credential: tuple = Depends(get_current_credential),
):
    current_subject = credential[0]
    """Update a saved provider configuration."""
    existing = providers_db.get_provider(provider_id)
    if not existing:
        raise HTTPException(status_code = 404, detail = "Provider not found")

    if payload.clear_api_key and payload.encrypted_api_key:
        raise HTTPException(
            status_code = 400,
            detail = "Cannot replace and clear an API key in the same request",
        )

    metadata_fields = {"display_name", "base_url", "is_enabled", "models", "available_models"}
    metadata_requested = bool(payload.model_fields_set & metadata_fields)
    if payload.encrypted_api_key:
        credential_secrets.get_or_create_credential_encryption_key()

    with current_credential_write(credential):
        if metadata_requested:
            providers_db.update_provider(
                id = provider_id,
                display_name = payload.display_name,
                base_url = payload.base_url,
                is_enabled = payload.is_enabled,
                models = payload.models,
                available_models = payload.available_models,
            )
        if payload.encrypted_api_key:
            api_key = resolve_provider_api_key_or_400(
                current_subject, provider_id, payload.encrypted_api_key
            )
            if not api_key:
                raise HTTPException(status_code = 400, detail = "API key cannot be empty")
            credential_secrets.save_provider_api_key(current_subject, provider_id, api_key)
        elif payload.clear_api_key:
            credential_secrets.delete_provider_api_key(current_subject, provider_id)

    if not metadata_requested and not payload.encrypted_api_key and not payload.clear_api_key:
        raise HTTPException(status_code = 400, detail = "No fields to update")

    row = providers_db.get_provider(provider_id)
    return _provider_response(row, current_subject)


@router.delete("/{provider_id}", status_code = 204)
async def delete_provider_config(
    provider_id: str, credential: tuple = Depends(get_current_credential)
):
    current_subject = credential[0]
    """Idempotently delete a saved provider and this caller's scoped key."""
    with current_credential_write(credential):
        credential_secrets.delete_provider_api_key(current_subject, provider_id)
        providers_db.delete_provider(provider_id)


def _bind_saved_provider_target(payload):
    """Use the saved provider's endpoint whenever its saved credential may be used."""
    if not payload.provider_id:
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
    return payload.model_copy(
        update = {
            "provider_type": config["provider_type"],
            "base_url": config["base_url"],
        }
    )


# ── Test connectivity ─────────────────────────────────────────────


@router.post("/test", response_model = ProviderTestResult)
async def test_provider(
    payload: ProviderTestRequest, current_subject: str = Depends(get_current_subject)
):
    """
    Test connectivity to an external provider.

    Makes a lightweight GET /models call to verify the API key works. Generic
    custom endpoints use a chat-completions probe because /models is optional.
    An explicit encrypted key takes precedence over the saved provider key.
    """

    payload = _bind_saved_provider_target(payload)
    info = get_provider_info(payload.provider_type)
    if info is None:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown provider type: {payload.provider_type}",
        )

    api_key = resolve_provider_api_key_or_400(
        current_subject, payload.provider_id, payload.encrypted_api_key
    )

    base_url = payload.base_url or info["base_url"]
    if payload.provider_type == "custom":
        if not base_url:
            return ProviderTestResult(
                success = False,
                message = "Connection failed: Base URL is required for custom providers.",
                models_count = None,
            )

    client = ExternalProviderClient(
        provider_type = payload.provider_type,
        base_url = base_url,
        api_key = api_key,
        timeout = 15.0,
    )

    try:
        if payload.provider_type == "custom":
            model_id = (payload.model_id or "").strip()
            if not model_id:
                return ProviderTestResult(
                    success = False,
                    message = "Connection failed: add a model ID to test custom providers.",
                    models_count = None,
                )
            await client.chat_completion(
                messages = [{"role": "user", "content": "ping"}],
                model = model_id,
                temperature = 0.0,
                top_p = 1.0,
                max_tokens = 1,
            )
            return ProviderTestResult(
                success = True,
                message = "Connected successfully. Chat completions endpoint responded.",
                models_count = None,
            )
        if info.get("model_list_mode") == "curated":
            await client.verify_models_endpoint_lightweight()
            return ProviderTestResult(
                success = True,
                message = (
                    "Connected successfully. Full model list is not fetched for this provider — "
                    "use suggestions and manual model IDs in the dialog."
                ),
                models_count = None,
            )
        models = await client.list_models()
        return ProviderTestResult(
            success = True,
            message = f"Connected successfully. Found {len(models)} model(s).",
            models_count = len(models),
        )
    except Exception as exc:
        logger.error(
            "providers.test_failed",
            provider_type = payload.provider_type,
            error = str(exc),
            exc_info = True,
        )
        return ProviderTestResult(
            success = False,
            message = f"Connection failed: {safe_curated_detail(exc)}",
            models_count = None,
        )
    finally:
        await client.close()


# ── List models from provider ─────────────────────────────────────


@router.post("/models", response_model = list[ProviderModelInfo])
async def list_provider_models(
    payload: ProviderModelsRequest, current_subject: str = Depends(get_current_subject)
):
    """
    List models available from an external provider.

    An explicit encrypted key takes precedence over the saved provider key.
    """

    payload = _bind_saved_provider_target(payload)
    info = get_provider_info(payload.provider_type)
    if info is None:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown provider type: {payload.provider_type}",
        )

    api_key = resolve_provider_api_key_or_400(
        current_subject, payload.provider_id, payload.encrypted_api_key
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
        ]

    base_url = payload.base_url or info["base_url"]
    client = ExternalProviderClient(
        provider_type = payload.provider_type,
        base_url = base_url,
        api_key = api_key,
        timeout = 15.0,
    )

    try:
        models = await client.list_models()
        # Registry model-id filters only apply to the native Gemini base. A
        # custom OAI-compatible proxy returns prefixed IDs the native allowlist
        # would strip, leaving the picker empty; match the host check here so the
        # model list and chat dispatch agree on what counts as "native".
        apply_registry_model_filters = True
        if payload.provider_type == "gemini":
            try:
                from urllib.parse import urlparse as _urlparse
                _host = (_urlparse(base_url).hostname or "").lower()
            except Exception:
                _host = ""
            apply_registry_model_filters = _host == "generativelanguage.googleapis.com"

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
        # Optional cap after filtering to keep large catalogs picker-sized.
        # Unsorted, so "first N matches"; pair with default_models for flagships.
        limit = info.get("model_id_limit")
        if isinstance(limit, int) and limit > 0:
            models = models[:limit]
        return [
            ProviderModelInfo(
                id = m.get("id", ""),
                display_name = m.get("id", ""),
                context_length = m.get("context_length") or m.get("context_window"),
                owned_by = m.get("owned_by"),
            )
            for m in models
        ]
    except Exception as exc:
        raise log_and_http_error(
            exc,
            502,
            f"Failed to list models from {payload.provider_type}.",
            event = "providers.list_models_failed",
            log = logger,
        )
    finally:
        await client.close()
