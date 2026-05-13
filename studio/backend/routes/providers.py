# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
API routes for external LLM provider management.

Provides endpoints for:
  - Discovering available provider types (registry)
  - CRUD for saved provider configurations (no API keys stored)
  - Fetching the RSA public key for API key encryption
  - Testing provider connectivity
  - Listing models from a provider
"""

import uuid
import structlog
from fastapi import APIRouter, Depends, HTTPException

from auth.authentication import get_current_subject
from core.inference.key_exchange import (
    decrypt_api_key,
    get_public_key_fingerprint,
    get_public_key_pem,
)
from core.inference.providers import (
    get_base_url,
    get_provider_info,
    list_available_providers,
)
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
from storage import providers_db

logger = structlog.get_logger(__name__)

router = APIRouter()


# ── Public key for API key encryption ─────────────────────────────


@router.get("/public-key")
async def get_public_key(
    current_subject: str = Depends(get_current_subject),
):
    """Return the RSA public key PEM for client-side API key encryption.

    The ``fingerprint`` field is a short SHA256 of the PEM and is meant
    purely for diagnostics — a mismatch between what the frontend
    captured at encrypt time and what the server reports here is a
    clear signal that the keypair rotated mid-flight (e.g. the server
    re-ran ``init_key_pair`` for any reason).
    """
    return {
        "public_key": get_public_key_pem(),
        "fingerprint": get_public_key_fingerprint(),
    }


# ── Provider registry (static) ───────────────────────────────────


@router.get("/registry", response_model = list[ProviderRegistryEntry])
async def list_registry(
    current_subject: str = Depends(get_current_subject),
):
    """List all supported provider types with their default configurations."""
    return list_available_providers()


# ── Provider config CRUD ──────────────────────────────────────────


@router.get("/", response_model = list[ProviderResponse])
async def list_provider_configs(
    current_subject: str = Depends(get_current_subject),
):
    """List all saved provider configurations."""
    rows = providers_db.list_providers()
    return [
        ProviderResponse(
            id = row["id"],
            provider_type = row["provider_type"],
            display_name = row["display_name"],
            base_url = row["base_url"],
            is_enabled = bool(row["is_enabled"]),
            created_at = row["created_at"],
            updated_at = row["updated_at"],
        )
        for row in rows
    ]


@router.post("/", response_model = ProviderResponse, status_code = 201)
async def create_provider_config(
    payload: ProviderCreate,
    current_subject: str = Depends(get_current_subject),
):
    """Create a new saved provider configuration (no API key stored)."""
    info = get_provider_info(payload.provider_type)
    if info is None:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown provider type: {payload.provider_type}. "
            f"Use GET /api/providers/registry to see available types.",
        )

    provider_id = uuid.uuid4().hex[:16]
    base_url = payload.base_url or info["base_url"]

    providers_db.create_provider(
        id = provider_id,
        provider_type = payload.provider_type,
        display_name = payload.display_name,
        base_url = base_url,
    )

    row = providers_db.get_provider(provider_id)
    return ProviderResponse(
        id = row["id"],
        provider_type = row["provider_type"],
        display_name = row["display_name"],
        base_url = row["base_url"],
        is_enabled = bool(row["is_enabled"]),
        created_at = row["created_at"],
        updated_at = row["updated_at"],
    )


@router.put("/{provider_id}", response_model = ProviderResponse)
async def update_provider_config(
    provider_id: str,
    payload: ProviderUpdate,
    current_subject: str = Depends(get_current_subject),
):
    """Update a saved provider configuration."""
    existing = providers_db.get_provider(provider_id)
    if not existing:
        raise HTTPException(status_code = 404, detail = "Provider not found")

    updated = providers_db.update_provider(
        id = provider_id,
        display_name = payload.display_name,
        base_url = payload.base_url,
        is_enabled = payload.is_enabled,
    )
    if not updated:
        raise HTTPException(status_code = 400, detail = "No fields to update")

    row = providers_db.get_provider(provider_id)
    return ProviderResponse(
        id = row["id"],
        provider_type = row["provider_type"],
        display_name = row["display_name"],
        base_url = row["base_url"],
        is_enabled = bool(row["is_enabled"]),
        created_at = row["created_at"],
        updated_at = row["updated_at"],
    )


@router.delete("/{provider_id}", status_code = 204)
async def delete_provider_config(
    provider_id: str,
    current_subject: str = Depends(get_current_subject),
):
    """Delete a saved provider configuration."""
    deleted = providers_db.delete_provider(provider_id)
    if not deleted:
        raise HTTPException(status_code = 404, detail = "Provider not found")


# ── Test connectivity ─────────────────────────────────────────────


@router.post("/test", response_model = ProviderTestResult)
async def test_provider(
    payload: ProviderTestRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Test connectivity to an external provider.

    Makes a lightweight GET /models call to verify the API key works.
    The encrypted_api_key is decrypted server-side and never stored.
    """
    info = get_provider_info(payload.provider_type)
    if info is None:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown provider type: {payload.provider_type}",
        )

    try:
        api_key = decrypt_api_key(payload.encrypted_api_key)
    except Exception as exc:
        logger.warning("Failed to decrypt API key (%s): %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code = 400,
            detail = "Failed to decrypt API key. The public key may have changed — try refreshing the page.",
        )

    base_url = payload.base_url or info["base_url"]
    client = ExternalProviderClient(
        provider_type = payload.provider_type,
        base_url = base_url,
        api_key = api_key,
        timeout = 15.0,
    )

    try:
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
        logger.warning("Provider test failed for %s: %s", payload.provider_type, exc)
        return ProviderTestResult(
            success = False,
            message = f"Connection failed: {exc}",
            models_count = None,
        )
    finally:
        await client.close()


# ── List models from provider ─────────────────────────────────────


@router.post("/models", response_model = list[ProviderModelInfo])
async def list_provider_models(
    payload: ProviderModelsRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    List models available from an external provider.

    The encrypted_api_key is decrypted server-side and never stored.
    """
    info = get_provider_info(payload.provider_type)
    if info is None:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown provider type: {payload.provider_type}",
        )

    try:
        api_key = decrypt_api_key(payload.encrypted_api_key)
    except Exception as exc:
        logger.warning("Failed to decrypt API key (%s): %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code = 400,
            detail = "Failed to decrypt API key. The public key may have changed — try refreshing the page.",
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
        allowlist = info.get("model_id_allowlist")
        if allowlist is not None:
            models = [m for m in models if allowlist.match(m.get("id", ""))]
        denylist = info.get("model_id_denylist")
        if denylist is not None:
            models = [m for m in models if not denylist.search(m.get("id", ""))]
        # Apply an optional cap after filtering so registry entries with a
        # large remote catalog (e.g. HF Inference Providers) can stay
        # picker-sized. No popularity sort happens server-side, so this is
        # "first N matches" — pair with default_models for any must-have
        # flagship ids.
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
        logger.error("Failed to list models from %s: %s", payload.provider_type, exc)
        raise HTTPException(
            status_code = 502,
            detail = f"Failed to list models from {payload.provider_type}: {exc}",
        )
    finally:
        await client.close()
