# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
API routes for external LLM provider management.

Endpoints:
  - Discover available provider types (registry)
  - CRUD for saved provider configurations and API keys
  - Fetch the RSA public key for API key encryption
  - Test provider connectivity
  - List models from a provider
"""

import uuid
from typing import Optional

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
    get_base_url,
    get_provider_info,
    list_available_providers,
    validate_provider_base_url,
)
from core.inference.pricing import pricing_snapshot
from core.inference.external_provider import ExternalProviderClient

from core.inference import openai_codex_auth, openai_codex_client
from models.providers import (
    ProviderCreate,
    ProviderCredentialMigration,
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


def _provider_response(row: dict) -> ProviderResponse:
    return ProviderResponse(
        id = row["id"],
        provider_type = row["provider_type"],
        display_name = row["display_name"],
        base_url = row["base_url"],
        is_enabled = bool(row["is_enabled"]),
        has_api_key = credential_secrets.has_secret(
            credential_secrets.PROVIDER_API_KEY_KIND,
            row["id"],
        ),
        auth_kind = ("chatgpt_oauth" if row["provider_type"] == "openai_codex" else "api_key"),
        auth_status = (
            openai_codex_auth.auth_status(row["id"])
            if row["provider_type"] == "openai_codex"
            else (
                "connected"
                if credential_secrets.has_secret(
                    credential_secrets.PROVIDER_API_KEY_KIND, row["id"]
                )
                else "disconnected"
            )
        ),
        models = row.get("models") or [],
        available_models = row.get("available_models") or [],
        max_output_tokens = row.get("max_output_tokens"),
        created_at = row["created_at"],
        updated_at = row["updated_at"],
    )


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
    # Same order of evidence the chat route uses, so a save cannot persist a model that
    # every send would then refuse: the plan's catalog once it is known, otherwise the
    # seed, plus what this row already carries unless it was left by another account.
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
            # Already accepted on this row once, so an upstream outage must not make an
            # unrelated edit such as a rename unsavable.
            allowed |= set(persisted_models)
    if not models or not set(models).issubset(allowed):
        raise HTTPException(status_code = 400, detail = "Choose only curated Codex models.")


def _validate_max_output_tokens_contract(
    provider_type: str,
    field_was_set: bool,
    value: Optional[int] = None,
) -> None:
    """Reject a non-null override on a ChatGPT subscription.

    Codex routing, model list and output cap are all fixed, so an override stored there
    would never be read. Every other type takes one: the frontend uses it to lower a
    model's documented cap, or to replace the 32,768-token fallback for a model with no
    documented cap.

    An explicit null is allowed everywhere, Codex included: a blank field serialises as
    null rather than as an omission, and clearing an absent override is a no-op.
    """
    if field_was_set and value is not None and provider_type == "openai_codex":
        raise HTTPException(
            status_code = 400,
            detail = "ChatGPT subscriptions use a fixed Max Tokens limit.",
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
async def list_registry(
    include_hidden: bool = False, current_subject: str = Depends(get_current_subject)
):
    """List all supported provider types with their default configurations.

    ``include_hidden=true`` also returns the backend-only entries (the
    self-hosted presets), which carry the studio-tools capability the composer
    needs. It is opt-in so that a browser still running a pre-capability bundle,
    which does not know to filter on ``hidden``, keeps seeing exactly the list
    it saw before and cannot render them as duplicate dropdown options.
    """
    return list_available_providers(include_hidden = include_hidden)


# ── Per-MTok pricing snapshot for client-side cost display ──────────


@router.get("/pricing")
async def get_pricing_snapshot(current_subject: str = Depends(get_current_subject)):
    """Static per-MTok pricing table the frontend uses to convert upstream
    usage into per-turn USD cost. See ``core/inference/pricing.py`` for sourcing."""
    return pricing_snapshot()


# ── Provider config CRUD ──────────────────────────────────────────


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
    info = get_provider_info(payload.provider_type)
    if info is None:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown provider type: {payload.provider_type}. "
            f"Use GET /api/providers/registry to see available types.",
        )

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
    # An empty base URL stays allowed (custom/vLLM entries carry none until the
    # user fills one in); anything present is checked before a key is decrypted.
    if base_url:
        try:
            base_url = validate_provider_base_url(base_url)
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
    existing = providers_db.get_provider(provider_id)
    if not existing:
        raise HTTPException(status_code = 404, detail = "Provider not found")

    existing_info = get_provider_info(existing["provider_type"]) or {}
    max_output_tokens_requested = "max_output_tokens" in payload.model_fields_set
    _validate_max_output_tokens_contract(
        existing["provider_type"],
        max_output_tokens_requested,
        payload.max_output_tokens,
    )
    persisted_models = list(existing.get("models") or [])
    # One reading of who owns this connection, used for every decision in this request:
    # a second lookup later could name a different account than the one the selection was
    # actually judged against.
    validated_account: str | None = None
    if existing_info.get("auth_kind") == "chatgpt_oauth":
        # The OAuth bundle is shared through the installation DB while the catalog is per
        # process, so another worker may have rebound this connection. The chat route
        # makes the same check; without it here a save would persist exactly what every
        # send then refuses.
        current_bundle = openai_codex_auth.load_oauth_bundle(provider_id)
        validated_account = current_bundle.get("account_id") if current_bundle else None
        current_account = validated_account
        if current_account and not openai_codex_client.subscription_catalog_matches_account(
            provider_id, current_account
        ):
            openai_codex_client.forget_subscription_models(provider_id)
            openai_codex_client.mark_subscription_catalog_stale(provider_id)
    if existing_info.get("auth_kind") == "chatgpt_oauth" and payload.models:
        # Only a slug that is neither seeded nor already saved here needs the plan
        # catalog. Reaching upstream for the others would make an ordinary save wait out
        # the 20s connect / 120s read timeout whenever ChatGPT is unreachable, and would
        # fail an unrelated edit to a connection whose selection was accepted long ago.
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
    }
    metadata_requested = bool(payload.model_fields_set & metadata_fields)

    # Only a *changed* base URL is validated. The dialog re-sends the stored value
    # on every edit, so validating an unchanged legacy row would lock the user out
    # of editing its models or API key. Outbound use is still checked.
    base_url = payload.base_url
    if base_url and base_url != existing["base_url"]:
        try:
            base_url = validate_provider_base_url(base_url)
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
        )
        if max_output_tokens_requested:
            metadata_updates["max_output_tokens"] = payload.max_output_tokens

    # The row snapshot this request found, keyed the way update_provider takes it.
    _restorable = dict(
        display_name = existing["display_name"],
        base_url = existing["base_url"],
        is_enabled = bool(existing["is_enabled"]),
        models = existing.get("models") or [],
        available_models = existing.get("available_models") or [],
        max_output_tokens = existing.get("max_output_tokens"),
    )

    def _current_matches(current: dict, field: str, written) -> bool:
        """Does the row still hold what this request wrote into that column?"""
        if field == "is_enabled":
            return bool(current.get("is_enabled")) == bool(written)
        return current.get(field) == written

    def _restore_metadata() -> None:
        """Undo this request's own metadata write, while it is still the row.

        update_provider commits and closes its own connection, so the row is already
        durable by the time any later step fails; a compensating write is the only undo
        there is. This handler suspends between that commit and the proof write, though
        (remember_catalog_account awaits a 30s file lock, and the failure worth undoing is
        exactly the one where that lock was contended), so a second save can land in
        between. Restoring the whole pre-request snapshot would silently erase it. Put
        back only the columns this request set, and only those the row still holds this
        request's value for: a column a later save has since claimed belongs to that save.
        """
        if not metadata_requested:
            return
        current = providers_db.get_provider(provider_id)
        if current is None:
            return
        undo = {}
        for field, written in metadata_updates.items():
            if field == "id":
                continue
            # None means "not sent" for every column but max_output_tokens, which is only
            # present here when it was explicitly requested. update_provider left the
            # unsent ones alone, so there is nothing of this request's to take back.
            if written is None and field != "max_output_tokens":
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
            # Metadata and the saved key share studio.db.  Commit them together so
            # another process can never route to the new endpoint with the old key.
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
        # Record the proof here rather than when a catalog is read: reading one only
        # shows which account answered, while this is the point where the row's models
        # were actually judged against it and stored.
        # The account the selection was judged against, not whatever owns the connection
        # by now. remember_catalog_account re-reads under the guard and declines to write
        # when the bundle has moved on, so a rebind in between records nothing.
        # Written after the row, never before: a proof recorded ahead of a commit that
        # then failed would license models this connection never saved. Recording it is
        # part of the save, so a failure here undoes the row too. Leaving the new models
        # behind without the proof is the state saved_models_proven_for exists to catch,
        # and it outlives the process: after a restart, with the plan catalog unreadable,
        # the row's own slugs stop authorizing chat and the next save is refused as well.
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
    if providers_db.get_provider(provider_id) is None:
        raise HTTPException(status_code = 404, detail = "Provider not found")
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
            # The plan catalog is held per connection in this process and is only
            # released by forget_subscription_models. Disconnecting the OAuth bundle
            # calls it, deleting the whole connection did not, so every ChatGPT
            # connection a user removed left its catalog, its account marker and its
            # request ticket behind for the life of the process. Ids come from uuid4 and
            # are never reused, so nothing stale could be consulted again; it simply
            # accumulated. Released here, after the row is gone for good, so a rolled
            # back delete keeps the catalog it is about to need again.
            openai_codex_client.forget_subscription_models(provider_id)


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
    return payload.model_copy(
        update = {
            "provider_type": config["provider_type"],
            "base_url": config["base_url"],
        }
    )


# ── Test connectivity ─────────────────────────────────────────────


async def _test_custom_provider_connectivity(client, model_id: str) -> ProviderTestResult:
    """Probe a custom OpenAI-compatible endpoint without assuming /chat/completions.

      TTS-only gateways such as Kokoro expose ``/models`` and ``/audio/speech`` but
    not ``/chat/completions``. Try those first, then fall back to a chat probe."""
    model_id = (model_id or "").strip()
    models_error: Exception | None = None
    try:
        models = await client.list_models()
        return ProviderTestResult(
            success = True,
            message = f"Connected successfully. Found {len(models)} model(s).",
            models_count = len(models),
        )
    except Exception as exc:
        models_error = exc

    if not model_id:
        return ProviderTestResult(
            success = False,
            message = (
                "Connection failed: could not reach /models and no model ID was "
                f"provided to test further. {safe_curated_detail(models_error)}"
            ),
            models_count = None,
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
            top_p = 1.0,
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
            message = f"Connection failed: {safe_curated_detail(exc)}",
            models_count = None,
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

    payload = _bind_saved_provider_target(payload)
    info = get_provider_info(payload.provider_type)
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
        base_url = validate_provider_base_url(base_url)
    except ValueError as exc:
        return ProviderTestResult(
            success = False,
            message = f"Connection failed: {exc}",
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
            return await _test_custom_provider_connectivity(client, payload.model_id or "")
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
    payload: ProviderModelsRequest,
    _current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
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
        ]

    base_url = payload.base_url or info["base_url"]
    try:
        base_url = validate_provider_base_url(base_url)
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from None

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
