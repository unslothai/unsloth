# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""UI-session-only lifecycle routes for ChatGPT/Codex subscription OAuth."""

import secrets
from typing import Literal

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.authentication import authenticated_via_api_key, get_current_credential
from core.inference import openai_codex_auth as codex_auth
from core.inference import openai_codex_client as codex_client
from core.inference.providers import get_provider_info
from routes.provider_credentials import current_credential_write, require_ui_session
from storage import providers_db

router = APIRouter()
logger = structlog.get_logger(__name__)


class OAuthStartRequest(BaseModel):
    method: Literal["browser", "device"]


class OAuthCompleteRequest(BaseModel):
    callback_url: str = Field(..., min_length = 1, max_length = 8192)


def _provider(provider_id: str) -> dict:
    row = providers_db.get_provider(provider_id)
    if row is None:
        raise HTTPException(status_code = 404, detail = "Provider not found")
    if row["provider_type"] != "openai_codex":
        raise HTTPException(
            status_code = 400, detail = "This provider does not use ChatGPT authorization."
        )
    return row


def _safe_error(exc: Exception) -> HTTPException:
    if isinstance(exc, codex_auth.CodexAuthError):
        return HTTPException(status_code = 400, detail = str(exc))
    return HTTPException(status_code = 502, detail = "ChatGPT authorization failed. Please retry.")


def _bundle_persister(credential: tuple, marker: str):
    async def persist(scope_id: str, bundle: dict) -> None:
        async with codex_auth.provider_oauth_write_guard(scope_id):
            with current_credential_write(credential):
                _provider(scope_id)
                if not codex_auth.oauth_flow_marker_matches(scope_id, marker):
                    raise codex_auth.CodexAuthError("Authorization flow is no longer active.")
                codex_auth.save_oauth_bundle(scope_id, bundle)
                codex_auth.set_oauth_flow_marker_status(scope_id, marker, "connected")

    return persist


@router.post("/{provider_id}/oauth/start")
async def start_oauth(
    provider_id: str,
    payload: OAuthStartRequest,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    require_ui_session(via_api_key)
    _provider(provider_id)
    codex_auth.credential_secrets.get_or_create_credential_encryption_key()
    marker = secrets.token_urlsafe(32)

    guarded_persist = _bundle_persister(credential, marker)
    flow = None
    try:
        async with codex_auth.provider_oauth_write_guard(provider_id):
            with current_credential_write(credential):
                codex_auth.save_oauth_flow_marker(provider_id, marker)
        flow = await codex_auth.start_flow(provider_id, payload.method, guarded_persist, marker)
        async with codex_auth.provider_oauth_write_guard(provider_id):
            with current_credential_write(credential):
                if codex_auth.oauth_flow_marker_matches(provider_id, marker):
                    codex_auth.save_oauth_flow_marker(provider_id, marker, flow)
        return codex_auth.safe_flow(flow)
    except Exception as exc:
        if flow is not None:
            await codex_auth.cancel_flow(flow.id)
        try:
            async with codex_auth.provider_oauth_write_guard(provider_id):
                with current_credential_write(credential):
                    codex_auth.delete_oauth_flow_marker(provider_id, marker)
        except codex_auth.CodexAuthError:
            pass
        raise _safe_error(exc) from exc


@router.get("/{provider_id}/oauth/flows/{flow_id}")
async def oauth_status(
    provider_id: str,
    flow_id: str,
    _credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    require_ui_session(via_api_key)
    _provider(provider_id)
    try:
        return codex_auth.safe_flow(codex_auth.get_flow(provider_id, flow_id))
    except Exception as exc:
        raise _safe_error(exc) from exc


@router.post("/{provider_id}/oauth/flows/{flow_id}/complete")
async def complete_oauth(
    provider_id: str,
    flow_id: str,
    payload: OAuthCompleteRequest,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    require_ui_session(via_api_key)
    _provider(provider_id)
    try:
        flow = codex_auth.get_flow(provider_id, flow_id)
        if flow.persist_bundle is None:
            flow.persist_bundle = _bundle_persister(credential, flow.marker)
        flow = await codex_auth.complete_browser_flow(provider_id, flow_id, payload.callback_url)
        return codex_auth.safe_flow(flow)
    except Exception as exc:
        raise _safe_error(exc) from exc


@router.delete("/{provider_id}/oauth/flows/{flow_id}", status_code = 204)
async def cancel_oauth(
    provider_id: str,
    flow_id: str,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    require_ui_session(via_api_key)
    _provider(provider_id)
    try:
        flow = codex_auth.get_flow(provider_id, flow_id)
        # Closing a loopback server can wait for an in-flight callback. Do not
        # hold the credential guard while waiting for that handler to finish.
        await codex_auth.cancel_flow(flow.id)
        async with codex_auth.provider_oauth_write_guard(provider_id):
            with current_credential_write(credential):
                codex_auth.delete_oauth_flow_marker(provider_id, flow.marker)
    except Exception as exc:
        raise _safe_error(exc) from exc


@router.get("/{provider_id}/codex/models")
async def list_subscription_models(
    provider_id: str,
    refresh: bool = False,
    _credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    """Models this plan can reach, falling back to the curated seed."""
    require_ui_session(via_api_key)
    _provider(provider_id)
    curated = [
        {"id": model, "display_name": model, "context_length": None}
        for model in get_provider_info("openai_codex")["default_models"]
    ]
    status = codex_auth.auth_status(provider_id)
    if status == "reauthorization_required":
        # Something already marked this bundle, possibly another worker, after the last
        # provider sync the browser saw. Saying only "curated" here would leave the
        # editor presenting a dead connection as healthy.
        return {"models": curated, "source": "reauthorization_required"}
    if status != "connected":
        return {"models": curated, "source": "curated"}
    try:
        token, account_id = await codex_auth.resolve_access(provider_id)
        models = await codex_client.list_subscription_models(
            provider_id, token, account_id, force = refresh
        )
    except (codex_auth.CodexAuthError, codex_client.CodexReauthorizationError) as exc:
        # resolve_access has already marked the connection as needing reauthorization, so
        # say so in the answer rather than through a 401: the client's authFetch reads
        # every 401 as an expired Unsloth session, refreshes it and retries, and the retry
        # would come back as a plain curated list with the connection looking healthy.
        # A source the picker does not treat as authoritative carries the signal instead.
        logger.info(
            "openai_codex.model_list_reauthorization_required",
            provider_id = provider_id,
            error_type = type(exc).__name__,
        )
        return {"models": curated, "source": "reauthorization_required"}
    except Exception as exc:
        logger.warning(
            "openai_codex.model_list_failed",
            provider_id = provider_id,
            error_type = type(exc).__name__,
        )
        return {"models": curated, "source": "curated"}
    # Only listed slugs are offered, but every slug the plan returned is reported so the
    # picker can tell one it should stop offering from one this account cannot reach.
    offered = [model for model in models if model.get("listed")]
    if not offered:
        return {"models": curated, "source": "curated"}
    return {
        "models": offered,
        # Full entries, not just ids: a hidden slug stays selectable, so the client needs
        # its capabilities too or it will guess and offer what the chat route refuses.
        "known": models,
        "source": "subscription",
    }


@router.delete("/{provider_id}/oauth", status_code = 204)
async def delete_oauth(
    provider_id: str,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    require_ui_session(via_api_key)
    _provider(provider_id)
    codex_auth.credential_secrets.get_or_create_credential_encryption_key()

    await codex_auth.cancel_provider_flows(provider_id)
    codex_client.forget_subscription_models(provider_id)
    async with codex_auth.provider_oauth_write_guard(provider_id):
        with current_credential_write(credential):
            codex_auth.delete_oauth_bundle(provider_id)
            codex_auth.delete_oauth_flow_marker(provider_id)
