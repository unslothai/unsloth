# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""UI-session-only lifecycle routes for ChatGPT/Codex subscription OAuth."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from auth.authentication import authenticated_via_api_key, get_current_credential
from core.inference import openai_codex_auth as codex_auth
from routes.provider_credentials import current_credential_write, require_ui_session
from storage import providers_db

router = APIRouter()


class OAuthStartRequest(BaseModel):
    method: Literal["browser", "device"]


class OAuthCompleteRequest(BaseModel):
    callback_url: str = Field(..., min_length=1, max_length=8192)


def _provider(provider_id: str) -> dict:
    row = providers_db.get_provider(provider_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Provider not found")
    if row["provider_type"] != "openai_codex":
        raise HTTPException(status_code=400, detail="This provider does not use ChatGPT authorization.")
    return row


def _safe_error(exc: Exception) -> HTTPException:
    if isinstance(exc, codex_auth.CodexAuthError):
        return HTTPException(status_code=400, detail=str(exc))
    return HTTPException(status_code=502, detail="ChatGPT authorization failed. Please retry.")


@router.post("/{provider_id}/oauth/start")
async def start_oauth(
    provider_id: str,
    payload: OAuthStartRequest,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    require_ui_session(via_api_key)
    _provider(provider_id)
    # Warm the auth-owned key before a captured generation guard takes its lock.
    codex_auth.credential_secrets.get_or_create_credential_encryption_key()

    def guarded_persist(scope_id: str, bundle: dict) -> None:
        with current_credential_write(credential):
            codex_auth.save_oauth_bundle(scope_id, bundle)

    try:
        return codex_auth.safe_flow(
            await codex_auth.start_flow(provider_id, payload.method, guarded_persist)
        )
    except Exception as exc:
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
        # Persistence uses the generation-guarded callback captured at start.
        flow = await codex_auth.complete_browser_flow(
            provider_id, flow_id, payload.callback_url
        )
        return codex_auth.safe_flow(flow)
    except Exception as exc:
        raise _safe_error(exc) from exc


@router.delete("/{provider_id}/oauth/flows/{flow_id}", status_code=204)
async def cancel_oauth(
    provider_id: str,
    flow_id: str,
    _credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    require_ui_session(via_api_key)
    _provider(provider_id)
    flow = codex_auth.get_flow(provider_id, flow_id)
    await codex_auth.cancel_flow(flow.id)



@router.delete("/{provider_id}/oauth", status_code=204)
async def delete_oauth(
    provider_id: str,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    require_ui_session(via_api_key)
    _provider(provider_id)
    codex_auth.credential_secrets.get_or_create_credential_encryption_key()

    # Never hold SQLite's credential-generation write lock across an await.
    # Active callback/listener work is cancelled first; only the synchronous
    # credential deletion runs inside the guard.
    await codex_auth.cancel_provider_flows(provider_id)
    with current_credential_write(credential):
        codex_auth.delete_oauth_bundle(provider_id)
