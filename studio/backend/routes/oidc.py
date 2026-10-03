# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Generic OpenID Connect authorization-code flow routes."""

from __future__ import annotations

from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import RedirectResponse

from auth.authentication import create_access_token, create_refresh_token
from auth.oidc_config import oidc_config
from auth.oidc_discovery import OIDCDiscoveryClient, OIDCProviderUnavailable
from auth.oidc_handoff import oidc_session_handoffs
from auth.oidc_state import oidc_state_manager
from auth.oidc_storage import create_external_account, record_external_login
from auth.oidc_tokens import (
    OIDCAuthenticationError,
    exchange_authorization_code,
    validate_id_token,
)
from auth.storage import get_user_record
from models.auth import OIDCConfigResponse, OIDCHandoffRequest
from models.users import Token


router = APIRouter()
_discovery_client: OIDCDiscoveryClient | None = None


def _enabled_config():
    config = oidc_config()
    if not config.enabled:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail = "OIDC is disabled")
    return config


def _discovery(config) -> OIDCDiscoveryClient:
    global _discovery_client
    if _discovery_client is None:
        _discovery_client = OIDCDiscoveryClient(config)
    return _discovery_client


@router.get("/config", response_model = OIDCConfigResponse)
def public_oidc_config(response: Response) -> OIDCConfigResponse:
    response.headers["Cache-Control"] = "no-store"
    config = oidc_config()
    return OIDCConfigResponse(
        enabled = config.enabled,
        display_name = config.display_name if config.enabled else None,
    )


@router.get("/login")
def oidc_login():
    config = _enabled_config()
    try:
        metadata = _discovery(config).metadata()
    except OIDCProviderUnavailable as exc:
        raise HTTPException(status_code = 503, detail = "The SSO provider is unavailable") from exc
    attempt = oidc_state_manager.create()
    query = urlencode(
        {
            "response_type": "code",
            "client_id": config.client_id,
            "redirect_uri": config.redirect_uri,
            "scope": config.scope,
            "state": attempt.state,
            "nonce": attempt.nonce,
            "code_challenge": attempt.code_challenge,
            "code_challenge_method": "S256",
        }
    )
    return RedirectResponse(
        f"{metadata.authorization_endpoint}?{query}",
        status_code = 302,
        headers = {"Cache-Control": "no-store"},
    )


@router.get("/callback")
def oidc_callback(
    code: str = "",
    state: str = "",
    error: str = "",
):
    config = _enabled_config()
    attempt = oidc_state_manager.consume(state)
    if attempt is None:
        raise HTTPException(status_code = 400, detail = "Invalid or expired OIDC login state")
    if error:
        raise HTTPException(status_code = 401, detail = "SSO authentication was rejected")
    if not code:
        raise HTTPException(status_code = 400, detail = "Invalid or expired OIDC login state")
    discovery = _discovery(config)
    try:
        token_response = exchange_authorization_code(
            code,
            code_verifier = attempt.code_verifier,
            config = config,
            discovery = discovery,
        )
        claims = validate_id_token(
            token_response["id_token"],
            expected_nonce = attempt.nonce,
            config = config,
            discovery = discovery,
        )
    except OIDCProviderUnavailable as exc:
        raise HTTPException(status_code = 503, detail = "The SSO provider is unavailable") from exc
    except OIDCAuthenticationError as exc:
        raise HTTPException(status_code = 401, detail = "SSO authentication failed") from exc

    if config.allowed_groups:
        groups = claims.get("groups")
        memberships = set(groups) if isinstance(groups, list) else set()
        if not memberships.intersection(config.allowed_groups):
            raise HTTPException(status_code = 403, detail = "This SSO identity is not allowed")

    account = record_external_login(
        issuer = config.issuer,
        subject = claims["sub"],
        email = claims.get(config.email_claim),
        preferred_username = claims.get(config.username_claim),
    )
    if account is None:
        if not config.auto_create_users:
            raise HTTPException(status_code = 403, detail = "This SSO identity has no Unsloth account")
        account = create_external_account(
            issuer = config.issuer,
            subject = claims["sub"],
            preferred_username = claims.get(config.username_claim),
            email = claims.get(config.email_claim),
            display_name = claims.get("name"),
        )
    if not account["is_active"]:
        raise HTTPException(status_code = 403, detail = "This Unsloth account is disabled")
    record = get_user_record(account["username"])
    if record is None or record["account_id"] != account["account_id"]:
        raise HTTPException(status_code = 401, detail = "The mapped Unsloth account is unavailable")
    secret = record["jwt_secret"]
    access_token = create_access_token(subject = account["username"], secret = secret)
    refresh_token = create_refresh_token(subject = account["username"], secret = secret)
    handoff = oidc_session_handoffs.create(
        access_token = access_token,
        refresh_token = refresh_token,
        account_id = account["account_id"],
    )
    # Only an opaque, single-use code crosses the URL. Session credentials never enter logs,
    # browser history, Referer headers, or provider-visible URLs.
    return RedirectResponse(
        f"/auth/oidc/callback?handoff={handoff}",
        status_code = 302,
        headers = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"},
    )


@router.post("/handoff", response_model = Token)
def oidc_handoff(payload: OIDCHandoffRequest, response: Response) -> Token:
    response.headers["Cache-Control"] = "no-store"
    _enabled_config()
    handoff = oidc_session_handoffs.consume(payload.handoff)
    if handoff is None:
        raise HTTPException(status_code = 401, detail = "Invalid or expired SSO session handoff")
    return Token(
        access_token = handoff.access_token,
        refresh_token = handoff.refresh_token,
        token_type = "bearer",
        must_change_password = False,
        account_id = handoff.account_id,
    )
