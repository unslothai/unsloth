# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GitHub OAuth App device authorization, scoped to an authenticated Studio account."""

import hashlib
import json
import os
import secrets
import sqlite3
import sys
import time
import uuid
from contextlib import asynccontextmanager, closing
from functools import wraps

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from filelock import AsyncFileLock, Timeout

from auth.authentication import (
    authenticated_via_api_key,
    get_current_credential,
    get_current_subject,
)
from core.inference.providers import PROVIDER_REGISTRY
from routes.provider_credentials import (
    current_credential_write,
    require_ui_session,
    resolve_provider_api_key_or_400,
    serialize_provider_config,
)
from storage import credential_secrets, providers_db
from utils.paths import studio_db_path

router = APIRouter(dependencies = [Depends(get_current_subject)])
FLOW_KIND = "github_copilot_oauth_flow"
OAUTH_KIND = "github_copilot_oauth"
# A pending first sign-in uses this account-scoped vault slot until GitHub approves it.
_NEW_CONNECTION = "new"
# Set this to the public Client ID of the unslothai OAuth app with Device Flow enabled.
GITHUB_COPILOT_CLIENT_ID: str | None = None
CLIENT_ID_ENV = "UNSLOTH_GITHUB_COPILOT_CLIENT_ID"
_TOKEN_PATH = "login/oauth/access_token"
_RECONNECT = "GitHub Copilot authorization expired. Reconnect GitHub in Settings → Connections."
_REFRESH_SKEW_SECONDS = 60


def _client_id() -> str:
    return os.environ.get(CLIENT_ID_ENV, "").strip() or GITHUB_COPILOT_CLIENT_ID or ""


@asynccontextmanager
async def provider_write_guard(provider_id: str, *, poll: bool = False):
    # The vault path is account-scoped. Workers must take turns exchanging a rotating refresh token.
    key = hashlib.sha256(provider_id.encode()).hexdigest()[:24]
    suffix = "-poll" if poll else ""
    lock_path = studio_db_path().parent / f".github-copilot-{key}{suffix}.lock"
    try:
        async with AsyncFileLock(lock_path, timeout = 30):
            yield
    except Timeout:
        raise HTTPException(503, "GitHub Copilot sign-in is busy. Try again.") from None


def _serialize_github_credentials(handler):
    @wraps(handler)
    async def wrapped(provider_id: str, *args, **kwargs):
        async with provider_write_guard(provider_id):
            return await handler(provider_id, *args, **kwargs)

    return wrapped


def _load_bundle(provider_id: str) -> dict | None:
    raw = credential_secrets.get_secret(OAUTH_KIND, provider_id)
    return json.loads(raw) if raw else None


def _expiration(body: dict, field: str, now: float) -> float | None:
    value = body.get(field)
    if value is None:
        return None
    try:
        seconds = int(value)
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(502, "GitHub returned invalid token expiration data.") from None
    if seconds <= 0:
        raise HTTPException(502, "GitHub returned invalid token expiration data.")
    return now + seconds


def _token_bundle(body: dict, client_id: str) -> dict:
    now = time.time()
    return {
        "client_id": client_id,
        "access_token": body["access_token"],
        "expires_at": _expiration(body, "expires_in", now),
        "refresh_token": body.get("refresh_token"),
        "refresh_expires_at": _expiration(body, "refresh_token_expires_in", now),
    }


def _can_refresh(bundle: dict, now: float) -> bool:
    expires = bundle.get("refresh_expires_at")
    return bool(bundle.get("refresh_token")) and (expires is None or expires > now)


def auth_status(provider_id: str) -> str:
    bundle = _load_bundle(provider_id)
    if bundle is None:
        # Preserve connections made before refresh support: they stored the access token alone.
        return (
            "connected"
            if credential_secrets.provider_api_key_saved(provider_id)
            else "disconnected"
        )
    if bundle.get("reauthorization_required"):
        return "reauthorization_required"
    now = time.time()
    expires = bundle.get("expires_at")
    if expires is not None and expires <= now and not _can_refresh(bundle, now):
        return "reauthorization_required"
    return "connected"


def _usable_access(bundle: dict, now: float) -> str | None:
    if bundle.get("reauthorization_required"):
        raise HTTPException(401, _RECONNECT)
    expires = bundle.get("expires_at")
    if expires is None or expires > now + _REFRESH_SKEW_SECONDS:
        return bundle["access_token"]
    if not _can_refresh(bundle, now):
        if expires > now:
            return bundle["access_token"]
        raise HTTPException(401, _RECONNECT)
    return None


async def resolve_access(
    provider_id: str | None, encrypted_api_key: str | None, *, allow_saved_key: bool
) -> str:
    if encrypted_api_key or not provider_id or not allow_saved_key:
        return resolve_provider_api_key_or_400(
            provider_id,
            encrypted_api_key,
            allow_saved_key = allow_saved_key,
        )
    bundle = _load_bundle(provider_id)
    if bundle is None:
        return resolve_provider_api_key_or_400(provider_id, None)
    access = _usable_access(bundle, time.time())
    if access is not None:
        return access

    async with provider_write_guard(provider_id):
        # A concurrent request may have refreshed, reconnected, or disconnected.
        bundle = _load_bundle(provider_id)
        if bundle is None:
            raise HTTPException(401, _RECONNECT)
        access = _usable_access(bundle, time.time())
        if access is not None:
            return access
        body = await _github_post(
            _TOKEN_PATH,
            {
                "client_id": bundle["client_id"],
                "grant_type": "refresh_token",
                "refresh_token": bundle["refresh_token"],
            },
        )
        error = body.get("error")
        if error in ("bad_refresh_token", "invalid_grant", "expired_token"):
            credential_secrets.upsert_secret(
                OAUTH_KIND,
                provider_id,
                json.dumps({"reauthorization_required": True}),
            )
            raise HTTPException(401, _RECONNECT)
        if error or not isinstance(body.get("access_token"), str) or not body["access_token"]:
            raise HTTPException(502, "GitHub could not refresh Copilot access. Try again.")
        refreshed = _token_bundle(body, bundle["client_id"])
        credential_secrets.upsert_secret(OAUTH_KIND, provider_id, json.dumps(refreshed))
        return refreshed["access_token"]


def delete_credentials(provider_id: str, *, connection: sqlite3.Connection) -> None:
    credential_secrets.delete_secret(FLOW_KIND, provider_id, connection = connection)
    credential_secrets.delete_secret(OAUTH_KIND, provider_id, connection = connection)
    credential_secrets.delete_provider_api_key(provider_id, connection = connection)


def _require_provider(
    provider_id: str,
    via_api_key: bool,
    *,
    allow_new: bool = False,
):
    require_ui_session(via_api_key)
    if allow_new and provider_id == _NEW_CONNECTION:
        return
    row = providers_db.get_own_provider(provider_id)
    if row is None or row["provider_type"] != "github_copilot":
        raise HTTPException(404, "GitHub Copilot connection not found.")
    return row


async def _github_post(path: str, data: dict) -> dict:
    try:
        async with httpx.AsyncClient(timeout = 20, follow_redirects = False) as client:
            response = await client.post(
                "https://github.com/" + path,
                data = data,
                headers = {"Accept": "application/json"},
            )
            response.raise_for_status()
            return response.json()
    except (httpx.HTTPError, ValueError):
        raise HTTPException(502, "Could not contact GitHub. Try signing in again.") from None


@router.post("/{provider_id}/github/start")
@serialize_provider_config
@_serialize_github_credentials
async def start(
    provider_id: str,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    _require_provider(provider_id, via_api_key, allow_new = True)
    if sys.version_info < (3, 11):
        raise HTTPException(
            503, "GitHub Copilot subscriptions require Studio running on Python 3.11 or newer."
        )
    client_id = _client_id()
    if not client_id:
        raise HTTPException(
            503,
            "GitHub Copilot sign-in is unavailable in this Studio build. "
            "Ask your Studio administrator to configure the OAuth app.",
        )
    result = await _github_post("login/device/code", {"client_id": client_id})
    if not all(result.get(key) for key in ("device_code", "user_code", "expires_in")):
        raise HTTPException(
            502, "GitHub refused device authorization. Check the OAuth App's Device Flow setting."
        )
    flow_id = secrets.token_urlsafe(24)
    interval = max(5, int(result.get("interval", 5)))
    flow = {
        "id": flow_id,
        "device_code": result["device_code"],
        "client_id": client_id,
        "expires_at": time.time() + int(result["expires_in"]),
        "interval": interval,
        "next_poll": time.time() + interval,
    }
    # The first-use encryption key lives in auth.db. Initialize it before taking that database's
    # credential-generation write lock, since creating the key also needs a write transaction.
    credential_secrets.get_or_create_credential_encryption_key()
    with current_credential_write(credential):
        credential_secrets.upsert_secret(FLOW_KIND, provider_id, json.dumps(flow))
    return {
        "flow_id": flow_id,
        "user_code": result["user_code"],
        "verification_uri": "https://github.com/login/device",
        "expires_in": result["expires_in"],
        "interval": interval,
    }


def _matching_flow(provider_id: str, flow_id: str) -> dict | None:
    raw = credential_secrets.get_secret(FLOW_KIND, provider_id)
    flow = json.loads(raw) if raw else None
    if flow and secrets.compare_digest(flow["id"], flow_id):
        return flow
    return None


@router.post("/{provider_id}/github/flows/{flow_id}/poll")
async def poll(
    provider_id: str,
    flow_id: str,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    _require_provider(provider_id, via_api_key, allow_new = True)
    # Serialize polls across workers while letting cancel or disconnect take the write lock
    # during the GitHub request. Recheck the flow before applying any response.
    async with provider_write_guard(provider_id, poll = True):
        async with provider_write_guard(provider_id):
            flow = _matching_flow(provider_id, flow_id)
            if not flow or "provider_id" in flow:
                raise HTTPException(404, "Sign-in attempt not found. Start again.")
            now = time.time()
            if now >= flow["expires_at"]:
                with current_credential_write(credential):
                    credential_secrets.delete_secret(FLOW_KIND, provider_id)
                return {"status": "expired"}
            if now < flow["next_poll"]:
                return {"status": "pending", "interval": flow["interval"]}
            # Persist before network I/O, including on failures, to honor GitHub's polling limit.
            flow["next_poll"] = now + flow["interval"]
            with current_credential_write(credential):
                credential_secrets.upsert_secret(FLOW_KIND, provider_id, json.dumps(flow))
        result = await _github_post(
            _TOKEN_PATH,
            {
                "client_id": flow["client_id"],
                "device_code": flow["device_code"],
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
        )
        async with provider_write_guard(provider_id):
            if not _matching_flow(provider_id, flow_id):
                raise HTTPException(404, "Sign-in attempt not found. Start again.")
            return _apply_poll_result(provider_id, flow, result, credential)


def _apply_poll_result(provider_id: str, flow: dict, result: dict, credential: tuple) -> dict:
    error = result.get("error")
    if error in ("authorization_pending", "slow_down"):
        if error == "slow_down":
            flow["interval"] += 5
            flow["next_poll"] = time.time() + flow["interval"]
            with current_credential_write(credential):
                credential_secrets.upsert_secret(FLOW_KIND, provider_id, json.dumps(flow))
        return {"status": "pending", "interval": flow["interval"]}
    token = result.get("access_token")
    if not error and isinstance(token, str) and token:
        bundle = _token_bundle(result, flow["client_id"])
        # Retain just the outcome so a late cancel can report that approval already finished.
        if provider_id == _NEW_CONNECTION:
            new_id = uuid.uuid4().hex[:16]
            info = PROVIDER_REGISTRY["github_copilot"]
            with (
                current_credential_write(credential),
                providers_db.provider_bundle_transaction() as conn,
            ):
                providers_db.create_provider(
                    id = new_id,
                    provider_type = "github_copilot",
                    display_name = info["display_name"],
                    base_url = info["base_url"],
                    connection = conn,
                )
                credential_secrets.upsert_secret(
                    OAUTH_KIND,
                    new_id,
                    json.dumps(bundle),
                    connection = conn,
                )
                credential_secrets.upsert_secret(
                    FLOW_KIND,
                    provider_id,
                    json.dumps({"id": flow["id"], "provider_id": new_id}),
                    connection = conn,
                )
            return {"status": "connected", "provider_id": new_id}
        with current_credential_write(credential):
            with closing(credential_secrets.get_connection()) as conn, conn:
                credential_secrets.upsert_secret(
                    OAUTH_KIND,
                    provider_id,
                    json.dumps(bundle),
                    connection = conn,
                )
                credential_secrets.delete_provider_api_key(provider_id, connection = conn)
                credential_secrets.upsert_secret(
                    FLOW_KIND,
                    provider_id,
                    json.dumps({"id": flow["id"], "provider_id": provider_id}),
                    connection = conn,
                )
        return {"status": "connected"}
    with current_credential_write(credential):
        credential_secrets.delete_secret(FLOW_KIND, provider_id)
    if error in ("expired_token", "token_expired"):
        return {"status": "expired"}
    if error == "access_denied":
        return {"status": "denied"}
    raise HTTPException(502, "GitHub refused authorization. Start sign-in again.")


@router.delete("/{provider_id}/github/flows/{flow_id}", status_code = 204)
@serialize_provider_config
@_serialize_github_credentials
async def cancel(
    provider_id: str,
    flow_id: str,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    _require_provider(provider_id, via_api_key, allow_new = True)
    flow = _matching_flow(provider_id, flow_id)
    if flow is None:
        return
    connected_id = flow.get("provider_id")
    if (
        connected_id
        and providers_db.get_own_provider(connected_id)
        and auth_status(connected_id) == "connected"
    ):
        return JSONResponse({"provider_id": connected_id})
    with current_credential_write(credential):
        credential_secrets.delete_secret(FLOW_KIND, provider_id)


@router.delete("/{provider_id}/github", status_code = 204)
@serialize_provider_config
@_serialize_github_credentials
async def disconnect(
    provider_id: str,
    credential: tuple = Depends(get_current_credential),
    via_api_key: bool = Depends(authenticated_via_api_key),
):
    _require_provider(provider_id, via_api_key)
    with current_credential_write(credential):
        with closing(credential_secrets.get_connection()) as conn, conn:
            delete_credentials(provider_id, connection = conn)
