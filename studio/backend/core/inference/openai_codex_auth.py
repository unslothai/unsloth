# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ChatGPT/Codex public-client OAuth owned by the Studio backend.

OAuth transient material never crosses the API boundary: callers receive only an
opaque flow id and safe user-facing authorization metadata.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from filelock import FileLock, Timeout as FileLockTimeout

from storage import credential_secrets

from utils.paths import studio_db_path

OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_CODEX_ISSUER = "https://auth.openai.com"
OPENAI_CODEX_AUTHORIZE_URL = f"{OPENAI_CODEX_ISSUER}/oauth/authorize"
OPENAI_CODEX_TOKEN_URL = f"{OPENAI_CODEX_ISSUER}/oauth/token"
OPENAI_CODEX_DEVICE_CODE_URL = f"{OPENAI_CODEX_ISSUER}/api/accounts/deviceauth/usercode"
OPENAI_CODEX_DEVICE_TOKEN_URL = f"{OPENAI_CODEX_ISSUER}/api/accounts/deviceauth/token"
OPENAI_CODEX_SCOPE = "openid profile email offline_access"
OPENAI_CODEX_LOOPBACK_PORTS = (1455,)
OPENAI_CODEX_CALLBACK_PATH = "/auth/callback"

OPENAI_CODEX_DEVICE_REDIRECT_URI = f"{OPENAI_CODEX_ISSUER}/deviceauth/callback"
OPENAI_CODEX_ORIGINATOR = "unsloth_studio"
OPENAI_CODEX_API_BASE = "https://chatgpt.com/backend-api"
OPENAI_CODEX_RESPONSES_URL = f"{OPENAI_CODEX_API_BASE}/codex/responses"
OPENAI_CODEX_USER_AGENT = "unsloth-studio/1"
OPENAI_CODEX_COMPATIBILITY_INSTRUCTIONS = (
    "You are operating inside Unsloth Studio. Follow the user's instructions, "
    "use only tools supplied in this request, and return concise, accurate results."
)

_FLOW_TTL_SECONDS = 15 * 60
_REFRESH_SKEW_SECONDS = 5 * 60

_FLOW_TERMINAL_RETENTION_SECONDS = 60


class CodexAuthError(RuntimeError):
    """Sanitized authentication failure safe to return to the UI."""


class CodexReauthorizationRequired(CodexAuthError):
    """The saved refresh credential was permanently rejected."""


@dataclass
class OAuthFlow:
    id: str
    provider_id: str
    method: Literal["browser", "device"]
    created_at: float
    expires_at: float
    state: str = ""
    verifier: str = ""
    redirect_uri: str = ""
    authorization_url: str = ""
    verification_url: str = ""
    user_code: str = ""
    device_auth_id: str = ""
    interval: float = 5.0
    status: Literal["pending", "connected", "error", "cancelled"] = "pending"
    message: str = ""
    consumed: bool = False
    server: asyncio.AbstractServer | None = field(default=None, repr=False)
    task: asyncio.Task | None = field(default=None, repr=False)

    cleanup_task: asyncio.Task | None = field(default=None, repr=False)

    persist_bundle: Callable[[str, dict[str, Any]], None] | None = field(
        default=None, repr=False
    )


_flows: dict[str, OAuthFlow] = {}
_flows_lock = asyncio.Lock()
_refresh_locks: dict[str, asyncio.Lock] = {}


def _flow_is_stale(flow: OAuthFlow, now: float) -> bool:
    if flow.status == "pending":
        return now >= flow.expires_at
    return now >= min(flow.expires_at, flow.created_at + _FLOW_TERMINAL_RETENTION_SECONDS)


async def _prune_flows() -> None:
    now = time.time()
    for flow_id, flow in list(_flows.items()):
        if _flow_is_stale(flow, now):
            await cancel_flow(flow_id)



async def _expire_and_remove_flow(flow: OAuthFlow) -> None:
    """Expire a pending flow at its deadline, then discard terminal metadata."""
    await asyncio.sleep(max(0.0, flow.expires_at - time.time()))
    if _flows.get(flow.id) is not flow:
        return
    if flow.status == "pending":
        flow.status = "error"
        flow.message = "Authorization expired. Start a new connection."
        if flow.task:
            flow.task.cancel()
        if flow.server:
            flow.server.close()
            await flow.server.wait_closed()
            flow.server = None
    await asyncio.sleep(_FLOW_TERMINAL_RETENTION_SECONDS)
    if _flows.get(flow.id) is flow:
        await cancel_flow(flow.id)


async def shutdown_flows() -> None:
    """Cancel every loopback listener/device poll during application shutdown."""
    for flow_id in list(_flows):
        await cancel_flow(flow_id)


def mark_reauthorization_required(provider_id: str) -> None:
    bundle = load_oauth_bundle(provider_id)
    if bundle:
        bundle["reauthorization_required"] = True
        save_oauth_bundle(provider_id, bundle)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def create_pkce() -> tuple[str, str]:
    verifier = _b64url(secrets.token_bytes(48))
    challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def extract_chatgpt_account_id(access_token: str) -> str:
    """Decode only the bounded JWT payload needed as an upstream routing hint."""
    parts = access_token.split(".")
    if len(parts) < 2 or len(parts[1]) > 16_384:
        raise CodexAuthError("ChatGPT returned an invalid access token.")
    try:
        raw = base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4))
        payload = json.loads(raw)
    except Exception as exc:
        raise CodexAuthError("ChatGPT returned an invalid access token.") from exc
    account_id = payload.get("https://api.openai.com/auth", {}).get("chatgpt_account_id")
    if not isinstance(account_id, str) or not account_id or len(account_id) > 512:
        account_id = payload.get("https://api.openai.com/auth.chatgpt_account_id")
    if not isinstance(account_id, str) or not account_id or len(account_id) > 512:
        raise CodexAuthError("The ChatGPT account identifier was missing.")
    return account_id


def _validate_token_payload(body: Any, previous_refresh_token: str = "") -> dict[str, Any]:
    if not isinstance(body, dict):
        raise CodexAuthError("ChatGPT returned an invalid token response.")
    access_token = body.get("access_token")
    refresh_token = body.get("refresh_token") or previous_refresh_token
    expires_in = body.get("expires_in", 3600)
    if not isinstance(access_token, str) or not access_token:
        raise CodexAuthError("ChatGPT returned an invalid token response.")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise CodexAuthError("ChatGPT did not return a refresh token.")
    try:
        expires_in = max(60, min(int(expires_in), 30 * 24 * 3600))
    except (TypeError, ValueError) as exc:
        raise CodexAuthError("ChatGPT returned an invalid token lifetime.") from exc
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": int(time.time()) + expires_in,
        "account_id": extract_chatgpt_account_id(access_token),
    }


def save_oauth_bundle(provider_id: str, bundle: dict[str, Any]) -> None:
    credential_secrets.upsert_secret(
        credential_secrets.OPENAI_CODEX_OAUTH_KIND,
        provider_id,
        json.dumps(bundle, separators=(",", ":")),
    )


def load_oauth_bundle(provider_id: str) -> dict[str, Any] | None:
    raw = credential_secrets.get_secret(credential_secrets.OPENAI_CODEX_OAUTH_KIND, provider_id)
    if not raw:
        return None
    try:
        bundle = json.loads(raw)
    except Exception:
        return None
    required = ("access_token", "refresh_token", "expires_at", "account_id")
    return bundle if isinstance(bundle, dict) and all(bundle.get(key) for key in required) else None


def auth_status(provider_id: str) -> str:
    bundle = load_oauth_bundle(provider_id)
    if not bundle:
        return "disconnected"
    # An expired access token is still usable after refresh. Only a permanent
    # refresh rejection should ask the user to reconnect.
    return "reauthorization_required" if bundle.get("reauthorization_required") else "connected"


async def _token_request(data: dict[str, Any]) -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False, trust_env=False) as client:
            response = await client.post(OPENAI_CODEX_TOKEN_URL, data=data)
    except httpx.HTTPError as exc:
        raise CodexAuthError("Could not reach ChatGPT authentication.") from exc
    if response.status_code >= 400:
        error_code = ""
        try:
            error = response.json().get("error")
            error_code = error.get("code", "") if isinstance(error, dict) else str(error or "")
        except Exception:
            pass
        if data.get("grant_type") == "refresh_token" and error_code in {
            "invalid_grant", "invalid_refresh_token", "refresh_token_expired",
        }:
            raise CodexReauthorizationRequired(
                "ChatGPT authorization is no longer valid. Please reconnect."
            )
        raise CodexAuthError("ChatGPT authorization failed. Please reconnect.")
    try:
        return response.json()
    except Exception as exc:
        raise CodexAuthError("ChatGPT returned an invalid authorization response.") from exc


async def _exchange_code(
    flow: OAuthFlow,
    code: str,
    *,
    verifier: str | None = None,
    redirect_uri: str | None = None,
) -> None:
    if flow.consumed:
        raise CodexAuthError("Authorization callback was already used.")
    flow.consumed = True
    try:
        body = await _token_request({
            "grant_type": "authorization_code",
            "client_id": OPENAI_CODEX_CLIENT_ID,
            "code": code,
            "redirect_uri": redirect_uri or flow.redirect_uri,
            "code_verifier": verifier or flow.verifier,
        })
        bundle = _validate_token_payload(body)
        if flow.persist_bundle is None:
            raise CodexAuthError("Authorization flow can no longer save credentials.")
        flow.persist_bundle(flow.provider_id, bundle)
    except Exception:
        flow.status = "error"
        flow.message = "ChatGPT authorization failed. Please reconnect."
        raise
    flow.status = "connected"
    if flow.server:
        flow.server.close()
        await flow.server.wait_closed()
        flow.server = None


async def _loopback_handler(flow: OAuthFlow, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        first = await asyncio.wait_for(reader.readline(), timeout=5)
        target = first.decode("ascii", "ignore").split(" ")[1]
        parsed = urlparse(target)
        query = parse_qs(parsed.query)
        if parsed.path != OPENAI_CODEX_CALLBACK_PATH or query.get("state", [""])[0] != flow.state:
            raise CodexAuthError("Authorization state did not match.")
        code = query.get("code", [""])[0]
        if not code or flow.consumed:
            raise CodexAuthError("Authorization callback was invalid or already used.")
        await _exchange_code(flow, code)
        message = "ChatGPT connected. You can close this window."
    except Exception as exc:
        # Stray requests and state mismatches must not poison the active flow.
        if flow.consumed and flow.status == "pending":
            flow.status = "error"
            flow.message = (
                str(exc) if isinstance(exc, CodexAuthError) else "Authorization failed."
            )
        message = "Authorization failed. Return to Unsloth Studio and try again."
    body = ("<!doctype html><meta charset=utf-8><title>Unsloth Studio</title>" +
            "<p>" + message.replace("&", "&amp;").replace("<", "&lt;") + "</p>").encode()
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nCache-Control: no-store\r\nContent-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body)
    await writer.drain()
    writer.close()
    await writer.wait_closed()


async def _start_browser_flow(
    provider_id: str,
    persist_bundle: Callable[[str, dict[str, Any]], None],
) -> OAuthFlow:
    verifier, challenge = create_pkce()
    flow = OAuthFlow(
        id=secrets.token_urlsafe(24), provider_id=provider_id, method="browser",
        created_at=time.time(), expires_at=time.time() + _FLOW_TTL_SECONDS,
        state=secrets.token_urlsafe(32), verifier=verifier,
        persist_bundle=persist_bundle,
    )
    bound_port: int | None = None
    for port in OPENAI_CODEX_LOOPBACK_PORTS:
        try:
            flow.server = await asyncio.start_server(
                lambda r, w, f=flow: _loopback_handler(f, r, w), "127.0.0.1", port
            )
            bound_port = port
            break
        except OSError:
            continue
    # The callback URL must match the verifier exchange even when no listener
    # can bind. Manual completion remains valid with the canonical URL.
    callback_port = bound_port or OPENAI_CODEX_LOOPBACK_PORTS[0]
    # OpenAI registers localhost redirect URIs; the listener itself remains
    # pinned to 127.0.0.1 so it can never accept non-loopback traffic.
    flow.redirect_uri = f"http://localhost:{callback_port}{OPENAI_CODEX_CALLBACK_PATH}"
    flow.authorization_url = OPENAI_CODEX_AUTHORIZE_URL + "?" + urlencode({
        "client_id": OPENAI_CODEX_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": flow.redirect_uri,
        "scope": OPENAI_CODEX_SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": flow.state,
        "originator": OPENAI_CODEX_ORIGINATOR,
        "codex_cli_simplified_flow": "true",

        "id_token_add_organizations": "true",
    })
    return flow


async def _device_poll(flow: OAuthFlow) -> None:
    while flow.status == "pending" and time.time() < flow.expires_at:
        await asyncio.sleep(flow.interval)
        try:
            async with httpx.AsyncClient(
                timeout=30.0, follow_redirects=False, trust_env=False
            ) as client:
                response = await client.post(OPENAI_CODEX_DEVICE_TOKEN_URL, json={
                    "device_auth_id": flow.device_auth_id,
                    "user_code": flow.user_code,
                })
            if response.status_code in (403, 404):
                continue
            if response.status_code >= 400:
                error_code = ""
                server_interval: Any = None
                try:
                    error_body = response.json()
                    error = error_body.get("error")
                    error_code = (
                        error.get("code", "") if isinstance(error, dict) else str(error or "")
                    )
                    server_interval = error_body.get("interval")
                except Exception:
                    pass
                if error_code == "deviceauth_authorization_pending":
                    continue
                if error_code == "slow_down" or response.status_code == 429:
                    try:
                        requested = float(server_interval)
                    except (TypeError, ValueError):
                        requested = flow.interval + 5
                    flow.interval = min(30.0, max(flow.interval + 5, requested))
                    continue
                raise CodexAuthError(
                    "Device authorization failed. Enable device-code login in ChatGPT settings and retry."
                )
            body = response.json()
            code = body.get("authorization_code")
            verifier = body.get("code_verifier")
            if (
                not isinstance(code, str)
                or not code
                or not isinstance(verifier, str)
                or not verifier
            ):
                raise CodexAuthError(
                    "ChatGPT returned an invalid device authorization response."
                )
            await _exchange_code(
                flow,
                code,
                verifier=verifier,
                redirect_uri=OPENAI_CODEX_DEVICE_REDIRECT_URI,
            )
            return
        except asyncio.CancelledError:
            return
        except CodexAuthError as exc:
            flow.status = "error"
            flow.message = str(exc)
            return
        except Exception:
            flow.status = "error"
            flow.message = "Device authorization failed. Please retry."
            return
    if flow.status == "pending":
        flow.status = "error"
        flow.message = "Device authorization expired. Start a new connection."


async def _start_device_flow(
    provider_id: str,
    persist_bundle: Callable[[str, dict[str, Any]], None],
) -> OAuthFlow:
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False, trust_env=False) as client:
            response = await client.post(OPENAI_CODEX_DEVICE_CODE_URL, json={"client_id": OPENAI_CODEX_CLIENT_ID})
    except httpx.HTTPError as exc:
        raise CodexAuthError("Could not reach ChatGPT authentication.") from exc
    if response.status_code >= 400:
        raise CodexAuthError("Device login is unavailable. Enable device-code login in ChatGPT settings.")
    try:
        body = response.json()
        device_auth_id = body["device_auth_id"]
        user_code = body["user_code"]
        verification_url = body.get("verification_uri_complete") or body.get("verification_uri") or "https://auth.openai.com/codex/device"
    except Exception as exc:
        raise CodexAuthError("ChatGPT returned an invalid device authorization response.") from exc
    flow = OAuthFlow(
        id=secrets.token_urlsafe(24), provider_id=provider_id, method="device",
        created_at=time.time(), expires_at=time.time() + min(int(body.get("expires_in", _FLOW_TTL_SECONDS)), _FLOW_TTL_SECONDS),
        device_auth_id=str(device_auth_id), user_code=str(user_code), verification_url=str(verification_url),
        interval=max(1.0, min(float(body.get("interval", 5)), 30.0)),
        redirect_uri=OPENAI_CODEX_DEVICE_REDIRECT_URI,
        persist_bundle=persist_bundle,
    )
    flow.task = asyncio.create_task(_device_poll(flow))
    return flow


async def start_flow(
    provider_id: str,
    method: Literal["browser", "device"],
    persist_bundle: Callable[[str, dict[str, Any]], None],
) -> OAuthFlow:
    async with _flows_lock:

        await _prune_flows()
        for old in list(_flows.values()):
            if old.provider_id == provider_id and old.status == "pending":
                await cancel_flow(old.id)
        flow = await (
            _start_browser_flow(provider_id, persist_bundle)
            if method == "browser"
            else _start_device_flow(provider_id, persist_bundle)
        )
        _flows[flow.id] = flow

        flow.cleanup_task = asyncio.create_task(_expire_and_remove_flow(flow))
        return flow


def safe_flow(flow: OAuthFlow) -> dict[str, Any]:
    return {
        "flow_id": flow.id,
        "method": flow.method,
        "status": flow.status,
        "expires_at": int(flow.expires_at),
        "authorization_url": flow.authorization_url or None,
        "verification_url": flow.verification_url or None,
        "user_code": flow.user_code or None,
        "message": flow.message or None,
    }


def get_flow(provider_id: str, flow_id: str) -> OAuthFlow:
    flow = _flows.get(flow_id)
    if flow is None or flow.provider_id != provider_id:
        raise CodexAuthError("Authorization flow was not found or expired.")
    if time.time() >= flow.expires_at and flow.status == "pending":
        flow.status = "error"
        flow.message = "Authorization expired. Start a new connection."
        if flow.task:
            flow.task.cancel()
        if flow.server:
            flow.server.close()
    return flow


async def complete_browser_flow(provider_id: str, flow_id: str, callback_url: str) -> OAuthFlow:
    flow = get_flow(provider_id, flow_id)
    if flow.method != "browser" or flow.status != "pending" or flow.consumed:
        raise CodexAuthError("Authorization flow is no longer active.")
    parsed = urlparse(callback_url)
    expected = urlparse(flow.redirect_uri)
    if (
        parsed.scheme != expected.scheme
        or parsed.hostname != expected.hostname
        or parsed.port != expected.port
        or parsed.path != expected.path
        or parsed.fragment
    ):
        raise CodexAuthError("Paste the complete localhost ChatGPT callback URL.")
    query = parse_qs(parsed.query)
    if not secrets.compare_digest(query.get("state", [""])[0], flow.state):
        raise CodexAuthError("Authorization state did not match.")
    code = query.get("code", [""])[0]
    if not code:
        raise CodexAuthError("The callback URL did not contain an authorization code.")
    await _exchange_code(flow, code)
    return flow


async def cancel_flow(flow_id: str) -> None:
    flow = _flows.pop(flow_id, None)
    if not flow:
        return
    flow.status = "cancelled"
    if flow.task:
        flow.task.cancel()
    if flow.cleanup_task and flow.cleanup_task is not asyncio.current_task():
        flow.cleanup_task.cancel()
    if flow.server:
        flow.server.close()
        await flow.server.wait_closed()


async def cancel_provider_flows(provider_id: str) -> None:
    """Stop transient authorization work without touching persisted credentials."""
    for flow_id, flow in list(_flows.items()):
        if flow.provider_id == provider_id:
            await cancel_flow(flow_id)


def delete_oauth_bundle(provider_id: str) -> None:
    """Delete persisted authorization synchronously inside a credential guard."""
    credential_secrets.delete_secret(
        credential_secrets.OPENAI_CODEX_OAUTH_KIND,
        provider_id,
    )


async def disconnect(provider_id: str) -> None:
    """Compatibility wrapper for callers without an external write guard."""
    await cancel_provider_flows(provider_id)
    delete_oauth_bundle(provider_id)


async def resolve_access(provider_id: str) -> tuple[str, str]:
    lock = _refresh_locks.setdefault(provider_id, asyncio.Lock())
    async with lock:
        bundle = load_oauth_bundle(provider_id)
        if not bundle:
            raise CodexAuthError("ChatGPT connection requires authorization.")

        if bundle.get("reauthorization_required"):
            raise CodexAuthError("ChatGPT authorization is no longer valid. Please reconnect.")
        if bundle["expires_at"] > time.time() + _REFRESH_SKEW_SECONDS:
            return bundle["access_token"], bundle["account_id"]

        # Multiple Studio workers may share the installation DB. Take a
        # provider-scoped filesystem lock off the event loop, then re-read so a
        # refresh completed by another worker is reused rather than duplicated.
        lock_name = hashlib.sha256(provider_id.encode()).hexdigest()[:24]
        file_lock = FileLock(
            str(studio_db_path().parent / f".openai-codex-refresh-{lock_name}.lock"),
            timeout = 30,
        )
        try:
            await asyncio.to_thread(file_lock.acquire)
        except FileLockTimeout as exc:
            raise CodexAuthError("ChatGPT credential refresh is busy. Please retry.") from exc
        try:
            bundle = load_oauth_bundle(provider_id)
            if not bundle:
                raise CodexAuthError("ChatGPT connection requires authorization.")
            if bundle["expires_at"] > time.time() + _REFRESH_SKEW_SECONDS:
                return bundle["access_token"], bundle["account_id"]
            try:
                body = await _token_request({
                    "grant_type": "refresh_token",
                    "client_id": OPENAI_CODEX_CLIENT_ID,
                    "refresh_token": bundle["refresh_token"],
                })
            except CodexReauthorizationRequired:
                bundle["reauthorization_required"] = True
                save_oauth_bundle(provider_id, bundle)
                raise
            refreshed = _validate_token_payload(body, bundle["refresh_token"])
            save_oauth_bundle(provider_id, refreshed)
            return refreshed["access_token"], refreshed["account_id"]
        finally:
            await asyncio.to_thread(file_lock.release)
