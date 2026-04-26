# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Server-side settings API.

Exposes the SSL configuration so the frontend Settings → Server tab can
read/update it. Settings are stored as rows in the ``app_secrets`` table
and only take effect after the server restarts.
"""

from __future__ import annotations

import asyncio
import os
import ssl
import sys
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, model_validator

from auth import storage
from auth.authentication import get_current_subject
from ssl_config import (
    SSL_CERTFILE_KEY,
    SSL_ENABLED_KEY,
    SSL_KEYFILE_KEY,
    SSL_SELF_SIGNED_KEY,
    ensure_self_signed_cert,
)


router = APIRouter()


class SslSettingsResponse(BaseModel):
    """Saved SSL settings (paths only — never includes cert/key bytes)."""

    ssl_enabled: bool = Field(
        ..., description = "Whether SSL is enabled in saved settings"
    )
    ssl_self_signed: bool = Field(
        ...,
        description = "When true, a self-signed cert is generated/reused at startup",
    )
    ssl_certfile: Optional[str] = Field(
        None, description = "Path to PEM cert (when not self-signed)"
    )
    ssl_keyfile: Optional[str] = Field(
        None, description = "Path to PEM private key (when not self-signed)"
    )
    active_scheme: str = Field(
        ...,
        description = "Scheme the *running* server is currently bound to (http/https)",
    )
    active_port: Optional[int] = Field(
        None, description = "Port the running server is currently bound to"
    )
    active_source: str = Field(
        "default",
        description = (
            "Which precedence layer produced the active config: cli_no_ssl, "
            "cli_paths, cli_self_signed, env_paths, env_self_signed, db_paths, "
            "db_self_signed, or default."
        ),
    )
    restart_supported: bool = Field(
        ...,
        description = "True when the backend can self-restart in place to apply changes",
    )


class SslSettingsUpdate(BaseModel):
    """Payload to update SSL settings."""

    ssl_enabled: bool
    ssl_self_signed: bool = False
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None

    @model_validator(mode = "after")
    def _check_paths(self) -> "SslSettingsUpdate":
        if self.ssl_enabled and not self.ssl_self_signed:
            if not (self.ssl_certfile and self.ssl_keyfile):
                raise ValueError(
                    "ssl_certfile and ssl_keyfile are required when SSL is enabled "
                    "without --ssl-self-signed."
                )
        return self


class SslTestRequest(BaseModel):
    """Payload to validate a cert/key pair before saving."""

    ssl_certfile: str
    ssl_keyfile: str


class SslTestResponse(BaseModel):
    ok: bool
    error: Optional[str] = None


class RestartResponse(BaseModel):
    restarting: bool
    supported: bool


def _restart_supported() -> bool:
    """Self-restart works on every platform we ship today, but expose this
    as a flag so the frontend can degrade gracefully if that changes."""
    return True


@router.get("/server", response_model = SslSettingsResponse)
async def get_server_settings(
    request: Request,
    current_subject: str = Depends(get_current_subject),
) -> SslSettingsResponse:
    """Return saved SSL settings + the scheme the live server is using."""
    enabled_raw = storage.get_app_secret(SSL_ENABLED_KEY)
    self_signed_raw = storage.get_app_secret(SSL_SELF_SIGNED_KEY)
    certfile = storage.get_app_secret(SSL_CERTFILE_KEY)
    keyfile = storage.get_app_secret(SSL_KEYFILE_KEY)

    def _truthy(value: Optional[str]) -> bool:
        return bool(value) and value.strip().lower() in ("1", "true", "yes", "on")

    return SslSettingsResponse(
        ssl_enabled = _truthy(enabled_raw),
        ssl_self_signed = _truthy(self_signed_raw),
        ssl_certfile = certfile or None,
        ssl_keyfile = keyfile or None,
        active_scheme = getattr(request.app.state, "scheme", "http"),
        active_port = getattr(request.app.state, "server_port", None),
        active_source = getattr(request.app.state, "ssl_source", "default"),
        restart_supported = _restart_supported(),
    )


@router.post("/server", response_model = SslSettingsResponse)
async def update_server_settings(
    payload: SslSettingsUpdate,
    request: Request,
    current_subject: str = Depends(get_current_subject),
) -> SslSettingsResponse:
    """Persist SSL settings to ``app_secrets``. Takes effect on next restart."""
    if payload.ssl_enabled and not payload.ssl_self_signed:
        cert_path = Path(payload.ssl_certfile or "").expanduser()
        key_path = Path(payload.ssl_keyfile or "").expanduser()
        if not cert_path.is_file():
            raise HTTPException(
                status_code = status.HTTP_400_BAD_REQUEST,
                detail = f"Certificate file not found: {cert_path}",
            )
        if not key_path.is_file():
            raise HTTPException(
                status_code = status.HTTP_400_BAD_REQUEST,
                detail = f"Key file not found: {key_path}",
            )
        try:
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile = str(cert_path), keyfile = str(key_path))
        except ssl.SSLError as exc:
            raise HTTPException(
                status_code = status.HTTP_400_BAD_REQUEST,
                detail = f"Certificate/key are not a valid pair: {exc}",
            ) from exc

    if payload.ssl_enabled and payload.ssl_self_signed:
        # Pre-generate so the cert exists before the user restarts.
        ensure_self_signed_cert(
            bind_host = getattr(request.app.state, "bind_host", "0.0.0.0")
        )

    storage.set_app_secret(SSL_ENABLED_KEY, "1" if payload.ssl_enabled else "0")
    storage.set_app_secret(SSL_SELF_SIGNED_KEY, "1" if payload.ssl_self_signed else "0")
    if payload.ssl_certfile:
        storage.set_app_secret(SSL_CERTFILE_KEY, payload.ssl_certfile)
    else:
        storage.delete_app_secret(SSL_CERTFILE_KEY)
    if payload.ssl_keyfile:
        storage.set_app_secret(SSL_KEYFILE_KEY, payload.ssl_keyfile)
    else:
        storage.delete_app_secret(SSL_KEYFILE_KEY)

    return await get_server_settings(request, current_subject = current_subject)


@router.post("/server/test", response_model = SslTestResponse)
async def test_certificate(
    payload: SslTestRequest,
    current_subject: str = Depends(get_current_subject),
) -> SslTestResponse:
    """Validate a certfile/keyfile pair without saving anything."""
    cert_path = Path(payload.ssl_certfile).expanduser()
    key_path = Path(payload.ssl_keyfile).expanduser()
    if not cert_path.is_file():
        return SslTestResponse(ok = False, error = f"Cert file not found: {cert_path}")
    if not key_path.is_file():
        return SslTestResponse(ok = False, error = f"Key file not found: {key_path}")
    try:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile = str(cert_path), keyfile = str(key_path))
    except ssl.SSLError as exc:
        return SslTestResponse(ok = False, error = str(exc))
    return SslTestResponse(ok = True)


@router.post("/server/restart", response_model = RestartResponse)
async def restart_server(
    request: Request,
    current_subject: str = Depends(get_current_subject),
) -> RestartResponse:
    """Re-exec the studio process so saved settings take effect.

    The HTTP response is sent first; the actual restart happens on a
    short delay so the client can finish reading the response.
    """
    if not _restart_supported():
        return RestartResponse(restarting = False, supported = False)

    async def _delayed_restart() -> None:
        await asyncio.sleep(0.3)
        from run import trigger_self_restart  # late import to avoid circular load

        try:
            trigger_self_restart()
        except Exception:
            # asyncio drops unhandled task exceptions into a logger few
            # people watch — log it ourselves so a failed restart never
            # looks like a silent no-op.
            import structlog

            structlog.get_logger(__name__).exception(
                "trigger_self_restart raised; server is still running"
            )

    request.app.state._restart_task = asyncio.create_task(_delayed_restart())
    return RestartResponse(restarting = True, supported = True)
