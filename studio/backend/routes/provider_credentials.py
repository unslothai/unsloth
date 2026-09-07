# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared request-layer resolution for external-provider credentials."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from functools import wraps
from typing import Iterator
from weakref import WeakKeyDictionary

from auth import storage as auth_storage

import structlog
from fastapi import HTTPException


from storage import credential_secrets

logger = structlog.get_logger(__name__)

_provider_config_locks: WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = (
    WeakKeyDictionary()
)


def provider_config_guard(provider_id: str) -> asyncio.Lock:
    """Serialize one provider's routing metadata and installation credential."""
    loop = asyncio.get_running_loop()
    locks = _provider_config_locks.setdefault(loop, {})
    return locks.setdefault(provider_id, asyncio.Lock())


def serialize_provider_config(handler):
    """Keep provider mutations atomic with saved routing/credential snapshots."""

    @wraps(handler)
    async def _serialized(provider_id: str, *args, **kwargs):
        async with provider_config_guard(provider_id):
            return await handler(provider_id, *args, **kwargs)

    _serialized._provider_config_serialized = True
    return _serialized


@contextmanager
def current_credential_write(credential: tuple[str, str | None]) -> Iterator[None]:
    """Reject a credential-derived write if password rotation revoked its request."""
    subject, generation = credential
    try:
        with auth_storage.credential_generation_guard(subject, generation):
            yield
    except auth_storage.CredentialRotated as exc:
        raise HTTPException(status_code = 401, detail = "Invalid or expired token") from exc


def require_ui_session(via_api_key: bool) -> None:
    """Keep installation-owned credentials behind an interactive UI session."""
    if via_api_key:
        raise HTTPException(status_code = 403, detail = "Remote access requires a UI session.")


def resolve_provider_api_key_or_400(
    provider_id: str | None,
    encrypted_api_key: str | None,
    *,
    allow_saved_key: bool = True,
    prefer_saved_key: bool = False,
) -> str:
    """Resolve an explicit key, or a saved key only for an interactive UI session."""

    try:
        saved_provider_id = provider_id if allow_saved_key else None
        if prefer_saved_key and saved_provider_id:
            saved_key = credential_secrets.get_provider_api_key(saved_provider_id)
            if saved_key is not None:
                return saved_key
        return credential_secrets.resolve_provider_api_key(saved_provider_id, encrypted_api_key)
    except Exception as exc:
        logger.warning(
            "external_provider.api_key_decrypt_failed",
            error_type = type(exc).__name__,
        )
        raise HTTPException(
            status_code = 400,
            detail = (
                "Failed to decrypt API key. The server public key may have changed — "
                "try refreshing the page."
            ),
        ) from exc
