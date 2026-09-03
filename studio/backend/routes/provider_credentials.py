# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared request-layer resolution for external-provider credentials."""

from __future__ import annotations


from contextlib import contextmanager
from typing import Iterator

from auth import storage as auth_storage

import structlog
from fastapi import HTTPException


from storage import credential_secrets

logger = structlog.get_logger(__name__)


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
) -> str:
    """Resolve an explicit key, or a saved key only for an interactive UI session."""

    try:
        saved_provider_id = provider_id if allow_saved_key else None
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
