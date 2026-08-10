# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared request-layer resolution for external-provider credentials."""

from __future__ import annotations

import structlog
from fastapi import HTTPException

from storage import credential_secrets

logger = structlog.get_logger(__name__)


def resolve_provider_api_key_or_400(
    current_subject: str,
    provider_id: str | None,
    encrypted_api_key: str | None,
) -> str:
    """Resolve an explicit or saved provider key with one safe HTTP error contract."""
    try:
        return credential_secrets.resolve_provider_api_key(
            current_subject,
            provider_id,
            encrypted_api_key,
        )
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
