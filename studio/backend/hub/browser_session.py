# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from fastapi import HTTPException, Request


async def signed_in(request: Request) -> bool:
    """Whether the request carries a valid Unsloth session, for routes that also serve anonymous loads."""
    from fastapi.security import HTTPAuthorizationCredentials

    from auth.authentication import get_current_subject

    scheme, _, token = (request.headers.get("authorization") or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        return False
    try:
        await get_current_subject(HTTPAuthorizationCredentials(scheme = "Bearer", credentials = token))
    except HTTPException:
        return False
    return True
