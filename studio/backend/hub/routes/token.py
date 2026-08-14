# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hugging Face token validation endpoint."""

from __future__ import annotations

import asyncio
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from hub.dependencies import get_hf_token
from utils.client_ip import client_ip
from utils.hf_token_validation import validate_hf_token


router = APIRouter()


class HfTokenValidationResponse(BaseModel):
    status: Literal["missing", "valid", "invalid", "rate_limited", "unavailable"]
    retry_after_seconds: Optional[int] = None


class RepositoryAccessRequest(BaseModel):
    repo_id: str = Field(min_length = 3, max_length = 192)


class RepositoryAccessResponse(BaseModel):
    status: Literal[
        "ready",
        "authentication_required",
        "invalid_token",
        "not_found",
        "no_write_permission",
        "unavailable",
    ]


def _check_repository_access(repo_id: str, token: str) -> str:
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

    api = HfApi(token = token)
    try:
        identity = api.whoami(token = token)
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code in (401, 403):
            return "invalid_token"
        return "unavailable"

    try:
        api.model_info(repo_id = repo_id, token = token)
    except RepositoryNotFoundError:
        return "not_found"
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return "not_found"
        if exc.response is not None and exc.response.status_code in (401, 403):
            return "no_write_permission"
        return "unavailable"

    token_role = ((identity.get("auth") or {}).get("accessToken") or {}).get("role")
    return "ready" if token_role == "write" else "no_write_permission"


@router.post("/token/validate", response_model = HfTokenValidationResponse)
async def validate_token(
    request: Request,
    hf_token: Optional[str] = Depends(get_hf_token),
    current_subject: str = Depends(get_current_subject),
):
    if not hf_token:
        return HfTokenValidationResponse(status = "missing")
    result = await asyncio.to_thread(
        validate_hf_token,
        hf_token,
        rate_key = f"{current_subject}:{client_ip(request)}",
    )
    return HfTokenValidationResponse(
        status = result.status,
        retry_after_seconds = result.retry_after_seconds,
    )


@router.post("/repository/access", response_model = RepositoryAccessResponse)
async def validate_repository_access(
    payload: RepositoryAccessRequest,
    hf_token: Optional[str] = Depends(get_hf_token),
    _current_subject: str = Depends(get_current_subject),
):
    if not hf_token:
        return RepositoryAccessResponse(status = "authentication_required")
    status = await asyncio.to_thread(_check_repository_access, payload.repo_id, hf_token)
    return RepositoryAccessResponse(status = status)
