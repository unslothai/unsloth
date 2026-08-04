# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hugging Face token validation endpoint."""

from __future__ import annotations

import asyncio
from typing import Literal, Optional

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from auth.authentication import get_current_subject
from hub.dependencies import get_hf_token
from utils.client_ip import client_ip
from utils.hf_token_validation import validate_hf_token


router = APIRouter()


class HfTokenValidationResponse(BaseModel):
    status: Literal["missing", "valid", "invalid", "rate_limited", "unavailable"]
    retry_after_seconds: Optional[int] = None


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
