# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import token_limit
from routes.provider_credentials import require_ui_session

router = APIRouter()


class TokenLimitUpdate(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    enabled: Optional[StrictBool] = None
    period: Optional[Literal["day", "week", "month"]] = None
    mode: Optional[Literal["combined", "split"]] = None
    method: Optional[Literal["total", "incremental"]] = None
    limit: Optional[int] = Field(None, gt = 0, le = token_limit.MAX_LIMIT)
    localLimit: Optional[int] = Field(None, gt = 0, le = token_limit.MAX_LIMIT)
    apiLimit: Optional[int] = Field(None, gt = 0, le = token_limit.MAX_LIMIT)
    timezone: Optional[str] = Field(None, max_length = 64)
    password: Optional[str] = Field(None, max_length = 256)
    newPassword: Optional[str] = Field(None, max_length = 256)


class TokenLimitOverride(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    pool: Literal["combined", "local", "api"]


@router.get("/status")
async def get_token_limit_status(
    current_subject: str = Depends(get_current_subject),
) -> dict[str, Any]:
    return await asyncio.to_thread(token_limit.status)


@router.put("")
async def update_token_limit(
    payload: TokenLimitUpdate,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> dict[str, Any]:
    require_ui_session(via_api_key)
    patch = payload.model_dump(exclude_none = True, exclude = {"password", "newPassword"})
    try:
        return await asyncio.to_thread(
            token_limit.update_settings,
            patch,
            password = payload.password,
            new_password = payload.newPassword,
        )
    except token_limit.TokenLimitPasswordError:
        raise HTTPException(status_code = 403, detail = "Incorrect token limit password.") from None


@router.post("/override")
async def override_token_limit(
    payload: TokenLimitOverride,
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> dict[str, Any]:
    require_ui_session(via_api_key)
    return await asyncio.to_thread(token_limit.grant_override, payload.pool)
