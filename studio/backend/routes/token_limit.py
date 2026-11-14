# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.inference import token_limit
from routes.provider_credentials import require_ui_session

router = APIRouter()


class TokenLimitUpdate(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    enabled: Optional[StrictBool] = None
    period: Optional[Literal["day", "week", "month"]] = None
    limit: Optional[int] = Field(None, gt = 0, le = token_limit.MAX_LIMIT)
    timezone: Optional[str] = Field(None, max_length = 64)


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
    return await asyncio.to_thread(
        token_limit.update_settings, payload.model_dump(exclude_none = True)
    )


@router.post("/override")
async def override_token_limit(
    current_subject: str = Depends(get_current_subject),
    via_api_key: bool = Depends(authenticated_via_api_key),
) -> dict[str, Any]:
    require_ui_session(via_api_key)
    return await asyncio.to_thread(token_limit.grant_override)
