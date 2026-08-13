# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from utils.pill_settings import get_pill_settings, update_pill_settings

router = APIRouter()


class PillSettingsUpdate(BaseModel):
    enabled: Optional[bool] = None
    defaultModel: Optional[str] = Field(None, max_length = 2_000)
    defaultGgufVariant: Optional[str] = Field(None, max_length = 200)
    autoLoad: Optional[bool] = None
    excludedApps: Optional[list[str]] = Field(None, max_length = 500)


@router.get("/settings")
def get_settings(current_subject: str = Depends(get_current_subject)):
    return get_pill_settings()


@router.put("/settings")
def put_settings(
    req: PillSettingsUpdate,
    current_subject: str = Depends(get_current_subject),
):
    return update_pill_settings(
        enabled = req.enabled,
        default_model = req.defaultModel,
        default_gguf_variant = req.defaultGgufVariant,
        auto_load = req.autoLoad,
        excluded_apps = req.excludedApps,
    )
