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


_ARG_BY_FIELD = {
    "enabled": "enabled",
    "defaultModel": "default_model",
    "defaultGgufVariant": "default_gguf_variant",
    "autoLoad": "auto_load",
    "excludedApps": "excluded_apps",
}


@router.put("/settings")
def put_settings(req: PillSettingsUpdate, current_subject: str = Depends(get_current_subject)):
    # An absent field and an explicit null both arrive as None, so pass on only
    # what the client actually sent: clearing the default model is a null.
    sent = req.model_dump(exclude_unset = True)
    return update_pill_settings(**{_ARG_BY_FIELD[field]: value for field, value in sent.items()})
