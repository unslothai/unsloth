# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Optional

from pydantic import BaseModel, Field


class ValidateChatTemplateRequest(BaseModel):
    template: str = Field(default = "")


class ValidateChatTemplateResponse(BaseModel):
    valid: bool
    error: Optional[str] = None


class ModelTemplateResponse(BaseModel):
    model_name: str
    chat_template: Optional[str] = None
