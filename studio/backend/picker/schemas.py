# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Mirror the frontend's 64 KiB chat-template contract (per-model-config.ts) at
# the API boundary so a direct caller cannot make Jinja parse an oversized
# template. MaxBodyMiddleware only caps the whole request body, not this field.
MAX_CHAT_TEMPLATE_BYTES = 65_536


class ValidateChatTemplateRequest(BaseModel):
    template: str = Field(default = "")

    @field_validator("template")
    @classmethod
    def _enforce_template_size(cls, value: str) -> str:
        if len(value.encode("utf-8")) > MAX_CHAT_TEMPLATE_BYTES:
            raise ValueError(f"Chat template exceeds the {MAX_CHAT_TEMPLATE_BYTES}-byte limit.")
        return value


class ValidateChatTemplateResponse(BaseModel):
    valid: bool
    error: Optional[str] = None


class ModelTemplateResponse(BaseModel):
    model_name: str
    chat_template: Optional[str] = None
