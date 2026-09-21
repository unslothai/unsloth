# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Optional

from pydantic import BaseModel, Field, field_validator

# Mirror the frontend's 64 KiB chat-template contract (per-model-config.ts) at the API boundary:
# MaxBodyMiddleware only caps the whole request body, not this field.
MAX_CHAT_TEMPLATE_BYTES = 65_536


def chat_template_byte_length(value: str) -> Optional[int]:
    """UTF-8 length, or None if the string cannot be encoded at all.

    JSON can carry an unpaired surrogate, as a truncated emoji paste produces.
    json decodes it fine and .encode("utf-8") then raises. Callers treat None as
    "reject": such a template can never render.
    """
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError:
        return None


class ValidateChatTemplateRequest(BaseModel):
    template: str = Field(default = "")

    @field_validator("template")
    @classmethod
    def _enforce_template_size(cls, value: str) -> str:
        size = chat_template_byte_length(value)
        if size is None:
            raise ValueError("Chat template contains unpaired surrogate characters.")
        if size > MAX_CHAT_TEMPLATE_BYTES:
            raise ValueError(f"Chat template exceeds the {MAX_CHAT_TEMPLATE_BYTES}-byte limit.")
        return value


class ValidateChatTemplateResponse(BaseModel):
    valid: bool
    error: Optional[str] = None


class ModelTemplateResponse(BaseModel):
    model_name: str
    chat_template: Optional[str] = None
