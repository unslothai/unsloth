# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Body, Depends, Query

from auth.authentication import get_current_subject
from hub.dependencies import get_request_hf_token
from hub.utils.hf_tokens import HfTokenArg

from ..schemas import (
    MAX_CHAT_TEMPLATE_BYTES,
    ModelTemplateResponse,
    ValidateChatTemplateRequest,
    ValidateChatTemplateResponse,
)
from ..service import read_default_chat_template, validate_chat_template

router = APIRouter()


@router.post("/validate-chat-template", response_model = ValidateChatTemplateResponse)
async def validate_chat_template_route(
    body: ValidateChatTemplateRequest = Body(...),
    current_subject: str = Depends(get_current_subject),
) -> ValidateChatTemplateResponse:
    return await asyncio.to_thread(validate_chat_template, body.template)


@router.get("/chat-template/{model_name:path}", response_model = ModelTemplateResponse)
async def get_default_chat_template_route(
    model_name: str,
    gguf_variant: Optional[str] = Query(None),
    hf_token: HfTokenArg = Depends(get_request_hf_token),
    current_subject: str = Depends(get_current_subject),
) -> ModelTemplateResponse:
    # A cache miss falls through to the hub, and offline that costs one retry backoff per candidate template file.
    from core.inference.llama_cpp import _hf_offline_if_unreachable_for

    def _read():
        with _hf_offline_if_unreachable_for(model_name):
            return read_default_chat_template(model_name, hf_token, gguf_variant)

    template = await asyncio.to_thread(_read)
    if template is not None and len(template.encode("utf-8")) > MAX_CHAT_TEMPLATE_BYTES:
        template = None
    return ModelTemplateResponse(model_name = model_name, chat_template = template)
