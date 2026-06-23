# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
OpenAI-compatible audio endpoints.
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from auth.authentication import get_current_subject
from loggers import get_logger

router = APIRouter()
logger = get_logger(__name__)


class SpeechRequest(BaseModel):
    input: str
    voice: Optional[str] = None


@router.post("/speech")
async def create_speech(
    payload: SpeechRequest, current_subject: str = Depends(get_current_subject)
):
    """
    Generate speech from text using the currently loaded TTS model.
    Returns raw WAV audio bytes (Content-Type: audio/wav).

    Mirrors the OpenAI POST /v1/audio/speech interface; `voice` is accepted
    but ignored until per-voice routing is implemented.
    """
    from routes.inference import get_llama_cpp_backend, get_voice_llama_backend
    from core.inference import get_inference_backend

    text = payload.input.strip()
    if not text:
        raise HTTPException(status_code = 400, detail = "input must not be empty.")

    # Priority: voice slot → main llama slot → transformers backend
    voice_backend = get_voice_llama_backend()
    llama_backend = get_llama_cpp_backend()

    if voice_backend.is_loaded and getattr(voice_backend, "_is_audio", False):
        gen = lambda: voice_backend.generate_audio_response(
            text = text,
            audio_type = voice_backend._audio_type,
        )
    elif llama_backend.is_loaded and getattr(llama_backend, "_is_audio", False):
        gen = lambda: llama_backend.generate_audio_response(
            text = text,
            audio_type = llama_backend._audio_type,
        )
    else:
        backend = get_inference_backend()
        if not backend.active_model_name:
            raise HTTPException(status_code = 400, detail = "No model loaded.")
        model_info = backend.models.get(backend.active_model_name, {})
        if not model_info.get("is_audio"):
            raise HTTPException(status_code = 400, detail = "Active model is not a TTS model.")
        gen = lambda: backend.generate_audio_response(text = text)

    try:
        wav_bytes, _ = await asyncio.get_event_loop().run_in_executor(None, gen)
    except Exception as e:
        logger.error("Speech generation error: %s", e, exc_info = True)
        raise HTTPException(status_code = 500, detail = str(e))

    return Response(content = wav_bytes, media_type = "audio/wav")
