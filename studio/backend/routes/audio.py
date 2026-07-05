# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
OpenAI-compatible audio endpoints.
"""

import asyncio
import io
import os
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from auth.authentication import get_current_subject
from loggers import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Default Whisper STT model; override per-request or via UNSLOTH_WHISPER_MODEL.
DEFAULT_WHISPER_MODEL = os.environ.get("UNSLOTH_WHISPER_MODEL", "openai/whisper-base")
WHISPER_SAMPLE_RATE = 16000

# Lazily-built transformers ASR pipeline, cached and reused across requests.
_whisper_pipeline = None
_whisper_pipeline_model = None


def _get_whisper_pipeline(model_id: str):
    """Build (or reuse) a transformers automatic-speech-recognition pipeline.

    Uses the transformers Whisper path (the same family the inference backend
    already loads), so no openai-whisper / ffmpeg dependency is required. Runs on
    GPU when available, else CPU."""
    global _whisper_pipeline, _whisper_pipeline_model
    if _whisper_pipeline is not None and _whisper_pipeline_model == model_id:
        return _whisper_pipeline
    from transformers import pipeline
    import torch

    # transformers.is_torchcodec_available() only checks that the torchcodec
    # package is importable, not that its compiled DLLs actually load; on this
    # ROCm-torch/Windows setup the package is present but libtorchcodec_core*.dll
    # fails to load, so the ASR pipeline's unconditional `import torchcodec`
    # crashes every request. We already hand it a pre-decoded numpy array (see
    # transcribe() below), so torchcodec is never actually needed here — make the
    # pipeline think it's absent so it takes the plain-array path instead.
    import transformers.pipelines.automatic_speech_recognition as _asr_module

    _asr_module.is_torchcodec_available = lambda: False

    device = 0 if torch.cuda.is_available() else -1
    # fp16 on GPU: about 2x faster and half the VRAM vs the default fp32, and
    # Whisper inference is numerically fine in half precision. CPU stays fp32.
    dtype = torch.float16 if device == 0 else torch.float32
    _whisper_pipeline = pipeline(
        "automatic-speech-recognition",
        model = model_id,
        device = device,
        torch_dtype = dtype,
    )
    _whisper_pipeline_model = model_id
    return _whisper_pipeline


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
    from routes.inference import (
        get_llama_cpp_backend,
        get_qwen_tts_backend,
        get_voice_llama_backend,
    )
    from core.inference import get_inference_backend

    text = payload.input.strip()
    if not text:
        raise HTTPException(status_code = 400, detail = "input must not be empty.")

    # Priority: in-process voice slot (Qwen3-TTS) → GGUF voice slot
    # → main llama slot → transformers backend
    qwen_tts_backend = get_qwen_tts_backend()
    voice_backend = get_voice_llama_backend()
    llama_backend = get_llama_cpp_backend()

    # Named speaker for the GGUF voice slot (Orpheus). "default"/blank -> tara.
    voice_name = (payload.voice or "").strip().lower() or "tara"

    if qwen_tts_backend.is_loaded:
        gen = lambda: qwen_tts_backend.generate_audio_response(text = text)
    elif voice_backend.is_loaded and getattr(voice_backend, "_is_audio", False):
        gen = lambda: voice_backend.generate_audio_response(
            text = text,
            audio_type = voice_backend._audio_type,
            voice = voice_name,
        )
    elif llama_backend.is_loaded and getattr(llama_backend, "_is_audio", False):
        gen = lambda: llama_backend.generate_audio_response(
            text = text,
            audio_type = llama_backend._audio_type,
            voice = voice_name,
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
        wav_bytes, _ = await asyncio.to_thread(gen)
    except Exception as e:
        logger.error("Speech generation error: %s", e, exc_info = True)
        raise HTTPException(status_code = 500, detail = str(e))

    return Response(content = wav_bytes, media_type = "audio/wav")


class TranscriptionResponse(BaseModel):
    text: str


@router.post("/transcribe", response_model = TranscriptionResponse)
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_WHISPER_MODEL),
    current_subject: str = Depends(get_current_subject),
):
    """
    Transcribe uploaded audio to text with Whisper (speech-to-text).

    Backend STT alternative to the browser's Web Speech API, so voice mode works
    in browsers/webviews without a cloud speech service (Edge, Brave, the Tauri
    desktop app). Send mono PCM WAV (16 kHz preferred); audio is decoded with
    soundfile (no ffmpeg) and resampled to 16 kHz if needed. Roughly mirrors the
    OpenAI POST /v1/audio/transcriptions interface.
    """
    import numpy as np
    import soundfile as sf

    data = await file.read()
    if not data:
        raise HTTPException(status_code = 400, detail = "Empty audio upload.")

    try:
        audio, sample_rate = sf.read(io.BytesIO(data), dtype = "float32")
    except Exception as e:
        raise HTTPException(
            status_code = 400,
            detail = f"Could not decode audio (send mono PCM WAV): {e}",
        )

    if audio.ndim > 1:  # stereo -> mono
        audio = audio.mean(axis = 1)
    if sample_rate != WHISPER_SAMPLE_RATE:
        import librosa

        audio = librosa.resample(
            audio, orig_sr = sample_rate, target_sr = WHISPER_SAMPLE_RATE
        )
    audio = np.ascontiguousarray(audio, dtype = np.float32)

    if audio.size == 0:
        return TranscriptionResponse(text = "")

    def run() -> str:
        pipe = _get_whisper_pipeline(model)
        result = pipe(audio)
        return (result.get("text") if isinstance(result, dict) else str(result)) or ""

    try:
        text = await asyncio.to_thread(run)
    except Exception as e:
        logger.error("Transcription error: %s", e, exc_info = True)
        raise HTTPException(status_code = 500, detail = str(e))

    return TranscriptionResponse(text = text.strip())


class WarmupRequest(BaseModel):
    model: Optional[str] = None


@router.post("/warmup")
async def warmup(
    payload: Optional[WarmupRequest] = None,
    current_subject: str = Depends(get_current_subject),
):
    """
    Pre-build and exercise the Whisper pipeline on a short silent clip.

    Whisper always processes a fixed 30-second (padded) mel window, so warming on
    any clip tunes the exact conv shapes every later utterance uses. On ROCm that
    moves MIOpen's one-time kernel autotune off the critical path (before the user
    speaks) instead of freezing the first real transcription. Call it when voice
    mode starts with the Whisper STT engine; idempotent and cheap once warmed.
    """
    import numpy as np

    model = ((payload.model if payload else None) or DEFAULT_WHISPER_MODEL).strip()

    def run() -> None:
        pipe = _get_whisper_pipeline(model)
        pipe(np.zeros(WHISPER_SAMPLE_RATE, dtype = np.float32))

    try:
        await asyncio.to_thread(run)
    except Exception as e:
        logger.warning("Whisper warmup failed: %s", e)
        return {"status": "warmup_failed", "detail": str(e)}
    return {"status": "warmed", "model": model}
