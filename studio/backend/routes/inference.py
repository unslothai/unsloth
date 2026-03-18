# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference API routes for model loading and text generation.
"""

import sys
import time
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional
import json
import structlog
from loggers import get_logger
import asyncio
import threading


# Add backend directory to path
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import backend functions
try:
    from core.inference import get_inference_backend
    from core.inference.llama_cpp import LlamaCppBackend
    from utils.models import ModelConfig
    from utils.inference import load_inference_config
    from utils.models.model_config import load_model_defaults
except ImportError:
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from core.inference import get_inference_backend
    from core.inference.llama_cpp import LlamaCppBackend
    from utils.models import ModelConfig
    from utils.inference import load_inference_config
    from utils.models.model_config import load_model_defaults

from models.inference import (
    LoadRequest,
    UnloadRequest,
    GenerateRequest,
    LoadResponse,
    UnloadResponse,
    InferenceStatusResponse,
    ChatCompletionRequest,
    ChatCompletionChunk,
    ChatCompletion,
    ChunkChoice,
    ChoiceDelta,
    CompletionChoice,
    CompletionMessage,
    ValidateModelRequest,
    ValidateModelResponse,
)
from auth.authentication import get_current_subject

import io
import wave
import base64
import numpy as np

router = APIRouter()
logger = get_logger(__name__)


# GGUF inference backend (llama-server)
_llama_cpp_backend = LlamaCppBackend()


def get_llama_cpp_backend() -> LlamaCppBackend:
    return _llama_cpp_backend


@router.post("/load", response_model = LoadResponse)
async def load_model(
    request: LoadRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Load a model for inference.

    The model_path should be a clean identifier from GET /models/list.
    Returns inference configuration parameters (temperature, top_p, top_k, min_p)
    from the model's YAML config, falling back to default.yaml for missing values.

    GGUF models are loaded via llama-server (llama.cpp) instead of Unsloth.
    """
    try:
        # Version switching is handled automatically by the subprocess-based
        # inference backend — no need for ensure_transformers_version() here.

        # ── Already-loaded check: skip reload if the exact model is active ──
        backend = get_inference_backend()
        llama_backend = get_llama_cpp_backend()

        if request.gguf_variant:
            if (
                llama_backend.is_loaded
                and llama_backend.hf_variant
                and llama_backend.hf_variant.lower() == request.gguf_variant.lower()
                and llama_backend.model_identifier
                and llama_backend.model_identifier.lower() == request.model_path.lower()
            ):
                logger.info(
                    f"Model already loaded (GGUF): {request.model_path} variant={request.gguf_variant}, skipping reload"
                )
                inference_config = load_inference_config(llama_backend.model_identifier)
                from utils.models import is_audio_input_type

                _gguf_audio = (
                    llama_backend._audio_type
                    if hasattr(llama_backend, "_audio_type")
                    else None
                )
                _gguf_is_audio = getattr(llama_backend, "_is_audio", False)
                return LoadResponse(
                    status = "already_loaded",
                    model = llama_backend.model_identifier,
                    display_name = llama_backend.model_identifier,
                    is_vision = llama_backend._is_vision,
                    is_lora = False,
                    is_gguf = True,
                    is_audio = _gguf_is_audio,
                    audio_type = _gguf_audio,
                    has_audio_input = is_audio_input_type(_gguf_audio)
                    if _gguf_audio
                    else False,
                    inference = inference_config,
                    context_length = llama_backend.context_length,
                    supports_reasoning = llama_backend.supports_reasoning,
                    chat_template = llama_backend.chat_template,
                )
        else:
            if (
                backend.active_model_name
                and backend.active_model_name.lower() == request.model_path.lower()
            ):
                logger.info(
                    f"Model already loaded (Unsloth): {request.model_path}, skipping reload"
                )
                inference_config = load_inference_config(backend.active_model_name)
                _model_info = backend.models.get(backend.active_model_name, {})
                _chat_template = None
                try:
                    _tpl_info = _model_info.get("chat_template_info", {})
                    _chat_template = _tpl_info.get("template")
                except Exception as e:
                    logger.warning(
                        f"Could not retrieve chat template for {backend.active_model_name}: {e}"
                    )
                return LoadResponse(
                    status = "already_loaded",
                    model = backend.active_model_name,
                    display_name = backend.active_model_name,
                    is_vision = _model_info.get("is_vision", False),
                    is_lora = _model_info.get("is_lora", False),
                    is_gguf = False,
                    is_audio = _model_info.get("is_audio", False),
                    audio_type = _model_info.get("audio_type"),
                    has_audio_input = _model_info.get("has_audio_input", False),
                    inference = inference_config,
                    chat_template = _chat_template,
                )

        # Create config using clean factory method
        # is_lora is auto-detected from adapter_config.json on disk/HF
        config = ModelConfig.from_identifier(
            model_id = request.model_path,
            hf_token = request.hf_token,
            gguf_variant = request.gguf_variant,
        )

        if not config:
            raise HTTPException(
                status_code = 400,
                detail = f"Invalid model identifier: {request.model_path}",
            )

        # ── GGUF path: load via llama-server ──────────────────────
        if config.is_gguf:
            llama_backend = get_llama_cpp_backend()
            unsloth_backend = get_inference_backend()

            # Unload any active Unsloth model first to free VRAM
            if unsloth_backend.active_model_name:
                logger.info(
                    f"Unloading Unsloth model '{unsloth_backend.active_model_name}' before loading GGUF"
                )
                unsloth_backend.unload_model(unsloth_backend.active_model_name)

            # Route to HF mode or local mode based on config
            # Run in a thread so the event loop stays free for progress
            # polling and other requests during the (potentially long)
            # GGUF download + llama-server startup.
            if config.gguf_hf_repo:
                # HF mode: download via huggingface_hub then start llama-server
                success = await asyncio.to_thread(
                    llama_backend.load_model,
                    hf_repo = config.gguf_hf_repo,
                    hf_variant = config.gguf_variant,
                    hf_token = request.hf_token,
                    model_identifier = config.identifier,
                    is_vision = config.is_vision,
                    n_ctx = request.max_seq_length,
                    chat_template_override = request.chat_template_override,
                    cache_type_kv = request.cache_type_kv,
                )
            else:
                # Local mode: llama-server loads via -m <path>
                success = await asyncio.to_thread(
                    llama_backend.load_model,
                    gguf_path = config.gguf_file,
                    mmproj_path = config.gguf_mmproj_file,
                    model_identifier = config.identifier,
                    is_vision = config.is_vision,
                    n_ctx = request.max_seq_length,
                    chat_template_override = request.chat_template_override,
                    cache_type_kv = request.cache_type_kv,
                )

            if not success:
                raise HTTPException(
                    status_code = 500,
                    detail = f"Failed to load GGUF model: {config.display_name}",
                )

            logger.info(f"Loaded GGUF model via llama-server: {config.identifier}")

            # Detect TTS audio by probing the loaded model's vocabulary
            from utils.models import is_audio_input_type

            _gguf_audio = llama_backend.detect_audio_type()
            _gguf_is_audio = _gguf_audio in ("snac", "bicodec", "dac")
            llama_backend._is_audio = _gguf_is_audio
            llama_backend._audio_type = _gguf_audio
            if _gguf_is_audio:
                logger.info(f"GGUF model detected as audio: audio_type={_gguf_audio}")
                await asyncio.to_thread(llama_backend.init_audio_codec, _gguf_audio)

            inference_config = load_inference_config(config.identifier)

            return LoadResponse(
                status = "loaded",
                model = config.identifier,
                display_name = config.display_name,
                is_vision = config.is_vision,
                is_lora = False,
                is_gguf = True,
                is_audio = _gguf_is_audio,
                audio_type = _gguf_audio,
                has_audio_input = is_audio_input_type(_gguf_audio),
                inference = inference_config,
                context_length = llama_backend.context_length,
                supports_reasoning = llama_backend.supports_reasoning,
                supports_tools = llama_backend.supports_tools,
                cache_type_kv = llama_backend.cache_type_kv,
                chat_template = llama_backend.chat_template,
            )

        # ── Standard path: load via Unsloth/transformers ──────────
        backend = get_inference_backend()

        # Unload any active GGUF model first
        llama_backend = get_llama_cpp_backend()
        if llama_backend.is_loaded:
            logger.info("Unloading GGUF model before loading Unsloth model")
            llama_backend.unload_model()

        # Shut down any export subprocess to free VRAM
        try:
            from core.export import get_export_backend

            exp_backend = get_export_backend()
            if exp_backend.current_checkpoint:
                logger.info(
                    "Shutting down export subprocess to free GPU memory for inference"
                )
                exp_backend._shutdown_subprocess()
                exp_backend.current_checkpoint = None
                exp_backend.is_vision = False
                exp_backend.is_peft = False
        except Exception as e:
            logger.warning("Could not shut down export subprocess: %s", e)

        # Auto-detect quantization for LoRA adapters from adapter_config.json
        # The training pipeline patches this file with "unsloth_training_method"
        # which is 'qlora' or 'lora'. Only LoRA (16-bit) needs load_in_4bit=False.
        load_in_4bit = request.load_in_4bit
        if config.is_lora and config.path:
            import json
            from pathlib import Path

            adapter_cfg_path = Path(config.path) / "adapter_config.json"
            if adapter_cfg_path.exists():
                try:
                    with open(adapter_cfg_path) as f:
                        adapter_cfg = json.load(f)
                    training_method = adapter_cfg.get("unsloth_training_method")
                    if training_method == "lora" and load_in_4bit:
                        logger.info(
                            f"adapter_config.json says unsloth_training_method='lora' — "
                            f"setting load_in_4bit=False to match 16-bit training"
                        )
                        load_in_4bit = False
                    elif training_method == "qlora" and not load_in_4bit:
                        logger.info(
                            f"adapter_config.json says unsloth_training_method='qlora' — "
                            f"setting load_in_4bit=True to match QLoRA training"
                        )
                        load_in_4bit = True
                    elif training_method:
                        logger.info(
                            f"Training method: {training_method}, load_in_4bit={load_in_4bit}"
                        )
                    else:
                        # No unsloth_training_method — fallback to base model name
                        if (
                            config.base_model
                            and "-bnb-4bit" not in config.base_model.lower()
                            and load_in_4bit
                        ):
                            logger.info(
                                f"No unsloth_training_method in adapter_config.json. "
                                f"Base model '{config.base_model}' has no -bnb-4bit suffix — "
                                f"setting load_in_4bit=False"
                            )
                            load_in_4bit = False
                except Exception as e:
                    logger.warning(f"Could not read adapter_config.json: {e}")

        # Load the model in a thread so the event loop stays free
        # for download progress polling and other requests.
        success = await asyncio.to_thread(
            backend.load_model,
            config = config,
            max_seq_length = request.max_seq_length,
            load_in_4bit = load_in_4bit,
            hf_token = request.hf_token,
            trust_remote_code = request.trust_remote_code,
        )

        if not success:
            # Check if YAML says this model needs trust_remote_code
            if not request.trust_remote_code:
                model_defaults = load_model_defaults(config.identifier)
                yaml_trust = model_defaults.get("inference", {}).get(
                    "trust_remote_code", False
                )
                if yaml_trust:
                    raise HTTPException(
                        status_code = 400,
                        detail = (
                            f"Model '{config.display_name}' requires trust_remote_code to be enabled. "
                            f"Please enable 'Trust remote code' in Chat Settings and try again."
                        ),
                    )
            raise HTTPException(
                status_code = 500, detail = f"Failed to load model: {config.display_name}"
            )

        logger.info(f"Loaded model: {config.identifier}")

        # Load inference configuration parameters
        inference_config = load_inference_config(config.identifier)

        # Get chat template from tokenizer
        _chat_template = None
        try:
            _model_info = backend.models.get(config.identifier, {})
            _tpl_info = _model_info.get("chat_template_info", {})
            _chat_template = _tpl_info.get("template")
        except Exception:
            pass

        return LoadResponse(
            status = "loaded",
            model = config.identifier,
            display_name = config.display_name,
            is_vision = config.is_vision,
            is_lora = config.is_lora,
            is_gguf = False,
            is_audio = config.is_audio,
            audio_type = config.audio_type,
            has_audio_input = config.has_audio_input,
            inference = inference_config,
            chat_template = _chat_template,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info = True)
        msg = str(e)
        # Surface a friendlier message for models that Unsloth cannot load
        not_supported_hints = [
            "No config file found",
            "not yet supported",
            "is not supported",
            "does not support",
        ]
        if any(h.lower() in msg.lower() for h in not_supported_hints):
            msg = f"This model is not supported yet. Try a different model. (Original error: {msg})"
        raise HTTPException(status_code = 500, detail = f"Failed to load model: {msg}")


@router.post("/validate", response_model = ValidateModelResponse)
async def validate_model(
    request: ValidateModelRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Lightweight validation endpoint for model identifiers.

    This checks that ModelConfig.from_identifier() can resolve the given
    model_path, but it does NOT actually load model weights into GPU memory.
    """
    try:
        config = ModelConfig.from_identifier(
            model_id = request.model_path,
            hf_token = request.hf_token,
            gguf_variant = request.gguf_variant,
        )

        if not config:
            raise HTTPException(
                status_code = 400,
                detail = f"Invalid model identifier: {request.model_path}",
            )

        return ValidateModelResponse(
            valid = True,
            message = "Model identifier is valid.",
            identifier = config.identifier,
            display_name = getattr(config, "display_name", config.identifier),
            is_gguf = getattr(config, "is_gguf", False),
            is_lora = getattr(config, "is_lora", False),
            is_vision = getattr(config, "is_vision", False),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error validating model identifier '{request.model_path}': {e}",
            exc_info = True,
        )
        raise HTTPException(
            status_code = 400,
            detail = f"Invalid model: {str(e)}",
        )


@router.post("/unload", response_model = UnloadResponse)
async def unload_model(
    request: UnloadRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Unload a model from memory.
    Routes to the correct backend (llama-server for GGUF, Unsloth otherwise).
    """
    try:
        # Check if the GGUF backend has this model loaded or is loading it
        llama_backend = get_llama_cpp_backend()
        if llama_backend.is_active and (
            llama_backend.model_identifier == request.model_path
            or not llama_backend.is_loaded
        ):
            llama_backend.unload_model()
            logger.info(f"Unloaded GGUF model: {request.model_path}")
            return UnloadResponse(status = "unloaded", model = request.model_path)

        # Otherwise, unload from Unsloth backend
        backend = get_inference_backend()
        backend.unload_model(request.model_path)
        logger.info(f"Unloaded model: {request.model_path}")
        return UnloadResponse(status = "unloaded", model = request.model_path)

    except Exception as e:
        logger.error(f"Error unloading model: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = f"Failed to unload model: {str(e)}")


@router.post("/generate/stream")
async def generate_stream(
    request: GenerateRequest,
    current_subject: str = Depends(get_current_subject),
):
    """
    Generate a chat response with Server-Sent Events (SSE) streaming.

    For vision models, provide image_base64 with the base64-encoded image.
    """
    backend = get_inference_backend()

    if not backend.active_model_name:
        raise HTTPException(
            status_code = 400, detail = "No model loaded. Call POST /inference/load first."
        )

    # Decode image if provided (for vision models)
    image = None
    if request.image_base64:
        try:
            import base64
            from PIL import Image
            from io import BytesIO

            # Check if current model supports vision
            model_info = backend.models.get(backend.active_model_name, {})
            if not model_info.get("is_vision"):
                raise HTTPException(
                    status_code = 400,
                    detail = "Image provided but current model is text-only. Load a vision model.",
                )

            image_data = base64.b64decode(request.image_base64)
            image = Image.open(BytesIO(image_data))
            image = backend.resize_image(image)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code = 400, detail = f"Failed to decode image: {str(e)}"
            )

    async def stream():
        try:
            for chunk in backend.generate_chat_response(
                messages = request.messages,
                system_prompt = request.system_prompt,
                image = image,
                temperature = request.temperature,
                top_p = request.top_p,
                top_k = request.top_k,
                max_new_tokens = request.max_new_tokens,
                repetition_penalty = request.repetition_penalty,
            ):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            backend.reset_generation_state()
            logger.error(f"Error during generation: {e}", exc_info = True)
            yield f"data: {json.dumps({'error': 'An internal error occurred'})}\n\n"

    return StreamingResponse(
        stream(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.get("/status", response_model = InferenceStatusResponse)
async def get_status(
    current_subject: str = Depends(get_current_subject),
):
    """
    Get current inference backend status.
    Reports whichever backend (Unsloth or llama-server) is currently active.
    """
    try:
        llama_backend = get_llama_cpp_backend()

        # If a GGUF model is loaded via llama-server, report that
        if llama_backend.is_loaded:
            _model_id = llama_backend.model_identifier
            _inference_cfg = load_inference_config(_model_id) if _model_id else None
            return InferenceStatusResponse(
                active_model = _model_id,
                is_vision = llama_backend.is_vision,
                is_gguf = True,
                gguf_variant = llama_backend.hf_variant,
                is_audio = getattr(llama_backend, "_is_audio", False),
                audio_type = getattr(llama_backend, "_audio_type", None),
                loading = [],
                loaded = [_model_id],
                inference = _inference_cfg,
                supports_reasoning = llama_backend.supports_reasoning,
                supports_tools = llama_backend.supports_tools,
                context_length = llama_backend.context_length,
            )

        # Otherwise, report Unsloth backend status
        backend = get_inference_backend()

        is_vision = False
        is_audio = False
        audio_type = None
        has_audio_input = False
        if backend.active_model_name:
            model_info = backend.models.get(backend.active_model_name, {})
            is_vision = model_info.get("is_vision", False)
            is_audio = model_info.get("is_audio", False)
            audio_type = model_info.get("audio_type")
            has_audio_input = model_info.get("has_audio_input", False)

        # gpt-oss safetensors models support reasoning via harmony channels
        supports_reasoning = False
        if backend.active_model_name and hasattr(backend, "_is_gpt_oss_model"):
            supports_reasoning = backend._is_gpt_oss_model()

        return InferenceStatusResponse(
            active_model = backend.active_model_name,
            is_vision = is_vision,
            is_gguf = False,
            is_audio = is_audio,
            audio_type = audio_type,
            has_audio_input = has_audio_input,
            loading = list(getattr(backend, "loading_models", set())),
            loaded = list(backend.models.keys()),
            supports_reasoning = supports_reasoning,
        )

    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = f"Failed to get status: {str(e)}")


# =====================================================================
# Audio (TTS) Generation  (/audio/generate)
# =====================================================================


@router.post("/audio/generate")
async def generate_audio(
    payload: ChatCompletionRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    Generate audio (TTS) from the latest user message.
    Returns a JSON response with base64-encoded WAV audio.
    Works with both GGUF (llama-server) and Unsloth/transformers backends.
    """
    import base64

    # Extract text from the last user message
    _, chat_messages, _ = _extract_content_parts(payload.messages)
    if not chat_messages:
        raise HTTPException(status_code = 400, detail = "No messages provided.")
    last_user_msg = next(
        (m for m in reversed(chat_messages) if m["role"] == "user"), None
    )
    if not last_user_msg:
        raise HTTPException(status_code = 400, detail = "No user message found.")
    text = last_user_msg["content"]

    # Pick backend — both return (wav_bytes, sample_rate)
    llama_backend = get_llama_cpp_backend()
    if llama_backend.is_loaded and getattr(llama_backend, "_is_audio", False):
        model_name = llama_backend.model_identifier
        gen = lambda: llama_backend.generate_audio_response(
            text = text,
            audio_type = llama_backend._audio_type,
            temperature = payload.temperature,
            top_p = payload.top_p,
            top_k = payload.top_k,
            min_p = payload.min_p,
            max_new_tokens = payload.max_tokens or 2048,
            repetition_penalty = payload.repetition_penalty,
        )
    else:
        backend = get_inference_backend()
        if not backend.active_model_name:
            raise HTTPException(status_code = 400, detail = "No model loaded.")
        model_info = backend.models.get(backend.active_model_name, {})
        if not model_info.get("is_audio"):
            raise HTTPException(
                status_code = 400, detail = "Active model is not an audio model."
            )
        model_name = backend.active_model_name
        gen = lambda: backend.generate_audio_response(
            text = text,
            temperature = payload.temperature,
            top_p = payload.top_p,
            top_k = payload.top_k,
            min_p = payload.min_p,
            max_new_tokens = payload.max_tokens or 2048,
            repetition_penalty = payload.repetition_penalty,
            use_adapter = payload.use_adapter,
        )

    try:
        wav_bytes, sample_rate = await asyncio.get_event_loop().run_in_executor(
            None, gen
        )
    except Exception as e:
        logger.error(f"Audio generation error: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = str(e))

    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    return JSONResponse(
        content = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion.audio",
            "model": model_name,
            "audio": {"data": audio_b64, "format": "wav", "sample_rate": sample_rate},
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f'[Generated audio from: "{text[:100]}"]',
                    },
                    "finish_reason": "stop",
                }
            ],
        }
    )


# =====================================================================
# OpenAI-Compatible Chat Completions  (/chat/completions)
# =====================================================================


def _decode_audio_base64(b64: str) -> np.ndarray:
    """Decode base64 audio (any format) → float32 numpy array at 16kHz."""
    import torch
    import torchaudio
    import tempfile
    import os
    from utils.paths import ensure_dir, tmp_root

    raw = base64.b64decode(b64)
    # torchaudio.load needs a file path or file-like object with format hint
    # Write to a temp file so torchaudio can auto-detect the format
    with tempfile.NamedTemporaryFile(
        suffix = ".audio",
        delete = False,
        dir = str(ensure_dir(tmp_root())),
    ) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        waveform, sr = torchaudio.load(tmp_path)
    finally:
        os.unlink(tmp_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim = 0, keepdim = True)

    # Resample to 16kHz if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq = sr, new_freq = 16000)
        waveform = resampler(waveform)

    return waveform.squeeze(0).numpy()


def _extract_content_parts(
    messages: list,
) -> tuple[str, list[dict], "Optional[str]"]:
    """
    Parse OpenAI-format messages into components the inference backend expects.

    Handles both plain-string ``content`` and multimodal content-part arrays
    (``[{type: "text", ...}, {type: "image_url", ...}]``).

    Returns:
        system_prompt:  The system message text (empty string if none provided).
        chat_messages:  Non-system messages with content flattened to strings.
        image_base64:   Base64 data of the *first* image found, or ``None``.
    """
    system_prompt = ""
    chat_messages: list[dict] = []
    first_image_b64: Optional[str] = None

    for msg in messages:
        # ── System messages → extract as system_prompt ────────
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_prompt = msg.content
            elif isinstance(msg.content, list):
                # Unlikely but handle: join text parts
                system_prompt = "\n".join(
                    p.text for p in msg.content if p.type == "text"
                )
            continue

        # ── User / assistant messages ─────────────────────────
        if isinstance(msg.content, str):
            # Plain string content — pass through
            chat_messages.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg.content, list):
            # Multimodal content parts
            text_parts: list[str] = []
            for part in msg.content:
                if part.type == "text":
                    text_parts.append(part.text)
                elif part.type == "image_url" and first_image_b64 is None:
                    url = part.image_url.url
                    if url.startswith("data:"):
                        # data:image/png;base64,<DATA> → extract <DATA>
                        first_image_b64 = url.split(",", 1)[1] if "," in url else None
                    else:
                        logger.warning(
                            f"Remote image URLs not yet supported: {url[:80]}..."
                        )
            combined_text = "\n".join(text_parts) if text_parts else ""
            chat_messages.append({"role": msg.role, "content": combined_text})

    return system_prompt, chat_messages, first_image_b64


@router.post("/chat/completions")
async def openai_chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    OpenAI-compatible chat completions endpoint.

    Supports multimodal messages: ``content`` may be a plain string or a
    list of content parts (``text`` / ``image_url``).

    Streaming (default):  returns SSE chunks matching OpenAI's format.
    Non-streaming:        returns a single ChatCompletion JSON object.

    Automatically routes to the correct backend:
    - GGUF models → llama-server via LlamaCppBackend
    - Other models → Unsloth/transformers via InferenceBackend
    """
    llama_backend = get_llama_cpp_backend()
    using_gguf = llama_backend.is_loaded

    # ── Determine which backend is active ─────────────────────
    if using_gguf:
        model_name = llama_backend.model_identifier or payload.model
        if getattr(llama_backend, "_is_audio", False):
            return await generate_audio(payload, request)
    else:
        backend = get_inference_backend()
        if not backend.active_model_name:
            raise HTTPException(
                status_code = 400,
                detail = "No model loaded. Call POST /inference/load first.",
            )
        model_name = backend.active_model_name or payload.model

        # ── Audio TTS path: auto-route to audio generation ────
        # (Whisper is ASR not TTS — handled below in audio input path)
        model_info = backend.models.get(backend.active_model_name, {})
        if model_info.get("is_audio") and model_info.get("audio_type") != "whisper":
            return await generate_audio(payload, request)

        # ── Whisper without audio: return clear error ──
        if model_info.get("audio_type") == "whisper" and not payload.audio_base64:
            raise HTTPException(
                status_code = 400,
                detail = "Whisper models require audio input. Please upload an audio file.",
            )

        # ── Audio INPUT path: decode WAV and route to audio input generation ──
        if payload.audio_base64 and model_info.get("has_audio_input"):
            audio_array = _decode_audio_base64(payload.audio_base64)
            system_prompt, chat_messages, _ = _extract_content_parts(payload.messages)
            cancel_event = threading.Event()
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            def audio_input_generate():
                if model_info.get("audio_type") == "whisper":
                    return backend.generate_whisper_response(
                        audio_array = audio_array,
                        cancel_event = cancel_event,
                    )
                return backend.generate_audio_input_response(
                    messages = chat_messages,
                    system_prompt = system_prompt,
                    audio_array = audio_array,
                    temperature = payload.temperature,
                    top_p = payload.top_p,
                    top_k = payload.top_k,
                    min_p = payload.min_p,
                    max_new_tokens = payload.max_tokens or 2048,
                    repetition_penalty = payload.repetition_penalty,
                    cancel_event = cancel_event,
                )

            if payload.stream:

                async def audio_input_stream():
                    try:
                        first_chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [
                                ChunkChoice(
                                    delta = ChoiceDelta(role = "assistant"),
                                    finish_reason = None,
                                )
                            ],
                        )
                        yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

                        for chunk_text in audio_input_generate():
                            if await request.is_disconnected():
                                cancel_event.set()
                                return
                            if chunk_text:
                                chunk = ChatCompletionChunk(
                                    id = completion_id,
                                    created = created,
                                    model = model_name,
                                    choices = [
                                        ChunkChoice(
                                            delta = ChoiceDelta(content = chunk_text),
                                            finish_reason = None,
                                        )
                                    ],
                                )
                                yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

                        final_chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [
                                ChunkChoice(delta = ChoiceDelta(), finish_reason = "stop")
                            ],
                        )
                        yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                        yield "data: [DONE]\n\n"
                    except asyncio.CancelledError:
                        cancel_event.set()
                        raise
                    except Exception as e:
                        logger.error(
                            f"Error during audio input streaming: {e}", exc_info = True
                        )
                        yield f"data: {json.dumps({'error': {'message': 'An internal error occurred', 'type': 'server_error'}})}\n\n"

                return StreamingResponse(
                    audio_input_stream(),
                    media_type = "text/event-stream",
                    headers = {
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                full_text = "".join(audio_input_generate())
                response = ChatCompletion(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        CompletionChoice(
                            message = CompletionMessage(content = full_text),
                            finish_reason = "stop",
                        )
                    ],
                )
                return JSONResponse(content = response.model_dump())

    # ── Parse messages (handles multimodal content parts) ─────
    system_prompt, chat_messages, extracted_image_b64 = _extract_content_parts(
        payload.messages
    )

    if not chat_messages:
        raise HTTPException(
            status_code = 400,
            detail = "At least one non-system message is required.",
        )

    # ── GGUF path: proxy to llama-server /v1/chat/completions ──
    if using_gguf:
        # Reject images if this GGUF model doesn't support vision
        image_b64 = extracted_image_b64 or payload.image_base64
        if image_b64 and not llama_backend.is_vision:
            raise HTTPException(
                status_code = 400,
                detail = "Image provided but current GGUF model does not support vision.",
            )

        # Convert image to PNG for llama-server (stb_image has limited format support)
        if image_b64:
            try:
                import base64 as _b64
                from io import BytesIO as _BytesIO
                from PIL import Image as _Image

                raw = _b64.b64decode(image_b64)
                img = _Image.open(_BytesIO(raw))
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                buf = _BytesIO()
                img.save(buf, format = "PNG")
                image_b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
            except Exception as e:
                raise HTTPException(
                    status_code = 400, detail = f"Failed to process image: {e}"
                )

        # Build message list with system prompt prepended
        gguf_messages = []
        if system_prompt:
            gguf_messages.append({"role": "system", "content": system_prompt})
        gguf_messages.extend(chat_messages)

        cancel_event = threading.Event()

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # ── Tool-calling path (agentic loop) ──────────────────
        use_tools = (
            payload.enable_tools and llama_backend.supports_tools and not image_b64
        )

        if use_tools:
            from core.inference.tools import ALL_TOOLS

            if payload.enabled_tools is not None:
                tools_to_use = [
                    t
                    for t in ALL_TOOLS
                    if t["function"]["name"] in payload.enabled_tools
                ]
            else:
                tools_to_use = ALL_TOOLS

            def gguf_generate_with_tools():
                return llama_backend.generate_chat_completion_with_tools(
                    messages = gguf_messages,
                    tools = tools_to_use,
                    temperature = payload.temperature,
                    top_p = payload.top_p,
                    top_k = payload.top_k,
                    min_p = payload.min_p,
                    max_tokens = payload.max_tokens,
                    repetition_penalty = payload.repetition_penalty,
                    presence_penalty = payload.presence_penalty,
                    cancel_event = cancel_event,
                    enable_thinking = payload.enable_thinking,
                    auto_heal_tool_calls = payload.auto_heal_tool_calls
                    if payload.auto_heal_tool_calls is not None
                    else True,
                    max_tool_iterations = payload.max_tool_calls_per_message
                    if payload.max_tool_calls_per_message is not None
                    else 10,
                    tool_call_timeout = payload.tool_call_timeout
                    if payload.tool_call_timeout is not None
                    else 300,
                    session_id = payload.session_id,
                )

            _tool_sentinel = object()

            async def gguf_tool_stream():
                try:
                    first_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(role = "assistant"),
                                finish_reason = None,
                            )
                        ],
                    )
                    yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

                    # Iterate the synchronous generator in a thread so
                    # the event loop stays free for disconnect detection.
                    gen = gguf_generate_with_tools()
                    prev_text = ""
                    while True:
                        if await request.is_disconnected():
                            cancel_event.set()
                            return

                        event = await asyncio.to_thread(next, gen, _tool_sentinel)
                        if event is _tool_sentinel:
                            break

                        if event["type"] == "status":
                            # Emit tool status as a custom SSE event
                            status_data = json.dumps(
                                {
                                    "type": "tool_status",
                                    "content": event["text"],
                                }
                            )
                            yield f"data: {status_data}\n\n"
                            continue

                        if event["type"] in ("tool_start", "tool_end"):
                            yield f"data: {json.dumps(event)}\n\n"
                            continue

                        # "content" type -- cumulative text
                        cumulative = event.get("text", "")
                        new_text = cumulative[len(prev_text) :]
                        prev_text = cumulative
                        if not new_text:
                            continue
                        chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [
                                ChunkChoice(
                                    delta = ChoiceDelta(content = new_text),
                                    finish_reason = None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

                    final_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(),
                                finish_reason = "stop",
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                    yield "data: [DONE]\n\n"

                except asyncio.CancelledError:
                    cancel_event.set()
                    raise
                except Exception as e:
                    import traceback

                    tb = traceback.format_exc()
                    logger.error(f"Error during GGUF tool streaming: {e}\n{tb}")
                    error_chunk = {
                        "error": {
                            "message": "An internal error occurred",
                            "type": "server_error",
                        },
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                gguf_tool_stream(),
                media_type = "text/event-stream",
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # ── Standard GGUF path (no tools) ─────────────────────

        def gguf_generate():
            return llama_backend.generate_chat_completion(
                messages = gguf_messages,
                image_b64 = image_b64,
                temperature = payload.temperature,
                top_p = payload.top_p,
                top_k = payload.top_k,
                min_p = payload.min_p,
                max_tokens = payload.max_tokens,
                repetition_penalty = payload.repetition_penalty,
                presence_penalty = payload.presence_penalty,
                cancel_event = cancel_event,
                enable_thinking = payload.enable_thinking,
            )

        _gguf_sentinel = object()

        if payload.stream:

            async def gguf_stream_chunks():
                try:
                    # First chunk: role
                    first_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(role = "assistant"),
                                finish_reason = None,
                            )
                        ],
                    )
                    yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

                    # Iterate the synchronous generator in a thread so
                    # the event loop stays free for disconnect detection.
                    gen = gguf_generate()
                    prev_text = ""
                    while True:
                        if await request.is_disconnected():
                            cancel_event.set()
                            return
                        cumulative = await asyncio.to_thread(next, gen, _gguf_sentinel)
                        if cumulative is _gguf_sentinel:
                            break
                        new_text = cumulative[len(prev_text) :]
                        prev_text = cumulative
                        if not new_text:
                            continue
                        chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [
                                ChunkChoice(
                                    delta = ChoiceDelta(content = new_text),
                                    finish_reason = None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

                    # Final chunk
                    final_chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(),
                                finish_reason = "stop",
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                    yield "data: [DONE]\n\n"

                except asyncio.CancelledError:
                    cancel_event.set()
                    raise
                except Exception as e:
                    logger.error(f"Error during GGUF streaming: {e}", exc_info = True)
                    error_chunk = {
                        "error": {
                            "message": "An internal error occurred",
                            "type": "server_error",
                        },
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                gguf_stream_chunks(),
                media_type = "text/event-stream",
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            try:
                full_text = ""
                for token in gguf_generate():
                    full_text = token

                response = ChatCompletion(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        CompletionChoice(
                            message = CompletionMessage(content = full_text),
                            finish_reason = "stop",
                        )
                    ],
                )
                return JSONResponse(content = response.model_dump())

            except Exception as e:
                logger.error(f"Error during GGUF completion: {e}", exc_info = True)
                raise HTTPException(status_code = 500, detail = str(e))

    # ── Standard Unsloth path ─────────────────────────────────

    # Decode image (from content parts OR legacy field)
    image_b64 = extracted_image_b64 or payload.image_base64
    image = None

    if image_b64:
        try:
            import base64
            from PIL import Image
            from io import BytesIO

            model_info = backend.models.get(backend.active_model_name, {})
            if not model_info.get("is_vision"):
                raise HTTPException(
                    status_code = 400,
                    detail = "Image provided but current model is text-only. Load a vision model.",
                )

            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data))
            image = backend.resize_image(image)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code = 400, detail = f"Failed to decode image: {e}")

    # Shared generation kwargs
    gen_kwargs = dict(
        messages = chat_messages,
        system_prompt = system_prompt,
        image = image,
        temperature = payload.temperature,
        top_p = payload.top_p,
        top_k = payload.top_k,
        min_p = payload.min_p,
        max_new_tokens = payload.max_tokens or 2048,
        repetition_penalty = payload.repetition_penalty,
    )

    # Choose generation path (adapter-controlled or standard)
    cancel_event = threading.Event()

    if payload.use_adapter is not None:

        def generate():
            return backend.generate_with_adapter_control(
                use_adapter = payload.use_adapter,
                cancel_event = cancel_event,
                **gen_kwargs,
            )
    else:

        def generate():
            return backend.generate_chat_response(
                cancel_event = cancel_event, **gen_kwargs
            )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # ── Streaming response ────────────────────────────────────────
    if payload.stream:

        async def stream_chunks():
            try:
                first_chunk = ChatCompletionChunk(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        ChunkChoice(
                            delta = ChoiceDelta(role = "assistant"),
                            finish_reason = None,
                        )
                    ],
                )
                yield f"data: {first_chunk.model_dump_json(exclude_none = True)}\n\n"

                prev_text = ""
                # Run sync generator in thread pool to avoid blocking
                # the event loop. Critical for compare mode: two SSE
                # requests arrive concurrently but the orchestrator
                # serializes them via _gen_lock. Without run_in_executor
                # the second request's blocking lock acquisition would
                # freeze the entire event loop, stalling both streams.
                _DONE = object()  # sentinel for generator exhaustion
                loop = asyncio.get_event_loop()
                gen = generate()
                while True:
                    # next(gen, _DONE) returns _DONE instead of raising
                    # StopIteration — StopIteration cannot propagate
                    # through asyncio futures (Python limitation).
                    cumulative = await loop.run_in_executor(None, next, gen, _DONE)
                    if cumulative is _DONE:
                        break
                    if await request.is_disconnected():
                        cancel_event.set()
                        backend.reset_generation_state()
                        return
                    new_text = cumulative[len(prev_text) :]
                    prev_text = cumulative
                    if not new_text:
                        continue
                    chunk = ChatCompletionChunk(
                        id = completion_id,
                        created = created,
                        model = model_name,
                        choices = [
                            ChunkChoice(
                                delta = ChoiceDelta(content = new_text),
                                finish_reason = None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

                final_chunk = ChatCompletionChunk(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        ChunkChoice(
                            delta = ChoiceDelta(),
                            finish_reason = "stop",
                        )
                    ],
                )
                yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                yield "data: [DONE]\n\n"

            except asyncio.CancelledError:
                cancel_event.set()
                backend.reset_generation_state()
                raise
            except Exception as e:
                backend.reset_generation_state()
                logger.error(f"Error during OpenAI streaming: {e}", exc_info = True)
                error_chunk = {
                    "error": {
                        "message": "An internal error occurred",
                        "type": "server_error",
                    },
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(
            stream_chunks(),
            media_type = "text/event-stream",
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming response ────────────────────────────────────
    else:
        try:
            full_text = ""
            for token in generate():
                full_text = token

            response = ChatCompletion(
                id = completion_id,
                created = created,
                model = model_name,
                choices = [
                    CompletionChoice(
                        message = CompletionMessage(content = full_text),
                        finish_reason = "stop",
                    )
                ],
            )
            return JSONResponse(content = response.model_dump())

        except Exception as e:
            backend.reset_generation_state()
            logger.error(f"Error during OpenAI completion: {e}", exc_info = True)
            raise HTTPException(status_code = 500, detail = str(e))


# =====================================================================
# OpenAI-Compatible Models Listing  (/models → /v1/models)
# =====================================================================


@router.get("/models")
async def openai_list_models(
    current_subject: str = Depends(get_current_subject),
):
    """
    OpenAI-compatible model listing endpoint.

    Returns the currently loaded model in the format expected by
    OpenAI-compatible clients (``GET /v1/models``).
    """
    models = []

    # Check GGUF backend
    llama_backend = get_llama_cpp_backend()
    if llama_backend.is_loaded:
        models.append(
            {
                "id": llama_backend.model_identifier,
                "object": "model",
                "owned_by": "local",
            }
        )

    # Check Unsloth backend
    backend = get_inference_backend()
    if backend.active_model_name:
        models.append(
            {
                "id": backend.active_model_name,
                "object": "model",
                "owned_by": "local",
            }
        )

    return {"object": "list", "data": models}
