"""
Inference API routes for model loading and text generation.
"""
import sys
import time
import uuid
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from typing import Optional
import json
import logging
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
except ImportError:
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from core.inference import get_inference_backend
    from core.inference.llama_cpp import LlamaCppBackend
    from utils.models import ModelConfig
    from utils.inference import load_inference_config

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
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Configure logger
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# GGUF inference backend (llama-server)
_llama_cpp_backend = LlamaCppBackend()

def get_llama_cpp_backend() -> LlamaCppBackend:
    return _llama_cpp_backend


@router.post("/load", response_model=LoadResponse)
async def load_model(request: LoadRequest):
    """
    Load a model for inference.

    The model_path should be a clean identifier from GET /models/list.
    Returns inference configuration parameters (temperature, top_p, top_k, min_p)
    from the model's YAML config, falling back to default.yaml for missing values.

    GGUF models are loaded via llama-server (llama.cpp) instead of Unsloth.
    """
    try:
        # Create config using clean factory method
        # is_lora is auto-detected from adapter_config.json on disk/HF
        config = ModelConfig.from_identifier(
            model_id=request.model_path,
            hf_token=request.hf_token,
            gguf_variant=request.gguf_variant,
        )

        if not config:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model identifier: {request.model_path}"
            )

        # ── GGUF path: load via llama-server ──────────────────────
        if config.is_gguf:
            llama_backend = get_llama_cpp_backend()
            unsloth_backend = get_inference_backend()

            # Unload any active Unsloth model first to free VRAM
            if unsloth_backend.active_model_name:
                logger.info(f"Unloading Unsloth model '{unsloth_backend.active_model_name}' before loading GGUF")
                unsloth_backend.unload_model(unsloth_backend.active_model_name)

            # Route to HF mode or local mode based on config
            if config.gguf_hf_repo:
                # HF mode: llama-server downloads via -hf "repo:quant"
                success = llama_backend.load_model(
                    hf_repo=config.gguf_hf_repo,
                    hf_variant=config.gguf_variant,
                    hf_token=request.hf_token,
                    model_identifier=config.identifier,
                    is_vision=config.is_vision,
                    n_ctx=request.max_seq_length,
                )
            else:
                # Local mode: llama-server loads via -m <path>
                success = llama_backend.load_model(
                    gguf_path=config.gguf_file,
                    model_identifier=config.identifier,
                    is_vision=config.is_vision,
                    n_ctx=request.max_seq_length,
                )

            if not success:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to load GGUF model: {config.display_name}"
                )

            logger.info(f"Loaded GGUF model via llama-server: {config.identifier}")

            inference_config = load_inference_config(config.identifier)

            return LoadResponse(
                status="loaded",
                model=config.identifier,
                display_name=config.display_name,
                is_vision=config.is_vision,
                is_lora=False,
                is_gguf=True,
                inference=inference_config,
            )

        # ── Standard path: load via Unsloth/transformers ──────────
        backend = get_inference_backend()

        # Unload any active GGUF model first
        llama_backend = get_llama_cpp_backend()
        if llama_backend.is_loaded:
            logger.info("Unloading GGUF model before loading Unsloth model")
            llama_backend.unload_model()

        success = backend.load_model(
            config=config,
            max_seq_length=request.max_seq_length,
            load_in_4bit=request.load_in_4bit,
            hf_token=request.hf_token,
        )

        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model: {config.display_name}"
            )

        logger.info(f"Loaded model: {config.identifier}")

        # Load inference configuration parameters
        inference_config = load_inference_config(config.identifier)

        return LoadResponse(
            status="loaded",
            model=config.identifier,
            display_name=config.display_name,
            is_vision=config.is_vision,
            is_lora=config.is_lora,
            is_gguf=False,
            inference=inference_config,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load model: {str(e)}"
        )


@router.post("/unload", response_model=UnloadResponse)
async def unload_model(request: UnloadRequest):
    """
    Unload a model from memory.
    Routes to the correct backend (llama-server for GGUF, Unsloth otherwise).
    """
    try:
        # Check if the GGUF backend has this model loaded
        llama_backend = get_llama_cpp_backend()
        if llama_backend.is_loaded and llama_backend.model_identifier == request.model_path:
            llama_backend.unload_model()
            logger.info(f"Unloaded GGUF model: {request.model_path}")
            return UnloadResponse(status="unloaded", model=request.model_path)

        # Otherwise, unload from Unsloth backend
        backend = get_inference_backend()
        backend.unload_model(request.model_path)
        logger.info(f"Unloaded model: {request.model_path}")
        return UnloadResponse(status="unloaded", model=request.model_path)

    except Exception as e:
        logger.error(f"Error unloading model: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to unload model: {str(e)}"
        )


@router.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """
    Generate a chat response with Server-Sent Events (SSE) streaming.
    
    For vision models, provide image_base64 with the base64-encoded image.
    """
    backend = get_inference_backend()
    
    if not backend.active_model_name:
        raise HTTPException(
            status_code=400,
            detail="No model loaded. Call POST /inference/load first."
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
                    status_code=400,
                    detail="Image provided but current model is text-only. Load a vision model."
                )
            
            image_data = base64.b64decode(request.image_base64)
            image = Image.open(BytesIO(image_data))
            image = backend.resize_image(image)
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode image: {str(e)}"
            )
    
    async def stream():
        try:
            for chunk in backend.generate_chat_response(
                messages=request.messages,
                system_prompt=request.system_prompt,
                image=image,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_new_tokens=request.max_new_tokens,
                repetition_penalty=request.repetition_penalty,
            ):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            backend.reset_generation_state()
            logger.error(f"Error during generation: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/status", response_model=InferenceStatusResponse)
async def get_status():
    """
    Get current inference backend status.
    Reports whichever backend (Unsloth or llama-server) is currently active.
    """
    try:
        llama_backend = get_llama_cpp_backend()

        # If a GGUF model is loaded via llama-server, report that
        if llama_backend.is_loaded:
            return InferenceStatusResponse(
                active_model=llama_backend.model_identifier,
                is_vision=llama_backend.is_vision,
                is_gguf=True,
                gguf_variant=llama_backend.hf_variant,
                loading=[],
                loaded=[llama_backend.model_identifier],
            )

        # Otherwise, report Unsloth backend status
        backend = get_inference_backend()

        is_vision = False
        if backend.active_model_name:
            model_info = backend.models.get(backend.active_model_name, {})
            is_vision = model_info.get("is_vision", False)

        return InferenceStatusResponse(
            active_model=backend.active_model_name,
            is_vision=is_vision,
            is_gguf=False,
            loading=list(getattr(backend, 'loading_models', set())),
            loaded=list(backend.models.keys()),
        )

    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get status: {str(e)}"
        )


# =====================================================================
# OpenAI-Compatible Chat Completions  (/chat/completions)
# =====================================================================


def _extract_content_parts(
    messages: list,
) -> tuple[str, list[dict], "Optional[str]"]:
    """
    Parse OpenAI-format messages into components the inference backend expects.

    Handles both plain-string ``content`` and multimodal content-part arrays
    (``[{type: "text", ...}, {type: "image_url", ...}]``).

    Returns:
        system_prompt:  The system message text (or a default).
        chat_messages:  Non-system messages with content flattened to strings.
        image_base64:   Base64 data of the *first* image found, or ``None``.
    """
    system_prompt = "You are a helpful AI assistant."
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
async def openai_chat_completions(payload: ChatCompletionRequest, request: Request):
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
    else:
        backend = get_inference_backend()
        if not backend.active_model_name:
            raise HTTPException(
                status_code=400,
                detail="No model loaded. Call POST /inference/load first.",
            )
        model_name = backend.active_model_name or payload.model

    # ── Parse messages (handles multimodal content parts) ─────
    system_prompt, chat_messages, extracted_image_b64 = _extract_content_parts(
        payload.messages
    )

    if not chat_messages:
        raise HTTPException(
            status_code=400,
            detail="At least one non-system message is required.",
        )

    # ── GGUF path: proxy to llama-server /v1/chat/completions ──
    if using_gguf:
        # Reject images if this GGUF model doesn't support vision
        image_b64 = extracted_image_b64 or payload.image_base64
        if image_b64 and not llama_backend.is_vision:
            raise HTTPException(
                status_code=400,
                detail="Image provided but current GGUF model does not support vision.",
            )

        # Build message list with system prompt prepended
        gguf_messages = []
        if system_prompt:
            gguf_messages.append({"role": "system", "content": system_prompt})
        gguf_messages.extend(chat_messages)

        cancel_event = threading.Event()

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        def gguf_generate():
            return llama_backend.generate_chat_completion(
                messages=gguf_messages,
                image_b64=image_b64,
                temperature=payload.temperature,
                top_p=payload.top_p,
                top_k=payload.top_k,
                min_p=payload.min_p,
                max_tokens=payload.max_tokens or 512,
                repetition_penalty=payload.repetition_penalty,
                cancel_event=cancel_event,
            )

        if payload.stream:
            async def gguf_stream_chunks():
                try:
                    # First chunk: role
                    first_chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_name,
                        choices=[ChunkChoice(
                            delta=ChoiceDelta(role="assistant"),
                            finish_reason=None,
                        )],
                    )
                    yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

                    # Content chunks — llama backend yields cumulative text
                    prev_text = ""
                    for cumulative in gguf_generate():
                        if await request.is_disconnected():
                            cancel_event.set()
                            return
                        new_text = cumulative[len(prev_text):]
                        prev_text = cumulative
                        if not new_text:
                            continue
                        chunk = ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=model_name,
                            choices=[ChunkChoice(
                                delta=ChoiceDelta(content=new_text),
                                finish_reason=None,
                            )],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                    # Final chunk
                    final_chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_name,
                        choices=[ChunkChoice(
                            delta=ChoiceDelta(),
                            finish_reason="stop",
                        )],
                    )
                    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
                    yield "data: [DONE]\n\n"

                except asyncio.CancelledError:
                    cancel_event.set()
                    raise
                except Exception as e:
                    logger.error(f"Error during GGUF streaming: {e}", exc_info=True)
                    error_chunk = {
                        "error": {"message": str(e), "type": "server_error"},
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"

            return StreamingResponse(
                gguf_stream_chunks(),
                media_type="text/event-stream",
                headers={
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
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[CompletionChoice(
                        message=CompletionMessage(content=full_text),
                        finish_reason="stop",
                    )],
                )
                return JSONResponse(content=response.model_dump())

            except Exception as e:
                logger.error(f"Error during GGUF completion: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))

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
                    status_code=400,
                    detail="Image provided but current model is text-only. Load a vision model.",
                )

            image_data = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_data))
            image = backend.resize_image(image)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    # Shared generation kwargs
    gen_kwargs = dict(
        messages=chat_messages,
        system_prompt=system_prompt,
        image=image,
        temperature=payload.temperature,
        top_p=payload.top_p,
        top_k=payload.top_k,
        min_p=payload.min_p,
        max_new_tokens=payload.max_tokens or 512,
        repetition_penalty=payload.repetition_penalty,
    )

    # Choose generation path (adapter-controlled or standard)
    cancel_event = threading.Event()

    if payload.use_adapter is not None:
        def generate():
            return backend.generate_with_adapter_control(
                use_adapter=payload.use_adapter,
                cancel_event=cancel_event,
                **gen_kwargs,
            )
    else:
        def generate():
            return backend.generate_chat_response(cancel_event=cancel_event, **gen_kwargs)

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # ── Streaming response ────────────────────────────────────────
    if payload.stream:
        async def stream_chunks():
            try:
                first_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[ChunkChoice(
                        delta=ChoiceDelta(role="assistant"),
                        finish_reason=None,
                    )],
                )
                yield f"data: {first_chunk.model_dump_json(exclude_none=True)}\n\n"

                prev_text = ""
                for cumulative in generate():
                    if await request.is_disconnected():
                        cancel_event.set()
                        backend.reset_generation_state()
                        return
                    new_text = cumulative[len(prev_text):]
                    prev_text = cumulative
                    if not new_text:
                        continue
                    chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_name,
                        choices=[ChunkChoice(
                            delta=ChoiceDelta(content=new_text),
                            finish_reason=None,
                        )],
                    )
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

                final_chunk = ChatCompletionChunk(
                    id=completion_id,
                    created=created,
                    model=model_name,
                    choices=[ChunkChoice(
                        delta=ChoiceDelta(),
                        finish_reason="stop",
                    )],
                )
                yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n"
                yield "data: [DONE]\n\n"

            except asyncio.CancelledError:
                cancel_event.set()
                backend.reset_generation_state()
                raise
            except Exception as e:
                backend.reset_generation_state()
                logger.error(f"Error during OpenAI streaming: {e}", exc_info=True)
                error_chunk = {
                    "error": {"message": str(e), "type": "server_error"},
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"

        return StreamingResponse(
            stream_chunks(),
            media_type="text/event-stream",
            headers={
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
                id=completion_id,
                created=created,
                model=model_name,
                choices=[CompletionChoice(
                    message=CompletionMessage(content=full_text),
                    finish_reason="stop",
                )],
            )
            return JSONResponse(content=response.model_dump())

        except Exception as e:
            backend.reset_generation_state()
            logger.error(f"Error during OpenAI completion: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
