# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference API routes for model loading and text generation.
"""

import os
import sys
import time
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse, JSONResponse, Response
from typing import Any, Optional, Union
import json
import httpx
import structlog
from loggers import get_logger
import asyncio
import threading


import re as _re

# Model size extraction (shared with core/inference/llama_cpp.py)
from utils.models import extract_model_size_b as _extract_model_size_b


def _install_httpcore_asyncgen_silencer() -> None:
    """Silence benign httpx/httpcore asyncgen GC noise on Python 3.13.

    When Studio proxies a streaming response from llama-server via httpx,
    the innermost ``HTTP11ConnectionByteStream.__aiter__`` async generator
    is finalised by Python's asyncgen GC hook on a task different from the
    one that opened it. Its ``aclose`` path then calls
    ``anyio.Lock.acquire`` → ``cancel_shielded_checkpoint`` which enters a
    ``CancelScope`` on the finaliser task — Python 3.13 flags the
    cross-task exit as ``"Attempted to exit cancel scope in a different
    task"`` and prints ``"async generator ignored GeneratorExit"`` as an
    unraisable warning.

    This is a known httpx + httpcore + anyio interaction (see MCP SDK
    python-sdk#831, agno #3556, chainlit #2361, langchain-mcp-adapters
    #254). It is benign: the response has already been delivered with a
    200. The streaming pass-throughs (``/v1/chat/completions``,
    ``/v1/messages``, ``/v1/responses``, ``/v1/completions``) already
    manage their httpx lifecycle inside a single task with explicit
    ``aclose()`` of the lines iterator, response, and client; the errant
    generator is not one we hold a reference to and therefore cannot
    close ourselves.

    We install a single process-wide unraisable hook that swallows just
    this specific interaction — identified by the tuple of (RuntimeError
    mentioning cancel scope / GeneratorExit) + (object repr referencing
    HTTP11ConnectionByteStream) — and defers to the default hook for
    everything else. The filter is idempotent.
    """
    prior_hook = sys.unraisablehook
    if getattr(prior_hook, "_unsloth_httpcore_silencer", False):
        return

    def _hook(unraisable):
        exc_value = getattr(unraisable, "exc_value", None)
        obj = getattr(unraisable, "object", None)
        obj_repr = repr(obj) if obj is not None else ""
        if (
            isinstance(exc_value, RuntimeError)
            and "HTTP11ConnectionByteStream" in obj_repr
            and ("cancel scope" in str(exc_value) or "GeneratorExit" in str(exc_value))
        ):
            return
        prior_hook(unraisable)

    _hook._unsloth_httpcore_silencer = True  # type: ignore[attr-defined]
    sys.unraisablehook = _hook


_install_httpcore_asyncgen_silencer()


def _friendly_error(exc: Exception) -> str:
    """Extract a user-friendly message from known llama-server errors."""
    # httpx transport-layer failures reaching the managed llama-server —
    # raised by the async pass-through helpers that talk to llama-server
    # directly. Treat any RequestError subclass (ConnectError, ReadError,
    # RemoteProtocolError, WriteError, PoolTimeout, ...) as "the upstream
    # subprocess is unreachable", which for Studio always means the
    # llama-server subprocess crashed or is still coming up.
    if isinstance(exc, httpx.RequestError):
        return "Lost connection to the model server. It may have crashed -- try reloading the model."
    msg = str(exc)
    m = _re.search(
        r"request \((\d+) tokens?\) exceeds the available context size \((\d+) tokens?\)",
        msg,
    )
    if m:
        return (
            f"Message too long: {m.group(1)} tokens exceeds the {m.group(2)}-token "
            f"context window. Try increasing the Context Length in Model settings, "
            f"or shorten the conversation."
        )
    if "Lost connection to llama-server" in msg:
        return "Lost connection to the model server. It may have crashed -- try reloading the model."
    return "An internal error occurred"


# Add backend directory to path
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

# Import backend functions
try:
    from core.inference import get_inference_backend
    from core.inference.llama_cpp import (
        LlamaCppBackend,
        _DEFAULT_MAX_TOKENS_FLOOR,
        _DEFAULT_T_MAX_PREDICT_MS,
        detect_reasoning_flags,
    )
    from core.inference.llama_server_args import validate_extra_args
    from utils.models import ModelConfig
    from utils.inference import load_inference_config
    from utils.models.model_config import load_model_defaults
    from utils.native_path_leases import (
        NativePathLeaseError,
        display_label_for_native_path,
        is_registered_native_path_label,
        redact_native_paths,
        verify_native_path_lease,
    )
except ImportError:
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from core.inference import get_inference_backend
    from core.inference.llama_cpp import (
        LlamaCppBackend,
        _DEFAULT_MAX_TOKENS_FLOOR,
        _DEFAULT_T_MAX_PREDICT_MS,
        detect_reasoning_flags,
    )
    from core.inference.llama_server_args import validate_extra_args
    from utils.models import ModelConfig
    from utils.inference import load_inference_config
    from utils.models.model_config import load_model_defaults
    from utils.native_path_leases import (
        NativePathLeaseError,
        display_label_for_native_path,
        is_registered_native_path_label,
        redact_native_paths,
        verify_native_path_lease,
    )

from models.inference import (
    LoadRequest,
    UnloadRequest,
    GenerateRequest,
    LoadResponse,
    LoadProgressResponse,
    UnloadResponse,
    InferenceStatusResponse,
    ChatCompletionRequest,
    ChatCompletionChunk,
    ChatCompletion,
    ChatMessage,
    ChunkChoice,
    ChoiceDelta,
    CompletionChoice,
    CompletionMessage,
    CompletionUsage,
    ValidateModelRequest,
    ValidateModelResponse,
    TextContentPart,
    ImageContentPart,
    ImageUrl,
    ResponsesRequest,
    ResponsesInputMessage,
    ResponsesInputTextPart,
    ResponsesInputImagePart,
    ResponsesOutputTextPart,
    ResponsesUnknownContentPart,
    ResponsesUnknownInputItem,
    ResponsesFunctionCallInputItem,
    ResponsesFunctionCallOutputInputItem,
    ResponsesOutputTextContent,
    ResponsesOutputMessage,
    ResponsesOutputFunctionCall,
    ResponsesUsage,
    ResponsesResponse,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicResponseTextBlock,
    AnthropicResponseToolUseBlock,
    AnthropicUsage,
)
from core.inference.anthropic_compat import (
    anthropic_messages_to_openai,
    anthropic_tools_to_openai,
    anthropic_tool_choice_to_openai,
    AnthropicStreamEmitter,
    AnthropicPassthroughEmitter,
)
from auth.authentication import get_current_subject

from core.inference.key_exchange import decrypt_api_key
from core.inference.providers import get_provider_info, get_base_url
from core.inference.external_provider import ExternalProviderClient
from storage import providers_db

import io
import wave
import base64
import numpy as np
from datetime import date as _date

router = APIRouter()
# Studio-only router (not mounted on /v1 OpenAI-compat).
studio_router = APIRouter()


def _effective_enable_tools(payload) -> Optional[bool]:
    """Resolve `payload.enable_tools` against the process-level tool policy.

    Returns the policy value when set (CLI hard-override from `unsloth run`),
    otherwise the per-request value.
    """
    from state.tool_policy import get_tool_policy

    policy = get_tool_policy()
    return policy if policy is not None else payload.enable_tools


# Cancel registry. Proxies (e.g. Colab) can swallow client fetch aborts
# so is_disconnected() never fires. POST /inference/cancel looks up
# in-flight cancel_events here by cancel_id (per-run) or session_id /
# completion_id (fallbacks).
_CANCEL_REGISTRY: dict[str, set[threading.Event]] = {}
_CANCEL_LOCK = threading.Lock()

# Cancel POSTs that arrive before registration are stashed; the next
# matching __enter__ replays set() within the TTL.
_PENDING_CANCELS: dict[str, float] = {}
_PENDING_CANCEL_TTL_S = 30.0


def _prune_pending(now: float) -> None:
    for k in [
        k for k, ts in _PENDING_CANCELS.items() if now - ts > _PENDING_CANCEL_TTL_S
    ]:
        _PENDING_CANCELS.pop(k, None)


class _TrackedCancel:
    """Register cancel_event in _CANCEL_REGISTRY for the block's duration."""

    def __init__(self, event: threading.Event, *keys):
        self.event = event
        self.keys = tuple(k for k in keys if k)

    def __enter__(self):
        # Register + consume-pending must be one critical section to close
        # the TOCTOU race against a concurrent cancel POST.
        should_cancel = False
        with _CANCEL_LOCK:
            for k in self.keys:
                _CANCEL_REGISTRY.setdefault(k, set()).add(self.event)
            now = time.monotonic()
            _prune_pending(now)
            for k in self.keys:
                if k and _PENDING_CANCELS.pop(k, None) is not None:
                    should_cancel = True
        if should_cancel:
            self.event.set()
        return self.event

    def __exit__(self, *exc):
        with _CANCEL_LOCK:
            for k in self.keys:
                bucket = _CANCEL_REGISTRY.get(k)
                if bucket is None:
                    continue
                bucket.discard(self.event)
                if not bucket:
                    _CANCEL_REGISTRY.pop(k, None)
        return False


def _cancel_by_keys(keys) -> int:
    """Set cancel_event for matching registry entries; no stash.
    session_id/completion_id are shared across runs on the same thread,
    so stashing them would ghost-cancel the user's next request. Only
    cancel_id is per-run unique (see _cancel_by_cancel_id_or_stash)."""
    if not keys:
        return 0
    events: set[threading.Event] = set()
    with _CANCEL_LOCK:
        _prune_pending(time.monotonic())
        for k in keys:
            bucket = _CANCEL_REGISTRY.get(k)
            if bucket:
                events.update(bucket)
    for ev in events:
        ev.set()
    return len(events)


def _cancel_by_cancel_id_or_stash(cancel_id: str) -> int:
    """Atomic lookup-or-stash; pairs with _TrackedCancel.__enter__ to
    close the TOCTOU race."""
    now = time.monotonic()
    events: set[threading.Event] = set()
    with _CANCEL_LOCK:
        _prune_pending(now)
        bucket = _CANCEL_REGISTRY.get(cancel_id)
        if bucket:
            events.update(bucket)
        else:
            _PENDING_CANCELS[cancel_id] = now
    for ev in events:
        ev.set()
    return len(events)


async def _await_cancel_then_close(cancel_event, resp) -> None:
    """Watch a threading.Event from asyncio and close ``resp`` when it fires.

    Used by the passthrough streamers so a /cancel POST can interrupt
    while the async iterator is blocked waiting for llama-server prefill.
    Without this watcher the in-loop ``cancel_event.is_set()`` check is
    unreachable until the first SSE chunk arrives, which is exactly the
    proxy/Colab scenario the cancel POST exists to handle.

    Polls a threading.Event because the cancel registry is keyed by
    threading.Event so the synchronous /cancel handler can call .set().
    50ms cadence adds at most that much latency to a prefill cancel; the
    common-case streaming cancel path still observes the event in the
    iterator's first iteration after the next chunk.
    """
    try:
        while not cancel_event.is_set():
            await asyncio.sleep(0.05)
        try:
            await resp.aclose()
        except Exception:
            pass
    except asyncio.CancelledError:
        return


# Appended to tool-use nudge to discourage plan-without-action
_TOOL_ACTION_NUDGE = (
    " IMPORTANT: Always call tools directly -- never write code yourself."
    " Never describe what you plan to do -- just call the tool immediately."
    " For any code request, call the python tool. For any factual question, call web_search."
    " Do NOT output code blocks -- use the python tool instead."
)

# Regex for stripping leaked tool-call XML from assistant messages/stream
_TOOL_XML_RE = _re.compile(
    r"<tool_call>.*?</tool_call>|<function=\w+>.*?</function>",
    _re.DOTALL,
)
logger = get_logger(__name__)


def _validate_native_mmproj_companion(
    mmproj_path: str | None, gguf_path: str | None
) -> None:
    if not mmproj_path or not gguf_path:
        return
    import stat as _stat_module

    mm = Path(mmproj_path)
    gguf = Path(gguf_path)
    try:
        mm_lstat = os.lstat(mm)
    except OSError as exc:
        raise HTTPException(
            status_code = 400,
            detail = "Native vision companion is no longer accessible.",
        ) from exc
    if _stat_module.S_ISLNK(mm_lstat.st_mode) or not _stat_module.S_ISREG(
        mm_lstat.st_mode
    ):
        raise HTTPException(
            status_code = 400,
            detail = "Native vision companion must be a regular file.",
        )
    try:
        if mm.resolve(strict = True).parent != gguf.resolve(strict = True).parent:
            raise HTTPException(
                status_code = 400,
                detail = "Native vision companion must live next to the selected GGUF.",
            )
    except OSError as exc:
        raise HTTPException(
            status_code = 400,
            detail = "Native vision companion is no longer accessible.",
        ) from exc


def _resolve_model_identifier_for_request(
    request: LoadRequest | ValidateModelRequest,
    *,
    operation: str,
) -> tuple[str, str, bool]:
    if not request.native_path_lease:
        return request.model_path, request.model_path, False
    try:
        grant = verify_native_path_lease(
            request.native_path_lease,
            operation = operation,
            expected_kind = "model",
            expected_path_type = "file",
            allowed_suffixes = (".gguf",),
        )
    except NativePathLeaseError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc
    display_label = (
        grant.display_label or Path(request.model_path).name or "Native model"
    )
    return str(grant.canonical_path), display_label, True


# GGUF inference backend (llama-server)
_llama_cpp_backend = LlamaCppBackend()


def get_llama_cpp_backend() -> LlamaCppBackend:
    return _llama_cpp_backend


@router.post("/load", response_model = LoadResponse)
async def load_model(
    request: LoadRequest,
    fastapi_request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    Load a model for inference.

    The model_path should be a clean identifier from GET /models/list.
    Returns inference configuration parameters (temperature, top_p, top_k, min_p)
    from the model's YAML config, falling back to default.yaml for missing values.

    GGUF models are loaded via llama-server (llama.cpp) instead of Unsloth.
    """
    native_grant_backed = False
    model_log_label = request.model_path
    try:
        # Validate user-supplied llama-server pass-through args up front
        # so a managed-flag collision returns 400 before any model work.
        try:
            extra_llama_args = validate_extra_args(request.llama_extra_args)
        except ValueError as exc:
            raise HTTPException(status_code = 400, detail = str(exc))

        model_identifier, model_log_label, native_grant_backed = (
            _resolve_model_identifier_for_request(request, operation = "load-model")
        )
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
                and llama_backend.model_identifier.lower() == model_identifier.lower()
            ):
                logger.info(
                    f"Model already loaded (GGUF): {model_log_label} variant={request.gguf_variant}, skipping reload"
                )
                inference_config = load_inference_config(llama_backend.model_identifier)

                _gguf_audio = (
                    llama_backend._audio_type
                    if hasattr(llama_backend, "_audio_type")
                    else None
                )
                _gguf_is_audio = getattr(llama_backend, "_is_audio", False)
                return LoadResponse(
                    status = "already_loaded",
                    model = model_log_label
                    if native_grant_backed
                    else llama_backend.model_identifier,
                    display_name = model_log_label
                    if native_grant_backed
                    else llama_backend.model_identifier,
                    is_vision = llama_backend._is_vision,
                    is_lora = False,
                    is_gguf = True,
                    is_audio = _gguf_is_audio,
                    audio_type = _gguf_audio,
                    has_audio_input = False,
                    inference = inference_config,
                    requires_trust_remote_code = bool(
                        inference_config.get("trust_remote_code", False)
                    ),
                    context_length = llama_backend.context_length,
                    max_context_length = llama_backend.max_context_length,
                    native_context_length = llama_backend.native_context_length,
                    supports_reasoning = llama_backend.supports_reasoning,
                    reasoning_style = llama_backend.reasoning_style,
                    reasoning_always_on = llama_backend.reasoning_always_on,
                    supports_preserve_thinking = llama_backend.supports_preserve_thinking,
                    chat_template = llama_backend.chat_template,
                    speculative_type = llama_backend.speculative_type,
                )
        else:
            if (
                backend.active_model_name
                and backend.active_model_name.lower() == model_identifier.lower()
            ):
                logger.info(
                    f"Model already loaded (Unsloth): {model_log_label}, skipping reload"
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
                # Non-GGUF: only advertise reasoning for gpt-oss Harmony,
                # which emits reasoning via channels at the tokenizer level.
                # Template-level chat_template_kwargs (enable_thinking /
                # preserve_thinking / tools) are not yet forwarded through
                # the transformers generation path, so avoid advertising
                # controls the server cannot honour outside GGUF.
                _sf_supports_reasoning = False
                _sf_reasoning_style = "enable_thinking"
                if hasattr(backend, "_is_gpt_oss_model"):
                    try:
                        if backend._is_gpt_oss_model():
                            _sf_supports_reasoning = True
                            _sf_reasoning_style = "reasoning_effort"
                    except Exception:
                        pass
                return LoadResponse(
                    status = "already_loaded",
                    model = model_log_label
                    if native_grant_backed
                    else backend.active_model_name,
                    display_name = model_log_label
                    if native_grant_backed
                    else backend.active_model_name,
                    is_vision = _model_info.get("is_vision", False),
                    is_lora = _model_info.get("is_lora", False),
                    is_gguf = False,
                    is_audio = _model_info.get("is_audio", False),
                    audio_type = _model_info.get("audio_type"),
                    has_audio_input = _model_info.get("has_audio_input", False),
                    inference = inference_config,
                    requires_trust_remote_code = bool(
                        inference_config.get("trust_remote_code", False)
                    ),
                    supports_reasoning = _sf_supports_reasoning,
                    reasoning_style = _sf_reasoning_style,
                    reasoning_always_on = False,
                    supports_preserve_thinking = False,
                    supports_tools = False,
                    chat_template = _chat_template,
                )

        # Create config using clean factory method
        # is_lora is auto-detected from adapter_config.json on disk/HF
        config = ModelConfig.from_identifier(
            model_id = model_identifier,
            hf_token = request.hf_token,
            gguf_variant = request.gguf_variant,
        )

        if not config:
            raise HTTPException(
                status_code = 400,
                detail = f"Invalid model identifier: {model_log_label}",
            )

        # Normalize gpu_ids: empty list means auto-selection, same as None
        effective_gpu_ids = request.gpu_ids if request.gpu_ids else None

        # ── GGUF path: load via llama-server ──────────────────────
        if config.is_gguf:
            if effective_gpu_ids is not None:
                raise HTTPException(
                    status_code = 400,
                    detail = "gpu_ids is not supported for GGUF models yet.",
                )

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
            _n_parallel = getattr(fastapi_request.app.state, "llama_parallel_slots", 1)

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
                    speculative_type = request.speculative_type,
                    n_parallel = _n_parallel,
                    extra_args = extra_llama_args,
                )
            else:
                # Local mode: llama-server loads via -m <path>
                if native_grant_backed and config.gguf_mmproj_file:
                    _validate_native_mmproj_companion(
                        config.gguf_mmproj_file, config.gguf_file
                    )
                success = await asyncio.to_thread(
                    llama_backend.load_model,
                    gguf_path = config.gguf_file,
                    mmproj_path = config.gguf_mmproj_file,
                    model_identifier = config.identifier,
                    is_vision = config.is_vision,
                    n_ctx = request.max_seq_length,
                    chat_template_override = request.chat_template_override,
                    cache_type_kv = request.cache_type_kv,
                    speculative_type = request.speculative_type,
                    n_parallel = _n_parallel,
                    extra_args = extra_llama_args,
                )

            if not success:
                raise HTTPException(
                    status_code = 500,
                    detail = f"Failed to load GGUF model: {model_log_label if native_grant_backed else config.display_name}",
                )

            logger.info(
                f"Loaded GGUF model via llama-server: {model_log_label if native_grant_backed else config.identifier}"
            )

            # Detect TTS/audio marker tokens by probing the loaded model's vocabulary.
            # GGUF audio input is not wired through the chat path yet, so do not
            # advertise has_audio_input for GGUF models until uploaded audio is
            # actually forwarded to llama-server.
            _gguf_audio = llama_backend.detect_audio_type()
            _gguf_is_audio = _gguf_audio in ("snac", "bicodec", "dac")
            llama_backend._is_audio = _gguf_is_audio
            llama_backend._audio_type = _gguf_audio
            llama_backend._native_display_label = (
                model_log_label if native_grant_backed else None
            )
            llama_backend._native_grant_backed = bool(native_grant_backed)
            if _gguf_is_audio:
                logger.info(f"GGUF model detected as audio: audio_type={_gguf_audio}")
                await asyncio.to_thread(llama_backend.init_audio_codec, _gguf_audio)

            inference_config = load_inference_config(config.identifier)

            return LoadResponse(
                status = "loaded",
                model = model_log_label if native_grant_backed else config.identifier,
                display_name = model_log_label
                if native_grant_backed
                else config.display_name,
                is_vision = config.is_vision,
                is_lora = False,
                is_gguf = True,
                is_audio = _gguf_is_audio,
                audio_type = _gguf_audio,
                has_audio_input = False,
                inference = inference_config,
                requires_trust_remote_code = bool(
                    inference_config.get("trust_remote_code", False)
                ),
                context_length = llama_backend.context_length,
                max_context_length = llama_backend.max_context_length,
                native_context_length = llama_backend.native_context_length,
                supports_reasoning = llama_backend.supports_reasoning,
                reasoning_style = llama_backend.reasoning_style,
                reasoning_always_on = llama_backend.reasoning_always_on,
                supports_preserve_thinking = llama_backend.supports_preserve_thinking,
                supports_tools = llama_backend.supports_tools,
                cache_type_kv = llama_backend.cache_type_kv,
                chat_template = llama_backend.chat_template,
                speculative_type = llama_backend.speculative_type,
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
            gpu_ids = effective_gpu_ids,
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
                status_code = 500,
                detail = f"Failed to load model: {model_log_label if native_grant_backed else config.display_name}",
            )

        logger.info(
            f"Loaded model: {model_log_label if native_grant_backed else config.identifier}"
        )

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

        # Non-GGUF: gpt-oss Harmony surfaces reasoning via tokenizer-level
        # channels; other safetensors reasoning/tools/preserve-thinking
        # knobs are not forwarded to tokenizer.apply_chat_template yet, so
        # we only advertise support for the Harmony case here.
        _sf_supports_reasoning = False
        _sf_reasoning_style = "enable_thinking"
        if hasattr(backend, "_is_gpt_oss_model"):
            try:
                if backend._is_gpt_oss_model():
                    _sf_supports_reasoning = True
                    _sf_reasoning_style = "reasoning_effort"
            except Exception:
                pass

        return LoadResponse(
            status = "loaded",
            model = model_log_label if native_grant_backed else config.identifier,
            display_name = model_log_label
            if native_grant_backed
            else config.display_name,
            is_vision = config.is_vision,
            is_lora = config.is_lora,
            is_gguf = False,
            is_audio = config.is_audio,
            audio_type = config.audio_type,
            has_audio_input = config.has_audio_input,
            inference = inference_config,
            requires_trust_remote_code = bool(
                inference_config.get("trust_remote_code", False)
            ),
            supports_reasoning = _sf_supports_reasoning,
            reasoning_style = _sf_reasoning_style,
            reasoning_always_on = False,
            supports_preserve_thinking = False,
            supports_tools = False,
            chat_template = _chat_template,
        )

    except HTTPException:
        raise
    except ValueError as e:
        if native_grant_backed:
            redacted_msg = redact_native_paths(str(e))
            logger.warning(
                "Rejected inference selection for native model %s: %s",
                model_log_label,
                redacted_msg,
            )
            raise HTTPException(status_code = 400, detail = redacted_msg)
        logger.warning("Rejected inference GPU selection: %s", e)
        raise HTTPException(status_code = 400, detail = str(e))
    except Exception as e:
        # Surface a friendlier message for models that Unsloth cannot load
        not_supported_hints = [
            "No config file found",
            "not yet supported",
            "is not supported",
            "does not support",
        ]
        if native_grant_backed:
            redacted_msg = redact_native_paths(str(e))
            logger.error(
                "Error loading native model %s: %s",
                model_log_label,
                redacted_msg,
            )
            msg = redacted_msg
            if any(h.lower() in msg.lower() for h in not_supported_hints):
                msg = f"This model is not supported yet. Try a different model. (Original error: {msg})"
            raise HTTPException(
                status_code = 500,
                detail = f"Failed to load native model {model_log_label}: {msg}",
            )
        logger.error(f"Error loading model: {e}", exc_info = True)
        msg = str(e)
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
    native_grant_backed = False
    model_log_label = request.model_path
    try:
        model_identifier, model_log_label, native_grant_backed = (
            _resolve_model_identifier_for_request(request, operation = "validate-model")
        )
        config = ModelConfig.from_identifier(
            model_id = model_identifier,
            hf_token = request.hf_token,
            gguf_variant = request.gguf_variant,
        )

        if not config:
            raise HTTPException(
                status_code = 400,
                detail = f"Invalid model identifier: {model_log_label}",
            )

        return ValidateModelResponse(
            valid = True,
            message = "Model identifier is valid.",
            identifier = model_log_label if native_grant_backed else config.identifier,
            display_name = model_log_label
            if native_grant_backed
            else getattr(config, "display_name", config.identifier),
            is_gguf = getattr(config, "is_gguf", False),
            is_lora = getattr(config, "is_lora", False),
            is_vision = getattr(config, "is_vision", False),
            requires_trust_remote_code = bool(
                load_inference_config(config.identifier).get("trust_remote_code", False)
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        not_supported_hints = [
            "No config file found",
            "not yet supported",
            "is not supported",
            "does not support",
        ]
        if native_grant_backed:
            redacted_msg = redact_native_paths(str(e))
            logger.error(
                "Error validating native model %s: %s",
                model_log_label,
                redacted_msg,
            )
            msg = redacted_msg
            if any(h.lower() in msg.lower() for h in not_supported_hints):
                msg = f"This model is not supported yet. Try a different model. (Original error: {msg})"
            raise HTTPException(
                status_code = 400,
                detail = f"Invalid native model {model_log_label}: {msg}",
            )
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
            or is_registered_native_path_label(
                llama_backend.model_identifier, request.model_path
            )
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


@studio_router.post("/cancel")
async def cancel_inference(
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """Cancel in-flight inference requests.

    Body (JSON, at least one key required):
      cancel_id    - preferred: per-run UUID, matched exclusively.
      session_id   - fallback when cancel_id is absent.
      completion_id - fallback when cancel_id is absent.

    A cancel_id arriving before its stream registers is stashed briefly
    and replayed on registration. Returns {"cancelled": N}.
    """
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception as e:
        logger.debug("Failed to parse cancel request body: %s", e)
        body = {}

    cancel_id = body.get("cancel_id")
    if isinstance(cancel_id, str) and cancel_id:
        return {"cancelled": _cancel_by_cancel_id_or_stash(cancel_id)}

    keys = []
    # `message_id` is the Anthropic passthrough's per-run identifier --
    # included so /v1/messages clients can cancel by their native id.
    for k in ("completion_id", "session_id", "message_id"):
        v = body.get(k)
        if isinstance(v, str) and v:
            keys.append(v)

    if not keys:
        return {"cancelled": 0}

    n = _cancel_by_keys(keys)
    return {"cancelled": n}


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
            yield f"data: {json.dumps({'error': _friendly_error(e)})}\n\n"

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
            _native_grant_backed = getattr(llama_backend, "_native_grant_backed", False)
            _display_model_id = getattr(
                llama_backend, "_native_display_label", None
            ) or display_label_for_native_path(_model_id)
            if (
                _native_grant_backed
                and _model_id
                and _display_model_id == _model_id
                and os.path.isabs(_model_id)
            ):
                _display_model_id = os.path.basename(_model_id)
            _inference_cfg = load_inference_config(_model_id) if _model_id else None
            _audio_type = getattr(llama_backend, "_audio_type", None)
            return InferenceStatusResponse(
                active_model = _display_model_id,
                is_vision = llama_backend.is_vision,
                is_gguf = True,
                gguf_variant = llama_backend.hf_variant,
                is_audio = getattr(llama_backend, "_is_audio", False),
                audio_type = _audio_type,
                has_audio_input = False,
                loading = [],
                loaded = [_display_model_id] if _display_model_id else [],
                inference = _inference_cfg,
                requires_trust_remote_code = bool(
                    (_inference_cfg or {}).get("trust_remote_code", False)
                ),
                supports_reasoning = llama_backend.supports_reasoning,
                reasoning_style = llama_backend.reasoning_style,
                reasoning_always_on = llama_backend.reasoning_always_on,
                supports_preserve_thinking = llama_backend.supports_preserve_thinking,
                supports_tools = llama_backend.supports_tools,
                chat_template = llama_backend.chat_template,
                context_length = llama_backend.context_length,
                max_context_length = llama_backend.max_context_length,
                native_context_length = llama_backend.native_context_length,
                cache_type_kv = llama_backend.cache_type_kv,
                chat_template_override = llama_backend.chat_template_override,
                speculative_type = llama_backend.speculative_type,
            )

        # Otherwise, report Unsloth backend status
        backend = get_inference_backend()

        is_vision = False
        is_audio = False
        audio_type = None
        has_audio_input = False
        model_info = {}
        if backend.active_model_name:
            model_info = backend.models.get(backend.active_model_name, {})
            is_vision = model_info.get("is_vision", False)
            is_audio = model_info.get("is_audio", False)
            audio_type = model_info.get("audio_type")
            has_audio_input = model_info.get("has_audio_input", False)
        chat_template_info = model_info.get("chat_template_info", {})
        chat_template = (
            chat_template_info.get("template")
            if isinstance(chat_template_info, dict)
            else None
        )

        # Non-GGUF: only gpt-oss Harmony is wired through the transformers
        # generation path. Other template-level reasoning / tool kwargs
        # are not yet forwarded, so we do not advertise them here.
        supports_reasoning = False
        reasoning_style = "enable_thinking"
        if backend.active_model_name and hasattr(backend, "_is_gpt_oss_model"):
            try:
                if backend._is_gpt_oss_model():
                    supports_reasoning = True
                    reasoning_style = "reasoning_effort"
            except Exception:
                pass
        inference_config = (
            load_inference_config(backend.active_model_name)
            if backend.active_model_name
            else None
        )

        return InferenceStatusResponse(
            active_model = backend.active_model_name,
            is_vision = is_vision,
            is_gguf = False,
            is_audio = is_audio,
            audio_type = audio_type,
            has_audio_input = has_audio_input,
            loading = list(getattr(backend, "loading_models", set())),
            loaded = list(backend.models.keys()),
            inference = inference_config,
            requires_trust_remote_code = bool(
                (inference_config or {}).get("trust_remote_code", False)
            ),
            supports_reasoning = supports_reasoning,
            reasoning_style = reasoning_style,
            reasoning_always_on = False,
            supports_preserve_thinking = False,
            supports_tools = False,
            chat_template = chat_template,
        )

    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = f"Failed to get status: {str(e)}")


@router.get("/load-progress", response_model = LoadProgressResponse)
async def get_load_progress(
    current_subject: str = Depends(get_current_subject),
):
    """
    Return the active GGUF load's mmap/upload progress.

    During the warmup window after a GGUF download -- when llama-server
    is paging ~tens-to-hundreds of GB of shards into the page cache
    before pushing layers to VRAM -- ``/api/inference/status`` only
    shows a generic spinner. This endpoint exposes sampled progress so
    the UI can render a real bar plus rate/ETA during that window.

    Returns an empty payload (``phase=null, bytes=0``) when no load is
    in flight. The frontend should stop polling once ``phase`` becomes
    ``ready``.
    """
    try:
        llama_backend = get_llama_cpp_backend()
        progress = llama_backend.load_progress()
        if progress is None:
            return LoadProgressResponse()
        return LoadProgressResponse(**progress)
    except Exception as e:
        logger.warning(f"Error sampling load progress: {e}")
        return LoadProgressResponse()


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


# ── External provider proxy ──────────────────────────────────────


def _build_external_messages(
    messages: list,
    supports_vision: bool,
) -> list[dict]:
    """
    Convert ChatMessage list to OpenAI-compatible dicts for external providers.

    - Vision providers: preserve multimodal content arrays (image_url parts intact).
    - Non-vision providers: flatten to text-only (images silently dropped).
    """
    result = []
    for msg in messages:
        if isinstance(msg.content, str):
            # Skip assistant messages with empty content (some providers reject them)
            if msg.role == "assistant" and not msg.content.strip():
                continue
            result.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg.content, list):
            if supports_vision:
                parts = []
                for part in msg.content:
                    if part.type == "text":
                        parts.append({"type": "text", "text": part.text})
                    elif part.type == "image_url":
                        parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": part.image_url.url},
                            }
                        )
                result.append({"role": msg.role, "content": parts})
            else:
                # Non-vision provider — strip images, keep text only
                text = "\n".join(p.text for p in msg.content if p.type == "text")
                result.append({"role": msg.role, "content": text})
    return result


async def _proxy_to_external_provider(
    payload: ChatCompletionRequest,
    request: Request,
) -> StreamingResponse:
    """
    Proxy a chat completion request to an external LLM provider.

    Resolves provider config (from DB or registry), decrypts the API key,
    and streams the response back in OpenAI SSE format.
    """
    # Resolve provider type and base URL
    provider_type = payload.provider_type
    base_url = payload.provider_base_url

    if payload.provider_id:
        config = providers_db.get_provider(payload.provider_id)
        if config is None:
            raise HTTPException(
                status_code = 404,
                detail = f"Provider config not found: {payload.provider_id}",
            )
        if not config["is_enabled"]:
            raise HTTPException(
                status_code = 400,
                detail = f"Provider '{config['display_name']}' is disabled.",
            )
        provider_type = provider_type or config["provider_type"]
        base_url = base_url or config["base_url"]

    if not provider_type:
        raise HTTPException(
            status_code = 400,
            detail = "Either provider_id or provider_type is required for external provider routing.",
        )

    # Fall back to registry default base URL
    if not base_url:
        base_url = get_base_url(provider_type)
    if not base_url:
        raise HTTPException(
            status_code = 400,
            detail = f"Unknown provider type: {provider_type}",
        )

    api_key = ""
    if payload.encrypted_api_key:
        try:
            api_key = decrypt_api_key(payload.encrypted_api_key)
        except Exception as exc:
            logger.warning("external_provider.decrypt_failed", error = str(exc))
            raise HTTPException(
                status_code = 400,
                detail = "Failed to decrypt API key. The server key may have changed — try refreshing the page.",
            )

    model = payload.external_model or payload.model
    if model == "default":
        raise HTTPException(
            status_code = 400,
            detail = "external_model is required when using an external provider.",
        )

    # Build messages preserving multimodal content for vision-capable providers
    from core.inference.providers import get_provider_info as _get_provider_info

    _pinfo = _get_provider_info(provider_type) or {}
    _supports_vision = _pinfo.get("supports_vision", False)
    chat_messages = _build_external_messages(payload.messages, _supports_vision)

    client = ExternalProviderClient(
        provider_type = provider_type,
        base_url = base_url,
        api_key = api_key,
    )

    async def _stream():
        gen = client.stream_chat_completion(
            messages = chat_messages,
            model = model,
            temperature = payload.temperature,
            top_p = payload.top_p,
            max_tokens = payload.max_tokens,
            presence_penalty = payload.presence_penalty,
            top_k = payload.top_k,
            enable_thinking = payload.enable_thinking,
            reasoning_effort = payload.reasoning_effort,
            enabled_tools = payload.enabled_tools,
            enable_prompt_caching = payload.enable_prompt_caching,
            stream = payload.stream,
        )
        try:
            sent_done = False
            async for line in gen:
                yield f"{line}\n\n"
                if "[DONE]" in line:
                    sent_done = True
            if not sent_done:
                yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("external_provider.stream_error", error = str(exc))
        finally:
            try:
                await gen.aclose()
            except RuntimeError:
                pass  # suppress httpcore asyncgen cleanup error (Python 3.13 + httpcore 1.0.x)
            await client.close()

    return StreamingResponse(
        _stream(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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
    # ── External provider routing ────────────────────────────────
    # encrypted_api_key is optional — local providers (llama.cpp / vLLM / Ollama) may run without auth.
    if payload.provider_id or payload.provider_type:
        return await _proxy_to_external_provider(payload, request)

    llama_backend = get_llama_cpp_backend()
    using_gguf = llama_backend.is_loaded

    # OpenAI-SDK clients send ``chat_template_kwargs`` via ``extra_body``,
    # which the SDK spreads into the request body at the top level. Studio's
    # ChatCompletionRequest has ``extra="allow"`` so pydantic stashes them in
    # ``model_extra``, but the typed ``payload.enable_thinking`` path is what
    # downstream generators actually consume. Lift ``enable_thinking`` from
    # the extra-body chat_template_kwargs onto the typed field so clients
    # that only know the OpenAI shape (data_designer recipe runs, etc.)
    # can still control the reasoning preamble.
    _extra = getattr(payload, "model_extra", None)
    if payload.enable_thinking is None and isinstance(_extra, dict):
        _tpl_kw = _extra.get("chat_template_kwargs")
        if isinstance(_tpl_kw, dict) and "enable_thinking" in _tpl_kw:
            payload.enable_thinking = bool(_tpl_kw["enable_thinking"])

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
                _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
                _tracker = _TrackedCancel(cancel_event, *_cancel_keys)
                _tracker.__enter__()

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

                        gen = audio_input_generate()
                        _DONE = object()
                        while True:
                            if cancel_event.is_set():
                                break
                            if await request.is_disconnected():
                                cancel_event.set()
                                return
                            chunk_text = await asyncio.to_thread(next, gen, _DONE)
                            if chunk_text is _DONE:
                                break
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
                        yield f"data: {json.dumps({'error': {'message': _friendly_error(e), 'type': 'server_error'}})}\n\n"
                    finally:
                        _tracker.__exit__(None, None, None)

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

    # ── Standard OpenAI function-calling pass-through (GGUF only) ────
    # When a client (opencode / Claude Code via OpenAI compat / Cursor /
    # Continue / ...) sends standard OpenAI `tools` without Studio's
    # `enable_tools` shorthand, forward the request to llama-server
    # verbatim so structured `tool_calls` flow back to the client. This
    # branch runs BEFORE `_extract_content_parts` because that helper is
    # unaware of `role="tool"` messages and assistant messages that only
    # carry `tool_calls` (content=None) — both of which are valid in
    # multi-turn client-side tool loops.
    _has_tool_messages = any(m.role == "tool" or m.tool_calls for m in payload.messages)
    # Route guided-decoding requests through the verbatim passthrough so
    # ``response_format`` (JSON schema) actually reaches llama-server and
    # the model's GBNF-constrained output comes back unmodified. The
    # non-passthrough GGUF path below calls ``generate_chat_completion``
    # which has no response_format kwarg, so the schema gets silently
    # dropped and data_designer falls back to free-form sampling. Guided
    # decoding does not require ``supports_tools`` - the grammar machinery
    # is independent of tool-call parsing.
    _has_response_format = _extract_response_format(payload) is not None
    _tools_passthrough = llama_backend.supports_tools and (
        (payload.tools and len(payload.tools) > 0) or _has_tool_messages
    )
    if (
        using_gguf
        and not _effective_enable_tools(payload)
        and (_tools_passthrough or _has_response_format)
    ):
        if payload.audio_base64:
            raise HTTPException(
                status_code = 400,
                detail = "Audio input is not supported for GGUF chat models yet.",
            )

        # Preserve the vision guard that would otherwise run in the
        # non-passthrough path below: text-only tool-capable GGUFs
        # should return a clear 400 here rather than forwarding the
        # image to llama-server and surfacing an opaque upstream error.
        if not llama_backend.is_vision and (
            payload.image_base64
            or any(
                isinstance(m.content, list)
                and any(isinstance(p, ImageContentPart) for p in m.content)
                for m in payload.messages
            )
        ):
            raise HTTPException(
                status_code = 400,
                detail = "Image provided but current GGUF model does not support vision.",
            )

        cancel_event = threading.Event()
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        if payload.stream:
            return await _openai_passthrough_stream(
                request,
                cancel_event,
                llama_backend,
                payload,
                model_name,
                completion_id,
            )
        return await _openai_passthrough_non_streaming(
            llama_backend,
            payload,
            model_name,
        )

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
        if payload.audio_base64:
            raise HTTPException(
                status_code = 400,
                detail = "Audio input is not supported for GGUF chat models yet.",
            )

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
                from PIL import Image as _Image, UnidentifiedImageError as _UIE

                raw = _b64.b64decode(image_b64)
                # Normalize to RGB so PNG encoding succeeds regardless of
                # source mode (RGBA, P, L, CMYK, I, F, ...). Previously
                # we only converted RGBA, which left CMYK/I/F to raise at
                # img.save(PNG).
                img = _Image.open(_BytesIO(raw)).convert("RGB")
                buf = _BytesIO()
                img.save(buf, format = "PNG")
                image_b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
            except _UIE:
                raise HTTPException(
                    status_code = 400,
                    detail = "Unsupported or corrupt image format.",
                )
            except Exception:
                raise HTTPException(
                    status_code = 400,
                    detail = "Failed to process image.",
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
        # `_effective_enable_tools` lets `unsloth run --enable-tools/--disable-tools`
        # hard-override the per-request value. Without a CLI override, falls
        # back to `payload.enable_tools` (existing behavior).
        use_tools = (
            _effective_enable_tools(payload)
            and llama_backend.supports_tools
            and not image_b64
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

            # ── Tool-use system prompt nudge ──────────────────────
            _tool_names = {t["function"]["name"] for t in tools_to_use}
            _has_web = "web_search" in _tool_names
            _has_code = "python" in _tool_names or "terminal" in _tool_names

            _date_line = f"The current date is {_date.today().isoformat()}."

            # Small models (<9B) struggle with multi-step search plans,
            # so simplify the web tips to avoid plan-then-stall behavior.
            _model_size_b = _extract_model_size_b(model_name)
            _is_small_model = _model_size_b is not None and _model_size_b < 9

            if _is_small_model:
                _web_tips = "Do not repeat the same search query."
            else:
                _web_tips = (
                    "When you search and find a relevant URL in the results, "
                    "fetch its full content by calling web_search with the url parameter. "
                    "Do not repeat the same search query. If a search returns "
                    "no useful results, try rephrasing or fetching a result URL directly."
                )
            _code_tips = (
                "Use code execution for math, calculations, data processing, "
                "or to parse and analyze information from tool results."
            )

            if _has_web and _has_code:
                _nudge = (
                    _date_line + " "
                    "You have access to tools. When appropriate, prefer using "
                    "tools rather than answering from memory. "
                    + _web_tips
                    + " "
                    + _code_tips
                )
            elif _has_code:
                _nudge = (
                    _date_line + " "
                    "You have access to tools. When appropriate, prefer using "
                    "code execution rather than answering from memory. " + _code_tips
                )
            elif _has_web:
                _nudge = (
                    _date_line + " "
                    "You have access to tools. When appropriate, prefer using "
                    "web search for up-to-date or uncertain factual "
                    "information rather than answering from memory. " + _web_tips
                )
            else:
                _nudge = ""

            if _nudge:
                _nudge += _TOOL_ACTION_NUDGE
                # Append nudge to system prompt (preserve user's prompt)
                if system_prompt:
                    system_prompt = system_prompt.rstrip() + "\n\n" + _nudge
                else:
                    system_prompt = _nudge
                # Rebuild gguf_messages with updated system prompt
                gguf_messages = []
                if system_prompt:
                    gguf_messages.append({"role": "system", "content": system_prompt})
                gguf_messages.extend(chat_messages)

            # ── Strip stale tool-call XML from conversation history ─
            for _msg in gguf_messages:
                if _msg.get("role") == "assistant" and isinstance(
                    _msg.get("content"), str
                ):
                    _msg["content"] = _TOOL_XML_RE.sub("", _msg["content"]).strip()

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
                    reasoning_effort = payload.reasoning_effort,
                    preserve_thinking = payload.preserve_thinking,
                    auto_heal_tool_calls = payload.auto_heal_tool_calls
                    if payload.auto_heal_tool_calls is not None
                    else True,
                    max_tool_iterations = payload.max_tool_calls_per_message
                    if payload.max_tool_calls_per_message is not None
                    else 25,
                    tool_call_timeout = payload.tool_call_timeout
                    if payload.tool_call_timeout is not None
                    else 300,
                    session_id = payload.session_id,
                )

            _tool_sentinel = object()

            _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
            _tracker = _TrackedCancel(cancel_event, *_cancel_keys)
            _tracker.__enter__()

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
                    _stream_usage = None
                    _stream_timings = None
                    while True:
                        if cancel_event.is_set():
                            break
                        if await request.is_disconnected():
                            cancel_event.set()
                            return

                        event = await asyncio.to_thread(next, gen, _tool_sentinel)
                        if event is _tool_sentinel:
                            break

                        if event["type"] == "status":
                            # Empty status marks an iteration boundary
                            # in the GGUF tool loop (e.g. after a
                            # re-prompt).  Reset the cumulative cursor
                            # so the next assistant turn streams cleanly.
                            if not event["text"]:
                                prev_text = ""
                            # Emit tool status as a custom SSE event
                            # (including empty ones to clear UI badges)
                            status_data = json.dumps(
                                {
                                    "type": "tool_status",
                                    "content": event["text"],
                                }
                            )
                            yield f"data: {status_data}\n\n"
                            continue

                        if event["type"] in ("tool_start", "tool_end"):
                            if event["type"] == "tool_start":
                                prev_text = ""
                            yield f"data: {json.dumps(event)}\n\n"
                            continue

                        if event["type"] == "metadata":
                            _stream_usage = event.get("usage")
                            _stream_timings = event.get("timings")
                            continue

                        # "content" type -- cumulative text
                        # Sanitize the full cumulative then diff against
                        # the last sanitized snapshot so cross-chunk XML
                        # tags are handled correctly.
                        raw_cumulative = event.get("text", "")
                        clean_cumulative = _TOOL_XML_RE.sub("", raw_cumulative)
                        new_text = clean_cumulative[len(prev_text) :]
                        prev_text = clean_cumulative
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
                    # Usage chunk (OpenAI-standard: choices=[], usage populated)
                    if _stream_usage or _stream_timings:
                        usage_obj = CompletionUsage(
                            prompt_tokens = (_stream_usage or {}).get("prompt_tokens", 0),
                            completion_tokens = (_stream_usage or {}).get(
                                "completion_tokens", 0
                            ),
                            total_tokens = (_stream_usage or {}).get("total_tokens", 0),
                        )
                        usage_chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [],
                            usage = usage_obj,
                            timings = _stream_timings,
                        )
                        yield f"data: {usage_chunk.model_dump_json(exclude_none = True)}\n\n"
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
                            "message": _friendly_error(e),
                            "type": "server_error",
                        },
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                finally:
                    _tracker.__exit__(None, None, None)

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
                reasoning_effort = payload.reasoning_effort,
                preserve_thinking = payload.preserve_thinking,
            )

        _gguf_sentinel = object()

        if payload.stream:
            _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
            _tracker = _TrackedCancel(cancel_event, *_cancel_keys)
            _tracker.__enter__()

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
                    _stream_usage = None
                    _stream_timings = None
                    while True:
                        if cancel_event.is_set():
                            break
                        if await request.is_disconnected():
                            cancel_event.set()
                            return
                        cumulative = await asyncio.to_thread(next, gen, _gguf_sentinel)
                        if cumulative is _gguf_sentinel:
                            break
                        # Capture server metadata for final usage chunk
                        if isinstance(cumulative, dict):
                            if cumulative.get("type") == "metadata":
                                _stream_usage = cumulative.get("usage")
                                _stream_timings = cumulative.get("timings")
                            else:
                                logger.warning(
                                    "gguf_stream_chunks: unexpected dict event: %s",
                                    {
                                        k: v
                                        for k, v in cumulative.items()
                                        if k != "timings"
                                    },
                                )
                            continue
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
                    # Usage chunk (OpenAI-standard: choices=[], usage populated)
                    if _stream_usage or _stream_timings:
                        usage_obj = CompletionUsage(
                            prompt_tokens = (_stream_usage or {}).get("prompt_tokens", 0),
                            completion_tokens = (_stream_usage or {}).get(
                                "completion_tokens", 0
                            ),
                            total_tokens = (_stream_usage or {}).get("total_tokens", 0),
                        )
                        usage_chunk = ChatCompletionChunk(
                            id = completion_id,
                            created = created,
                            model = model_name,
                            choices = [],
                            usage = usage_obj,
                            timings = _stream_timings,
                        )
                        yield f"data: {usage_chunk.model_dump_json(exclude_none = True)}\n\n"
                    yield "data: [DONE]\n\n"

                except asyncio.CancelledError:
                    cancel_event.set()
                    raise
                except Exception as e:
                    logger.error(f"Error during GGUF streaming: {e}", exc_info = True)
                    error_chunk = {
                        "error": {
                            "message": _friendly_error(e),
                            "type": "server_error",
                        },
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                finally:
                    _tracker.__exit__(None, None, None)

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
                    if isinstance(token, dict):
                        continue  # skip metadata dict in non-streaming path
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
        _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
        _tracker = _TrackedCancel(cancel_event, *_cancel_keys)
        _tracker.__enter__()

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
                    if cancel_event.is_set():
                        backend.reset_generation_state()
                        break
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
                        "message": _friendly_error(e),
                        "type": "server_error",
                    },
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
            finally:
                _tracker.__exit__(None, None, None)

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
# Sandbox file serving  (/sandbox/{session_id}/{filename})
# =====================================================================

_SANDBOX_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


@router.get("/sandbox/{session_id}/{filename}")
async def serve_sandbox_file(
    session_id: str,
    filename: str,
    request: Request,
    token: Optional[str] = None,
):
    """
    Serve image files created by Python tool execution.

    Accepts auth via Authorization header OR ?token= query param
    (needed because <img src> cannot send custom headers).
    """
    from fastapi.responses import FileResponse

    # ── Authentication (header or query param) ──────────────────
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        jwt_token = auth_header[7:]
    elif token:
        jwt_token = token
    else:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Missing authentication token",
        )
    from fastapi.security import HTTPAuthorizationCredentials

    creds = HTTPAuthorizationCredentials(scheme = "Bearer", credentials = jwt_token)
    await get_current_subject(creds)

    # ── Filename sanitization ───────────────────────────────────
    safe_filename = os.path.basename(filename)
    if not safe_filename or safe_filename in (".", ".."):
        raise HTTPException(status_code = 404, detail = "Not found")

    # ── Extension allowlist ─────────────────────────────────────
    ext = os.path.splitext(safe_filename)[1].lower()
    media_type = _SANDBOX_MEDIA_TYPES.get(ext)
    if not media_type:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "File type not allowed",
        )

    # ── Path containment check ──────────────────────────────────
    home = os.path.expanduser("~")
    sandbox_root = os.path.realpath(os.path.join(home, "studio_sandbox"))
    safe_session = os.path.basename(session_id.replace("..", ""))
    if not safe_session:
        raise HTTPException(status_code = 404, detail = "Not found")

    file_path = os.path.realpath(
        os.path.join(sandbox_root, safe_session, safe_filename)
    )
    if not file_path.startswith(sandbox_root + os.sep):
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "Access denied",
        )

    if not os.path.isfile(file_path):
        raise HTTPException(status_code = 404, detail = "Not found")

    return FileResponse(
        path = file_path,
        media_type = media_type,
        headers = {
            "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


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


# =====================================================================
# OpenAI-Compatible Completions Proxy  (/completions → /v1/completions)
# =====================================================================


@router.post("/completions")
async def openai_completions(
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    OpenAI-compatible text completions endpoint (non-chat).

    Transparently proxies to the running llama-server's ``/v1/completions``.
    Only available when a GGUF model is loaded.
    """
    llama_backend = get_llama_cpp_backend()
    if not llama_backend.is_loaded:
        raise HTTPException(
            status_code = 503,
            detail = "No GGUF model loaded. Load a GGUF model first.",
        )

    body = await request.json()
    target_url = f"{llama_backend.base_url}/v1/completions"
    is_stream = body.get("stream", False)

    if is_stream:

        async def _stream():
            # Manual httpx client/response lifecycle AND explicit
            # aiter_bytes() iterator close — see _anthropic_passthrough_stream
            # for the full rationale. Saving `bytes_iter = resp.aiter_bytes()`
            # and `await bytes_iter.aclose()` in the finally block is the
            # part that matters for avoiding the Python 3.13 + httpcore
            # 1.0.x "Exception ignored in: <async_generator>" / anyio
            # cancel-scope trace: an anonymous async for leaves the
            # iterator unclosed, so Python's asyncgen GC finalizer runs
            # cleanup on a later pass in a different asyncio task.
            client = httpx.AsyncClient(timeout = 600)
            resp = None
            bytes_iter = None
            try:
                req = client.build_request("POST", target_url, json = body)
                resp = await client.send(req, stream = True)
                bytes_iter = resp.aiter_bytes()
                async for chunk in bytes_iter:
                    yield chunk
            except Exception as e:
                logger.error("openai_completions stream error: %s", e)
            finally:
                if bytes_iter is not None:
                    try:
                        await bytes_iter.aclose()
                    except Exception:
                        pass
                if resp is not None:
                    try:
                        await resp.aclose()
                    except Exception:
                        pass
                try:
                    await client.aclose()
                except Exception:
                    pass

        return StreamingResponse(_stream(), media_type = "text/event-stream")
    else:
        async with httpx.AsyncClient() as client:
            resp = await client.post(target_url, json = body, timeout = 600)
        return Response(
            content = resp.content,
            status_code = resp.status_code,
            media_type = "application/json",
        )


# =====================================================================
# OpenAI-Compatible Embeddings Proxy  (/embeddings → /v1/embeddings)
# =====================================================================


@router.post("/embeddings")
async def openai_embeddings(
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    OpenAI-compatible embeddings endpoint.

    Transparently proxies to the running llama-server's ``/v1/embeddings``.
    Only available when a GGUF model is loaded.
    Note: the loaded model must support pooling; otherwise llama-server
    will return an error (expected).
    """
    llama_backend = get_llama_cpp_backend()
    if not llama_backend.is_loaded:
        raise HTTPException(
            status_code = 503,
            detail = "No GGUF model loaded. Load a GGUF model first.",
        )

    body = await request.json()
    target_url = f"{llama_backend.base_url}/v1/embeddings"

    async with httpx.AsyncClient() as client:
        resp = await client.post(target_url, json = body, timeout = 600)
    return Response(
        content = resp.content,
        status_code = resp.status_code,
        media_type = "application/json",
    )


# =====================================================================
# OpenAI Responses API  (/responses → /v1/responses)
# =====================================================================


def _translate_responses_tools_to_chat(
    tools: Optional[list[dict]],
) -> Optional[list[dict]]:
    """Translate Responses-shape function tools to the Chat Completions nested shape.

    Responses uses a flat shape per tool entry::

        {"type": "function", "name": "...", "description": "...",
         "parameters": {...}, "strict": true}

    The Chat Completions / llama-server passthrough expects the nested shape::

        {"type": "function",
         "function": {"name": "...", "description": "...",
                      "parameters": {...}, "strict": true}}

    Only ``type=="function"`` entries are forwarded. Built-in Responses tools
    (``web_search``, ``file_search``, ``mcp``, ...) are dropped because
    llama-server does not implement them server-side; keeping them in the
    request would produce an opaque upstream 400.
    """
    if not tools:
        return None
    out: list[dict] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        fn: dict = {}
        if "name" in tool:
            fn["name"] = tool["name"]
        if tool.get("description") is not None:
            fn["description"] = tool["description"]
        if tool.get("parameters") is not None:
            fn["parameters"] = tool["parameters"]
        if tool.get("strict") is not None:
            fn["strict"] = tool["strict"]
        out.append({"type": "function", "function": fn})
    return out or None


def _translate_responses_tool_choice_to_chat(tool_choice: Any) -> Any:
    """Translate a Responses-shape ``tool_choice`` to the Chat Completions shape.

    String values (``"auto"``/``"none"``/``"required"``) pass through unchanged.
    The Responses forcing object ``{"type": "function", "name": "X"}`` is
    converted to Chat Completions' ``{"type": "function", "function": {"name": "X"}}``.
    Unknown / built-in tool choices are forwarded as-is; llama-server ignores
    what it doesn't recognise.
    """
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        return tool_choice
    if (
        isinstance(tool_choice, dict)
        and tool_choice.get("type") == "function"
        and "name" in tool_choice
        and "function" not in tool_choice
    ):
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return tool_choice


def _responses_message_text(content: Union[str, list]) -> str:
    """Flatten a ResponsesInputMessage ``content`` into a plain text string.

    Used for system/developer message hoisting and for assistant-replay
    (``output_text``) messages when images/unknown parts are irrelevant.
    Returns an empty string for empty input.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content or []:
        if isinstance(part, (ResponsesInputTextPart, ResponsesOutputTextPart)):
            parts.append(part.text)
    return "\n".join(parts)


def _normalise_responses_input(payload: ResponsesRequest) -> list[ChatMessage]:
    """Convert a ResponsesRequest's ``input`` into Chat-format ``ChatMessage`` list.

    Handles the three input item shapes allowed by the Responses API:

    - ``ResponsesInputMessage`` — regular chat messages (text or multimodal).
    - ``ResponsesFunctionCallInputItem`` — a prior assistant tool call replayed
      on a follow-up turn. Converted into an assistant message carrying a
      Chat Completions ``tool_calls`` entry keyed by ``call_id``.
    - ``ResponsesFunctionCallOutputInputItem`` — a tool result the client is
      returning. Converted into a ``role="tool"`` message with ``tool_call_id``
      set to the originating ``call_id`` so llama-server can reconcile the
      call with its result.

    System / developer content is collected from ``instructions`` *and* from
    any ``role="system"`` / ``role="developer"`` entries in ``input``, then
    merged into a single ``role="system"`` message placed at the top of the
    returned list. This satisfies strict chat templates (harmony / gpt-oss,
    Qwen3, ...) whose Jinja raises ``"System message must be at the
    beginning."`` when more than one system message is present or when a
    system message appears after a user turn — the exact pattern the OpenAI
    Codex CLI hits, since Codex sets ``instructions`` *and* also sends a
    developer message in ``input``.
    """
    system_parts: list[str] = []
    messages: list[ChatMessage] = []

    if payload.instructions:
        system_parts.append(payload.instructions)

    # Simple string input
    if isinstance(payload.input, str):
        if payload.input:
            messages.append(ChatMessage(role = "user", content = payload.input))
        if system_parts:
            merged = "\n\n".join(p for p in system_parts if p)
            return [ChatMessage(role = "system", content = merged), *messages]
        return messages

    for item in payload.input:
        if isinstance(item, ResponsesFunctionCallInputItem):
            messages.append(
                ChatMessage(
                    role = "assistant",
                    content = None,
                    tool_calls = [
                        {
                            "id": item.call_id,
                            "type": "function",
                            "function": {
                                "name": item.name,
                                "arguments": item.arguments,
                            },
                        }
                    ],
                )
            )
            continue

        if isinstance(item, ResponsesFunctionCallOutputInputItem):
            # Chat Completions `role="tool"` requires a string content; if a
            # Responses client sends a content-array output, serialize it.
            output = item.output
            if not isinstance(output, str):
                output = json.dumps(output)
            messages.append(
                ChatMessage(
                    role = "tool",
                    tool_call_id = item.call_id,
                    content = output,
                )
            )
            continue

        if isinstance(item, ResponsesUnknownInputItem):
            # Reasoning items and any other unmodelled top-level Responses
            # item types are silently dropped — llama-server-backed GGUFs
            # cannot consume them and our lenient validation let them in so
            # unrelated turns don't 422.
            continue

        # ResponsesInputMessage — hoist system/developer to the top, merge.
        if item.role in ("system", "developer"):
            hoisted = _responses_message_text(item.content)
            if hoisted:
                system_parts.append(hoisted)
            continue

        if isinstance(item.content, str):
            messages.append(ChatMessage(role = item.role, content = item.content))
            continue

        # Assistant-replay turns come back as content = [output_text, ...].
        # Chat Completions' assistant role expects a plain string, not a
        # multimodal content array, so flatten output_text (and any stray
        # input_text / unknown text) to a single string.
        if item.role == "assistant":
            text = _responses_message_text(item.content)
            if text:
                messages.append(ChatMessage(role = "assistant", content = text))
            continue

        # User (and any other remaining roles) — keep multimodal when
        # present, drop unknown content parts silently.
        parts: list = []
        for part in item.content:
            if isinstance(part, (ResponsesInputTextPart, ResponsesOutputTextPart)):
                parts.append(TextContentPart(type = "text", text = part.text))
            elif isinstance(part, ResponsesInputImagePart):
                parts.append(
                    ImageContentPart(
                        type = "image_url",
                        image_url = ImageUrl(url = part.image_url, detail = part.detail),
                    )
                )
            # ResponsesUnknownContentPart and anything else: drop.
        if parts:
            # Collapse single-text-part content to a plain string so roles
            # that reject multimodal arrays (e.g. legacy templates) still
            # accept the message.
            if len(parts) == 1 and isinstance(parts[0], TextContentPart):
                messages.append(ChatMessage(role = item.role, content = parts[0].text))
            else:
                messages.append(ChatMessage(role = item.role, content = parts))

    if system_parts:
        merged = "\n\n".join(p for p in system_parts if p)
        return [ChatMessage(role = "system", content = merged), *messages]
    return messages


def _build_chat_request(
    payload: ResponsesRequest, messages: list[ChatMessage], stream: bool
) -> ChatCompletionRequest:
    """Build a ChatCompletionRequest from a ResponsesRequest.

    Tools and ``tool_choice`` are translated from the flat Responses shape to
    the nested Chat Completions shape here so the existing #5099
    ``/v1/chat/completions`` client-side pass-through picks them up without
    further modification.
    """
    chat_kwargs: dict = dict(
        model = payload.model,
        messages = messages,
        stream = stream,
    )
    if payload.temperature is not None:
        chat_kwargs["temperature"] = payload.temperature
    if payload.top_p is not None:
        chat_kwargs["top_p"] = payload.top_p
    if payload.max_output_tokens is not None:
        chat_kwargs["max_tokens"] = payload.max_output_tokens

    chat_tools = _translate_responses_tools_to_chat(payload.tools)
    if chat_tools is not None:
        chat_kwargs["tools"] = chat_tools

    chat_tool_choice = _translate_responses_tool_choice_to_chat(payload.tool_choice)
    if chat_tool_choice is not None:
        chat_kwargs["tool_choice"] = chat_tool_choice

    req = ChatCompletionRequest(**chat_kwargs)
    # `parallel_tool_calls` is not a first-class field on ChatCompletionRequest,
    # but the model allows extras and _build_openai_passthrough_body forwards
    # only explicitly-known fields. Llama-server does not currently implement
    # parallel_tool_calls semantics, so we accept-and-ignore it on the
    # Responses side to avoid breaking SDK clients that always send it.
    return req


def _chat_tool_calls_to_responses_output(tool_calls: list[dict]) -> list[dict]:
    """Map Chat Completions ``tool_calls`` into Responses ``function_call`` output items.

    The Chat Completions id (``call_xxx``) is the shared correlation key across
    turns in the OpenAI Responses API — it is stored as ``call_id`` on the
    output item and must be echoed back by the client as
    ``function_call_output.call_id`` on the next turn.
    """
    items: list[dict] = []
    for tc in tool_calls:
        if tc.get("type") != "function":
            continue
        fn = tc.get("function") or {}
        items.append(
            ResponsesOutputFunctionCall(
                call_id = tc.get("id", ""),
                name = fn.get("name", ""),
                arguments = fn.get("arguments", "") or "",
                status = "completed",
            ).model_dump()
        )
    return items


async def _responses_non_streaming(
    payload: ResponsesRequest,
    messages: list[ChatMessage],
    request: Request,
) -> JSONResponse:
    """Handle a non-streaming Responses API call."""
    chat_req = _build_chat_request(payload, messages, stream = False)
    result = await openai_chat_completions(chat_req, request)

    # openai_chat_completions returns a JSONResponse for non-streaming
    if isinstance(result, JSONResponse):
        body = json.loads(result.body.decode())
    elif isinstance(result, Response):
        body = json.loads(result.body.decode())
    else:
        body = result

    choices = body.get("choices", [])
    text = ""
    tool_calls: list[dict] = []
    if choices:
        msg = choices[0].get("message", {}) or {}
        text = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls") or []

    usage_data = body.get("usage", {})
    input_tokens = usage_data.get("prompt_tokens", 0)
    output_tokens = usage_data.get("completion_tokens", 0)

    resp_id = f"resp_{uuid.uuid4().hex[:12]}"

    # Responses API emits each tool call as its own top-level output item,
    # alongside an optional assistant text message. Emit the text message
    # only when the model actually produced content, so clients that expect
    # a pure tool-call turn (finish_reason="tool_calls") don't see a spurious
    # empty message item.
    output_items: list[dict] = []
    if text:
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        output_items.append(
            ResponsesOutputMessage(
                id = msg_id,
                status = "completed",
                role = "assistant",
                content = [ResponsesOutputTextContent(text = text)],
            ).model_dump()
        )
    output_items.extend(_chat_tool_calls_to_responses_output(tool_calls))

    response = ResponsesResponse(
        id = resp_id,
        created_at = int(time.time()),
        status = "completed",
        model = body.get("model", payload.model),
        output = output_items,
        usage = ResponsesUsage(
            input_tokens = input_tokens,
            output_tokens = output_tokens,
            total_tokens = input_tokens + output_tokens,
        ),
        temperature = payload.temperature,
        top_p = payload.top_p,
        max_output_tokens = payload.max_output_tokens,
        instructions = payload.instructions,
    )
    return JSONResponse(content = response.model_dump())


async def _responses_stream(
    payload: ResponsesRequest,
    messages: list[ChatMessage],
    request: Request,
):
    """Handle a streaming Responses API call, emitting named SSE events.

    For GGUF models the request goes directly to llama-server's
    ``/v1/chat/completions`` endpoint from inside the StreamingResponse
    child task — a single httpx lifecycle, a single async generator.
    Wrapping the existing ``openai_chat_completions`` pass-through (which
    already does its own httpx lifecycle) stacks two generators: Python
    3.13 + httpcore 1.0.x then loses the close-propagation chain on the
    innermost ``HTTP11ConnectionByteStream`` at asyncgen finalisation,
    tripping "Attempted to exit cancel scope in a different task" /
    "async generator ignored GeneratorExit". The direct path avoids that
    altogether. Non-GGUF falls back to the wrapper (which doesn't use
    httpx, so the issue doesn't apply).

    Text deltas arrive as ``response.output_text.delta`` on a single
    ``message`` output item at ``output_index=0``. Each tool call from
    ``delta.tool_calls[]`` is promoted to its own top-level ``function_call``
    output item (one per distinct ``tool_calls[].index``), and relayed as
    ``response.function_call_arguments.delta`` / ``.done`` events so clients
    (Codex, OpenAI Python SDK) can reconstruct the call incrementally and
    reply with a ``function_call_output`` item on the next turn.
    """
    resp_id = f"resp_{uuid.uuid4().hex[:12]}"
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    created_at = int(time.time())

    chat_req = _build_chat_request(payload, messages, stream = True)

    llama_backend = get_llama_cpp_backend()
    if not llama_backend.is_loaded:
        # The direct pass-through is GGUF-only. Non-GGUF /v1/responses
        # streaming isn't a Codex-compatible path today and wrapping the
        # transformers backend's streaming generator here would re-
        # introduce the double-layer asyncgen close pattern that produces
        # "Attempted to exit cancel scope in a different task" on Python
        # 3.13. Surface a typed 400 so the client sees a useful error
        # instead of a dangling stream.
        raise HTTPException(
            status_code = 400,
            detail = (
                "Streaming /v1/responses requires a GGUF model loaded via "
                "llama-server. Use non-streaming /v1/responses, "
                "/v1/chat/completions, or load a GGUF model."
            ),
        )

    body = _build_openai_passthrough_body(
        chat_req, backend_ctx = llama_backend.context_length
    )
    target_url = f"{llama_backend.base_url}/v1/chat/completions"

    async def event_generator():
        full_text = ""
        input_tokens = 0
        output_tokens = 0
        # Per-tool-call state keyed by the Chat Completions `tool_calls[].index`
        # which stays stable across chunks for the same call. Values are:
        #   {output_index, item_id, call_id, name, arguments, opened}
        tool_call_state: dict[int, dict] = {}
        # Text message lives at output_index 0; tool calls claim 1, 2, ...
        next_output_index = 1

        def _snapshot_output() -> list[dict]:
            """Snapshot of all completed output items for response.completed."""
            items: list[dict] = [
                {
                    "type": "message",
                    "id": msg_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": full_text,
                            "annotations": [],
                        }
                    ],
                }
            ]
            for st in sorted(tool_call_state.values(), key = lambda s: s["output_index"]):
                items.append(
                    {
                        "type": "function_call",
                        "id": st["item_id"],
                        "status": "completed",
                        "call_id": st["call_id"],
                        "name": st["name"],
                        "arguments": st["arguments"],
                    }
                )
            return items

        # ── Preamble events ──
        yield f"event: response.created\ndata: {json.dumps({'type': 'response.created', 'response': {'id': resp_id, 'object': 'response', 'created_at': created_at, 'status': 'in_progress', 'model': payload.model, 'output': [], 'usage': {'input_tokens': 0, 'output_tokens': 0, 'total_tokens': 0}}})}\n\n"

        # output_item.added (text message at output_index 0)
        output_item = {
            "type": "message",
            "id": msg_id,
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        yield f"event: response.output_item.added\ndata: {json.dumps({'type': 'response.output_item.added', 'output_index': 0, 'item': output_item})}\n\n"

        # content_part.added
        content_part = {"type": "output_text", "text": "", "annotations": []}
        yield f"event: response.content_part.added\ndata: {json.dumps({'type': 'response.content_part.added', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'part': content_part})}\n\n"

        # ── Direct httpx lifecycle to llama-server ──
        # Full same-task open + close, identical pattern to
        # _openai_passthrough_stream and _anthropic_passthrough_stream:
        # no `async with`, explicit aclose of lines_iter BEFORE resp /
        # client so the innermost httpcore byte stream is finalised in
        # this task (not via Python's asyncgen GC in a sibling task).
        client = httpx.AsyncClient(timeout = 600)
        resp = None
        lines_iter = None
        try:
            req = client.build_request("POST", target_url, json = body)
            try:
                resp = await client.send(req, stream = True)
            except httpx.RequestError as e:
                logger.error("responses stream: upstream unreachable: %s", e)
                yield f"event: response.failed\ndata: {json.dumps({'type': 'response.failed', 'response': {'id': resp_id, 'object': 'response', 'created_at': created_at, 'status': 'failed', 'model': payload.model, 'output': [], 'error': {'code': 502, 'message': _friendly_error(e)}}})}\n\n"
                return

            if resp.status_code != 200:
                err_bytes = await resp.aread()
                err_text = err_bytes.decode("utf-8", errors = "replace")
                logger.error(
                    "responses stream upstream error: status=%s body=%s",
                    resp.status_code,
                    err_text[:500],
                )
                yield f"event: response.failed\ndata: {json.dumps({'type': 'response.failed', 'response': {'id': resp_id, 'object': 'response', 'created_at': created_at, 'status': 'failed', 'model': payload.model, 'output': [], 'error': {'code': resp.status_code, 'message': f'llama-server error: {err_text[:500]}'}}})}\n\n"
                return

            lines_iter = resp.aiter_lines()
            async for raw_line in lines_iter:
                if await request.is_disconnected():
                    break
                if not raw_line:
                    continue
                if not raw_line.startswith("data: "):
                    continue
                data_str = raw_line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk_data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk_data.get("choices", [])
                if not choices:
                    usage = chunk_data.get("usage")
                    if usage:
                        input_tokens = usage.get("prompt_tokens", input_tokens)
                        output_tokens = usage.get("completion_tokens", output_tokens)
                    continue

                delta = choices[0].get("delta", {}) or {}
                content = delta.get("content")
                if content:
                    full_text += content
                    delta_event = {
                        "type": "response.output_text.delta",
                        "item_id": msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": content,
                    }
                    yield f"event: response.output_text.delta\ndata: {json.dumps(delta_event)}\n\n"

                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index", 0)
                    st = tool_call_state.get(idx)
                    fn = tc.get("function") or {}
                    if st is None:
                        # First chunk for this tool call — allocate an
                        # output_index and emit output_item.added.
                        st = {
                            "output_index": next_output_index,
                            "item_id": f"fc_{uuid.uuid4().hex[:12]}",
                            "call_id": tc.get("id") or "",
                            "name": fn.get("name") or "",
                            "arguments": "",
                            "opened": False,
                        }
                        next_output_index += 1
                        tool_call_state[idx] = st
                    else:
                        # Later chunks sometimes carry the id/name only
                        # once; merge when present.
                        if tc.get("id") and not st["call_id"]:
                            st["call_id"] = tc["id"]
                        if fn.get("name") and not st["name"]:
                            st["name"] = fn["name"]

                    if not st["opened"] and st["call_id"] and st["name"]:
                        item_added = {
                            "type": "response.output_item.added",
                            "output_index": st["output_index"],
                            "item": {
                                "type": "function_call",
                                "id": st["item_id"],
                                "status": "in_progress",
                                "call_id": st["call_id"],
                                "name": st["name"],
                                "arguments": "",
                            },
                        }
                        yield f"event: response.output_item.added\ndata: {json.dumps(item_added)}\n\n"
                        st["opened"] = True

                    arg_delta = fn.get("arguments") or ""
                    if arg_delta and st["opened"]:
                        st["arguments"] += arg_delta
                        args_delta_event = {
                            "type": "response.function_call_arguments.delta",
                            "item_id": st["item_id"],
                            "output_index": st["output_index"],
                            "delta": arg_delta,
                        }
                        yield f"event: response.function_call_arguments.delta\ndata: {json.dumps(args_delta_event)}\n\n"
                    elif arg_delta:
                        # Buffer the args until we can open the item
                        # (id/name arrive in the same chunk as the first
                        # arg delta for some models — but if not, stash).
                        st["arguments"] += arg_delta

                usage = chunk_data.get("usage")
                if usage:
                    input_tokens = usage.get("prompt_tokens", input_tokens)
                    output_tokens = usage.get("completion_tokens", output_tokens)
        except Exception as e:
            logger.error("responses stream error: %s", e)
        finally:
            if lines_iter is not None:
                try:
                    await lines_iter.aclose()
                except Exception:
                    pass
            if resp is not None:
                try:
                    await resp.aclose()
                except Exception:
                    pass
            try:
                await client.aclose()
            except Exception:
                pass

        # ── Closing events for tool calls ──
        for st in sorted(tool_call_state.values(), key = lambda s: s["output_index"]):
            # If id/name never arrived (malformed upstream), synthesise so
            # the client still sees a coherent frame sequence.
            if not st["opened"]:
                if not st["call_id"]:
                    st["call_id"] = f"call_{uuid.uuid4().hex[:12]}"
                item_added = {
                    "type": "response.output_item.added",
                    "output_index": st["output_index"],
                    "item": {
                        "type": "function_call",
                        "id": st["item_id"],
                        "status": "in_progress",
                        "call_id": st["call_id"],
                        "name": st["name"],
                        "arguments": "",
                    },
                }
                yield f"event: response.output_item.added\ndata: {json.dumps(item_added)}\n\n"
                if st["arguments"]:
                    yield (
                        "event: response.function_call_arguments.delta\n"
                        "data: "
                        + json.dumps(
                            {
                                "type": "response.function_call_arguments.delta",
                                "item_id": st["item_id"],
                                "output_index": st["output_index"],
                                "delta": st["arguments"],
                            }
                        )
                        + "\n\n"
                    )
                st["opened"] = True

            args_done = {
                "type": "response.function_call_arguments.done",
                "item_id": st["item_id"],
                "output_index": st["output_index"],
                "name": st["name"],
                "arguments": st["arguments"],
            }
            yield f"event: response.function_call_arguments.done\ndata: {json.dumps(args_done)}\n\n"

            item_done = {
                "type": "response.output_item.done",
                "output_index": st["output_index"],
                "item": {
                    "type": "function_call",
                    "id": st["item_id"],
                    "status": "completed",
                    "call_id": st["call_id"],
                    "name": st["name"],
                    "arguments": st["arguments"],
                },
            }
            yield f"event: response.output_item.done\ndata: {json.dumps(item_done)}\n\n"

        # ── Closing events for text message ──
        yield f"event: response.output_text.done\ndata: {json.dumps({'type': 'response.output_text.done', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'text': full_text})}\n\n"

        yield f"event: response.content_part.done\ndata: {json.dumps({'type': 'response.content_part.done', 'item_id': msg_id, 'output_index': 0, 'content_index': 0, 'part': {'type': 'output_text', 'text': full_text, 'annotations': []}})}\n\n"

        yield f"event: response.output_item.done\ndata: {json.dumps({'type': 'response.output_item.done', 'output_index': 0, 'item': {'type': 'message', 'id': msg_id, 'status': 'completed', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': full_text, 'annotations': []}]}})}\n\n"

        # response.completed
        total_tokens = input_tokens + output_tokens
        completed_response = {
            "type": "response.completed",
            "response": {
                "id": resp_id,
                "object": "response",
                "created_at": created_at,
                "status": "completed",
                "model": payload.model,
                "output": _snapshot_output(),
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            },
        }
        yield f"event: response.completed\ndata: {json.dumps(completed_response)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/responses")
async def openai_responses(
    payload: ResponsesRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    OpenAI Responses API endpoint.

    Accepts the Responses-format request, converts it to a
    ChatCompletionRequest internally, and returns a response
    matching the OpenAI Responses API schema (output array,
    input_tokens/output_tokens, named SSE events for streaming).
    """
    messages = _normalise_responses_input(payload)
    if not messages:
        raise HTTPException(status_code = 400, detail = "No input provided.")

    if payload.stream:
        return await _responses_stream(payload, messages, request)
    return await _responses_non_streaming(payload, messages, request)


# =====================================================================
# Anthropic-Compatible Messages API  (/messages → /v1/messages)
# =====================================================================


def _normalize_anthropic_openai_images(
    openai_messages: list[dict], is_vision: bool
) -> bool:
    """Enforce the vision guard on translated Anthropic messages and
    normalize any ``image_url`` parts with base64 data URLs to PNG.

    llama-server's stb_image only handles a few formats (JPEG/PNG/BMP/…);
    Anthropic clients commonly send JPEG or WebP, and Claude Code sends
    WebP. Re-encoding everything to PNG mirrors the behavior of
    `_openai_messages_for_passthrough` / the GGUF branch of
    `/v1/chat/completions` so the two endpoints agree.

    Mutates ``openai_messages`` in place. Returns ``True`` when any
    image part was seen (so the caller can skip a second scan). Raises
    HTTPException(400) when images are present but the active model is
    not a vision model, or when an image cannot be decoded.
    """
    from PIL import Image

    has_image = False
    for msg in openai_messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") != "image_url":
                continue

            has_image = True
            if not is_vision:
                raise HTTPException(
                    status_code = 400,
                    detail = "Image provided but current GGUF model does not support vision.",
                )

            url = (part.get("image_url") or {}).get("url", "")
            if not url.startswith("data:"):
                # Remote URLs are forwarded as-is; llama-server will
                # fetch (or fail) per its own support matrix.
                continue

            try:
                _, b64data = url.split(",", 1)
                raw = base64.b64decode(b64data)
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format = "PNG")
                png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            except Exception:
                raise HTTPException(
                    status_code = 400,
                    detail = "Failed to process image.",
                )
            part["image_url"] = {"url": f"data:image/png;base64,{png_b64}"}

    return has_image


@router.post("/messages")
async def anthropic_messages(
    payload: AnthropicMessagesRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    Anthropic-compatible Messages API endpoint.

    Translates Anthropic message format to internal OpenAI format, runs
    through the existing agentic tool loop when tools are provided, and
    returns responses in Anthropic Messages API format (streaming SSE or
    non-streaming JSON).
    """
    llama_backend = get_llama_cpp_backend()
    if not llama_backend.is_loaded:
        raise HTTPException(
            status_code = 503,
            detail = "No GGUF model loaded. Load a GGUF model first.",
        )

    model_name = getattr(llama_backend, "model_identifier", None) or payload.model
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # ── Translate Anthropic → OpenAI ──────────────────────────
    openai_messages = anthropic_messages_to_openai(
        [m.model_dump() for m in payload.messages],
        payload.system,
    )
    openai_messages = _drop_empty_assistant_sentinels(openai_messages)

    # Enforce vision guard + re-encode embedded images to PNG so the
    # Anthropic endpoint matches the behavior of /v1/chat/completions.
    _has_image = _normalize_anthropic_openai_images(
        openai_messages, llama_backend.is_vision
    )

    temperature = payload.temperature if payload.temperature is not None else 0.6
    top_p = payload.top_p if payload.top_p is not None else 0.95
    top_k = payload.top_k if payload.top_k is not None else 20
    min_p = payload.min_p if payload.min_p is not None else 0.01
    repetition_penalty = (
        payload.repetition_penalty if payload.repetition_penalty is not None else 1.0
    )
    presence_penalty = (
        payload.presence_penalty if payload.presence_penalty is not None else 0.0
    )
    stop = payload.stop_sequences or None

    # Translate Anthropic tool_choice to OpenAI format for forwarding to
    # llama-server. Falls back to "auto" when unset or unrecognized, which
    # matches the prior hardcoded behavior.
    openai_tool_choice = anthropic_tool_choice_to_openai(payload.tool_choice)
    if openai_tool_choice is None:
        openai_tool_choice = "auto"

    cancel_event = threading.Event()

    # ── Tool routing ──────────────────────────────────────────
    # Three paths:
    # 1. enable_tools=true → server-side execution of built-in tools (Unsloth shorthand)
    # 2. tools=[...] only  → client-side pass-through (standard Anthropic behavior)
    # 3. neither           → plain chat
    # Server-side agentic loop doesn't support multimodal input — matches
    # the `not image_b64` gate in /v1/chat/completions.
    server_tools = (
        _effective_enable_tools(payload)
        and llama_backend.supports_tools
        and not _has_image
    )
    client_tools = (
        not server_tools
        and payload.tools
        and len(payload.tools) > 0
        and llama_backend.supports_tools
    )

    # ── Client-side pass-through path ─────────────────────────
    if client_tools:
        openai_tools = anthropic_tools_to_openai(payload.tools)

        if payload.stream:
            return await _anthropic_passthrough_stream(
                request,
                cancel_event,
                llama_backend,
                openai_messages,
                openai_tools,
                temperature,
                top_p,
                top_k,
                payload.max_tokens,
                message_id,
                model_name,
                stop = stop,
                min_p = min_p,
                repetition_penalty = repetition_penalty,
                presence_penalty = presence_penalty,
                tool_choice = openai_tool_choice,
                session_id = payload.session_id,
                cancel_id = payload.cancel_id,
            )
        return await _anthropic_passthrough_non_streaming(
            llama_backend,
            openai_messages,
            openai_tools,
            temperature,
            top_p,
            top_k,
            payload.max_tokens,
            message_id,
            model_name,
            stop = stop,
            min_p = min_p,
            repetition_penalty = repetition_penalty,
            presence_penalty = presence_penalty,
            tool_choice = openai_tool_choice,
        )

    if server_tools:
        from core.inference.tools import ALL_TOOLS

        if payload.enabled_tools is not None:
            openai_tools = [
                t for t in ALL_TOOLS if t["function"]["name"] in payload.enabled_tools
            ]
        else:
            openai_tools = ALL_TOOLS

        # Build tool-use system prompt nudge (same logic as /chat/completions)
        _tool_names = {t["function"]["name"] for t in openai_tools}
        _has_web = "web_search" in _tool_names
        _has_code = "python" in _tool_names or "terminal" in _tool_names

        _date_line = f"The current date is {_date.today().isoformat()}."
        _model_size_b = _extract_model_size_b(model_name)
        _is_small_model = _model_size_b is not None and _model_size_b < 9

        if _is_small_model:
            _web_tips = "Do not repeat the same search query."
        else:
            _web_tips = (
                "When you search and find a relevant URL in the results, "
                "fetch its full content by calling web_search with the url parameter. "
                "Do not repeat the same search query. If a search returns "
                "no useful results, try rephrasing or fetching a result URL directly."
            )
        _code_tips = (
            "Use code execution for math, calculations, data processing, "
            "or to parse and analyze information from tool results."
        )

        if _has_web and _has_code:
            _nudge = (
                _date_line + " "
                "You have access to tools. When appropriate, prefer using "
                "tools rather than answering from memory. "
                + _web_tips
                + " "
                + _code_tips
            )
        elif _has_code:
            _nudge = (
                _date_line + " "
                "You have access to tools. When appropriate, prefer using "
                "code execution rather than answering from memory. " + _code_tips
            )
        elif _has_web:
            _nudge = (
                _date_line + " "
                "You have access to tools. When appropriate, prefer using "
                "web search for up-to-date or uncertain factual "
                "information rather than answering from memory. " + _web_tips
            )
        else:
            _nudge = ""

        if _nudge:
            _nudge += _TOOL_ACTION_NUDGE
            # Inject into system prompt
            if openai_messages and openai_messages[0].get("role") == "system":
                openai_messages[0]["content"] = (
                    openai_messages[0]["content"].rstrip() + "\n\n" + _nudge
                )
            else:
                openai_messages.insert(0, {"role": "system", "content": _nudge})

        # Strip stale tool-call XML from conversation
        for _msg in openai_messages:
            if _msg.get("role") == "assistant" and isinstance(_msg.get("content"), str):
                _msg["content"] = _TOOL_XML_RE.sub("", _msg["content"]).strip()

        def _run_tool_gen():
            return llama_backend.generate_chat_completion_with_tools(
                messages = openai_messages,
                tools = openai_tools,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
                repetition_penalty = repetition_penalty,
                presence_penalty = presence_penalty,
                max_tokens = payload.max_tokens,
                stop = stop,
                cancel_event = cancel_event,
                max_tool_iterations = 25,
                auto_heal_tool_calls = True,
                tool_call_timeout = 300,
                session_id = payload.session_id,
            )

        if payload.stream:
            return await _anthropic_tool_stream(
                request,
                cancel_event,
                _run_tool_gen,
                message_id,
                model_name,
            )
        return await _anthropic_tool_non_streaming(
            _run_tool_gen,
            message_id,
            model_name,
        )

    # ── No-tool path ──────────────────────────────────────────
    def _run_plain_gen():
        return llama_backend.generate_chat_completion(
            messages = openai_messages,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            min_p = min_p,
            repetition_penalty = repetition_penalty,
            presence_penalty = presence_penalty,
            max_tokens = payload.max_tokens,
            stop = stop,
            cancel_event = cancel_event,
        )

    if payload.stream:
        return await _anthropic_plain_stream(
            request,
            cancel_event,
            _run_plain_gen,
            message_id,
            model_name,
        )
    return await _anthropic_plain_non_streaming(
        _run_plain_gen,
        message_id,
        model_name,
    )


async def _anthropic_tool_stream(
    request,
    cancel_event,
    run_gen,
    message_id,
    model_name,
):
    """Streaming response for the tool-calling path."""
    _sentinel = object()

    async def _stream():
        emitter = AnthropicStreamEmitter()
        for line in emitter.start(message_id, model_name):
            yield line

        gen = run_gen()
        try:
            while True:
                if await request.is_disconnected():
                    cancel_event.set()
                    return
                event = await asyncio.to_thread(next, gen, _sentinel)
                if event is _sentinel:
                    break
                # Strip leaked tool-call XML from content events
                if event.get("type") == "content":
                    event = dict(event)
                    event["text"] = _TOOL_XML_RE.sub("", event["text"])
                for line in emitter.feed(event):
                    yield line
        except Exception as e:
            logger.error("anthropic_messages stream error: %s", e)

        for line in emitter.finish("end_turn"):
            yield line

    return StreamingResponse(
        _stream(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _anthropic_plain_stream(
    request,
    cancel_event,
    run_gen,
    message_id,
    model_name,
):
    """Streaming response for the no-tool path."""
    _sentinel = object()

    async def _stream():
        emitter = AnthropicStreamEmitter()
        for line in emitter.start(message_id, model_name):
            yield line

        gen = run_gen()
        try:
            while True:
                if await request.is_disconnected():
                    cancel_event.set()
                    return
                cumulative = await asyncio.to_thread(next, gen, _sentinel)
                if cumulative is _sentinel:
                    break
                if isinstance(cumulative, dict):
                    if cumulative.get("type") == "metadata":
                        for line in emitter.feed(cumulative):
                            yield line
                    continue
                # Plain generator yields cumulative text strings
                for line in emitter.feed({"type": "content", "text": cumulative}):
                    yield line
        except Exception as e:
            logger.error("anthropic_messages stream error: %s", e)

        for line in emitter.finish("end_turn"):
            yield line

    return StreamingResponse(
        _stream(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _anthropic_tool_non_streaming(run_gen, message_id, model_name):
    """Non-streaming response for the tool-calling path.

    Builds ``content_blocks`` in generation order (text → tool_use → text →
    tool_use → ...), mirroring the streaming emitter's behavior. Deltas
    within a single synthesis turn are merged into the trailing text block;
    tool_use blocks interrupt the text sequence and open a new text block on
    the next content event.

    ``prev_text`` is reset on ``tool_end`` because
    ``generate_chat_completion_with_tools`` yields cumulative content *per
    turn* — the first content event of turn N+1 must diff against an empty
    baseline, not against turn N's final length.
    """
    content_blocks: list = []
    usage = {}
    prev_text = ""

    for event in run_gen():
        etype = event.get("type", "")
        if etype == "content":
            # Strip leaked tool-call XML
            clean = _TOOL_XML_RE.sub("", event["text"])
            new = clean[len(prev_text) :]
            prev_text = clean
            if new:
                if content_blocks and isinstance(
                    content_blocks[-1], AnthropicResponseTextBlock
                ):
                    content_blocks[-1].text += new
                else:
                    content_blocks.append(AnthropicResponseTextBlock(text = new))
        elif etype == "tool_start":
            content_blocks.append(
                AnthropicResponseToolUseBlock(
                    id = event["tool_call_id"],
                    name = event["tool_name"],
                    input = event.get("arguments", {}),
                )
            )
        elif etype == "tool_end":
            prev_text = ""
        elif etype == "metadata":
            usage = event.get("usage", {})

    resp = AnthropicMessagesResponse(
        id = message_id,
        model = model_name,
        content = content_blocks,
        stop_reason = "end_turn",
        usage = AnthropicUsage(
            input_tokens = usage.get("prompt_tokens", 0),
            output_tokens = usage.get("completion_tokens", 0),
        ),
    )
    return JSONResponse(content = resp.model_dump())


async def _anthropic_plain_non_streaming(run_gen, message_id, model_name):
    """Non-streaming response for the no-tool path."""
    text_parts = []
    usage = {}
    prev_text = ""

    for cumulative in run_gen():
        if isinstance(cumulative, dict):
            if cumulative.get("type") == "metadata":
                usage = cumulative.get("usage", {})
            continue
        new = cumulative[len(prev_text) :]
        prev_text = cumulative
        if new:
            text_parts.append(new)

    full_text = "".join(text_parts)
    content_blocks = []
    if full_text:
        content_blocks.append(AnthropicResponseTextBlock(text = full_text))

    resp = AnthropicMessagesResponse(
        id = message_id,
        model = model_name,
        content = content_blocks,
        stop_reason = "end_turn",
        usage = AnthropicUsage(
            input_tokens = usage.get("prompt_tokens", 0),
            output_tokens = usage.get("completion_tokens", 0),
        ),
    )
    return JSONResponse(content = resp.model_dump())


# =====================================================================
# Client-side tool pass-through (Anthropic-native tools field)
# =====================================================================


def _build_passthrough_payload(
    openai_messages,
    openai_tools,
    temperature,
    top_p,
    top_k,
    max_tokens,
    stream,
    stop = None,
    min_p = None,
    repetition_penalty = None,
    presence_penalty = None,
    tool_choice = "auto",
    response_format = None,
    chat_template_kwargs = None,
    backend_ctx = None,
):
    body = {
        "messages": openai_messages,
        "tools": openai_tools,
        "tool_choice": tool_choice,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    body["max_tokens"] = (
        max_tokens
        if max_tokens is not None
        else (backend_ctx or _DEFAULT_MAX_TOKENS_FLOOR)
    )
    body["t_max_predict_ms"] = _DEFAULT_T_MAX_PREDICT_MS
    if stop:
        body["stop"] = stop
    if min_p is not None:
        body["min_p"] = min_p
    if repetition_penalty is not None:
        # llama-server's field is "repeat_penalty", not "repetition_penalty"
        body["repeat_penalty"] = repetition_penalty
    if presence_penalty is not None:
        body["presence_penalty"] = presence_penalty
    if response_format is not None:
        # llama-server applies a GBNF grammar derived from the JSON schema
        # when response_format is present. Field is documented flat at the
        # request root (tools/server/README.md), which is also what the
        # OpenAI SDK produces by spreading extra_body into the body top.
        body["response_format"] = response_format
    if chat_template_kwargs is not None:
        # Propagate reasoning / template overrides (e.g. enable_thinking)
        # so llama-server renders the Jinja template in the mode the caller
        # asked for instead of whatever default the model was loaded with.
        body["chat_template_kwargs"] = chat_template_kwargs
    return body


async def _anthropic_passthrough_stream(
    request,
    cancel_event,
    llama_backend,
    openai_messages,
    openai_tools,
    temperature,
    top_p,
    top_k,
    max_tokens,
    message_id,
    model_name,
    stop = None,
    min_p = None,
    repetition_penalty = None,
    presence_penalty = None,
    tool_choice = "auto",
    session_id = None,
    cancel_id = None,
):
    """Streaming client-side pass-through: forward tools to llama-server and
    translate its streaming response to Anthropic SSE without executing anything."""
    target_url = f"{llama_backend.base_url}/v1/chat/completions"
    body = _build_passthrough_payload(
        openai_messages,
        openai_tools,
        temperature,
        top_p,
        top_k,
        max_tokens,
        True,
        stop = stop,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        tool_choice = tool_choice,
        backend_ctx = llama_backend.context_length,
    )

    # cancel_id mirrors the OpenAI passthrough so a per-run cancel POST
    # works without the caller having to know the local message_id.
    _tracker = _TrackedCancel(cancel_event, cancel_id, session_id, message_id)
    _tracker.__enter__()

    async def _stream():
        emitter = AnthropicPassthroughEmitter()
        for line in emitter.start(message_id, model_name):
            yield line

        # Manage the httpx client, response, AND the aiter_lines() async
        # generator MANUALLY — no `async with`, no anonymous iterator.
        #
        # On Python 3.13 + httpcore 1.0.x, `async for raw_line in
        # resp.aiter_lines():` creates an anonymous async generator. When
        # the loop exits via `break` (or the generator is orphaned when a
        # client disconnects mid-stream), Python's `async for` protocol
        # does NOT auto-close the iterator the way a sync `for` loop
        # would. The iterator remains reachable only from the current
        # coroutine frame; once `_stream()` returns, the frame is GC'd
        # and the iterator becomes unreachable. Python's asyncgen
        # finalizer hook then runs its aclose() on a LATER GC pass in a
        # DIFFERENT asyncio task, where httpcore's
        # `HTTP11ConnectionByteStream.aclose()` enters
        # `anyio.CancelScope.__exit__` with a mismatched task and prints
        # `RuntimeError: Attempted to exit cancel scope in a different
        # task` / `RuntimeError: async generator ignored GeneratorExit`
        # as "Exception ignored in:" unraisable warnings.
        #
        # The fix: save `resp.aiter_lines()` as `lines_iter`, and in the
        # finally block explicitly `await lines_iter.aclose()` BEFORE
        # `resp.aclose()` / `client.aclose()`. This closes the iterator
        # inside our own task's event loop, so the internal httpcore
        # byte-stream is cleaned up before Python's asyncgen finalizer
        # has anything orphaned to finalize. Each aclose is wrapped in
        # `try: ... except Exception: pass` so anyio cleanup noise from
        # nested aclose paths can't bubble out.
        client = httpx.AsyncClient(
            timeout = 600,
            limits = httpx.Limits(max_keepalive_connections = 0),
        )
        resp = None
        lines_iter = None
        cancel_watcher = None
        try:
            req = client.build_request("POST", target_url, json = body)
            resp = await client.send(req, stream = True)

            # See _openai_passthrough_stream for rationale: aiter_lines()
            # blocks during llama-server prefill, so the in-loop cancel
            # check is unreachable until the first SSE chunk arrives.
            # The watcher closes `resp` on cancel, raising in aiter_lines.
            cancel_watcher = asyncio.create_task(
                _await_cancel_then_close(cancel_event, resp)
            )
            lines_iter = resp.aiter_lines()
            async for raw_line in lines_iter:
                if cancel_event.is_set():
                    break
                if await request.is_disconnected():
                    cancel_event.set()
                    break
                if not raw_line or not raw_line.startswith("data: "):
                    continue
                data_str = raw_line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                for line in emitter.feed_chunk(chunk):
                    yield line
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.CloseError):
            if not cancel_event.is_set():
                raise
        except Exception as e:
            logger.error("anthropic_messages passthrough stream error: %s", e)
        finally:
            if cancel_watcher is not None:
                cancel_watcher.cancel()
                try:
                    await cancel_watcher
                except (asyncio.CancelledError, Exception):
                    pass
            if lines_iter is not None:
                try:
                    await lines_iter.aclose()
                except Exception:
                    pass
            if resp is not None:
                try:
                    await resp.aclose()
                except Exception:
                    pass
            try:
                await client.aclose()
            except Exception:
                pass
            _tracker.__exit__(None, None, None)

        for line in emitter.finish():
            yield line

    return StreamingResponse(
        _stream(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _anthropic_passthrough_non_streaming(
    llama_backend,
    openai_messages,
    openai_tools,
    temperature,
    top_p,
    top_k,
    max_tokens,
    message_id,
    model_name,
    stop = None,
    min_p = None,
    repetition_penalty = None,
    presence_penalty = None,
    tool_choice = "auto",
):
    """Non-streaming client-side pass-through."""
    target_url = f"{llama_backend.base_url}/v1/chat/completions"
    body = _build_passthrough_payload(
        openai_messages,
        openai_tools,
        temperature,
        top_p,
        top_k,
        max_tokens,
        False,
        stop = stop,
        min_p = min_p,
        repetition_penalty = repetition_penalty,
        presence_penalty = presence_penalty,
        tool_choice = tool_choice,
        backend_ctx = llama_backend.context_length,
    )

    async with httpx.AsyncClient() as client:
        resp = await client.post(target_url, json = body, timeout = 600)

    if resp.status_code != 200:
        raise HTTPException(
            status_code = resp.status_code,
            detail = f"llama-server error: {resp.text[:500]}",
        )

    data = resp.json()
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")

    content_blocks = []
    text = message.get("content") or ""
    if text:
        text = _TOOL_XML_RE.sub("", text).strip()
        if text:
            content_blocks.append(AnthropicResponseTextBlock(text = text))

    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        content_blocks.append(
            AnthropicResponseToolUseBlock(
                id = tc.get("id", ""),
                name = fn.get("name", ""),
                input = args,
            )
        )

    if tool_calls:
        stop_reason = "tool_use"
    elif finish_reason == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    usage = data.get("usage") or {}
    resp_obj = AnthropicMessagesResponse(
        id = message_id,
        model = model_name,
        content = content_blocks,
        stop_reason = stop_reason,
        usage = AnthropicUsage(
            input_tokens = usage.get("prompt_tokens", 0),
            output_tokens = usage.get("completion_tokens", 0),
        ),
    )
    return JSONResponse(content = resp_obj.model_dump())


# =====================================================================
# Client-side tool pass-through (OpenAI-native /v1/chat/completions)
# =====================================================================


def _drop_empty_assistant_sentinels(messages: list[dict]) -> list[dict]:
    """Drop bare ``{"role":"assistant"}`` Stop-button sentinels; passthrough backends reject them."""
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "assistant":
            has_content = bool(m.get("content"))
            has_tool_calls = bool(m.get("tool_calls"))
            if not has_content and not has_tool_calls:
                continue
        out.append(m)
    return out


def _openai_messages_for_passthrough(payload) -> list[dict]:
    """Build OpenAI-format message dicts for the /v1/chat/completions
    passthrough path.

    Messages from ``payload.messages`` are dumped through Pydantic (dropping
    unset optional fields) so they are already in standard OpenAI format
    — including ``role="tool"`` tool-result messages and assistant messages
    that carry structured ``tool_calls``. Content-parts images already in
    the message list are left untouched.

    When a client uses Studio's legacy ``image_base64`` top-level field, the
    image is re-encoded to PNG (llama-server's stb_image has limited format
    support) and spliced into the last user message as an OpenAI
    ``image_url`` content part so vision + function-calling requests work
    transparently.
    """
    messages = _drop_empty_assistant_sentinels(
        [m.model_dump(exclude_none = True) for m in payload.messages]
    )

    if not payload.image_base64:
        return messages

    try:
        import base64 as _b64
        from io import BytesIO as _BytesIO
        from PIL import Image as _Image

        raw = _b64.b64decode(payload.image_base64)
        img = _Image.open(_BytesIO(raw)).convert("RGB")
        buf = _BytesIO()
        img.save(buf, format = "PNG")
        png_b64 = _b64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        raise HTTPException(
            status_code = 400,
            detail = "Failed to process image.",
        )

    data_url = f"data:image/png;base64,{png_b64}"
    image_part = {"type": "image_url", "image_url": {"url": data_url}}

    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        existing = msg.get("content")
        if isinstance(existing, str):
            msg["content"] = [{"type": "text", "text": existing}, image_part]
        elif isinstance(existing, list):
            existing.append(image_part)
        else:
            msg["content"] = [image_part]
        break
    else:
        messages.append({"role": "user", "content": [image_part]})

    return messages


def _extract_response_format(payload):
    """Return the ``response_format`` field on an incoming ChatCompletionRequest
    (or None). The model is declared with ``extra="allow"`` so pydantic stashes
    unknown top-level fields in ``model_extra``; OpenAI-SDK clients spread
    ``extra_body`` into the request body top level, which is where guided-
    decoding recipes park their JSON-schema response_format.
    """
    extra = getattr(payload, "model_extra", None)
    if not isinstance(extra, dict):
        return None
    rf = extra.get("response_format")
    return rf if isinstance(rf, dict) else None


def _build_openai_passthrough_body(payload, backend_ctx = None) -> dict:
    """Assemble the llama-server request body from a ChatCompletionRequest.

    Only explicitly-known OpenAI / llama-server fields are forwarded so that
    Studio-specific extensions (``enable_tools``, ``enabled_tools``,
    ``session_id``, ...) never leak to the backend.
    """
    messages = _openai_messages_for_passthrough(payload)
    tool_choice = payload.tool_choice if payload.tool_choice is not None else "auto"
    # When the caller asked for a specific reasoning mode, forward it to
    # llama-server via chat_template_kwargs so the Jinja template renders
    # with (or without) the reasoning preamble.
    tpl_kwargs = None
    if payload.enable_thinking is not None:
        tpl_kwargs = {"enable_thinking": bool(payload.enable_thinking)}
    return _build_passthrough_payload(
        messages,
        payload.tools,
        payload.temperature,
        payload.top_p,
        payload.top_k,
        payload.max_tokens,
        payload.stream,
        stop = payload.stop,
        min_p = payload.min_p,
        repetition_penalty = payload.repetition_penalty,
        presence_penalty = payload.presence_penalty,
        tool_choice = tool_choice,
        response_format = _extract_response_format(payload),
        chat_template_kwargs = tpl_kwargs,
        backend_ctx = backend_ctx,
    )


async def _openai_passthrough_stream(
    request,
    cancel_event,
    llama_backend,
    payload,
    model_name,
    completion_id,
):
    """Streaming client-side pass-through for /v1/chat/completions.

    Forwards the client's OpenAI function-calling request to llama-server and
    relays the SSE stream back verbatim. This preserves llama-server's
    native response ``id``, ``finish_reason`` (including ``"tool_calls"``),
    ``delta.tool_calls``, and the trailing ``usage`` chunk so the client
    observes a standard OpenAI response.
    """
    target_url = f"{llama_backend.base_url}/v1/chat/completions"
    body = _build_openai_passthrough_body(
        payload, backend_ctx = llama_backend.context_length
    )

    _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
    _tracker = _TrackedCancel(cancel_event, *_cancel_keys)
    _tracker.__enter__()

    # Outer guard: asyncio.CancelledError at `await client.send(...)` is
    # a BaseException that bypasses `except httpx.RequestError`; without
    # this the tracker leaks. The generator's finally only runs once
    # iteration starts.
    try:
        # Dispatch BEFORE returning StreamingResponse so transport errors
        # and non-200 upstream statuses surface as real HTTP errors --
        # OpenAI SDKs rely on status codes to raise APIError/BadRequestError.
        client = httpx.AsyncClient(
            timeout = 600,
            limits = httpx.Limits(max_keepalive_connections = 0),
        )
        resp = None
        try:
            req = client.build_request("POST", target_url, json = body)
            resp = await client.send(req, stream = True)
        except httpx.RequestError as e:
            # llama-server subprocess crashed / still starting / unreachable.
            logger.error("openai passthrough stream: upstream unreachable: %s", e)
            if resp is not None:
                try:
                    await resp.aclose()
                except Exception:
                    pass
            try:
                await client.aclose()
            except Exception:
                pass
            raise HTTPException(
                status_code = 502,
                detail = _friendly_error(e),
            )

        if resp.status_code != 200:
            err_bytes = await resp.aread()
            err_text = err_bytes.decode("utf-8", errors = "replace")
            logger.error(
                "openai passthrough upstream error: status=%s body=%s",
                resp.status_code,
                err_text[:500],
            )
            upstream_status = resp.status_code
            try:
                await resp.aclose()
            except Exception:
                pass
            try:
                await client.aclose()
            except Exception:
                pass
            raise HTTPException(
                status_code = upstream_status,
                detail = f"llama-server error: {err_text[:500]}",
            )

        async def _stream():
            # Same httpx lifecycle pattern as _anthropic_passthrough_stream:
            # save resp.aiter_lines() so the finally block can aclose() it
            # on our task. See that function for full rationale.
            lines_iter = None
            # During llama-server prefill, `aiter_lines()` blocks until the
            # first SSE chunk arrives. The in-loop `cancel_event` check
            # cannot fire until then, which is the exact proxy/Colab
            # scenario the cancel POST is meant to recover from. Run a
            # tiny watcher that closes `resp` as soon as cancel fires,
            # unblocking the iterator with a RemoteProtocolError caught
            # in the except clause below.
            cancel_watcher = asyncio.create_task(
                _await_cancel_then_close(cancel_event, resp)
            )
            try:
                lines_iter = resp.aiter_lines()
                async for raw_line in lines_iter:
                    if cancel_event.is_set():
                        break
                    if await request.is_disconnected():
                        cancel_event.set()
                        break
                    if not raw_line:
                        continue
                    if not raw_line.startswith("data: "):
                        continue
                    # Relay verbatim to preserve llama-server's native id,
                    # finish_reason, delta.tool_calls, and usage chunks.
                    yield raw_line + "\n\n"
                    if raw_line[6:].strip() == "[DONE]":
                        break
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.CloseError):
                # Watcher closed resp on cancel. Emit nothing extra; the
                # client either initiated the cancel or already disconnected.
                if not cancel_event.is_set():
                    raise
            except Exception as e:
                # 200 headers are already flushed; errors must be in the SSE body.
                logger.error("openai passthrough stream error: %s", e)
                err = {
                    "error": {
                        "message": _friendly_error(e),
                        "type": "server_error",
                    },
                }
                yield f"data: {json.dumps(err)}\n\n"
            finally:
                cancel_watcher.cancel()
                try:
                    await cancel_watcher
                except (asyncio.CancelledError, Exception):
                    pass
                if lines_iter is not None:
                    try:
                        await lines_iter.aclose()
                    except Exception:
                        pass
                try:
                    await resp.aclose()
                except Exception:
                    pass
                try:
                    await client.aclose()
                except Exception:
                    pass
                _tracker.__exit__(None, None, None)

        return StreamingResponse(
            _stream(),
            media_type = "text/event-stream",
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    except BaseException:
        _tracker.__exit__(None, None, None)
        raise


async def _openai_passthrough_non_streaming(
    llama_backend,
    payload,
    model_name,
):
    """Non-streaming client-side pass-through for /v1/chat/completions.

    Returns llama-server's JSON response verbatim (via JSONResponse) so the
    client sees the native response ``id``, ``finish_reason`` (including
    ``"tool_calls"``), structured ``tool_calls``, and accurate ``usage``
    token counts.
    """
    target_url = f"{llama_backend.base_url}/v1/chat/completions"
    body = _build_openai_passthrough_body(
        payload, backend_ctx = llama_backend.context_length
    )

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(target_url, json = body, timeout = 600)
    except httpx.RequestError as e:
        # llama-server subprocess crashed / still starting / unreachable.
        # Surface the same friendly message the sync chat path emits so
        # operators don't see a bare 500 with no diagnostic.
        logger.error("openai passthrough non-streaming: upstream unreachable: %s", e)
        raise HTTPException(
            status_code = 502,
            detail = _friendly_error(e),
        )

    if resp.status_code != 200:
        raise HTTPException(
            status_code = resp.status_code,
            detail = f"llama-server error: {resp.text[:500]}",
        )

    # Guided-decoding fence wrap. llama-server returns raw JSON that matches
    # the schema (no surrounding markdown) because the GBNF grammar only
    # emits the JSON object itself. data_designer's llm-structured parser
    # looks for a ```json ... ``` markdown fence and discards unfenced
    # output, which collapses a 100%-valid guided-decoding run to 0/N.
    # Wrap each choice's content in the expected fence when the caller
    # asked for guided decoding, leaving already-fenced content alone.
    if _extract_response_format(payload) is not None:
        try:
            data = resp.json()
            changed = False
            for choice in data.get("choices", []):
                if not isinstance(choice, dict):
                    continue
                msg = choice.get("message")
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, str):
                    continue
                stripped = content.strip()
                if not stripped or stripped.startswith("```"):
                    continue
                msg["content"] = f"```json\n{stripped}\n```"
                changed = True
            if changed:
                return JSONResponse(content = data)
        except Exception as exc:
            # Wrap is best-effort; fall through to the verbatim body if
            # the response is not JSON-shaped or the structure is unusual.
            logger.warning(
                "response_format fence wrap skipped: %s",
                exc,
            )

    # Pass the upstream body through as raw bytes — skips a redundant
    # parse+re-serialize round-trip and keeps the response truly
    # verbatim (matches the docstring). Status is guaranteed 200 by
    # the check above.
    return Response(content = resp.content, media_type = "application/json")
