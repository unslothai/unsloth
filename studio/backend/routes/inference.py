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

from utils.api_errors import openai_error_body, anthropic_error_body


def _positive_int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None
    return value_int if value_int > 0 else None


def _install_httpcore_asyncgen_silencer() -> None:
    """Silence benign httpx/httpcore asyncgen GC noise on Python 3.13.

    When Studio proxies a llama-server stream via httpx, the innermost
    ``HTTP11ConnectionByteStream.__aiter__`` async generator is finalised by
    the asyncgen GC hook on a task different from the one that opened it. Its
    ``aclose`` calls ``anyio.Lock.acquire`` → ``cancel_shielded_checkpoint``,
    entering a ``CancelScope`` on the finaliser task; Python 3.13 flags the
    cross-task exit as ``"Attempted to exit cancel scope in a different task"``
    and prints ``"async generator ignored GeneratorExit"`` as an unraisable
    warning.

    Known httpx + httpcore + anyio interaction (MCP SDK python-sdk#831, agno
    #3556, chainlit #2361, langchain-mcp-adapters #254). Benign: the 200
    response is already delivered. The streaming pass-throughs
    (``/v1/chat/completions``, ``/v1/messages``, ``/v1/responses``,
    ``/v1/completions``) manage their httpx lifecycle in one task with explicit
    ``aclose()``; we don't hold a reference to the errant generator and can't
    close it ourselves.

    Install one process-wide unraisable hook that swallows only this
    interaction -- identified by (RuntimeError mentioning cancel scope /
    GeneratorExit) + (object repr referencing HTTP11ConnectionByteStream) --
    and defers to the default hook otherwise. Idempotent.
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
            and (
                "cancel scope" in str(exc_value)
                or "GeneratorExit" in str(exc_value)
                or "no running event loop" in str(exc_value)
            )
        ):
            return
        prior_hook(unraisable)

    _hook._unsloth_httpcore_silencer = True  # type: ignore[attr-defined]
    sys.unraisablehook = _hook


_install_httpcore_asyncgen_silencer()


def _loaded_chat_template() -> Optional[str]:
    """Chat template of the currently loaded GGUF model, if any."""
    try:
        return get_llama_cpp_backend().chat_template
    except Exception:
        return None


def _template_raise_message(error_text: str, chat_template: Optional[str]) -> Optional[str]:
    """A chat-template raise_exception message to surface, but only when it appears
    verbatim in chat_template (simple substring check), so we never leak arbitrary
    llama-server text. Anchors on llama.cpp's "Jinja Exception:" prefix."""
    if not chat_template:
        return None
    marker = "Jinja Exception:"
    idx = error_text.find(marker)
    if idx == -1:
        return None
    candidate = error_text[idx + len(marker) :]
    # llama-server appends JSON after the message; cut at the first boundary.
    for stop in ('"', "\n"):
        cut = candidate.find(stop)
        if cut != -1:
            candidate = candidate[:cut]
    candidate = candidate.strip()
    return candidate if candidate and candidate in chat_template else None


def _friendly_error(exc: Exception) -> str:
    """Extract a user-friendly message from known llama-server errors."""
    # httpx transport failures from the async pass-through helpers. Any
    # RequestError subclass (ConnectError, ReadError, RemoteProtocolError,
    # WriteError, PoolTimeout, ...) means the llama-server subprocess is
    # unreachable -- crashed or still coming up.
    if isinstance(exc, httpx.RequestError):
        return (
            "Lost connection to the model server. It may have crashed -- try reloading the model."
        )
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
        return (
            "Lost connection to the model server. It may have crashed -- try reloading the model."
        )
    template_msg = _template_raise_message(msg, _loaded_chat_template())
    if template_msg:
        return f"An internal error occurred: {template_msg}"
    return "An internal error occurred"


def _clamp_finish_reason(value) -> str:
    """Coerce an upstream finish_reason into OpenAI's known chat values.

    Unknown values (including ``None``) become ``"stop"`` so local upstream
    quirks do not leak into the public API shape.
    """
    return (
        value
        if value
        in (
            "stop",
            "length",
            "tool_calls",
            "content_filter",
            "function_call",
        )
        else "stop"
    )


def _normalize_stop_sequences(raw):
    """Coerce an OpenAI/Anthropic ``stop`` value into the list-of-non-empty-strings
    shape llama-server expects, or ``None`` when absent. A bare string becomes a
    single-element list; empty strings are dropped (an empty stop sequence would
    terminate generation immediately at position 0)."""
    if isinstance(raw, str):
        return [raw] if raw else None
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, str) and s] or None
    return None


def _effective_max_tokens(payload):
    """Resolve the generation cap, preferring OpenAI's replacement field.

    ``max_tokens`` is deprecated in favor of ``max_completion_tokens``; honor
    either for compatibility, but let the replacement field win when both are
    supplied.
    """
    return (
        payload.max_completion_tokens
        if payload.max_completion_tokens is not None
        else payload.max_tokens
    )


def _wants_multiple_choices(payload) -> bool:
    return (payload.n or 1) > 1


def _raise_unsupported_openai_parameter(param: str, message: str) -> None:
    raise HTTPException(
        status_code = 400,
        detail = openai_error_body(
            message,
            status = 400,
            code = "unsupported_parameter",
            param = param,
        ),
    )


def _raise_unsupported_n(path_label: str) -> None:
    _raise_unsupported_openai_parameter("n", f"n > 1 is not supported for {path_label}.")


def _openai_stream_error_chunk(exc) -> dict:
    """Build an in-band OpenAI error chunk for a mid-stream failure. Once the
    stream's 200 headers are flushed the status can't change, so the error must
    ride in the SSE body. An upstream context-window overflow is mapped to
    code=context_length_exceeded so client compaction/trim loops can detect it
    (a code-less error hides it)."""
    _cls = _classify_llama_generation_error(exc)
    if _cls:
        return openai_error_body(_friendly_error(exc), status = 400, code = "context_length_exceeded")
    if _cls is False:
        return openai_error_body(_friendly_error(exc), status = 400)
    return openai_error_body(_friendly_error(exc), status = 500)


def _openai_passthrough_error(status_code, text) -> "HTTPException":
    """HTTPException for a non-200 upstream response on the OpenAI passthrough
    (tools / response_format). An over-context upstream error is mapped to a 400
    with code="context_length_exceeded" so these paths deliver the same signal as
    the non-passthrough path; any other upstream error keeps llama-server's
    message verbatim."""
    if _classify_llama_generation_error(Exception(text)):
        return HTTPException(
            status_code = 400,
            detail = openai_error_body(
                _friendly_error(Exception(text)),
                status = 400,
                code = "context_length_exceeded",
                param = "messages",
            ),
        )
    return HTTPException(
        status_code = status_code,
        detail = f"llama-server error: {text[:500]}",
    )


_OVERFLOW_TRUNCATE_MAX_RETRIES = 3
# Truncated-prompt share of the real window; the rest is generation headroom
# so a near-full prompt cannot cut a tool call mid-JSON at the wall.
_OVERFLOW_PROMPT_TARGET_FRACTION = 0.75


def _overflow_truncation_requested(payload) -> bool:
    """True when the request (or the UNSLOTH_CONTEXT_OVERFLOW server default,
    for clients that cannot send custom fields) opted into truncation."""
    requested = getattr(payload, "context_overflow", None)
    if requested is not None:
        return requested == "truncate_middle"
    return os.environ.get("UNSLOTH_CONTEXT_OVERFLOW", "").strip().lower() == "truncate_middle"


def _parse_overflow_counts(err_text: str):
    """(n_prompt_tokens, n_ctx) from an exceed_context_size_error body, or
    None. Tolerates \\" around keys (body may be a re-wrapped JSON string)."""
    m_prompt = _re.search(r'n_prompt_tokens\\?"?\s*:\s*(\d+)', err_text)
    m_ctx = _re.search(r'n_ctx\\?"?\s*:\s*(\d+)', err_text)
    if m_prompt and m_ctx:
        return int(m_prompt.group(1)), int(m_ctx.group(1))
    return None


def _estimate_message_tokens(msg: dict) -> int:
    try:
        return max(1, len(json.dumps(msg, ensure_ascii = False)) // 4)
    except Exception:
        return 1


def _truncate_middle_messages(messages: list, keep_ratio: float):
    """Drop whole turn-groups from the middle of an OpenAI message list.

    Always kept: leading system message(s), the first group (task anchor),
    and the trailing groups. A group is a user message, or an assistant
    message plus its following tool results, so surviving tool_calls stay
    paired with their results as chat templates require.
    Returns (new_messages, dropped_message_count).
    """
    if not messages or keep_ratio >= 1.0:
        return messages, 0

    head: list = []
    idx = 0
    while idx < len(messages) and messages[idx].get("role") in ("system", "developer"):
        head.append(messages[idx])
        idx += 1

    groups: list[list] = []
    for msg in messages[idx:]:
        role = msg.get("role")
        if role == "tool" and groups:
            groups[-1].append(msg)
        elif role == "tool":
            groups.append([msg])  # orphan tool result; treat as its own group
        else:
            groups.append([msg])

    # Anchor group plus the last 3 groups stay.
    protected_tail = min(3, max(1, len(groups) - 1))
    if len(groups) <= 1 + protected_tail:
        return messages, 0

    total_est = sum(_estimate_message_tokens(m) for m in messages)
    target_est = int(total_est * keep_ratio)

    anchor = groups[0]
    middle = groups[1:-protected_tail]
    tail = groups[-protected_tail:]

    current_est = total_est
    kept_middle: list[list] = list(middle)
    dropped = 0
    # Drop oldest-first until the estimate fits the target.
    while kept_middle and current_est > target_est:
        victim = kept_middle.pop(0)
        dropped += len(victim)
        current_est -= sum(_estimate_message_tokens(m) for m in victim)

    if dropped == 0:
        return messages, 0

    new_messages = head + anchor
    for grp in kept_middle:
        new_messages.extend(grp)
    for grp in tail:
        new_messages.extend(grp)
    return new_messages, dropped


_CLIP_MARKER = "\n[... truncated by context_overflow=truncate_middle ...]\n"
# Generous head+tail first; cut harder if the estimate still misses the target.
_CLIP_KEEP_CHARS = (1500, 400)


def _clip_long_contents(messages: list, target_est: int) -> int:
    """Clip oversized string contents middle-out until ``target_est`` is met.

    Tool results first, then earlier user turns, the final message last.
    Message count and roles never change, so tool pairing holds even when
    group-dropping could not free enough. Returns messages clipped.
    """

    def _candidates():
        tools = [m for m in messages if m.get("role") == "tool"]
        users = [m for m in messages[:-1] if m.get("role") == "user"]
        last = [messages[-1]] if messages else []
        return tools + users + last

    clipped = 0
    for keep in _CLIP_KEEP_CHARS:
        for msg in _candidates():
            if sum(_estimate_message_tokens(m) for m in messages) <= target_est:
                return clipped
            content = msg.get("content")
            if not isinstance(content, str) or len(content) <= 2 * keep + len(_CLIP_MARKER):
                continue
            msg["content"] = content[:keep] + _CLIP_MARKER + content[-keep:]
            clipped += 1
    return clipped


def _apply_overflow_truncation(body: dict, err_text: str) -> bool:
    """Shrink a passthrough body after an upstream context overflow: drop
    middle turn-groups, clip still-oversized contents, clamp ``max_tokens``
    to the generation headroom. Returns False when nothing could shrink."""
    counts = _parse_overflow_counts(err_text)
    messages = body.get("messages") or []
    total_est = sum(_estimate_message_tokens(m) for m in messages)
    if counts:
        n_prompt, n_ctx = counts
        keep_ratio = min(0.95, (_OVERFLOW_PROMPT_TARGET_FRACTION * n_ctx) / max(1, n_prompt))
        # Scale the server-token target into char-estimate units.
        target_est = int(total_est * keep_ratio)
    else:
        n_ctx = None
        keep_ratio = 0.6  # no counts in the error; cut conservatively
        target_est = int(total_est * keep_ratio)

    new_messages, dropped = _truncate_middle_messages(messages, keep_ratio)
    if dropped:
        body["messages"] = new_messages
    clipped = 0
    if sum(_estimate_message_tokens(m) for m in body.get("messages") or []) > target_est:
        clipped = _clip_long_contents(body.get("messages") or [], target_est)
    if not dropped and not clipped:
        return False
    if n_ctx:
        headroom = max(1024, int(n_ctx * (1.0 - _OVERFLOW_PROMPT_TARGET_FRACTION)))
        cur_max = body.get("max_tokens")
        body["max_tokens"] = min(cur_max, headroom) if cur_max else headroom
    logger.warning(
        "context_overflow=truncate_middle: dropped %d middle messages, clipped "
        "%d contents (keep_ratio %.2f); retrying within the real window",
        dropped,
        clipped,
        keep_ratio,
    )
    return True


def _anthropic_stream_error_event(exc):
    """Anthropic in-band SSE ``error`` event for a mid-stream failure, or ``None``
    to fall through to a normal message_delta finish. Returns an event only for a
    classifiable upstream client error (context overflow / 4xx) so a streaming
    over-context request surfaces a real error instead of a silent empty
    end_turn message."""
    if _classify_llama_generation_error(exc) is None:
        return None
    return build_anthropic_sse_event(
        "error",
        anthropic_error_body(_friendly_error(exc), status = 400),
    )


def _drop_parallel_tool_call_deltas(chunk) -> bool:
    """In-place: drop tool_call deltas whose index >= 1 from a parsed OpenAI
    streaming chunk so only the first tool call survives (parallel_tool_calls=false
    / disable_parallel_tool_use, best-effort). Returns True if anything changed."""
    if not isinstance(chunk, dict):
        return False
    changed = False
    for ch in chunk.get("choices") or []:
        delta = ch.get("delta") or {}
        tcs = delta.get("tool_calls")
        if isinstance(tcs, list):
            kept = [tc for tc in tcs if isinstance(tc, dict) and (tc.get("index") or 0) == 0]
            if len(kept) != len(tcs):
                delta["tool_calls"] = kept
                changed = True
    return changed


def _cap_parallel_tool_calls_sse_line(raw_line: str) -> str:
    """Drop tool_call deltas whose index >= 1 from one streamed OpenAI SSE
    ``data:`` line so only the first tool call survives (parallel_tool_calls=false,
    best-effort). Non-tool / unparseable payloads are returned byte-for-byte."""
    payload = raw_line[len("data: ") :]
    if payload.strip() in ("", "[DONE]"):
        return raw_line
    try:
        obj = json.loads(payload)
    except Exception:
        return raw_line
    if not _drop_parallel_tool_call_deltas(obj):
        return raw_line
    return "data: " + json.dumps(obj, separators = (",", ":"))


def _prompt_tokens_details(upstream):
    """Surface llama-server's real ``cached_tokens`` (KV-cache prompt hits) while
    keeping the full OpenAI ``prompt_tokens_details`` shape. Defaults to zero when
    the upstream usage doesn't carry it, so the field is always present."""
    out = {"cached_tokens": 0, "audio_tokens": 0}
    if isinstance(upstream, dict):
        out.update({k: v for k, v in upstream.items() if v is not None})
    return out


def _wants_stream_usage(payload) -> bool:
    return bool((payload.stream_options or {}).get("include_usage"))


def _openai_stream_usage_chunk(
    payload, completion_id, created, model_name, stream_usage, stream_timings
):
    """Build the final OpenAI-standard usage chunk (choices=[], usage populated)
    for a chat stream. Returns the SSE ``data:`` line, or None when the client
    did not opt in via ``stream_options.include_usage`` (or no usage exists)."""
    if not _wants_stream_usage(payload):
        return None
    if not (stream_usage or stream_timings):
        return None
    _usage = stream_usage or {}
    _prompt_tokens = _usage.get("prompt_tokens") or 0
    _completion_tokens = _usage.get("completion_tokens") or 0
    _total_tokens = _usage.get("total_tokens") or (_prompt_tokens + _completion_tokens)
    usage_chunk = ChatCompletionChunk(
        id = completion_id,
        created = created,
        model = model_name,
        choices = [],
        usage = CompletionUsage(
            prompt_tokens = _prompt_tokens,
            completion_tokens = _completion_tokens,
            total_tokens = _total_tokens,
            prompt_tokens_details = _prompt_tokens_details(_usage.get("prompt_tokens_details")),
        ),
        timings = stream_timings,
    )
    return f"data: {usage_chunk.model_dump_json(exclude_none = True)}\n\n"


def _rewrite_cmpl_id(raw: bytes) -> bytes:
    """Rewrite llama-server's chat-style ``chatcmpl-`` ids to the ``cmpl-``
    prefix OpenAI's legacy /v1/completions use. Anchored on the ``"id":`` key
    (both spacing variants) so the rest of the body stays byte-exact."""
    return raw.replace(b'"id":"chatcmpl-', b'"id":"cmpl-').replace(
        b'"id": "chatcmpl-', b'"id": "cmpl-'
    )


def _cmpl_stream_event_out(event: bytes, include_usage: bool) -> Optional[bytes]:
    """Process one legacy /v1/completions SSE event (text between blank-line
    separators).

    Always rewrites the ``chatcmpl-`` -> ``cmpl-`` id prefix. When the client
    did NOT request ``stream_options.include_usage``, also removes the usage
    statistics so the stream matches OpenAI's contract.

    Shape note: on /v1/completions, llama-server attaches ``usage`` to the
    FINAL content chunk (the ``finish_reason`` chunk, which has a populated
    ``choices`` array) -- unlike the chat stream, which emits a standalone
    ``choices: []`` usage chunk. Both shapes are handled: a standalone
    usage-only chunk is dropped; an inline ``usage`` field is stripped from a
    content chunk while keeping ``choices``/``finish_reason`` intact.

    Returns the event bytes to emit, or ``None`` to drop the event. Only a
    usage-bearing event is re-serialized; every other event keeps exact bytes.
    """
    if include_usage:
        return _rewrite_cmpl_id(event)
    lines = event.split(b"\n")
    changed = False
    for i, ln in enumerate(lines):
        if not ln.startswith(b"data:"):
            continue
        payload = ln[len(b"data:") :].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except Exception:
            continue
        if not isinstance(obj, dict) or obj.get("usage") is None:
            continue
        # Standalone usage-only chunk (chat-style) -> drop the whole event.
        if obj.get("choices") == []:
            return None
        # Usage on a content/finish chunk (completions-style) -> strip it.
        obj.pop("usage", None)
        lines[i] = b"data: " + json.dumps(obj, separators = (",", ":")).encode("utf-8")
        changed = True
    return _rewrite_cmpl_id(b"\n".join(lines) if changed else event)


def _classify_llama_generation_error(exc: Exception) -> Optional[bool]:
    """Classify an error raised while consuming the GGUF generator.

    Returns True for a context-window overflow, False for any other upstream
    4xx (a client error), or None when it should stay a 500. Distinguishes a
    real client error from a genuine crash by the explicit "llama-server
    returned 4xx" marker, not a bare "tokens"/"exceed" substring.
    """
    msg = str(exc)
    msg_l = msg.lower()
    if "n_ctx" in msg_l or (
        "context" in msg_l and any(t in msg_l for t in ("exceed", "length", "window", "too long"))
    ):
        return True
    if _re.search(r"llama-server returned (4\d\d)", msg):
        return False
    return None


# Add backend directory to path
backend_path = Path(__file__).parent.parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))

try:
    from core.inference import get_inference_backend
    from core.inference.llama_cpp import (
        LlamaCppBackend,
        _DEFAULT_MAX_TOKENS_FLOOR,
        _DEFAULT_T_MAX_PREDICT_MS,
        _canonicalize_spec_mode,
        _extra_args_set_spec_type,
        _hf_offline_if_dns_dead,
        detect_reasoning_flags,
    )
    from core.inference.llama_server_args import (
        resolve_tensor_parallel,
        strip_shadowing_flags,
        validate_extra_args,
    )
    from core.inference.tensor_fallback import load_with_tensor_fallback
    from utils.models import ModelConfig
    from utils.inference import load_inference_config
    from utils.models.model_config import (
        detect_mtp_file,
        load_model_defaults,
    )
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
        _canonicalize_spec_mode,
        _extra_args_set_spec_type,
        _hf_offline_if_dns_dead,
        detect_reasoning_flags,
    )
    from core.inference.llama_server_args import (
        resolve_tensor_parallel,
        strip_shadowing_flags,
        validate_extra_args,
    )
    from core.inference.tensor_fallback import load_with_tensor_fallback
    from utils.models import ModelConfig
    from utils.inference import load_inference_config
    from utils.models.model_config import (
        detect_mtp_file,
        load_model_defaults,
    )
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
    ToolConfirmRequest,
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
    CreateOpenAIContainerBody,
    DeleteOpenAIContainerBody,
    ListOpenAIContainersResponse,
    OpenAIContainerRequest,
    OpenAIContainerSummary,
)
from core.inference.anthropic_compat import (
    anthropic_messages_to_openai,
    anthropic_tools_to_openai,
    anthropic_tool_choice_to_openai,
    openai_finish_to_anthropic_stop,
    anthropic_tool_use_id,
    build_anthropic_sse_event,
    AnthropicStreamEmitter,
    AnthropicPassthroughEmitter,
)
from auth.authentication import get_current_subject
from state.tool_approvals import resolve_tool_decision

from core.inference.key_exchange import decrypt_api_key
from core.inference.providers import get_provider_info, get_base_url
from core.inference.external_provider import ExternalProviderClient
from storage import providers_db
from utils.utils import safe_error_detail, log_and_http_error

import io
import wave
import base64
import numpy as np
from datetime import date as _date

router = APIRouter()
# Studio-only router (not mounted on /v1 OpenAI-compat).
studio_router = APIRouter()


_ARTIFACT_PREVIEW_FRAME_ANCESTORS = "'self' tauri://localhost http://tauri.localhost"
_ARTIFACT_PREVIEW_FRAME_STRICT_CSP = (
    "default-src 'none'; "
    "script-src 'unsafe-inline'; "
    "style-src 'unsafe-inline'; "
    "img-src data: blob:; "
    "font-src data:; "
    "media-src data: blob:; "
    "connect-src 'none'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    f"frame-ancestors {_ARTIFACT_PREVIEW_FRAME_ANCESTORS}; "
    "sandbox allow-scripts"
)
_ARTIFACT_PREVIEW_FRAME_NETWORK_CSP = (
    "default-src http: https: data: blob:; "
    "script-src 'unsafe-inline' 'unsafe-eval' http: https: data: blob:; "
    "script-src-elem 'unsafe-inline' http: https: data: blob:; "
    "style-src 'unsafe-inline' http: https: data: blob:; "
    "style-src-elem 'unsafe-inline' http: https: data: blob:; "
    "img-src http: https: data: blob:; "
    "font-src http: https: data: blob:; "
    "media-src http: https: data: blob:; "
    "connect-src http: https: ws: wss: data: blob:; "
    "worker-src http: https: blob:; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    f"frame-ancestors {_ARTIFACT_PREVIEW_FRAME_ANCESTORS}; "
    "sandbox allow-scripts"
)
_ARTIFACT_PREVIEW_FRAME_HTML = """<!doctype html>
<html>
  <head><meta charset=\"utf-8\" /></head>
  <body>
    <script>
      (() => {
        const createMemoryStorage = () => {
          const data = new Map();
          return {
            get length() { return data.size; },
            key: (index) => Array.from(data.keys())[index] ?? null,
            getItem: (key) => data.has(String(key)) ? data.get(String(key)) : null,
            setItem: (key, value) => data.set(String(key), String(value)),
            removeItem: (key) => data.delete(String(key)),
            clear: () => data.clear(),
          };
        };
        const installStorageFallback = (name) => {
          try {
            void window[name];
            return;
          } catch {
            // Opaque-origin sandboxed frames throw SecurityError for Web Storage.
          }
          try {
            Object.defineProperty(window, name, {
              value: createMemoryStorage(),
              configurable: true,
            });
          } catch {
            // Leave the sandbox failure contained in the artifact if the
            // browser refuses to shadow the Web Storage accessor.
          }
        };
        const installStorageFallbacks = () => {
          installStorageFallback("localStorage");
          installStorageFallback("sessionStorage");
        };
        const render = (html) => {
          installStorageFallbacks();
          document.open();
          document.write(html);
          document.close();
        };
        installStorageFallbacks();
        window.addEventListener("message", (event) => {
          const data = event.data;
          if (!data || data.type !== "unsloth:artifact-html" || typeof data.html !== "string") return;
          render(data.html);
        });
      })();
    </script>
  </body>
</html>"""


@studio_router.get("/artifact-preview-frame", include_in_schema = False)
async def artifact_preview_frame(
    request: Request,
    allow_network: bool = False,
    token: Optional[str] = None,
):
    """Serve the opaque sandbox shell used for client-side HTML artifacts."""

    if allow_network:
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

    csp = (
        _ARTIFACT_PREVIEW_FRAME_NETWORK_CSP if allow_network else _ARTIFACT_PREVIEW_FRAME_STRICT_CSP
    )
    return Response(
        content = _ARTIFACT_PREVIEW_FRAME_HTML,
        media_type = "text/html; charset=utf-8",
        headers = {
            "Cache-Control": "no-store",
            "Content-Security-Policy": csp,
            "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff",
        },
    )


def _detect_safetensors_features(backend, chat_template: Optional[str]) -> dict:
    """Classify reasoning/tool capabilities via the GGUF classifier so flags
    match across backends. gpt-oss is overridden: Harmony routes reasoning and
    tools through tokenizer channels, not template markup."""
    model_id = getattr(backend, "active_model_name", None)
    flags = (
        detect_reasoning_flags(
            chat_template,
            model_identifier = model_id,
            log_source = "safetensors",
        )
        if chat_template
        else {
            "supports_reasoning": False,
            "reasoning_style": "enable_thinking",
            "reasoning_always_on": False,
            "supports_preserve_thinking": False,
            "supports_tools": False,
        }
    )
    # Our safetensors loop only parses <tool_call>{json}</tool_call> and
    # <function=name>...</function>. Llama uses <|python_tag|>, Mistral uses
    # [TOOL_CALLS]; advertising tools for those enables a pill the parser
    # can't honour. GGUF is unaffected -- llama-server normalises every
    # format into structured deltas.
    if (
        flags.get("supports_tools")
        and chat_template
        and "<tool_call>" not in chat_template
        and "<function=" not in chat_template
    ):
        logger.info(
            "safetensors: template advertises tools but uses an "
            "emission format the loop cannot parse; suppressing "
            "supports_tools"
        )
        flags["supports_tools"] = False

    # gpt-oss: keep reasoning on, drop tools (Harmony channel, not the
    # <tool_call> XML this loop parses).
    try:
        if hasattr(backend, "_is_gpt_oss_model") and backend._is_gpt_oss_model():
            flags["supports_reasoning"] = True
            flags["reasoning_style"] = "reasoning_effort"
            flags["supports_tools"] = False
    except Exception:
        logger.debug("gpt_oss_check_failed", exc_info = True)
    return flags


def _effective_enable_tools(payload) -> Optional[bool]:
    """Resolve `payload.enable_tools` against the process-level tool policy.

    Returns the policy value when set (CLI hard-override from `unsloth run`),
    else the per-request value.
    """
    from state.tool_policy import get_tool_policy

    policy = get_tool_policy()
    return policy if policy is not None else payload.enable_tools


# Cancel registry. Proxies (e.g. Colab) can swallow client fetch aborts so
# is_disconnected() never fires. POST /inference/cancel looks up in-flight
# cancel_events here by cancel_id (per-run) or session_id / completion_id
# (fallbacks).
_CANCEL_REGISTRY: dict[str, set[threading.Event]] = {}
_CANCEL_LOCK = threading.Lock()

# Cancel POSTs arriving before registration are stashed; the next matching
# __enter__ replays set() within the TTL.
_PENDING_CANCELS: dict[str, float] = {}
_PENDING_CANCEL_TTL_S = 30.0


def _prune_pending(now: float) -> None:
    for k in [k for k, ts in _PENDING_CANCELS.items() if now - ts > _PENDING_CANCEL_TTL_S]:
        _PENDING_CANCELS.pop(k, None)


class _TrackedCancel:
    """Register cancel_event in _CANCEL_REGISTRY for the block's duration."""

    def __init__(self, event: threading.Event, *keys):
        self.event = event
        self.keys = tuple(k for k in keys if k)

    def __enter__(self):
        # Register + consume-pending in one critical section to close the
        # TOCTOU race against a concurrent cancel POST.
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
    session_id/completion_id are shared across runs on the same thread, so
    stashing them would ghost-cancel the user's next request. Only cancel_id
    is per-run unique (see _cancel_by_cancel_id_or_stash)."""
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

    Used by passthrough streamers so a /cancel POST can interrupt while the
    async iterator is blocked on llama-server prefill. Without it the in-loop
    ``cancel_event.is_set()`` check is unreachable until the first SSE chunk
    arrives -- exactly the proxy/Colab case the cancel POST exists for.

    Polls a threading.Event since the cancel registry is keyed by
    threading.Event (so the sync /cancel handler can call .set()). The 50ms
    cadence adds at most that latency to a prefill cancel; the common
    streaming-cancel path still sees the event on the iterator's next chunk.
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


# Centralized local/server tool nudge. Keep render_html guidance gated to turns
# where the artifact tool is actually present in the tool schema; otherwise
# small local models can hallucinate a missing tool call instead of following
# the fenced-HTML fallback prompt.
_TOOL_BASE_NUDGE = (
    "Tools are available when they materially improve the answer. Use an enabled "
    "tool for current facts, calculations, code execution, or artifacts when it "
    "materially helps; otherwise answer normally and follow the user's requested "
    "format."
)
_TOOL_WEB_COMPACT_TIP = "When using web_search, do not repeat the same search query."
_TOOL_WEB_EXPANDED_TIP = (
    "When using web_search and a result URL is relevant, fetch its full content "
    "by calling web_search with the url parameter. Do not repeat the same search "
    "query. If a search returns no useful results, try rephrasing or fetching a "
    "result URL directly."
)
_TOOL_CODE_TIP = (
    "Use code execution for math, calculations, data processing, or to parse "
    "and analyze information from tool results."
)
_TOOL_ARTIFACT_TIP = (
    "For HTML, CSS, or JavaScript artifact requests, call render_html once when "
    "it is available with one complete self-contained HTML document in the code "
    "argument. After render_html succeeds, do not call it again in the same "
    "response unless the user asks for changes. Future user requests for new "
    "artifacts may call render_html once."
)


def _build_tool_action_nudge(*, tools: list[dict], model_name: str) -> str:
    tool_names = {
        (tool.get("function") or {}).get("name")
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    has_web = "web_search" in tool_names
    has_code = "python" in tool_names or "terminal" in tool_names
    has_artifact = "render_html" in tool_names
    if not (has_web or has_code or has_artifact):
        return ""

    model_size_b = _extract_model_size_b(model_name)
    compact_web_tip = model_size_b is not None and model_size_b < 9
    tool_tip_parts: list[str] = []
    if has_web:
        tool_tip_parts.append(_TOOL_WEB_COMPACT_TIP if compact_web_tip else _TOOL_WEB_EXPANDED_TIP)
    if has_code:
        tool_tip_parts.append(_TOOL_CODE_TIP)
    if has_artifact:
        tool_tip_parts.append(_TOOL_ARTIFACT_TIP)
    return (
        f"The current date is {_date.today().isoformat()}. "
        + _TOOL_BASE_NUDGE
        + " "
        + " ".join(tool_tip_parts)
    )


# Strip tool-call XML the speculative buffer in core/inference/llama_cpp.py
# split across the visible/DRAIN boundary. Four leak shapes:
#   1. well-formed `<tool_call>...</tool_call>` / `<function=...>...</function>`
#   2. orphan opening to EOF (close was DRAINED)
#   3. bare orphan close (open was DRAINED)
#   4. tail-only `</parameter>` (outer close truncated by EOS); anchored to
#      `\Z` so mid-text `<parameter>` in user code samples survives.
_TOOL_XML_RE = _re.compile(
    # Hyphen in the name char-class matches MCP tool names with dashes
    # (mcp__srv__list-issues) that would otherwise leak past this strip.
    r"<(?:tool_call|function=[\w-]+)>.*?(?:</(?:tool_call|function)>|\Z)"
    r"|</(?:tool_call|function)>"
    r"|</parameter>\s*\Z",
    _re.DOTALL,
)


def _strip_tool_xml_for_display(text: str, *, auto_heal_tool_calls: bool) -> str:
    """Apply route-level XML leak cleanup only when Auto-Heal is enabled."""
    if not auto_heal_tool_calls:
        return text
    return _TOOL_XML_RE.sub("", text)


logger = get_logger(__name__)


def _validate_native_gguf_companion(
    companion_path: str | None, gguf_path: str | None, label: str
) -> None:
    """Reject a companion GGUF (mmproj / MTP drafter) that a native-lease load
    would otherwise hand to llama-server: must be a regular file (no symlink
    escaping the leased directory) living next to the selected GGUF."""
    if not companion_path or not gguf_path:
        return
    import stat as _stat_module

    companion = Path(companion_path)
    gguf = Path(gguf_path)
    try:
        companion_lstat = os.lstat(companion)
    except OSError as exc:
        raise HTTPException(
            status_code = 400,
            detail = f"Native {label} is no longer accessible.",
        ) from exc
    if _stat_module.S_ISLNK(companion_lstat.st_mode) or not _stat_module.S_ISREG(
        companion_lstat.st_mode
    ):
        raise HTTPException(
            status_code = 400,
            detail = f"Native {label} must be a regular file.",
        )
    try:
        if companion.resolve(strict = True).parent != gguf.resolve(strict = True).parent:
            raise HTTPException(
                status_code = 400,
                detail = f"Native {label} must live next to the selected GGUF.",
            )
    except OSError as exc:
        raise HTTPException(
            status_code = 400,
            detail = f"Native {label} is no longer accessible.",
        ) from exc


def _normalise_settings_str(value: Optional[str]) -> Optional[str]:
    """Lowercase + strip a settings string, mapping blank/None to None."""
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip().lower()
        return stripped or None
    return value


def _should_strip_split_mode(request: LoadRequest, backend_extra: Optional[list[str]]) -> bool:
    """Whether an inherited --split-mode should be stripped on reload.

    The binary Tensor Parallelism toggle can't carry --split-mode's row/none/
    layer modes, so only strip when the toggle overrides it: tensor being turned
    on, or the inherited mode is tensor (toggle turning it off). Non-tensor modes
    survive. Shared by the inheritance strip and the already-loaded stale check
    so they agree on what reload would do.
    """
    fields_set = getattr(request, "model_fields_set", set())
    return "tensor_parallel" in fields_set and (
        request.tensor_parallel or resolve_tensor_parallel(backend_extra, False)
    )


def _request_matches_loaded_settings(request: LoadRequest, llama_backend: LlamaCppBackend) -> bool:
    """True iff every runtime setting on the request matches the loaded server.
    Caller has already checked model+variant+is_loaded. See #5401."""
    # Compare requested n_ctx (not effective) so VRAM-cap doesn't mask an
    # Auto-vs-explicit slider flip.
    if request.max_seq_length != llama_backend.requested_n_ctx:
        return False
    if _normalise_settings_str(request.cache_type_kv) != _normalise_settings_str(
        llama_backend.cache_type_kv
    ):
        return False
    # Reconcile a user --split-mode in extras into the effective tensor state.
    # When the request omits llama_extra_args ("inherit"), compare using the
    # stored extras stripped the way the reload strips them, so an extras-driven
    # tensor load isn't seen as a mismatch that needlessly reloads the server.
    backend_extra = list(llama_backend.extra_args) if llama_backend.extra_args else []
    effective_extra = (
        request.llama_extra_args
        if request.llama_extra_args is not None
        else strip_shadowing_flags(
            backend_extra,
            strip_split_mode = _should_strip_split_mode(request, backend_extra),
        )
    )
    if resolve_tensor_parallel(effective_extra, request.tensor_parallel) != llama_backend.tensor_parallel:
        return False
    # Spec decoding works on vision models too (MTP is mmproj-compatible,
    # llama.cpp #22673; the old ``not is_vision`` gate is gone), so compare
    # the real requested mode -- coercing vision to ``off`` here used to
    # swallow every spec-mode change on a vision model as already_loaded.
    req_mode = _canonicalize_spec_mode(request.speculative_type) or "auto"
    backend_mode = llama_backend.requested_spec_mode or "auto"
    if req_mode != backend_mode:
        return False
    # spec_draft_n_max only matters with an MTP variant; None means "platform
    # default" and matches whatever the backend chose.
    if backend_mode in ("mtp", "mtp+ngram") and request.spec_draft_n_max is not None:
        if int(request.spec_draft_n_max) != (llama_backend.spec_draft_n_max or 0):
            return False
    if (request.chat_template_override or None) != (llama_backend.chat_template_override or None):
        return False
    # llama_extra_args=None means "inherit"; only an explicit differing list
    # forces a reload. On the inherit path, refuse to match if stored extras
    # contain any shadow flag, so the reload path strips them rather than
    # leaving a stale override in effect. (backend_extra computed above.)
    if request.llama_extra_args is None:
        # Mirror the reload's conditional split-mode strip, so a preserved
        # non-tensor mode (row/none/layer) isn't seen as stale and doesn't
        # trigger a needless reload of a healthy server.
        if (
            backend_extra
            and strip_shadowing_flags(
                backend_extra,
                strip_split_mode = _should_strip_split_mode(request, backend_extra),
            )
            != backend_extra
        ):
            return False
    else:
        if list(request.llama_extra_args) != backend_extra:
            return False
    # A separate drafter (Gemma's root mtp-*.gguf) appearing or disappearing
    # next to the loaded weights changes the launch command (--model-draft),
    # so a duplicate /load must reload rather than dedupe. Always compare the
    # detected vs stored drafter when the mode can use one and the user does
    # not own --spec-type: the resolved-path compare is cheap and handles all
    # four cases (both None -> match; one None -> reload; equal -> match;
    # different -> reload), including a drafter deleted out from under a
    # running server. Runs last: it stats the filesystem, so every pure-memory
    # comparison above short-circuits first. Resolve both sides since the
    # stored launch path may be a snapshot symlink while detect_mtp_file
    # returns the resolved blob.
    if req_mode in ("auto", "mtp", "mtp+ngram") and llama_backend.gguf_path:
        effective_extras = (
            request.llama_extra_args
            if request.llama_extra_args is not None
            else llama_backend.extra_args
        )
        if not _extra_args_set_spec_type(effective_extras):
            detected = detect_mtp_file(llama_backend.gguf_path)
            stored = llama_backend.mtp_draft_path
            try:
                detected_resolved = Path(detected).resolve() if detected else None
                stored_resolved = Path(stored).resolve() if stored else None
            except OSError:
                return False
            if detected_resolved != stored_resolved:
                return False
    return True


def _resolve_model_identifier_for_request(
    request: LoadRequest | ValidateModelRequest, *, operation: str
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
        # Curated, client-correctable lease error (expired / wrong type / re-select);
        # keep the actionable message, just redact paths.
        logger.warning("inference.native_path_lease_failed: %s", exc)
        raise HTTPException(
            status_code = 400,
            detail = redact_native_paths(str(exc)),
        ) from exc
    display_label = grant.display_label or Path(request.model_path).name or "Native model"
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

    model_path is a clean identifier from GET /models/list. Returns inference
    config (temperature, top_p, top_k, min_p) from the model's YAML, falling
    back to default.yaml for missing values.

    GGUF models load via llama-server (llama.cpp) instead of Unsloth.
    """
    native_grant_backed = False
    model_log_label = request.model_path
    try:
        # Validate user pass-through args up front so a managed-flag collision
        # returns 400 before any model work.
        try:
            extra_llama_args = validate_extra_args(request.llama_extra_args)
        except ValueError as exc:
            # Keep the curated validation message (names the flag); just strip paths.
            logger.warning("inference.validate_extra_args_failed: %s", exc)
            raise HTTPException(
                status_code = 400,
                detail = redact_native_paths(str(exc)),
            )
        # Re-narrow []-from-None back to None so the inheritance path below can
        # tell "caller omitted" from "caller explicit []".
        extra_llama_args: Optional[list[str]] = (
            None if request.llama_extra_args is None else extra_llama_args
        )

        model_identifier, model_log_label, native_grant_backed = (
            _resolve_model_identifier_for_request(request, operation = "load-model")
        )
        # Version switching is handled by the subprocess-based inference
        # backend -- no ensure_transformers_version() needed here.

        # ── Already-loaded check: skip reload if the exact model is active ──
        backend = get_inference_backend()
        llama_backend = get_llama_cpp_backend()

        is_direct_gguf_request = model_identifier.lower().endswith(".gguf")
        if request.gguf_variant or is_direct_gguf_request:
            gguf_variant_matches = is_direct_gguf_request or bool(
                llama_backend.hf_variant
                and request.gguf_variant
                and llama_backend.hf_variant.lower() == request.gguf_variant.lower()
            )
            if (
                llama_backend.is_loaded
                and gguf_variant_matches
                and llama_backend.model_identifier
                and llama_backend.model_identifier.lower() == model_identifier.lower()
                # Match runtime settings so Apply isn't dropped (#5401).
                and _request_matches_loaded_settings(request, llama_backend)
                # Skip if a prior audio probe failed -- let load_model retry.
                and getattr(llama_backend, "_audio_probed", True)
            ):
                logger.info(
                    "Model already loaded (GGUF): "
                    f"{model_log_label} variant={request.gguf_variant or llama_backend.hf_variant}, skipping reload"
                )
                inference_config = load_inference_config(llama_backend.model_identifier)

                _gguf_audio = (
                    llama_backend._audio_type if hasattr(llama_backend, "_audio_type") else None
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
                    has_audio_input = getattr(llama_backend, "_has_audio_input", False),
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
                    chat_template = llama_backend.chat_template,
                    speculative_type = llama_backend.requested_spec_mode,
                    spec_draft_n_max = llama_backend.spec_draft_n_max,
                    tensor_parallel = llama_backend.tensor_parallel,
                )
        else:
            if (
                backend.active_model_name
                and backend.active_model_name.lower() == model_identifier.lower()
            ):
                logger.info(f"Model already loaded (Unsloth): {model_log_label}, skipping reload")
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
                # Classify via the same path as GGUF.
                _sf_flags = _detect_safetensors_features(backend, _chat_template)
                _sf_supports_reasoning = _sf_flags["supports_reasoning"]
                _sf_reasoning_style = _sf_flags["reasoning_style"]
                return LoadResponse(
                    status = "already_loaded",
                    model = model_log_label if native_grant_backed else backend.active_model_name,
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
                    reasoning_always_on = _sf_flags["reasoning_always_on"],
                    supports_preserve_thinking = _sf_flags["supports_preserve_thinking"],
                    supports_tools = _sf_flags["supports_tools"],
                    context_length = _positive_int_or_none(_model_info.get("context_length")),
                    chat_template = _chat_template,
                )

        # is_lora auto-detected from adapter_config.json on disk/HF.
        # DNS-probe wrap so offline loads skip 30-60s of soft-failed network
        # checks before the worker starts.
        with _hf_offline_if_dns_dead():
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

            # Unload any active Unsloth model to free VRAM
            if unsloth_backend.active_model_name:
                logger.info(
                    f"Unloading Unsloth model '{unsloth_backend.active_model_name}' before loading GGUF"
                )
                unsloth_backend.unload_model(unsloth_backend.active_model_name)

            # Inherit llama_extra_args from the previous load when the request
            # omits the field (the chat-settings Apply path doesn't round-trip
            # them; explicit [] still clears). Gated on (model_identifier,
            # hf_variant) to refuse cross-model pickup, and shadowing flags are
            # stripped so an inherited override can't win the last-wins CLI
            # parse against a freshly-supplied first-class field.
            if request.llama_extra_args is None and llama_backend.extra_args:
                source = llama_backend.extra_args_source
                # Compare against the resolved variant, not the request
                # field: callers commonly omit gguf_variant for local
                # ``.gguf`` paths and HF auto-pick flows. ``config.gguf_
                # variant`` is the variant load_model was actually
                # invoked with (see the HF / local branches below), so
                # both sides of the comparison key off the same string.
                resolved_variant = (config.gguf_variant or "").lower()
                request_variant = (request.gguf_variant or "").lower()
                stored_variant = (source[1] or "").lower() if source else ""
                same_model = bool(
                    source and source[0] and source[0].lower() == model_identifier.lower()
                )
                if request.gguf_variant:
                    variant_mismatch = request_variant != stored_variant
                else:
                    variant_mismatch = bool(stored_variant and resolved_variant != stored_variant)
                same_source = same_model and not variant_mismatch
                if not same_source:
                    logger.info(
                        "Not inheriting llama_extra_args: stored args came from %s, loading %s",
                        source,
                        (model_identifier, resolved_variant),
                    )
                    # Cross-model: clear explicitly so the backend doesn't
                    # inherit via "no opinion" semantics.
                    extra_llama_args = []
                else:
                    # Strip only the groups whose first-class field was set by
                    # the caller, so an inherited --chat-template-file survives
                    # an Apply that omits chat_template_override.
                    fields_set = getattr(request, "model_fields_set", set())
                    stripped = strip_shadowing_flags(
                        llama_backend.extra_args,
                        strip_context = "max_seq_length" in fields_set,
                        strip_cache = "cache_type_kv" in fields_set,
                        strip_spec = (
                            "speculative_type" in fields_set or "spec_draft_n_max" in fields_set
                        ),
                        strip_template = "chat_template_override" in fields_set,
                        strip_split_mode = _should_strip_split_mode(
                            request, llama_backend.extra_args
                        ),
                    )
                    try:
                        extra_llama_args = validate_extra_args(stripped)
                    except ValueError:
                        # Shouldn't happen on already-validated args; degrade to
                        # no-extras rather than 400 if managed flags changed.
                        logger.warning(
                            "Stored llama_extra_args failed revalidation; loading without them: %s",
                            stripped,
                        )
                        extra_llama_args = []
                    else:
                        if extra_llama_args:
                            logger.info(
                                "Inheriting llama_extra_args from previous "
                                "load (same model, shadow-stripped): %s",
                                extra_llama_args,
                            )

            # Route to HF or local mode based on config. Run in a thread so the
            # event loop stays free for progress polling and other requests
            # during the (potentially long) GGUF download + llama-server start.
            _n_parallel = getattr(fastapi_request.app.state, "llama_parallel_slots", 1)

            # Load kwargs common to HF and local modes; the two differ only by
            # the model-source args (hf_repo/-token vs gguf_path/mmproj).
            _common_load_kwargs = dict(
                model_identifier = config.identifier,
                is_vision = config.is_vision,
                n_ctx = request.max_seq_length,
                chat_template_override = request.chat_template_override,
                cache_type_kv = request.cache_type_kv,
                speculative_type = request.speculative_type,
                spec_draft_n_max = request.spec_draft_n_max,
                n_parallel = _n_parallel,
                extra_args = extra_llama_args,
            )
            if config.gguf_hf_repo:
                # HF mode: download via huggingface_hub then start llama-server
                _source_load_kwargs = dict(
                    hf_repo = config.gguf_hf_repo,
                    hf_variant = config.gguf_variant,
                    hf_token = request.hf_token,
                )
            else:
                # Local mode: llama-server loads via -m <path>
                if native_grant_backed:
                    if config.gguf_mmproj_file:
                        _validate_native_gguf_companion(
                            config.gguf_mmproj_file, config.gguf_file, "vision companion"
                        )
                    if config.gguf_mtp_file:
                        # The drafter is optional (unlike mmproj for a vision
                        # model): drop it rather than fail the load.
                        try:
                            _validate_native_gguf_companion(
                                config.gguf_mtp_file, config.gguf_file, "MTP drafter"
                            )
                        except HTTPException as exc:
                            logger.warning("Dropping MTP drafter for native load: %s", exc.detail)
                            config.gguf_mtp_file = None
                _source_load_kwargs = dict(
                    gguf_path = config.gguf_file,
                    mmproj_path = config.gguf_mmproj_file,
                    mtp_draft_path = config.gguf_mtp_file,
                    # Pass the resolved variant so _extra_args_source keys off
                    # the same string the inheritance check at the top of /load
                    # uses (#5401 followup).
                    hf_variant = config.gguf_variant,
                )

            # Run a single load attempt with the given tensor flag + extras.
            async def _attempt_gguf_load(
                tensor_parallel: bool, attempt_extra_args: Optional[list[str]]
            ) -> bool:
                attempt_kwargs = {
                    **_common_load_kwargs,
                    "extra_args": attempt_extra_args,
                }
                return await asyncio.to_thread(
                    llama_backend.load_model,
                    **_source_load_kwargs,
                    **attempt_kwargs,
                    tensor_parallel = tensor_parallel,
                )

            # Tensor parallelism is arch-gated in llama.cpp and crashes some loads
            # outright (e.g. Gemma 3n aborts with a GGML_ASSERT). The helper auto-
            # falls back to layer split so the checkbox never blocks a model from
            # loading; the response reports the backend's actual tensor_parallel
            # state so the UI toggle reflects the fallback.
            success = await load_with_tensor_fallback(
                _attempt_gguf_load,
                requested_tensor = request.tensor_parallel,
                extra_args = extra_llama_args,
                label = config.identifier,
                cancelled = llama_backend.load_cancelled,
            )

            if not success:
                raise HTTPException(
                    status_code = 500,
                    detail = f"Failed to load GGUF model: {model_log_label if native_grant_backed else config.display_name}",
                )

            logger.info(
                f"Loaded GGUF model via llama-server: {model_log_label if native_grant_backed else config.identifier}"
            )

            # Audio detection moved into load_model under _serial_load_lock (#5642).
            _gguf_audio = llama_backend._audio_type
            _gguf_is_audio = llama_backend._is_audio
            llama_backend._native_display_label = model_log_label if native_grant_backed else None
            llama_backend._native_grant_backed = bool(native_grant_backed)
            if _gguf_is_audio:
                logger.info(f"GGUF model detected as audio: audio_type={_gguf_audio}")

            inference_config = load_inference_config(config.identifier)

            return LoadResponse(
                status = "loaded",
                model = model_log_label if native_grant_backed else config.identifier,
                display_name = model_log_label if native_grant_backed else config.display_name,
                is_vision = llama_backend.is_vision,
                is_lora = False,
                is_gguf = True,
                is_audio = _gguf_is_audio,
                audio_type = _gguf_audio,
                has_audio_input = llama_backend._has_audio_input,
                inference = inference_config,
                requires_trust_remote_code = bool(inference_config.get("trust_remote_code", False)),
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
                speculative_type = llama_backend.requested_spec_mode,
                spec_draft_n_max = llama_backend.spec_draft_n_max,
                tensor_parallel = llama_backend.tensor_parallel,
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
                logger.info("Shutting down export subprocess to free GPU memory for inference")
                exp_backend._shutdown_subprocess()
                exp_backend.current_checkpoint = None
                exp_backend.is_vision = False
                exp_backend.is_peft = False
        except Exception as e:
            logger.warning("Could not shut down export subprocess: %s", e)

        # Auto-detect quantization for LoRA adapters from adapter_config.json.
        # The training pipeline writes "unsloth_training_method" ('qlora' or
        # 'lora'); only LoRA (16-bit) needs load_in_4bit=False.
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
                        # No unsloth_training_method -- fall back to base model name
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

        # Load in a thread so the event loop stays free for download progress
        # polling and other requests.
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
            # Check if YAML says this model needs trust_remote_code.
            if not request.trust_remote_code:
                model_defaults = load_model_defaults(config.identifier)
                yaml_trust = model_defaults.get("inference", {}).get("trust_remote_code", False)
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

        # Classify reasoning/tool flags via the GGUF sniffer.
        _sf_flags = _detect_safetensors_features(backend, _chat_template)

        return LoadResponse(
            status = "loaded",
            model = model_log_label if native_grant_backed else config.identifier,
            display_name = model_log_label if native_grant_backed else config.display_name,
            is_vision = config.is_vision,
            is_lora = config.is_lora,
            is_gguf = False,
            is_audio = config.is_audio,
            audio_type = config.audio_type,
            has_audio_input = config.has_audio_input,
            inference = inference_config,
            requires_trust_remote_code = bool(inference_config.get("trust_remote_code", False)),
            supports_reasoning = _sf_flags["supports_reasoning"],
            reasoning_style = _sf_flags["reasoning_style"],
            reasoning_always_on = _sf_flags["reasoning_always_on"],
            supports_preserve_thinking = _sf_flags["supports_preserve_thinking"],
            supports_tools = _sf_flags["supports_tools"],
            context_length = _positive_int_or_none(_model_info.get("context_length")),
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
        # User-facing validation (e.g. "Invalid gpu_ids [99]"): redact paths, keep detail.
        raise HTTPException(status_code = 400, detail = redact_native_paths(str(e)))
    except Exception as e:
        # Friendlier message for models Unsloth cannot load.
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
        msg = redact_native_paths(str(e))
        if any(h.lower() in msg.lower() for h in not_supported_hints):
            msg = f"This model is not supported yet. Try a different model. (Original error: {msg})"
        raise HTTPException(status_code = 500, detail = f"Failed to load model: {msg}")


@router.post("/validate", response_model = ValidateModelResponse)
async def validate_model(
    request: ValidateModelRequest, current_subject: str = Depends(get_current_subject)
):
    """
    Lightweight validation endpoint for model identifiers.

    Checks that ModelConfig.from_identifier() can resolve model_path, but does
    NOT load model weights into GPU memory.
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
            detail = "Invalid model",
        )


@router.post("/unload", response_model = UnloadResponse)
async def unload_model(request: UnloadRequest, current_subject: str = Depends(get_current_subject)):
    """
    Unload a model from memory.
    Routes to the correct backend (llama-server for GGUF, Unsloth otherwise).
    """
    try:
        # Check if the GGUF backend has this model loaded or is loading it.
        llama_backend = get_llama_cpp_backend()
        if llama_backend.is_active and (
            llama_backend.model_identifier == request.model_path
            or is_registered_native_path_label(llama_backend.model_identifier, request.model_path)
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
        raise HTTPException(status_code = 500, detail = "Failed to unload model")


@studio_router.post("/cancel")
async def cancel_inference(request: Request, current_subject: str = Depends(get_current_subject)):
    """Cancel in-flight inference requests.

    Body (JSON, at least one key required):
      cancel_id    - preferred: per-run UUID, matched exclusively.
      session_id   - fallback when cancel_id is absent.
      completion_id - fallback when cancel_id is absent.

    A cancel_id arriving before its stream registers is stashed briefly and
    replayed on registration. Returns {"cancelled": N}.
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
    # `message_id` is the Anthropic passthrough's per-run identifier, so
    # /v1/messages clients can cancel by their native id.
    for k in ("completion_id", "session_id", "message_id"):
        v = body.get(k)
        if isinstance(v, str) and v:
            keys.append(v)

    if not keys:
        return {"cancelled": 0}

    n = _cancel_by_keys(keys)
    return {"cancelled": n}


@studio_router.post("/tool-confirm")
async def confirm_tool_call(
    request: ToolConfirmRequest, current_subject: str = Depends(get_current_subject)
):
    matched = resolve_tool_decision(
        request.approval_id,
        request.decision,
        session_id = request.session_id,
    )
    if not matched:
        raise HTTPException(status_code = 404, detail = "No pending tool call confirmation")
    return {"resolved": True}


@router.post("/generate/stream")
async def generate_stream(
    request: GenerateRequest, current_subject: str = Depends(get_current_subject)
):
    """
    Generate a chat response with Server-Sent Events (SSE) streaming.

    For vision models, provide image_base64 (base64-encoded image).
    """
    backend = get_inference_backend()

    if not backend.active_model_name:
        raise HTTPException(
            status_code = 400, detail = "No model loaded. Call POST /inference/load first."
        )

    # Decode image if provided (vision models)
    image = None
    if request.image_base64:
        try:
            import base64
            from PIL import Image
            from io import BytesIO

            # Check current model supports vision
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
            raise log_and_http_error(
                e,
                400,
                "Failed to decode image",
                event = "inference.decode_image_failed",
                log = logger,
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
async def get_status(current_subject: str = Depends(get_current_subject)):
    """
    Get current inference backend status.
    Reports whichever backend (Unsloth or llama-server) is active.
    """
    try:
        llama_backend = get_llama_cpp_backend()

        # MTP probe + freshness check (both cached); drive the UI banner.
        try:
            _bin = type(llama_backend)._find_llama_server_binary()
            _caps = type(llama_backend).probe_server_capabilities(_bin)
            _supports_mtp = bool(_caps.get("supports_mtp", False))
        except Exception:
            _bin = None
            _supports_mtp = True  # fail open
        try:
            from utils.llama_cpp_freshness import check_prebuilt_freshness
            _freshness = check_prebuilt_freshness(_bin)
        except Exception:
            _freshness = {}
        _stale = bool(_freshness.get("stale"))
        _installed_tag = _freshness.get("installed_tag")
        _latest_tag = _freshness.get("latest_tag")

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
                model_identifier = None if _native_grant_backed else _model_id,
                is_vision = llama_backend.is_vision,
                is_gguf = True,
                gguf_variant = llama_backend.hf_variant,
                is_audio = getattr(llama_backend, "_is_audio", False),
                audio_type = _audio_type,
                has_audio_input = getattr(llama_backend, "_has_audio_input", False),
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
                speculative_type = llama_backend.requested_spec_mode,
                spec_draft_n_max = llama_backend.spec_draft_n_max,
                tensor_parallel = llama_backend.tensor_parallel,
                llama_cpp_supports_mtp = _supports_mtp,
                spec_fallback_reason = llama_backend.spec_fallback_reason,
                llama_cpp_prebuilt_stale = _stale,
                llama_cpp_installed_tag = _installed_tag,
                llama_cpp_latest_tag = _latest_tag,
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
            chat_template_info.get("template") if isinstance(chat_template_info, dict) else None
        )

        # Non-GGUF: classify from the loaded template.
        _sf_flags = _detect_safetensors_features(backend, chat_template)
        inference_config = (
            load_inference_config(backend.active_model_name) if backend.active_model_name else None
        )

        return InferenceStatusResponse(
            active_model = backend.active_model_name,
            model_identifier = backend.active_model_name,
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
            supports_reasoning = _sf_flags["supports_reasoning"],
            reasoning_style = _sf_flags["reasoning_style"],
            reasoning_always_on = _sf_flags["reasoning_always_on"],
            supports_preserve_thinking = _sf_flags["supports_preserve_thinking"],
            supports_tools = _sf_flags["supports_tools"],
            context_length = _positive_int_or_none(model_info.get("context_length")),
            chat_template = chat_template,
            llama_cpp_supports_mtp = _supports_mtp,
            llama_cpp_prebuilt_stale = _stale,
            llama_cpp_installed_tag = _installed_tag,
            llama_cpp_latest_tag = _latest_tag,
        )

    except Exception as e:
        logger.error(f"Error getting status: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = "Failed to get status")


@router.get("/load-progress", response_model = LoadProgressResponse)
async def get_load_progress(current_subject: str = Depends(get_current_subject)):
    """
    Return the active GGUF load's mmap/upload progress.

    During the warmup window after a GGUF download -- when llama-server pages
    ~tens-to-hundreds of GB of shards into the page cache before pushing layers
    to VRAM -- ``/api/inference/status`` only shows a generic spinner. This
    exposes sampled progress so the UI can render a real bar plus rate/ETA.

    Returns an empty payload (``phase=null, bytes=0``) when no load is in
    flight. The frontend should stop polling once ``phase`` becomes ``ready``.
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
    Returns JSON with base64-encoded WAV audio.
    Works with both GGUF (llama-server) and Unsloth/transformers backends.
    """
    import base64

    # Extract text from the last user message
    _, chat_messages, _ = _extract_content_parts(payload.messages)
    if not chat_messages:
        raise HTTPException(status_code = 400, detail = "No messages provided.")
    last_user_msg = next((m for m in reversed(chat_messages) if m["role"] == "user"), None)
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
            max_new_tokens = _effective_max_tokens(payload) or 2048,
            repetition_penalty = payload.repetition_penalty,
        )
    else:
        backend = get_inference_backend()
        if not backend.active_model_name:
            raise HTTPException(status_code = 400, detail = "No model loaded.")
        model_info = backend.models.get(backend.active_model_name, {})
        if not model_info.get("is_audio"):
            raise HTTPException(status_code = 400, detail = "Active model is not an audio model.")
        model_name = backend.active_model_name
        gen = lambda: backend.generate_audio_response(
            text = text,
            temperature = payload.temperature,
            top_p = payload.top_p,
            top_k = payload.top_k,
            min_p = payload.min_p,
            max_new_tokens = _effective_max_tokens(payload) or 2048,
            repetition_penalty = payload.repetition_penalty,
            use_adapter = payload.use_adapter,
        )

    try:
        wav_bytes, sample_rate = await asyncio.get_event_loop().run_in_executor(None, gen)
    except Exception as e:
        logger.error(f"Audio generation error: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = safe_error_detail(e))

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
    # torchaudio.load needs a path or file-like with a format hint; write a
    # temp file so it can auto-detect the format.
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


# Reject oversized audio before decoding. base64 inflates raw bytes by ~4/3, so
# cap the encoded length to bound the upload. _MAX_AUDIO_SECONDS additionally
# bounds the *decoded* length, since a small compressed file (opus/flac/etc.)
# can expand to a far larger PCM array than the encoded-size cap implies.
_MAX_AUDIO_RAW_BYTES = 25 * 1024 * 1024
_MAX_AUDIO_B64_CHARS = _MAX_AUDIO_RAW_BYTES * 4 // 3
_MAX_AUDIO_SECONDS = 30 * 60
_WAV_HEADER_BYTES = 44
_MIN_TRANSCODE_AUDIO_SAMPLE_RATE = 8000


def _sniff_audio_container(raw: bytes) -> Optional[str]:
    """Return 'wav' or 'mp3' if the bytes are a container llama-server accepts
    directly (so we can forward them untouched), else None (needs transcoding)."""
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WAVE":
        return "wav"
    # mp3: ID3 tag, or an MPEG audio frame sync (no other accepted format leads
    # with 0xFF, so the simple sync check doesn't collide).
    if raw[:3] == b"ID3" or (len(raw) >= 2 and raw[0] == 0xFF and (raw[1] & 0xE0) == 0xE0):
        return "mp3"
    return None


def _mono_f32_to_wav_bytes(arr: np.ndarray, sample_rate: int) -> bytes:
    """Encode a mono float32 array as 16-bit PCM WAV bytes.

    Torch-free (numpy + stdlib only) so it works on no-torch GGUF-only installs;
    the shared audio_codecs helper pulls in torch at import time.
    """
    import io
    import wave

    arr = np.nan_to_num(np.asarray(arr, dtype = np.float32).flatten(), posinf = 0.0, neginf = 0.0)
    if arr.size == 0:
        raise ValueError("decoded audio is empty")
    peak = float(np.abs(arr).max())
    if peak > 1.0:
        arr = arr / peak
    pcm = (arr * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _resample_mono_linear(arr: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Small numpy-only resampler for upload size limiting."""
    if source_rate <= 0 or target_rate <= 0 or source_rate == target_rate:
        return arr
    duration = len(arr) / float(source_rate)
    target_len = max(1, int(round(duration * target_rate)))
    if target_len == len(arr):
        return arr
    source_x = np.linspace(0.0, duration, num = len(arr), endpoint = False)
    target_x = np.linspace(0.0, duration, num = target_len, endpoint = False)
    return np.interp(target_x, source_x, arr).astype(np.float32)


def _fit_transcoded_audio_to_wav_cap(arr: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    """Downsample only when needed so transcoded WAV stays within the upload cap."""
    if sample_rate <= 0:
        raise ValueError("decoded audio has an invalid sample rate")
    wav_bytes = _WAV_HEADER_BYTES + len(arr) * 2
    if wav_bytes <= _MAX_AUDIO_RAW_BYTES:
        return arr, sample_rate

    duration = len(arr) / float(sample_rate)
    max_samples = max(1, (_MAX_AUDIO_RAW_BYTES - _WAV_HEADER_BYTES) // 2)
    target_rate = int(max_samples // duration)
    if target_rate < _MIN_TRANSCODE_AUDIO_SAMPLE_RATE:
        raise ValueError("decoded audio exceeds the transcoded WAV size limit")
    target_rate = min(sample_rate, target_rate)
    fitted = _resample_mono_linear(arr, sample_rate, target_rate)
    if _WAV_HEADER_BYTES + len(fitted) * 2 > _MAX_AUDIO_RAW_BYTES:
        raise ValueError("decoded audio exceeds the transcoded WAV size limit")
    return fitted, target_rate


def _decode_audio_mono(raw: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes to (mono float32 array, native sample_rate).

    soundfile (libsndfile) reads wav/mp3/ogg/flac straight from memory. librosa
    (ffmpeg-backed) additionally covers m4a/webm but needs a real path and is
    absent on no-torch GGUF-only installs. Both imports are inside the fallback
    so a missing decoder degrades to the next one (and finally a clear error)
    rather than crashing.
    """
    import io

    try:
        import soundfile as sf
        arr, sr = sf.read(io.BytesIO(raw), dtype = "float32")
    except Exception:
        try:
            import librosa
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "this audio format needs librosa, which is not installed in "
                "GGUF-only environments; use wav, mp3, ogg or flac"
            ) from e
        import os
        import tempfile
        from utils.paths import ensure_dir, tmp_root

        with tempfile.NamedTemporaryFile(
            suffix = ".audio",
            delete = False,
            dir = str(ensure_dir(tmp_root())),
        ) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        try:
            arr, sr = librosa.load(tmp_path, sr = None, mono = True)
        finally:
            os.unlink(tmp_path)
    if arr.ndim > 1:
        arr = arr.mean(axis = 1)
    if sr > 0 and len(arr) > sr * _MAX_AUDIO_SECONDS:
        raise ValueError(f"decoded audio exceeds the {_MAX_AUDIO_SECONDS // 60}-minute limit")
    return arr, sr


def _prepare_audio_for_llama(b64: str) -> tuple[str, str]:
    """Return (base64, format) ready for llama-server's input_audio part.

    llama-server's API only accepts wav/mp3, and decodes/resamples/down-mixes
    them itself, so wav and mp3 uploads are forwarded untouched (no decode, no
    PCM payload inflation). Other containers (m4a/ogg/webm/flac) are decoded to
    a mono WAV. Blocking; call via a thread from async paths.
    """
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1] if "," in b64 else ""
    raw = base64.b64decode(b64)
    passthrough = _sniff_audio_container(raw)
    if passthrough is not None:
        return b64, passthrough

    arr, sr = _decode_audio_mono(raw)
    arr, sr = _fit_transcoded_audio_to_wav_cap(arr, sr)
    return base64.b64encode(_mono_f32_to_wav_bytes(arr, sr)).decode("ascii"), "wav"


def _inject_audio_part(messages: list[dict], audio_b64: str, audio_format: str) -> None:
    """Append an input_audio part to the last user message, in place.

    Audio rides in the message list like image_url parts do, so it flows through
    both the plain and tool-calling generation paths.
    """
    part = {
        "type": "input_audio",
        "input_audio": {"data": audio_b64, "format": audio_format},
    }
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                content.append(part)
            else:
                msg["content"] = [{"type": "text", "text": content or ""}, part]
            return


def _extract_content_parts(messages: list) -> tuple[str, list[dict], "Optional[str]"]:
    """
    Parse OpenAI-format messages into components the inference backend expects.

    Handles both plain-string ``content`` and multimodal content-part arrays
    (``[{type: "text", ...}, {type: "image_url", ...}]``).

    Returns:
        system_prompt:  System message text (empty string if none).
        chat_messages:  Non-system messages with content flattened to strings.
        image_base64:   Base64 of the *first* image found, or ``None``.
    """
    system_parts: list[str] = []
    chat_messages: list[dict] = []
    first_image_b64: Optional[str] = None

    for msg in messages:
        # ── System / developer messages → extract as system_prompt ────────
        if msg.role in ("system", "developer"):
            if isinstance(msg.content, str):
                system_parts.append(msg.content)
            elif isinstance(msg.content, list):
                # Unlikely but handle: join text parts
                system_parts.append("\n".join(p.text for p in msg.content if p.type == "text"))
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
                        # data:image/png;base64,<DATA> -> extract <DATA>
                        first_image_b64 = url.split(",", 1)[1] if "," in url else None
                    else:
                        logger.warning(f"Remote image URLs not yet supported: {url[:80]}...")
            combined_text = "\n".join(text_parts) if text_parts else ""
            chat_messages.append({"role": msg.role, "content": combined_text})

    return "\n\n".join(p for p in system_parts if p), chat_messages, first_image_b64


# ── External provider proxy ──────────────────────────────────────


# Providers whose stream helper translates `input_document` parts into a
# native attachment block on the wire. Anthropic: `_stream_anthropic` ->
# {type:"document", source:...}; OpenAI: `_stream_openai_responses` ->
# {type:"input_file", file_data|file_url}. Every other provider (gemini /
# mistral / kimi / openrouter / deepseek / custom OpenAI-compat) goes through
# the generic /chat/completions passthrough that forwards messages verbatim,
# so handing them an `input_document` part would 400 with an unknown
# content_part type.
_INPUT_DOCUMENT_PROVIDERS = frozenset({"anthropic", "openai"})


def _build_external_messages(
    messages: list,
    supports_vision: bool,
    provider_type: Optional[str] = None,
    base_url: Optional[str] = None,
) -> list[dict]:
    """
    Convert ChatMessage list to OpenAI-compatible dicts for external providers.

    Behaviour per content-part type:
    - `text`: always preserved.
    - `image_url`: preserved on vision providers; stripped on non-vision.
    - `input_document`: preserved ONLY when the provider's stream helper has
      explicit translation logic (Anthropic + OpenAI today, see
      ``_INPUT_DOCUMENT_PROVIDERS``). Stripped for every other provider so the
      unknown type doesn't reach generic /chat/completions and 400.
    - `reasoning`: OpenAI-only Responses reasoning item paired with a prior
      tool output. Forwarded ONLY when provider_type=="openai" so follow-up
      image edits can replay the required reasoning item.
    - `image_generation_call`: OpenAI-only Responses image reference. Forwarded
      ONLY when provider_type=="openai" so follow-up image edits can reference
      prior generated images.
    - `compaction`: Anthropic-only synthetic part (round-trips server-side
      compaction state). Forwarded ONLY when provider_type=="anthropic";
      stripped elsewhere so the unknown part doesn't reach generic
      /chat/completions and 400 (DeepSeek, Mistral, Gemini, Kimi, OpenRouter).
    """
    document_provider = provider_type in _INPUT_DOCUMENT_PROVIDERS
    anthropic = provider_type == "anthropic"
    openai = provider_type == "openai"
    # `extra_content` carries the assistant's text-part `thoughtSignature`
    # round-trip on Gemini's native streamGenerateContent endpoint. Custom
    # Gemini OpenAI-compat gateways (LiteLLM etc.) route through
    # /chat/completions where the field is unknown and can be rejected -- gate
    # strictly on the Google-hosted Gemini base.
    _native_gemini = False
    if provider_type == "gemini" and base_url:
        try:
            from urllib.parse import urlparse as _urlparse
            _host = (_urlparse(base_url).hostname or "").lower()
            _native_gemini = _host == "generativelanguage.googleapis.com"
        except Exception:
            _native_gemini = False
    emit_extra_content = _native_gemini

    _SERVER_BUILTIN_TOOL_NAMES = frozenset(
        {"web_search", "web_fetch", "code_execution", "image_generation"}
    )

    def _is_marked_server_builtin_tool_call(tc: Any) -> bool:
        """Return True iff `tc` is a synthetic provider-side tool card with a
        canonical builtin name and either:
          - the `args._server_tool` marker stamped by the backend, or
          - a Gemini `args.google.native_part` payload (durable replay signal
            for code_execution / image_generation that predates the marker).
        Such cards must not be forwarded to non-native providers: they aren't
        real user functions, so the receiving API rejects the orphan tool
        history. Real user functions with these names normally have neither
        signal.
        """
        if not isinstance(tc, dict):
            return False
        fn = tc.get("function")
        if not isinstance(fn, dict):
            return False
        name = (fn.get("name") or "").lower()
        if name not in _SERVER_BUILTIN_TOOL_NAMES:
            return False
        raw_args = fn.get("arguments") or ""
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except Exception:
            return False
        if not isinstance(args, dict):
            return False
        if args.get("_server_tool") is True:
            return True
        google = args.get("google")
        return isinstance(google, dict) and isinstance(google.get("native_part"), dict)

    # When we drop a server-side builtin tool_call, the matching `role="tool"`
    # follow-up must also be dropped -- else the provider gets an orphan
    # tool_call_id with no matching assistant call, which OpenAI Responses and
    # Anthropic both reject.
    dropped_server_builtin_tool_call_ids: set[str] = set()

    def _filter_tool_calls(tool_calls: Any) -> Optional[list]:
        """Sanitize assistant `tool_calls` for non-native-Gemini providers.

        Two concerns:
          1. `tool_calls[i].extra_content` carries Gemini-only thoughtSignature
             metadata; strip it for providers that can't parse the unknown key.
          2. Marked server-side builtin cards (`_server_tool: true` on a
             canonical builtin name, or a Gemini `native_part` payload) are
             Studio-internal tool cards from a prior native Gemini turn;
             forwarding them to OpenAI / Anthropic / custom OAI-compat gateways
             sends an orphan `tool_calls` entry (no matching tool declaration,
             often no matching `role="tool"` reply) that can be rejected. We
             record the dropped call_ids so the matching role=tool message is
             skipped below.
        Native Gemini keeps both untouched so the translator can replay them
        via `native_part`.
        """
        if not tool_calls:
            return None
        if not isinstance(tool_calls, list):
            return tool_calls
        if emit_extra_content:
            return tool_calls
        cleaned: list = []
        for _tc in tool_calls:
            if _is_marked_server_builtin_tool_call(_tc):
                _tc_id = _tc.get("id") if isinstance(_tc, dict) else None
                if isinstance(_tc_id, str) and _tc_id:
                    dropped_server_builtin_tool_call_ids.add(_tc_id)
                continue
            if not isinstance(_tc, dict):
                cleaned.append(_tc)
                continue
            if "extra_content" not in _tc:
                cleaned.append(_tc)
                continue
            _stripped = {k: v for k, v in _tc.items() if k != "extra_content"}
            cleaned.append(_stripped)
        return cleaned

    result = []
    for msg in messages:
        # Drop role=tool messages whose matching server-builtin tool_call was
        # filtered above. An orphan tool_result with no matching tool_call is
        # rejected by OpenAI Responses and Anthropic.
        if (
            msg.role == "tool"
            and isinstance(msg.tool_call_id, str)
            and msg.tool_call_id in dropped_server_builtin_tool_call_ids
        ):
            continue
        if isinstance(msg.content, str):
            # Drop bare assistant messages with no content AND no tool_calls
            # (some providers reject empty assistant turns). Preserve assistant
            # turns whose only payload is tool_calls so multi-turn
            # function-call loops round-trip.
            if msg.role == "assistant" and not msg.content.strip() and not msg.tool_calls:
                continue
            out: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.role == "assistant" and msg.tool_calls:
                _tcs = _filter_tool_calls(msg.tool_calls)
                if _tcs:
                    out["tool_calls"] = _tcs
                elif not msg.content.strip():
                    # Every tool_call was a dropped synthetic provider card;
                    # the turn would be an empty
                    # `{"role":"assistant","content":""}` that some providers
                    # reject. Skip it entirely.
                    continue
            if msg.role == "tool":
                if msg.tool_call_id:
                    out["tool_call_id"] = msg.tool_call_id
                if msg.name:
                    out["name"] = msg.name
            if emit_extra_content and msg.role == "assistant" and msg.extra_content:
                out["extra_content"] = msg.extra_content
            result.append(out)
            continue
        # Assistant messages with content=None but populated tool_calls are
        # valid (post-tool-call turn). Forward them so the provider helper can
        # rebuild the functionCall part.
        if msg.content is None and msg.role == "assistant" and msg.tool_calls:
            _filtered_tcs = _filter_tool_calls(msg.tool_calls)
            if not _filtered_tcs:
                # Every tool_call was provider-side synthetic and dropped;
                # skip the whole message to avoid an empty assistant turn.
                continue
            _assistant_only: dict[str, Any] = {
                "role": "assistant",
                "content": "",
                "tool_calls": _filtered_tcs,
            }
            if emit_extra_content and msg.extra_content:
                _assistant_only["extra_content"] = msg.extra_content
            result.append(_assistant_only)
            continue
        if isinstance(msg.content, list):
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
                    elif part.type == "reasoning" and openai and msg.role == "assistant":
                        reasoning: dict[str, Any] = {
                            "type": "reasoning",
                            "id": part.id,
                            "summary": part.summary,
                        }
                        if part.status:
                            reasoning["status"] = part.status
                        parts.append(reasoning)
                    elif (
                        part.type == "image_generation_call" and openai and msg.role == "assistant"
                    ):
                        # ExternalProviderClient maps this onto a top-level
                        # Responses input item after the current user prompt,
                        # or onto `previous_response_id` when response_id is
                        # available from the prior turn.
                        image_ref = {"type": "image_generation_call", "id": part.id}
                        if getattr(part, "response_id", None):
                            image_ref["response_id"] = part.response_id
                        parts.append(image_ref)
                    elif part.type == "input_document" and document_provider:
                        # ExternalProviderClient maps this onto Anthropic's
                        # `document` or OpenAI Responses' `input_file` block;
                        # every other provider would 400 on the unknown part.
                        doc: dict[str, Any] = {"type": "input_document"}
                        if part.file_data:
                            doc["file_data"] = part.file_data
                        if part.file_url:
                            doc["file_url"] = part.file_url
                        if part.filename:
                            doc["filename"] = part.filename
                        if part.media_type:
                            doc["media_type"] = part.media_type
                        parts.append(doc)
                    elif part.type == "compaction" and anthropic:
                        # Anthropic stream helper forwards this as a native
                        # `compaction` block; every other provider would 400 on
                        # the unknown part, so gate by provider_type.
                        parts.append({"type": "compaction", "content": part.content})
                entry: dict[str, Any] = {"role": msg.role, "content": parts}
                if msg.role == "assistant" and msg.tool_calls:
                    _tcs = _filter_tool_calls(msg.tool_calls)
                    if _tcs:
                        entry["tool_calls"] = _tcs
                    elif not parts:
                        # All tool_calls were synthetic and dropped, and no
                        # content parts survived. Skip rather than forward an
                        # empty assistant turn that downstream providers reject.
                        continue
                elif msg.role == "assistant" and not parts:
                    continue
                if msg.role == "tool":
                    if msg.tool_call_id:
                        entry["tool_call_id"] = msg.tool_call_id
                    if msg.name:
                        entry["name"] = msg.name
                if emit_extra_content and msg.role == "assistant" and msg.extra_content:
                    entry["extra_content"] = msg.extra_content
                result.append(entry)
            else:
                # Non-vision provider: strip images / documents, keep text,
                # optionally keep compaction (Anthropic only --
                # compaction-capable Anthropic models all report
                # supports_vision=True today, but gate here for safety).
                preserved = []
                for p in msg.content:
                    if p.type == "text":
                        preserved.append({"type": "text", "text": p.text})
                    elif p.type == "reasoning" and openai and msg.role == "assistant":
                        reasoning: dict[str, Any] = {
                            "type": "reasoning",
                            "id": p.id,
                            "summary": p.summary,
                        }
                        if p.status:
                            reasoning["status"] = p.status
                        preserved.append(reasoning)
                    elif p.type == "image_generation_call" and openai and msg.role == "assistant":
                        image_ref = {"type": "image_generation_call", "id": p.id}
                        if getattr(p, "response_id", None):
                            image_ref["response_id"] = p.response_id
                        preserved.append(image_ref)
                    elif p.type == "compaction" and anthropic:
                        preserved.append({"type": "compaction", "content": p.content})
                if msg.role == "assistant" and not preserved:
                    continue
                if len(preserved) == 1 and preserved[0]["type"] == "text":
                    # Single text part collapses to a string for providers that
                    # don't accept content arrays.
                    entry = {"role": msg.role, "content": preserved[0]["text"]}
                else:
                    entry = {"role": msg.role, "content": preserved}
                if msg.role == "assistant" and msg.tool_calls:
                    _tcs = _filter_tool_calls(msg.tool_calls)
                    if _tcs:
                        entry["tool_calls"] = _tcs
                    else:
                        # All tool_calls were synthetic and dropped; skip if no
                        # content survived either.
                        _entry_content = entry.get("content")
                        _has_text = (
                            isinstance(_entry_content, str) and _entry_content.strip()
                        ) or (isinstance(_entry_content, list) and len(_entry_content) > 0)
                        if not _has_text:
                            continue
                if msg.role == "tool":
                    if msg.tool_call_id:
                        entry["tool_call_id"] = msg.tool_call_id
                    if msg.name:
                        entry["name"] = msg.name
                if emit_extra_content and msg.role == "assistant" and msg.extra_content:
                    entry["extra_content"] = msg.extra_content
                result.append(entry)
    return result


async def _proxy_to_external_provider(
    payload: ChatCompletionRequest, request: Request
) -> StreamingResponse:
    """
    Proxy a chat completion request to an external LLM provider.

    Resolves provider config (DB or registry), decrypts the API key, and
    streams the response back in OpenAI SSE format.
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

    # Build messages, preserving multimodal content for vision providers
    from core.inference.providers import get_provider_info as _get_provider_info

    _pinfo = _get_provider_info(provider_type) or {}
    _supports_vision = _pinfo.get("supports_vision", False)
    chat_messages = _build_external_messages(
        payload.messages,
        _supports_vision,
        provider_type = provider_type,
        base_url = base_url,
    )

    client = ExternalProviderClient(
        provider_type = provider_type,
        base_url = base_url,
        api_key = api_key,
    )

    # `top_k` defaults to 20 in ChatCompletionRequest because the local path
    # expects an int, but the external-provider path treats "field omitted from
    # JSON" as "use provider default" so callers sending only model/messages
    # don't silently get different sampling than before this PR. Pydantic's
    # `model_fields_set` tracks explicit-vs-default per request.
    _top_k_explicit = payload.top_k if "top_k" in payload.model_fields_set else None

    async def _stream():
        gen = client.stream_chat_completion(
            messages = chat_messages,
            model = model,
            temperature = payload.temperature,
            top_p = payload.top_p,
            # Honor max_completion_tokens when max_tokens is absent, so a
            # provider-routed request capped only by the newer field still gets
            # a limit instead of falling back to the provider default.
            max_tokens = _effective_max_tokens(payload),
            presence_penalty = payload.presence_penalty,
            top_k = _top_k_explicit,
            enable_thinking = payload.enable_thinking,
            reasoning_effort = payload.reasoning_effort,
            enabled_tools = payload.enabled_tools,
            enable_prompt_caching = payload.enable_prompt_caching,
            openai_code_exec_container_id = payload.openai_code_exec_container_id,
            anthropic_code_exec_container_id = payload.anthropic_code_exec_container_id,
            prompt_cache_ttl = payload.prompt_cache_ttl,
            compaction_threshold = payload.compaction_threshold,
            tools = payload.tools,
            tool_choice = payload.tool_choice,
            fast_mode = payload.fast_mode,
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


# ── OpenAI shell-tool container management ───────────────────────


def _resolve_openai_cloud_client(body: OpenAIContainerRequest) -> ExternalProviderClient:
    """
    Decrypt the API key + validate the base URL points at OpenAI cloud, then
    build an ExternalProviderClient for the three container CRUD endpoints
    below. The shell tool only exists on api.openai.com, so rejecting non-cloud
    bases up front prevents confusing 404s on ollama / llama.cpp / vLLM /
    custom presets.
    """
    base_url = body.provider_base_url or get_base_url("openai")
    if not base_url or "api.openai.com" not in base_url:
        raise HTTPException(
            status_code = 400,
            detail = (
                "OpenAI container management is only available on the "
                "managed cloud (api.openai.com). The provider's base URL "
                f"points at {base_url!r}."
            ),
        )
    try:
        api_key = decrypt_api_key(body.encrypted_api_key)
    except Exception as exc:
        logger.warning("external_provider.decrypt_failed", error = str(exc))
        raise HTTPException(
            status_code = 400,
            detail = "Failed to decrypt API key. The server key may have changed — try refreshing the page.",
        )
    return ExternalProviderClient(
        provider_type = "openai",
        base_url = base_url,
        api_key = api_key,
    )


def _summarize_container(raw: dict) -> OpenAIContainerSummary:
    expires = raw.get("expires_after")
    expires_minutes: Optional[int] = None
    if isinstance(expires, dict):
        minutes = expires.get("minutes")
        if isinstance(minutes, int):
            expires_minutes = minutes
    return OpenAIContainerSummary(
        id = str(raw.get("id") or ""),
        name = raw.get("name"),
        created_at = raw.get("created_at") if isinstance(raw.get("created_at"), int) else None,
        last_active_at = raw.get("last_active_at")
        if isinstance(raw.get("last_active_at"), int)
        else None,
        expires_after_minutes = expires_minutes,
        status = raw.get("status") if isinstance(raw.get("status"), str) else None,
    )


@router.post(
    "/external/openai/containers/list",
    response_model = ListOpenAIContainersResponse,
)
async def list_openai_containers(
    body: OpenAIContainerRequest, current_subject: str = Depends(get_current_subject)
) -> ListOpenAIContainersResponse:
    """List the user's OpenAI shell-tool containers."""
    client = _resolve_openai_cloud_client(body)
    try:
        try:
            raw = await client.list_openai_containers()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            raise HTTPException(
                status_code = exc.response.status_code if exc.response else 502,
                detail = f"OpenAI rejected /containers list: {detail}",
            )
        except httpx.HTTPError as exc:
            raise log_and_http_error(
                exc,
                502,
                "Could not reach OpenAI.",
                event = "openai_container_list.transport_error",
                log = logger,
            )
        # OpenAI keeps expired containers in /v1/containers indefinitely with
        # status="expired" -- dead but still listed. Hide them so the picker
        # only shows usable containers.
        return ListOpenAIContainersResponse(
            containers = [
                _summarize_container(c)
                for c in raw
                if isinstance(c, dict) and c.get("status") != "expired"
            ],
        )
    finally:
        await client.close()


@router.post(
    "/external/openai/containers/create",
    response_model = OpenAIContainerSummary,
)
async def create_openai_container(
    body: CreateOpenAIContainerBody, current_subject: str = Depends(get_current_subject)
) -> OpenAIContainerSummary:
    """Create a named container with the user-chosen idle TTL."""
    client = _resolve_openai_cloud_client(body)
    try:
        try:
            raw = await client.create_openai_container(
                name = body.name,
                ttl_minutes = body.ttl_minutes,
            )
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            raise HTTPException(
                status_code = exc.response.status_code if exc.response else 502,
                detail = f"OpenAI rejected /containers create: {detail}",
            )
        except httpx.HTTPError as exc:
            raise log_and_http_error(
                exc,
                502,
                "Could not reach OpenAI.",
                event = "openai_container_create.transport_error",
                log = logger,
            )
        if not isinstance(raw, dict):
            raise HTTPException(
                status_code = 502,
                detail = "OpenAI returned an unexpected container payload.",
            )
        return _summarize_container(raw)
    finally:
        await client.close()


@router.post("/external/openai/containers/delete", status_code = 204)
async def delete_openai_container(
    body: DeleteOpenAIContainerBody, current_subject: str = Depends(get_current_subject)
) -> None:
    """Delete a named container by id."""
    logger.info(
        "openai_container_delete.request subject=%s container_id=%s base_url=%s",
        current_subject,
        body.container_id,
        body.provider_base_url,
    )
    client = _resolve_openai_cloud_client(body)
    try:
        try:
            await client.delete_openai_container(body.container_id)
            logger.info(
                "openai_container_delete.success container_id=%s",
                body.container_id,
            )
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500] if exc.response is not None else str(exc)
            logger.warning(
                "openai_container_delete.openai_rejected container_id=%s status=%s body=%s",
                body.container_id,
                exc.response.status_code if exc.response else None,
                detail,
            )
            raise HTTPException(
                status_code = exc.response.status_code if exc.response else 502,
                detail = f"OpenAI rejected /containers delete: {detail}",
            )
        except httpx.HTTPError as exc:
            raise log_and_http_error(
                exc,
                502,
                "Could not reach OpenAI.",
                event = "openai_container_delete.transport_error",
                log = logger,
            )
    finally:
        await client.close()


@router.post("/chat/completions")
async def openai_chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    OpenAI-compatible chat completions endpoint.

    Supports multimodal messages: ``content`` may be a plain string or a list
    of content parts (``text`` / ``image_url``).

    Non-streaming (default): returns a single ChatCompletion JSON object.
    Streaming:               returns SSE chunks matching OpenAI's format.

    ``stream`` defaults to ``false`` per OpenAI's spec; clients opt into SSE by
    sending ``stream: true``.

    Routes to the correct backend automatically:
    - GGUF models → llama-server via LlamaCppBackend
    - Other models → Unsloth/transformers via InferenceBackend
    """
    # OpenAI's newer "developer" role is equivalent to "system". Normalize it
    # before provider routing so external providers (which may not accept the
    # "developer" role) get "system" too, matching the local path.
    for _m in payload.messages:
        if _m.role == "developer":
            _m.role = "system"

    if payload.logprobs:
        _raise_unsupported_openai_parameter(
            "logprobs", "logprobs is not supported for chat completions."
        )
    if payload.top_logprobs is not None:
        _raise_unsupported_openai_parameter(
            "top_logprobs", "top_logprobs is not supported for chat completions."
        )

    # ── External provider routing ────────────────────────────────
    # encrypted_api_key is optional -- local providers (llama.cpp / vLLM / Ollama) may run without auth.
    if payload.provider_id or payload.provider_type:
        if payload.confirm_tool_calls and (
            payload.enable_tools is True
            or bool(payload.enabled_tools)
            or bool(payload.tools)
            or bool(payload.openai_code_exec_container_id)
            or bool(payload.anthropic_code_exec_container_id)
        ):
            raise HTTPException(
                status_code = 400,
                detail = openai_error_body(
                    "confirm_tool_calls is only supported for local streaming tools.",
                    status = 400,
                    code = "invalid_request_error",
                    param = "confirm_tool_calls",
                ),
            )
        if _wants_multiple_choices(payload):
            _raise_unsupported_n("external provider chat completions")
        return await _proxy_to_external_provider(payload, request)

    # Reject a malformed function tool here: it would otherwise reach
    # llama-server and surface as an opaque 500 "Failed to parse tools".
    if payload.tools:
        for _tool in payload.tools:
            if not isinstance(_tool, dict):
                continue
            # llama-server 500s ("Failed to parse tools: Missing tool type") when
            # a function tool omits "type". Default it to "function" so a
            # well-formed tool isn't rejected over a missing discriminator (and a
            # malformed one still surfaces as a clean 400 below, not a 500).
            if _tool.get("type") is None and isinstance(_tool.get("function"), dict):
                _tool["type"] = "function"
            if _tool.get("type") != "function":
                continue
            _fn = _tool.get("function")
            _name = _fn.get("name") if isinstance(_fn, dict) else None
            if not isinstance(_name, str) or not _name.strip():
                raise HTTPException(
                    status_code = 400,
                    detail = openai_error_body(
                        "Invalid 'tools': each tool must have a 'function' with a 'name'.",
                        status = 400,
                        code = "invalid_value",
                        param = "tools",
                    ),
                )

    llama_backend = get_llama_cpp_backend()
    using_gguf = llama_backend.is_loaded

    # OpenAI-SDK clients send ``chat_template_kwargs`` via ``extra_body``, which
    # the SDK spreads into the request body at the top level. Studio's
    # ChatCompletionRequest has ``extra="allow"`` so pydantic stashes them in
    # ``model_extra``, but downstream generators consume the typed
    # ``payload.enable_thinking``. Lift ``enable_thinking`` from the extra-body
    # chat_template_kwargs onto the typed field so clients that only know the
    # OpenAI shape (data_designer recipe runs, etc.) can still control the
    # reasoning preamble.
    _extra = getattr(payload, "model_extra", None)
    if payload.enable_thinking is None and isinstance(_extra, dict):
        _tpl_kw = _extra.get("chat_template_kwargs")
        if isinstance(_tpl_kw, dict) and "enable_thinking" in _tpl_kw:
            payload.enable_thinking = bool(_tpl_kw["enable_thinking"])

    # ── Determine which backend is active ─────────────────────
    # Single-model server: any model name serves the loaded model (drop-in
    # OpenAI compat), so payload.model is only a fallback label here.
    if using_gguf:
        model_name = llama_backend.model_identifier or payload.model
        if getattr(llama_backend, "_is_audio", False):
            if _wants_multiple_choices(payload):
                _raise_unsupported_n("GGUF audio chat completions")
            return await generate_audio(payload, request)
    else:
        backend = get_inference_backend()
        if not backend.active_model_name:
            raise HTTPException(
                status_code = 400,
                detail = "No model loaded. Call POST /inference/load first.",
            )
        model_name = backend.active_model_name or payload.model
        if _wants_multiple_choices(payload):
            _raise_unsupported_n("non-GGUF chat completions")

        # ── Audio TTS path: auto-route to audio generation ────
        # (Whisper is ASR not TTS -- handled below in audio input path)
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
                    max_new_tokens = _effective_max_tokens(payload) or 2048,
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
                            choices = [ChunkChoice(delta = ChoiceDelta(), finish_reason = "stop")],
                        )
                        yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                        yield "data: [DONE]\n\n"
                    except asyncio.CancelledError:
                        cancel_event.set()
                        raise
                    except Exception as e:
                        logger.error(f"Error during audio input streaming: {e}", exc_info = True)
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
    effective_max_tokens = _effective_max_tokens(payload)

    normalized_stop = _normalize_stop_sequences(payload.stop)

    _has_tool_messages = any(m.role == "tool" or m.tool_calls for m in payload.messages)
    # Route guided-decoding requests through the verbatim passthrough so
    # ``response_format`` (JSON schema) reaches llama-server and the model's
    # GBNF-constrained output comes back unmodified. The non-passthrough GGUF
    # path below calls ``generate_chat_completion`` which has no response_format
    # kwarg, so the schema gets silently dropped and data_designer falls back to
    # free-form sampling. Guided decoding does not require ``supports_tools`` --
    # the grammar machinery is independent of tool-call parsing.
    _has_response_format = _extract_response_format(payload) is not None
    _tools_passthrough = llama_backend.supports_tools and (
        (payload.tools and len(payload.tools) > 0) or _has_tool_messages
    )
    if (
        using_gguf
        and not _effective_enable_tools(payload)
        and (_tools_passthrough or _has_response_format)
    ):
        if _wants_multiple_choices(payload):
            _raise_unsupported_n("GGUF tool or response_format passthrough")
        if payload.audio_base64:
            # This path forwards the request verbatim, so the transcoded audio
            # never gets injected. (The agentic tool loop below does support
            # audio.)
            raise HTTPException(
                status_code = 400,
                detail = "Audio input is not supported together with guided decoding or client-supplied tools yet.",
            )

        # Preserve the vision guard from the non-passthrough path below:
        # text-only tool-capable GGUFs should return a clear 400 here rather
        # than forwarding the image to llama-server and surfacing an opaque
        # upstream error.
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
        # `stream` defaults to False on ChatCompletionRequest (OpenAI spec
        # parity). Naive curl / .NET / System.Text.Json clients omitting the
        # field used to get SSE here and choke on deserialization (#5047).
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
    system_prompt, chat_messages, extracted_image_b64 = _extract_content_parts(payload.messages)

    if not chat_messages:
        raise HTTPException(
            status_code = 400,
            detail = "At least one non-system message is required.",
        )

    # ── GGUF path: proxy to llama-server /v1/chat/completions ──
    if using_gguf:
        # Forward uploaded audio as an input_audio part. wav/mp3 pass through
        # untouched (llama-server decodes and resamples them via the mmproj
        # audio encoder); other containers are transcoded to WAV here. The part
        # is injected into the message list below so it rides through both the
        # plain and tool-calling paths, exactly like image_url parts.
        audio_b64 = None
        audio_format = "wav"
        if payload.audio_base64:
            if not getattr(llama_backend, "_has_audio_input", False):
                raise HTTPException(
                    status_code = 400,
                    detail = "Audio provided but current GGUF model does not support audio input.",
                )
            if len(payload.audio_base64) > _MAX_AUDIO_B64_CHARS:
                raise HTTPException(
                    status_code = 413,
                    detail = "Audio file is too large (max ~25 MB).",
                )
            try:
                audio_b64, audio_format = await asyncio.to_thread(
                    _prepare_audio_for_llama, payload.audio_base64
                )
            except Exception as e:
                logger.warning("Audio decode failed: %s", e, exc_info = True)
                raise HTTPException(
                    status_code = 400,
                    detail = "Could not decode the provided audio file.",
                )

        gguf_messages, _ = _openai_messages_for_gguf_chat(
            payload,
            llama_backend.is_vision,
        )
        gguf_messages = _set_or_prepend_system_message(gguf_messages, system_prompt)
        image_b64 = None
        if audio_b64:
            _inject_audio_part(gguf_messages, audio_b64, audio_format)

        cancel_event = threading.Event()

        completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())

        # ── Tool-calling path (agentic loop) ──────────────────
        # `_effective_enable_tools` lets `unsloth run --enable-tools/--disable-tools`
        # hard-override the per-request value, else falls back to
        # `payload.enable_tools`. `mcp_enabled=true` also opens the tool loop so
        # MCP-only callers needn't flip a second flag, BUT must still honor a
        # CLI `--disable-tools` policy -- checking the raw policy here keeps
        # `mcp_enabled` from re-enabling tools the operator explicitly forbade.
        from state.tool_policy import get_tool_policy as _get_tool_policy_g

        _cli_policy = _get_tool_policy_g()
        _tools_on = _effective_enable_tools(payload)
        _mcp_allowed = bool(payload.mcp_enabled) and _cli_policy is not False
        use_tools = (_tools_on or _mcp_allowed) and llama_backend.supports_tools

        if use_tools:
            from core.inference.tools import ALL_TOOLS, get_enabled_mcp_tools

            if not _tools_on:
                # MCP-only request: skip built-ins, leave room for MCP tools.
                tools_to_use = []
            elif payload.enabled_tools is not None:
                tools_to_use = [
                    t for t in ALL_TOOLS if t["function"]["name"] in payload.enabled_tools
                ]
            else:
                tools_to_use = ALL_TOOLS

            # Drop the RAG tool without a scope: nothing to search over.
            if not payload.rag_scope:
                tools_to_use = [
                    t for t in tools_to_use if t["function"]["name"] != "search_knowledge_base"
                ]

            if _mcp_allowed:
                tools_to_use = tools_to_use + await get_enabled_mcp_tools()

            # Skip the tool loop when no tool survived, so the safetensors
            # loop's "empty = allow all" semantic can't reach built-in tools
            # the caller didn't opt into. Callers who omit enabled_tools still
            # get ALL_TOOLS here, so this only suppresses the loop when
            # discovery + opt-in left it genuinely empty.
            if not tools_to_use:
                use_tools = False

        if use_tools:
            if payload.confirm_tool_calls and not payload.stream:
                raise HTTPException(
                    status_code = 400,
                    detail = openai_error_body(
                        "confirm_tool_calls requires stream=true for local tool execution.",
                        status = 400,
                        code = "invalid_request_error",
                        param = "confirm_tool_calls",
                    ),
                )
            if _wants_multiple_choices(payload):
                _raise_unsupported_n("GGUF tool chat completions")
            # ── Tool-use system prompt nudge ──────────────────────
            _nudge = _build_tool_action_nudge(
                tools = tools_to_use,
                model_name = model_name,
            )

            # Nudge the model to ground in attached documents instead of memory.
            _tool_names = {(t.get("function") or {}).get("name") for t in (tools_to_use or [])}
            _rag_active = "search_knowledge_base" in _tool_names and payload.rag_scope
            if _rag_active:
                _rag_nudge = (
                    "The user has attached documents to this conversation. Relevant "
                    "passages are retrieved and provided to you automatically; base "
                    "your answer on them and cite them. You can also call "
                    "search_knowledge_base to look for more. Do not answer from "
                    "memory when the attached documents are relevant."
                )
                # Prefix the date when the tool nudge is empty (RAG-only tool set).
                _date_line = f"The current date is {_date.today().isoformat()}."
                _nudge = _date_line + " " + _rag_nudge if not _nudge else _nudge + " " + _rag_nudge

            if _nudge:
                # Append nudge to system prompt (preserve user's prompt)
                if system_prompt:
                    system_prompt = system_prompt.rstrip() + "\n\n" + _nudge
                else:
                    system_prompt = _nudge
                gguf_messages = _set_or_prepend_system_message(gguf_messages, system_prompt)

            _gguf_auto_heal_tool_calls = (
                payload.auto_heal_tool_calls if payload.auto_heal_tool_calls is not None else True
            )

            # ── Strip stale tool-call XML from conversation history ─
            for _msg in gguf_messages:
                if _msg.get("role") == "assistant" and isinstance(_msg.get("content"), str):
                    _msg["content"] = _strip_tool_xml_for_display(
                        _msg["content"],
                        auto_heal_tool_calls = _gguf_auto_heal_tool_calls,
                    ).strip()

            def gguf_generate_with_tools():
                return llama_backend.generate_chat_completion_with_tools(
                    messages = gguf_messages,
                    tools = tools_to_use,
                    temperature = payload.temperature,
                    top_p = payload.top_p,
                    top_k = payload.top_k,
                    min_p = payload.min_p,
                    max_tokens = effective_max_tokens,
                    repetition_penalty = payload.repetition_penalty,
                    presence_penalty = payload.presence_penalty,
                    stop = normalized_stop,
                    cancel_event = cancel_event,
                    seed = payload.seed,
                    enable_thinking = payload.enable_thinking,
                    reasoning_effort = payload.reasoning_effort,
                    preserve_thinking = payload.preserve_thinking,
                    auto_heal_tool_calls = _gguf_auto_heal_tool_calls,
                    max_tool_iterations = payload.max_tool_calls_per_message
                    if payload.max_tool_calls_per_message is not None
                    else 25,
                    tool_call_timeout = payload.tool_call_timeout
                    if payload.tool_call_timeout is not None
                    else 300,
                    session_id = payload.session_id,
                    rag_scope = payload.rag_scope,
                    disable_parallel_tool_use = payload.parallel_tool_calls is False,
                    confirm_tool_calls = bool(payload.confirm_tool_calls),
                )

            _tool_sentinel = object()

            _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
            _tracker = _TrackedCancel(cancel_event, *_cancel_keys)
            _tracker.__enter__()

            async def gguf_tool_stream():
                gen = None
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

                    # Iterate the sync generator in a thread so the event loop
                    # stays free for disconnect detection.
                    gen = gguf_generate_with_tools()
                    prev_text = ""
                    _stream_usage = None
                    _stream_timings = None
                    _stream_finish = None
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
                            # Empty status marks an iteration boundary in the
                            # GGUF tool loop (e.g. after a re-prompt). Reset the
                            # cumulative cursor so the next assistant turn
                            # streams cleanly.
                            if not event["text"]:
                                prev_text = ""
                            # Emit tool status as a custom SSE event (including
                            # empty ones to clear UI badges)
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
                            _stream_finish = event.get("finish_reason")
                            continue

                        # "content" type -- cumulative text. Sanitize the full
                        # cumulative then diff against the last sanitized
                        # snapshot so cross-chunk XML tags are handled correctly.
                        raw_cumulative = event.get("text", "")
                        clean_cumulative = _strip_tool_xml_for_display(
                            raw_cumulative,
                            auto_heal_tool_calls = _gguf_auto_heal_tool_calls,
                        )
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
                                finish_reason = _clamp_finish_reason(_stream_finish),
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                    usage_line = _openai_stream_usage_chunk(
                        payload,
                        completion_id,
                        created,
                        model_name,
                        _stream_usage,
                        _stream_timings,
                    )
                    if usage_line is not None:
                        yield usage_line
                    yield "data: [DONE]\n\n"

                except asyncio.CancelledError:
                    cancel_event.set()
                    raise
                except Exception as e:
                    import traceback

                    tb = traceback.format_exc()
                    logger.error(f"Error during GGUF tool streaming: {e}\n{tb}")
                    error_chunk = _openai_stream_error_chunk(e)
                    yield f"data: {json.dumps(error_chunk)}\n\n"
                finally:
                    if gen is not None:
                        try:
                            gen.close()
                        except (RuntimeError, ValueError):
                            pass
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

        def gguf_generate(choice_index: int = 0):
            _seed = payload.seed
            if _seed is not None and _seed >= 0 and choice_index:
                _seed += choice_index
            return llama_backend.generate_chat_completion(
                messages = gguf_messages,
                image_b64 = image_b64,
                temperature = payload.temperature,
                top_p = payload.top_p,
                top_k = payload.top_k,
                min_p = payload.min_p,
                max_tokens = effective_max_tokens,
                repetition_penalty = payload.repetition_penalty,
                presence_penalty = payload.presence_penalty,
                stop = normalized_stop,
                cancel_event = cancel_event,
                enable_thinking = payload.enable_thinking,
                reasoning_effort = payload.reasoning_effort,
                preserve_thinking = payload.preserve_thinking,
                seed = _seed,
            )

        _gguf_sentinel = object()

        if payload.stream:
            if _wants_multiple_choices(payload):
                _raise_unsupported_n("streaming GGUF chat completions")
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

                    # Iterate the sync generator in a thread so the event loop
                    # stays free for disconnect detection.
                    gen = gguf_generate()
                    prev_text = ""
                    _stream_usage = None
                    _stream_timings = None
                    _stream_finish = None
                    while True:
                        if cancel_event.is_set():
                            break
                        if await request.is_disconnected():
                            cancel_event.set()
                            return
                        cumulative = await asyncio.to_thread(next, gen, _gguf_sentinel)
                        if cumulative is _gguf_sentinel:
                            break
                        # Capture server metadata for the final usage chunk
                        if isinstance(cumulative, dict):
                            if cumulative.get("type") == "metadata":
                                _stream_usage = cumulative.get("usage")
                                _stream_timings = cumulative.get("timings")
                                _stream_finish = cumulative.get("finish_reason")
                            else:
                                logger.warning(
                                    "gguf_stream_chunks: unexpected dict event: %s",
                                    {k: v for k, v in cumulative.items() if k != "timings"},
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
                                finish_reason = _clamp_finish_reason(_stream_finish),
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json(exclude_none = True)}\n\n"
                    usage_line = _openai_stream_usage_chunk(
                        payload,
                        completion_id,
                        created,
                        model_name,
                        _stream_usage,
                        _stream_timings,
                    )
                    if usage_line is not None:
                        yield usage_line
                    yield "data: [DONE]\n\n"

                except asyncio.CancelledError:
                    cancel_event.set()
                    raise
                except Exception as e:
                    logger.error(f"Error during GGUF streaming: {e}", exc_info = True)
                    error_chunk = _openai_stream_error_chunk(e)
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
                # ``n`` requests several independent completions; the single
                # decode slot yields one at a time, so loop sequentially.
                _n = payload.n or 1

                _choices = []
                _prompt_tokens = 0
                _sum_completion = 0
                _prompt_details = None
                for _idx in range(_n):
                    # Stop spawning the remaining choices once cancelled.
                    if cancel_event.is_set():
                        break
                    full_text = ""
                    completion_usage = None
                    completion_finish = None
                    for token in gguf_generate(_idx):
                        if isinstance(token, dict):
                            if token.get("type") == "metadata":
                                completion_usage = token.get("usage")
                                completion_finish = token.get("finish_reason")
                            continue
                        full_text = token

                    _choices.append(
                        CompletionChoice(
                            index = _idx,
                            message = CompletionMessage(content = full_text),
                            finish_reason = _clamp_finish_reason(completion_finish),
                        )
                    )
                    if completion_usage:
                        # The prompt is shared across all n choices, so count its
                        # tokens ONCE (OpenAI bills only generated tokens for each
                        # extra choice). Only completion_tokens accumulates.
                        _prompt_tokens = completion_usage.get("prompt_tokens") or _prompt_tokens
                        _sum_completion += completion_usage.get("completion_tokens") or 0
                        if _prompt_details is None:
                            _prompt_details = completion_usage.get("prompt_tokens_details")

                response = ChatCompletion(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = _choices,
                    usage = CompletionUsage(
                        prompt_tokens = _prompt_tokens,
                        completion_tokens = _sum_completion,
                        total_tokens = _prompt_tokens + _sum_completion,
                        prompt_tokens_details = _prompt_tokens_details(_prompt_details),
                    ),
                )
                return JSONResponse(content = response.model_dump())

            except Exception as e:
                logger.error(f"Error during GGUF completion: {e}", exc_info = True)
                # An over-context prompt makes llama-server return 400; map any
                # upstream 4xx to a 400 client error rather than leaking a 500.
                _cls = _classify_llama_generation_error(e)
                if _cls is not None:
                    raise HTTPException(
                        status_code = 400,
                        detail = openai_error_body(
                            _friendly_error(e),
                            status = 400,
                            code = "context_length_exceeded" if _cls else None,
                            param = "messages",
                        ),
                    )
                raise HTTPException(status_code = 500, detail = safe_error_detail(e))

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
            raise log_and_http_error(
                e,
                400,
                "Failed to decode image",
                event = "inference.decode_image_failed",
                log = logger,
            )

    # Classify capability flags from the loaded template.
    _sf_model_info = backend.models.get(backend.active_model_name, {})
    _sf_tpl = (_sf_model_info.get("chat_template_info") or {}).get("template")
    _sf_features = _detect_safetensors_features(backend, _sf_tpl)

    cancel_event = threading.Event()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    # ── Safetensors tool-calling path ─────────────────────────
    # Mirrors the GGUF agentic loop's event shape. Disabled for vision turns
    # (untested overlap with image render slot) and for gpt-oss (Harmony uses
    # dedicated channels, not <tool_call> XML -- gpt-oss tools still work via
    # the GGUF path).
    _sf_is_gptoss = False
    try:
        _sf_is_gptoss = bool(hasattr(backend, "_is_gpt_oss_model") and backend._is_gpt_oss_model())
    except Exception:
        _sf_is_gptoss = False

    _sf_tool_budget = (
        payload.max_tool_calls_per_message if payload.max_tool_calls_per_message is not None else 25
    )

    # Match the GGUF path: mcp_enabled also opens the tool loop on its own
    # but must still honor a CLI `--disable-tools` policy.
    from state.tool_policy import get_tool_policy as _get_tool_policy_sf

    _sf_cli_policy = _get_tool_policy_sf()
    _sf_tools_on = _effective_enable_tools(payload)
    _sf_mcp_allowed = bool(payload.mcp_enabled) and _sf_cli_policy is not False
    _sf_use_tools = (
        (_sf_tools_on or _sf_mcp_allowed)
        and _sf_features.get("supports_tools", False)
        and image is None
        and not _sf_is_gptoss
        and _sf_tool_budget > 0
    )

    if _sf_use_tools:
        from core.inference.tools import ALL_TOOLS, get_enabled_mcp_tools

        if not _sf_tools_on:
            _sf_tools_to_use = []
        elif payload.enabled_tools is not None:
            _sf_tools_to_use = [
                t for t in ALL_TOOLS if t["function"]["name"] in payload.enabled_tools
            ]
        else:
            _sf_tools_to_use = ALL_TOOLS

        # Drop the RAG tool unless the request carries a retrieval scope.
        if not payload.rag_scope:
            _sf_tools_to_use = [
                t for t in _sf_tools_to_use if t["function"]["name"] != "search_knowledge_base"
            ]

        if _sf_mcp_allowed:
            _sf_tools_to_use = _sf_tools_to_use + await get_enabled_mcp_tools()

        # Mirror the GGUF path: refuse to enter the tool loop when nothing
        # survived, so a model-emitted built-in call can't piggy-back on the
        # empty allow-list.
        if not _sf_tools_to_use:
            _sf_use_tools = False

    if _sf_use_tools:
        if payload.confirm_tool_calls and not payload.stream:
            raise HTTPException(
                status_code = 400,
                detail = openai_error_body(
                    "confirm_tool_calls requires stream=true for local tool execution.",
                    status = 400,
                    code = "invalid_request_error",
                    param = "confirm_tool_calls",
                ),
            )
        _sf_nudge = _build_tool_action_nudge(
            tools = _sf_tools_to_use,
            model_name = model_name,
        )

        # RAG nudge, mirroring the GGUF path.
        _sf_tool_names = {(t.get("function") or {}).get("name") for t in (_sf_tools_to_use or [])}
        _sf_rag_active = "search_knowledge_base" in _sf_tool_names and payload.rag_scope
        if _sf_rag_active:
            _sf_rag_nudge = (
                "The user has attached documents to this conversation. Relevant "
                "passages are retrieved and provided to you automatically; base "
                "your answer on them and cite them. You can also call "
                "search_knowledge_base to look for more. Do not answer from "
                "memory when the attached documents are relevant."
            )
            # Prefix the date when the tool nudge is empty (RAG-only tool set).
            _sf_date_line = f"The current date is {_date.today().isoformat()}."
            _sf_nudge = (
                _sf_date_line + " " + _sf_rag_nudge
                if not _sf_nudge
                else _sf_nudge + " " + _sf_rag_nudge
            )

        _sf_system_prompt = system_prompt
        if _sf_nudge:
            if _sf_system_prompt:
                _sf_system_prompt = _sf_system_prompt.rstrip() + "\n\n" + _sf_nudge
            else:
                _sf_system_prompt = _sf_nudge

        _sf_auto_heal_tool_calls = (
            payload.auto_heal_tool_calls if payload.auto_heal_tool_calls is not None else True
        )

        # Strip stale tool-call XML from prior assistant turns.
        _sf_chat_messages = []
        for _msg in chat_messages:
            if _msg.get("role") == "assistant" and isinstance(_msg.get("content"), str):
                _sf_chat_messages.append(
                    {
                        **_msg,
                        "content": _strip_tool_xml_for_display(
                            _msg["content"],
                            auto_heal_tool_calls = _sf_auto_heal_tool_calls,
                        ).strip(),
                    }
                )
            else:
                _sf_chat_messages.append(_msg)

        # Request-scoped usage/timings receptacle (filled at gen_done).
        _sf_stats_holder: dict = {}

        def sf_generate_with_tools():
            return backend.generate_chat_completion_with_tools(
                messages = _sf_chat_messages,
                tools = _sf_tools_to_use,
                system_prompt = _sf_system_prompt or "",
                temperature = payload.temperature,
                top_p = payload.top_p,
                top_k = payload.top_k,
                min_p = payload.min_p,
                max_tokens = effective_max_tokens,
                repetition_penalty = payload.repetition_penalty,
                cancel_event = cancel_event,
                enable_thinking = payload.enable_thinking,
                reasoning_effort = payload.reasoning_effort,
                preserve_thinking = payload.preserve_thinking,
                auto_heal_tool_calls = _sf_auto_heal_tool_calls,
                max_tool_iterations = _sf_tool_budget,
                tool_call_timeout = payload.tool_call_timeout
                if payload.tool_call_timeout is not None
                else 300,
                session_id = payload.session_id,
                rag_scope = payload.rag_scope,
                confirm_tool_calls = bool(payload.confirm_tool_calls),
                use_adapter = payload.use_adapter,
                stats_holder = _sf_stats_holder,
            )

        _sf_tool_sentinel = object()
        _sf_cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
        _sf_tracker = _TrackedCancel(cancel_event, *_sf_cancel_keys)
        _sf_tracker.__enter__()

        async def sf_tool_stream():
            gen = None
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

                gen = sf_generate_with_tools()
                prev_text = ""
                while True:
                    if cancel_event.is_set():
                        backend.reset_generation_state()
                        break
                    if await request.is_disconnected():
                        cancel_event.set()
                        backend.reset_generation_state()
                        return

                    event = await asyncio.to_thread(next, gen, _sf_tool_sentinel)
                    if event is _sf_tool_sentinel:
                        break

                    if event["type"] == "status":
                        if not event["text"]:
                            prev_text = ""
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

                    # Diff cumulative cleaned text against last snapshot.
                    raw_cumulative = event.get("text", "")
                    clean_cumulative = _strip_tool_xml_for_display(
                        raw_cumulative,
                        auto_heal_tool_calls = _sf_auto_heal_tool_calls,
                    )
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
                # Usage chunk from the last turn, same shape as the
                # GGUF tool loop's metadata. Request-scoped holder, so
                # concurrent streams cannot read each other's stats.
                _stats = _sf_stats_holder.get("stats")
                if _stats:
                    usage_line = _openai_stream_usage_chunk(
                        payload,
                        completion_id,
                        created,
                        model_name,
                        _stats.get("usage"),
                        _stats.get("timings"),
                    )
                    if usage_line is not None:
                        yield usage_line
                yield "data: [DONE]\n\n"

            except asyncio.CancelledError:
                cancel_event.set()
                backend.reset_generation_state()
                raise
            except Exception:
                backend.reset_generation_state()
                # Generic wire message; full trace stays in the log (CWE-209:
                # transformers/torch errors may leak paths).
                logger.exception("safetensors tool stream error")
                error_chunk = {
                    "error": {
                        "message": "An internal error occurred.",
                        "type": "server_error",
                    },
                }
                yield f"data: {json.dumps(error_chunk)}\n\n"
            finally:
                if gen is not None:
                    try:
                        gen.close()
                    except (RuntimeError, ValueError):
                        pass
                _sf_tracker.__exit__(None, None, None)

        if payload.stream:
            return StreamingResponse(
                sf_tool_stream(),
                media_type = "text/event-stream",
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming JSON: drain the loop, build one ChatCompletion.
        try:

            def _drain_to_text():
                full_text = ""
                gen = sf_generate_with_tools()
                for event in gen:
                    if cancel_event.is_set():
                        break
                    if event.get("type") == "content":
                        full_text = _strip_tool_xml_for_display(
                            event.get("text", ""),
                            auto_heal_tool_calls = _sf_auto_heal_tool_calls,
                        )
                return full_text

            content_text = await asyncio.to_thread(_drain_to_text)
            response = ChatCompletion(
                id = completion_id,
                created = created,
                model = model_name,
                choices = [
                    CompletionChoice(
                        message = CompletionMessage(content = content_text),
                        finish_reason = "stop",
                    )
                ],
            )
            return JSONResponse(content = response.model_dump())
        except Exception:
            backend.reset_generation_state()
            # CWE-209: generic detail; full trace in log.
            logger.exception("safetensors tool completion error")
            raise HTTPException(
                status_code = 500,
                detail = "An internal error occurred.",
            )
        finally:
            _sf_tracker.__exit__(None, None, None)

    # Shared generation kwargs
    gen_kwargs = dict(
        messages = chat_messages,
        system_prompt = system_prompt,
        image = image,
        temperature = payload.temperature,
        top_p = payload.top_p,
        top_k = payload.top_k,
        min_p = payload.min_p,
        max_new_tokens = effective_max_tokens or 2048,
        repetition_penalty = payload.repetition_penalty,
    )
    # Forward reasoning kwargs; the worker/template wrapper peels off any the
    # template doesn't accept.
    if payload.enable_thinking is not None:
        gen_kwargs["enable_thinking"] = payload.enable_thinking
    if payload.reasoning_effort is not None:
        gen_kwargs["reasoning_effort"] = payload.reasoning_effort
    if payload.preserve_thinking is not None:
        gen_kwargs["preserve_thinking"] = payload.preserve_thinking

    # Request-scoped usage/timings receptacle (filled at gen_done).
    stats_holder: dict = {}

    if payload.use_adapter is not None:

        def generate():
            return backend.generate_with_adapter_control(
                use_adapter = payload.use_adapter,
                cancel_event = cancel_event,
                stats_holder = stats_holder,
                **gen_kwargs,
            )
    else:

        def generate():
            return backend.generate_chat_response(
                cancel_event = cancel_event,
                stats_holder = stats_holder,
                **gen_kwargs,
            )

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
                # Run the sync generator in a thread pool to avoid blocking the
                # event loop. Critical for compare mode: two SSE requests arrive
                # concurrently but the orchestrator serializes them via
                # _gen_lock; without run_in_executor the second request's
                # blocking lock acquisition would freeze the entire event loop,
                # stalling both streams.
                _DONE = object()  # sentinel for generator exhaustion
                loop = asyncio.get_event_loop()
                gen = generate()
                while True:
                    if cancel_event.is_set():
                        backend.reset_generation_state()
                        break
                    # next(gen, _DONE) returns _DONE instead of raising
                    # StopIteration -- StopIteration can't propagate through
                    # asyncio futures (Python limitation).
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
                # Usage chunk (choices=[], usage set), same shape as the
                # GGUF path so the speed popover works for MLX too.
                # Request-scoped holder, so concurrent streams cannot
                # read each other's stats.
                _stats = stats_holder.get("stats")
                if _stats:
                    usage_line = _openai_stream_usage_chunk(
                        payload,
                        completion_id,
                        created,
                        model_name,
                        _stats.get("usage"),
                        _stats.get("timings"),
                    )
                    if usage_line is not None:
                        yield usage_line
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
            raise HTTPException(status_code = 500, detail = safe_error_detail(e))


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

    Accepts auth via Authorization header OR ?token= query param (needed
    because <img src> cannot send custom headers).
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
    # Defense-in-depth allowlist (clears CodeQL py/path-injection), still allowing
    # names like "loss curve.png"; basename + extension + realpath below are the guards.
    if not _re.fullmatch(r"[^/\\\x00-\x1f]{1,255}", safe_filename):
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
    from core.inference.tools import get_sandbox_workdir

    sandbox_dir = os.path.realpath(get_sandbox_workdir(session_id))
    file_path = os.path.realpath(os.path.join(sandbox_dir, safe_filename))
    if file_path != sandbox_dir and not file_path.startswith(sandbox_dir + os.sep):
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


def _openai_model_objects() -> list[dict]:
    """The model objects GET /v1/models exposes (one per loaded local backend).

    Shared by the LIST and RETRIEVE handlers so both report the same ids and
    field shape.
    """
    models: list[dict] = []
    _created = int(time.time())

    # Check GGUF backend
    llama_backend = get_llama_cpp_backend()
    if llama_backend.is_loaded:
        entry = {
            "id": llama_backend.model_identifier,
            "object": "model",
            "created": _created,
            "owned_by": "local",
        }
        _ctx = _positive_int_or_none(getattr(llama_backend, "context_length", None))
        if _ctx is not None:
            entry["context_length"] = _ctx
        _max_ctx = _positive_int_or_none(getattr(llama_backend, "max_context_length", None))
        if _max_ctx is not None:
            entry["max_context_length"] = _max_ctx
        _native_ctx = _positive_int_or_none(getattr(llama_backend, "native_context_length", None))
        if _native_ctx is not None:
            entry["native_context_length"] = _native_ctx
        models.append(entry)

    # Check Unsloth backend
    backend = get_inference_backend()
    if backend.active_model_name:
        model_info = backend.models.get(backend.active_model_name, {})
        entry = {
            "id": backend.active_model_name,
            "object": "model",
            "created": _created,
            "owned_by": "local",
        }
        _ctx = _positive_int_or_none(model_info.get("context_length"))
        if _ctx is None:
            for _candidate in (
                getattr(backend, "context_length", None),
                getattr(backend, "max_seq_length", None),
            ):
                _ctx = _positive_int_or_none(_candidate)
                if _ctx is not None:
                    break
        if _ctx is not None:
            entry["context_length"] = _ctx
        models.append(entry)

    return models


@router.get("/models")
async def openai_list_models(current_subject: str = Depends(get_current_subject)):
    """
    OpenAI-compatible model listing endpoint.

    Returns the currently loaded model in the format expected by
    OpenAI-compatible clients (``GET /v1/models``).
    """
    return {"object": "list", "data": _openai_model_objects()}


@router.get("/models/{model_id:path}")
async def openai_retrieve_model(model_id: str, current_subject: str = Depends(get_current_subject)):
    """
    OpenAI-compatible single-model retrieval endpoint (``GET /v1/models/{id}``).

    Returns the bare model object when ``model_id`` matches a loaded local
    model, or 404 model_not_found otherwise. Defined after the LIST route so
    it does not shadow it; ``{model_id:path}`` keeps ids with slashes intact.
    """
    for model in _openai_model_objects():
        if model["id"] == model_id:
            return model
    raise HTTPException(
        status_code = 404,
        detail = openai_error_body(
            f"The model '{model_id}' does not exist",
            status = 404,
            code = "model_not_found",
            param = "id",
        ),
    )


# =====================================================================
# OpenAI-Compatible Completions Proxy  (/completions → /v1/completions)
# =====================================================================


@router.post("/completions")
async def openai_completions(request: Request, current_subject: str = Depends(get_current_subject)):
    """
    OpenAI-compatible text completions endpoint (non-chat).

    Proxies to the running llama-server's ``/v1/completions``. Only available
    when a GGUF model is loaded.
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
            # Manual httpx client/response lifecycle AND explicit iterator
            # close — see _anthropic_passthrough_stream for the full rationale.
            # Saving the iterator and closing it in the finally block avoids the
            # Python 3.13 + httpcore 1.0.x "Exception ignored in:
            # <async_generator>" / anyio cancel-scope trace.
            #
            # Buffer the relay into whole SSE events (split on the blank-line
            # separator) so _cmpl_stream_event_out can rewrite the cmpl- id and
            # honor stream_options.include_usage per event, while keeping SSE
            # framing and token bytes intact.
            _include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
            client = httpx.AsyncClient(timeout = 600)
            resp = None
            bytes_iter = None
            try:
                req = client.build_request("POST", target_url, json = body)
                resp = await client.send(req, stream = True)
                bytes_iter = resp.aiter_bytes()
                buffer = b""
                async for chunk in bytes_iter:
                    buffer += chunk
                    while b"\n\n" in buffer:
                        event, buffer = buffer.split(b"\n\n", 1)
                        out = _cmpl_stream_event_out(event, _include_usage)
                        if out is not None:
                            yield out + b"\n\n"
                if buffer:
                    out = _cmpl_stream_event_out(buffer, _include_usage)
                    if out is not None:
                        # Re-add the SSE separator the split consumed, so a final
                        # event arriving without a trailing blank line is still
                        # terminated for the client's parser.
                        yield out + b"\n\n"
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

        if resp.status_code != 200:
            raise _openai_passthrough_error(resp.status_code, resp.text)

        return Response(
            content = _rewrite_cmpl_id(resp.content),
            status_code = resp.status_code,
            media_type = "application/json",
        )


# =====================================================================
# OpenAI-Compatible Embeddings Proxy  (/embeddings → /v1/embeddings)
# =====================================================================


@router.post("/embeddings")
async def openai_embeddings(request: Request, current_subject: str = Depends(get_current_subject)):
    """
    OpenAI-compatible embeddings endpoint.

    Proxies to the running llama-server's ``/v1/embeddings``. Only available
    when a GGUF model is loaded.
    Note: the loaded model must support pooling, else llama-server returns an
    error (expected).
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


def _translate_responses_tools_to_chat(tools: Optional[list[dict]]) -> Optional[list[dict]]:
    """Translate Responses-shape function tools to the Chat Completions nested shape.

    Responses uses a flat shape per tool entry::

        {"type": "function", "name": "...", "description": "...",
         "parameters": {...}, "strict": true}

    The Chat Completions / llama-server passthrough expects the nested shape::

        {"type": "function",
         "function": {"name": "...", "description": "...",
                      "parameters": {...}, "strict": true}}

    Only ``type=="function"`` entries are forwarded. Built-in Responses tools
    (``web_search``, ``file_search``, ``mcp``, ...) are dropped: llama-server
    doesn't implement them server-side, so keeping them would produce an opaque
    upstream 400.
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
    The Responses forcing object ``{"type": "function", "name": "X"}`` becomes
    Chat Completions' ``{"type": "function", "function": {"name": "X"}}``.
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


def _responses_tool_output_text(output: Union[str, list]) -> str:
    """Return Chat Completions-safe content for a Responses tool result."""
    if isinstance(output, str):
        return output if output.strip() else "(no output)"

    if output:
        return json.dumps(output)

    return "(no output)"


def _normalise_responses_input(payload: ResponsesRequest) -> list[ChatMessage]:
    """Convert a ResponsesRequest's ``input`` into a Chat-format ``ChatMessage`` list.

    Handles the three input item shapes allowed by the Responses API:

    - ``ResponsesInputMessage`` -- regular chat messages (text or multimodal).
    - ``ResponsesFunctionCallInputItem`` -- a prior assistant tool call
      replayed on a follow-up turn. Becomes an assistant message carrying a
      Chat Completions ``tool_calls`` entry keyed by ``call_id``.
    - ``ResponsesFunctionCallOutputInputItem`` -- a tool result the client is
      returning. Becomes a ``role="tool"`` message with ``tool_call_id`` set to
      the originating ``call_id`` so llama-server can reconcile call with result.

    System / developer content is collected from ``instructions`` *and* any
    ``role="system"`` / ``role="developer"`` entries in ``input``, then merged
    into a single top-of-list ``role="system"`` message. This satisfies strict
    chat templates (harmony / gpt-oss, Qwen3, ...) whose Jinja raises
    ``"System message must be at the beginning."`` when more than one system
    message is present or a system message follows a user turn -- the exact
    pattern the OpenAI Codex CLI hits, since Codex sets ``instructions`` *and*
    also sends a developer message in ``input``.
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
            # Chat Completions `role="tool"` requires string content; serialize
            # a Responses content-array output and keep empty outputs from
            # tripping the stricter ChatMessage role validator.
            output = _responses_tool_output_text(item.output)
            messages.append(
                ChatMessage(
                    role = "tool",
                    tool_call_id = item.call_id,
                    content = output,
                )
            )
            continue

        if isinstance(item, ResponsesUnknownInputItem):
            # Reasoning items and other unmodelled top-level Responses item
            # types are silently dropped -- llama-server-backed GGUFs can't
            # consume them; lenient validation lets them in so unrelated turns
            # don't 422.
            continue

        # ResponsesInputMessage -- hoist system/developer to the top, merge.
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
        # multimodal array, so flatten output_text (and any stray input_text /
        # unknown text) to a single string.
        if item.role == "assistant":
            text = _responses_message_text(item.content)
            if text:
                messages.append(ChatMessage(role = "assistant", content = text))
            continue

        # User (and any other remaining roles) -- keep multimodal when present,
        # drop unknown content parts silently.
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
            # Collapse single-text-part content to a plain string so roles that
            # reject multimodal arrays (e.g. legacy templates) still accept it.
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
    ``/v1/chat/completions`` client-side pass-through picks them up unchanged.
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
    if payload.parallel_tool_calls is not None:
        chat_kwargs["parallel_tool_calls"] = payload.parallel_tool_calls

    return ChatCompletionRequest(**chat_kwargs)


def _chat_tool_calls_to_responses_output(tool_calls: list[dict]) -> list[dict]:
    """Map Chat Completions ``tool_calls`` into Responses ``function_call`` output items.

    The Chat Completions id (``call_xxx``) is the shared correlation key across
    turns in the Responses API -- stored as ``call_id`` on the output item and
    echoed back by the client as ``function_call_output.call_id`` next turn.
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
    payload: ResponsesRequest, messages: list[ChatMessage], request: Request
) -> JSONResponse:
    """Handle a non-streaming Responses API call."""
    chat_req = _build_chat_request(payload, messages, stream = False)
    result = await openai_chat_completions(chat_req, request)

    # openai_chat_completions returns a JSONResponse for non-streaming.
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
    # plus an optional assistant text message. Emit the text message only when
    # the model produced content, so clients expecting a pure tool-call turn
    # (finish_reason="tool_calls") don't see a spurious empty message item.
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
    payload: ResponsesRequest, messages: list[ChatMessage], request: Request
):
    """Handle a streaming Responses API call, emitting named SSE events.

    For GGUF models the request goes directly to llama-server's
    ``/v1/chat/completions`` from inside the StreamingResponse child task -- one
    httpx lifecycle, one async generator. Wrapping the existing
    ``openai_chat_completions`` pass-through (which has its own httpx lifecycle)
    stacks two generators: Python 3.13 + httpcore 1.0.x then loses the
    close-propagation chain on the innermost ``HTTP11ConnectionByteStream`` at
    asyncgen finalisation, tripping "Attempted to exit cancel scope in a
    different task" / "async generator ignored GeneratorExit". The direct path
    avoids that. Non-GGUF falls back to the wrapper (which doesn't use httpx, so
    the issue doesn't apply).

    Text deltas arrive as ``response.output_text.delta`` on a single
    ``message`` output item at ``output_index=0``. Each tool call from
    ``delta.tool_calls[]`` is promoted to its own top-level ``function_call``
    output item (one per distinct ``tool_calls[].index``) and relayed as
    ``response.function_call_arguments.delta`` / ``.done`` events so clients
    (Codex, OpenAI Python SDK) can reconstruct the call incrementally and reply
    with a ``function_call_output`` item next turn.
    """
    resp_id = f"resp_{uuid.uuid4().hex[:12]}"
    msg_id = f"msg_{uuid.uuid4().hex[:12]}"
    created_at = int(time.time())

    chat_req = _build_chat_request(payload, messages, stream = True)

    llama_backend = get_llama_cpp_backend()
    if not llama_backend.is_loaded:
        # The direct pass-through is GGUF-only. Non-GGUF /v1/responses streaming
        # isn't a Codex-compatible path today, and wrapping the transformers
        # backend's streaming generator here would re-introduce the
        # double-layer asyncgen close pattern that produces "Attempted to exit
        # cancel scope in a different task" on Python 3.13. Surface a typed 400
        # so the client sees a useful error instead of a dangling stream.
        raise HTTPException(
            status_code = 400,
            detail = (
                "Streaming /v1/responses requires a GGUF model loaded via "
                "llama-server. Use non-streaming /v1/responses, "
                "/v1/chat/completions, or load a GGUF model."
            ),
        )

    # Direct pass-through bypasses the openai_chat_completions image gate.
    if not llama_backend.is_vision and any(
        isinstance(m.content, list) and any(isinstance(p, ImageContentPart) for p in m.content)
        for m in messages
    ):
        raise HTTPException(
            status_code = 400,
            detail = "Image provided but current GGUF model does not support vision.",
        )

    body = _build_openai_passthrough_body(
        chat_req, backend_ctx = llama_backend.context_length, llama_backend = llama_backend
    )
    body["stream_options"] = {"include_usage": True}
    target_url = f"{llama_backend.base_url}/v1/chat/completions"

    async def event_generator():
        full_text = ""
        input_tokens = 0
        output_tokens = 0
        # Per-tool-call state keyed by Chat Completions `tool_calls[].index`,
        # stable across chunks for the same call. Values:
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
        # Full same-task open + close, same pattern as
        # _openai_passthrough_stream and _anthropic_passthrough_stream: no
        # `async with`, explicit aclose of lines_iter BEFORE resp / client so
        # the innermost httpcore byte stream is finalised in this task (not via
        # the asyncgen GC in a sibling task).
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
                if payload.parallel_tool_calls is False:
                    _drop_parallel_tool_call_deltas(chunk_data)

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
                        # First chunk for this tool call -- allocate an
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
                        # Later chunks sometimes carry id/name only once; merge
                        # when present.
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
                        # Buffer args until we can open the item (some models
                        # send id/name in the same chunk as the first arg delta;
                        # if not, stash).
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
            # If id/name never arrived (malformed upstream), synthesise so the
            # client still sees a coherent frame sequence.
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

    Accepts a Responses-format request, converts it to a ChatCompletionRequest
    internally, and returns a response matching the Responses API schema
    (output array, input_tokens/output_tokens, named SSE events for streaming).
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


_STUDIO_ANTHROPIC_TOOL_ALIASES = {
    "web_search": "web_search",
    "web_search_20250305": "web_search",
    "web_fetch": "web_search",
    "web_fetch_20250910": "web_search",
    "web_fetch_20260209": "web_search",
    "python": "python",
    "terminal": "terminal",
}


def _anthropic_requested_studio_tools(tools: Optional[list]) -> set[str]:
    requested: set[str] = set()
    for tool in tools or []:
        td = tool if isinstance(tool, dict) else tool.model_dump()
        # Client tools always carry input_schema; server tools never do.
        if td.get("input_schema") is not None:
            continue
        # Anthropic dispatches server tools by `type`, not bare `name`; matching
        # name too would let a malformed client tool like `{"name": "python"}`
        # silently flip into server-execution mode.
        type_ = td.get("type")
        if isinstance(type_, str) and type_ in _STUDIO_ANTHROPIC_TOOL_ALIASES:
            requested.add(_STUDIO_ANTHROPIC_TOOL_ALIASES[type_])
    return requested


def _select_anthropic_server_tools(
    all_tools: list[dict], requested_studio_tools: set[str], enabled_tools: Optional[list[str]]
) -> list[dict]:
    """Select Studio tools requested through Anthropic tools and extensions."""
    if not requested_studio_tools and enabled_tools is None:
        return all_tools

    selected_names = set(requested_studio_tools)
    if enabled_tools is not None:
        selected_names.update(enabled_tools)

    return [tool for tool in all_tools if tool["function"]["name"] in selected_names]


def _normalize_anthropic_openai_images(openai_messages: list[dict], is_vision: bool) -> bool:
    """Enforce the vision guard on translated Anthropic messages and normalize
    any base64-data-URL ``image_url`` parts to PNG.

    llama-server's stb_image only handles a few formats (JPEG/PNG/BMP/…);
    Anthropic clients commonly send JPEG or WebP, and Claude Code sends WebP.
    Re-encoding everything to PNG mirrors `_openai_messages_for_passthrough` /
    the GGUF branch of `/v1/chat/completions` so the two endpoints agree.

    Mutates ``openai_messages`` in place. Returns ``True`` when any image part
    was seen (so the caller can skip a second scan). Raises HTTPException(400)
    when images are present but the active model isn't a vision model, or when
    an image cannot be decoded.
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


@router.post("/messages/count_tokens")
async def anthropic_count_tokens(
    payload: AnthropicMessagesRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """Anthropic-compatible token-counting endpoint (POST /v1/messages/count_tokens).

    Translates the Anthropic request to OpenAI form (the same translation the
    /messages handler uses), counts prompt tokens with the loaded GGUF model's
    tokenizer, and returns ``{"input_tokens": int}`` only. Unlike /messages,
    max_tokens is NOT required here.
    """
    llama_backend = get_llama_cpp_backend()
    if not llama_backend.is_loaded:
        raise HTTPException(
            status_code = 503,
            detail = "No GGUF model loaded. Load a GGUF model first.",
        )

    # Same Anthropic → OpenAI translation as anthropic_messages: system is
    # folded into the messages list, so pass system=None to the counter.
    openai_messages = anthropic_messages_to_openai(
        [m.model_dump() for m in payload.messages],
        payload.system,
    )
    # Apply the same sanitization /messages does before generation, so the count
    # matches the prompt the real request would build (otherwise empty-assistant
    # sentinels / synthetic tool history inflate the count or hit the fallback).
    openai_messages = _strip_provider_synthetic_tool_history(
        _drop_empty_assistant_sentinels(openai_messages)
    )
    openai_tools = anthropic_tools_to_openai(payload.tools or []) or None

    try:
        count = await asyncio.to_thread(
            llama_backend.count_chat_tokens,
            openai_messages,
            None,
            openai_tools,
            strict = True,
        )
    except Exception:
        raise HTTPException(
            status_code = 503,
            detail = "Unable to count tokens with the loaded model tokenizer.",
        )
    return JSONResponse(content = {"input_tokens": int(count)})


def _set_or_prepend_system_message(
    messages: Optional[list[dict]], system_prompt: str
) -> list[dict]:
    """Return messages with a single leading system prompt, preserving multimodal parts."""
    safe_messages = messages or []
    if not system_prompt:
        return safe_messages

    # Drop existing system/developer turns so the backend never sees duplicate
    # or conflicting system instructions, then prepend the resolved prompt.
    others = [dict(msg) for msg in safe_messages if msg.get("role") not in ("system", "developer")]
    return [{"role": "system", "content": system_prompt}, *others]


@router.post("/messages")
async def anthropic_messages(
    payload: AnthropicMessagesRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    Anthropic-compatible Messages API endpoint.

    Translates Anthropic message format to internal OpenAI format, runs through
    the existing agentic tool loop when tools are provided, and returns
    responses in Anthropic Messages API format (streaming SSE or non-streaming
    JSON).
    """
    llama_backend = get_llama_cpp_backend()
    if not llama_backend.is_loaded:
        raise HTTPException(
            status_code = 503,
            detail = "No GGUF model loaded. Load a GGUF model first.",
        )

    # max_tokens is a required field on the Anthropic Messages API; real
    # Anthropic returns a 400 invalid_request_error when it is omitted.
    if payload.max_tokens is None:
        raise HTTPException(
            status_code = 400,
            detail = anthropic_error_body(
                "max_tokens: field required",
                status = 400,
                err_type = "invalid_request_error",
            ),
        )

    model_name = getattr(llama_backend, "model_identifier", None) or payload.model
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # ── Translate Anthropic → OpenAI ──────────────────────────
    openai_messages = anthropic_messages_to_openai(
        [m.model_dump() for m in payload.messages],
        payload.system,
    )
    # Strip synthetic provider-side builtin tool history (web_search,
    # web_fetch, code_execution, image_generation cards tagged with
    # _server_tool or extra_content.google.native_part) before handing off to
    # local llama-server. The local /v1/chat/completions and GGUF passthrough
    # builders apply the same strip; without it an Anthropic /v1/messages caller
    # replaying a prior provider-side tool_use forwards fake builtin tool
    # history to a backend with no matching function declarations.
    openai_messages = _strip_provider_synthetic_tool_history(
        _drop_empty_assistant_sentinels(openai_messages)
    )

    # Enforce vision guard + re-encode embedded images to PNG so the Anthropic
    # endpoint matches /v1/chat/completions.
    _has_image = _normalize_anthropic_openai_images(openai_messages, llama_backend.is_vision)

    temperature = payload.temperature if payload.temperature is not None else 0.6
    top_p = payload.top_p if payload.top_p is not None else 0.95
    top_k = payload.top_k if payload.top_k is not None else 20
    min_p = payload.min_p if payload.min_p is not None else 0.01
    repetition_penalty = (
        payload.repetition_penalty if payload.repetition_penalty is not None else 1.0
    )
    presence_penalty = payload.presence_penalty if payload.presence_penalty is not None else 0.0
    stop = payload.stop_sequences or None

    # Translate Anthropic tool_choice to OpenAI format for llama-server. Falls
    # back to "auto" when unset or unrecognized (prior hardcoded behavior).
    openai_tool_choice = anthropic_tool_choice_to_openai(payload.tool_choice)
    if openai_tool_choice is None:
        openai_tool_choice = "auto"

    cancel_event = threading.Event()

    # ── Tool routing ──────────────────────────────────────────
    # Three paths:
    # 1. enable_tools=true → server-side execution of built-in tools (Unsloth shorthand)
    # 2. tools=[...] only  → client-side pass-through (standard Anthropic behavior)
    # 3. neither           → plain chat
    # The server-side agentic loop doesn't support multimodal input -- matches
    # the `not image_b64` gate in /v1/chat/completions.
    requested_studio_tools = _anthropic_requested_studio_tools(payload.tools)

    # Reject malformed client tools at the boundary. AnthropicTool was relaxed
    # to Optional[name]/Optional[input_schema] for server tools, so the
    # converter silently drops incomplete entries -- surface them as 400. A
    # `type` field marks a server-tool declaration per spec (unrecognized server
    # tools are accepted as no-ops); anything else without input_schema or name
    # is malformed and must not be allowed to silently flip execution mode or
    # disable tool calling.
    for tool in payload.tools or []:
        td = tool if isinstance(tool, dict) else tool.model_dump()
        name, type_, schema = td.get("name"), td.get("type"), td.get("input_schema")
        if schema is None and not isinstance(type_, str):
            raise HTTPException(
                status_code = 400,
                detail = f"Tool {name!r} is missing required field 'input_schema'.",
            )
        if schema is not None and (not isinstance(name, str) or not name):
            raise HTTPException(
                status_code = 400,
                detail = "Client tool is missing required field 'name'.",
            )

    # Detect client tools from the raw payload (presence of input_schema) so the
    # mixed-mode check below isn't fooled by a name collision with a server-tool
    # alias that the post-filter would silently drop.
    _has_client_tool = any(
        (t if isinstance(t, dict) else t.model_dump()).get("input_schema") is not None
        for t in payload.tools or []
    )

    # The server-tool agentic loop executes tools in-process and can't relay
    # unknown client functions back to the caller, so mixed requests would
    # silently drop the client tools. Reject explicitly instead.
    if requested_studio_tools and _has_client_tool:
        raise HTTPException(
            status_code = 400,
            detail = (
                "Mixing Anthropic server tools (e.g. web_search_20250305) "
                "with custom client tools in a single request is not "
                "supported. Send them in separate requests."
            ),
        )

    openai_client_tools = [
        tool
        for tool in anthropic_tools_to_openai(payload.tools or [])
        if tool.get("function", {}).get("name") not in requested_studio_tools
    ]

    # An Anthropic server-tool declaration implies server-tool mode, but only
    # when tools aren't explicitly disabled (CLI --disable-tools or per-request
    # enable_tools=false). Explicit False always wins.
    _enable = _effective_enable_tools(payload)
    server_tools = (
        (_enable or (_enable is None and bool(requested_studio_tools)))
        and llama_backend.supports_tools
        and not _has_image
    )
    client_tools = (
        not server_tools and len(openai_client_tools) > 0 and llama_backend.supports_tools
    )

    # Anthropic tool_choice.disable_parallel_tool_use caps the response to a
    # single tool_use block. Computed here so BOTH the client-tool passthrough
    # and the server-tool path honor it.
    _disable_parallel = bool(
        isinstance(payload.tool_choice, dict)
        and payload.tool_choice.get("disable_parallel_tool_use")
    )

    # ── Client-side pass-through path ─────────────────────────
    if client_tools:
        openai_tools = openai_client_tools

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
                disable_parallel_tool_use = _disable_parallel,
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
            disable_parallel_tool_use = _disable_parallel,
        )

    if server_tools:
        if bool(getattr(payload, "confirm_tool_calls", False)):
            raise HTTPException(
                status_code = 400,
                detail = anthropic_error_body(
                    "confirm_tool_calls is not supported for Anthropic Messages server tools.",
                    status = 400,
                    err_type = "invalid_request_error",
                ),
            )
        from core.inference.tools import ALL_TOOLS

        openai_tools = _select_anthropic_server_tools(
            ALL_TOOLS,
            requested_studio_tools,
            payload.enabled_tools,
        )

        # Build tool-use system prompt nudge (same logic as /chat/completions)
        _nudge = _build_tool_action_nudge(
            tools = openai_tools,
            model_name = model_name,
        )

        if _nudge:
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
                # Anthropic passthrough has no rag_scope field (RAG is local-only).
                rag_scope = getattr(payload, "rag_scope", None),
                disable_parallel_tool_use = _disable_parallel,
            )

        if payload.stream:
            return await _anthropic_tool_stream(
                request,
                cancel_event,
                _run_tool_gen,
                message_id,
                model_name,
                llama_backend = llama_backend,
                openai_messages = openai_messages,
                openai_tools = openai_tools,
                disable_parallel_tool_use = _disable_parallel,
            )
        return await _anthropic_tool_non_streaming(
            _run_tool_gen,
            message_id,
            model_name,
            disable_parallel_tool_use = _disable_parallel,
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
            llama_backend = llama_backend,
            openai_messages = openai_messages,
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
    llama_backend = None,
    openai_messages = None,
    openai_tools = None,
    disable_parallel_tool_use = False,
):
    """Streaming response for the tool-calling path."""
    _sentinel = object()

    # Prompt-token count for message_start.usage.input_tokens. count_chat_tokens
    # makes blocking HTTP calls to llama-server, so run it off the event loop.
    # Pass the tools so tool-schema tokens are counted (the generator renders
    # them too), matching the non-stream / count_tokens / passthrough paths.
    input_tokens = 0
    if llama_backend is not None and openai_messages is not None:
        input_tokens = await asyncio.to_thread(
            llama_backend.count_chat_tokens, openai_messages, None, openai_tools
        )

    async def _stream():
        emitter = AnthropicStreamEmitter()
        for line in emitter.start(message_id, model_name, input_tokens = input_tokens):
            yield line

        captured_finish_reason = None
        # Whether the response currently ends on a pending tool_use block (the
        # client must act → stop_reason "tool_use") as opposed to final text.
        # The server may run a tool and then keep generating, which flips this
        # back to False — that is an end_turn (or max_tokens) response.
        ends_on_tool_use = False
        tool_blocks_emitted = 0
        drop_until_tool_end = False

        gen = run_gen()
        try:
            while True:
                if await request.is_disconnected():
                    cancel_event.set()
                    return
                event = await asyncio.to_thread(next, gen, _sentinel)
                if event is _sentinel:
                    break
                etype = event.get("type")
                if drop_until_tool_end:
                    # disable_parallel_tool_use: a later tool call is being
                    # dropped — skip every event until (and including) its tool_end.
                    if etype == "tool_end":
                        drop_until_tool_end = False
                    continue
                if etype == "metadata":
                    _fr = event.get("finish_reason")
                    if _fr is not None:
                        captured_finish_reason = _fr
                # Strip leaked tool-call XML from content events first, so a
                # content event that was purely tool XML doesn't count as text.
                if etype == "content":
                    event = dict(event)
                    event["text"] = _TOOL_XML_RE.sub("", event["text"])
                # disable_parallel_tool_use: keep only the first tool_use block,
                # dropping every later tool_start and its paired tool_end (robust
                # to empty tool-call ids — tracked by state, not id matching).
                if etype == "tool_start":
                    if disable_parallel_tool_use and tool_blocks_emitted >= 1:
                        drop_until_tool_end = True
                        continue
                    ends_on_tool_use = True
                elif etype == "tool_end":
                    tool_blocks_emitted += 1
                    # A tool_end means Studio executed the tool server-side, so
                    # the response no longer ends on a pending client action.
                    # Without this, a server tool that produces no trailing text
                    # would be mislabeled stop_reason "tool_use", telling the
                    # client to run a tool Studio already ran.
                    ends_on_tool_use = False
                elif etype == "content" and event.get("text"):
                    ends_on_tool_use = False
                for line in emitter.feed(event):
                    yield line
        except Exception as e:
            logger.error("anthropic_messages stream error: %s", e)
            _error_event = _anthropic_stream_error_event(e)
            if _error_event is not None:
                yield _error_event
                return

        stop_reason = openai_finish_to_anthropic_stop(
            captured_finish_reason, had_tool_calls = ends_on_tool_use
        )
        for line in emitter.finish(stop_reason = stop_reason, stop_sequence = None):
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
    llama_backend = None,
    openai_messages = None,
):
    """Streaming response for the no-tool path."""
    _sentinel = object()

    # Prompt-token count for message_start.usage.input_tokens. count_chat_tokens
    # makes blocking HTTP calls to llama-server, so run it off the event loop.
    input_tokens = 0
    if llama_backend is not None and openai_messages is not None:
        input_tokens = await asyncio.to_thread(llama_backend.count_chat_tokens, openai_messages)

    async def _stream():
        emitter = AnthropicStreamEmitter()
        for line in emitter.start(message_id, model_name, input_tokens = input_tokens):
            yield line

        captured_finish_reason = None

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
                        _fr = cumulative.get("finish_reason")
                        if _fr is not None:
                            captured_finish_reason = _fr
                        for line in emitter.feed(cumulative):
                            yield line
                    continue
                # Plain generator yields cumulative text strings
                for line in emitter.feed({"type": "content", "text": cumulative}):
                    yield line
        except Exception as e:
            logger.error("anthropic_messages stream error: %s", e)
            _error_event = _anthropic_stream_error_event(e)
            if _error_event is not None:
                yield _error_event
                return

        stop_reason = openai_finish_to_anthropic_stop(captured_finish_reason, had_tool_calls = False)
        for line in emitter.finish(stop_reason = stop_reason, stop_sequence = None):
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


def _anthropic_map_generation_error(e: Exception) -> HTTPException:
    """Map an upstream 4xx / context-overflow generation error to a clean
    Anthropic 400 invalid_request_error. Genuine 5xx errors stay 500."""
    if _classify_llama_generation_error(e) is not None:
        return HTTPException(
            status_code = 400,
            detail = anthropic_error_body(
                _friendly_error(e),
                status = 400,
                err_type = "invalid_request_error",
            ),
        )
    return HTTPException(status_code = 500, detail = _friendly_error(e))


def _collect_anthropic_events(run_gen) -> list:
    """Drain the generator into a list, mapping an upstream 4xx / context
    overflow to a clean Anthropic 400 instead of leaking a 500."""
    try:
        return list(run_gen())
    except HTTPException:
        raise
    except Exception as e:
        raise _anthropic_map_generation_error(e)


async def _anthropic_tool_non_streaming(
    run_gen,
    message_id,
    model_name,
    disable_parallel_tool_use = False,
):
    """Non-streaming response for the tool-calling path.

    Builds ``content_blocks`` in generation order (text → tool_use → text →
    tool_use → ...), mirroring the streaming emitter. Deltas within one
    synthesis turn merge into the trailing text block; tool_use blocks interrupt
    the text sequence and open a new text block on the next content event.

    ``prev_text`` is reset on ``tool_end`` because
    ``generate_chat_completion_with_tools`` yields cumulative content *per
    turn* -- the first content event of turn N+1 must diff against an empty
    baseline, not turn N's final length.
    """
    content_blocks: list = []
    tool_blocks_by_id: dict[str, AnthropicResponseToolUseBlock] = {}
    usage = {}
    prev_text = ""
    captured_finish_reason = None
    # Pending client tool_use; cleared by tool_end (server execution) or
    # trailing text. See the stop_reason mapping below.
    ends_on_tool_use = False

    events = _collect_anthropic_events(run_gen)

    for event in events:
        etype = event.get("type", "")
        if etype == "content":
            # Strip leaked tool-call XML
            clean = _TOOL_XML_RE.sub("", event["text"])
            new = clean[len(prev_text) :]
            prev_text = clean
            if new:
                ends_on_tool_use = False
                if content_blocks and isinstance(content_blocks[-1], AnthropicResponseTextBlock):
                    content_blocks[-1].text += new
                else:
                    content_blocks.append(AnthropicResponseTextBlock(text = new))
        elif etype == "tool_start":
            tool_call_id = event["tool_call_id"]
            arguments = event.get("arguments", {})
            existing_tool_block = tool_blocks_by_id.get(tool_call_id) if tool_call_id else None
            if existing_tool_block is not None:
                if arguments or not existing_tool_block.input:
                    existing_tool_block.input = arguments
                if event.get("tool_name") and not existing_tool_block.name:
                    existing_tool_block.name = event["tool_name"]
            else:
                tool_block = AnthropicResponseToolUseBlock(
                    id = anthropic_tool_use_id(tool_call_id),
                    name = event["tool_name"],
                    input = arguments,
                )
                if tool_call_id:
                    tool_blocks_by_id[tool_call_id] = tool_block
                content_blocks.append(tool_block)
            ends_on_tool_use = True
        elif etype == "tool_end":
            prev_text = ""
            # Server-executed: no longer pending a client action (see above).
            ends_on_tool_use = False
        elif etype == "metadata":
            usage = event.get("usage", {})
            _fr = event.get("finish_reason")
            if _fr is not None:
                captured_finish_reason = _fr

    # disable_parallel_tool_use: cap the response to at most one tool_use
    # block. Keep the first tool_use and drop any later ones.
    if disable_parallel_tool_use:
        _seen_tool_use = False
        _capped: list = []
        for block in content_blocks:
            if isinstance(block, AnthropicResponseToolUseBlock):
                if _seen_tool_use:
                    continue
                _seen_tool_use = True
            _capped.append(block)
        content_blocks = _capped

    # stop_reason "tool_use" only when the response still ends on a pending
    # tool_use (client must act). `ends_on_tool_use` is tracked through the
    # event stream above: it is True only if the last tool_start had no
    # following tool_end (server execution) or trailing text.
    stop_reason = openai_finish_to_anthropic_stop(
        captured_finish_reason, had_tool_calls = ends_on_tool_use
    )

    resp = AnthropicMessagesResponse(
        id = message_id,
        model = model_name,
        content = content_blocks,
        stop_reason = stop_reason,
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
    captured_finish_reason = None

    events = _collect_anthropic_events(run_gen)

    for cumulative in events:
        if isinstance(cumulative, dict):
            if cumulative.get("type") == "metadata":
                usage = cumulative.get("usage", {})
                _fr = cumulative.get("finish_reason")
                if _fr is not None:
                    captured_finish_reason = _fr
            continue
        new = cumulative[len(prev_text) :]
        prev_text = cumulative
        if new:
            text_parts.append(new)

    full_text = "".join(text_parts)
    content_blocks = []
    if full_text:
        content_blocks.append(AnthropicResponseTextBlock(text = full_text))

    stop_reason = openai_finish_to_anthropic_stop(captured_finish_reason, had_tool_calls = False)

    resp = AnthropicMessagesResponse(
        id = message_id,
        model = model_name,
        content = content_blocks,
        stop_reason = stop_reason,
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
    seed = None,
    stream_options = None,
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
    if seed is not None:
        body["seed"] = seed
    if stream and stream_options is not None:
        body["stream_options"] = stream_options
    body["max_tokens"] = (
        max_tokens if max_tokens is not None else (backend_ctx or _DEFAULT_MAX_TOKENS_FLOOR)
    )
    body["t_max_predict_ms"] = _DEFAULT_T_MAX_PREDICT_MS
    # Normalize stop the same way the non-passthrough path does (the passthrough
    # was previously the one path that forwarded an empty stop string verbatim).
    _stop = _normalize_stop_sequences(stop)
    if _stop:
        body["stop"] = _stop
    if min_p is not None:
        body["min_p"] = min_p
    if repetition_penalty is not None:
        # llama-server's field is "repeat_penalty", not "repetition_penalty".
        body["repeat_penalty"] = repetition_penalty
    if presence_penalty is not None:
        body["presence_penalty"] = presence_penalty
    if response_format is not None:
        # llama-server applies a GBNF grammar derived from the JSON schema when
        # response_format is present. The field is documented flat at the
        # request root (tools/server/README.md), which is also what the OpenAI
        # SDK produces by spreading extra_body into the body top.
        body["response_format"] = response_format
    if chat_template_kwargs is not None:
        # Propagate reasoning / template overrides (e.g. enable_thinking) so
        # llama-server renders the Jinja template in the caller's mode instead
        # of the model's load-time default.
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
    disable_parallel_tool_use = False,
):
    """Streaming client-side pass-through: forward tools to llama-server and
    translate its stream to Anthropic SSE without executing anything."""
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
        stream_options = {"include_usage": True},
    )

    # Prompt-token count for message_start.usage.input_tokens. count_chat_tokens
    # makes blocking HTTP calls to llama-server, so run it off the event loop.
    # Pass the tools through so tool-schema tokens are counted (otherwise the
    # streaming input_tokens undercounts vs the non-stream / count_tokens paths).
    input_tokens = await asyncio.to_thread(
        llama_backend.count_chat_tokens, openai_messages, None, openai_tools
    )

    # cancel_id mirrors the OpenAI passthrough so a per-run cancel POST
    # works without the caller having to know the local message_id.
    _tracker = _TrackedCancel(cancel_event, cancel_id, session_id, message_id)
    _tracker.__enter__()

    async def _stream():
        emitter = AnthropicPassthroughEmitter()
        for line in emitter.start(message_id, model_name, input_tokens = input_tokens):
            yield line

        # Manage the httpx client, response, AND the aiter_lines() async
        # generator MANUALLY -- no `async with`, no anonymous iterator.
        #
        # On Python 3.13 + httpcore 1.0.x, `async for raw_line in
        # resp.aiter_lines():` creates an anonymous async generator. When the
        # loop exits via `break` (or the generator is orphaned by a mid-stream
        # client disconnect), `async for` does NOT auto-close the iterator like
        # a sync `for` would. The iterator stays reachable only from the current
        # coroutine frame; once `_stream()` returns, the frame is GC'd and the
        # iterator becomes unreachable. The asyncgen finalizer then runs aclose()
        # on a LATER GC pass in a DIFFERENT asyncio task, where httpcore's
        # `HTTP11ConnectionByteStream.aclose()` enters `anyio.CancelScope.__exit__`
        # with a mismatched task and prints `RuntimeError: Attempted to exit
        # cancel scope in a different task` / `RuntimeError: async generator
        # ignored GeneratorExit` as "Exception ignored in:" unraisable warnings.
        #
        # Fix: save `resp.aiter_lines()` as `lines_iter`, and in finally
        # explicitly `await lines_iter.aclose()` BEFORE `resp.aclose()` /
        # `client.aclose()`. This closes the iterator in our own task's event
        # loop, cleaning up the httpcore byte-stream before the asyncgen
        # finalizer has anything orphaned to finalize. Each aclose is wrapped in
        # `try: ... except Exception: pass` so nested anyio cleanup noise can't
        # bubble out.
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

            # Upstream client error (e.g. over-context 400) arrives before any
            # SSE. The 200 stream headers are already flushed, so surface it as
            # an in-band Anthropic ``error`` event instead of silently finishing
            # with an empty end_turn message.
            if resp.status_code != 200:
                _err_bytes = await resp.aread()
                _err_text = _err_bytes.decode("utf-8", "replace")[:500]
                logger.error(
                    "anthropic passthrough upstream error: status=%s body=%s",
                    resp.status_code,
                    _err_text,
                )
                yield build_anthropic_sse_event(
                    "error",
                    anthropic_error_body(
                        f"llama-server error: {_err_text}",
                        status = resp.status_code,
                    ),
                )
                return

            # See _openai_passthrough_stream for rationale: aiter_lines()
            # blocks during llama-server prefill, so the in-loop cancel
            # check is unreachable until the first SSE chunk arrives.
            # The watcher closes `resp` on cancel, raising in aiter_lines.
            cancel_watcher = asyncio.create_task(_await_cancel_then_close(cancel_event, resp))
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
                if disable_parallel_tool_use:
                    _drop_parallel_tool_call_deltas(chunk)
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
    disable_parallel_tool_use = False,
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
    # disable_parallel_tool_use: keep only the first tool_use block.
    if disable_parallel_tool_use and len(tool_calls) > 1:
        tool_calls = tool_calls[:1]
    for tc in tool_calls:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        content_blocks.append(
            AnthropicResponseToolUseBlock(
                id = anthropic_tool_use_id(tc.get("id")),
                name = fn.get("name", ""),
                input = args,
            )
        )

    stop_reason = openai_finish_to_anthropic_stop(finish_reason, had_tool_calls = bool(tool_calls))

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


_LOCAL_SERVER_BUILTIN_TOOL_NAMES = frozenset(
    {"web_search", "web_fetch", "code_execution", "image_generation"}
)


def _strip_provider_synthetic_tool_history(messages: list[dict]) -> list[dict]:
    """Drop synthetic provider-side tool_calls + matching role=tool replies on
    the local-backend (llama-server / GGUF) dispatch path.

    A Gemini chat that ran code_execution / image_generation persists the
    server-side tool card into history as an assistant tool_calls entry tagged
    with ``args._server_tool`` (or a Gemini ``args.google.native_part`` payload)
    plus a follow-up role=tool reply. When the user switches the SAME thread to
    a local GGUF model, those synthetic tool_calls aren't real user functions,
    llama-server has no matching declaration, and Gemini-only ``extra_content``
    / ``native_part`` payloads are meaningless. Forward only ordinary user
    function calls; strip the matched role=tool replies too so the backend never
    sees an orphan tool_call_id.
    """
    dropped_ids: set[str] = set()
    sanitized_assistant: list[dict] = []
    for m in messages:
        if m.get("role") != "assistant":
            sanitized_assistant.append(m)
            continue
        tool_calls = m.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            # Plain text Gemini reply: still strip message-level
            # `extra_content` (carries `google.thought_signature` replay
            # metadata) so a text-only Gemini turn switched to a local GGUF
            # backend doesn't leak Gemini-only fields to llama-server.
            # ChatMessage didn't used to have `extra_content` (implicitly
            # dropped); round-22 added it, which made this leak possible.
            if "extra_content" in m:
                m = {k: v for k, v in m.items() if k != "extra_content"}
            sanitized_assistant.append(m)
            continue
        cleaned: list[dict] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                cleaned.append(tc)
                continue
            fn = tc.get("function")
            name = ""
            if isinstance(fn, dict):
                name = (fn.get("name") or "").lower()
            if name in _LOCAL_SERVER_BUILTIN_TOOL_NAMES:
                raw_args = fn.get("arguments") if isinstance(fn, dict) else None
                args_obj: Any = None
                if isinstance(raw_args, str):
                    try:
                        args_obj = json.loads(raw_args) if raw_args else None
                    except Exception:
                        args_obj = None
                elif isinstance(raw_args, dict):
                    args_obj = raw_args
                is_synthetic = False
                if isinstance(args_obj, dict):
                    if args_obj.get("_server_tool") is True:
                        is_synthetic = True
                    google = args_obj.get("google")
                    if isinstance(google, dict) and isinstance(google.get("native_part"), dict):
                        is_synthetic = True
                if is_synthetic:
                    tc_id = tc.get("id")
                    if isinstance(tc_id, str) and tc_id:
                        dropped_ids.add(tc_id)
                    continue
            # Strip Gemini-only `extra_content` on real user tool_calls too --
            # llama-server has no use for it and may pass it to the model
            # unchanged.
            if "extra_content" in tc:
                tc = {k: v for k, v in tc.items() if k != "extra_content"}
            cleaned.append(tc)
        # Drop message-level `extra_content` (Gemini thoughtSignature replay
        # metadata) on local dispatch.
        m_clean = {k: v for k, v in m.items() if k != "extra_content"}
        if cleaned:
            m_clean["tool_calls"] = cleaned
        else:
            m_clean.pop("tool_calls", None)
        if not m_clean.get("content") and not m_clean.get("tool_calls"):
            continue  # assistant turn now empty, drop
        sanitized_assistant.append(m_clean)

    if not dropped_ids:
        return sanitized_assistant
    out: list[dict] = []
    for m in sanitized_assistant:
        if (
            m.get("role") == "tool"
            and isinstance(m.get("tool_call_id"), str)
            and m["tool_call_id"] in dropped_ids
        ):
            continue
        out.append(m)
    return out


def _openai_messages_for_passthrough(payload) -> list[dict]:
    """Build OpenAI-format message dicts for the /v1/chat/completions
    passthrough path.

    ``payload.messages`` are dumped through Pydantic (dropping unset optional
    fields), so they're already standard OpenAI format -- including
    ``role="tool"`` tool-result messages and assistant messages carrying
    structured ``tool_calls``. Content-parts images already in the list are
    left untouched.

    When a client uses Studio's legacy ``image_base64`` top-level field, the
    image is re-encoded to PNG (llama-server's stb_image has limited format
    support) and spliced into the last user message as an OpenAI ``image_url``
    content part so vision + function-calling requests work transparently.
    """
    messages = _strip_provider_synthetic_tool_history(
        _drop_empty_assistant_sentinels([m.model_dump(exclude_none = True) for m in payload.messages])
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


def _openai_messages_for_gguf_chat(payload, is_vision: bool) -> tuple[list[dict], bool]:
    """Build llama-server messages for the standard GGUF chat path.

    llama-server accepts OpenAI multimodal content parts directly. Preserve all
    per-turn ``image_url`` parts so multi-image chat history keeps each image
    attached to its original turn.
    """
    messages = _strip_provider_synthetic_tool_history(
        _drop_empty_assistant_sentinels([m.model_dump(exclude_none = True) for m in payload.messages])
    )
    has_message_image = any(
        isinstance(msg.get("content"), list)
        and any(part.get("type") == "image_url" for part in msg["content"])
        for msg in messages
    )
    if payload.image_base64 and not has_message_image:
        # Legacy bytes can be any format; the normalizer below sniffs and
        # re-encodes to PNG, so the declared mime is rewritten anyway.
        image_part = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{payload.image_base64}",
            },
        }
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
    has_image = _normalize_anthropic_openai_images(messages, is_vision)
    return messages, has_image


def _extract_response_format(payload):
    """Return the ``response_format`` field on an incoming ChatCompletionRequest
    (or None). The model uses ``extra="allow"`` so pydantic stashes unknown
    top-level fields in ``model_extra``; OpenAI-SDK clients spread ``extra_body``
    into the request body top level, where guided-decoding recipes park their
    JSON-schema response_format.
    """
    extra = getattr(payload, "model_extra", None)
    if not isinstance(extra, dict):
        return None
    rf = extra.get("response_format")
    return rf if isinstance(rf, dict) else None


def _build_openai_passthrough_body(
    payload,
    backend_ctx = None,
    llama_backend = None,
) -> dict:
    """Assemble the llama-server request body from a ChatCompletionRequest.

    Only known OpenAI / llama-server fields are forwarded, so Studio-specific
    extensions (``enable_tools``, ``enabled_tools``, ``session_id``, ...) never
    leak to the backend.
    """
    messages = _openai_messages_for_passthrough(payload)
    system_prompt, _, _ = _extract_content_parts(payload.messages)
    messages = _set_or_prepend_system_message(messages, system_prompt)
    tool_choice = payload.tool_choice if payload.tool_choice is not None else "auto"
    # Forward per-request reasoning fields (enable_thinking / reasoning_effort /
    # preserve_thinking) via chat_template_kwargs so the Jinja template renders
    # in the caller's mode, gated on the active template's capabilities exactly
    # like the non-passthrough paths.
    tpl_kwargs = (
        llama_backend._request_reasoning_kwargs(
            payload.enable_thinking,
            payload.reasoning_effort,
            payload.preserve_thinking,
        )
        if llama_backend is not None
        else None
    )
    return _build_passthrough_payload(
        messages,
        payload.tools,
        payload.temperature,
        payload.top_p,
        payload.top_k,
        # Honor max_completion_tokens on the tools/response_format passthrough too.
        _effective_max_tokens(payload),
        payload.stream,
        stop = payload.stop,
        min_p = payload.min_p,
        repetition_penalty = payload.repetition_penalty,
        presence_penalty = payload.presence_penalty,
        tool_choice = tool_choice,
        response_format = _extract_response_format(payload),
        chat_template_kwargs = tpl_kwargs,
        backend_ctx = backend_ctx,
        seed = payload.seed,
        stream_options = payload.stream_options,
    )


async def _openai_passthrough_stream(
    request, cancel_event, llama_backend, payload, model_name, completion_id
):
    """Streaming client-side pass-through for /v1/chat/completions.

    Forwards the client's OpenAI function-calling request to llama-server and
    relays the SSE stream back verbatim, preserving llama-server's native
    response ``id``, ``finish_reason`` (including ``"tool_calls"``),
    ``delta.tool_calls``, and any client-requested trailing ``usage`` chunk so
    the client sees a standard OpenAI response.
    """
    target_url = f"{llama_backend.base_url}/v1/chat/completions"
    body = _build_openai_passthrough_body(
        payload, backend_ctx = llama_backend.context_length, llama_backend = llama_backend
    )

    _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
    _tracker = _TrackedCancel(cancel_event, *_cancel_keys)
    _tracker.__enter__()

    # Outer guard: asyncio.CancelledError at `await client.send(...)` is a
    # BaseException that bypasses `except httpx.RequestError`; without this the
    # tracker leaks. The generator's finally only runs once iteration starts.
    try:
        # Dispatch BEFORE returning StreamingResponse so transport errors and
        # non-200 upstream statuses surface as real HTTP errors -- OpenAI SDKs
        # rely on status codes to raise APIError/BadRequestError.
        client = httpx.AsyncClient(
            timeout = 600,
            limits = httpx.Limits(max_keepalive_connections = 0),
        )
        resp = None
        _truncate_budget = (
            _OVERFLOW_TRUNCATE_MAX_RETRIES if _overflow_truncation_requested(payload) else 0
        )
        while True:
            try:
                req = client.build_request("POST", target_url, json = body)
                resp = await client.send(req, stream = True)
            except httpx.RequestError as e:
                # llama-server subprocess crashed / starting / unreachable.
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

            if resp.status_code == 200:
                break
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
            # Opt-in overflow policy: shrink and retry instead of a fatal 400.
            if (
                _truncate_budget > 0
                and _classify_llama_generation_error(Exception(err_text))
                and _apply_overflow_truncation(body, err_text)
            ):
                _truncate_budget -= 1
                continue
            try:
                await client.aclose()
            except Exception:
                pass
            raise _openai_passthrough_error(upstream_status, err_text)

        async def _stream():
            # Same httpx lifecycle pattern as _anthropic_passthrough_stream:
            # save resp.aiter_lines() so the finally block can aclose() it on
            # our task. See that function for full rationale.
            lines_iter = None
            # During llama-server prefill, `aiter_lines()` blocks until the
            # first SSE chunk arrives. The in-loop `cancel_event` check can't
            # fire until then -- the exact proxy/Colab scenario the cancel POST
            # recovers from. Run a tiny watcher that closes `resp` as soon as
            # cancel fires, unblocking the iterator with a RemoteProtocolError
            # caught in the except clause below.
            cancel_watcher = asyncio.create_task(_await_cancel_then_close(cancel_event, resp))
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
                    # Honor parallel_tool_calls=false (best-effort): drop tool_call
                    # deltas with index>=1 so only the first call streams. Only
                    # lines carrying tool_calls are reparsed; everything else is
                    # relayed byte-for-byte.
                    if payload.parallel_tool_calls is False and '"tool_calls"' in raw_line:
                        raw_line = _cap_parallel_tool_calls_sse_line(raw_line)
                    # Relay verbatim to preserve llama-server's native id,
                    # finish_reason, delta.tool_calls, and usage chunks.
                    yield raw_line + "\n\n"
                    if raw_line[6:].strip() == "[DONE]":
                        break
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.CloseError):
                # Watcher closed resp on cancel. Emit nothing extra; the client
                # initiated the cancel or already disconnected.
                if not cancel_event.is_set():
                    raise
            except Exception as e:
                # 200 headers already flushed; errors must go in the SSE body.
                logger.error("openai passthrough stream error: %s", e)
                err = _openai_stream_error_chunk(e)
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


async def _openai_passthrough_non_streaming(llama_backend, payload, model_name):
    """Non-streaming client-side pass-through for /v1/chat/completions.

    Returns llama-server's JSON response verbatim so the client sees the native
    response ``id``, ``finish_reason`` (including ``"tool_calls"``), structured
    ``tool_calls``, and accurate ``usage`` token counts.
    """
    target_url = f"{llama_backend.base_url}/v1/chat/completions"
    body = _build_openai_passthrough_body(
        payload, backend_ctx = llama_backend.context_length, llama_backend = llama_backend
    )

    _truncate_budget = (
        _OVERFLOW_TRUNCATE_MAX_RETRIES if _overflow_truncation_requested(payload) else 0
    )
    while True:
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(target_url, json = body, timeout = 600)
        except httpx.RequestError as e:
            # llama-server subprocess crashed / starting / unreachable. Surface the
            # same friendly message the sync chat path emits so operators don't see
            # a bare 500 with no diagnostic.
            logger.error("openai passthrough non-streaming: upstream unreachable: %s", e)
            raise HTTPException(
                status_code = 502,
                detail = _friendly_error(e),
            )

        if resp.status_code == 200:
            break
        # Opt-in overflow policy: shrink and retry instead of a fatal 400.
        if (
            _truncate_budget > 0
            and _classify_llama_generation_error(Exception(resp.text))
            and _apply_overflow_truncation(body, resp.text)
        ):
            _truncate_budget -= 1
            continue
        raise _openai_passthrough_error(resp.status_code, resp.text)

    # The guided-decoding fence wraps each choice's JSON content in a
    # ```json ... ``` markdown fence that data_designer's structured parser
    # requires but which CORRUPTS output for standard OpenAI clients doing
    # ``json.loads(content)``. It is therefore opt-in: only the internal
    # data-recipe path sets ``_unsloth_guided_fence``; public response_format
    # clients get the raw upstream JSON verbatim.
    _guided_fence = bool((payload.model_extra or {}).get("_unsloth_guided_fence"))
    _do_fence = _guided_fence and _extract_response_format(payload) is not None
    _cap_parallel = payload.parallel_tool_calls is False

    try:
        data = resp.json()
    except Exception as exc:
        # Non-JSON / unparseable upstream body: relay verbatim as before.
        logger.warning(
            "openai passthrough non-streaming: response not JSON, relaying raw: %s",
            exc,
        )
        return Response(content = resp.content, media_type = "application/json")

    changed = False
    for choice in data.get("choices", []):
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message")
        if not isinstance(msg, dict):
            continue

        # OpenAI requires content=null on a pure tool-call turn; llama-server
        # emits content="".
        if msg.get("tool_calls") and msg.get("content") == "":
            msg["content"] = None
            changed = True

        # Honor parallel_tool_calls=false (best-effort) by capping to one call.
        if _cap_parallel:
            _tcs = msg.get("tool_calls")
            if isinstance(_tcs, list) and len(_tcs) > 1:
                msg["tool_calls"] = _tcs[:1]
                changed = True

        # Guided-decoding fence wrap (opt-in via _unsloth_guided_fence).
        if _do_fence:
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            stripped = content.strip()
            if not stripped or stripped.startswith("```"):
                continue
            msg["content"] = f"```json\n{stripped}\n```"
            changed = True

    # Nothing mutated: relay the upstream bytes verbatim, skipping a redundant
    # parse + re-serialize round-trip.
    if not changed:
        return Response(content = resp.content, media_type = "application/json")
    return JSONResponse(content = data)
