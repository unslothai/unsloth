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
import hashlib as _hashlib
import hmac as _hmac
import secrets as _secrets
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse, JSONResponse, Response
from starlette.requests import ClientDisconnect
from typing import Any, Callable, List, NamedTuple, Optional, Union
import json
import httpx
from loggers import get_logger
import asyncio
import threading
import weakref
from contextlib import ExitStack, contextmanager
from dataclasses import replace


import re as _re

# Model size extraction (shared with core/inference/llama_cpp.py)
from utils.models import extract_model_size_b as _extract_model_size_b

from utils.api_errors import openai_error_body, anthropic_error_body, error_body_for_path
from utils.upload_limits import STT_AUDIO_B64_MAX_CHARS, STT_AUDIO_RAW_MAX_BYTES
from hub.dependencies import get_hf_token
from core.inference.orchestrator import GenStreamError, GenStreamErrorRaised
from core.inference.llama_admission import (
    LlamaAdmissionCancelled,
    LlamaAdmissionConfig,
    LlamaAdmissionLease,
    LlamaAdmissionQueueFull,
    LlamaAdmissionReservation,
    LlamaAdmissionTimeout,
    get_llama_admission_queue,
    llama_admission_config_from_env,
    peek_llama_admission_snapshot,
)


def _positive_int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None
    return value_int if value_int > 0 else None


def _nonnegative_int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        value_int = int(value)
    except (TypeError, ValueError):
        return None
    return value_int if value_int >= 0 else None


_MLX_MPI_DISTRIBUTED_ENV_PAIRS = (
    ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
    ("PMI_RANK", "PMI_SIZE"),
    ("PMIX_RANK", "PMIX_SIZE"),
    ("MPI_RANK", "MPI_WORLD_SIZE"),
    ("MV2_COMM_WORLD_RANK", "MV2_COMM_WORLD_SIZE"),
)


def _mlx_distributed_launch_detected() -> bool:
    if _nonnegative_int_or_none(os.environ.get("MLX_RANK")) is not None:
        world_size = _positive_int_or_none(os.environ.get("MLX_WORLD_SIZE"))
        if world_size is not None and world_size > 1:
            return True
        return bool(
            os.environ.get("MLX_HOSTFILE")
            or os.environ.get("MLX_IBV_DEVICES")
            or os.environ.get("MLX_JACCL_COORDINATOR")
            or (os.environ.get("NCCL_HOST_IP") and os.environ.get("NCCL_PORT"))
        )
    return any(
        _nonnegative_int_or_none(os.environ.get(rank_env)) is not None
        and (_positive_int_or_none(os.environ.get(size_env)) or 0) > 1
        for rank_env, size_env in _MLX_MPI_DISTRIBUTED_ENV_PAIRS
    )


def _install_httpcore_asyncgen_silencer() -> None:
    """Silence benign httpx/httpcore asyncgen GC noise on Python 3.13.

    When Unsloth proxies a llama-server stream via httpx, the innermost
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


_LOST_CONNECTION_MSG = (
    "Lost connection to the model server. It may have crashed -- try reloading the model."
)


def _friendly_error(exc: Exception) -> str:
    """Extract a user-friendly message from known llama-server errors."""
    if isinstance(exc, httpx.ReadTimeout):
        if "stopped producing tokens" in str(exc).lower():
            return (
                "The model stopped producing tokens before the response "
                "completed. Try stopping and retrying, or reduce max tokens."
            )
        return (
            "The model is still processing the prompt but did not produce a "
            "first token within 20 minutes. Try reducing context length, "
            "using more GPU offload, or loading a smaller model."
        )
    if isinstance(exc, httpx.TimeoutException):
        return "Timed out communicating with the model server. Try again shortly."
    # httpx transport failures from the async pass-through helpers. Any
    # RequestError subclass (ConnectError, ReadError, RemoteProtocolError,
    # WriteError, PoolTimeout, ...) means the llama-server subprocess is
    # unreachable -- crashed or still coming up.
    if isinstance(exc, httpx.RequestError):
        return _LOST_CONNECTION_MSG
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
        return _LOST_CONNECTION_MSG
    template_msg = _template_raise_message(msg, _loaded_chat_template())
    if template_msg:
        return f"An internal error occurred: {template_msg}"
    return "An internal error occurred"


def _friendly_gen_stream_error(value) -> str:
    """Return a client-safe message for typed local generation errors."""
    text = str(value)
    if getattr(value, "public", False):
        return text
    return safe_error_detail(RuntimeError(text), fallback = "An internal error occurred.")


def _friendly_upstream_error(text: str) -> str:
    """Rewrite a raw llama-server error body into an actionable message where we can.

    The main case is a tool-calling grammar that llama-server can't compile ("failed to
    parse grammar" / "failed to initialize samplers"). This surfaces to coding agents as
    a hard 400 on every tool-bearing turn. It is a llama-server limitation with some
    model/quant + tool-schema combinations, and recent llama.cpp builds handle the common
    coding-agent tools, so point the user at updating Unsloth rather than the raw body.
    """
    lowered = text.lower()
    if "failed to parse grammar" in lowered or "failed to initialize samplers" in lowered:
        return (
            "The model couldn't compile a tool-calling grammar for this request. This is a "
            "llama-server limitation with some model/quant and tool-schema combinations. "
            "Update Unsloth (it installs the latest llama.cpp, which handles the common "
            "coding-agent tools) or try a different GGUF model."
        )
    return f"llama-server error: {text}"


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


_OPENAI_COMPAT_STREAM_STALL_TIMEOUT_ENV = "UNSLOTH_OPENAI_COMPAT_STREAM_STALL_TIMEOUT"


def _positive_float_env(env_name: str, default):
    """Parse a positive float from an env var. A parseable non-positive value
    returns ``None`` (0 disables the guarded feature); only unparseable or unset
    values fall back to ``default``."""
    raw_value = os.environ.get(env_name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        value = float(raw_value.strip())
    except ValueError:
        return default
    return value if value > 0 else None


def _effective_openai_max_tokens_from_values(max_tokens, max_completion_tokens = None):
    """Resolve the OpenAI-compatible generation cap from raw request values.

    Prefers ``max_completion_tokens`` over the deprecated ``max_tokens``, and
    returns ``None`` when both are omitted so callers keep their context-window
    default (OpenAI treats an omitted cap as bounded only by the context
    window). Explicit client caps pass through unchanged.
    """

    def _validate_explicit(value, param: str):
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise HTTPException(
                status_code = 400,
                detail = openai_error_body(
                    f"'{param}' must be an integer.",
                    status = 400,
                    code = "invalid_type",
                    param = param,
                ),
            )
        # The legacy completions spec declares ``minimum: 0`` for max_tokens,
        # so 0 is a valid (if degenerate) cap and only negatives are rejected.
        # The chat fields never reach here with 0 (pydantic enforces ge=1).
        if value < 0:
            raise HTTPException(
                status_code = 400,
                detail = openai_error_body(
                    f"'{param}' must be at least 0.",
                    status = 400,
                    code = "invalid_value",
                    param = param,
                ),
            )
        return value

    max_tokens = _validate_explicit(max_tokens, "max_tokens")
    max_completion_tokens = _validate_explicit(max_completion_tokens, "max_completion_tokens")
    return max_completion_tokens if max_completion_tokens is not None else max_tokens


def _effective_openai_max_tokens(payload):
    return _effective_openai_max_tokens_from_values(
        getattr(payload, "max_tokens", None),
        getattr(payload, "max_completion_tokens", None),
    )


def _wants_multiple_choices(payload) -> bool:
    return (payload.n or 1) > 1


def _has_openai_tool_history(messages) -> bool:
    for message in messages or []:
        if isinstance(message, dict):
            if message.get("role") == "tool" or message.get("tool_calls"):
                return True
            continue
        if getattr(message, "role", None) == "tool" or getattr(message, "tool_calls", None):
            return True
    return False


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


def _sse_streaming_response(content, *, unstarted_cleanup = None) -> StreamingResponse:
    """A ``text/event-stream`` response with the standard SSE headers used by
    every streaming path here: no client/proxy caching, no proxy buffering, and
    a one-shot connection. Two callers build their response inline instead: the
    external-provider proxy omits ``Connection: close``, and the OpenAI
    passthrough returns an empty ``keep-alive`` stream when the request is
    cancelled before the upstream response starts.

    Built on ``_SameTaskStreamingResponse`` (not Starlette's stock
    ``StreamingResponse``) so the SSE generator runs in the request task. The
    legacy AnyIO task-group wrapper trips "Attempted to exit a cancel scope in a
    different task" on Python 3.13 + httpx, which surfaced as a mid-stream
    ``response.failed``. The streaming paths that take their response inline use
    ``_SameTaskStreamingResponse`` directly for the same reason."""
    return _SameTaskStreamingResponse(
        content,
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "close",
            "X-Accel-Buffering": "no",
        },
        unstarted_cleanup = unstarted_cleanup,
    )


def _openai_stream_error_chunk(exc) -> dict:
    """Build an in-band OpenAI error chunk for a mid-stream failure. Once the
    stream's 200 headers are flushed the status can't change, so the error must
    ride in the SSE body. An upstream context-window overflow is mapped to
    code=context_length_exceeded so client compaction/trim loops can detect it
    (a code-less error hides it)."""
    _cls = _classify_llama_generation_error(exc)
    if _cls:
        return openai_error_body(
            _friendly_error(exc),
            status = 400,
            code = "context_length_exceeded",
        )
    if _cls is False:
        return openai_error_body(_friendly_error(exc), status = 400)
    return openai_error_body(_friendly_error(exc), status = 500)


def _openai_stream_error_sse(error: dict) -> str:
    return f"data: {json.dumps(error)}\n\ndata: [DONE]\n\n"


def _openai_stream_error_sse_bytes(error: dict) -> bytes:
    return _openai_stream_error_sse(error).encode("utf-8")


def _openai_passthrough_error(status_code, text) -> "HTTPException":
    """HTTPException for a non-200 upstream response on the OpenAI passthrough
    (tools / response_format). An over-context upstream error is mapped to a 400
    with code="context_length_exceeded" so these paths deliver the same signal as
    the non-passthrough path; a tool-grammar compile failure gets the same actionable
    guidance as the Anthropic passthrough; any other upstream error stays verbatim."""
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
        detail = _friendly_upstream_error(text[:500]),
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
    else:
        n_ctx = None
        keep_ratio = 0.6  # no counts in the error; cut conservatively
    # Scale the server-token target into char-estimate units.
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


def _anthropic_stream_error_event(exc, *, force: bool = False):
    """Return an Anthropic in-band stream error event when one is useful."""
    _cls = _classify_llama_generation_error(exc)
    if _cls is None and not force:
        return None
    status = 400 if _cls is not None else 500
    return build_anthropic_sse_event(
        "error",
        anthropic_error_body(_friendly_error(exc), status = status),
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


def _add_empty_content_to_reasoning_deltas(chunk: dict) -> bool:
    """Make reasoning-only deltas palatable to strict OpenAI adapters.

    Some clients built on OpenAI-compatible streams ignore or reject chunks whose
    delta only contains non-standard ``reasoning_content``. Preserve that field,
    but add an empty standard ``content`` member so the chunk is still a valid
    text-delta shape and downstream parsers keep the stream alive.
    """
    changed = False
    choices = chunk.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        if "reasoning_content" in delta and "content" not in delta:
            delta["content"] = ""
            changed = True
    return changed


def _normalize_openai_passthrough_sse_line(
    raw_line: str, *, cap_parallel_tool_calls: bool = False
) -> str:
    """Normalize one passthrough OpenAI SSE ``data:`` line before relaying.

    The function is intentionally narrow: it leaves comments, blank events,
    ``[DONE]``, and unparseable upstream bytes untouched; parsed chunks are
    re-serialized only when a compatibility mutation is actually required.
    """
    if not raw_line.startswith("data:"):
        return raw_line
    # Both mutations key off JSON object keys, so a line without either quoted
    # key can never change; skip the parse on the per-token common case.
    if '"reasoning_content"' not in raw_line and not (
        cap_parallel_tool_calls and '"tool_calls"' in raw_line
    ):
        return raw_line
    payload = raw_line[len("data:") :].lstrip()
    if payload.strip() in ("", "[DONE]"):
        return raw_line
    try:
        obj = json.loads(payload)
    except Exception:
        return raw_line
    if not isinstance(obj, dict):
        return raw_line
    changed = _add_empty_content_to_reasoning_deltas(obj)
    if cap_parallel_tool_calls and _drop_parallel_tool_call_deltas(obj):
        changed = True
    if not changed:
        return raw_line
    return "data: " + json.dumps(obj, separators = (",", ":"), ensure_ascii = False)


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


_OPENAI_PASSTHROUGH_TERMINAL_GRACE_S = 2.0
_SSE_DONE_LINE = "data: [DONE]"
_SSE_DONE_CHUNK = "data: [DONE]\n\n"


def _openai_passthrough_sse_line_terminal_state(raw_line: str) -> Optional[str]:
    """Classify OpenAI-compatible chat stream terminal markers.

    Some llama-server builds can emit the logical final chunk (``finish_reason``)
    and optional usage chunk, then keep the HTTP stream open without sending the
    OpenAI ``data: [DONE]`` sentinel. Classifying those chunks lets Unsloth close
    the client stream promptly while preserving an optional trailing usage chunk.
    """
    if not raw_line.startswith("data:"):
        return None
    data_str = raw_line[5:].lstrip()
    if data_str == "[DONE]":
        return "done"
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return None
    return _openai_passthrough_terminal_state_from_data(data)


def _openai_passthrough_terminal_state_from_data(data) -> Optional[str]:
    """Dict-level core of ``_openai_passthrough_sse_line_terminal_state`` for
    callers that already parsed the chunk (avoids a re-parse per relayed line)."""
    if not isinstance(data, dict):
        return None
    if _monitor_openai_error_message(data):
        return "error"
    choices = data.get("choices")
    if isinstance(choices, list):
        if not choices and isinstance(data.get("usage"), dict):
            return "usage"
        for choice in choices:
            if isinstance(choice, dict) and choice.get("finish_reason") is not None:
                return "finish"
    elif isinstance(data.get("usage"), dict):
        return "usage"
    return None


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


def _chat_chunk_sse(completion_id, created, model_name, *, delta, finish_reason) -> str:
    """One ``ChatCompletionChunk`` as an SSE ``data:`` line. The role / content /
    final chunks every in-process streamer emits differ only in their ``delta``
    and ``finish_reason``."""
    chunk = ChatCompletionChunk(
        id = completion_id,
        created = created,
        model = model_name,
        choices = [ChunkChoice(delta = delta, finish_reason = finish_reason)],
    )
    return f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"


def _chat_role_chunk(completion_id, created, model_name) -> str:
    """Opening assistant-role chunk for a chat stream."""
    return _chat_chunk_sse(
        completion_id,
        created,
        model_name,
        delta = ChoiceDelta(role = "assistant"),
        finish_reason = None,
    )


def _chat_content_chunk(completion_id, created, model_name, text) -> str:
    """A content-delta chunk carrying ``text``."""
    return _chat_chunk_sse(
        completion_id,
        created,
        model_name,
        delta = ChoiceDelta(content = text),
        finish_reason = None,
    )


def _chat_reasoning_chunk(completion_id, created, model_name, text) -> str:
    """Like ``_chat_content_chunk`` but on ``reasoning_content`` (renders the UI thinking block).

    Carries ``content: ""`` alongside, like the GGUF and passthrough paths, so
    strict OpenAI adapters don't drop the reasoning-only delta.
    """
    return _chat_chunk_sse(
        completion_id,
        created,
        model_name,
        delta = ChoiceDelta(content = "", reasoning_content = text),
        finish_reason = None,
    )


def _chat_final_chunk(completion_id, created, model_name, finish_reason) -> str:
    """Terminal stop chunk (empty delta) carrying the finish reason."""
    return _chat_chunk_sse(
        completion_id,
        created,
        model_name,
        delta = ChoiceDelta(),
        finish_reason = finish_reason,
    )


def _chat_tool_calls_chunk(completion_id, created, model_name, tool_calls) -> str:
    """Delta chunk carrying OpenAI tool-call deltas (sibling of ``_chat_content_chunk``)."""
    return _chat_chunk_sse(
        completion_id,
        created,
        model_name,
        delta = ChoiceDelta(tool_calls = tool_calls),
        finish_reason = None,
    )


def _sf_heal_events_to_sse(
    events,
    completion_id,
    created,
    model_name,
    state,
    parallel_tool_calls,
    monitor_id = None,
):
    """Serialize ``StreamToolCallHealer`` events into chat SSE lines.

    ``state["idx"]`` tracks the call index across ``feed``/``finalize``;
    ``parallel_tool_calls is False`` caps promotion to one call (GGUF parity).
    The monitor is fed from the same events the client receives, never the
    healed-away markup."""
    lines = []
    for kind, value in events:
        if kind == "text":
            if value:
                lines.append(_chat_content_chunk(completion_id, created, model_name, value))
                api_monitor.append_reply(monitor_id, value)
            continue
        if parallel_tool_calls is False and state["idx"] >= 1:
            continue
        lines.append(
            _chat_tool_calls_chunk(
                completion_id,
                created,
                model_name,
                [
                    {
                        "index": state["idx"],
                        "id": value["id"],
                        "type": "function",
                        "function": value["function"],
                    }
                ],
            )
        )
        _fn = value.get("function") or {}
        api_monitor.append_reply(
            monitor_id,
            ("[tool_calls] " if state["idx"] == 0 else "; ")
            + f"{_fn.get('name', '')}({_fn.get('arguments', '')})",
        )
        state["idx"] += 1
    return lines


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
        GgufLoadIntent,
        LlamaCppBackend,
        _DEFAULT_FIRST_TOKEN_TIMEOUT_S,
        _DEFAULT_MAX_TOKENS_FLOOR,
        _DEFAULT_STREAM_STALL_TIMEOUT_S,
        _extra_args_draft_device_pin,
        _extra_args_n_ubatch,
        _hf_offline_if_unreachable,
        _hf_offline_if_unreachable_for,
        _kv_bytes_per_elem,
        _kv_unified_from_args,
        _metal_device_is_paravirtual,
        _planned_main_cache_types,
        _swa_full_from_args_or_env,
        detect_reasoning_flags,
        paravirtual_normalized_request,
    )
    from core.inference.llama_server_args import (
        _effective_tensor_parallel,
        extra_args_disable_mmproj,
        parse_gpu_layers_override,
        parse_split_mode_override,
        resolve_tensor_parallel,
        strip_shadowing_flags,
        validate_extra_args,
    )
    from core.inference.tensor_fallback import load_with_tensor_fallback
    from utils.models import ModelConfig
    from utils.paths import is_local_path
    from utils.inference import load_inference_config
    from utils.models.model_config import (
        _local_gguf_companion_search_root,
        colocated_split_shards,
        detect_mtp_file,
        load_model_defaults,
    )
    from utils.native_path_leases import (
        NativePathLeaseError,
        display_label_for_native_path,
        is_registered_native_path_label,
        native_gguf_companion_parent_allowed,
        redact_native_paths,
        verify_native_path_lease,
    )
except ImportError:
    parent_backend = backend_path.parent / "backend"
    if str(parent_backend) not in sys.path:
        sys.path.insert(0, str(parent_backend))
    from core.inference import get_inference_backend
    from core.inference.llama_cpp import (
        GgufLoadIntent,
        LlamaCppBackend,
        _DEFAULT_FIRST_TOKEN_TIMEOUT_S,
        _DEFAULT_MAX_TOKENS_FLOOR,
        _DEFAULT_STREAM_STALL_TIMEOUT_S,
        _extra_args_draft_device_pin,
        _extra_args_n_ubatch,
        _hf_offline_if_unreachable,
        _hf_offline_if_unreachable_for,
        _kv_bytes_per_elem,
        _kv_unified_from_args,
        _metal_device_is_paravirtual,
        _planned_main_cache_types,
        _swa_full_from_args_or_env,
        detect_reasoning_flags,
        paravirtual_normalized_request,
    )
    from core.inference.llama_server_args import (
        _effective_tensor_parallel,
        extra_args_disable_mmproj,
        parse_gpu_layers_override,
        parse_split_mode_override,
        resolve_tensor_parallel,
        strip_shadowing_flags,
        validate_extra_args,
    )
    from core.inference.tensor_fallback import load_with_tensor_fallback
    from utils.models import ModelConfig
    from utils.paths import is_local_path
    from utils.inference import load_inference_config
    from utils.models.model_config import (
        _local_gguf_companion_search_root,
        colocated_split_shards,
        detect_mtp_file,
        load_model_defaults,
    )
    from utils.native_path_leases import (
        NativePathLeaseError,
        display_label_for_native_path,
        is_registered_native_path_label,
        native_gguf_companion_parent_allowed,
        redact_native_paths,
        verify_native_path_lease,
    )


def _llama_non_streaming_generation_timeout() -> httpx.Timeout:
    return httpx.Timeout(_DEFAULT_FIRST_TOKEN_TIMEOUT_S)


def _llama_streaming_generation_timeout() -> httpx.Timeout:
    return httpx.Timeout(_DEFAULT_FIRST_TOKEN_TIMEOUT_S)


def _set_stream_response_read_timeout(
    response: httpx.Response, read_timeout_s: Optional[float] = _DEFAULT_STREAM_STALL_TIMEOUT_S
) -> None:
    # ``read_timeout_s = None`` clears httpx's read timeout (wait indefinitely),
    # used when the stall guard is disabled so a stale first-token deadline
    # can't keep timing out post-first-chunk gaps.
    try:
        timeout_ext = response.request.extensions.get("timeout")
        if isinstance(timeout_ext, dict):
            timeout_ext["read"] = read_timeout_s
    except Exception:
        pass


_STREAM_DISCONNECT_POLL_TIMEOUT_S = 0.25
_OPENAI_PASSTHROUGH_PREHEADER_STATUS_WINDOW_S = 0.1
_OPENAI_PASSTHROUGH_PENDING_RESPONSE_KEEPALIVE_S = 5.0
_OPENAI_PASSTHROUGH_SSE_KEEPALIVE = ": keep-alive\n\n"
_OPENAI_LLAMA_ADMISSION_POLL_S = 0.25
# Cap on waiting for a cancelled teardown task. Request.is_disconnected() can swallow
# cancel() (#7617), so teardown abandons the task rather than hold the response, and
# the process-wide slot, open forever.
_TEARDOWN_TASK_STOP_TIMEOUT_S = 5.0
# Idle window before a local tool-loop stream emits an SSE keepalive comment
# (e.g. prompt prefill between tool iterations). A second layer atop the
# tool_stream_exec heartbeats, keeping proxies (Cloudflare drops idle at ~100s).
_LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S = 15.0


def _openai_llama_admission_capacity(request: Optional[Request], llama_backend = None) -> int:
    """Serving slots available for one local llama-server backend.

    The loaded backend is the source of truth because it may have reduced
    ``--parallel`` at load time to keep the model on GPU. The app state is a
    launch-intent fallback for tests and for the short window before a backend
    reports its committed runtime slots.
    """
    slots = _positive_int_or_none(getattr(llama_backend, "effective_parallel_slots", None))
    if slots is not None:
        return slots
    try:
        slots = getattr(request.app.state, "llama_parallel_slots", None)
    except Exception:
        slots = None
    return _positive_int_or_none(slots) or 1


def _openai_llama_admission_reserve(
    *, request: Optional[Request], llama_backend
) -> tuple[LlamaAdmissionReservation, LlamaAdmissionConfig]:
    config = llama_admission_config_from_env()
    capacity = _openai_llama_admission_capacity(request, llama_backend)
    key = str(getattr(llama_backend, "base_url", "llama-server"))
    reservation = get_llama_admission_queue(key).reserve(
        capacity = capacity,
        config = config,
    )
    return reservation, config


def _openai_admission_request_path(request: Optional[Request]) -> Optional[str]:
    try:
        return str(request.url.path) if request is not None else None
    except Exception:
        return None


def _llama_admission_log(
    event: str,
    reservation: Optional[LlamaAdmissionReservation] = None,
    *,
    snapshot = None,
    request: Optional[Request],
    mode: str,
    wait_started_at: Optional[float] = None,
    completion_id: Optional[str] = None,
    level: str = "debug",
) -> None:
    if snapshot is None and reservation is not None:
        snapshot = reservation.snapshot_now()
    wait_ms = None
    if wait_started_at is not None:
        wait_ms = int(max(0.0, time.monotonic() - wait_started_at) * 1000)
    log = getattr(logger, level, logger.debug)
    log(
        "llama admission %s: mode=%s path=%s completion_id=%s "
        "pool=%s/%s free=%s queued=%s wait_ms=%s",
        event,
        mode,
        _openai_admission_request_path(request),
        completion_id,
        getattr(snapshot, "active", None),
        getattr(snapshot, "capacity", None),
        getattr(snapshot, "free", None),
        getattr(snapshot, "queued", None),
        wait_ms,
    )


def _openai_admission_error_body(exc: Exception, *, status_code: int) -> dict:
    snapshot = getattr(exc, "snapshot", None)
    message = str(exc)
    if snapshot is not None:
        message = (
            f"{message} "
            f"(active={snapshot.active}, queued={snapshot.queued}, capacity={snapshot.capacity})"
        )
    return openai_error_body(message, status = status_code)


def _openai_admission_http_exception(exc: Exception, *, status_code: int) -> HTTPException:
    return HTTPException(
        status_code = status_code,
        detail = _openai_admission_error_body(exc, status_code = status_code),
    )


def _anthropic_admission_http_exception(exc: Exception, *, status_code: int) -> HTTPException:
    """Anthropic-shaped error for an admission reject/timeout/cancel (429/503/499)."""
    snapshot = getattr(exc, "snapshot", None)
    message = str(exc)
    if snapshot is not None:
        message = (
            f"{message} "
            f"(active={snapshot.active}, queued={snapshot.queued}, capacity={snapshot.capacity})"
        )
    # Types come from ANTHROPIC_TYPE_BY_STATUS (429 -> rate_limit_error, which is
    # what Anthropic SDKs back off on); overloaded_error is reserved for 529.
    return HTTPException(
        status_code = status_code,
        detail = anthropic_error_body(message, status = status_code),
    )


def _openai_admission_timeout_error(
    reservation: LlamaAdmissionReservation,
) -> LlamaAdmissionTimeout:
    return LlamaAdmissionTimeout(
        "Timed out waiting for an available local llama-server generation slot",
        snapshot = reservation.snapshot_now(),
    )


def _openai_admission_cancelled_error(
    reservation: LlamaAdmissionReservation,
) -> LlamaAdmissionCancelled:
    return LlamaAdmissionCancelled(
        "Client disconnected before an upstream llama-server generation slot was available",
        snapshot = reservation.snapshot_now(),
    )


async def _raise_if_openai_admission_cancelled(
    reservation: LlamaAdmissionReservation, *, request: Optional[Request], cancel_event
) -> None:
    if reservation.is_cancelled:
        raise _openai_admission_cancelled_error(reservation)
    if await _preheader_cancelled(cancel_event, request):
        reservation.cancel()
        raise _openai_admission_cancelled_error(reservation)


async def _wait_for_openai_admission_non_streaming(
    reservation: LlamaAdmissionReservation,
    config: LlamaAdmissionConfig,
    *,
    request: Optional[Request],
    cancel_event,
) -> LlamaAdmissionLease:
    lease = reservation.lease_nowait()
    if lease is not None:
        try:
            await _raise_if_openai_admission_cancelled(
                reservation,
                request = request,
                cancel_event = cancel_event,
            )
        except asyncio.CancelledError:
            lease.release()
            raise
        except LlamaAdmissionCancelled:
            lease.release()
            raise
        return lease
    await _raise_if_openai_admission_cancelled(
        reservation,
        request = request,
        cancel_event = cancel_event,
    )
    deadline = None if config.queue_timeout_s is None else time.monotonic() + config.queue_timeout_s
    try:
        while True:
            await _raise_if_openai_admission_cancelled(
                reservation,
                request = request,
                cancel_event = cancel_event,
            )
            lease = reservation.lease_nowait()
            if lease is not None:
                try:
                    await _raise_if_openai_admission_cancelled(
                        reservation,
                        request = request,
                        cancel_event = cancel_event,
                    )
                except asyncio.CancelledError:
                    lease.release()
                    raise
                except LlamaAdmissionCancelled:
                    lease.release()
                    raise
                return lease
            wait_s = _OPENAI_LLAMA_ADMISSION_POLL_S
            if deadline is not None:
                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0:
                    reservation.cancel()
                    raise _openai_admission_timeout_error(reservation)
                wait_s = min(wait_s, max(remaining_s, 0.001))
            try:
                lease = await reservation.wait(wait_s)
            except asyncio.TimeoutError:
                continue
            if lease is not None:
                return lease
            await _raise_if_openai_admission_cancelled(
                reservation,
                request = request,
                cancel_event = cancel_event,
            )
    except asyncio.CancelledError:
        reservation.cancel()
        raise


async def _openai_admission_wait_stream_chunks(
    reservation: LlamaAdmissionReservation,
    config: LlamaAdmissionConfig,
    *,
    request: Optional[Request],
    cancel_event,
):
    lease = reservation.lease_nowait()
    if lease is not None:
        yield lease
        return

    await _raise_if_openai_admission_cancelled(
        reservation,
        request = request,
        cancel_event = cancel_event,
    )
    deadline = None if config.queue_timeout_s is None else time.monotonic() + config.queue_timeout_s
    keepalive_interval_s = max(0.001, config.keepalive_interval_s)
    next_keepalive_at = time.monotonic() + keepalive_interval_s
    try:
        while True:
            await _raise_if_openai_admission_cancelled(
                reservation,
                request = request,
                cancel_event = cancel_event,
            )
            lease = reservation.lease_nowait()
            if lease is not None:
                yield lease
                return

            now = time.monotonic()
            wait_s = min(_OPENAI_LLAMA_ADMISSION_POLL_S, max(next_keepalive_at - now, 0.001))
            if deadline is not None:
                remaining_s = deadline - now
                if remaining_s <= 0:
                    reservation.cancel()
                    raise _openai_admission_timeout_error(reservation)
                wait_s = min(wait_s, max(remaining_s, 0.001))
            try:
                lease = await reservation.wait(wait_s)
            except asyncio.TimeoutError:
                lease = None
            if lease is not None:
                yield lease
                return
            await _raise_if_openai_admission_cancelled(
                reservation,
                request = request,
                cancel_event = cancel_event,
            )
            now = time.monotonic()
            if now >= next_keepalive_at:
                next_keepalive_at = now + keepalive_interval_s
                yield _OPENAI_PASSTHROUGH_SSE_KEEPALIVE
    except asyncio.CancelledError:
        reservation.cancel()
        raise


async def _close_openai_admitted_stream_iterator(iterator, *, cancelled: bool) -> None:
    if iterator is None:
        return
    if cancelled:
        athrow = getattr(iterator, "athrow", None)
        if athrow is not None:
            try:
                await athrow(asyncio.CancelledError())
            except (asyncio.CancelledError, StopAsyncIteration, RuntimeError):
                return
    aclose = getattr(iterator, "aclose", None)
    if aclose is not None:
        await aclose()


def _openai_compat_stream_stall_timeout():
    """Max silent gap after an OpenAI passthrough stream has produced data.

    If the socket goes silent after valid SSE data, this bounds how long the
    client is kept open. Defaults to the backend-wide stall timeout so this
    path stalls out like every sibling stream; set the env var to tighten it
    for local serving, or to 0 to disable the guard.
    """
    return _positive_float_env(
        _OPENAI_COMPAT_STREAM_STALL_TIMEOUT_ENV,
        _DEFAULT_STREAM_STALL_TIMEOUT_S,
    )


def _openai_passthrough_upstream_headers(*, llama_backend = None) -> dict:
    headers = {}
    auth_headers = getattr(llama_backend, "_auth_headers", None)
    if isinstance(auth_headers, dict):
        headers.update(auth_headers)
    headers["Connection"] = "close"
    return headers


class _CompatSameTaskTimeout:
    """Same-task timeout fallback for Python versions before asyncio.timeout."""

    def __init__(self, timeout_s: float):
        self.timeout_s = timeout_s
        self._task = None
        self._handle = None
        self._timed_out = False
        self._cancelling = 0

    async def __aenter__(self):
        self._task = asyncio.current_task()
        if self._task is None:
            return self
        if hasattr(self._task, "cancelling"):
            self._cancelling = self._task.cancelling()
        loop = asyncio.get_running_loop()
        self._handle = loop.call_later(max(self.timeout_s, 0), self._cancel_task)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.cancel()
        if exc_type is not None and issubclass(exc_type, asyncio.CancelledError):
            if self._timed_out:
                if self._task is not None and hasattr(self._task, "uncancel"):
                    if self._task.uncancel() > self._cancelling:
                        return None
                raise asyncio.TimeoutError from exc
        return None

    def _cancel_task(self) -> None:
        self._timed_out = True
        if self._task is not None:
            self._task.cancel()


def _same_task_timeout(timeout_s: float):
    timeout_ctx = getattr(asyncio, "timeout", None)
    if timeout_ctx is not None:
        return timeout_ctx(timeout_s)
    return _CompatSameTaskTimeout(timeout_s)


class _SameTaskStreamingResponse(StreamingResponse):
    """StreamingResponse without Starlette's legacy AnyIO task-group wrapper."""

    def __init__(
        self,
        *args,
        unstarted_cleanup = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Released when the client disconnects before the body iterator starts:
        # its try/finally never runs, so a stream that opens resources before the
        # first yield (the passthrough's upstream httpx stream) passes this.
        self._unstarted_cleanup = unstarted_cleanup

    async def __call__(self, scope, receive, send) -> None:
        # send() emits a body message only after the first chunk, so no body
        # message means the generator never entered its try/finally.
        body_started = False

        async def _tracking_send(message) -> None:
            nonlocal body_started
            if message.get("type") == "http.response.body":
                body_started = True
            await send(message)

        try:
            await self.stream_response(_tracking_send)
        except OSError:  # client disconnected mid-send
            if body_started:
                # Generator is suspended in its try/finally: throw CancelledError
                # (not aclose's GeneratorExit) so its handler finishes the
                # api_monitor entry. Fall back to aclose() without athrow.
                athrow = getattr(self.body_iterator, "athrow", None)
                if athrow is not None:
                    try:
                        await athrow(asyncio.CancelledError())
                    except (asyncio.CancelledError, StopAsyncIteration, RuntimeError):
                        pass
                else:
                    aclose = getattr(self.body_iterator, "aclose", None)
                    if aclose is not None:
                        await aclose()
            else:
                # Generator never started; aclose()/athrow() are no-ops on it, so
                # release eager resources via the hook. getattr guards a response
                # built through __new__ without __init__ (tests, pickling).
                aclose = getattr(self.body_iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
                cleanup = getattr(self, "_unstarted_cleanup", None)
                if cleanup is not None:
                    try:
                        await cleanup()
                    except Exception:
                        pass
            raise ClientDisconnect()
        if self.background is not None:
            await self.background()


async def _release_unstarted_anthropic_stream(iterator, prior_cleanup) -> None:
    """Close a stream whose body never started, running the response's own
    pre-start hook. aclose() on an unstarted async generator is a no-op, so its
    finally never runs and anything the builder acquired eagerly (the passthrough
    cancel tracker) would leak without the hook."""
    aclose = getattr(iterator, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
        except Exception:
            pass
    if prior_cleanup is not None:
        try:
            await prior_cleanup()
        except Exception:
            pass


def _tracked_cancel_unstarted_cleanup(tracker):
    """unstarted_cleanup that exits ``tracker`` on a pre-start disconnect, when
    the generator's finally (which normally exits it) never runs."""

    async def _cleanup() -> None:
        tracker.__exit__(None, None, None)

    return _cleanup


# Cloudflare quick tunnels (--secure) drop a request whose origin has sent no body
# bytes for ~100s, and a 600 GB GGUF load runs 100-330s. Measured on a real quick
# tunnel: headers at t=0 with no body still 524s, one space every 20s survives. So
# a slow call commits a 200 and pads until its payload is ready; leading whitespace
# is legal JSON, so clients parse the body as-is.
_TUNNEL_KEEPALIVE_AFTER_S = 15.0
_TUNNEL_KEEPALIVE_EVERY_S = 20.0

# Underscored so it cannot collide with a real field or the OpenAI ``error`` envelope.
_DEFERRED_ERROR_KEY = "_deferred_error"


def _deferred_error_body(status_code: int, detail) -> bytes:
    body = {_DEFERRED_ERROR_KEY: {"status_code": status_code, "detail": detail}}
    return json.dumps(body).encode()


async def _tunnel_safe_json(coro, *, label: str):
    """Await ``coro``, padding the response body if it outruns the tunnel timer.

    A call finishing within ``_TUNNEL_KEEPALIVE_AFTER_S`` keeps the current
    contract exactly, HTTPException status code included; every early failure
    (validation, unknown identifier, download-manager and sidecar 409s) raises in
    that window. Only a slower call switches to a padded stream, and it must
    report a late failure in the body because the status line is already gone.

    A client disconnect does not cancel the work: the model stays resident, as
    it does today.
    """
    task = asyncio.ensure_future(coro)
    # A client that disconnects mid-pad leaves nobody to await the task, and an
    # unretrieved exception logs "Task exception was never retrieved". Retrieving
    # it here does not consume it: result() below still raises.
    task.add_done_callback(lambda t: t.cancelled() or t.exception())
    done, _ = await asyncio.wait({task}, timeout = _TUNNEL_KEEPALIVE_AFTER_S)
    if done:
        return task.result()  # re-raises exactly as an un-wrapped await would

    logger.info(
        f"{label} exceeded {_TUNNEL_KEEPALIVE_AFTER_S:.0f}s; "
        "padding the response so a proxy cannot time it out"
    )

    async def _body():
        while True:
            finished, _ = await asyncio.wait({task}, timeout = _TUNNEL_KEEPALIVE_EVERY_S)
            if not finished:
                yield b" "
                continue
            try:
                payload = task.result()
            except HTTPException as exc:
                logger.info(f"{label} failed with {exc.status_code} after the response committed")
                yield _deferred_error_body(exc.status_code, exc.detail)
            except Exception as exc:
                logger.exception(f"{label} failed after the response was committed")
                yield _deferred_error_body(500, f"{type(exc).__name__}: {exc}")
            else:
                yield json.dumps(jsonable_encoder(payload)).encode()
            return

    return _SameTaskStreamingResponse(
        _body(),
        media_type = "application/json",
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _aclose_stream_resources(
    *,
    watchers = (),
    iterator = None,
    resp = None,
    client = None,
) -> None:
    """Tear down an httpx streaming generator's resources in the required order:
    cancel + bounded-wait each watcher task, then aclose() the byte/line iterator,
    the response, and the client. Each step swallows its own exceptions so teardown
    always completes; a close-time CancelledError is re-raised only after every
    step has run. See _anthropic_passthrough_stream for the ordering rationale."""
    # Bounded: a watcher parked in Request.is_disconnected() can swallow cancel(), so an
    # unbounded await holds the response open. Stopped together, so N watchers cost one
    # bound before the closes, which are what stop llama-server decoding. #7617
    live = [w for w in watchers if w is not None]
    if live:
        # Cancel before the first await, else a cancel here stops the gather before it
        # steps its children, leaving watchers never cancelled.
        for watcher in live:
            watcher.cancel()
        try:
            await asyncio.gather(
                *(_stop_local_disconnect_cancel_watcher(w) for w in live),
                return_exceptions = True,
            )
        except (asyncio.CancelledError, Exception):
            pass
    close_cancelled = False
    if iterator is not None:
        try:
            await iterator.aclose()
        except asyncio.CancelledError:
            close_cancelled = True
        except Exception:
            pass
    if resp is not None:
        try:
            await resp.aclose()
        except asyncio.CancelledError:
            close_cancelled = True
        except Exception:
            pass
    if client is not None:
        try:
            await client.aclose()
        except asyncio.CancelledError:
            close_cancelled = True
        except Exception:
            pass
    if close_cancelled:
        raise asyncio.CancelledError()


# The loop holds only weak refs to tasks, so a bare ensure_future() close can be collected
# before it runs. Strong ref until done.
_LATE_CLOSE_TASKS: set = set()


async def _aclose_quietly(obj) -> None:
    try:
        await obj.aclose()
    except Exception:
        pass


def _discard_task_outcome(task: asyncio.Task) -> None:
    """Drain an abandoned teardown task, closing a late response rather than dropping it.

    Closing the per-request client does not close a response the send produces on a
    connection opened after that close. Never raises: this is a done callback. #7617
    """
    try:
        if task.cancelled():
            return
        if task.exception() is not None:
            return
        result = task.result()
    except Exception:
        return
    if result is not None and hasattr(result, "aclose") and not getattr(result, "is_closed", False):
        try:
            closing = asyncio.ensure_future(_aclose_quietly(result))
        except RuntimeError:
            return
        _LATE_CLOSE_TASKS.add(closing)
        closing.add_done_callback(_LATE_CLOSE_TASKS.discard)


def _release_admission(admission_lease = None, tracker = None) -> None:
    """Give back the process-wide llama-server slot and the cancel-registry entry.

    Must run after the upstream response is closed: on disconnect llama-server keeps
    decoding until ``resp`` is closed, so releasing first admits a second request past
    --parallel. Safe behind the closes only because every teardown await is bounded. #7617
    """
    try:
        if admission_lease is not None:
            admission_lease.release()
    finally:
        if tracker is not None:
            tracker.__exit__(None, None, None)


async def _preheader_cancelled(cancel_event = None, request: Optional[Request] = None) -> bool:
    if cancel_event is not None and cancel_event.is_set():
        return True
    if request is not None and await request.is_disconnected():
        if cancel_event is not None:
            cancel_event.set()
        return True
    return False


async def _wait_preheader_cancel(cancel_event = None, request: Optional[Request] = None) -> None:
    while not await _preheader_cancelled(cancel_event, request):
        await asyncio.sleep(0.05)


async def _send_stream_with_preheader_cancel(
    client: httpx.AsyncClient,
    req: httpx.Request,
    cancel_event = None,
    request: Optional[Request] = None,
    mark_cancel_on_cancel: bool = True,
) -> Optional[httpx.Response]:
    if cancel_event is None and request is None:
        return await client.send(req, stream = True)
    if await _preheader_cancelled(cancel_event, request):
        return None

    send_task = asyncio.create_task(client.send(req, stream = True))
    cancel_task = asyncio.create_task(_wait_preheader_cancel(cancel_event, request))

    async def _stop_send_task() -> None:
        try:
            await client.aclose()
        except Exception:
            pass
        # Bounded: the client is already closed, so an abandoned send owns nothing and
        # the callback drains its result. #7617
        send_task.cancel()
        done, _pending = await asyncio.wait({send_task}, timeout = _TEARDOWN_TASK_STOP_TIMEOUT_S)
        if not done:
            send_task.add_done_callback(_discard_task_outcome)
            return
        try:
            # The aclose() above does not close a response the send produced during it.
            sent = send_task.result()
            if sent is not None:
                await _aclose_quietly(sent)
        except (asyncio.CancelledError, Exception):
            pass

    try:
        done, _pending = await asyncio.wait(
            {send_task, cancel_task},
            return_when = asyncio.FIRST_COMPLETED,
        )
        if send_task in done:
            return await send_task

        await _stop_send_task()
        return None
    except asyncio.CancelledError:
        if mark_cancel_on_cancel and cancel_event is not None:
            cancel_event.set()
        await _stop_send_task()
        raise
    finally:
        # Bounded: cancel_task polls Request.is_disconnected(), which can swallow cancel(),
        # and this finally also runs on the success path, before the first byte. #7617
        try:
            await _stop_local_disconnect_cancel_watcher(cancel_task)
        except (asyncio.CancelledError, Exception):
            pass


async def _aiter_llama_stream_items(
    async_iter,
    *,
    cancel_event = None,
    request: Optional[Request] = None,
    first_token_deadline: Optional[float] = None,
    response: Optional[httpx.Response] = None,
    post_first_item_read_timeout_s: Optional[
        Union[float, Callable[[], Optional[float]]]
    ] = _DEFAULT_STREAM_STALL_TIMEOUT_S,
):
    if first_token_deadline is None:
        first_token_deadline = time.monotonic() + _DEFAULT_FIRST_TOKEN_TIMEOUT_S
    last_item_at: Optional[float] = None

    def _post_first_timeout_s() -> Optional[float]:
        if callable(post_first_item_read_timeout_s):
            return post_first_item_read_timeout_s()
        return post_first_item_read_timeout_s

    while True:
        if cancel_event is not None and cancel_event.is_set():
            return
        if request is not None and await request.is_disconnected():
            if cancel_event is not None:
                cancel_event.set()
            return
        waiting_first_item = last_item_at is None
        try:
            if waiting_first_item:
                remaining_s = first_token_deadline - time.monotonic()
                if remaining_s <= 0:
                    raise httpx.ReadTimeout("The model did not produce a first token in time.")
                if response is not None:
                    _set_stream_response_read_timeout(response, remaining_s)
                # Keep httpx/httpcore's AnyIO cancel scope in this task.
                # asyncio.wait_for would drive __anext__ in a child task.
                async with _same_task_timeout(remaining_s):
                    item = await async_iter.__anext__()
            else:
                timeout_s = _post_first_timeout_s()
                if (
                    request is not None
                    and response is not None
                    and timeout_s is not None
                    and last_item_at is not None
                ):
                    stall_remaining_s = timeout_s - (time.monotonic() - last_item_at)
                    if stall_remaining_s <= 0:
                        raise httpx.ReadTimeout("The model stopped producing tokens mid-response.")
                    _set_stream_response_read_timeout(response, stall_remaining_s)
                item = await async_iter.__anext__()
        except asyncio.TimeoutError as exc:
            if waiting_first_item:
                raise httpx.ReadTimeout("The model did not produce a first token in time.") from exc
            raise
        except StopAsyncIteration:
            return
        except httpx.ReadTimeout:
            now = time.monotonic()
            if last_item_at is None:
                if now >= first_token_deadline:
                    raise
                continue
            timeout_s = _post_first_timeout_s()
            if request is not None and timeout_s is not None and now - last_item_at < timeout_s:
                continue
            raise httpx.ReadTimeout("The model stopped producing tokens mid-response.")
        if last_item_at is None and response is not None:
            # The first-token read deadline no longer applies once a chunk has
            # arrived: switch to the stall timeout, or clear the read timeout
            # entirely when the stall guard is disabled (callable returns None)
            # so a long gap can't trip the stale first-token deadline.
            _set_stream_response_read_timeout(response, _post_first_timeout_s())
        last_item_at = time.monotonic()
        yield item


from models.inference import (
    _InferenceRuntimeFields,
    LoadRequest,
    UnloadRequest,
    TranscribeRequest,
    SttLoadRequest,
    GenerateRequest,
    DiffusionLoadRequest,
    DiffusionGenerateRequest,
    DiffusionGenerateResponse,
    DiffusionGenerateProgressResponse,
    DiffusionStatusResponse,
    DiffusionDownloadPlanResponse,
    DiffusionInferenceInfoResponse,
    DiffusionLoadProgressResponse,
    GalleryImage,
    GalleryListResponse,
    ImageGenerationRequest,
    ImageGenerationData,
    ImageGenerationResponse,
    LoadResponse,
    LoadProgressResponse,
    UnloadResponse,
    InferenceStatusResponse,
    ChatCompletionRequest,
    ChatCountTokensRequest,
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
    TransformersUpgradeInfo,
    InstallLatestTransformersRequest,
    InstallLatestTransformersResponse,
    TextContentPart,
    ImageContentPart,
    ImageUrl,
    ResponsesRequest,
    ResponsesInputTextPart,
    ResponsesInputImagePart,
    ResponsesOutputTextPart,
    ResponsesUnknownInputItem,
    ResponsesFunctionCallInputItem,
    ResponsesFunctionCallOutputInputItem,
    ResponsesOutputTextContent,
    ResponsesOutputMessage,
    ResponsesOutputReasoning,
    ResponsesOutputReasoningContent,
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
    anthropic_schema_client_tool_kind,
    anthropic_tools_to_openai,
    anthropic_tool_choice_to_openai,
    openai_finish_to_anthropic_stop,
    anthropic_tool_use_id,
    build_anthropic_sse_event,
    AnthropicStreamEmitter,
    AnthropicPassthroughEmitter,
)
from auth.authentication import API_KEY_PREFIX, get_current_subject
from state import active_generations


def _request_used_api_key(request: Any) -> bool:
    """True when this request authenticated with an sk-unsloth key.

    Studio's own chat hits these same endpoints with a session JWT, so this is
    what separates "someone is using Unsloth as an API server" from "someone is
    using Unsloth".
    """
    # Total by construction: this only decides a monitor label and must never fail a
    # load. Only a real Request hands back a string; the load routes take stand-ins too.
    try:
        header = request.headers.get("authorization")
    except Exception:
        return False
    if not isinstance(header, str):
        return False
    scheme, _, token = header.partition(" ")
    return scheme.lower() == "bearer" and token.startswith(API_KEY_PREFIX)


from state.tool_approvals import resolve_tool_decision

from core.inference.key_exchange import decrypt_api_key
from core.inference.model_ids import model_id_matches, public_model_id
from core.inference.api_monitor import api_monitor
from core.inference.llama_http import nonstreaming_client
from core.inference.tool_call_parser import (
    _strip_function_xml_calls,
    _strip_gemma_wrapperless_calls,
    _strip_glm_calls,
    _strip_mistral_closed_calls,
)
from core.inference.tool_call_parser import TOOL_XML_SIGNALS as _PARSER_TOOL_SIGNALS
from core.inference.passthrough_healing import (
    StreamToolCallHealer,
    heal_gate,
    heal_openai_message,
    heal_openai_message_events,
    nudge_enabled,
    nudge_messages,
    nudge_should_retry,
    response_has_promotable_calls,
)
from core.inference.providers import get_base_url
from core.inference.external_provider import ExternalProviderClient
from core.inference.chat_templates import resolve_effective_chat_template_override
from storage import providers_db
from utils.utils import is_hf_authentication_error, safe_error_detail, log_and_http_error

import io
import base64
import numpy as np
from datetime import date as _date

router = APIRouter()
# Unsloth-only router (not mounted on /v1 OpenAI-compat).
studio_router = APIRouter()


# Packaged desktop runs at tauri://localhost (macOS/Linux) or http://tauri.localhost
# (Windows WebView2); the web build is same-origin ('self'). The `tauri dev` shell,
# however, serves the frontend from the Vite dev origin (http://localhost:5173),
# so the packaged allowlist alone leaves the preview blocked in dev with an
# "ancestor violates frame-ancestors" error. This shell exposes no server resource
# (it only renders postMessage'd HTML in a no-same-origin sandbox), so also allowing
# any localhost/127.0.0.1 dev origin to frame it is safe and unblocks the dev shell.
_ARTIFACT_PREVIEW_FRAME_ANCESTORS = (
    "'self' tauri://localhost http://tauri.localhost http://localhost:* http://127.0.0.1:*"
)
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
            // Leave the sandbox failure contained in the canvas if the
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


async def _authenticate_header_or_query(request: Request, token: Optional[str]) -> str:
    """Resolve the bearer token from the Authorization header or the ``?token=``
    query param (needed for <img src> / <iframe>, which can't send custom
    headers), validate it, and return the subject. Raises 401 when absent."""
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
    return await get_current_subject(creds)


@studio_router.get("/artifact-preview-frame", include_in_schema = False)
async def artifact_preview_frame(allow_network: bool = False):
    """Serve the opaque sandbox shell for client-side HTML canvases.

    No auth token by design: the URL is readable by the untrusted canvas via
    location.href, and this static shell exposes no server resource (frame-ancestors
    plus the sandbox already gate it), so the CSP is chosen from allow_network alone.
    """

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


# Whitespace/escape-tolerant bare-JSON tool-template detector (matches pretty-printed and
# JSON-escaped ``{"name":`` plus the ``"function"`` alias), mirroring the parser's tolerance.
_BARE_JSON_NAME_MARKER_RE = _re.compile(r'\{\s*\\?"(?:name|function)\\?"\s*:')


def _detect_safetensors_features(
    backend,
    chat_template: Optional[str],
    tools = None,
) -> dict:
    """Classify reasoning/tool capabilities via the GGUF classifier so flags
    match across backends. gpt-oss is overridden: Harmony routes reasoning and
    tools through tokenizer channels, not template markup."""
    model_id = getattr(backend, "active_model_name", None)
    feature_template = chat_template
    try:
        from core.inference.chat_template_helpers import _selected_template_strings_from_value
        selected_templates = _selected_template_strings_from_value(chat_template, tools)
        if selected_templates:
            feature_template = selected_templates[0]
    except Exception:
        logger.debug("safetensors_named_template_selection_failed", exc_info = True)
    flags = detect_reasoning_flags(
        feature_template,
        model_identifier = model_id,
        log_source = "safetensors",
    )
    if not flags.get("supports_reasoning"):
        try:
            from core.inference.chat_template_helpers import (
                detect_reasoning_channel_markers_from_template,
            )

            templates = [chat_template]
            models = getattr(backend, "models", None)
            model_info = (
                models.get(model_id, {})
                if isinstance(models, dict) and model_id is not None
                else {}
            )
            if isinstance(model_info, dict):
                templates.extend(
                    (
                        model_info.get("native_chat_template"),
                        (model_info.get("chat_template_info") or {}).get("template"),
                    )
                )
            if any(
                detect_reasoning_channel_markers_from_template(template, tools = tools) is not None
                for template in templates
            ):
                flags["supports_reasoning"] = True
                flags["reasoning_always_on"] = True
                logger.info("safetensors: model always reasons (native channel markers)")
        except Exception:
            logger.debug("safetensors_native_reasoning_marker_check_failed", exc_info = True)
    # Markers any supported parser recognises (template advertises tools but
    # uses none -> drop the pill). Reuse the parser's own signal list so this
    # gate never drifts (a hand-maintained copy lost the DeepSeek variants);
    # ``<arg_key>`` is GLM's unique signal, absent from the shared set. The
    # bare-JSON ``{"name":`` form is matched below with the whitespace/escape-
    # tolerant ``_BARE_JSON_NAME_MARKER_RE`` so pretty-printed or escaped
    # templates are not mis-classified as tool-less.
    _PARSER_MARKERS = (
        *_PARSER_TOOL_SIGNALS,
        "<arg_key>",
    )
    if (
        flags.get("supports_tools")
        and isinstance(feature_template, str)
        and not any(m in feature_template for m in _PARSER_MARKERS)
        and not _BARE_JSON_NAME_MARKER_RE.search(feature_template)
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


def _generation_prompt_opens_think(template: Optional[str]) -> bool:
    """True when rendering the template's generation prompt ends INSIDE an unclosed ``<think>``.

    Distinguishes templates that PREFILL an open ``<think>`` in the assistant generation
    prompt (DeepSeek-R1, QwQ, Qwen3-Thinking) -- where the model emits only the closing
    ``</think>`` and the extractor must start in reasoning mode -- from templates that merely
    render PAST assistant ``<think>...</think>`` history while leaving the generation prompt
    open with no ``<think>`` (e.g. Kimi-K2-Thinking), where the model self-emits its own block
    and the extractor must start in normal mode. Renders a single-user-message probe with the
    same sandbox transformers uses; on any failure returns True, preserving the historical
    always-on prefill for templates that cannot be rendered here.
    """
    if not template:
        return False
    try:
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def _raise_exception(message: str):
            raise RuntimeError(message)

        env = ImmutableSandboxedEnvironment(
            trim_blocks = True,
            lstrip_blocks = True,
            extensions = ["jinja2.ext.loopcontrols"],
        )
        env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii = False)
        env.globals["raise_exception"] = _raise_exception
        rendered = env.from_string(template).render(
            messages = [{"role": "user", "content": "hi"}],
            add_generation_prompt = True,
            bos_token = "",
            eos_token = "",
        )
    except Exception:
        return True
    # ``<think>`` is not a substring of ``</think>`` (the ``/`` breaks it), so the last open
    # tag sitting after the last close tag means the prompt ends inside an open block.
    return rendered.rfind("<think>") > rendered.rfind("</think>")


def _sf_reasoning_prefill_mode(
    features: dict,
    enable_thinking: Optional[bool],
    template: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> bool:
    """Whether a safetensors/MLX generation begins INSIDE an unclosed ``<think>``.

    ``enable_thinking`` templates (Qwen3/GLM) prefill an open ``<think>`` so the model
    emits only the closing ``</think>``, and the extractor must start in reasoning mode.
    Gated on the STANDARD ``<think>``/``</think>`` markers: bespoke channels (gemma's
    ``<|think|>``) never emit ``</think>`` and would swallow the answer, so they and
    gpt-oss and thinking-disabled requests return False. ``enable_thinking`` None
    defaults thinking ON, so a plain request still prefills.
    """
    if features.get("reasoning_style") not in ("enable_thinking", "enable_thinking_effort"):
        return False
    tpl = template or ""
    if "</think>" not in tpl and "<think>" not in tpl:
        return False
    if features.get("reasoning_always_on"):
        # enable_thinking_effort + always-on: the effort mechanism (not the prompt shape) keeps
        # thinking on, so always-on wins over reasoning_effort and we prefill.
        if features.get("reasoning_style") == "enable_thinking_effort":
            return True
        # ``reasoning_always_on`` fires on paired ``<think>...</think>`` anywhere in the
        # template, including markup that only renders PAST assistant history (Kimi-K2-Thinking)
        # while the generation prompt opens none. Prefill only when the generation prompt opens
        # one, else the extractor captures a normal answer as reasoning_content and returns blank.
        return _generation_prompt_opens_think(tpl)
    if not features.get("supports_reasoning"):
        return False
    if enable_thinking is False:
        return False
    # Thinking-off arrives as reasoning_effort "none" on enable_thinking_effort models; honor it
    # so we don't prefill and capture the answer. Plain enable_thinking models ignore effort.
    if features.get("reasoning_style") == "enable_thinking_effort" and reasoning_effort == "none":
        return False
    return True


def _effective_enable_tools(payload) -> Optional[bool]:
    """Resolve `payload.enable_tools` against the process-level tool policy.

    Returns the policy value when set (CLI hard-override from `unsloth run`),
    else the per-request value.
    """
    from state.tool_policy import get_tool_policy

    policy = get_tool_policy()
    return policy if policy is not None else payload.enable_tools


def _explicit_studio_tool_loop_requested(payload) -> bool:
    """True when the request itself asks Unsloth to execute local tools.

    Process-wide CLI policy can default Unsloth's tool loop on for ordinary chat,
    but it must not steal OpenAI-compatible client tools or response_format
    requests from the llama-server passthrough path. A policy of ``False``
    (--disable-tools) vetoes even an explicit ``enable_tools: true`` ask.
    """
    from state.tool_policy import get_tool_policy

    policy = get_tool_policy()
    return policy is not False and (payload.enable_tools is True or bool(payload.mcp_enabled))


def _takes_tool_passthrough(payload, llama_backend) -> bool:
    """True when a GGUF request is forwarded to llama-server verbatim.

    The passthrough sends the caller's own tools, no built-in schema and no nudge, so the counter
    must decide this BEFORE applying the process tool policy: `unsloth run --enable-tools` sets
    that policy without asking for the tool loop, so its catalog would price a prompt never sent.
    """
    supports_tools = getattr(llama_backend, "supports_tools", False)
    if supports_tools and _explicit_studio_tool_loop_requested(payload):
        return False
    # Read defensively: a count request carries no tool_choice, and absent withdraws nothing.
    has_client_contract = (
        bool(payload.tools) and getattr(payload, "tool_choice", None) != "none"
    ) or _has_openai_tool_history(payload.messages)
    supports_passthrough = getattr(llama_backend, "supports_tool_passthrough", supports_tools)
    if supports_passthrough and has_client_contract:
        return True
    return _extract_response_format(payload) is not None


def _passthrough_client_tools(payload):
    """The caller's own tool catalog exactly as the passthrough puts it on the wire.

    ``tool_choice: "none"`` withdraws it, unless tool history needs those schemas to replay.
    /apply-template renders any ``tools`` regardless of tool_choice, so the counter shares the rule.
    """
    if getattr(payload, "tool_choice", None) == "none" and not _has_openai_tool_history(
        payload.messages
    ):
        return None
    return payload.tools or None


def _permission_mode_confirm(payload) -> bool:
    """Effective confirm-gate intent for Unsloth's own local tool loop.

    An explicit confirm_tool_calls (True or False) wins; explicit ask/auto always
    engage the gate (a non-streaming one is then rejected, since it cannot prompt);
    off/full never prompt. An unset mode stays lenient here even though the loop
    defaults it to "auto": a non-streaming request keeps the legacy
    run-without-gate behavior instead of 400ing, so non-streaming clients and
    health checks keep working. Used at the pre-switch guard and the per-backend
    tool paths so a forced tool loop (CLI --enable-tools) still gates streaming.
    """
    if payload.confirm_tool_calls is not None:
        return bool(payload.confirm_tool_calls)
    mode = getattr(payload, "permission_mode", None)
    if mode in ("ask", "auto"):
        return True
    if mode in ("off", "full"):
        return False
    return bool(getattr(payload, "stream", False))


def _confirm_gate_needs_stream(payload) -> bool:
    """Whether Unsloth's local tool-loop confirm gate still requires stream=true.

    The gate can only prompt while streaming, so a non-streaming request that will
    prompt must 400 up front. auto ("Approve for me") only prompts for a call the
    classifier flags, so an auto request whose confirm is derived from the mode
    (not an explicit confirm_tool_calls=true) and whose selectable tools are all
    always-safe (web_search / RAG) never prompts and needs no stream. ask,
    an explicit confirm flag, MCP tools, and an unrestricted or unsafe selection
    still require streaming.
    """
    if not _permission_mode_confirm(payload):
        return False
    if getattr(payload, "permission_mode", None) != "auto":
        return True
    if payload.confirm_tool_calls is True:
        return True
    if getattr(payload, "mcp_enabled", False):
        return True
    enabled = getattr(payload, "enabled_tools", None)
    if enabled is None:
        return True  # omitted enabled_tools resolves to ALL tools (incl. terminal/python)
    if not enabled:
        # An explicit empty selection runs no built-in tool (_select_request_tools
        # skips the loop), so there is nothing to prompt and no stream is needed.
        return False
    from core.inference.tools import is_always_safe_tool

    return not all(is_always_safe_tool(t) for t in enabled)


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
    """Register cancel_event in _CANCEL_REGISTRY for the block's duration.

    Also records the run in state.active_generations so /load and /unload can
    see which chats a reload would interrupt. Both registries share this event,
    so either one cancels down the same per-request path.
    """

    def __init__(
        self,
        event: threading.Event,
        *keys,
        thread_id = None,
        model = None,
        kind = "chat",
    ):
        self.event = event
        self.keys = tuple(k for k in keys if k)
        # kind reaches the swap prompt: embeddings and raw completions have no conversation, so
        # naming them chats would offer to stop something the user never started from a thread.
        self._active = active_generations.ActiveGeneration(
            event, thread_id = thread_id, model = model, kind = kind
        )

    @classmethod
    def for_payload(cls, event: threading.Event, payload, *keys):
        """Track the run against the conversation its request names."""
        return cls(
            event,
            *keys,
            thread_id = getattr(payload, "thread_id", None),
            model = getattr(payload, "model", None),
        )

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
        self._active.__enter__()
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
        self._active.__exit__(*exc)
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


async def _await_disconnect_then_close(request, resp, cancel_event) -> None:
    """Close ``resp`` on client disconnect; sets ``cancel_event`` first so
    the streamer's RemoteProtocolError handler treats it as cancellation.
    Catches aborts the in-loop /cancel check misses during prefill. #5692.
    """
    try:
        while not await request.is_disconnected():
            await asyncio.sleep(0.1)
        cancel_event.set()
        try:
            await resp.aclose()
        except Exception as e:
            logger.debug("Failed to close response on disconnect: %s", e)
    except asyncio.CancelledError:
        return


async def _await_disconnect_then_cancel(request, cancel_event) -> None:
    """Set ``cancel_event`` when a same-task local stream disconnects."""
    try:
        while not await request.is_disconnected():
            await asyncio.sleep(0.1)
        cancel_event.set()
    except asyncio.CancelledError:
        return


def _cancelable_nonstreaming_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        limits = httpx.Limits(max_connections = 1, max_keepalive_connections = 0),
        trust_env = False,
    )


async def _await_cancel_or_disconnect_then_close_client(
    *, cancel_event, request: Optional[Request], client: httpx.AsyncClient
) -> None:
    """Close a dedicated non-streaming upstream client on cancel/disconnect.

    The shared ``nonstreaming_client()`` is pooled, so cancelable generation calls
    use a per-request client. Closing it interrupts a blocked llama-server
    request without affecting unrelated pooled non-streaming calls.
    """
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                break
            if request is not None and await request.is_disconnected():
                if cancel_event is not None:
                    cancel_event.set()
                break
            await asyncio.sleep(0.1)
        try:
            await client.aclose()
        except Exception:
            pass
    except asyncio.CancelledError:
        return


async def _stop_local_disconnect_cancel_watcher(
    watcher, timeout_s: float = _TEARDOWN_TASK_STOP_TIMEOUT_S
) -> None:
    # Bounded: this runs in the stream's finally, so awaiting the watcher outright would let a
    # wedged poll loop hold the response open forever. asyncio.wait neither cancels nor re-raises,
    # and an abandoned watcher owns no resources.
    watcher.cancel()
    done, _pending = await asyncio.wait({watcher}, timeout = timeout_s)
    if not done:
        # _wait_preheader_cancel has no exception handler, so a raise after we stop
        # waiting would surface as "Task exception was never retrieved".
        watcher.add_done_callback(_discard_task_outcome)
        return
    try:
        watcher.result()
    except (asyncio.CancelledError, Exception):
        pass


async def _drain_pending_next_task(task, cancel_event) -> None:
    """Wait for a pending ``asyncio.to_thread(next, gen, ...)`` task to finish
    before its generator is closed.

    On disconnect a ``next(gen)`` call may still run in a worker thread;
    cancelling the awaiting task does NOT stop it, and ``gen.close()`` mid-
    ``next(gen)`` raises ``ValueError: generator already executing``, leaking the
    generator's cleanup. So re-set the cancel flag (the generator polls it) and
    shield the task until the worker returns. No-op when there is no pending task.
    """
    if task is None:
        return
    if cancel_event is not None:
        cancel_event.set()
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if cancel_event is not None:
                cancel_event.set()
            continue
        except Exception:
            break
    if task.done():
        try:
            task.exception()
        except (asyncio.CancelledError, Exception):
            pass


# Centralized local/server tool nudge. Keep render_html guidance gated to turns
# where the canvas tool is actually present in the tool schema; otherwise
# small local models can hallucinate a missing tool call instead of following
# the fenced-HTML fallback prompt.
_TOOL_BASE_NUDGE = (
    "Tools are available when they materially improve the answer. Use an enabled "
    "tool for current facts, calculations, code execution, or canvases when it "
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
    "For HTML, CSS, or JavaScript canvas requests, call render_html once when "
    "it is available with one complete self-contained HTML document in the code "
    "argument. After render_html succeeds, do not call it again in the same "
    "response unless the user asks for changes. Future user requests for new "
    "canvases may call render_html once."
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


# Nudge appended when the RAG knowledge-base tool is active: ground answers in
# the attached documents instead of model memory.
_RAG_GROUNDING_NUDGE = (
    "The user has attached documents to this conversation. Relevant "
    "passages are retrieved and provided to you automatically; base "
    "your answer on them and cite them. You can also call "
    "search_knowledge_base to look for more. Do not answer from "
    "memory when the attached documents are relevant."
)


async def _select_request_tools(
    payload: ChatCompletionRequest, *, tools_on: bool, mcp_allowed: bool
) -> list[dict]:
    """Resolve the tool list for a chat request: built-ins filtered by the
    caller's opt-in (empty when MCP-only), the RAG tool dropped without a
    retrieval scope, then enabled MCP tools appended. An empty result means the
    caller should skip the tool loop, so a model-emitted built-in call can't
    piggy-back on the empty allow-list."""
    from core.inference.tools import ALL_TOOLS, get_enabled_mcp_tools

    if not tools_on:
        # MCP-only request: skip built-ins, leave room for MCP tools.
        tools = []
    elif payload.enabled_tools is not None:
        tools = [t for t in ALL_TOOLS if t["function"]["name"] in payload.enabled_tools]
    else:
        # Copy so the shared module-global tool list can't be mutated by callers.
        tools = list(ALL_TOOLS)
    # Drop the RAG tool without a scope: nothing to search over.
    if not payload.rag_scope:
        tools = [t for t in tools if t["function"]["name"] != "search_knowledge_base"]
    if mcp_allowed:
        tools = tools + await get_enabled_mcp_tools()
    return tools


def _apply_rag_nudge(nudge: str, tools: list[dict], *, rag_scope) -> str:
    """Append the RAG grounding nudge to ``nudge`` when the knowledge-base tool
    is active (search_knowledge_base present and a retrieval scope is set). The
    date is prefixed when the tool nudge is empty (RAG-only tool set). Returns
    ``nudge`` unchanged when RAG isn't active."""
    tool_names = {(t.get("function") or {}).get("name") for t in (tools or [])}
    if "search_knowledge_base" not in tool_names or not rag_scope:
        return nudge
    if not nudge:
        date_line = f"The current date is {_date.today().isoformat()}."
        return date_line + " " + _RAG_GROUNDING_NUDGE
    return nudge + " " + _RAG_GROUNDING_NUDGE


# Strip leaked tool-call markup: every shared-parser format plus the leak shapes
# llama_cpp.py's speculative buffer splits across the visible/DRAIN boundary:
#   1. well-formed `<tool_call>...</tool_call>` / `<function=...>...</function>`
#   2. orphan opening to EOF (close was DRAINED)
#   3. bare orphan close (open was DRAINED)
#   4. tail-only `</parameter>` (outer close truncated by EOS); anchored to
#      `\Z` so mid-text `<parameter>` in user code samples survives.
#   5. Mistral `[TOOL_CALLS]name{json}` / rehearsal `name[ARGS]{json}`: the balanced
#      scan removes the whole call (a non-greedy regex would truncate nested JSON).
# DeepSeek/GLM/Kimi envelopes are covered by the parser's own arms/scans, so a signal
# we parse is never left un-stripped; the DeepSeek opener alternation is the parser's own.
from core.inference.tool_call_parser import _DEEPSEEK_OPEN_RE_SRC as _DS_OPEN_SRC

_TOOL_XML_RE = _re.compile(
    # Arm order/notes: the closed ``<function=...>`` arm runs first and extends
    # to the call's REAL close so a literal ``</function>`` in a value does not
    # leak the tail; the combined arm still catches ``<tool_call>`` and orphan
    # tails. The python_tag arm bounds only on REAL Llama control sentinels
    # (stopping at any ``<|`` truncated on literal ``<|x|>`` tokens in values).
    # The last arms cover DeepSeek envelopes (all opener variants), Kimi section
    # blocks, and bare Kimi calls. Name class ``[\w.\-]`` mirrors the parser.
    # Those three arms carry a call-shaped lookahead (matching the parser's
    # ``_TOOL_ALL_PATS``): a prose answer that merely mentions a marker
    # (``See <|tool_call_begin|> in the docs``) is only stripped when a real
    # call actually follows the marker, or the marker is a bare fragment at EOF.
    r'<function(?:=[\w.\-]+|\s+name="[\w.\-]+")>(?:(?!<function(?:=[\w.\-]+|\s+name="[\w.\-]+")>).)*</function>'
    r'|<(?:tool_call|function(?:=[\w.\-]+|\s+name="[\w.\-]+"))>.*?(?:</(?:tool_call|function)>|\Z)'
    r"|<\|tool_call>.*?(?:<tool_call\|>|\Z)"
    r"|</(?:tool_call|function)>"
    r"|<tool_call\|>"
    r"|<\|python_tag\|>(?:[^<]|<(?!\|(?:eot_id|eom_id|python_tag|start_header_id|end_header_id|begin_of_text|finetune_right_pad_id)\|))*"
    r"|\[/TOOL_CALLS\]"
    # Truncated canonical array (closing ``]`` lost to EOS): the balanced scan cannot remove
    # it, so strip its tail here.
    r"|\[TOOL_CALLS\]\s*\[.*\Z"
    # Named / v11 forms and bare rehearsal; arms aligned with the parser regexes.
    r"|\[TOOL_CALLS\]\s*[\w-]+(?:\[CALL_ID\][\w-]+)?(?:\[ARGS\])?\s*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|.*?\Z)"
    # Rehearsal: balanced/truncated body or bare marker at EOS only (prose ``foo[ARGS]``
    # survives); NAME captured as ``reh`` for the inactive-name display gate.
    r"|(?<!\[CALL_ID\])\b(?P<reh>[\w-]+)\[ARGS\]\s*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\{.*\Z|\Z)"
    # DeepSeek envelopes (all opener variants), Kimi section blocks, and bare Kimi calls;
    # each arm carries a call-shaped lookahead so prose merely mentioning a marker survives.
    r"|"
    + _DS_OPEN_SRC
    + r"(?=\s*(?:<｜tool▁call▁begin｜>|function)|\s*$).*?(?:<｜tool▁calls▁end｜>|\Z)"
    r"|<\|tool_calls_section_begin\|>(?=\s*<\|tool_call_begin\|>|\s*$).*?(?:<\|tool_calls_section_end\|>|\Z)"
    r"|<\|tool_call_begin\|>(?=\s*[A-Za-z_][\w.\-]*:\d|\s*$).*?(?:<\|tool_call_end\|>|\Z)"
    # ``</param>`` is the attribute-form alias of ``</parameter>`` (the parser accepts
    # both); strip a tail-only orphan close of either spelling.
    r"|</(?:parameter|param)>\s*\Z",
    _re.DOTALL,
)

# Closed-only variant for segments before the last think block: the ``\Z``-anchored arms
# would treat a segment boundary as EOS and strip prose ``foo[ARGS]``.
_TOOL_XML_CLOSED_RE = _re.compile(
    r"<(?:tool_call|function=[\w-]+)>.*?</(?:tool_call|function)>"
    r"|<\|tool_call>.*?<tool_call\|>"
    r"|</(?:tool_call|function)>"
    r"|<tool_call\|>"
    r"|\[/TOOL_CALLS\]",
    _re.DOTALL,
)


def _gemma_strip_gate(tools) -> set:
    """Enabled tool NAMES gating the wrapper-less Gemma strip (mirrors the
    parser/loop gate: only an enabled ``call:foo{...}`` is a call). With NO tools
    enabled this returns an EMPTY set, not ``None``: every ``call:NAME{...}`` is
    then prose, and ``None`` would strip-all and delete a legitimate answer."""
    names = {
        (t.get("function") or {}).get("name")
        for t in (tools or [])
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    }
    names.discard(None)
    return names


def _display_tool_name_gate(active_tools):
    """Active tool NAMES for gating the rehearsal display strip, or None when no tools
    are enabled. ``None`` keeps the legacy strip-all behavior, mirroring the loop gate:
    a bare ``NAME[ARGS]`` is a call only when NAME is active; without a tool list every
    identifier stays ambiguous, so strip."""
    names = {
        (t.get("function") or {}).get("name")
        for t in (active_tools or [])
        if isinstance(t, dict) and isinstance(t.get("function"), dict)
    }
    names.discard(None)
    return names or None


def _strip_tool_xml_for_display(
    text: str,
    *,
    auto_heal_tool_calls: bool,
    enabled_tool_names: Optional[set] = None,
) -> str:
    """Apply route-level XML leak cleanup only when Auto-Heal is enabled.

    Mirrors the parser-side segment scan: balanced strips first (Mistral, gated Gemma
    wrapper-less, GLM real-close, guarded function-XML close at each call's REAL terminator
    so literal markup inside a value is data), then the ``_TOOL_XML_RE`` arms cover the
    DeepSeek / Kimi / orphan forms. ``<think>`` blocks are preserved verbatim and the
    ``\\Z``-anchored tail arms run only on the last segment (prose ``foo[ARGS]`` before a
    block survives). ``enabled_tool_names`` (when not None) gates the ambiguous bare-rehearsal
    ``NAME[ARGS]{...}`` and wrapper-less Gemma ``call:NAME{...}`` strips on the active tool
    list; an inactive NAME is prose and is kept. The ``[TOOL_CALLS]`` control-token arms strip
    unconditionally regardless of NAME."""
    if not auto_heal_tool_calls:
        return text
    from core.tool_healing import _strip_bracket_tag_calls, strip_outside_think

    def _keep_inactive_rehearsal(m) -> str:
        # Only the bare-rehearsal arm captures ``reh``; with a tool list an inactive
        # NAME[ARGS]{...} is prose -- keep it.
        if enabled_tool_names is not None:
            name = m.groupdict().get("reh")
            if name is not None and name not in enabled_tool_names:
                return m.group(0)
        return ""

    def _strip_segment(seg: str, is_last: bool) -> str:
        # Scan strips close at each call's REAL terminator (a literal ``</function>`` or a
        # nested marker quoted inside a value cannot truncate the strip); the regex arms below
        # cover the attribute form and the DeepSeek / Kimi / orphan families.
        seg = _strip_mistral_closed_calls(seg)
        seg = _strip_bracket_tag_calls(seg, enabled_tool_names = enabled_tool_names)
        if is_last:
            seg = _strip_gemma_wrapperless_calls(seg, enabled_tool_names)
        seg = _strip_glm_calls(seg, final = is_last)
        seg = _strip_function_xml_calls(seg, final = is_last)
        if is_last:
            return _TOOL_XML_RE.sub(_keep_inactive_rehearsal, seg)
        return _TOOL_XML_CLOSED_RE.sub("", seg)

    return strip_outside_think(text, _strip_segment)


def _strip_tool_xml(text: str, enabled_tool_names: Optional[set] = None) -> str:
    # Mistral balanced-brace pre-strip (kept explicit so the regression guards see it), then
    # the shared think-aware display strip -- the one raw _TOOL_XML_RE.sub lives inside
    # _strip_tool_xml_for_display, so every route cleanup site shares it. ``enabled_tool_names``
    # gates the Gemma wrapper-less strip; ``None`` strips every closed call.
    text = _strip_mistral_closed_calls(text)
    return _strip_tool_xml_for_display(
        text, auto_heal_tool_calls = True, enabled_tool_names = enabled_tool_names
    )


logger = get_logger(__name__)


def _monitor_content_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                ptype = part.get("type")
                if ptype in ("text", "input_text", "output_text"):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                elif ptype in ("image_url", "input_image", "image"):
                    parts.append("[image]")
                else:
                    parts.append(f"[{ptype or 'content'}]")
            else:
                ptype = getattr(part, "type", None)
                text = getattr(part, "text", None)
                if isinstance(text, str):
                    parts.append(text)
                elif ptype in ("image_url", "input_image", "image"):
                    parts.append("[image]")
                elif ptype:
                    parts.append(f"[{ptype}]")
        return "\n".join(parts)
    return str(content)


def _monitor_prompt_from_messages(messages) -> str:
    lines: list[str] = []
    for msg in messages or []:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        tool_calls = (
            msg.get("tool_calls") if isinstance(msg, dict) else getattr(msg, "tool_calls", None)
        )
        text = _monitor_content_text(content)
        if tool_calls and not text:
            text = "[tool calls]"
        if text:
            lines.append(f"{role or 'message'}: {text}")
    return "\n\n".join(lines)


def _monitor_usage(
    monitor_id: Optional[str],
    usage: Optional[dict],
    context_length = None,
    *,
    timings: Optional[dict] = None,
    stop_reason: Optional[str] = None,
):
    if usage:
        api_monitor.set_usage(
            monitor_id,
            prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens"),
            completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens"),
            total_tokens = usage.get("total_tokens"),
            context_length = context_length,
        )
    tok_per_sec = prompt_ms = None
    if isinstance(timings, dict):
        tok_per_sec = timings.get("predicted_per_second")
        prompt_ms = timings.get("prompt_ms")
    if tok_per_sec is not None or prompt_ms is not None or stop_reason is not None:
        api_monitor.set_perf(
            monitor_id,
            tok_per_sec = tok_per_sec,
            prompt_ms = prompt_ms,
            stop_reason = stop_reason,
        )


def _monitor_call_text(name: Any, arguments: Any = None) -> str:
    call_name = str(name or "tool")
    if arguments is None or arguments == "":
        return f"Tool call: {call_name}"
    if not isinstance(arguments, str):
        args_text = json.dumps(arguments, default = str)
    else:
        args_text = arguments
    if len(args_text) > 500:
        args_text = args_text[:497] + "..."
    return f"Tool call: {call_name}({args_text})"


def _monitor_tool_calls_text(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return ""
    parts: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        fn = tool_call.get("function") or {}
        if not isinstance(fn, dict):
            fn = {}
        name = fn.get("name") or tool_call.get("name") or "tool"
        args = fn.get("arguments")
        if args is None:
            args = tool_call.get("arguments")
        parts.append(_monitor_call_text(name, args))
    return "\n".join(parts)


def _monitor_openai_chunk(
    monitor_id: Optional[str],
    data: dict,
    context_length = None,
):
    if not monitor_id:
        return
    # Defensive: ignore malformed shapes so the helper never raises into the
    # streaming generator and aborts the user's response.
    choices = data.get("choices")
    finish_reason = None
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict) and choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])
                break
    timings = data.get("timings")
    _monitor_usage(
        monitor_id,
        data.get("usage"),
        context_length,
        timings = timings if isinstance(timings, dict) else None,
        stop_reason = finish_reason,
    )
    if not isinstance(choices, list) or not choices:
        return
    reply_parts: list[tuple[int, str]] = []
    for idx, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}
        if isinstance(delta, dict) and delta.get("reasoning_content"):
            api_monitor.mark_first_token(monitor_id)
        content = delta.get("content") if isinstance(delta, dict) else None
        if content:
            api_monitor.append_reply(monitor_id, content)
            continue
        if isinstance(delta, dict):
            tool_text = _monitor_tool_calls_text(delta.get("tool_calls"))
            if tool_text:
                api_monitor.append_reply(monitor_id, tool_text)
                continue
        if isinstance(choice.get("text"), str):
            reply_parts.append((idx, choice["text"]))
        elif isinstance(message, dict):
            text = message.get("content")
            if isinstance(text, str):
                reply_parts.append((idx, text))
            else:
                tool_text = _monitor_tool_calls_text(message.get("tool_calls"))
                if tool_text:
                    reply_parts.append((idx, tool_text))
    if not reply_parts:
        return
    if len(choices) == 1:
        api_monitor.append_reply(monitor_id, reply_parts[0][1], stamp_first_token = False)
        return
    api_monitor.append_reply(
        monitor_id,
        "\n\n".join(f"Choice {idx + 1}:\n{text}" for idx, text in reply_parts),
        stamp_first_token = False,
    )


def _monitor_openai_error_message(data: dict) -> Optional[str]:
    error = data.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
        return json.dumps(error)
    if isinstance(error, str) and error:
        return error
    return None


def _is_openai_sse_done(raw_line: str) -> bool:
    """Whether the line is the terminal `data: [DONE]` frame.

    Deliberately independent of the monitor: framing the client sees must not
    change just because recording is off.
    """
    # SSE spec allows `data:value` and `data: value`; accept both.
    if not raw_line.startswith("data:"):
        return False
    return raw_line[5:].lstrip() == "[DONE]"


def _monitor_openai_sse_line(
    monitor_id: Optional[str],
    raw_line: str,
    context_length = None,
) -> Optional[str]:
    if not monitor_id:
        return None
    if not raw_line.startswith("data:"):
        return None
    data_str = raw_line[5:].lstrip()
    if data_str == "[DONE]":
        api_monitor.finish(monitor_id)
        return "done"
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        error_message = _monitor_openai_error_message(data)
        if error_message:
            api_monitor.fail(monitor_id, error_message)
            return "error"
        _monitor_openai_chunk(monitor_id, data, context_length)
    return None


def _monitor_openai_sse_event(
    monitor_id: Optional[str],
    event: bytes,
    context_length = None,
) -> None:
    for line in event.decode("utf-8", errors = "ignore").splitlines():
        _monitor_openai_sse_line(monitor_id, line.strip(), context_length)


def _monitor_anthropic_usage(
    monitor_id: Optional[str],
    usage: Optional[dict],
    context_length = None,
) -> None:
    if not usage:
        return
    _monitor_usage(
        monitor_id,
        {
            "prompt_tokens": usage.get("input_tokens") or usage.get("prompt_tokens"),
            "completion_tokens": usage.get("output_tokens") or usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        },
        context_length,
    )


_ANTHROPIC_MONITOR_TOOL_BLOCKS: dict[str, dict[int, bool]] = {}


def _monitor_anthropic_index(data: dict) -> int:
    try:
        return int(data.get("index") or 0)
    except (TypeError, ValueError):
        return 0


def _monitor_anthropic_payload(
    monitor_id: Optional[str],
    data: dict,
    context_length = None,
) -> Optional[str]:
    if not monitor_id or not isinstance(data, dict):
        return None
    event_type = data.get("type")
    if event_type == "message_start":
        message = data.get("message") or {}
        if isinstance(message, dict):
            _monitor_anthropic_usage(monitor_id, message.get("usage"), context_length)
        return None
    if event_type == "content_block_start":
        content_block = data.get("content_block") or {}
        if isinstance(content_block, dict) and content_block.get("type") == "tool_use":
            index = _monitor_anthropic_index(data)
            _ANTHROPIC_MONITOR_TOOL_BLOCKS.setdefault(monitor_id, {})[index] = False
            api_monitor.append_reply(monitor_id, _monitor_call_text(content_block.get("name")))
        return None
    if event_type == "content_block_delta":
        delta = data.get("delta") or {}
        text = delta.get("text") if isinstance(delta, dict) else None
        if isinstance(delta, dict) and delta.get("type") == "thinking_delta":
            api_monitor.mark_first_token(monitor_id)
        if isinstance(text, str) and text:
            api_monitor.append_reply(monitor_id, text)
        elif isinstance(delta, dict) and delta.get("type") == "input_json_delta":
            index = _monitor_anthropic_index(data)
            tool_blocks = _ANTHROPIC_MONITOR_TOOL_BLOCKS.get(monitor_id) or {}
            if index in tool_blocks:
                if not tool_blocks[index]:
                    api_monitor.append_reply(monitor_id, "\nInput: ")
                    tool_blocks[index] = True
                partial_json = delta.get("partial_json")
                if isinstance(partial_json, str) and partial_json:
                    api_monitor.append_reply(monitor_id, partial_json)
        return None
    if event_type == "content_block_stop":
        index = _monitor_anthropic_index(data)
        tool_blocks = _ANTHROPIC_MONITOR_TOOL_BLOCKS.get(monitor_id)
        if tool_blocks is not None:
            tool_blocks.pop(index, None)
            if not tool_blocks:
                _ANTHROPIC_MONITOR_TOOL_BLOCKS.pop(monitor_id, None)
        return None
    if event_type == "message_delta":
        delta = data.get("delta")
        if isinstance(delta, dict) and delta.get("stop_reason"):
            api_monitor.set_perf(monitor_id, stop_reason = str(delta["stop_reason"]))
        _monitor_anthropic_usage(monitor_id, data.get("usage"), context_length)
        return None
    if event_type == "error":
        error = data.get("error") or {}
        if isinstance(error, dict):
            message = error.get("message") or json.dumps(error, default = str)
        else:
            message = str(error)
        api_monitor.fail(monitor_id, message)
        return "error"
    return None


def _monitor_anthropic_sse_line(
    monitor_id: Optional[str],
    raw_line: str,
    context_length = None,
) -> Optional[str]:
    if not monitor_id or not raw_line.startswith("data:"):
        return None
    data_str = raw_line[5:].lstrip()
    try:
        data = json.loads(data_str)
    except json.JSONDecodeError:
        return None
    return _monitor_anthropic_payload(monitor_id, data, context_length)


def _monitor_anthropic_content_blocks(content: Any) -> str:
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block.get("type") == "tool_use":
            parts.append(_monitor_call_text(block.get("name"), block.get("input")))
    return "".join(parts)


def _monitor_anthropic_json_response(
    response,
    monitor_id: Optional[str],
    context_length = None,
) -> None:
    if not monitor_id:
        return
    body = getattr(response, "body", b"")
    try:
        data = json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)
    except Exception:
        api_monitor.finish(monitor_id)
        return
    if not isinstance(data, dict):
        api_monitor.finish(monitor_id)
        return
    text = _monitor_anthropic_content_blocks(data.get("content"))
    if text:
        api_monitor.set_reply(monitor_id, text)
    if data.get("stop_reason"):
        api_monitor.set_perf(monitor_id, stop_reason = str(data["stop_reason"]))
    _monitor_anthropic_usage(monitor_id, data.get("usage"), context_length)
    api_monitor.finish(monitor_id)


def _monitor_anthropic_response(
    response,
    monitor_id,
    context_length = None,
    cancel_event = None,
):
    if not monitor_id:
        return response
    body_iterator = getattr(response, "body_iterator", None)
    if body_iterator is None:
        _monitor_anthropic_json_response(response, monitor_id, context_length)
        return response

    async def _monitored_body():
        terminal = False
        try:
            async for chunk in body_iterator:
                text = (
                    chunk.decode("utf-8", errors = "ignore")
                    if isinstance(chunk, (bytes, bytearray))
                    else str(chunk)
                )
                for line in text.splitlines():
                    if (
                        _monitor_anthropic_sse_line(
                            monitor_id,
                            line.strip(),
                            context_length,
                        )
                        == "error"
                    ):
                        terminal = True
                yield chunk
            if not terminal:
                api_monitor.finish(
                    monitor_id,
                    "cancelled"
                    if cancel_event is not None and cancel_event.is_set()
                    else "completed",
                )
        except asyncio.CancelledError:
            if cancel_event is not None:
                cancel_event.set()
            api_monitor.finish(monitor_id, "cancelled")
            raise
        except Exception as exc:
            api_monitor.fail(monitor_id, _friendly_error(exc))
            raise
        finally:
            _ANTHROPIC_MONITOR_TOOL_BLOCKS.pop(monitor_id, None)

    response.body_iterator = _monitored_body()
    return response


def _peek_inference_backend() -> Any:
    """The orchestrator if one already exists, else None. Never constructs one.

    Constructing reaches get_default_models() -> get_device(), so during the warm a caller
    that only describes what is loaded would block uvicorn on the torch import to answer
    "nothing". A patched module getter still wins: that is this module's injection seam.
    """
    from core.inference import orchestrator as _orch

    if get_inference_backend is not _orch.get_inference_backend:
        return get_inference_backend()
    return _orch.peek_inference_backend()


def _monitor_context_length() -> Optional[int]:
    llama_backend = get_llama_cpp_backend()
    if getattr(llama_backend, "is_loaded", False):
        context_length = _positive_int_or_none(getattr(llama_backend, "context_length", None))
        if context_length is not None:
            return context_length
    # Peek, not the constructing getter: called inline from the OpenAI, Responses and
    # Anthropic monitor paths, and no orchestrator already means nothing is loaded.
    backend = _peek_inference_backend()
    if backend is None or not backend.active_model_name:
        return None
    models = getattr(backend, "models", {}) or {}
    model_info = models.get(backend.active_model_name, {}) if isinstance(models, dict) else {}
    context_length = _positive_int_or_none(model_info.get("context_length"))
    if context_length is not None:
        return context_length
    for candidate in (
        getattr(backend, "context_length", None),
        getattr(backend, "max_seq_length", None),
    ):
        context_length = _positive_int_or_none(candidate)
        if context_length is not None:
            return context_length
    return None


def _lifecycle_model_label(model: Optional[str], variant: Optional[str] = None) -> str:
    """A path-free ``repo`` / ``repo:QUANT`` label for a monitor lifecycle row."""
    clean = public_model_id(model) or model or "model"
    return f"{clean}:{variant}" if variant and ":" not in clean else clean


def _close_load_event(
    entry_id: Optional[str], model: Optional[str], variant: Optional[str]
) -> None:
    """Close a monitor load row, relabelled with the id the load resolved: the row
    opened on the request's model_path, which may be an HF snapshot dir."""
    api_monitor.relabel(entry_id, _lifecycle_model_label(model, variant))
    api_monitor.finish(entry_id)


# Requests proxied straight to llama-server without admission (/v1/completions)
# still occupy decode slots; counted here so the monitor's readout includes them.
_direct_llama_inflight = 0
_direct_llama_inflight_lock = threading.Lock()


def _direct_llama_request_started() -> None:
    global _direct_llama_inflight
    with _direct_llama_inflight_lock:
        _direct_llama_inflight += 1


def _direct_llama_request_finished() -> None:
    global _direct_llama_inflight
    with _direct_llama_inflight_lock:
        _direct_llama_inflight = max(0, _direct_llama_inflight - 1)


def _monitor_queue_state() -> Optional[dict]:
    """Live slot/queue occupancy of the loaded llama-server, for the API monitor."""
    # Disabled admission tracks nothing: its queues stay at the default capacity
    # of 1 and never take leases, so a snapshot would misreport a multi-slot server.
    if not llama_admission_config_from_env().enabled:
        return None
    llama_backend = get_llama_cpp_backend()
    if not getattr(llama_backend, "is_loaded", False) or getattr(
        llama_backend, "is_diffusion", False
    ):
        return None
    direct = _direct_llama_inflight
    snapshot = peek_llama_admission_snapshot(
        str(getattr(llama_backend, "base_url", "llama-server"))
    )
    if snapshot is not None:
        active = min(snapshot.capacity, snapshot.active + direct)
        return {
            "capacity": snapshot.capacity,
            "active": active,
            "queued": snapshot.queued,
            "free": max(0, snapshot.capacity - active),
        }
    capacity = _positive_int_or_none(getattr(llama_backend, "effective_parallel_slots", None)) or 1
    active = min(capacity, direct)
    return {"capacity": capacity, "active": active, "queued": 0, "free": capacity - active}


def _monitor_active_model() -> Optional[str]:
    """The loaded model as a client-facing id, quant included when known.

    Cleaned like /v1/models: rendered in the settings UI and served over the public
    --secure tunnel, so it must never be the on-disk load path.
    """
    llama_backend = get_llama_cpp_backend()
    if getattr(llama_backend, "is_loaded", False):
        model_id = _llama_public_model_id(llama_backend)
        variant = getattr(llama_backend, "hf_variant", None)
        if model_id and variant and ":" not in model_id:
            return f"{model_id}:{variant}"
        return model_id
    # Peek: the monitor overlay is on by default and polls this read-only, so building
    # the singleton to answer "nothing loaded" would import torch on a warm-disabled host.
    backend = _peek_inference_backend()
    if backend is None:
        return None
    return public_model_id(backend.active_model_name) or backend.active_model_name


def _validate_native_gguf_companion(
    companion_path: str | None,
    gguf_path: str | None,
    label: str,
    *,
    allow_mtp_subdir: bool = False,
    mtp_search_root: str | Path | None = None,
) -> None:
    """Reject a companion GGUF (mmproj / MTP drafter) that a native-lease load
    would otherwise hand to llama-server: must be a regular file (no symlink
    escaping the leased directory) in a permitted location."""
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
        if not native_gguf_companion_parent_allowed(
            companion,
            gguf,
            allow_mtp_subdir = allow_mtp_subdir,
            mtp_search_root = mtp_search_root,
        ):
            location = (
                "beside the selected GGUF or in its MTP directory"
                if allow_mtp_subdir
                else "next to the selected GGUF"
            )
            raise HTTPException(
                status_code = 400,
                detail = f"Native {label} must live {location}.",
            )
    except OSError as exc:
        raise HTTPException(
            status_code = 400,
            detail = f"Native {label} is no longer accessible.",
        ) from exc


def _loaded_is_local_model(
    llama_backend: LlamaCppBackend, native_grant_backed: bool, model_id: str | None
) -> bool:
    """Provenance of the running model, preferring what the load recorded.

    Falls back to the filesystem for a server started before the flag existed.
    """
    if native_grant_backed:
        return True
    stored = getattr(llama_backend, "_is_local_model", None)
    if stored is not None:
        return bool(stored)
    return bool(model_id and is_local_path(model_id))


def _validate_native_mtp_drafter(
    companion_path: str | None,
    gguf_path: str | None,
    *,
    mtp_search_root: str | Path | None = None,
) -> None:
    """Validate an MTP drafter for a native load, every shard of it.

    llama-server opens the sibling shards of a split drafter implicitly, so
    checking only the launch path would let a later shard be a symlink out of
    the permitted directory without ever facing the native rules.
    """
    if not companion_path or not gguf_path:
        return
    shards, _ = colocated_split_shards(Path(companion_path))
    for shard in shards or [Path(companion_path)]:
        _validate_native_gguf_companion(
            str(shard),
            gguf_path,
            "MTP drafter",
            allow_mtp_subdir = True,
            mtp_search_root = mtp_search_root,
        )


def _native_gguf_companion_usable(
    companion_path: str | None,
    gguf_path: str | None,
    *,
    mtp_search_root: str | Path | None = None,
    log_rejection: bool = False,
) -> bool:
    """Whether a native load would accept this MTP drafter, as a predicate for
    reload dedup. Same rules, so the two cannot disagree."""
    try:
        _validate_native_mtp_drafter(companion_path, gguf_path, mtp_search_root = mtp_search_root)
    except HTTPException as exc:
        if log_rejection:
            logger.warning("Dropping MTP drafter for native load: %s", exc.detail)
        return False
    return True


def _should_strip_split_mode(request: LoadRequest, backend_extra: Optional[list[str]]) -> bool:
    """Whether an inherited --split-mode (and its coupled --tensor-split) should
    be stripped on reload.

    The binary Tensor Parallelism toggle can't carry --split-mode's row/none/
    layer modes, so only strip when the toggle overrides it: tensor being turned
    on, or the inherited mode is tensor (toggle turning it off). Non-tensor modes
    survive. A manual per-GPU ratio is handled by _should_strip_tensor_split,
    which strips only --tensor-split so the inherited mode is kept. Shared by the
    inheritance strip and the already-loaded stale check so they agree on what
    reload would do.
    """
    fields_set = getattr(request, "model_fields_set", set())
    return "tensor_parallel" in fields_set and (
        request.tensor_parallel or resolve_tensor_parallel(backend_extra, False)
    )


def _should_strip_tensor_split(request: LoadRequest) -> bool:
    """Whether an inherited --tensor-split alone should be stripped on reload.

    Manual explicit offload (gpu_layers >= 0) owns the per-GPU split: with a ratio
    it emits its own --tensor-split (an inherited one, appended last, would
    override it), and with the ratio cleared it wants llama.cpp's default
    free-VRAM split. Either way an inherited --tensor-split must go, else the
    cleared case silently keeps the stale ratio while status reports None.
    Unlike _should_strip_split_mode this leaves --split-mode untouched, so a
    user's row/none/layer mode survives a Studio split-ratio edit. When the
    Tensor Parallelism toggle IS overriding the mode, _should_strip_split_mode
    (called alongside this at every site) strips --split-mode anyway.
    """
    return (
        getattr(request, "gpu_memory_mode", "auto") == "manual"
        and getattr(request, "gpu_layers", -1) >= 0
    )


def _carry_preserved_tensor_intent(
    *, preserved: bool, same_model: bool, explicit_drop: bool
) -> bool:
    """Carry a preserved multi-GPU layer fallback forward only for a reload of the
    SAME loaded model that doesn't explicitly drop tensor intent, so a fitting model
    isn't collapsed to one GPU on a ctx-only change -- but an unrelated model switch
    (without /unload) or an explicit tensor-off doesn't inherit it (#6659)."""
    return preserved and same_model and not explicit_drop


def _is_explicit_tensor_drop(request: LoadRequest) -> bool:
    """True only when the request explicitly selects a non-tensor --split-mode (e.g.
    layer/row/none), a deliberate departure from a preserved tensor->layer fallback.

    A bare tensor_parallel field is NOT a drop: the Unsloth UI always sends it and echoes
    the /load response's resolved value back, so after a fallback every reload carries
    tensor_parallel=false even though the user never changed it -- treating that as a drop
    would collapse the preserved multi-GPU placement on the next ctx/settings reload. An
    empty clear is not a drop either (a fallback always stores --split-mode layer, never a
    tensor split mode, so a clear never wipes tensor intent), nor is an unrelated extra
    (--top-k) or inherit (None). tensor_parallel=true / --split-mode tensor re-engage
    tensor. Shared by the already-loaded dedup and the load carry-forward (#6659)."""
    override = parse_split_mode_override(request.llama_extra_args)
    return override is not None and override.strip().lower() != "tensor"


def _llama_runtime_fields(llama_backend: LlamaCppBackend) -> dict:
    """Runtime state shared by load, dedupe, and status; duplicates echo active settings."""
    fields = {
        name: getattr(llama_backend, name, getattr(llama_backend, f"_{name}", None))
        for name in _InferenceRuntimeFields.model_fields
        if hasattr(llama_backend, name) or hasattr(llama_backend, f"_{name}")
    }
    fields.update(
        speculative_type = llama_backend.requested_spec_mode,
        requested_parallel_slots = (
            None if llama_backend.is_diffusion else llama_backend.requested_parallel_slots
        ),
        parallel_slots = (
            None if llama_backend.is_diffusion else llama_backend.effective_parallel_slots
        ),
    )
    unresolved = (
        set(_InferenceRuntimeFields.model_fields) - fields.keys() - {"requires_trust_remote_code"}
    )
    if unresolved:
        raise AttributeError(
            f"GGUF backend is missing runtime response fields: {sorted(unresolved)}"
        )
    return fields


def _gguf_load_response(
    llama_backend: LlamaCppBackend,
    status: str,
    model: str,
    *,
    display_name: Optional[str] = None,
    is_local_model: bool,
    inference_identifier: Optional[str] = None,
) -> LoadResponse:
    return LoadResponse(
        status = status,
        model = model,
        display_name = display_name or model,
        is_lora = False,
        is_gguf = True,
        is_local_model = is_local_model,
        inference = load_inference_config(
            inference_identifier or llama_backend.model_identifier or model
        ),
        **_llama_runtime_fields(llama_backend),
    )


def _gguf_request_intent(
    source: GgufLoadIntent,
    request: LoadRequest,
    *,
    chat_template_override: Optional[str],
    extra_args: Optional[list[str]],
    gpu_ids: Optional[list[int]],
    n_parallel: int,
    **changes,
) -> GgufLoadIntent:
    settings = {
        name: getattr(request, name)
        for name in vars(source)
        if hasattr(request, name) and (name != "hf_token" or source.hf_repo)
    }
    settings.update(
        n_ctx = request.max_seq_length,
        chat_template_override = chat_template_override,
        extra_args = extra_args,
        gpu_ids = gpu_ids,
        n_parallel = n_parallel,
    )
    settings.update(changes)
    return replace(source, **settings)


def _mtp_draft_for_path(
    gguf_path: Optional[str],
    native_grant_backed: bool,
    *,
    log_native_fallback: bool = False,
) -> Optional[str]:
    if not gguf_path:
        return None
    root = _local_gguf_companion_search_root(gguf_path, gguf_path)
    rejected = False
    accept = None
    if native_grant_backed:

        def accept(candidate):
            nonlocal rejected
            usable = _native_gguf_companion_usable(
                candidate,
                gguf_path,
                mtp_search_root = root,
                log_rejection = log_native_fallback,
            )
            rejected |= not usable
            return usable

    detected = detect_mtp_file(
        gguf_path,
        search_root = root,
        accept = accept,
    )
    if log_native_fallback and rejected and detected:
        logger.info("Using MTP subdirectory drafter for native load: %s", detected)
    return detected


def _active_gguf_intent(
    request: LoadRequest,
    llama_backend: LlamaCppBackend,
    *,
    model_identifier: str,
    chat_template_override: Optional[str],
    n_parallel: int,
    native_grant_backed: bool,
) -> GgufLoadIntent:
    backend_extra = list(llama_backend.extra_args or ())
    effective_extra = (
        request.llama_extra_args
        if request.llama_extra_args is not None
        else strip_shadowing_flags(
            backend_extra,
            strip_split_mode = _should_strip_split_mode(request, backend_extra),
            strip_tensor_split = _should_strip_tensor_split(request),
            strip_offload = request.gpu_memory_mode == "manual",
        )
    )
    source = llama_backend.last_load_intent or GgufLoadIntent(
        model_identifier = model_identifier,
        gguf_path = None if llama_backend.hf_repo else llama_backend.gguf_path,
        hf_repo = llama_backend.hf_repo,
        hf_variant = llama_backend.hf_variant,
    )
    return _gguf_request_intent(
        source,
        request,
        model_identifier = model_identifier,
        # A repo or directory variant has not been resolved to a file yet. Do
        # not inherit the resident file or source matching would compare that
        # file with itself and ignore a requested quant switch.
        gguf_path = source.gguf_path if model_identifier.lower().endswith(".gguf") else None,
        hf_variant = request.gguf_variant or source.hf_variant,
        chat_template_override = chat_template_override,
        extra_args = effective_extra,
        gpu_ids = request.gpu_ids,
        n_parallel = n_parallel,
        preserve_multi_gpu_on_layer = (
            llama_backend.layer_preserves_tensor_intent and not _is_explicit_tensor_drop(request)
        ),
        mtp_draft_path = _mtp_draft_for_path(llama_backend.gguf_path, native_grant_backed),
        compare_mtp_draft = True,
        extra_args_inherited = request.llama_extra_args is None,
    )


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


# Serializes opt-in auto-switch loads so two requests can't race a swap. One
# lock per running loop, since a module-level asyncio.Lock binds to a single
# loop and breaks multi-loop runners (e.g. pytest's per-test loops on pre-3.10).
_auto_switch_locks: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_auto_switch_locks_guard = threading.Lock()


def _auto_switch_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    # WeakKeyDictionary mutation isn't thread-safe; guard get-or-create so two
    # loops on different threads can't race it.
    with _auto_switch_locks_guard:
        lock = _auto_switch_locks.get(loop)
        if lock is None:
            lock = _auto_switch_locks[loop] = asyncio.Lock()
        return lock


# Process-wide gate so a swap on another event loop in this process can't race
# this one for the single model slot: the asyncio lock above is per loop, but the
# backend slot and _load_model_impl are process-wide. threading.Lock so it serializes
# across loops/threads; released from the loop thread (Lock allows cross-thread release).
_auto_switch_process_lock = threading.Lock()


async def _acquire_swap_gate() -> None:
    # Non-blocking first for the common single-loop case; otherwise poll off a
    # short sleep rather than awaiting to_thread(acquire). A cancelled to_thread
    # (client disconnect mid-wait) leaves its worker thread still acquiring, so the
    # gate gets taken but the finally that releases it never runs -- deadlocking
    # later swaps. Polling keeps the wait off this loop AND cancellation-safe: a
    # cancel lands during the sleep, when the gate is not held.
    while not _auto_switch_process_lock.acquire(blocking = False):
        await asyncio.sleep(0.02)


# Counts auto-switch requests queued to load each (target, variant). They are not
# generating, so the drain wait below excludes them from the active inference count.
_auto_switch_waiters: dict[tuple[str, str], int] = {}
_auto_switch_waiters_guard = threading.Lock()


def _switch_key(override_id: str, variant: Optional[str]) -> tuple[str, str]:
    return (override_id.lower(), (variant or "").lower())


def _note_switch_waiter(key: tuple[str, str], delta: int) -> None:
    with _auto_switch_waiters_guard:
        n = _auto_switch_waiters.get(key, 0) + delta
        if n > 0:
            _auto_switch_waiters[key] = n
        else:
            _auto_switch_waiters.pop(key, None)


def _switch_waiter_count() -> int:
    with _auto_switch_waiters_guard:
        return sum(max(0, count) for count in _auto_switch_waiters.values())


async def _wait_for_model_switch_idle(
    *,
    current_request_counted: bool,
    cancel_pending: bool = False,
    timeout_s: Optional[float] = None,
) -> None:
    """Wait until a model replacement cannot interrupt active inference.

    The caller holds ``inference_lifecycle_gate``, which prevents new inference
    from starting while existing requests drain. Auto-switch requests that have
    resolved their targets are scheduler waiters, not active generations, so
    exclude them to avoid a queue deadlock.

    ``cancel_pending`` is set by a forced swap that has NOT cancelled yet: the
    registered generations are the ones it is about to stop, so waiting on them
    would wait out exactly what the force exists to end. Excluding them lets the
    drain finish ahead of the cancel, which keeps every check that can still
    reject the swap in front of the destructive step. Recomputed each poll (not
    snapshotted) so a generation that ends on its own stops being discounted and
    the remaining, non-cancellable requests are still waited out.

    ``timeout_s`` bounds the wait and returns rather than raising. Only the
    post-cancel drains pass it: what they wait on may never observe its cancel
    (TTS on the subprocess backend has no observer), and they hold the lifecycle
    gate, so an unbounded wait pins every load and unload behind one
    uninterruptible generation. Expiring there just proceeds, which is what they
    do anyway once drained. Pre-cancel drains stay unbounded -- the swap can
    still be refused, so they must not shorten the protection they provide.
    """
    from core.inference.llama_keepwarm import other_inference_request_count

    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    while True:
        queued_switches = _switch_waiter_count()
        if current_request_counted and queued_switches > 0:
            queued_switches -= 1
        active_others = other_inference_request_count(
            current_request_counted = current_request_counted,
            include_pending = False,
        )
        if cancel_pending:
            active_others -= min(active_others, active_generations.count())
        if active_others <= queued_switches:
            return
        if deadline is not None and time.monotonic() >= deadline:
            logger.warning(
                "model_switch_drain_timed_out",
                extra = {
                    "event": "inference.switch_drain_timeout",
                    "remaining": active_others - queued_switches,
                },
            )
            return
        await asyncio.sleep(0.02)


def _llama_public_model_id(llama_backend, fallback: Optional[str] = None) -> Optional[str]:
    """The id to report for the loaded GGUF in API responses: the advertised repo
    id from an auto-switch load, else the cleaned public id, never the on-disk
    .gguf path (see core.inference.model_ids.public_model_id)."""
    return (
        getattr(llama_backend, "_openai_advertised_id", None)
        or public_model_id(getattr(llama_backend, "model_identifier", None))
        or public_model_id(fallback)
        or fallback
    )


def _llama_status_model_ids(llama_backend) -> "tuple[Optional[str], Optional[str]]":
    """The ``(active_model, model_identifier)`` pair ``/api/inference/status`` publishes
    for a loaded GGUF. A native-lease load reports only the display label, never the
    leased on-disk path."""
    model_id = getattr(llama_backend, "model_identifier", None)
    native_grant_backed = getattr(llama_backend, "_native_grant_backed", False)
    display_model_id = getattr(
        llama_backend, "_native_display_label", None
    ) or display_label_for_native_path(model_id)
    if (
        native_grant_backed
        and model_id
        and display_model_id == model_id
        and os.path.isabs(model_id)
    ):
        display_model_id = os.path.basename(model_id)
    elif not native_grant_backed and display_model_id == model_id:
        # No label registered, so report the clean public id, not the snapshot's sha.
        display_model_id = _llama_public_model_id(llama_backend) or display_model_id
    return display_model_id, (None if native_grant_backed else model_id)


def _llama_status_checkpoint_id(llama_backend) -> Optional[str]:
    """The exact string a Studio client holds as ``params.checkpoint`` for the loaded
    GGUF: ``status.model_identifier ?? status.active_model``. Built from the same pair the
    status handler returns so the two cannot drift."""
    display_model_id, model_identifier = _llama_status_model_ids(llama_backend)
    return display_model_id if model_identifier is None else model_identifier


_DISABLE_OPENAI_AUTO_SWITCH_SCOPE_KEY = "_unsloth_disable_openai_auto_switch"
# Sentinel a raw-body endpoint passes when the request omits ``model``: it must
# only restore an idle-freed model, never run the resolver (so a downloaded GGUF
# literally named "default" can't be swapped to). The NUL keeps it off any index.
_RELOAD_ONLY_MODEL = "\x00reload-only"
# One cold scan is worth paying to avoid answering a named model with another,
# bounded so a pathological install cannot hang the request behind it.
_COLD_INDEX_WAIT_S = 10.0


def _switch_model_for_payload(payload) -> str:
    # A pydantic request fills an omitted ``model`` with "default"; only an
    # explicitly set model may switch, else reload-only so a GGUF named "default"
    # is never matched (mirrors the raw-body sentinel path).
    return payload.model if "model" in payload.model_fields_set else _RELOAD_ONLY_MODEL


def _target_is_vision(load_path: str) -> bool:
    # A local GGUF's vision capability is its companion mmproj, a filesystem check
    # (no model load). Matches the loaded backend's is_vision, so rejecting a swap
    # here can't differ from the post-load guard. Thread the ambient HF token so the
    # probe keeps the capability-probe invariant (the resolver only yields local
    # paths, where the token is unused, but the rule requires it regardless).
    from utils.models.model_config import is_vision_model
    try:
        # Deliberately unguarded: the resolver only yields local paths, so this returns
        # from the mmproj filesystem branch without touching the hub. A reachability
        # probe here would add seconds per request and prevent nothing.
        return bool(is_vision_model(load_path, hf_token = os.environ.get("HF_TOKEN")))
    except Exception as exc:
        # Detection failure: don't block the swap, let the load decide.
        logger.debug("auto-switch: vision probe failed for %s: %s", load_path, exc)
        return True


def _messages_have_image(messages) -> bool:
    return any(
        isinstance(m.content, list) and any(isinstance(p, ImageContentPart) for p in m.content)
        for m in messages
    )


def _request_has_image(payload) -> bool:
    if getattr(payload, "image_base64", None):
        return True
    return _messages_have_image(payload.messages)


def _anthropic_request_has_image(payload) -> bool:
    # Mirror anthropic_messages_to_openai: an Anthropic image block carries
    # ``type == "image"`` (typed AnthropicImageBlock or a raw dict).
    for msg in getattr(payload, "messages", None) or []:
        content = getattr(msg, "content", None)
        if not isinstance(content, list):
            continue
        for block in content:
            bt = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
            if bt == "image":
                return True
    return False


def disable_openai_auto_switch_for_request(scope) -> None:
    """Opt a request out of OpenAI auto-switch. The public preview route uses this:
    it always serves its pinned checkpoint, so a caller-supplied model must never
    swap the loaded model."""
    if isinstance(scope, dict):
        scope[_DISABLE_OPENAI_AUTO_SWITCH_SCOPE_KEY] = True


def _automatic_model_load_may_run() -> bool:
    """True when a request can trigger an automatic load: either resolver-based
    auto-switch is on, or a standalone idle TTL can reload an idle-freed model. The
    validate-before-switch guards key off this so an invalid request never loads."""
    from utils.openai_auto_switch_settings import (
        get_openai_auto_switch_enabled,
        get_auto_unload_idle_seconds,
    )
    return get_openai_auto_switch_enabled() or get_auto_unload_idle_seconds() > 0


def _no_model_loaded_detail(base: str) -> str:
    """Append a pointer to the opt-in auto-switch toggle to a "no model loaded"
    error, but only when it's off. Auto-switch (default off) cold-loads a
    requested downloaded GGUF, so an off toggle is the usual reason a request
    naming a listed model still 400/503s; surface the fix. With it on the name
    simply didn't resolve to a local GGUF, so the hint would mislead and is omitted."""
    from utils.openai_auto_switch_settings import get_openai_auto_switch_enabled

    if get_openai_auto_switch_enabled():
        return base
    return base + (
        " Or enable Model auto-switch (Settings > API) to load a requested model automatically."
    )


# Cap on ids listed by a "not downloaded" error, so it stays readable in a terminal.
_MAX_LISTED_AVAILABLE_MODELS = 8


def _raw_body_model(body) -> Optional[str]:
    """The ``model`` a raw-body endpoint was given, else None (same value
    :func:`_auto_switch_from_request_body` fed the switch hook)."""
    return body.get("model") if isinstance(body, dict) else None


async def _available_model_ids() -> list[str]:
    """Sorted ids a /v1 request may name, from the catalog ``GET /v1/models``
    serves, so an error and the listing can't disagree."""
    return sorted(
        mid
        for mid in (m.get("id") for m in await _openai_catalog_objects())
        if isinstance(mid, str) and mid
    )


def _format_available_models(ids: list[str]) -> str:
    if not ids:
        return ""
    shown = ", ".join(ids[:_MAX_LISTED_AVAILABLE_MODELS])
    extra = len(ids) - _MAX_LISTED_AVAILABLE_MODELS
    return f"{shown} and {extra} more" if extra > 0 else shown


async def _unavailable_model_message(requested_model: str) -> str:
    """Why a named model can't serve this request, and what can.

    Auto-switch only loads downloaded GGUFs, so a request naming a real model
    usually fails because it is not on this machine, which /inference/load cannot
    fix; say what is actually wrong.
    """
    from core.inference.local_model_resolver import (
        MISS_VARIANT_NOT_FOUND,
        describe_local_miss,
    )

    reason, variants = await asyncio.to_thread(describe_local_miss, requested_model)
    if reason == MISS_VARIANT_NOT_FOUND:
        # Repo downloaded, only the quant missing: sibling quants beat the catalog.
        base_id, _, wanted = requested_model.strip().rpartition(":")
        return (
            f"The model '{base_id}' is downloaded, but the quant '{wanted}' is not. "
            f"Available quants: {', '.join(variants)}."
        )
    available = _format_available_models(await _available_model_ids())
    if not available:
        return (
            f"The model '{requested_model}' is not downloaded on this server, and no "
            "models are downloaded yet. Download one in Unsloth Studio."
        )
    return (
        f"The model '{requested_model}' is not downloaded on this server. "
        f"Available models: {available}. Download more in Unsloth Studio, "
        "or list them with GET /v1/models."
    )


async def _no_model_loaded_error(
    base: str, requested_model: Optional[str], fastapi_request: Optional[Request], *, status: int
):
    """``(status, detail)`` for the /v1 sites that fail because nothing is loaded.

    Changes only the case the generic text gets wrong (auto-switch on, a model
    named, that name resolving to nothing local, so the switch silently did
    nothing) into a 404 model_not_found. Everything else keeps ``status`` and the
    :func:`_no_model_loaded_detail` text verbatim.
    """
    from utils.openai_auto_switch_settings import get_openai_auto_switch_enabled
    from core.inference.local_model_resolver import resolve_local_gguf

    named = (
        requested_model
        if isinstance(requested_model, str)
        and requested_model.strip()
        and requested_model != _RELOAD_ONLY_MODEL
        else None
    )
    if named is None or not get_openai_auto_switch_enabled():
        return status, _no_model_loaded_detail(base)
    try:
        if await asyncio.to_thread(_loaded_satisfies, named):
            # Resident but on a backend this endpoint can't use, so "not downloaded" is false.
            return status, _no_model_loaded_detail(base)
        if await asyncio.to_thread(resolve_local_gguf, named) is not None:
            # Resolvable but unloaded: the switch failed, which the generic text covers.
            return status, _no_model_loaded_detail(base)
        message = await _unavailable_model_message(named)
    except Exception as exc:
        # The diagnosis is a nicety; never let it turn a 4xx into a 500.
        logger.debug("no-model-loaded diagnosis failed for %r: %s", named, exc)
        return status, _no_model_loaded_detail(base)
    path = getattr(getattr(fastapi_request, "url", None), "path", None)
    if not isinstance(path, str):
        # No request in hand: let the global /v1/* handler pick the envelope.
        return 404, message
    return 404, error_body_for_path(
        path,
        message,
        status = 404,
        code = "model_not_found",
        param = "model",
    )


def _auto_download_hf_token(fastapi_request: Optional[Request]) -> Optional[str]:
    """The token to fetch with: only one the caller sent themselves.

    Never the server's ambient token, and never the OpenAI bearer key. The repo is
    named by whoever holds an API key, so borrowing the owner's Hub identity would
    let that key pull the owner's private repos and publish them in /v1/models.
    """
    from hub.dependencies import HUB_HF_TOKEN_HEADER, HUB_HF_TOKEN_MAX_LENGTH

    headers = getattr(fastapi_request, "headers", None)
    if headers is None:
        return None
    supplied = (headers.get(HUB_HF_TOKEN_HEADER) or "").strip()
    if supplied and len(supplied) <= HUB_HF_TOKEN_MAX_LENGTH:
        return supplied
    return None


async def _maybe_auto_download_model(
    requested_model: str,
    fastapi_request: Optional[Request],
    *,
    require_vision: bool = False,
    current_subject: Optional[str] = None,
) -> None:
    """Opt-in: start fetching a named GGUF this server doesn't have.

    Raises to stop the request while the model is downloading or cannot be fetched.
    Off by default, and never fires on a name not shaped like a Hub repo, so an
    unknown id like "gpt-4" still falls through to the resident model.
    """
    from utils.openai_auto_switch_settings import get_openai_auto_download_enabled
    from core.inference.openai_auto_download import is_downloadable_ref, maybe_auto_download

    if not requested_model or not get_openai_auto_download_enabled():
        return
    if not is_downloadable_ref(requested_model):
        return
    # An Ollama-style tag (":latest") names no quant, so the resolver misses a servable model.
    if await asyncio.to_thread(_loaded_satisfies, requested_model):
        return
    try:
        refusal = await maybe_auto_download(
            requested_model,
            hf_token = _auto_download_hf_token(fastapi_request),
            require_vision = require_vision,
            subject = current_subject,
            # These endpoints also serve Studio's chat on a JWT, so only mark real API traffic.
            via_api_key = _request_used_api_key(fastapi_request),
        )
    except Exception as exc:
        # Never turn a servable request into a 500 over the download attempt.
        logger.warning("auto-download failed for %r: %s", requested_model, exc)
        return
    if refusal is None:
        return
    path = getattr(getattr(fastapi_request, "url", None), "path", None)
    detail = (
        error_body_for_path(
            path,
            refusal.message,
            status = refusal.status,
            code = refusal.code,
            param = "model",
        )
        if isinstance(path, str)
        else refusal.message
    )
    _record_refused_request(fastapi_request, requested_model, refusal, current_subject)
    raise HTTPException(
        status_code = refusal.status,
        detail = detail,
        headers = ({"Retry-After": str(refusal.retry_after)} if refusal.retry_after else None),
    )


def _record_refused_request(
    fastapi_request: Optional[Request],
    requested_model: str,
    refusal: Any,
    current_subject: Optional[str],
) -> None:
    """Log the refused call itself, not just the download it is waiting on.

    The refusal replaces the request, so the handler's own ``api_monitor.start``
    never runs. Only the caller that dispatched a download gets a row from
    ``record_lifecycle``; anyone refused while it runs left no trace at all, and a
    download some other caller started carries their attribution, so an API-key
    client waiting on it never opened the overlay and read as Studio's own traffic.
    """
    state = getattr(fastapi_request, "state", None)
    if getattr(state, "skip_api_monitor", False):
        return
    path = getattr(getattr(fastapi_request, "url", None), "path", None)
    entry_id = api_monitor.start(
        endpoint = path if isinstance(path, str) else "/v1",
        method = str(getattr(fastapi_request, "method", "") or "POST"),
        model = requested_model,
        prompt = "",
        subject = current_subject,
        via_api_key = _request_used_api_key(fastapi_request),
    )
    api_monitor.fail(entry_id, refusal.message)


def _loaded_satisfies(requested: str) -> bool:
    """Whether what is serving right now actually answers to *requested*.

    A bare ``org/model`` is satisfied by any loaded quant of that repo; an explicit
    ``:QUANT`` must match the loaded one.
    """
    from core.inference.openai_auto_download import looks_like_quant, split_model_ref

    base, variant = split_model_ref(requested)
    llama_backend = get_llama_cpp_backend()
    if getattr(llama_backend, "is_loaded", False):
        candidates = [
            candidate
            for candidate in (
                getattr(llama_backend, "model_identifier", None),
                getattr(llama_backend, "_openai_advertised_id", None),
                _llama_public_model_id(llama_backend),
            )
            if candidate
        ]
        if not _matches_any(base, candidates):
            return False
        if not looks_like_quant(variant):
            # An Ollama-style tag (":latest", ":8b") names no file, so the repo is enough.
            return True
        return (getattr(llama_backend, "hf_variant", None) or "").lower() == variant.lower()
    active = getattr(get_inference_backend(), "active_model_name", None)
    if not active:
        return False
    # Only llama.cpp carries a quant identity, so this backend can only match on the repo.
    if looks_like_quant(variant):
        return False
    return _matches_any(base, [active, public_model_id(active)])


def _raise_still_indexing(requested_model: str, fastapi_request) -> None:
    """Refuse a name we cannot yet place, rather than answer it with another model."""
    path = getattr(getattr(fastapi_request, "url", None), "path", None)
    message = (
        f"This server is still indexing its local models, so it cannot confirm "
        f"'{requested_model}' yet. Retry shortly."
    )
    raise HTTPException(
        status_code = 503,
        detail = (
            error_body_for_path(path, message, status = 503, code = "model_indexing")
            if isinstance(path, str)
            else message
        ),
        headers = {"Retry-After": "5"},
    )


def _matches_any(requested: str, candidates) -> bool:
    """Whether *requested* names any of *candidates*.

    A repo alias is case-insensitive, a filesystem path is not: lowercasing both
    made /srv/models/foo.gguf and /srv/models/Foo.gguf the same weights.
    """
    lowered = requested.strip().lower()
    for candidate in candidates:
        if not candidate:
            continue
        if _looks_like_local_path(requested) or _looks_like_local_path(candidate):
            if _norm_path(requested) == _norm_path(candidate):
                return True
            continue
        if lowered == str(candidate).strip().lower():
            return True
    return False


def _looks_like_local_path(value: str) -> bool:
    """A filesystem path rather than a repo id, so case matters."""
    text = str(value)
    return text.startswith("/") or text.startswith("~") or ":\\" in text or "\\" in text


def _norm_path(value: str) -> str:
    """Compare-ready path. normcase, not lower: on a case-sensitive filesystem
    /srv/models/Foo and /srv/models/foo are different models."""
    import os

    # normcase after, not before: on Windows it folds case *and* rewrites "/" to a
    # backslash, leaving the descendant checks below comparing against a path with none.
    return os.path.normcase(str(value)).replace("\\", "/").rstrip("/")


def _resident_quant_is(variant: Optional[str]) -> bool:
    """Whether the loaded GGUF is that exact quant."""
    resident = getattr(get_llama_cpp_backend(), "hf_variant", None) or ""
    return bool(variant) and resident.lower() == variant.strip().lower()


def _resolves_to_resident(load_path: Optional[str], *, llama_only: bool = False) -> bool:
    """Whether a resolved on-disk path is what is already loaded.

    ``llama_only`` drops the Transformers backend: only llama.cpp carries a quant
    identity, so a Transformers model active from a directory that also holds GGUF
    exports would otherwise answer a request for one of those quants.
    """
    if not load_path:
        return False
    target = _norm_path(load_path)
    llama_backend = get_llama_cpp_backend()
    for candidate in (
        getattr(llama_backend, "gguf_path", None)
        if getattr(llama_backend, "is_loaded", False)
        else None,
        getattr(llama_backend, "model_identifier", None)
        if getattr(llama_backend, "is_loaded", False)
        else None,
        None if llama_only else getattr(get_inference_backend(), "active_model_name", None),
    ):
        if not candidate:
            continue
        current = _norm_path(candidate)
        if current == target:
            return True
        if current.startswith(f"{target}/"):
            # A model directory holding the weights loaded from it. Nested entries
            # (/models/A alongside /models/A/sub/B) matched too, so a request for A was
            # answered with B. The innermost indexed model owns the file; with none
            # indexed there is no nesting to tell apart, so keep matching.
            owner = _innermost_indexed_owner(current)
            if owner is None or owner == target:
                return True
            continue
        if target.startswith(f"{current}/"):
            return True
    return False


def _innermost_indexed_owner(path: str) -> Optional[str]:
    """Longest catalog-listed model path containing *path*, or None if none does."""
    best = None
    for info in _CATALOG_CACHE["models"] or ():
        listed = getattr(info, "path", None)
        if not listed:
            continue
        normalized = _norm_path(listed)
        if path == normalized or path.startswith(f"{normalized}/"):
            if best is None or len(normalized) > len(best):
                best = normalized
    return best


async def _reject_unservable_model(
    requested_model: Optional[str], fastapi_request: Optional[Request]
) -> None:
    """Refuse rather than answer a named model with a different one.

    Only for a reference this server can tell was meant for it: an explicit GGUF
    quant, or a model that is actually here. A namespace decides nothing either way
    (``vendor/model`` is how LiteLLM and OpenRouter name every provider, and a
    standalone GGUF is advertised without one), so a slashless id that resolves
    locally is still a concrete reference. Only runs while something is serving:
    with nothing loaded, :func:`_no_model_loaded_error` already says the right thing.
    """
    from core.inference.openai_auto_download import looks_like_quant, split_model_ref

    if (
        not isinstance(requested_model, str)
        or not requested_model.strip()
        or requested_model == _RELOAD_ONLY_MODEL
    ):
        return
    base, variant = split_model_ref(requested_model)
    quantified = looks_like_quant(variant)
    from core.inference.local_model_resolver import (
        index_is_built,
        recently_downloaded,
        resolve_local_gguf,
        warm_index_soon,
    )
    from utils.openai_auto_switch_settings import get_openai_auto_switch_enabled

    still_indexing = False
    try:
        if await asyncio.to_thread(_loaded_satisfies, requested_model):
            return
        if not (
            get_llama_cpp_backend().is_loaded
            or getattr(await asyncio.to_thread(get_inference_backend), "active_model_name", None)
        ):
            return
        # Refresh in the background and read the index as-is: scanning here would stall the
        # request, and a cold index only costs evidence (the gate below fails safe without it).
        if index_is_built():
            warm_index_soon()
            resolved = resolve_local_gguf(requested_model, allow_scan = False)
        else:
            # Nothing cached to reason from yet, and falling through would answer a
            # named model with the resident one. Pay the scan once, off the loop and
            # bounded, rather than read "not scanned yet" as "not here".
            try:
                resolved = await asyncio.wait_for(
                    asyncio.to_thread(resolve_local_gguf, requested_model),
                    _COLD_INDEX_WAIT_S,
                )
            except (TimeoutError, asyncio.TimeoutError):
                # Still scanning, so nothing is known about this name: say "not yet"
                # rather than guess and put the resident model behind it.
                warm_index_soon()
                still_indexing = True
                resolved = None
        # A manual load stores the on-disk path the resolver advertises under an alias,
        # so match on the path too. Quants of one repo share a directory, so the path
        # alone cannot tell them apart: without the variant check an explicit :Q8_0
        # would be answered by a resident Q4_K_M.
        # Off-loop: reads the Transformers singleton, and the llama.cpp short-circuits above
        # skip the offloaded reads, so on a restart this built the singleton on the loop.
        if (
            resolved is not None
            and await asyncio.to_thread(_resolves_to_resident, resolved[0], llama_only = quantified)
            and (not quantified or _resident_quant_is(variant))
        ):
            return
        downloaded = resolved is not None
        # /v1/models may have advertised this id off its own scan while the index is cold.
        advertised = _advertised_local_path(base)
        if (
            advertised is not None
            and await asyncio.to_thread(_resolves_to_resident, advertised, llama_only = quantified)
            and (not quantified or _resident_quant_is(variant))
        ):
            return
        # The exact ref may miss on the quant alone, so ask about the repo too.
        here = (
            downloaded
            or advertised is not None
            # Just landed, so no scan has indexed it yet and neither of the above sees it.
            or recently_downloaded(base)
            or (variant is not None and resolve_local_gguf(base, allow_scan = False) is not None)
        )
        switchable = downloaded and get_openai_auto_switch_enabled()
    except HTTPException:
        # A refusal decided above is the answer, not a failure to decide: without this
        # the handler below logs it and falls through to the resident model.
        raise
    except Exception as exc:
        # Can't verify: an explicit quant still proves intent, so refuse; let anything else by.
        logger.debug("unservable-model check failed for %r: %s", requested_model, exc)
        if not quantified:
            return
        downloaded = here = switchable = False
    if still_indexing:
        _raise_still_indexing(requested_model, fastapi_request)
    if not (quantified or here):
        return
    if switchable:
        # On disk and switching allowed, so the swap failed: the resident model is wrong weights.
        status_code, code = 503, "model_switch_failed"
        message = (
            f"The model '{requested_model}' is downloaded, but this server could not "
            "switch to it. Retry shortly, or load it in Unsloth Studio."
        )
    elif downloaded:
        status_code, code = 404, "model_not_found"
        message = (
            f"The model '{requested_model}' is downloaded but not loaded, and "
            "'Switch model by request' is off, so this server can only serve the "
            "loaded model. Turn it on in Unsloth Studio under Settings > API."
        )
    else:
        status_code, code = 404, "model_not_found"
        try:
            message = await _unavailable_model_message(requested_model)
        except Exception as exc:
            # Only the wording is uncertain; the mismatch is already established.
            logger.debug("unavailable-model diagnosis failed for %r: %s", requested_model, exc)
            message = f"The model '{requested_model}' is not the model this server is serving."
    path = getattr(getattr(fastapi_request, "url", None), "path", None)
    raise HTTPException(
        status_code = status_code,
        detail = (
            error_body_for_path(path, message, status = status_code, code = code, param = "model")
            if isinstance(path, str)
            else message
        ),
        headers = {"Retry-After": "5"} if status_code == 503 else None,
    )


async def _maybe_auto_switch_model(
    requested_model: Optional[str],
    fastapi_request: Request,
    current_subject: str,
    *,
    require_vision: bool = False,
) -> None:
    """Load a downloaded local GGUF named by an OpenAI request when auto-switch is on.

    No-op unless enabled and ``requested_model`` resolves to a downloaded local
    model different from the loaded one. Unknown names fall through (drop-in
    compat); a miss only reaches the network when auto-download is also on, and
    even then only for ``namespace/name`` ids. ``require_vision`` rejects a swap
    to a text-only target before it runs, so an image request can't evict the
    resident vision model only to 400 afterwards.
    """
    from utils.openai_auto_switch_settings import (
        get_openai_auto_switch_enabled,
        get_auto_unload_idle_seconds,
        get_model_override,
        model_override_load_kwargs,
    )
    from core.inference.local_model_resolver import resolve_local_gguf
    from core.inference.llama_keepwarm import (
        get_last_unloaded_model,
        inference_lifecycle_gate,
    )

    # Treat a non-string model (e.g. {"model": 123} on a raw-body endpoint) as
    # absent so it falls through instead of raising in the membership checks below.
    if not isinstance(requested_model, str) or not requested_model:
        return
    # The public preview route opts out so a caller cannot switch away from the
    # pinned preview checkpoint it just loaded.
    scope = getattr(fastapi_request, "scope", None)
    if isinstance(scope, dict) and scope.get(_DISABLE_OPENAI_AUTO_SWITCH_SCOPE_KEY):
        return
    auto_switch_on = get_openai_auto_switch_enabled()
    # The reload-stash path also runs when idle-unload is active on its own (a
    # standalone UNSLOTH_MODEL_IDLE_TTL with auto-switch off), so a model the idle
    # loop freed is restored on the next request. The resolver-based switch still
    # requires the auto-switch toggle.
    if not auto_switch_on and get_auto_unload_idle_seconds() <= 0:
        # No switching to do, but a named model must still not be answered by another.
        await _reject_unservable_model(requested_model, fastapi_request)
        return

    async def _resolve_and_switch() -> None:
        # Off the loop: a cold-cache rebuild walks several model dirs + HF caches.
        # With auto-switch off (or an omitted-model reload-only request), skip the
        # resolve so only the reload-stash path runs and no name is ever matched.
        reload_only = requested_model == _RELOAD_ONLY_MODEL
        resolved = (
            await asyncio.to_thread(resolve_local_gguf, requested_model)
            if auto_switch_on and not reload_only
            else None
        )
        if resolved is None:
            # Not on disk. Opt-in: fetch in the background and ask the caller to retry.
            if auto_switch_on and not reload_only:
                await _maybe_auto_download_model(
                    requested_model,
                    fastapi_request,
                    require_vision = require_vision,
                    current_subject = current_subject,
                )
            # Idle-unload may have freed the model; reload exactly what it freed
            # (path + quant + advertised id) so an alias/unknown name stays servable
            # and keeps the override keyed by the advertised id, not the load path.
            last = get_last_unloaded_model()
            # A non-GGUF (Unsloth/Transformers) model loaded after the idle-unload
            # leaves the GGUF slot empty but is the live model, so don't resurrect
            # the stale GGUF over it (that load would tear the active model down).
            if (
                not last
                or get_llama_cpp_backend().is_loaded
                or getattr(
                    await asyncio.to_thread(get_inference_backend), "active_model_name", None
                )
            ):
                return
            if len(last) == 3:
                target_id, variant, override_id = last
            else:  # pre-3-tuple stash: fall back to the path as the override key
                target_id, variant = last
                override_id = target_id
        else:
            # load_path is a concrete local path (never the bare repo id), so /load
            # takes the local branch and cannot trigger a download. override_id is the
            # advertised repo id, the launch-override key and the public model id.
            target_id, variant, override_id = resolved
        backend = get_llama_cpp_backend()
        # A bare model id (no :VARIANT) is satisfied by any loaded quant of that
        # repo, so it never reloads a different local quant that already serves it.
        from core.inference.openai_auto_download import looks_like_quant, split_model_ref

        # A tag that names no quant (":latest", ":8b") means the repo, as
        # _loaded_satisfies and the resolver read it. Treating it as a quant tears down
        # a serving Q8 to load the preferred Q4 for a request either satisfies.
        _, _requested_variant = split_model_ref(requested_model)
        bare = not looks_like_quant(_requested_variant)

        def _already_serving() -> bool:
            # Match against both the concrete load path and the advertised repo id,
            # so a model loaded manually by repo id (identifier = repo id) and one
            # loaded by auto-switch (identifier = path, advertised = repo id) both
            # count as already serving rather than triggering a needless reswap.
            if not backend.is_loaded or not backend.model_identifier:
                return False
            loaded_keys = {backend.model_identifier.lower()}
            advertised = getattr(backend, "_openai_advertised_id", None)
            if advertised:
                loaded_keys.add(advertised.lower())
            if loaded_keys.isdisjoint({target_id.lower(), override_id.lower()}):
                return False
            if bare:
                return True
            if variant:
                loaded_variant = (getattr(backend, "hf_variant", None) or "").lower()
                return loaded_variant == variant.lower()
            return True

        def _record_serving_alias() -> None:
            # When an advertised alias already resolves to the loaded model (e.g. a
            # model loaded by local path, requested by its repo/LM Studio id), record
            # the alias as the public id so /v1/models and responses report it (and
            # mark it loaded) instead of the path-derived basename. Resolver branch
            # only: the reload-stash override_id can be the bare path, not a repo id.
            # Lock-free is safe here: an in-flight request blocks any concurrent swap
            # (single-slot busy guard), so the loaded model can't change under this.
            if resolved is None or not override_id:
                return
            b = get_llama_cpp_backend()
            if getattr(b, "_openai_advertised_id", None) != override_id:
                b._openai_advertised_id = override_id

        if _already_serving():
            _record_serving_alias()
            return
        # An image/audio request naming a different text-only GGUF would load it
        # here and only 400 below, evicting the working model. Reject before the
        # swap. Only the resolver branch (an explicit new target); the reload-stash
        # path just restores the model the request was already using. Both vision and
        # audio input come from a companion mmproj (a filesystem probe) -- run it off
        # the loop, like the resolver above.
        if (
            require_vision
            and resolved is not None
            and not await asyncio.to_thread(_target_is_vision, target_id)
        ):
            raise HTTPException(
                status_code = 400,
                detail = openai_error_body(
                    "The requested model does not support the image or audio input in this request.",
                    status = 400,
                    code = "invalid_value",
                    param = "model",
                ),
            )
        key = _switch_key(override_id, variant)
        _note_switch_waiter(key, 1)
        waiter_noted = True
        try:
            async with _auto_switch_lock():
                # The asyncio lock is per loop; add a process-wide gate so a swap on
                # another loop in this process can't race the single slot.
                await _acquire_swap_gate()
                try:
                    # Hold the keep-warm gate across the swap so no new inference can
                    # start on the model while it is being torn down and replaced.
                    async with inference_lifecycle_gate():
                        if _already_serving():
                            _record_serving_alias()
                            return
                        # Apply the saved launch config so an API swap loads as the picker
                        # would. Order: variant-qualified keys before bare ids, and the
                        # load path before the advertised id, since the settings UI keys
                        # local rows by that path while override_id is a derived alias, so
                        # reading the alias first let an older entry shadow a fresh save. A
                        # cached repo has no path entry and resolves on the second try; an
                        # early build keyed a loose .gguf by its filename label, so
                        # "<path>:LABEL" is read too, after the bare path used today.
                        file_variant = None
                        if not variant and target_id.lower().endswith(".gguf"):
                            from hub.utils.gguf import extract_quant_label
                            file_variant = extract_quant_label(os.path.basename(target_id))
                        override = {}
                        for override_key in (
                            f"{target_id}:{variant}" if variant else None,
                            f"{override_id}:{variant}" if variant else None,
                            target_id,
                            f"{target_id}:{file_variant}" if file_variant else None,
                            override_id,
                        ):
                            if not override_key:
                                continue
                            override = get_model_override(override_key)
                            if override:
                                break
                        load_kwargs = {"model_path": target_id, "gguf_variant": variant}
                        load_kwargs.update(
                            model_override_load_kwargs(
                                override,
                                # Set for every GGUF the resolver returns; the reload
                                # stash carries the quant it froze.
                                is_gguf = bool(variant) or target_id.lower().endswith(".gguf"),
                            )
                        )
                        saved_gpu_ids = load_kwargs.get("gpu_ids")
                        if saved_gpu_ids and not await _override_gpu_ids_still_resolve(
                            saved_gpu_ids
                        ):
                            # Stale pin (GPU removed, another host): drop the one dead
                            # field rather than 400 the whole load.
                            load_kwargs.pop("gpu_ids", None)
                            logger.warning(
                                "Dropping saved gpu_ids %s for %s: not available here.",
                                saved_gpu_ids,
                                override_id,
                            )
                        # Reuse the load impl so its dedup, tensor fallback, and threading
                        # apply. Call the impl directly: we already hold the lifecycle gate
                        # the /load route would otherwise take, so the route would deadlock.
                        try:
                            await _load_model_impl(
                                LoadRequest(**load_kwargs),
                                fastapi_request,
                                current_subject,
                                current_request_counted = True,
                            )
                        except HTTPException as exc:
                            # The pre-flight check cannot mirror every loader gpu_ids rule,
                            # and a stale pin must never block a request, so retry without it.
                            if not (
                                exc.status_code == 400
                                and load_kwargs.get("gpu_ids")
                                and "gpu" in str(exc.detail).lower()
                            ):
                                raise
                            logger.warning(
                                "Retrying %s without saved gpu_ids %s: %s",
                                override_id,
                                load_kwargs.get("gpu_ids"),
                                exc.detail,
                            )
                            load_kwargs.pop("gpu_ids", None)
                            await _load_model_impl(
                                LoadRequest(**load_kwargs),
                                fastapi_request,
                                current_subject,
                                current_request_counted = True,
                            )
                        # Advertise the repo id (not the concrete load path) as the loaded
                        # model's public id and override key for /v1/models and idle stash.
                        get_llama_cpp_backend()._openai_advertised_id = override_id
                finally:
                    # Deregister before releasing the gate: otherwise a swap on another
                    # loop counts this finished request as queued and unloads its model.
                    _note_switch_waiter(key, -1)
                    waiter_noted = False
                    _auto_switch_process_lock.release()
        finally:
            if waiter_noted:
                _note_switch_waiter(key, -1)

    await _resolve_and_switch()
    # The switch may have missed, so refuse rather than answer as whatever is resident.
    await _reject_unservable_model(requested_model, fastapi_request)


async def _auto_switch_from_request_body(request: Request, current_subject: str):
    """Run auto-switch from a raw-body endpoint's ``model`` without changing its
    pre-feature status codes: a malformed/non-dict body yields no model (so an
    unloaded backend still 503s, not 500), and the caller re-reads to surface the
    original parse error after the loaded-state check. Returns the parsed body, or
    None if it could not be parsed."""
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(body, dict):
        # A raw-body client may omit ``model`` and rely on the loaded backend. Pass
        # a reload-only sentinel so the idle-stash reload still runs (an idle-freed
        # model is restored) without the resolver ever matching a real name.
        model = body.get("model") or _RELOAD_ONLY_MODEL
    else:
        model = None
    await _maybe_auto_switch_model(model, request, current_subject)
    return body


def _effective_load_in_4bit(config: ModelConfig, requested: bool) -> bool:
    """Effective quantization the loader will use: a LoRA adapter can flip 4-bit to
    16-bit via adapter_config.json, so the guard sizes this, not the raw request."""
    load_in_4bit = requested
    if not getattr(config, "is_lora", False) or not getattr(config, "path", None):
        return load_in_4bit
    adapter_cfg_path = Path(config.path) / "adapter_config.json"
    if not adapter_cfg_path.exists():
        return load_in_4bit
    try:
        with open(adapter_cfg_path, encoding = "utf-8-sig") as f:
            adapter_cfg = json.load(f)
        if not isinstance(adapter_cfg, dict):  # malformed -> keep requested
            return load_in_4bit
    except Exception as e:
        logger.warning(f"Could not read adapter_config.json: {e}")
        return load_in_4bit
    training_method = adapter_cfg.get("unsloth_training_method")
    if training_method == "lora":
        return False
    if training_method == "qlora":
        return True
    if not training_method and config.base_model and "-bnb-4bit" not in config.base_model.lower():
        return False
    return load_in_4bit


def _remote_gguf_companion_bytes(
    repo: str, *, hf_token: Optional[str], include_mmproj: bool
) -> int:
    """Bytes of MTP/mmproj companion GGUFs llama-server auto-downloads. 0 on error,
    so it can only add headroom, never refuse a load by itself."""
    try:
        from huggingface_hub import model_info

        info = model_info(repo, token = hf_token, files_metadata = True)
        total = 0
        for sibling in info.siblings or []:
            name = sibling.rfilename or ""
            base = Path(name).name.lower()
            if not base.endswith(".gguf"):
                continue
            # Root-level mtp- only: -hf auto-fetches the repo-root drafter, not
            # the MTP/ subdir copies (which now share the mtp- prefix too).
            is_root_mtp = "/" not in name and base.startswith("mtp-")
            if is_root_mtp or (include_mmproj and "mmproj" in base):
                total += getattr(sibling, "size", 0) or 0
        return total
    except Exception as e:
        logger.warning(f"Could not size GGUF companions for {repo}: {e}")
        return 0


def _estimate_gguf_kv_gb(
    gguf_path: str,
    max_seq_length: int,
    llama_extra_args: Optional[list[str]] = None,
    n_parallel: int = 1,
    cache_type_kv: Optional[str] = None,
    tensor_parallel: bool = False,
) -> float:
    """KV-cache VRAM (GB) at the larger of max_seq_length and any `--ctx-size`/`-c`
    override, over n_parallel slots, using the effective cache settings and managed
    launcher defaults. 0 if metadata is unreadable."""
    try:
        from core.inference.llama_server_args import parse_ctx_override

        probe = LlamaCppBackend()
        probe._read_gguf_metadata(gguf_path)
        if not probe._can_estimate_kv():
            return 0.0
        try:
            ctx_override = parse_ctx_override(llama_extra_args) or 0
        except Exception:
            ctx_override = 0  # malformed extras are rejected upstream; fall back
        ctx = max(max_seq_length or 0, ctx_override) or (probe._context_length or 0)
        if ctx <= 0:
            return 0.0
        slots = max(1, n_parallel or 1)
        managed_kv_unified = bool(
            slots > 1
            and LlamaCppBackend.probe_server_capabilities().get("supports_kv_unified", False)
        )
        planned_cache_types = _planned_main_cache_types(
            cache_type_kv,
            llama_extra_args,
        )
        if tensor_parallel and any(
            cache_type not in LlamaCppBackend._TENSOR_PARALLEL_KV_TYPES
            for cache_type in planned_cache_types
        ):
            # Tensor mode strips quantized axes, but a layer fallback restores
            # the original settings. Size for the larger successful outcome.
            tensor_cache_types = _planned_main_cache_types(None, None)
            cache_type_for_budget = max(
                (*planned_cache_types, *tensor_cache_types, "f16"),
                key = _kv_bytes_per_elem,
            )
        else:
            cache_type_for_budget = max(
                planned_cache_types,
                key = _kv_bytes_per_elem,
            )
        kv = probe._estimate_kv_cache_bytes(
            ctx,
            cache_type_for_budget,
            n_parallel = slots,
            swa_full = _swa_full_from_args_or_env(llama_extra_args),
            kv_unified = _kv_unified_from_args(
                llama_extra_args,
                default = managed_kv_unified,
            ),
            n_ubatch = _extra_args_n_ubatch(llama_extra_args, n_ctx = ctx),
            flash_attn = False,
        )
        return kv / (1024**3)
    except Exception as e:
        logger.warning(f"Could not size GGUF KV cache for training guard: {e}")
        return 0.0


def _estimate_gguf_required_gb(
    config: ModelConfig,
    hf_token: Optional[str] = None,
    max_seq_length: int = 0,
    llama_extra_args: Optional[list[str]] = None,
    n_parallel: int = 1,
    cache_type_kv: Optional[str] = None,
    tensor_parallel: bool = False,
) -> Optional[float]:
    """Approximate GGUF VRAM (GB): quantized weights + companions, plus the KV
    cache for local files (unreadable pre-download for remote). None when nothing
    resolves so the caller default-denies."""
    try:
        total_bytes = 0
        main = getattr(config, "gguf_file", None)
        if main and Path(main).is_file():
            total_bytes += LlamaCppBackend._get_gguf_size_bytes(str(main))
        for attr in ("gguf_mmproj_file", "gguf_mtp_file"):
            f = getattr(config, attr, None)
            if f and Path(f).is_file():
                total_bytes += Path(f).stat().st_size
        if total_bytes > 0:
            return total_bytes / (1024**3) + _estimate_gguf_kv_gb(
                main,
                max_seq_length,
                llama_extra_args,
                n_parallel,
                cache_type_kv,
                tensor_parallel,
            )

        repo = getattr(config, "gguf_hf_repo", None)
        variant = getattr(config, "gguf_variant", None)
        if repo and variant:
            from utils.models.model_config import list_gguf_variants

            variants, has_vision = list_gguf_variants(repo, hf_token = hf_token)
            main_bytes = next(
                (v.size_bytes for v in variants if v.quant.lower() == variant.lower()), None
            )
            if main_bytes is None:
                return None
            companions = _remote_gguf_companion_bytes(
                repo, hf_token = hf_token, include_mmproj = bool(has_vision)
            )
            return (main_bytes + companions) / (1024**3)
        return None
    except Exception as e:
        logger.warning(f"Could not size GGUF model for training guard: {e}")
        return None


def _gguf_layer_count(config: ModelConfig) -> Optional[int]:
    """Total block count from a local GGUF header, or None (remote / unreadable)."""
    try:
        main = getattr(config, "gguf_file", None)
        if not (main and Path(main).is_file()):
            repo = getattr(config, "gguf_hf_repo", None)
            variant = getattr(config, "gguf_variant", None)
            if repo and variant:
                from hub.utils.gguf import resolve_local_gguf_path
                main = resolve_local_gguf_path(repo, variant)
        if main and Path(main).is_file():
            probe = LlamaCppBackend()
            probe._read_gguf_metadata(str(main))
            return getattr(probe, "_n_layers", None) or None
    except Exception as e:
        logger.debug("Could not read GGUF layer count for training guard: %s", e)
    return None


def _classify_diffusion_gguf(config: ModelConfig) -> Optional[bool]:
    """Classify a GGUF as diffusion, normal, or unknown before loading."""
    identity = " ".join(
        str(getattr(config, attr, "") or "") for attr in ("identifier", "gguf_hf_repo", "gguf_file")
    ).lower()
    # Only use the specific DiffusionGemma family name as a header fallback.
    name_says_diffusion = "diffusiongemma" in _re.sub(r"[^a-z0-9]+", "", identity)

    try:
        main = getattr(config, "gguf_file", None)
        if not (main and Path(main).is_file()):
            repo = getattr(config, "gguf_hf_repo", None)
            variant = getattr(config, "gguf_variant", None)
            if repo and variant:
                from hub.utils.gguf import resolve_local_gguf_path
                main = resolve_local_gguf_path(repo, variant)
        if main and Path(main).is_file():
            probe = LlamaCppBackend()
            probe._read_gguf_metadata(str(main))
            if probe.is_diffusion:
                return True
            if getattr(probe, "_architecture", None):
                return False
    except Exception as e:
        logger.debug("Could not identify diffusion GGUF for training guard: %s", e)
    return True if name_says_diffusion else None


async def _override_gpu_ids_still_resolve(gpu_ids: List[int]) -> bool:
    """Whether a per-model GPU pin is usable on this machine right now.

    normalize_model_override cannot know the device list, so it stores whatever
    was valid where the config was written. This is the load-time reconciliation
    for the device-availability rules, which are the ones that go stale.

    Deliberately not exhaustive: model-dependent rules (a Vulkan diffusion GGUF
    refuses gpu_ids outright) need a ModelConfig this has no reason to build.
    The caller's retry-without-the-pin covers those, and covers rules added
    later, so a check missing here costs one extra attempt, not the load.
    """
    try:
        from utils.hardware import DeviceType, get_device
        from utils.hardware.hardware import resolve_requested_gpu_ids

        # One hop for the whole device-dependent block: resolve_requested_gpu_ids() reaches
        # get_device() itself, so both wait on the detection lock during the warm.
        def _device_and_resolution() -> tuple[object, bool, list]:
            is_vulkan = LlamaCppBackend._is_vulkan_backend()
            return (
                get_device(),
                is_vulkan,
                resolve_requested_gpu_ids(gpu_ids, is_vulkan = is_vulkan),
            )

        device, is_vulkan, resolved = await asyncio.to_thread(_device_and_resolution)
        if device == DeviceType.XPU and not is_vulkan:
            # Rejected outright on XPU.
            return False
        if is_vulkan and resolved:
            # Vulkan ordinals are their own index space, so presence needs the ggml probe.
            binary = LlamaCppBackend._find_llama_server_binary()
            if binary:
                probed = {
                    gpu[0]
                    for gpu in await asyncio.to_thread(LlamaCppBackend._get_gpu_memory, binary)
                }
                if not {int(gpu_id) for gpu_id in resolved}.issubset(probed):
                    return False
        return True
    except Exception:
        return False


def _reject_draft_device_with_gpu_ids(
    gpu_ids: Optional[List[int]],
    extra_args: Optional[list[str]],
    *,
    gpu_ids_are_vulkan_ordinals: bool,
) -> None:
    """Reject a physical drafter pin beside Vulkan-ordinal main placement."""
    if not gpu_ids or not gpu_ids_are_vulkan_ordinals:
        return
    draft_device = _extra_args_draft_device_pin(extra_args)
    if draft_device is not None:
        raise HTTPException(
            status_code = 400,
            detail = (
                f"A draft-model device override ('{draft_device}') cannot be combined "
                "with explicit gpu_ids: it would place the speculative drafter outside "
                "the pinned GPUs the training guard budgeted. Remove the draft-device "
                "flag to follow gpu_ids, or set it to none."
            ),
        )


_DIFFUSION_KIND_UNSET = object()


async def _resolve_gguf_gpu_ids_for_request(
    config: ModelConfig,
    gpu_ids: Optional[List[int]],
    *,
    diffusion_kind: Optional[bool] | object = _DIFFUSION_KIND_UNSET,
) -> tuple[Optional[List[int]], bool]:
    """Validate GGUF GPU IDs and report whether they are Vulkan ordinals."""
    if not gpu_ids:
        return None, False

    from utils.hardware import DeviceType, get_device
    from utils.hardware.hardware import resolve_requested_gpu_ids

    llama_backend = get_llama_cpp_backend()
    is_vulkan_build = await asyncio.to_thread(llama_backend.is_vulkan_build)
    if diffusion_kind is _DIFFUSION_KIND_UNSET:
        diffusion_kind = _classify_diffusion_gguf(config)
    confirmed_diffusion = diffusion_kind is True
    definitively_non_diffusion = diffusion_kind is False
    # Off-loop: get_device() waits on the detection lock, i.e. the cold torch import.
    device = await asyncio.to_thread(get_device)
    lacks_gpu_lib = getattr(llama_backend, "_backend_lacks_gpu_lib", None)

    # ROCm is deliberately DeviceType.CUDA internally because it uses
    # torch.cuda.*. Only the API label changes to "rocm", so this accepts both
    # CUDA and ROCm physical IDs while rejecting device namespaces the
    # diffusion runner cannot apply.
    diffusion_physical_ids_supported = device == DeviceType.CUDA
    if confirmed_diffusion and not diffusion_physical_ids_supported:
        raise HTTPException(
            status_code = 400,
            detail = (
                "GPU selection (gpu_ids) for DiffusionGemma requires CUDA or ROCm. "
                "Omit gpu_ids on this host."
            ),
        )

    if confirmed_diffusion and is_vulkan_build:
        raise HTTPException(
            status_code = 400,
            detail = (
                "GPU selection (gpu_ids) is not supported for a DiffusionGemma "
                "GGUF on a Vulkan llama.cpp build: the picker uses Vulkan ordinals, "
                "which have no defined mapping to CUDA physical indices. Omit gpu_ids "
                "to use the default device."
            ),
        )

    ids_are_vulkan_ordinals = is_vulkan_build

    if device == DeviceType.XPU and not ids_are_vulkan_ordinals:
        raise HTTPException(
            status_code = 400,
            detail = (
                "GPU selection (gpu_ids) is not supported on Intel XPU. "
                "Omit gpu_ids to use all devices."
            ),
        )

    if (
        device == DeviceType.CUDA
        and not ids_are_vulkan_ordinals
        and definitively_non_diffusion
        and callable(lacks_gpu_lib)
        and await asyncio.to_thread(lacks_gpu_lib)
    ):
        raise HTTPException(
            status_code = 400,
            detail = (
                f"Requested gpu_ids {list(gpu_ids)} but the llama.cpp build has "
                "no GPU backend (CPU-only build); it would ignore the pin and run "
                "on CPU. Omit gpu_ids to run on CPU."
            ),
        )

    try:
        resolved = resolve_requested_gpu_ids(
            gpu_ids,
            is_vulkan = ids_are_vulkan_ordinals,
        )
    except ValueError as exc:
        raise HTTPException(status_code = 400, detail = str(exc)) from exc

    if ids_are_vulkan_ordinals and resolved:
        binary = LlamaCppBackend._find_llama_server_binary()
        if binary:
            probed = {
                gpu[0] for gpu in await asyncio.to_thread(LlamaCppBackend._get_gpu_memory, binary)
            }
            wanted = {int(gpu_id) for gpu_id in resolved}
            if not wanted.issubset(probed):
                raise HTTPException(
                    status_code = 400,
                    detail = (
                        f"Requested Vulkan GPU ordinal(s) {sorted(wanted)} not "
                        f"present. Available Vulkan devices: {sorted(probed)}."
                    ),
                )

    return resolved, ids_are_vulkan_ordinals


class _LoadPlacement(NamedTuple):
    requested_gpu_ids: Optional[List[int]]
    resolved_gpu_ids: Optional[List[int]]
    gpu_ids_are_vulkan_ordinals: bool
    diffusion_kind: Optional[bool]


def _resolve_parallel_slots(request, fastapi_request: Optional[Request]) -> int:
    if request.n_parallel is not None:
        return request.n_parallel
    state = getattr(getattr(fastapi_request, "app", None), "state", None)
    return getattr(state, "llama_parallel_slots", 1)


async def _prepare_load_placement(
    config: ModelConfig,
    request: LoadRequest | ValidateModelRequest,
    extra_args: Optional[list[str]],
) -> _LoadPlacement:
    requested = request.gpu_ids or None
    if not config.is_gguf:
        return _LoadPlacement(requested, None, False, False)
    diffusion_kind = _classify_diffusion_gguf(config)
    resolved, is_vulkan = await _resolve_gguf_gpu_ids_for_request(
        config, requested, diffusion_kind = diffusion_kind
    )
    _reject_draft_device_with_gpu_ids(resolved, extra_args, gpu_ids_are_vulkan_ordinals = is_vulkan)
    return _LoadPlacement(requested, resolved, is_vulkan, diffusion_kind)


def _resolve_gguf_load_intent(
    config: ModelConfig,
    request: LoadRequest,
    *,
    native_grant_backed: bool,
    chat_template_override: Optional[str],
    extra_args: Optional[list[str]],
    placement: _LoadPlacement,
    n_parallel: int,
) -> GgufLoadIntent:
    """Resolve source, companions, settings, and placement into one load value."""
    if config.gguf_hf_repo:
        source = GgufLoadIntent(
            model_identifier = config.identifier,
            hf_repo = config.gguf_hf_repo,
            hf_variant = config.gguf_variant,
            hf_token = request.hf_token,
        )
    else:
        if native_grant_backed:
            if config.gguf_mmproj_file:
                _validate_native_gguf_companion(
                    config.gguf_mmproj_file, config.gguf_file, "vision companion"
                )
            if config.gguf_mtp_file:
                config.gguf_mtp_file = _mtp_draft_for_path(
                    config.gguf_file,
                    True,
                    log_native_fallback = True,
                )
        source = GgufLoadIntent(
            model_identifier = config.identifier,
            gguf_path = config.gguf_file,
            mmproj_path = config.gguf_mmproj_file,
            mtp_draft_path = config.gguf_mtp_file,
            hf_variant = config.gguf_variant,
        )

    return _gguf_request_intent(
        source,
        request,
        chat_template_override = chat_template_override,
        extra_args = extra_args,
        gpu_ids = placement.resolved_gpu_ids,
        n_parallel = n_parallel,
        is_vision = config.is_vision,
        gpu_ids_are_vulkan_ordinals = placement.gpu_ids_are_vulkan_ordinals,
        extra_args_inherited = getattr(request, "llama_extra_args", None) is None,
    )


def _guard_chat_load_against_training(
    config: ModelConfig,
    request: LoadRequest | ValidateModelRequest,
    *,
    load_in_4bit: bool,
    placement: _LoadPlacement,
    llama_extra_args: Optional[list[str]] = None,
    n_parallel: int = 1,
) -> None:
    """Protect active training from automatically placed chat-model loads.

    No-op when training is inactive or unknown. `load_in_4bit` must be the
    effective quantization (see _effective_load_in_4bit). Manual chat-GGUF
    placement is an explicit override: Auto layers delegate fitting to
    llama.cpp's ``--fit`` and pinned layers are owned by the user, so neither is
    estimated here. Diffusion is still guarded because its runner uses one GPU, except
    for an explicit zero-layer split, which places no layers at all; an unclassified
    GGUF is guarded as potentially diffusion until its local header proves otherwise.
    Other loads raise HTTP 409 when they would not fit beside training.
    """
    from core.training import get_training_backend
    from routes.training_vram import can_load_chat_during_training

    requested_gpu_ids = placement.requested_gpu_ids
    gpu_ids_are_vulkan_ordinals = placement.gpu_ids_are_vulkan_ordinals
    diffusion_kind = placement.diffusion_kind
    try:
        llm_active = get_training_backend().is_training_active()
    except Exception as e:
        # Independent probes: an unreadable LLM backend must still fall through to the diffusion check, which reads a different service.
        logger.warning("Could not check training state for chat-load guard: %s", e)
        llm_active = False

    if not llm_active:
        # An SDXL LoRA trainer runs in its own subprocess and cannot be cheaply fit-checked, so refuse the chat load while one is active.
        if _diffusion_training_active():
            raise HTTPException(
                status_code = 409,
                detail = (
                    "Can't load this model while diffusion (Images) training is running: "
                    "its GPU memory use can't be verified against the trainer, so the load "
                    "was refused to protect the run. Try again after training finishes."
                ),
            )
        return

    from core.inference.llama_cpp import _diffusion_manual_ngl, _scale_diffusion_required_gb

    is_gguf = bool(getattr(config, "is_gguf", False))
    # load_model pins a GGUF to CPU on a virtualised Metal device, so guard what will run:
    # sized as the raw Auto request, a CPU-only load is refused over VRAM it never takes.
    _guard_gpu_memory_mode = request.gpu_memory_mode
    _guard_gpu_layers = request.gpu_layers
    _guard_tensor_parallel = request.tensor_parallel
    _pv_guard_forced_cpu = is_gguf and _metal_device_is_paravirtual()
    if _pv_guard_forced_cpu:
        _pv = paravirtual_normalized_request(
            gpu_memory_mode = request.gpu_memory_mode,
            gpu_layers = request.gpu_layers,
            tensor_parallel = request.tensor_parallel,
            tensor_split = None,
            n_cpu_moe = 0,
            extra_args = llama_extra_args,
            log_dropped = False,
        )
        _guard_gpu_memory_mode = _pv.gpu_memory_mode
        _guard_gpu_layers = _pv.gpu_layers
        _guard_tensor_parallel = _pv.tensor_parallel
        llama_extra_args = _pv.extra_args
    # The pin leaves nothing on the GPU (--device none, no mmproj offload, drafter on CPU),
    # so there is nothing to budget whatever the GGUF turns out to be. Ahead of the checks
    # below, which only exempt a CONFIRMED diffusion GGUF and would size an unclassified
    # remote one as GPU-resident and 409 a load that never touches VRAM.
    if _pv_guard_forced_cpu:
        return
    if is_gguf and _guard_gpu_memory_mode == "manual" and (diffusion_kind is False):
        return
    # A zero-layer diffusion split places no model layers on any device, so it cannot compete
    # with training for VRAM. Mirrors the loader, which folds the same condition into its
    # cpu_only (core/inference/llama_cpp.py).
    diffusion_ngl = (
        _diffusion_manual_ngl(_guard_gpu_memory_mode, _guard_gpu_layers) if is_gguf else None
    )
    if diffusion_ngl is not None and diffusion_kind is not False:
        # The loader drops the split when the shim has no --ngl and launches GPU-resident.
        # Guard what will run, not what was asked, or a zero-layer request skips the VRAM
        # check while the child takes a whole GPU.
        try:
            if not get_llama_cpp_backend().diffusion_split_supported():
                diffusion_ngl = None
        except Exception as e:
            logger.warning("Could not probe diffusion shim for chat-load guard: %s", e)
            diffusion_ngl = None
    # `is True`, not `is not False`: only a CONFIRMED diffusion GGUF places nothing at ngl 0.
    # On a possibly-ordinary GGUF a device pin, tensor mode, mmproj or a GPU drafter keeps it
    # resident (see LlamaCppBackend._zero_offload_keeps_gpu_visible).
    if is_gguf and diffusion_kind is True and diffusion_ngl == 0:
        return

    diffusion_gpu = None
    if is_gguf and diffusion_kind is not False and not gpu_ids_are_vulkan_ordinals:
        # Use the same token selection as the runner: an explicit pick wins,
        # followed by DG_GPU, the first parent-visible token, then GPU 0. Suppressed
        # for a Vulkan-ordinal pin so single-device CUDA budgeting can't override the
        # Vulkan-ordinal path (single_device_gpu wins in can_load_chat_during_training).
        # No force_cpu, deliberately: a CONFIRMED zero-layer split already returned above, so
        # ngl 0 here means an UNCLASSIFIED GGUF -- and an empty token makes
        # can_load_chat_during_training short-circuit to "cpu_only" and always allow the
        # load, on an assumption that only holds for real diffusion. Let the picker choose a
        # device so an ordinary GGUF keeping VRAM at --gpu-layers 0 stays conservatively sized.
        diffusion_gpu = LlamaCppBackend._diffusion_gpu_arg(
            requested_gpu_ids,
            cpu_only = LlamaCppBackend._effective_gpu_count() == 0,
        )

    # Detected once: both the tensor-parallel KV sizing below and the Vulkan
    # free-VRAM view need the same answer. An ordinal pin only exists on a
    # Vulkan build, so it settles the question without probing the binary.
    binary = LlamaCppBackend._find_llama_server_binary() if is_gguf else None
    is_vulkan_backend = bool(
        is_gguf
        and (gpu_ids_are_vulkan_ordinals or (binary and LlamaCppBackend._is_vulkan_backend(binary)))
    )

    # Size with the count that will actually launch, or a load that fits gets a
    # 409: diffusion never receives --parallel, load_model clamps to 1 on an
    # llama-server without --kv-unified, and it clamps MTP to 1 as well. An
    # unclassified GGUF keeps the ask.
    if is_gguf and n_parallel > 1:
        if diffusion_kind is True:
            n_parallel = 1
        # MTP is deliberately NOT clamped here even though the launch clamps it to one
        # slot. _estimate_gguf_required_gb counts the drafter file and the main KV, but
        # not the draft KV, the duplicated target context MLA keeps, or the draft compute
        # reserve, all of which load_model does budget. Sizing for one slot would drop
        # the slot KV without replacing it with those, and a guard that under-sizes
        # evicts the training run it exists to protect: the spare slots stand in for
        # what is not modelled.
        else:
            try:
                caps = LlamaCppBackend.probe_server_capabilities()
                if caps.get("found") and not caps.get("supports_kv_unified"):
                    n_parallel = 1
            except Exception as e:
                logger.warning("Could not probe llama-server slots for chat-load guard: %s", e)

    required_override_gb = (
        _estimate_gguf_required_gb(
            config,
            hf_token = request.hf_token,
            max_seq_length = request.max_seq_length,
            llama_extra_args = llama_extra_args,
            n_parallel = n_parallel,
            cache_type_kv = request.cache_type_kv,
            tensor_parallel = (
                _effective_tensor_parallel(llama_extra_args, _guard_tensor_parallel)
                and (
                    is_vulkan_backend
                    or LlamaCppBackend._effective_gpu_count(requested_gpu_ids) >= 2
                )
            ),
        )
        if is_gguf
        else None
    )
    # A confirmed-diffusion positive split puts only ngl/n_layers of the weights on the GPU (a
    # split the loader would drop was nulled above). Unknown classification keeps the full
    # estimate: its header was unreadable, so the layer count is too.
    if (
        required_override_gb is not None
        and diffusion_kind is True
        and diffusion_ngl is not None
        and diffusion_ngl > 0
    ):
        required_override_gb = _scale_diffusion_required_gb(
            required_override_gb, diffusion_ngl, _gguf_layer_count(config)
        )

    vulkan_free_vram_gb = None
    if is_gguf:
        if is_vulkan_backend and (gpu_ids_are_vulkan_ordinals or diffusion_kind is False):
            gpu_memory = LlamaCppBackend._get_gpu_memory(binary)
            if not requested_gpu_ids:
                gpu_memory = LlamaCppBackend._vulkan_auto_gpu_memory(gpu_memory)
            vulkan_free_vram_gb = {
                index: free_mib / 1024.0 for index, free_mib, _total_mib in gpu_memory
            }
        elif is_vulkan_backend and diffusion_kind is None and requested_gpu_ids:
            # Until the header is available, the model may use either the Vulkan
            # llama-server or the CUDA-only diffusion runner, so an explicit pin
            # cannot be budgeted: neither device namespace can stand in for the
            # other. Automatic placement has no ordinal to mis-map, so it keeps
            # the torch view below rather than refusing every uncached remote
            # GGUF while training runs.
            vulkan_free_vram_gb = {}

    ok, info = can_load_chat_during_training(
        model_name = getattr(config, "identifier", request.model_path),
        hf_token = request.hf_token,
        load_in_4bit = load_in_4bit,
        max_seq_length = request.max_seq_length,
        requested_gpu_ids = requested_gpu_ids,
        is_gguf = is_gguf,
        gpu_ids_are_vulkan_ordinals = gpu_ids_are_vulkan_ordinals,
        vulkan_free_vram_gb = vulkan_free_vram_gb,
        required_override_gb = required_override_gb,
        single_device_gpu = diffusion_gpu,
    )
    if ok:
        return

    usable = info.get("usable_gb")
    needed = info.get("needed_gb")
    if needed is None:
        needed = info.get("required_gb")
    if needed is not None and usable is not None:
        detail = (
            f"Not enough free GPU memory to load this model while training is "
            f"running (needs ~{needed:.0f} GB including safety headroom, "
            f"~{usable:.0f} GB free). Training was left untouched. Use an external "
            f"provider, a smaller or more quantized model, or try again after "
            f"training finishes."
        )
    else:
        detail = (
            "Can't load this model while training is running: its GPU memory use "
            "could not be verified, so the load was refused to protect the "
            "training run. Use an external provider or try again after training "
            "finishes."
        )
    logger.info("Refusing chat-model load during training: %s", info)
    raise HTTPException(status_code = 409, detail = detail)


def _resolve_inherited_extra_args(
    request,
    config: ModelConfig,
    model_identifier: str,
    extra_llama_args: Optional[list[str]],
    effective_chat_template_override: Optional[str] = None,
) -> Optional[list[str]]:
    """Effective pass-through extras for a GGUF request that omitted the field:
    the previous same-model load's extras, shadow-stripped, so a settings-Apply
    reload (which does not round-trip the extras field) keeps them (#5401)."""
    if getattr(request, "llama_extra_args", None) is not None:
        return extra_llama_args
    if not getattr(config, "is_gguf", False):
        return extra_llama_args
    llama_backend = get_llama_cpp_backend()
    stored_args = getattr(llama_backend, "extra_args", None)
    if not stored_args:
        return extra_llama_args
    # Inherit the previous load's extras (the chat-settings Apply path doesn't
    # round-trip them; an explicit [] still clears). Gated on (model_identifier,
    # hf_variant) to refuse cross-model pickup, and shadowing flags are
    # stripped so an inherited override can't win the last-wins CLI
    # parse against a freshly-supplied first-class field.
    source = getattr(llama_backend, "extra_args_source", None)
    # Compare against the resolved variant, not the request field: callers
    # commonly omit gguf_variant for local ``.gguf`` paths and HF auto-pick
    # flows. ``config.gguf_variant`` is the variant load_model was actually
    # invoked with, so both sides of the comparison key off the same string.
    resolved_variant = (config.gguf_variant or "").lower()
    request_variant = (request.gguf_variant or "").lower()
    stored_variant = (source[1] or "").lower() if source else ""
    same_model = bool(source and source[0] and source[0].lower() == model_identifier.lower())
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
        # Strip only the groups whose first-class field was set by the caller, so
        # an inherited --chat-template-file survives an Apply that omits
        # chat_template_override. A bundled family template (e.g. gemma-4) counts as
        # a first-class template even when the request omits chat_template_override,
        # so strip the inherited --chat-template-file then too -- else the stale arg
        # (appended last) shadows the bundled template while Studio reports its caps.
        fields_set = getattr(request, "model_fields_set", set())
        stripped = strip_shadowing_flags(
            stored_args,
            strip_context = "max_seq_length" in fields_set,
            strip_cache = "cache_type_kv" in fields_set,
            strip_spec = ("speculative_type" in fields_set or "spec_draft_n_max" in fields_set),
            strip_template = (
                "chat_template_override" in fields_set
                or effective_chat_template_override is not None
            ),
            strip_split_mode = _should_strip_split_mode(request, stored_args),
            # manual + per-GPU ratio emits its own --tensor-split; drop
            # an inherited one (appended last would override it) while
            # keeping the user's --split-mode row/none/layer choice.
            strip_tensor_split = _should_strip_tensor_split(request),
            # manual emits its own --fit/--gpu-layers, so an inherited offload flag
            # must not last-wins-override it. auto leaves a user's inherited -ngl
            # alone. getattr: a validate request reuses this resolver, no offload fields.
            strip_offload = getattr(request, "gpu_memory_mode", "auto") == "manual",
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
    return extra_llama_args


def _model_json_response(model, status_code: int = 200) -> Response:
    """Serialize a pydantic response once via pydantic-core.

    Equivalent body to ``JSONResponse(content = model.model_dump())`` but
    avoids the dict round-trip plus Starlette's second ``json.dumps``.
    """
    return Response(
        content = model.model_dump_json(),
        media_type = "application/json",
        status_code = status_code,
    )


_NOT_SUPPORTED_HINTS = (
    "No config file found",
    "not yet supported",
    "is not supported",
    "does not support",
)

_NVFP4_INFERENCE_UNSUPPORTED_MESSAGE = (
    "We are working on supporting NVFP4 inference. For now it is not supported"
)


def _is_unsupported_nvfp4_inference_error(msg: str) -> bool:
    """Whether ``msg`` is the verbose MLX per-module metadata error emitted
    while loading an NVFP4 checkpoint."""
    lower_msg = msg.lower()
    return "nvfp4" in lower_msg and "per-module mlx quantization metadata" in lower_msg


def _maybe_unsupported_message(msg: str) -> str:
    """Rewrite a load/validate error into the friendly "not supported yet"
    message when it matches a known unsupported-model signature; otherwise
    return ``msg`` unchanged."""
    if any(h.lower() in msg.lower() for h in _NOT_SUPPORTED_HINTS):
        return f"This model is not supported yet. Try a different model. (Original error: {msg})"
    return msg


def _raise_if_sidecar_swap_in_progress() -> None:
    from utils.transformers_version import sidecar_swap_in_progress
    if sidecar_swap_in_progress():
        raise HTTPException(
            status_code = 409,
            detail = "A transformers installation is in progress. Retry when it completes.",
        )


def _raise_or_cancel_active_generations(
    *,
    force: bool,
    action: str,
    cancel: bool = True,
) -> int:
    """Gate a model swap on the chats currently generating.

    Every open conversation decodes on the single llama-server this route is
    about to replace, so refuse with 409 and name them. force_cancel_active
    instead stops them through the same events an explicit Stop uses. Returns
    how many were cancelled. The frontend guard is bypassable from a second tab
    or curl; this one is not.

    ``cancel = False`` runs the refusal half only. /load calls it that way once
    up front, so a non-forced swap still fails fast, and again with cancel just
    before teardown: cancelling is destructive and unrecoverable, so it must not
    run ahead of preflight checks that can still reject the load (see
    _load_model_impl).
    """
    if not active_generations.count():
        return 0
    if not force:
        thread_ids = active_generations.active_thread_ids()
        running = active_generations.count()
        raise HTTPException(
            status_code = 409,
            detail = {
                "error": "active_generations",
                "message": (
                    f"{action} would stop {running} chat"
                    f"{'s' if running != 1 else ''} that "
                    f"{'are' if running != 1 else 'is'} still generating. "
                    "Stop them first, or retry with force_cancel_active."
                ),
                "running": running,
                "thread_ids": thread_ids,
            },
        )
    if not cancel:
        # Refusal-only pass: the caller cancels later, once nothing can still reject the load.
        return 0
    cancelled = active_generations.cancel_all()
    if cancelled:
        logger.info(
            "model_swap_cancelled_active_generations",
            extra = {"event": "inference.reload_cancelled_generations", "count": cancelled},
        )
    return cancelled


_POST_CANCEL_DRAIN_TIMEOUT_S = 5.0


async def _cancel_and_drain_for_sidecar_swap(timeout_s: Optional[float] = None) -> None:
    """Clear the way for a confirmed sidecar swap, then stop the chats it interrupts.

    The installer gates on the middleware's in-flight count, not on
    active_generations, so it also sees requests the cancel cannot stop. Drain
    those FIRST, discounting the registered chats (they are what the cancel is
    for, so waiting on them would wait out the point of the force). Only then
    cancel, and let the survivors unwind. Cancelling first meant an unrelated
    counted request -- a /v1/messages/count_tokens, say -- was still there for
    the caller's recheck, which then refused an install that had already stopped
    every chat for nothing.

    Bounded on both halves: the requests being waited on may never observe a
    cancel, and this holds the lifecycle gate and the sidecar reservation inside
    ``asyncio.shield``, so an unbounded wait would wedge the process. Expiring in
    the first half returns without cancelling, so the caller's recheck refuses
    with the chats untouched.
    """
    from core.inference.llama_keepwarm import other_inference_request_count

    budget = _POST_CANCEL_DRAIN_TIMEOUT_S if timeout_s is None else timeout_s

    async def _drain(deadline: float, *, discount_registered: bool) -> bool:
        while True:
            counted = other_inference_request_count(
                current_request_counted = False, include_pending = False
            )
            if discount_registered:
                counted -= min(counted, active_generations.count())
            if counted <= 0:
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.02)

    # Weighted, not halved, so the total wait under the gate is unchanged. The first drain only
    # asks whether unrelated inference is in flight; cutting the second short refused installs
    # whose chats had already been stopped for nothing.
    if not await _drain(time.monotonic() + budget / 5, discount_registered = True):
        return
    _raise_or_cancel_active_generations(force = True, action = "Installing a new transformers version")
    await _drain(time.monotonic() + budget * 4 / 5, discount_registered = False)


async def _drain_and_recancel_before_teardown(*, force: bool, action: str) -> None:
    """Wait out inference the registry cannot see, then stop anything new.

    A request that passed the keep-warm middleware but has not reached its
    ``_TrackedCancel`` yet is counted in-flight and absent from the registry, so
    cancelling on the registry alone lets a teardown land on an already-admitted
    request. Drain on the middleware count instead, which covers both the runs
    just cancelled and the ones still in that window, then cancel again for
    anything that registered while waiting.

    Bounded and non-raising: an unload is a deliberate user action, so the worst
    case stays what it is today rather than becoming a refusal.
    """
    await _wait_for_model_switch_idle(
        current_request_counted = False,
        timeout_s = _POST_CANCEL_DRAIN_TIMEOUT_S,
    )
    if force:
        _raise_or_cancel_active_generations(force = True, action = action)


_UNRESOLVED_BACKEND_STATE = object()


def _names_the_resident_model(resident: Optional[str], model_path: str) -> bool:
    """Whether a client's ``model_path`` names ``resident``.

    A cached row can pin a snapshot directory, so the load sends that path while the status the
    client reads back reports the repo id it maps to. Both name the same model, and an unload
    arriving under either has to find it.
    """
    return bool(resident) and model_id_matches(model_path, resident)


def _names_the_loading_model(loading: str, model_path: str) -> bool:
    """Whether a client's ``model_path`` names the load already in flight.

    Cancel sends the id the picker shows, which for a pinned row is not the path the load is
    running as, so a raw compare left Stop loading reporting success on a load still running.
    """
    return (
        model_path == loading
        or model_path.lower() == loading.lower()
        or _names_the_resident_model(loading, model_path)
    )


def _resident_standard_model_name(backend, model_path: str) -> str:
    """The registry name to unload for ``model_path``, or ``model_path`` when nothing matches.

    The backend refuses a name it never loaded, so a pinned load has to be evicted under the
    name it was registered with rather than the id the client shows.
    """
    active = getattr(backend, "active_model_name", None)
    if isinstance(active, str) and _names_the_resident_model(active, model_path):
        return active
    loaded = getattr(backend, "models", None)
    if isinstance(loaded, dict):
        for name in loaded:
            if isinstance(name, str) and _names_the_resident_model(name, model_path):
                return name
    return model_path


def _unload_evicts_standard_backend(backend, model_path: str) -> bool:
    """Whether ``backend.unload_model(model_path)`` will really evict something.

    The standard backend refuses to unload a name it never loaded ("don't unload
    a stale model") and returns success, so /unload for a model another tab has
    already replaced is a no-op. That must not count as a teardown: cancelling
    the running chats for it would end them and leave the resident model up.

    Mirrors the backend's own guard (case-insensitive on the active name, since
    the load path canonicalizes casing). A backend that exposes neither field is
    reported as a real unload, which keeps the previous behaviour.
    """
    active = getattr(backend, "active_model_name", _UNRESOLVED_BACKEND_STATE)
    loaded = getattr(backend, "models", _UNRESOLVED_BACKEND_STATE)
    if active is _UNRESOLVED_BACKEND_STATE and loaded is _UNRESOLVED_BACKEND_STATE:
        return True
    if isinstance(active, str) and active and active.lower() == (model_path or "").lower():
        return True
    if isinstance(active, str) and _names_the_resident_model(active, model_path):
        return True
    if not isinstance(loaded, dict):
        return False
    return model_path in loaded or any(
        isinstance(name, str) and _names_the_resident_model(name, model_path) for name in loaded
    )


def _unload_may_evict(model_path: str) -> bool:
    """Whether POST /unload for ``model_path`` can still tear something down.

    The refusal passes gate on this. A request naming a model another tab has
    already replaced reaches none of the teardown branches and returns the
    documented idempotent no-op (see _unload_evicts_standard_backend), so
    refusing it counts a teardown that cannot happen and leaves a stale tab
    unable to clear its selection. Each disjunct mirrors one teardown branch, so
    True means "some branch may fire", never "this unload succeeds".

    Attribute reads only, no lifecycle gate, so the pre-gate pass still fails
    fast on a swap that would really stop chats. A stale answer is safe in both
    directions: the gated pass re-runs this under the gate, and every branch
    re-runs the refusal at its own point of no return, so a False here can never
    let a teardown through unrefused.
    """
    backend = get_inference_backend()
    loading = getattr(backend, "get_loading_model", lambda: None)()
    if (
        loading is not None
        and hasattr(backend, "cancel_load")
        and _names_the_loading_model(loading, model_path)
    ):
        return True
    llama_backend = get_llama_cpp_backend()
    if llama_backend.is_active and (
        llama_backend.model_identifier == model_path
        or is_registered_native_path_label(llama_backend.model_identifier, model_path)
        or _names_the_resident_model(llama_backend.model_identifier, model_path)
        # Up but not serving is mid-load, evicted whatever model was named.
        or not llama_backend.is_loaded
    ):
        return True
    return _unload_evicts_standard_backend(backend, model_path)


@studio_router.get("/active-generations")
async def get_active_generations(
    fastapi_request: Request, current_subject: str = Depends(get_current_subject)
):
    """Conversations currently generating, plus how many can decode at once.

    Lets a model swap name the chats it would interrupt, including runs this tab
    cannot see (another tab, or a reload behind a proxy). parallel_slots is the
    slot count actually in use, which the VRAM fit may have cut below the
    requested --parallel; chats beyond it queue rather than fail.
    """
    entries = active_generations.snapshot()
    # A tracker's model can be a native local path (the legacy stream records active_model_name
    # verbatim); redact here, the one place that serialises it.
    for _entry in entries:
        if isinstance(_entry.get("model"), str):
            _entry["model"] = redact_native_paths(_entry["model"])
    slots = 1
    try:
        slots = _openai_llama_admission_capacity(fastapi_request, get_llama_cpp_backend())
    except Exception:
        slots = int(getattr(fastapi_request.app.state, "llama_parallel_slots", 1) or 1)
    return {
        "active": entries,
        "count": len(entries),
        "thread_ids": active_generations.active_thread_ids(),
        "parallel_slots": max(1, int(slots)),
    }


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
    return await _tunnel_safe_json(
        load_model_gated(request, fastapi_request, current_subject), label = "Model load"
    )


async def load_model_gated(request: LoadRequest, fastapi_request: Request, current_subject: str):
    """Everything ``POST /load`` does except the tunnel-safe padding.

    In-process callers (preview) must await THIS, not the route: the route's slow
    path returns a StreamingResponse nobody in-process drains, so awaiting it would
    return mid-load and hide a late failure in an unread body. This blocks until the
    model is resident and raises the real exception.
    """
    # A sidecar install that has reserved the swap must not lose to a load that
    # then gets unloaded by the pre-swap teardown. Rechecked under the gate: an
    # install can reserve while this request queues on the gate, so the pre-gate
    # check alone is only a fast path.
    from core.inference.llama_keepwarm import inference_lifecycle_gate

    _raise_if_sidecar_swap_in_progress()
    # Hold the lifecycle gate across the load so idle auto-unload can't unload the
    # model mid-load. Auto-switch calls _load_model_impl directly since it already
    # holds this gate.
    async with inference_lifecycle_gate():
        _raise_if_sidecar_swap_in_progress()
        # The active-generation gate runs inside _load_model_impl, once it knows this is a real
        # reload, and still under the lifecycle gate so the check stays atomic with the teardown.
        return await _load_model_impl(
            request,
            fastapi_request,
            current_subject,
            on_reload_confirmed = lambda *, cancel: _raise_or_cancel_active_generations(
                force = request.force_cancel_active,
                action = "Loading a model",
                cancel = cancel,
            ),
        )


async def _load_model_impl(
    request: LoadRequest,
    fastapi_request: Request,
    current_subject: str,
    *,
    current_request_counted: bool = False,
    on_reload_confirmed = None,
):
    from core.inference.llama_cpp import LlamaServerNotFoundError

    # A new load starts here; arm the progress throttle so this load's first
    # sampled step logs even if it reports 100% immediately (cached/small load).
    _reset_load_progress_step()

    # Live "loading" row: discarded if already loaded, relabelled on the real id, closed on exit.
    _load_event = api_monitor.record_lifecycle(
        event = "load",
        model = _lifecycle_model_label(request.model_path, request.gguf_variant),
        running = True,
        # Auto-switch loads run before the request row opens, so a failure leaves only this.
        via_api_key = _request_used_api_key(fastapi_request),
        # The row is shared, so name its owner or the overlay pops open in unrelated tabs.
        subject = current_subject,
    )

    native_grant_backed = False
    model_log_label = request.model_path
    gguf_load_stack = ExitStack()
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

        # Manual mode owns the offload flags. Preserve an explicit layer count
        # by translating its last-wins value into the first-class field before
        # stripping the raw flags. This keeps CLI pass-through such as
        # ``-ngl 20`` from being silently replaced by the manual default (-1).
        # The inherited path already strips offload flags. Manual + per-GPU
        # ratio owns --tensor-split the same way.
        if request.gpu_memory_mode == "manual" and extra_llama_args:
            _gpu_layers_override = parse_gpu_layers_override(extra_llama_args)
            if _gpu_layers_override is not None:
                request = request.model_copy(update = {"gpu_layers": _gpu_layers_override})
            _stripped_explicit = strip_shadowing_flags(
                extra_llama_args,
                strip_context = False,
                strip_cache = False,
                strip_spec = False,
                strip_template = False,
                strip_split_mode = False,
                strip_tensor_split = _should_strip_tensor_split(request),
                strip_offload = True,
            )
            if _stripped_explicit != extra_llama_args:
                logger.info(
                    "Manual GPU memory owns the offload flags; stripping them "
                    "from explicit llama_extra_args: %s -> %s",
                    extra_llama_args,
                    _stripped_explicit,
                )
                extra_llama_args = _stripped_explicit

        # Keep every downstream consumer on the normalized explicit list. In
        # particular, the already-loaded comparator must not compare the raw
        # request's managed offload flags against the stripped launch state.
        request = request.model_copy(update = {"llama_extra_args": extra_llama_args})

        model_identifier, model_log_label, native_grant_backed = (
            _resolve_model_identifier_for_request(request, operation = "load-model")
        )
        # Version switching is handled by the subprocess-based inference
        # backend -- no ensure_transformers_version() needed here.

        # Resolve the effective chat-template override once, up front: an
        # explicit user override, else a bundled family template (e.g. the
        # gemma-4 override that ships preserve_thinking without re-downloading
        # quants), else None. Used for both the reload-dedup check below and the
        # load_model calls, so the live backend state and the incoming request
        # compare against the same template text.
        effective_chat_template_override = resolve_effective_chat_template_override(
            model_identifier = model_identifier,
            user_override = request.chat_template_override,
        )

        # Reclaim the GPU for chat (evicting a resident Images/Video pipeline) only once the load is known viable; the
        # already-loaded fast paths below re-assert CHAT themselves. Deferred past validation so a doomed load evicts nothing.
        from core.inference.gpu_arbiter import acquire_for, current_owner, release, CHAT

        # ── Already-loaded check: skip reload if the exact model is active ──
        backend = await asyncio.to_thread(get_inference_backend)
        llama_backend = get_llama_cpp_backend()

        # Resolve once so dedupe, admission and launch use the same slot count.
        _n_parallel = _resolve_parallel_slots(request, fastapi_request)

        def _reuse_loaded_gguf(
            intent: GgufLoadIntent, *, display_name: Optional[str] = None
        ) -> Optional[LoadResponse]:
            if not (
                llama_backend.adopt_load_intent_if_matched(intent)
                and getattr(llama_backend, "_audio_probed", True)
            ):
                return None
            api_monitor.discard(_load_event)
            logger.info("Model already loaded (GGUF): %s, skipping reload", model_log_label)
            return _gguf_load_response(
                llama_backend,
                "already_loaded",
                model_log_label if native_grant_backed else llama_backend.model_identifier,
                display_name = model_log_label if native_grant_backed else display_name,
                is_local_model = _loaded_is_local_model(
                    llama_backend, native_grant_backed, llama_backend.model_identifier
                ),
            )

        is_direct_gguf_request = model_identifier.lower().endswith(".gguf")
        if llama_backend.is_loaded and (request.gguf_variant or is_direct_gguf_request):
            reused = _reuse_loaded_gguf(
                _active_gguf_intent(
                    request,
                    llama_backend,
                    model_identifier = model_identifier,
                    chat_template_override = effective_chat_template_override,
                    n_parallel = _n_parallel,
                    native_grant_backed = native_grant_backed,
                )
            )
            if reused is not None:
                # Requested GGUF chat model already resident: assert CHAT ownership (no-op when
                # held) to correct a drifted owner. Unless the resident server is a confirmed
                # zero-VRAM one, which coexists with an image/video pipeline.
                if not llama_backend.holds_no_vram:
                    await asyncio.to_thread(acquire_for, CHAT)
                return reused
        if not (request.gguf_variant or is_direct_gguf_request):
            if (
                backend.active_model_name
                and backend.active_model_name.lower() == model_identifier.lower()
            ):
                api_monitor.discard(_load_event)  # nothing loaded, no monitor row
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
                # Requested chat model already resident: assert CHAT ownership (no-op when held) to correct a drifted owner.
                await asyncio.to_thread(acquire_for, CHAT)
                return LoadResponse(
                    status = "already_loaded",
                    model = model_log_label if native_grant_backed else backend.active_model_name,
                    display_name = model_log_label
                    if native_grant_backed
                    else backend.active_model_name,
                    is_vision = _model_info.get("is_vision", False),
                    is_lora = _model_info.get("is_lora", False),
                    is_gguf = False,
                    is_local_model = native_grant_backed or is_local_path(backend.active_model_name),
                    is_audio = _model_info.get("is_audio", False),
                    audio_type = _model_info.get("audio_type"),
                    has_audio_input = _model_info.get("has_audio_input", False),
                    inference = inference_config,
                    requires_trust_remote_code = _resolve_loaded_trust_remote_code(
                        backend.active_model_name, _model_info, inference_config
                    ),
                    supports_reasoning = _sf_supports_reasoning,
                    reasoning_style = _sf_reasoning_style,
                    reasoning_effort_levels = _sf_flags.get("reasoning_effort_levels", []),
                    reasoning_always_on = _sf_flags["reasoning_always_on"],
                    supports_preserve_thinking = _sf_flags["supports_preserve_thinking"],
                    supports_tools = _sf_flags["supports_tools"],
                    context_length = _positive_int_or_none(_model_info.get("context_length")),
                    chat_template = _chat_template,
                )

        # is_lora auto-detected from adapter_config.json on disk/HF.
        # Probe wrap so offline loads skip 30-60s of soft-failed network checks before
        # the worker starts. Off-loop: the guard can spend seconds on DNS plus a HEAD and
        # its TCP fallback, and this handler is awaited directly by the route, so running
        # it inline would stall every unrelated request. Same shape as /validate.
        def _resolve_config():
            with _hf_offline_if_unreachable_for(model_identifier):
                return ModelConfig.from_identifier(
                    model_id = model_identifier,
                    hf_token = request.hf_token,
                    gguf_variant = request.gguf_variant,
                )

        # Guard and call go to the worker together: from_identifier can import transformers
        # to build the detection registry, and the guard's probe is a network round trip.
        config = await asyncio.to_thread(_resolve_config)

        if not config:
            raise HTTPException(
                status_code = 400,
                detail = f"Invalid model identifier: {model_log_label}",
            )

        # Resolve inherited extras once before command-dependent preflights.
        extra_llama_args = _resolve_inherited_extra_args(
            request,
            config,
            model_identifier,
            extra_llama_args,
            effective_chat_template_override,
        )

        # Invalid GPU IDs must fail before the training coexistence guard.
        placement = await _prepare_load_placement(config, request, extra_llama_args)
        gguf_intent: Optional[GgufLoadIntent] = None
        _tensor_intent_overall = False
        if config.is_gguf:
            gguf_intent = _resolve_gguf_load_intent(
                config,
                request,
                native_grant_backed = native_grant_backed,
                chat_template_override = effective_chat_template_override,
                extra_args = extra_llama_args,
                placement = placement,
                n_parallel = _n_parallel,
            )
            same_loaded_model = llama_backend.matches_load_source(gguf_intent)
            if same_loaded_model and config.gguf_hf_repo and llama_backend.gguf_path:
                gguf_intent = replace(
                    gguf_intent,
                    mtp_draft_path = _mtp_draft_for_path(llama_backend.gguf_path, False),
                    compare_mtp_draft = True,
                )
            _effective_tensor = _effective_tensor_parallel(
                extra_llama_args, request.tensor_parallel
            )
            _tensor_intent_overall = _effective_tensor or _carry_preserved_tensor_intent(
                preserved = getattr(llama_backend, "layer_preserves_tensor_intent", False),
                same_model = same_loaded_model,
                explicit_drop = _is_explicit_tensor_drop(request),
            )
            gguf_intent = replace(
                gguf_intent,
                preserve_multi_gpu_on_layer = (_tensor_intent_overall and not _effective_tensor),
            )
            reused = _reuse_loaded_gguf(
                gguf_intent,
                display_name = config.display_name,
            )
            if reused is not None:
                return reused

        # Config-resolved dedupe must run first: a duplicate must not refuse/cancel active chats.
        # Refusal is non-destructive; defer forced cancellation past every remaining rejection.
        if on_reload_confirmed is not None:
            on_reload_confirmed(cancel = False)
        cancel_pending = on_reload_confirmed is not None and bool(request.force_cancel_active)

        if not config.is_gguf and _mlx_distributed_launch_detected():
            raise HTTPException(
                status_code = 400,
                detail = (
                    "Unsloth does not support distributed MLX inference under "
                    "mlx.launch. Use `mlx.launch ... unsloth chat` or run Unsloth "
                    "without the distributed launcher."
                ),
            )

        # Effective quantization (LoRA can flip 4-bit -> 16-bit); guard + load reuse it.
        effective_load_in_4bit = _effective_load_in_4bit(config, request.load_in_4bit)
        if effective_load_in_4bit != request.load_in_4bit:
            logger.info(
                f"Resolved load_in_4bit={effective_load_in_4bit} for '{model_log_label}' "
                f"from adapter_config.json / base model (requested {request.load_in_4bit})"
            )
        # Latest-sidecar models load 16-bit (worker refuses bnb 4-bit); size the guard
        # to match. Off-loop: tier resolution reads configs.
        if effective_load_in_4bit and not config.is_gguf:
            from utils.transformers_version import latest_tier_active_for
            if await asyncio.to_thread(
                _offline_guarded,
                (model_identifier, config.identifier, getattr(config, "base_model", None)),
                latest_tier_active_for,
                config.identifier,
                request.hf_token,
            ):
                effective_load_in_4bit = False
                logger.info(
                    f"Latest-transformers sidecar active for '{model_log_label}' - "
                    "sizing and loading in 16-bit (4-bit is disabled for brand-new "
                    "architectures)"
                )

        # Apply the training coexistence policy before the unload step below
        # frees the resident model. Off-loop and guarded: the guard does sync HF work.
        await asyncio.to_thread(
            _offline_guarded,
            (model_identifier, config.identifier, getattr(config, "base_model", None)),
            _guard_chat_load_against_training,
            config,
            request,
            load_in_4bit = effective_load_in_4bit,
            placement = placement,
            llama_extra_args = extra_llama_args,
            n_parallel = _n_parallel,
        )

        # Mark the load and refuse one the download manager already owns BEFORE the eviction below: this 409 leaves nothing
        # loaded. It runs after argument inheritance, since a carried --no-mmproj changes the companion requirement.
        if config.is_gguf and config.gguf_hf_repo:
            from core.inference.llama_cpp import gguf_load_in_flight

            gguf_load_stack.enter_context(gguf_load_in_flight(config.gguf_hf_repo))

            from core.inference.llama_cpp import _hub_download_blocks_gguf_load

            if await asyncio.to_thread(
                _hub_download_blocks_gguf_load,
                config.gguf_hf_repo,
                config.gguf_variant,
                require_mmproj = bool(
                    config.is_vision and not extra_args_disable_mmproj(extra_llama_args)
                ),
                hf_token = request.hf_token,
            ):
                raise HTTPException(
                    status_code = 409,
                    detail = (
                        f"'{model_log_label}' is currently being downloaded "
                        "by the download manager. Wait for the download to "
                        "finish (or cancel it), then load the model."
                    ),
                )

        # Load now known viable: reclaim the GPU for chat, evicting a resident Images/Video pipeline, so a doomed load evicts
        # nothing. The marker is entered UNDER the arbiter lock, since a chat load holds no process until its GGUF lands.
        from core.inference.llama_cpp import chat_load_in_flight, zero_vram_chat_load

        # ...but only when this load will actually use the GPU, exactly as the image and video loaders gate on their device:
        # a manual gpu_layers=0 load runs on CPU, so taking the arbiter would cancel an image/video generation for nothing.
        chat_load_needs_gpu = not (
            config.is_gguf
            and await asyncio.to_thread(
                zero_vram_chat_load,
                request.gpu_memory_mode,
                request.gpu_layers,
                extra_llama_args,
                bool(config.is_vision and not extra_args_disable_mmproj(extra_llama_args)),
                request.speculative_type,
            )
        )
        if chat_load_needs_gpu:
            await asyncio.to_thread(
                acquire_for,
                CHAT,
                lambda: gguf_load_stack.enter_context(chat_load_in_flight()),
            )
        else:
            # The marker still goes up (the download-manager handshake reads it, and it keeps this load cancellable). A stale CHAT claim is dropped AFTER the load.
            gguf_load_stack.enter_context(chat_load_in_flight())

        # ── GGUF path: load via llama-server ──────────────────────
        if config.is_gguf:
            llama_backend = get_llama_cpp_backend()
            unsloth_backend = await asyncio.to_thread(get_inference_backend)

            # Fast path only: a swap can still be reserved during the drain.
            _raise_if_sidecar_swap_in_progress()

            # Drain active generations first (the lifecycle gate blocks new starts); a forced swap
            # excludes the ones it is about to cancel rather than waiting them out.
            await _wait_for_model_switch_idle(
                current_request_counted = current_request_counted,
                cancel_pending = cancel_pending,
            )
            # Decisive recheck, and the last thing that can reject this load, so it runs BEFORE the
            # cancel: rejecting after would stop every chat for nothing.
            _raise_if_sidecar_swap_in_progress()

            # Point of no return for the GGUF path: nothing left can reject this load, so stop the
            # chats the swap interrupts (or refuse, if the caller never opted in).
            if on_reload_confirmed is not None:
                on_reload_confirmed(cancel = True)

            # Let the cancelled generations unwind before the teardown; no check follows, so this cannot
            # strand a cancelled chat behind a 409. Bounded: TTS observes no cancel event, so an
            # unbounded wait would hold the gate for a whole audio run.
            if cancel_pending:
                await _wait_for_model_switch_idle(
                    current_request_counted = current_request_counted,
                    timeout_s = _POST_CANCEL_DRAIN_TIMEOUT_S,
                )

            # Unload any active Unsloth model only after every hub conflict check.
            if unsloth_backend.active_model_name:
                logger.info(
                    f"Unloading Unsloth model '{unsloth_backend.active_model_name}' before loading GGUF"
                )
                await asyncio.to_thread(
                    unsloth_backend.unload_model, unsloth_backend.active_model_name
                )

            # Every rejection and source check has completed. The immutable
            # intent resolved before teardown is now the only launch input.
            if gguf_intent is None:
                raise RuntimeError("GGUF load intent was not resolved")
            load_intent = gguf_intent

            # Run a single load attempt with the given tensor flag + extras.
            async def _attempt_gguf_load(
                tensor_parallel: bool, attempt_extra_args: Optional[list[str]]
            ) -> bool:
                attempt = replace(
                    load_intent,
                    extra_args = (
                        tuple(attempt_extra_args) if attempt_extra_args is not None else None
                    ),
                    tensor_parallel = tensor_parallel,
                    preserve_multi_gpu_on_layer = bool(
                        _tensor_intent_overall
                        and not _effective_tensor_parallel(attempt_extra_args, tensor_parallel)
                    ),
                )
                return await asyncio.to_thread(
                    llama_backend.load_model,
                    intent = attempt,
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

            # An Images/Video acquire can land in the gap between the acquire above and load_model clearing the cancel event, so
            # its cancellation is lost. Ownership survives that gap, so this load undoes itself. A zero-VRAM load never yields.
            if chat_load_needs_gpu and current_owner() != CHAT:
                await asyncio.to_thread(llama_backend.unload_model)
                raise HTTPException(
                    status_code = 409,
                    detail = (
                        "An image or video model took the GPU while this model was loading, "
                        "so the load was cancelled. Unload that model, then try again."
                    ),
                )
            if not chat_load_needs_gpu:
                # Zero-VRAM load done, so drop a now-stale CHAT claim: leaving it would make the next image/video load "evict" a server holding nothing. Owner-guarded.
                await asyncio.to_thread(release, CHAT)

            logger.info(
                f"Loaded GGUF model via llama-server: {model_log_label if native_grant_backed else config.identifier}"
            )
            _close_load_event(
                _load_event,
                model_log_label if native_grant_backed else config.identifier,
                request.gguf_variant or getattr(llama_backend, "hf_variant", None),
            )
            # Clear any idle-unload reload stash now, not only on the next poll.
            from core.inference.llama_keepwarm import note_model_loaded

            await asyncio.to_thread(note_model_loaded, llama_backend)
            # A plain load advertises its own identifier; auto-switch overwrites
            # this with the repo id right after _load_model_impl returns.
            llama_backend._openai_advertised_id = None

            # Audio detection moved into load_model under _serial_load_lock (#5642).
            _gguf_audio = llama_backend._audio_type
            _gguf_is_audio = llama_backend._is_audio
            llama_backend._native_display_label = model_log_label if native_grant_backed else None
            llama_backend._native_grant_backed = bool(native_grant_backed)
            # Provenance is a load-time fact. Re-deriving it per status poll
            # would flip a local model to remote if its directory is deleted
            # or unmounted underneath a still-running server.
            llama_backend._is_local_model = bool(native_grant_backed or config.is_local)
            if _gguf_is_audio:
                logger.info(f"GGUF model detected as audio: audio_type={_gguf_audio}")

            return _gguf_load_response(
                llama_backend,
                "loaded",
                model_log_label if native_grant_backed else config.identifier,
                display_name = model_log_label if native_grant_backed else config.display_name,
                is_local_model = config.is_local,
                inference_identifier = config.identifier,
            )

        # ── Standard path: load via Unsloth/transformers ──────────
        backend = await asyncio.to_thread(get_inference_backend)

        # Same sidecar rejection as GGUF: fast path ahead of the drain, rechecked after.
        _raise_if_sidecar_swap_in_progress()

        llama_backend = get_llama_cpp_backend()
        await _wait_for_model_switch_idle(
            current_request_counted = current_request_counted,
            cancel_pending = cancel_pending,
        )
        _raise_if_sidecar_swap_in_progress()

        # Point of no return for the Unsloth path: cancel only once nothing can still reject the load.
        if on_reload_confirmed is not None:
            on_reload_confirmed(cancel = True)

        # Let the cancelled generations unwind before the teardown; no check follows. Bounded like GGUF.
        if cancel_pending:
            await _wait_for_model_switch_idle(
                current_request_counted = current_request_counted,
                timeout_s = _POST_CANCEL_DRAIN_TIMEOUT_S,
            )
        # Unload any active GGUF model first, off-loop: a 600 GB teardown measures
        # 160s and on-loop would block _tunnel_safe_json's own padding.
        if llama_backend.is_loaded:
            logger.info("Unloading GGUF model before loading Unsloth model")
            await asyncio.to_thread(llama_backend.unload_model)

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

        # Resolved before the guard so both size the same load.
        load_in_4bit = effective_load_in_4bit

        # Load in a thread so the event loop stays free for download progress
        # polling and other requests.
        success = await asyncio.to_thread(
            backend.load_model,
            config = config,
            max_seq_length = request.max_seq_length,
            load_in_4bit = load_in_4bit,
            hf_token = request.hf_token,
            trust_remote_code = request.trust_remote_code,
            approved_remote_code_fingerprint = request.approved_remote_code_fingerprint,
            gpu_ids = placement.requested_gpu_ids,
            subject = current_subject,
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

        # Same guard the GGUF branch runs above: an Images/Video acquire can land between this load's cancellation and its publish, so this load undoes itself.
        if current_owner() != CHAT:
            await asyncio.to_thread(backend.unload_model, config.identifier)
            # The worker's base CUDA context outlives the model unload, so kill it too.
            await asyncio.to_thread(backend._shutdown_subprocess, 5.0)
            raise HTTPException(
                status_code = 409,
                detail = (
                    "An image or video model took the GPU while this model was loading, "
                    "so the load was cancelled. Unload that model, then try again."
                ),
            )

        logger.info(
            f"Loaded model: {model_log_label if native_grant_backed else config.identifier}"
        )
        _close_load_event(
            _load_event, model_log_label if native_grant_backed else config.identifier, None
        )
        # Clear any idle-unload reload stash: a manual load supersedes an idle-freed
        # GGUF, so the next /v1 request must not resurrect it. Mirror the GGUF branch
        # above; without this a non-GGUF load leaves a stale stash until the idle
        # poll clears it (and never, while idle-unload is off).
        from core.inference.llama_keepwarm import note_model_loaded

        note_model_loaded()

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

        # Report validate_model's requirement (raw auto_map OR YAML) plus the value the
        # load used, and persist it, so a later retry/rollback doesn't send
        # trust_remote_code=false for a custom-code model (and status reports it too).
        _requires_rc = _resolve_loaded_trust_remote_code(
            config.identifier,
            None,
            inference_config,
            request.hf_token,
            trust_remote_code_used = bool(getattr(request, "trust_remote_code", False)),
        )
        try:
            backend.models.setdefault(config.identifier, {})["requires_trust_remote_code"] = (
                _requires_rc
            )
        except Exception:
            pass

        return LoadResponse(
            status = "loaded",
            model = model_log_label if native_grant_backed else config.identifier,
            display_name = model_log_label if native_grant_backed else config.display_name,
            is_vision = config.is_vision,
            is_lora = config.is_lora,
            is_gguf = False,
            is_local_model = config.is_local,
            # Post-load classification (mirrored from the worker) wins here.
            is_audio = _model_info.get("is_audio", config.is_audio),
            audio_type = _model_info.get("audio_type", config.audio_type),
            has_audio_input = _model_info.get("has_audio_input", config.has_audio_input),
            inference = inference_config,
            requires_trust_remote_code = _requires_rc,
            supports_reasoning = _sf_flags["supports_reasoning"],
            reasoning_style = _sf_flags["reasoning_style"],
            reasoning_effort_levels = _sf_flags.get("reasoning_effort_levels", []),
            reasoning_always_on = _sf_flags["reasoning_always_on"],
            supports_preserve_thinking = _sf_flags["supports_preserve_thinking"],
            supports_tools = _sf_flags["supports_tools"],
            context_length = _positive_int_or_none(_model_info.get("context_length")),
            chat_template = _chat_template,
        )

    except HTTPException:
        raise
    except ValueError as e:
        redacted_msg = redact_native_paths(str(e))
        if _is_unsupported_nvfp4_inference_error(redacted_msg):
            logger.warning(
                "NVFP4 inference is not supported yet while loading '%s'",
                model_log_label,
            )
            raise HTTPException(
                status_code = 500,
                detail = _NVFP4_INFERENCE_UNSUPPORTED_MESSAGE,
            )
        if native_grant_backed:
            logger.warning(
                "Rejected inference selection for native model %s: %s",
                model_log_label,
                redacted_msg,
            )
            raise HTTPException(status_code = 400, detail = redacted_msg)
        logger.warning("Rejected inference GPU selection: %s", e)
        # User-facing validation (e.g. "Invalid gpu_ids [99]"): redact paths, keep detail.
        raise HTTPException(status_code = 400, detail = redacted_msg)
    except LlamaServerNotFoundError as e:
        # Missing GGUF runtime: 400 with the install message, not a generic 500.
        logger.warning("GGUF runtime missing while loading '%s': %s", model_log_label, e)
        raise HTTPException(status_code = 400, detail = str(e))
    except Exception as e:
        from utils.transformers_version import SidecarSwapInProgress

        if isinstance(e, SidecarSwapInProgress):
            # Lost the spawn-time race to a sidecar install/repair: retryable 409.
            raise HTTPException(status_code = 409, detail = str(e))
        # Friendlier message for models Unsloth cannot load.
        redacted_msg = redact_native_paths(str(e))
        if _is_unsupported_nvfp4_inference_error(redacted_msg):
            logger.warning(
                "NVFP4 inference is not supported yet while loading '%s'",
                model_log_label,
            )
            raise HTTPException(
                status_code = 500,
                detail = _NVFP4_INFERENCE_UNSUPPORTED_MESSAGE,
            )
        if native_grant_backed:
            logger.error(
                "Error loading native model %s: %s",
                model_log_label,
                redacted_msg,
            )
            msg = _maybe_unsupported_message(redacted_msg)
            raise HTTPException(
                status_code = 500,
                detail = f"Failed to load native model {model_log_label}: {msg}",
            )
        logger.error(f"Error loading model: {e}", exc_info = True)
        msg = _maybe_unsupported_message(redacted_msg)
        raise HTTPException(status_code = 500, detail = f"Failed to load model: {msg}")
    finally:
        gguf_load_stack.close()
        # Catch-all: an error or cancelled load would otherwise leave the row "loading".
        api_monitor.fail_open(_load_event, "Load did not complete")


def _any_remote(targets) -> bool:
    """True unless every target is a local path. Falsy entries are skipped (no base to
    read); anything unresolvable counts as remote, since guarding a local read costs one
    memoised verdict while missing a remote one costs the retry backoff."""
    from utils.paths import is_local_path

    for target in (targets,) if isinstance(targets, str) else targets or ():
        if not target:
            continue  # no base is not an unknown base: nothing to read, nothing to guard
        try:
            if not (isinstance(target, str) and is_local_path(target)):
                return True
        except Exception:
            return True  # unresolvable: guard, since missing a remote read costs the backoff
    return False


def _offline_guarded(targets, fn, /, *args, **kwargs):
    """Run one blocking preflight inside the same forced-offline window as config
    resolution. The config is not the only remote read here: the upgrade, trust-remote-code
    and sizing preflights each fetch raw metadata, and would otherwise burn the retry
    backoff the guard exists to skip. The verdict is memoised, so this costs no extra
    probe. Call from a worker thread: the guard is process-global and blocks on a cold
    verdict.

    ``targets`` is what this call actually READS, not the outer request, because a local
    adapter can resolve to a remote base and the base is what gets fetched. Positional-only,
    so a wrapped call's own model_identifier kwarg cannot collide."""
    from contextlib import nullcontext

    # The module-level symbol, not a fresh import: route tests patch
    # routes.inference._hf_offline_if_unreachable to stay deterministic, and a local
    # re-import would bypass the patch and run a real probe.
    ctx = _hf_offline_if_unreachable() if _any_remote(targets) else nullcontext()
    with ctx:
        return fn(*args, **kwargs)


def _requires_trust_remote_code_for_model(
    model_identifier: str, hf_token: Optional[str] = None
) -> bool:
    """Whether loading this model would execute custom repo code, so the consent
    dialog must run first. True if the Unsloth YAML default enables
    ``trust_remote_code`` OR the raw config declares an ``auto_map`` (Hub/local,
    config.json or tokenizer_config.json). Reads raw JSON only; never imports
    model code."""
    from utils.inference import load_inference_config

    try:
        if bool(load_inference_config(model_identifier).get("trust_remote_code", False)):
            return True
    except Exception:
        pass
    try:
        from utils.security.consent import _config_has_auto_map
        return _config_has_auto_map(model_identifier, hf_token) is True
    except Exception:
        return False


def _resolve_loaded_trust_remote_code(
    model_id,
    model_info,
    inference_config,
    hf_token = None,
    trust_remote_code_used = False,
) -> bool:
    """TRC requirement to report for an ALREADY-LOADED model, consistent with
    ``validate_model``.

    ``validate_model`` reports ``requires_trust_remote_code`` from
    ``_requires_trust_remote_code_for_model`` (YAML default OR raw ``auto_map``), but
    the load / already-loaded / status responses historically reported only the YAML
    default. That dropped raw-``auto_map`` models: after approving and loading one, the
    response said ``false``, so the frontend stored ``false`` and a later retry/rollback
    sent ``trust_remote_code=false`` and failed.

    Resolution order: a value stored on the model at load time (so a status refresh does
    not re-derive it) -> the trust_remote_code the load actually used -> the YAML default
    -> the raw ``auto_map`` check (reads the loaded model's cached config; no network)."""
    stored = (model_info or {}).get("requires_trust_remote_code")
    if stored is not None:
        return bool(stored)
    if trust_remote_code_used or bool((inference_config or {}).get("trust_remote_code", False)):
        return True
    try:
        return bool(_requires_trust_remote_code_for_model(model_id, hf_token))
    except Exception:
        return False


def _requires_security_review_for_model(
    model_identifier: str, hf_token: Optional[str] = None
) -> bool:
    """Whether Hugging Face's security scan flagged unsafe files for this repo, so
    the consent dialog must open as a hard block before loading. Metadata-only;
    never downloads the flagged files. Fails open (False) on any error."""
    try:
        from utils.security import evaluate_file_security, security_load_subdirs
        return evaluate_file_security(
            model_identifier,
            hf_token,
            load_subdirs = security_load_subdirs(model_identifier, hf_token),
        ).blocked
    except Exception:
        return False


@router.post("/validate", response_model = ValidateModelResponse)
async def validate_model(
    request: ValidateModelRequest,
    fastapi_request: Request = None,
    current_subject: str = Depends(get_current_subject),
):
    """
    Lightweight validation endpoint for model identifiers.

    Checks that ModelConfig.from_identifier() can resolve model_path, but does
    NOT load model weights into GPU memory.
    """
    from core.inference.llama_cpp import (
        LlamaServerNotFoundError,
        _hf_offline_if_unreachable_for,
    )

    native_grant_backed = False
    model_log_label = request.model_path
    try:
        model_identifier, model_log_label, native_grant_backed = (
            _resolve_model_identifier_for_request(request, operation = "validate-model")
        )

        # The frontend validates before it loads, so this needs the same guard as
        # /load; otherwise the stall just moves here and /load is never reached.
        # Off-loop twice over: the guard is a network round trip, and the first
        # from_identifier builds the detection registry (transformers, or the warm's lock).
        def _resolve_config():
            with _hf_offline_if_unreachable_for(model_identifier):
                return ModelConfig.from_identifier(
                    model_id = model_identifier,
                    hf_token = request.hf_token,
                    gguf_variant = request.gguf_variant,
                )

        config = await asyncio.to_thread(_resolve_config)

        if not config:
            raise HTTPException(
                status_code = 400,
                detail = f"Invalid model identifier: {model_log_label}",
            )

        effective_extra_args = _resolve_inherited_extra_args(
            request, config, model_identifier, None
        )

        # Apply the same placement policy as /load before the frontend unloads
        # the current model.
        placement = await _prepare_load_placement(config, request, effective_extra_args)
        effective_load_in_4bit = _effective_load_in_4bit(config, request.load_in_4bit)

        # Both checks cover the [adapter, base] set (matching the scan route and workers):
        # either repo can ship auto_map code or a poisoned pickle.
        security_targets = [config.identifier]
        try:
            from utils.models.model_config import get_base_model_from_lora_identifier

            # Resolve a LOCAL or REMOTE adapter's base so its code/weights are reviewed too.
            _base = await asyncio.to_thread(
                _offline_guarded,
                model_identifier,
                get_base_model_from_lora_identifier,
                model_identifier,
                request.hf_token,
            )
            if _base:
                security_targets.append(_base)
        except Exception:
            pass
        security_targets = list(dict.fromkeys(security_targets))

        is_gguf = getattr(config, "is_gguf", False)
        # Does a newer transformers ship this model_type? Static overlay first, cached
        # PyPI/main snapshot only for unknown types. Never fails validation; run before
        # the training guard so an installable upgrade sizes as 16-bit.
        transformers_upgrade: Optional[TransformersUpgradeInfo] = None
        if not is_gguf:
            from utils.transformers_latest import check_upgrade_for_model

            # Cover [adapter, base]: the worker activates transformers for the base model.
            for _target in security_targets:
                _upgrade = await asyncio.to_thread(
                    _offline_guarded,
                    _target,
                    check_upgrade_for_model,
                    _target,
                    request.hf_token,
                )
                if _upgrade is not None:
                    transformers_upgrade = TransformersUpgradeInfo(**_upgrade)
                    break

        # Whether the model can load on the CURRENT transformers through its own remote
        # code (auto_map, or the YAML trust default). Computed before the 16-bit flip
        # because a model with this fallback still loads 4-bit without the offered install,
        # exactly as /load does.
        requires_trust_remote_code = False
        if not is_gguf:
            # Reads raw config/tokenizer JSON, so guarded and off-loop like the rest.
            requires_trust_remote_code = await asyncio.to_thread(
                _offline_guarded,
                security_targets,
                lambda: any(
                    _requires_trust_remote_code_for_model(_t, request.hf_token)
                    for _t in security_targets
                ),
            )

        # Mirror /load's latest-sidecar 16-bit flip so the guard sizes it the same way. An
        # ALREADY-ACTIVE latest sidecar always forces 16-bit (the worker will). A merely
        # OFFERED (not yet installed) upgrade forces 16-bit only when the model has NO
        # custom-code fallback: with auto_map it still loads 4-bit on the current
        # transformers (as /load does without a successful install), and the install route
        # refuses while training is active, so sizing 16-bit here would 409 the only viable
        # 4-bit path. /load re-sizes 16-bit after a successful install and re-guards there.
        if effective_load_in_4bit and not is_gguf:
            from utils.transformers_version import latest_tier_active_for
            _install_only_upgrade = (
                transformers_upgrade is not None
                and transformers_upgrade.supported_in_pypi
                and transformers_upgrade.pypi_version
                and not requires_trust_remote_code
            )
            if _install_only_upgrade or await asyncio.to_thread(
                _offline_guarded,
                (model_identifier, config.identifier, getattr(config, "base_model", None)),
                latest_tier_active_for,
                config.identifier,
                request.hf_token,
            ):
                effective_load_in_4bit = False
        # A metadata-only probe reads the GGUF header and allocates no VRAM, so the
        # training guard must not refuse it. Real loads omit include_context_length /
        # include_chat_template, and /load applies the guard again.
        if not (request.include_context_length or request.include_chat_template):
            # Off-loop and guarded: the guard does sync nvidia-smi / HF work.
            await asyncio.to_thread(
                _offline_guarded,
                (model_identifier, config.identifier, getattr(config, "base_model", None)),
                _guard_chat_load_against_training,
                config,
                request,
                load_in_4bit = effective_load_in_4bit,
                placement = placement,
                llama_extra_args = effective_extra_args,
                n_parallel = _resolve_parallel_slots(request, fastapi_request),
            )

        # A selected GGUF loads via llama.cpp: auto_map Python and root pickle weights in a
        # mixed repo are inert for this load, so gating on them is a false positive. Only
        # run the security preflight for non-GGUF loads (requires_trust_remote_code was
        # already resolved above for the sizing flip).
        requires_security_review = False
        if not is_gguf:
            # _fetch_security_status does hf_model_info with 10s and 20s timeouts, so this
            # needs the same window and worker thread as the preflights above.
            requires_security_review = await asyncio.to_thread(
                _offline_guarded,
                security_targets,
                lambda: any(
                    _requires_security_review_for_model(_t, request.hf_token)
                    for _t in security_targets
                ),
            )
        # Native context length, read from the local GGUF header when present.
        # Lets the staged ("Load on selection" off) flow populate the context
        # slider before the GPU load; None until the file is downloaded.
        # Staged header dims (one read): native context, total layer count, and
        # MoE expert-layer count -- let the staged flow size the context, GPU-
        # layers and manual --n-cpu-moe sliders before the load.
        context_length: Optional[int] = None
        layer_count: Optional[int] = None
        moe_layer_count: Optional[int] = None
        chat_template: Optional[str] = None
        # Both header probes read the same local GGUF, so resolve it once.
        if (request.include_context_length or request.include_chat_template) and is_gguf:
            from hub.utils.gguf import resolve_local_gguf_path
            from picker.schemas import MAX_CHAT_TEMPLATE_BYTES
            from utils.models.gguf_metadata import (
                read_gguf_chat_template,
                read_gguf_staged_dims,
            )

            # Best-effort: a header-read failure must never fail validation of an
            # otherwise-valid model (the outer except turns it into a 400).
            try:
                if native_grant_backed:
                    # model_identifier is the resolved canonical .gguf path.
                    local_gguf = model_identifier
                else:
                    # Local folder / exported GGUFs already have their file
                    # resolved on the config (gguf_file is None for HF repos, so
                    # those fall back to the HF-cache lookup).
                    local_gguf = config.gguf_file or resolve_local_gguf_path(
                        model_identifier, request.gguf_variant
                    )
                if local_gguf:
                    if request.include_context_length:
                        # Header walk reads tokenizer arrays (tens of ms); keep it
                        # off the event loop.
                        dims = await asyncio.to_thread(read_gguf_staged_dims, local_gguf)
                        if dims:
                            context_length = dims["context_length"]
                            layer_count = dims["layer_count"]
                            moe_layer_count = dims["moe_layer_count"]
                    if request.include_chat_template:
                        # Read only the leased GGUF's own embedded template (the copy
                        # llama.cpp loads), never a sibling sidecar: the native grant
                        # authorizes just this path, so neighbours would be scope escalation.
                        raw_template = await asyncio.to_thread(read_gguf_chat_template, local_gguf)
                        if (
                            raw_template is not None
                            and len(raw_template.encode("utf-8")) <= MAX_CHAT_TEMPLATE_BYTES
                        ):
                            chat_template = raw_template
            except Exception as e:
                logger.debug("Header probe failed for %s: %s", model_log_label, e)

        return ValidateModelResponse(
            valid = True,
            message = "Model identifier is valid.",
            identifier = model_log_label if native_grant_backed else config.identifier,
            display_name = model_log_label
            if native_grant_backed
            else getattr(config, "display_name", config.identifier),
            is_gguf = is_gguf,
            is_diffusion = is_gguf and placement.diffusion_kind is True,
            # An unavailable header is inconclusive, not proof of an ordinary GGUF.
            diffusion_unknown = is_gguf and placement.diffusion_kind is None,
            is_lora = getattr(config, "is_lora", False),
            is_vision = getattr(config, "is_vision", False),
            requires_trust_remote_code = requires_trust_remote_code,
            requires_security_review = requires_security_review,
            context_length = context_length,
            layer_count = layer_count,
            moe_layer_count = moe_layer_count,
            chat_template = chat_template,
            requires_transformers_upgrade = transformers_upgrade is not None,
            transformers_upgrade = transformers_upgrade,
        )

    except HTTPException:
        raise
    except LlamaServerNotFoundError as e:
        # Missing GGUF runtime: 400 with the install message, not a generic "Invalid model".
        logger.warning("GGUF runtime missing while validating '%s': %s", request.model_path, e)
        raise HTTPException(status_code = 400, detail = str(e))
    except Exception as e:
        redacted_msg = redact_native_paths(str(e))
        if is_hf_authentication_error(e):
            raise HTTPException(
                status_code = 400,
                detail = (
                    "Hugging Face authentication failed. Check or clear the token "
                    "in Settings, and confirm access to this gated repository."
                ),
            )
        if _is_unsupported_nvfp4_inference_error(redacted_msg):
            logger.warning(
                "NVFP4 inference is not supported yet while validating '%s'",
                model_log_label,
            )
            raise HTTPException(
                status_code = 400,
                detail = _NVFP4_INFERENCE_UNSUPPORTED_MESSAGE,
            )
        if native_grant_backed:
            logger.error(
                "Error validating native model %s: %s",
                model_log_label,
                redacted_msg,
            )
            msg = _maybe_unsupported_message(redacted_msg)
            raise HTTPException(
                status_code = 400,
                detail = f"Invalid native model {model_log_label}: {msg}",
            )
        logger.error(
            f"Error validating model identifier '{request.model_path}': {e}",
            exc_info = True,
        )
        # RuntimeError / ValueError carry intentional, actionable messages here
        # (e.g. "llama-server binary not found - cannot load GGUF models. Run
        # setup.sh ..."), so surface them instead of a blank "Invalid model".
        # Path-redact for safety and keep any other exception type generic so an
        # unexpected internal error never leaks its details to the client.
        if isinstance(e, (RuntimeError, ValueError)):
            msg = redacted_msg.strip()
            if msg:
                msg = _maybe_unsupported_message(msg)
                raise HTTPException(
                    status_code = 400,
                    detail = msg,
                )
        raise HTTPException(
            status_code = 400,
            detail = "Invalid model",
        )


# studio_router only: admin action, kept off the OpenAI-compatible /v1 mount.
@studio_router.post(
    "/install-latest-transformers", response_model = InstallLatestTransformersResponse
)
async def install_latest_transformers_route(
    request: InstallLatestTransformersRequest, current_subject: str = Depends(get_current_subject)
):
    """
    Consented install of the latest transformers release into the persistent
    .venv_t5_latest sidecar.

    Called after the user confirms the transformers-upgrade dialog raised by /validate
    (requires_transformers_upgrade). The requested version must match the current latest
    PyPI release (re-verified server-side); the sidecar then participates in routing on
    this and every future start. A pip install runs off-loop, so this can take a minute.
    """
    from utils.transformers_latest import install_latest_transformers
    from utils.transformers_version import end_sidecar_swap, try_begin_sidecar_swap

    # The install stage-and-swaps .venv_t5_latest in place; a live worker would
    # lazy-import from the new version mid-run, mixing incompatible modules. Gate on
    # worker LIVENESS not tier (no HF token here, so tier re-resolution is unreliable
    # for gated repos): training and export are refused, the chat model unloaded.
    # Reserve the swap FIRST, before any await: training/export starts check this
    # reservation, so raising it after the gate wait would let a worker slip in.
    if not try_begin_sidecar_swap():
        raise HTTPException(
            status_code = 409,
            detail = "A transformers installation is already in progress.",
        )
    # Until the installer thread takes over, this coroutine owns the reservation
    # and must release it on any early exit (the 409 refusals below).
    owns_reservation = True
    try:
        from core.export import get_export_backend
        from core.training import get_training_backend

        if get_training_backend().is_training_active():
            raise HTTPException(
                status_code = 409,
                detail = (
                    "A training run is active. Wait for it to finish before "
                    "installing a new transformers version."
                ),
            )
        _export = get_export_backend()
        if _export.is_export_active():
            raise HTTPException(
                status_code = 409,
                detail = (
                    "An export is running. Wait for it to finish before "
                    "installing a new transformers version."
                ),
            )
        # A loaded (idle) export checkpoint would be torn down by the pre-swap
        # cleanup; if the swap then failed, that state would be silently lost
        # with no rollback signal. Make the user unload it deliberately first.
        if getattr(_export, "current_checkpoint", None):
            raise HTTPException(
                status_code = 409,
                detail = (
                    "An export checkpoint is loaded. Unload it from the Export "
                    "page before installing a new transformers version."
                ),
            )
        # In-flight streams passed the middleware already, so the lifecycle gate can't
        # protect them and the swap's unload would kill them mid-stream; mirror the
        # auto-switch busy check. This route is not middleware-counted and pending
        # requests stay blocked in the middleware, so neither is subtracted here.
        from core.inference.llama_keepwarm import (
            inference_lifecycle_gate,
            note_model_unloaded,
            other_inference_request_count,
        )

        # A confirmed swap skips only this fast path; the recheck under the gate still has to pass,
        # so the guard is unchanged for anyone who did not confirm.
        if (
            not request.force_cancel_active
            and other_inference_request_count(current_request_counted = False, include_pending = False)
            > 0
        ):
            raise HTTPException(
                status_code = 409,
                detail = (
                    "Another inference request is in progress. Wait for it to "
                    "finish before installing a new transformers version."
                ),
            )

        # Hold the lifecycle gate /load holds so no HF worker can start (or be mid-load
        # with active_model_name unset) while the sidecar is swapped. Teardown runs via
        # before_swap, only once the staged install succeeded: a failed pip/compat check
        # must not leave the user with their model gone. GGUF stays loaded (llama-server
        # never imports transformers).
        backend = await asyncio.to_thread(get_inference_backend)
        export_backend = get_export_backend()

        unloaded_chat = {"v": False}

        def _unload_before_swap() -> None:
            # Runs on the install thread, inside the gate held by _gated_install. Any
            # failure raises so the previous sidecar stays untouched (a worker that did
            # not tear down cleanly may still lazy-import from it). Export teardown runs
            # FIRST so its failure aborts while the chat model is still loaded;
            # cleanup_memory shuts the subprocess down even when its command fails, so
            # judge by worker liveness, not its return value.
            export_backend.cleanup_memory()
            export_alive = getattr(export_backend, "is_worker_alive", None)
            if callable(export_alive) and export_alive():
                raise RuntimeError("Export worker still alive before the transformers swap")
            active = getattr(backend, "active_model_name", None)
            if active:
                if not backend.unload_model(active):
                    # A failed unload still clears the orchestrator's model state,
                    # so the model is gone from the parent's view even though the
                    # swap aborts: report it so the client rolls back instead of
                    # pointing at an unloaded model.
                    if getattr(backend, "active_model_name", None) != active:
                        unloaded_chat["v"] = True
                        note_model_unloaded()
                    raise RuntimeError(f"Could not unload '{active}' before the transformers swap")
                note_model_unloaded()
                unloaded_chat["v"] = True
                logger.info(
                    "Unloaded '%s' before swapping in transformers %s",
                    active,
                    request.version,
                )
            # A failed load can leave a live worker with no active model that
            # still holds sidecar modules (and blocks the rename on Windows).
            worker_alive = getattr(backend, "is_worker_alive", None)
            if callable(worker_alive) and worker_alive():
                # _shutdown_subprocess keeps the handle when the worker outlives SIGKILL,
                # so both its False result and the liveness recheck catch a survivor
                # rather than the recheck being fooled by a nulled handle.
                stopped = backend._shutdown_subprocess()
                if not stopped or worker_alive():
                    raise RuntimeError("Inference worker still alive before the transformers swap")

        def _run_install() -> dict:
            # Owns the reservation from here: releasing in the thread, not the route,
            # keeps it held if the request is cancelled while the install still stages.
            try:
                return install_latest_transformers(request.version, _unload_before_swap, True)
            finally:
                end_sidecar_swap()

        # Snapshot before waiting on the gate: a /load already holding it can
        # complete meanwhile (including a same-model reload with new settings),
        # and the installer must not unload a model whose successful LoadResponse
        # the client is about to render. The generation counter catches reloads
        # the name alone would miss.
        active_before_gate = (
            getattr(backend, "active_model_name", None),
            getattr(backend, "load_generation", 0),
        )

        async def _gated_install() -> dict:
            # Held by THIS task, not the request coroutine: a cancelled POST unwinding an
            # `async with` here would drop the only guard /load honors mid-install.
            async with inference_lifecycle_gate():
                _active_now = (
                    getattr(backend, "active_model_name", None),
                    getattr(backend, "load_generation", 0),
                )
                if _active_now != active_before_gate:
                    end_sidecar_swap()
                    raise HTTPException(
                        status_code = 409,
                        detail = (
                            "A model load completed while the install was waiting. "
                            "Retry the install."
                        ),
                    )
                # Carry a confirmed swap's decision through: the user already accepted the "stop N
                # chats" prompt, and refusing here would make that answer unactionable (Retry
                # cannot succeed while the same chats run). Deliberately LAST, after every check
                # that can still reject the install, so the cancel is spent only once nothing can
                # turn this request away -- /load's rule.
                if request.force_cancel_active:
                    await _cancel_and_drain_for_sidecar_swap()
                # Recheck under the gate: new streams bump their in-flight count while
                # holding it, so once held nothing slips past. A forced install that could
                # not drain in time lands here too, for the same 409 as without the flag.
                if (
                    other_inference_request_count(
                        current_request_counted = False, include_pending = False
                    )
                    > 0
                ):
                    end_sidecar_swap()
                    raise HTTPException(
                        status_code = 409,
                        detail = (
                            "Another inference request is in progress. Wait for "
                            "it to finish before installing a new transformers "
                            "version."
                        ),
                    )
                return await asyncio.to_thread(_run_install)

        install_task = asyncio.ensure_future(_gated_install())
        owns_reservation = False
        # shield: a cancelled request stops waiting, but the installer runs to
        # completion (holding the gate) instead of being torn down mid-swap.
        result = await asyncio.shield(install_task)
    finally:
        if owns_reservation:
            end_sidecar_swap()
    if not result["success"]:
        if result.get("latest_version"):
            # Structured failure so the dialog can update to the newer release
            # and offer a retry that can actually succeed.
            return InstallLatestTransformersResponse(**result, model_unloaded = unloaded_chat["v"])
        if unloaded_chat["v"]:
            # The chat model is already gone even though the swap failed; return a
            # structured failure (not a bare 400) so the client can restore its
            # model state instead of pointing at an unloaded model.
            return InstallLatestTransformersResponse(**result, model_unloaded = True)
        raise HTTPException(status_code = 400, detail = result["message"])
    return InstallLatestTransformersResponse(**result, model_unloaded = unloaded_chat["v"])


@router.post("/unload", response_model = UnloadResponse)
async def unload_model(request: UnloadRequest, current_subject: str = Depends(get_current_subject)):
    """Unload a model from memory.

    Padded like /load: a 600 GB GGUF teardown measures 160s, past the proxy timer.
    See ``_tunnel_safe_json``. No in-process caller today; a future one must await
    ``_unload_model_impl`` (as preview awaits ``load_model_gated``), since the padded
    path returns a StreamingResponse, not the payload.
    """
    return await _tunnel_safe_json(
        _unload_model_impl(request, current_subject), label = "Model unload"
    )


async def _unload_model_impl(request: UnloadRequest, current_subject: str):
    """
    Unload a model from memory.
    Routes to the correct backend (llama-server for GGUF, Unsloth otherwise).
    """
    # A deliberate unload means "stay unloaded": drop any idle reload stash so the
    # next /v1 request can't resurrect this model. The idle loop unloads via the
    # backend directly (not this route), so clearing here never fights keep-warm.
    from core.inference.llama_keepwarm import inference_lifecycle_gate, note_model_unloaded
    try:
        # "Stop loading" (frontend cancelLoading -> /unload) must abort a still-loading
        # model promptly, and /load holds the lifecycle gate for the whole load. cancel_load only
        # tears the loading subprocess down, so it is safe off-gate -- and ahead of the
        # active-generation refusal below, which it can never need (see there).
        backend = await asyncio.to_thread(get_inference_backend)
        loading = getattr(backend, "get_loading_model", lambda: None)()
        if (
            loading is not None
            and hasattr(backend, "cancel_load")
            and _names_the_loading_model(loading, request.model_path)
        ):
            # Cancel under the name the load runs as, which a pinned row states as a path.
            if await asyncio.to_thread(backend.cancel_load, loading):
                note_model_unloaded()
                logger.info(f"Cancelled in-flight load: {request.model_path}")
                return UnloadResponse(status = "unloaded", model = request.model_path)

        # Same "stop loading" fast path for a still-loading GGUF (spawned, health check not passed).
        # unload_model() sets the cancel_event load_model polls and kills the child without a
        # worker command, so it is safe off-gate like cancel_load; the gated branch below handles
        # the already-loaded case. Gated on the loading model so an unload for a different model
        # cannot cancel this load.
        llama_backend = get_llama_cpp_backend()
        if (
            llama_backend.is_active
            and not llama_backend.is_loaded
            and (
                llama_backend.model_identifier == request.model_path
                or is_registered_native_path_label(
                    llama_backend.model_identifier, request.model_path
                )
                or _names_the_resident_model(llama_backend.model_identifier, request.model_path)
            )
        ):
            await asyncio.to_thread(llama_backend.unload_model)
            note_model_unloaded()
            logger.info(f"Cancelled in-flight GGUF load: {request.model_path}")
            return UnloadResponse(status = "unloaded", model = request.model_path)

        # Same gate as /load: refusal only, so a non-forced unload fails fast before queueing on the
        # lifecycle gate. Skipped when no teardown branch can fire, or a request naming a model
        # another tab already replaced would 409 on chats it cannot interrupt.
        #
        # BEHIND the two "stop loading" fast paths above: both cancel a load that has not replaced
        # anything yet, so neither can interrupt a chat, and refusing them counted a teardown that
        # cannot happen (unretryably -- the frontend's Cancel sends this unload unforced and drops
        # the error). Any other name still falls through here.
        if await asyncio.to_thread(_unload_may_evict, request.model_path):
            _raise_or_cancel_active_generations(
                force = request.force_cancel_active,
                action = "Unloading the model",
                cancel = False,
            )

        # Serialize with /load under the same lifecycle gate: the Unsloth unload now runs
        # off the event loop (asyncio.to_thread), so without this a concurrent /load could
        # swap in a fresh subprocess mid-unload and the unload command would land on the
        # new worker. The gate makes load and unload exclusive.
        async with inference_lifecycle_gate():
            # Rechecked under the gate, like /load: a chat can register while this one queues here (the
            # middleware takes and releases the same gate). Still refusal only, and re-read rather
            # than carried down, since a load may have finished meanwhile.
            if await asyncio.to_thread(_unload_may_evict, request.model_path):
                _raise_or_cancel_active_generations(
                    force = request.force_cancel_active,
                    action = "Unloading the model",
                    cancel = False,
                )
            # Check if the GGUF backend has this model loaded or is loading it.
            llama_backend = get_llama_cpp_backend()
            if llama_backend.is_active and (
                llama_backend.model_identifier == request.model_path
                or is_registered_native_path_label(
                    llama_backend.model_identifier, request.model_path
                )
                or _names_the_resident_model(llama_backend.model_identifier, request.model_path)
                or not llama_backend.is_loaded
            ):
                # Read the identity before teardown clears it, so the row reads repo:QUANT.
                _unloaded = _llama_public_model_id(llama_backend, request.model_path)
                _unloaded_variant = getattr(llama_backend, "hf_variant", None)
                # Point of no return: this really does replace the running server, so stop the
                # chats. A manual unload is a deliberate user action, so it cancels mid-stream
                # requests rather than deferring to them the way the automatic idle loop does.
                _raise_or_cancel_active_generations(
                    force = request.force_cancel_active, action = "Unloading the model"
                )
                # Let what we just cancelled unwind first, like /load: tearing the server down under
                # streams told to stop but not yet finished turned a clean end into a dropped
                # connection. Bounded, since a manual unload is deliberate.
                await _drain_and_recancel_before_teardown(
                    force = request.force_cancel_active, action = "Unloading the model"
                )
                # Off-loop like the in-flight branch above: a 160s teardown on the
                # loop would block this route's own padding.
                await asyncio.to_thread(llama_backend.unload_model)
                note_model_unloaded()
                api_monitor.record_lifecycle(
                    event = "unload",
                    model = _lifecycle_model_label(_unloaded, _unloaded_variant),
                    reason = "manual",
                )
                logger.info(f"Unloaded GGUF model: {request.model_path}")
                return UnloadResponse(status = "unloaded", model = request.model_path)

            # Unload from Unsloth backend off the event loop: unload takes _gen_lock, which
            # a slow SSE stream paused between tokens still holds, so a sync call would block
            # the loop that drives the stream's next token and the lock release.
            backend = await asyncio.to_thread(get_inference_backend)
            if _unload_evicts_standard_backend(backend, request.model_path):
                # Point of no return for the standard path, same rule as above.
                _raise_or_cancel_active_generations(
                    force = request.force_cancel_active, action = "Unloading the model"
                )
                await _drain_and_recancel_before_teardown(
                    force = request.force_cancel_active, action = "Unloading the model"
                )
            await asyncio.to_thread(
                backend.unload_model, _resident_standard_model_name(backend, request.model_path)
            )
            note_model_unloaded()
            api_monitor.record_lifecycle(
                event = "unload",
                model = _lifecycle_model_label(request.model_path),
                reason = "manual",
            )
            logger.info(f"Unloaded model: {request.model_path}")
            return UnloadResponse(status = "unloaded", model = request.model_path)

    except HTTPException:
        # Typed refusals (the gate's 409) must not be rewritten as a 500 below.
        raise
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


@studio_router.get("/monitor")
async def get_api_monitor(current_subject: str = Depends(get_current_subject)):
    """Return recent OpenAI-compatible API activity for Unsloth."""
    # Off-loop: both helpers reach get_inference_backend(), whose first call waits on
    # hardware detection, and this is polled from first paint.
    active_model, context_length, queue = await asyncio.to_thread(
        lambda: (_monitor_active_model(), _monitor_context_length(), _monitor_queue_state())
    )
    active_requests = api_monitor.active_count(subject = current_subject)
    if active_requests:
        operating_status = "generating"
    elif active_model:
        operating_status = "ready"
    else:
        operating_status = "idle"
    # With request logging off, ``snapshot()`` returns an empty list -- the same shape
    # as a Studio that simply hasn't served a request yet. Signal the disabled state so
    # the UI can explain the empty list instead of claiming there was no API traffic.
    return {
        "status": operating_status,
        # The clock every entry's started_at is on. The monitor dates its first snapshot
        # against this, since the browser's clock need not agree over a tunnel.
        "server_time": time.time(),
        "active_model": active_model,
        "context_length": context_length,
        "active_requests": active_requests,
        "queue": queue,
        "logging_enabled": api_monitor.enabled,
        "entries": api_monitor.snapshot(include_details = False, subject = current_subject),
    }


@studio_router.delete("/monitor")
async def clear_api_monitor(current_subject: str = Depends(get_current_subject)):
    """Drop this caller's recorded API history so a debugging session starts clean.

    Scoped to the current subject, like every read on the monitor: an unscoped
    wipe would erase another user's history and zero their active-request count
    while their generation is still streaming.

    The caller's own in-flight requests are dropped from the log too; they keep
    streaming to their client, they just stop being reported here (a later append
    re-adds nothing, since the entry id no longer resolves).
    """
    api_monitor.clear(subject = current_subject)
    return {"cleared": True}


@studio_router.get("/monitor/{entry_id}")
async def get_api_monitor_entry(entry_id: str, current_subject: str = Depends(get_current_subject)):
    """Return full prompt/reply details for one OpenAI-compatible API request."""
    entry = api_monitor.get(entry_id, subject = current_subject)
    if entry is None:
        raise HTTPException(status_code = 404, detail = "Monitor entry not found")
    return entry


@router.post("/generate/stream")
async def generate_stream(
    request: GenerateRequest,
    fastapi_request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """
    Generate a chat response with Server-Sent Events (SSE) streaming.

    For vision models, provide image_base64 (base64-encoded image).
    """
    backend = await asyncio.to_thread(get_inference_backend)

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

    cancel_event = threading.Event()

    async def stream():
        gen = None
        completed = False
        # Cancel the generation when the client disconnects. The generator only
        # awaits asyncio.to_thread(next, gen, ...), so without a concurrent
        # watcher a disconnect during a long prefill/generation would go
        # unnoticed until the next send and the backend would keep generating.
        disconnect_watcher = asyncio.create_task(
            _await_disconnect_then_cancel(fastapi_request, cancel_event)
        )
        # Registered inside the generator, under the finally that unregisters it, so a response whose
        # body never starts leaves nothing behind. Unregistered, this run passes /unload's 409 gate
        # (which runs no idle drain) and a forced swap has no event to signal. GenerateRequest
        # carries no thread_id: counted, not nameable.
        _tracker = _TrackedCancel(cancel_event, model = backend.active_model_name)
        _tracker.__enter__()
        try:
            gen = backend.generate_chat_response(
                messages = request.messages,
                system_prompt = request.system_prompt,
                image = image,
                temperature = request.temperature,
                top_p = request.top_p,
                top_k = request.top_k,
                min_p = request.min_p,
                max_new_tokens = request.max_new_tokens,
                repetition_penalty = request.repetition_penalty,
                presence_penalty = request.presence_penalty,
                cancel_event = cancel_event,
            )
            _DONE = object()
            while True:
                if cancel_event.is_set():
                    # Watcher set cancel_event between chunks. Reset here: closing
                    # the generator does not signal a subprocess backend, so it would
                    # keep decoding. The finally's reset is guarded, so no double-run.
                    backend.reset_generation_state(cancel_event)
                    break
                chunk = await asyncio.to_thread(next, gen, _DONE)
                if chunk is _DONE:
                    completed = True
                    break
                if isinstance(chunk, GenStreamError):
                    yield f"data: {json.dumps({'error': _friendly_gen_stream_error(chunk)})}\n\n"
                    yield "data: [DONE]\n\n"
                    return
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            if completed:
                yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            cancel_event.set()
            backend.reset_generation_state(cancel_event)
            raise
        except Exception as e:
            cancel_event.set()
            backend.reset_generation_state(cancel_event)
            logger.error(f"Error during generation: {e}", exc_info = True)
            yield f"data: {json.dumps({'error': _friendly_error(e)})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            # Nested so a teardown failure still unregisters; a phantom entry 409s swaps.
            try:
                await _stop_local_disconnect_cancel_watcher(disconnect_watcher)
                if not completed and not cancel_event.is_set():
                    cancel_event.set()
                    backend.reset_generation_state(cancel_event)
                if gen is not None:
                    try:
                        await asyncio.to_thread(gen.close)
                    except (RuntimeError, ValueError):
                        pass
            finally:
                _tracker.__exit__(None, None, None)

    return _sse_streaming_response(stream())


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
            # Fail open on inconclusive probes: False means a definitive
            # "binary lacks MTP" to API consumers.
            _supports_mtp = bool(
                _caps.get("supports_mtp", False)
                or (_caps.get("found", False) and _caps.get("mtp_probe_inconclusive", False))
            )
        except Exception:
            _bin = None
            _supports_mtp = False  # no usable binary: MTP genuinely unavailable
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
            # is_local_model below needs the flag; the helper reports identities, not provenance.
            _native_grant_backed = getattr(llama_backend, "_native_grant_backed", False)
            # Shared with /chat/count_tokens, so a client can tell whose tokenizer counted.
            _display_model_id, _reported_model_identifier = _llama_status_model_ids(llama_backend)
            _inference_cfg = load_inference_config(_model_id) if _model_id else None
            # Don't surface Unsloth's auto-applied bundled family template (e.g. the
            # gemma-4 override) as a user-authored override: the frontend adopts
            # status.chat_template_override as editable state and would otherwise
            # re-send it as an explicit override for a later, unrelated model. Only
            # expose a genuine user override.
            _reported_chat_template_override = llama_backend.chat_template_override
            _auto_chat_template_override = resolve_effective_chat_template_override(
                model_identifier = _model_id,
                user_override = None,
            )
            if (
                _auto_chat_template_override is not None
                and _reported_chat_template_override == _auto_chat_template_override
            ):
                _reported_chat_template_override = None
            return InferenceStatusResponse(
                active_model = _display_model_id,
                model_identifier = _reported_model_identifier,
                is_gguf = True,
                is_local_model = _loaded_is_local_model(
                    llama_backend, _native_grant_backed, _model_id
                ),
                gguf_variant = llama_backend.hf_variant,
                loading = [],
                loaded = [_display_model_id] if _display_model_id else [],
                inference = _inference_cfg,
                **_llama_runtime_fields(llama_backend),
                chat_template_override = _reported_chat_template_override,
                requested_context_length = llama_backend.requested_n_ctx,
                llama_cpp_supports_mtp = _supports_mtp,
                spec_fallback_reason = llama_backend.spec_fallback_reason,
                llama_cpp_prebuilt_stale = _stale,
                llama_cpp_installed_tag = _installed_tag,
                llama_cpp_latest_tag = _latest_tag,
            )

        # Otherwise report Unsloth backend status. Peek rather than build: no singleton means
        # nothing is loaded, and the chat UI polls this from first paint.
        backend = _peek_inference_backend()
        if backend is None:
            return InferenceStatusResponse(
                llama_cpp_supports_mtp = _supports_mtp,
                llama_cpp_prebuilt_stale = _stale,
                llama_cpp_installed_tag = _installed_tag,
                llama_cpp_latest_tag = _latest_tag,
            )

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
            is_local_model = bool(
                backend.active_model_name and is_local_path(backend.active_model_name)
            ),
            is_audio = is_audio,
            audio_type = audio_type,
            has_audio_input = has_audio_input,
            loading = list(getattr(backend, "loading_models", set())),
            loaded = list(backend.models.keys()),
            inference = inference_config,
            requires_trust_remote_code = _resolve_loaded_trust_remote_code(
                backend.active_model_name, model_info, inference_config
            ),
            supports_reasoning = _sf_flags["supports_reasoning"],
            reasoning_style = _sf_flags["reasoning_style"],
            reasoning_effort_levels = _sf_flags.get("reasoning_effort_levels", []),
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


_load_progress_lock = threading.Lock()
_last_load_progress_step = -1


def _log_load_progress_step(fraction, phase):
    """One inference_load_progress line per 10% step, so a model load shows
    progress without a line per poll. Reset per load by _reset_load_progress_step."""
    global _last_load_progress_step
    step = int(max(0.0, min(float(fraction), 1.0)) * 10)
    with _load_progress_lock:
        prev = _last_load_progress_step
        if step == prev:
            return
        _last_load_progress_step = step
        if step < prev:
            return  # load regressed/restarted mid-poll; resync without logging
    logger.info("inference_load_progress", phase = phase or "", percent = step * 10)


def _reset_load_progress_step():
    """Arm the throttle for a new load so its first sampled step always logs,
    even a cached load that already reports fraction=1.0 on the first poll."""
    global _last_load_progress_step
    with _load_progress_lock:
        _last_load_progress_step = -1


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
        resp = LoadProgressResponse(**progress)
        _log_load_progress_step(resp.fraction, resp.phase)
        return resp
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

    # Restore an idle-evicted GGUF before selecting a backend: this path is
    # keep-warm-tracked but had no reload hook, so a standalone idle TTL could
    # unload an audio GGUF the next request then failed to restore. Validation
    # above ran first, so an invalid request never triggers a reload.
    #
    # Reload-only on purpose: a local GGUF's audio-input capability is not a cheap
    # pre-load probe (the companion mmproj signal can't tell an audio projector
    # from a vision one, and codec-based TTS ships no projector at all), so passing
    # the client model through the resolver could load a text- or vision-only target
    # and evict the working audio model before the audio backend check fails. Only
    # the idle-stash restore runs here; switching TTS models is an explicit /load.
    await _maybe_auto_switch_model(_RELOAD_ONLY_MODEL, request, current_subject)

    # Created before the backend pick so the GGUF lambda can close over it; the registration
    # that arms it is below, once the model name is known.
    _audio_cancel = threading.Event()

    # Pick backend — both return (wav_bytes, sample_rate)
    llama_backend = get_llama_cpp_backend()
    if llama_backend.is_loaded and getattr(llama_backend, "_is_audio", False):
        # Advertised repo id after an auto-switch load, else a clean public id,
        # never the absolute .gguf path.
        model_name = _llama_public_model_id(llama_backend)
        _audio_model_id = getattr(llama_backend, "model_identifier", None) or model_name
        gen = lambda: llama_backend.generate_audio_response(
            text = text,
            audio_type = llama_backend._audio_type,
            temperature = payload.temperature,
            top_p = payload.top_p,
            top_k = payload.top_k,
            min_p = payload.min_p,
            max_new_tokens = _effective_max_tokens(payload) or 2048,
            repetition_penalty = payload.repetition_penalty,
            cancel_event = _audio_cancel,
        )
    else:
        backend = await asyncio.to_thread(get_inference_backend)
        if not backend.active_model_name:
            raise HTTPException(status_code = 400, detail = "No model loaded.")
        model_info = backend.models.get(backend.active_model_name, {})
        if not model_info.get("is_audio"):
            raise HTTPException(status_code = 400, detail = "Active model is not an audio model.")
        model_name = public_model_id(backend.active_model_name)
        _audio_model_id = getattr(backend, "active_model_name", None) or model_name
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

    # Apply per-model recommended sampling + any operator UNSLOTH_SAMPLING_* pin before
    # generating, so `unsloth run --temperature` (and the other pins) and per-model
    # recommendations reach audio (TTS) generation too, not just chat. The gen lambdas read
    # payload.* lazily at call time, so filling here takes effect; this covers both the direct
    # /audio/generate route and the chat-completions audio branches that delegate here.
    _fill_recommended_sampling_openai(payload, _audio_model_id)

    # TTS holds the model for the whole request, so unregistered a non-forced swap counted zero
    # generations and tore the model down mid-generation. The GGUF path observes the event; the
    # subprocess backend blocks on its response queue with no cancel plumbing, so there it is
    # only advisory -- which is why the swap drains are bounded. No cancel keys: /cancel
    # addresses streams, and this route has none.
    with _TrackedCancel(
        _audio_cancel,
        thread_id = getattr(payload, "thread_id", None),
        model = model_name,
        kind = "audio",
    ):
        # Stop in the UI aborts the fetch and nothing more, and this route has no cancel id to
        # address, so without watching the disconnect llama-server kept generating for the rest
        # of the request timeout after the chat had already reported it stopped.
        _audio_watcher = asyncio.create_task(_await_disconnect_then_cancel(request, _audio_cancel))
        try:
            wav_bytes, sample_rate = await asyncio.to_thread(gen)
        except Exception as e:
            if _audio_cancel.is_set():
                raise HTTPException(status_code = 499, detail = "Audio generation cancelled")
            logger.error(f"Audio generation error: {e}", exc_info = True)
            raise HTTPException(status_code = 500, detail = safe_error_detail(e))
        finally:
            await _stop_local_disconnect_cancel_watcher(_audio_watcher)

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
# Speech-to-text (STT) sidecar  (/audio/transcribe, /audio/stt/*)
# =====================================================================


def _resolve_stt_engine(engine: Optional[str]) -> str:
    """Normalize the requested STT engine name; default is Transformers."""
    normalized = (engine or "transformers").strip().lower()
    if normalized in ("", "transformers", "whisper"):
        return "transformers"
    if normalized in ("gguf", "ggml", "whisper_cpp", "whisper.cpp"):
        return "gguf"
    if normalized in ("mtmd", "llama_cpp", "llama.cpp"):
        return "mtmd"
    raise HTTPException(
        status_code = 422,
        detail = f"Unknown STT engine '{engine}'. Use 'transformers', 'gguf', or 'mtmd'.",
    )


def _resolve_serving_stt_engine(engine: Optional[str]) -> str:
    """Resolve the engine that will actually serve a model.

    whisper.cpp (gguf) only accepts curated ids, which Transformers serves too,
    so when whisper-server is not installed (the common case: `unsloth studio
    update` does not yet build it) fall back to Transformers instead of 501-ing
    on every recording. Used for download/load/transcribe; unload targets a
    specific engine via _resolve_stt_engine.
    """
    resolved = _resolve_stt_engine(engine)
    if resolved == "gguf":
        from core.inference import stt_ggml_sidecar
        if not stt_ggml_sidecar.is_available():
            return "transformers"
    # mtmd models exist in no other engine, so there is nothing to fall back to.
    return resolved


def _stt_download_module(engine: str):
    """Module owning download/status for an engine."""
    if engine == "mtmd":
        from core.inference import stt_mtmd_sidecar
        return stt_mtmd_sidecar
    if engine == "gguf":
        from core.inference import stt_ggml_sidecar
        return stt_ggml_sidecar
    from core.inference import stt_sidecar

    return stt_sidecar


def _stt_sidecar_for(engine: str):
    """The sidecar serving an engine. One resolver, shared with the orchestrator."""
    from core.inference import stt_registry
    return stt_registry.sidecar_for(engine)


def _stt_lifecycle() -> tuple:
    """(load, unload) for dictation models, off the orchestrator when it exists.

    Same object Model Hub loads a chat model with, so one thing knows everything
    resident. Its methods only forward to `stt_registry`, so a cold process
    calls that directly rather than constructing an orchestrator (which blocks
    on hardware detection) to load a model that never touches the chat worker.
    """
    from core.inference import stt_registry
    from core.inference.orchestrator import peek_inference_backend

    backend = peek_inference_backend()
    if backend is None:
        return stt_registry.load, stt_registry.unload
    return backend.load_stt_model, backend.unload_stt_model


@studio_router.get("/audio/stt/status")
async def stt_status(
    model: Optional[str] = None, current_subject: str = Depends(get_current_subject)
):
    """Report STT availability and which model, if any, is resident.

    ``model`` extends the Transformers ``downloaded_models`` check to a
    custom Hugging Face repository beyond the curated defaults.
    """
    from core.inference import stt_ggml_sidecar, stt_mtmd_sidecar, stt_sidecar
    from core.inference.stt_sidecar import (
        DEFAULT_STT_MODEL,
        STT_MODELS,
        get_stt_sidecar,
        is_available,
    )

    sidecar = get_stt_sidecar()
    ggml = stt_ggml_sidecar.get_ggml_stt_sidecar()
    mtmd = stt_mtmd_sidecar.get_mtmd_stt_sidecar()
    transformers_downloaded = [
        model_id for model_id in STT_MODELS if stt_sidecar.is_model_downloaded(model_id)
    ]
    if model and model not in STT_MODELS and stt_sidecar.is_model_downloaded(model):
        transformers_downloaded.append(model)
    return JSONResponse(
        content = {
            "available": is_available(),
            "loaded_model": sidecar.loaded_model,
            "loading": sidecar.is_loading(),
            "device": sidecar.device,
            "keep_alive_seconds": sidecar.keep_alive_seconds,
            "default_model": DEFAULT_STT_MODEL,
            "models": list(STT_MODELS.keys()),
            # Transformers engine, same shape as "gguf" below so clients read
            # either generically. Top-level fields above kept for old clients.
            "transformers": {
                "available": is_available(),
                "loaded_model": sidecar.loaded_model,
                "loading": sidecar.is_loading(),
                "device": sidecar.device,
                "keep_alive_seconds": sidecar.keep_alive_seconds,
                "default_model": DEFAULT_STT_MODEL,
                "models": list(STT_MODELS.keys()),
                "downloaded_models": transformers_downloaded,
                "download": stt_sidecar.download_status(),
            },
            # llama.cpp (mtmd) engine: non-Whisper ASR models.
            "mtmd": {
                "available": stt_mtmd_sidecar.is_available(),
                "loaded_model": mtmd.loaded_model,
                "loading": mtmd.is_loading(),
                "device": mtmd.device,
                "keep_alive_seconds": mtmd.keep_alive_seconds,
                "default_model": None,
                "models": list(stt_mtmd_sidecar.MTMD_STT_MODELS),
                "downloaded_models": [
                    model_id
                    for model_id in stt_mtmd_sidecar.MTMD_STT_MODELS
                    if stt_mtmd_sidecar.is_model_downloaded(model_id)
                ],
                "download": stt_mtmd_sidecar.download_status(),
            },
            # whisper.cpp (GGUF) engine.
            "gguf": {
                "available": stt_ggml_sidecar.is_available(),
                "loaded_model": ggml.loaded_model,
                "loading": ggml.is_loading(),
                "device": ggml.device,
                "keep_alive_seconds": ggml.keep_alive_seconds,
                "default_model": stt_ggml_sidecar.DEFAULT_GGML_STT_MODEL,
                "models": list(stt_ggml_sidecar.GGML_STT_MODELS.keys()),
                "downloaded_models": [
                    model_id
                    for model_id in stt_ggml_sidecar.GGML_STT_MODELS
                    if stt_ggml_sidecar._cached_model_path(model_id) is not None
                ],
                "download": stt_ggml_sidecar.download_status(),
            },
        }
    )


@studio_router.post("/audio/stt/download")
async def stt_download(
    payload: SttLoadRequest,
    current_subject: str = Depends(get_current_subject),
    hf_token: Optional[str] = Depends(get_hf_token),
):
    """Start a background download of a dictation model.

    Both engines download directly (a GGML checkpoint is a single file the Model
    Hub's GGUF variant planner cannot express; a Transformers checkpoint is a
    whole snapshot). Progress is reported by /audio/stt/status.
    """
    from core.inference import stt_ggml_sidecar, stt_sidecar
    from core.inference.stt_sidecar import (
        SttModelCompatibilityError,
        SttModelIdError,
        validate_remote_model,
    )

    engine = _resolve_serving_stt_engine(payload.engine)
    module = _stt_download_module(engine)
    try:
        # Transformers accepts custom `owner/model` repos, so confirm the repo is
        # a Whisper checkpoint (metadata-only) before snapshot_download pulls a
        # possibly-large non-STT repo into the shared cache. Curated ids
        # short-circuit; GGUF and mtmd accept curated ids only, so they skip it.
        if engine == "transformers":
            validated = await asyncio.to_thread(validate_remote_model, payload.model, hf_token)
            # Pin the download to the commit that was just validated so the
            # repo cannot be swapped between validation and snapshot_download.
            await asyncio.to_thread(
                module.start_model_download,
                payload.model,
                hf_token,
                validated.get("revision"),
            )
        else:
            await asyncio.to_thread(module.start_model_download, payload.model, hf_token)
    except SttModelIdError as e:
        raise HTTPException(status_code = 422, detail = str(e))
    except SttModelCompatibilityError as e:
        raise HTTPException(status_code = 422, detail = str(e))
    return JSONResponse(content = module.download_status())


@studio_router.post("/audio/stt/download/cancel")
async def stt_download_cancel(
    payload: Optional[SttLoadRequest] = None, current_subject: str = Depends(get_current_subject)
):
    """Stop an in-flight dictation model download.

    Partial files stay cached, so the same download resumes. Cancelling when
    nothing is downloading is a no-op, so a double click cannot fail.
    """
    from core.inference import stt_ggml_sidecar, stt_sidecar

    engine = _resolve_serving_stt_engine(payload.engine if payload else None)
    module = _stt_download_module(engine)
    cancelled = await asyncio.to_thread(module.cancel_model_download)
    # This request's result last: download_status() carries its own historical
    # "cancelled", which would otherwise report a no-op as a cancellation.
    return JSONResponse(content = {**module.download_status(), "cancelled": cancelled})


@studio_router.post("/audio/stt/load")
async def stt_load(payload: SttLoadRequest, current_subject: str = Depends(get_current_subject)):
    """Load the selected STT model after the user starts local dictation."""
    from core.inference.stt_sidecar import (
        SttLoadCancelledError,
        SttModelBusyError,
        SttModelCompatibilityError,
        SttModelIdError,
        SttModelNotDownloadedError,
        SttUnavailableError,
        get_stt_sidecar,
    )

    engine = _resolve_serving_stt_engine(payload.engine)
    sidecar = _stt_sidecar_for(engine)
    load_stt, _ = _stt_lifecycle()
    try:
        await asyncio.to_thread(load_stt, payload.model, engine)
    except SttModelNotDownloadedError as e:
        raise HTTPException(status_code = 409, detail = str(e))
    except SttUnavailableError as e:
        raise HTTPException(status_code = 501, detail = str(e))
    except SttLoadCancelledError as e:
        raise HTTPException(status_code = 409, detail = str(e))
    except SttModelBusyError as e:
        raise HTTPException(status_code = 409, detail = str(e))
    except SttModelIdError as e:
        raise HTTPException(status_code = 422, detail = str(e))
    except SttModelCompatibilityError as e:
        raise HTTPException(status_code = 422, detail = str(e))
    except Exception as e:
        logger.error(f"STT load error: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = safe_error_detail(e))
    return JSONResponse(content = {"loaded_model": sidecar.loaded_model, "device": sidecar.device})


@studio_router.post("/audio/stt/validate")
async def stt_validate(
    payload: SttLoadRequest,
    current_subject: str = Depends(get_current_subject),
    hf_token: Optional[str] = Depends(get_hf_token),
):
    """Verify a Hub repository is a Whisper checkpoint before downloading it."""
    from core.inference.stt_sidecar import (
        SttModelCompatibilityError,
        SttModelIdError,
        validate_remote_model,
    )

    try:
        result = await asyncio.to_thread(validate_remote_model, payload.model, hf_token)
    except (SttModelIdError, SttModelCompatibilityError) as e:
        raise HTTPException(status_code = 422, detail = str(e))
    return JSONResponse(content = result)


@studio_router.post("/audio/stt/unload")
async def stt_unload(
    engine: Optional[str] = None, current_subject: str = Depends(get_current_subject)
):
    """Release the local STT model when dictation is idle.

    Without an engine, both sidecars unload so an engine switch in Voice
    settings always frees whichever backend was resident.
    """
    if engine is None:
        engines = None
    else:
        # Use the serving resolver: a "gguf" pick without whisper-server is
        # actually served by the Transformers fallback, so unload must target
        # that same engine or the resident model is never freed.
        engines = [_resolve_serving_stt_engine(engine)]
    # Every engine is attempted even if one raises, so failing to free one never
    # skips the other (both can be resident after a switch).
    _, unload_stt = _stt_lifecycle()
    failed: list[str] = await asyncio.to_thread(unload_stt, engines)
    if failed:
        raise HTTPException(
            status_code = 500,
            detail = f"Failed to unload STT engine(s): {', '.join(failed)}",
        )
    return JSONResponse(content = {"loaded_model": None, "device": None})


async def _transcribe_audio_bytes(
    raw: bytes,
    model: Optional[str],
    language: Optional[str],
    fast: bool,
    engine: Optional[str] = None,
) -> JSONResponse:
    """Run STT for already-decoded request bytes."""
    from core.inference.stt_sidecar import (
        SttAudioDecodeError,
        SttAudioTooLongError,
        SttLanguageError,
        SttLoadCancelledError,
        SttModelBusyError,
        SttModelCompatibilityError,
        SttModelIdError,
        SttModelNotDownloadedError,
        SttUnavailableError,
    )

    if not raw:
        raise HTTPException(status_code = 400, detail = "Audio is empty.")
    if len(raw) > _MAX_AUDIO_RAW_BYTES:
        raise HTTPException(status_code = 413, detail = "Audio is too large.")

    sidecar = _stt_sidecar_for(_resolve_serving_stt_engine(engine))
    try:
        result = await asyncio.to_thread(
            sidecar.transcribe,
            raw,
            model,
            language,
            fast,
        )
    except SttUnavailableError as e:
        raise HTTPException(status_code = 501, detail = str(e))
    except SttLoadCancelledError as e:
        raise HTTPException(status_code = 409, detail = str(e))
    except SttModelNotDownloadedError as e:
        raise HTTPException(status_code = 409, detail = str(e))
    except SttModelBusyError as e:
        # Another client switching the dictation model is ordinary concurrency,
        # so say retry rather than report a server failure.
        raise HTTPException(status_code = 409, detail = str(e))
    except SttModelIdError as e:
        raise HTTPException(status_code = 422, detail = str(e))
    except SttModelCompatibilityError as e:
        raise HTTPException(status_code = 422, detail = str(e))
    except SttLanguageError as e:
        raise HTTPException(status_code = 422, detail = str(e))
    except SttAudioTooLongError as e:
        raise HTTPException(status_code = 413, detail = str(e))
    except SttAudioDecodeError as e:
        raise HTTPException(status_code = 400, detail = str(e))
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info = True)
        raise HTTPException(status_code = 500, detail = safe_error_detail(e))
    return JSONResponse(content = result)


@studio_router.post("/audio/transcribe")
async def transcribe_audio(
    payload: TranscribeRequest, current_subject: str = Depends(get_current_subject)
):
    """Transcribe dictation audio to text via the STT sidecar.

    Runs alongside the chat model without evicting it, so any model (including
    text-only ones) can be driven by voice.
    """
    b64 = payload.audio or ""
    if not b64:
        raise HTTPException(status_code = 400, detail = "No audio provided.")
    if len(b64) > _MAX_AUDIO_B64_CHARS:
        raise HTTPException(status_code = 413, detail = "Audio is too large.")
    try:
        raw = base64.b64decode(b64, validate = True)
    except Exception:
        raise HTTPException(status_code = 400, detail = "Audio is not valid base64.")
    return await _transcribe_audio_bytes(
        raw, payload.model, payload.language, payload.fast, payload.engine
    )


@studio_router.post("/audio/transcribe/raw")
async def transcribe_audio_raw(
    request: Request,
    model: Optional[str] = None,
    language: Optional[str] = None,
    fast: bool = False,
    engine: Optional[str] = None,
    current_subject: str = Depends(get_current_subject),
):
    """Transcribe a raw audio body without base64 or JSON conversion overhead."""
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > _MAX_AUDIO_RAW_BYTES:
            raise HTTPException(status_code = 413, detail = "Audio is too large.")
        chunks.append(chunk)
    return await _transcribe_audio_bytes(b"".join(chunks), model, language, fast, engine)


# =====================================================================
# OpenAI-Compatible Chat Completions  (/chat/completions)
# =====================================================================


def _decode_audio_base64(b64: str) -> np.ndarray:
    """Decode base64 audio (any format) → float32 numpy array at 16kHz."""
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

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim = 0, keepdim = True)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq = sr, new_freq = 16000)
        waveform = resampler(waveform)

    return waveform.squeeze(0).numpy()


# Reject oversized audio before decoding. base64 inflates raw bytes by ~4/3, so
# cap the encoded length to bound the upload. _MAX_AUDIO_SECONDS additionally
# bounds the *decoded* length, since a small compressed file (opus/flac/etc.)
# can expand to a far larger PCM array than the encoded-size cap implies.
_MAX_AUDIO_RAW_BYTES = STT_AUDIO_RAW_MAX_BYTES
_MAX_AUDIO_B64_CHARS = STT_AUDIO_B64_MAX_CHARS
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
             Unsloth-internal tool cards from a prior native Gemini turn;
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

    def _openai_responses_part(item: Any) -> Optional[dict[str, Any]]:
        """Rebuild a forwarded OpenAI Responses assistant part (`reasoning` or
        `image_generation_call`); returns None for any other part type."""
        if item.type == "reasoning":
            reasoning: dict[str, Any] = {
                "type": "reasoning",
                "id": item.id,
                "summary": item.summary,
            }
            if item.status:
                reasoning["status"] = item.status
            return reasoning
        if item.type == "image_generation_call":
            image_ref: dict[str, Any] = {"type": "image_generation_call", "id": item.id}
            if getattr(item, "response_id", None):
                image_ref["response_id"] = item.response_id
            return image_ref
        return None

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
                    elif (
                        openai
                        and msg.role == "assistant"
                        and (_rp := _openai_responses_part(part)) is not None
                    ):
                        # ExternalProviderClient maps image_generation_call onto a
                        # top-level Responses input item after the current user
                        # prompt, or onto `previous_response_id` when response_id
                        # is available from the prior turn.
                        parts.append(_rp)
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
                    elif (
                        openai
                        and msg.role == "assistant"
                        and (_rp := _openai_responses_part(p)) is not None
                    ):
                        preserved.append(_rp)
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
    payload: ChatCompletionRequest,
    request: Request,
    current_subject: Optional[str] = None,
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
    monitor_id = None
    if not getattr(request.state, "skip_api_monitor", False):
        monitor_id = api_monitor.start(
            endpoint = request.url.path,
            via_api_key = _request_used_api_key(request),
            method = request.method,
            model = model,
            prompt = _monitor_prompt_from_messages(payload.messages),
            context_length = None,
            subject = current_subject,
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
            stream_failed = False
            async for line in gen:
                monitor_event = _monitor_openai_sse_line(monitor_id, line)
                if monitor_event is None:
                    try:
                        _monitor_openai_chunk(monitor_id, json.loads(line))
                    except Exception:
                        pass
                if monitor_event == "error":
                    stream_failed = True
                yield f"{line}\n\n"
                # Parsed from the line itself, not from monitor_event: with the
                # monitor disabled the helper returns None for every line, and
                # trusting it would append a second [DONE] after the provider's.
                if _is_openai_sse_done(line):
                    sent_done = True
            if not sent_done:
                if not stream_failed:
                    api_monitor.finish(monitor_id)
                yield "data: [DONE]\n\n"
        except asyncio.CancelledError:
            api_monitor.finish(monitor_id, "cancelled")
            raise
        except Exception as exc:
            logger.error("external_provider.stream_error", error = str(exc))
            api_monitor.fail(monitor_id, _friendly_error(exc))
            # Surface the failure: a bare EOF (e.g. after a read timeout) is treated
            # by the chat client as success, saving a partial answer with no error.
            yield (
                "data: "
                + json.dumps({"error": {"message": _friendly_error(exc), "type": "server_error"}})
                + "\n\n"
            )
            yield "data: [DONE]\n\n"
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


def _fill_recommended_sampling_openai(payload, model_id) -> None:
    """Apply per-model recommended sampling (and any operator UNSLOTH_SAMPLING_* pin) to a
    ChatCompletionRequest in place.

    Only the sampling fields the client did NOT explicitly send (tracked via
    ``model_fields_set``) are overwritten, so a client that sets a field stays byte-identical
    unless an operator pins it. Fields with neither a recommendation nor a pin keep their
    existing (schema-default) value.
    """
    from utils.inference.inference_config import resolve_effective_sampling, SAMPLING_FIELD_NAMES

    explicit = {
        f: (getattr(payload, f) if f in payload.model_fields_set else None)
        for f in SAMPLING_FIELD_NAMES
    }
    effective = resolve_effective_sampling(model_id, explicit)
    for field, value in effective.items():
        setattr(payload, field, value)


# /v1/completions is proxied to llama-server verbatim; its repetition knob is "repeat_penalty",
# and every other sampling field keeps its name (mirrors _build_passthrough_payload).
_COMPLETIONS_SAMPLING_BODY_KEY = {"repetition_penalty": "repeat_penalty"}


def _fill_recommended_sampling_completions(body: dict, model_id) -> None:
    """Apply per-model recommended sampling (and any operator UNSLOTH_SAMPLING_* pin) to a raw
    ``/v1/completions`` body in place, so the legacy (non-chat) endpoint honors the same pins as
    ``/v1/chat/completions``.

    Unlike :func:`_fill_recommended_sampling_openai`, which fills a ChatCompletionRequest whose
    schema already carries per-field defaults, this body is proxied to llama-server as-is. A field
    with no operator pin, client value, or per-model recommendation is therefore left untouched
    (``fill_defaults = False``) so llama-server keeps its own default rather than being forced onto
    this schema's value. llama-server names the repetition knob ``repeat_penalty``, so read and
    write that alias for the client-sent value and any pin.
    """
    from utils.inference.inference_config import resolve_effective_sampling, SAMPLING_FIELD_NAMES

    explicit = {f: body.get(_COMPLETIONS_SAMPLING_BODY_KEY.get(f, f)) for f in SAMPLING_FIELD_NAMES}
    effective = resolve_effective_sampling(model_id, explicit, fill_defaults = False)
    for field, value in effective.items():
        body[_COMPLETIONS_SAMPLING_BODY_KEY.get(field, field)] = value


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
        # External provider: this request won't touch the local GGUF, so drop it
        # from the keep-warm count or its in-flight stream would falsely block a
        # concurrent local model switch from proceeding.
        from core.inference.llama_keepwarm import untrack_current_request

        untrack_current_request(request.scope)
        # Bypass Permissions suppresses the confirm gate, so do not reject a
        # request that sets both flags (effective confirm is then False).
        if (
            payload.confirm_tool_calls
            and not payload.bypass_permissions
            and (
                payload.enable_tools is True
                or bool(payload.enabled_tools)
                or bool(payload.tools)
                or bool(payload.openai_code_exec_container_id)
                or bool(payload.anthropic_code_exec_container_id)
            )
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
        return await _proxy_to_external_provider(payload, request, current_subject)

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

    # Reject a system-only chat before any automatic load so an invalid request
    # never swaps or reloads the resident model (as /responses and /messages
    # already validate before switching). Gate on every automatic-load trigger,
    # not just auto-switch, since a standalone idle TTL can also reload here.
    # Parse once and reuse below.
    _pre_parsed = None
    _needs_vision = False
    if _automatic_model_load_may_run():
        _pre_parsed = _extract_content_parts(payload.messages)
        if not _pre_parsed[1]:
            raise HTTPException(
                status_code = 400, detail = "At least one non-system message is required."
            )
        # Reject confirm-without-stream local tool requests before the switch: the
        # local tool path requires stream=true for the confirm gate, so this shape
        # is invalid and must not evict the resident model first.
        #
        # Enter the local-loop arm exactly when the passthrough router below would
        # run Unsloth's own tool loop. That gate is `_tools_on or _mcp_allowed`
        # (see the use_tools block): _effective_enable_tools (which lets a
        # process-wide --enable-tools policy force the loop on) plus mcp_enabled
        # honoring --disable-tools, and tool_choice="none" disabling it unless the
        # request explicitly asked. enabled_tools never enters loop entry (it only
        # filters which tools run), so it is not a signal here.
        #
        # But a policy-forced loop must not steal client-tool passthrough: when the
        # request did not explicitly ask for the loop (enable_tools/mcp) and carries
        # client tools, the router forwards to the provider branch, so only treat it
        # as the local loop when the request explicitly asked OR there is no client
        # passthrough to defer to.
        from state.tool_policy import get_tool_policy as _get_tool_policy_pre

        _cli_policy_pre = _get_tool_policy_pre()
        _use_tools_intent = _effective_enable_tools(payload) or (
            bool(payload.mcp_enabled) and _cli_policy_pre is not False
        )
        if payload.tool_choice == "none" and not _explicit_studio_tool_loop_requested(payload):
            _use_tools_intent = False
        _client_tool_passthrough = (
            bool(payload.tools)
            or bool(payload.openai_code_exec_container_id)
            or bool(payload.anthropic_code_exec_container_id)
            # A JSON-schema response_format is guided-decoding structured output the
            # router forwards to the llama-server passthrough, not Unsloth's tool
            # loop, so a --enable-tools policy must not 400 it as a local-confirm
            # request under ask/auto.
            or bool(_extract_response_format(payload))
        )
        # permission_mode only implies the confirm gate for that local loop.
        # Client-tool passthrough forwards to the provider branch and the validator
        # intentionally leaves confirm_tool_calls unset there, so only an explicit
        # confirm_tool_calls=True should force the local-confirm rejection for it.
        _studio_local_tool_loop = bool(_use_tools_intent) and (
            _explicit_studio_tool_loop_requested(payload) or not _client_tool_passthrough
        )
        if (
            not payload.bypass_permissions
            and not payload.stream
            and (
                (_confirm_gate_needs_stream(payload) and _studio_local_tool_loop)
                or (payload.confirm_tool_calls is True and _client_tool_passthrough)
            )
        ):
            raise HTTPException(
                status_code = 400,
                detail = openai_error_body(
                    "confirm_tool_calls requires stream=true for local tool execution.",
                    status = 400,
                    code = "invalid_request_error",
                    param = "confirm_tool_calls",
                ),
            )
        # Reject a malformed tool_choice forcing object before the switch: a
        # {"type": "function", "function": {}} with no name would otherwise be
        # forwarded to llama-server and rejected only after the model swapped.
        _tc = payload.tool_choice
        if isinstance(_tc, dict) and _tc.get("type") == "function":
            _tc_fn = _tc.get("function")
            _tc_name = _tc_fn.get("name") if isinstance(_tc_fn, dict) else None
            if not isinstance(_tc_name, str) or not _tc_name.strip():
                raise HTTPException(
                    status_code = 400,
                    detail = openai_error_body(
                        "Invalid 'tool_choice': the forced function must have a 'name'.",
                        status = 400,
                        code = "invalid_value",
                        param = "tool_choice",
                    ),
                )
        # Reject an oversized audio upload before the switch: the size cap is a
        # cheap, target-independent length check, so a too-large payload must not
        # load a GGUF only to 413 afterward (the decode itself stays post-switch to
        # avoid decoding a valid upload twice).
        if payload.audio_base64 and len(payload.audio_base64) > _MAX_AUDIO_B64_CHARS:
            raise HTTPException(status_code = 413, detail = "Audio file is too large (max ~25 MB).")
        # Reject streaming n>1 before the switch: only the non-streaming GGUF path
        # returns multiple choices, so stream=true + n>1 is invalid on every local
        # serving path (the external path already rejected it before its early
        # return). Both fields are known here, so a bad shape must not load model B
        # only to 400. The non-streaming n>1 cases stay post-switch, where the
        # serving path decides whether the shape is supported.
        if payload.stream and _wants_multiple_choices(payload):
            _raise_unsupported_n("streaming chat completions")
        # Audio input rides the same companion-mmproj projector as vision, so a
        # text-only target can't serve it either; guard both before the switch.
        _needs_vision = (
            bool(_pre_parsed[2]) or _request_has_image(payload) or bool(payload.audio_base64)
        )

    await _maybe_auto_switch_model(
        _switch_model_for_payload(payload),
        request,
        current_subject,
        require_vision = _needs_vision,
    )

    llama_backend = get_llama_cpp_backend()
    using_gguf = llama_backend.is_loaded

    # OpenAI-SDK clients send ``chat_template_kwargs`` via ``extra_body``, which
    # the SDK spreads into the request body at the top level. Unsloth's
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
    monitor_id = None

    async def _monitored_generate_audio(model_label: str, context_length: Optional[int] = None):
        tts_monitor_id = None
        if not getattr(request.state, "skip_api_monitor", False):
            tts_monitor_id = api_monitor.start(
                endpoint = request.url.path,
                via_api_key = _request_used_api_key(request),
                method = request.method,
                model = model_label,
                prompt = _monitor_prompt_from_messages(payload.messages),
                context_length = context_length,
                subject = current_subject,
            )
        try:
            response = await generate_audio(payload, request)
        except asyncio.CancelledError:
            api_monitor.finish(tts_monitor_id, "cancelled")
            raise
        except Exception as e:
            api_monitor.fail(tts_monitor_id, _friendly_error(e))
            raise
        if isinstance(response, JSONResponse):
            try:
                body = json.loads(response.body.decode())
                choices = body.get("choices") or []
                message = (choices[0].get("message") or {}) if choices else {}
                content = message.get("content")
                if isinstance(content, str):
                    api_monitor.set_reply(tts_monitor_id, content)
            except Exception:
                pass
        api_monitor.finish(tts_monitor_id)
        return response

    if using_gguf:
        # Advertised repo id after an auto-switch load, else a clean public id,
        # never the absolute .gguf path.
        model_name = _llama_public_model_id(llama_backend, payload.model)
        if getattr(llama_backend, "_is_audio", False):
            if _wants_multiple_choices(payload):
                _raise_unsupported_n("GGUF audio chat completions")
            return await _monitored_generate_audio(
                model_name,
                context_length = llama_backend.context_length,
            )
    else:
        backend = await asyncio.to_thread(get_inference_backend)
        if not backend.active_model_name:
            _status, _detail = await _no_model_loaded_error(
                "No model loaded. Call POST /inference/load first.",
                _switch_model_for_payload(payload),
                request,
                status = 400,
            )
            raise HTTPException(status_code = _status, detail = _detail)
        # Clean public id so the response never echoes a local path; the audio
        # branch below receives this sanitized label too.
        model_name = public_model_id(backend.active_model_name) or payload.model
        if _wants_multiple_choices(payload):
            _raise_unsupported_n("non-GGUF chat completions")

        # ── Audio TTS path: auto-route to audio generation ────
        # (Whisper is ASR not TTS -- handled below in audio input path)
        model_info = backend.models.get(backend.active_model_name, {})
        if model_info.get("is_audio") and model_info.get("audio_type") != "whisper":
            return await _monitored_generate_audio(model_name)

        # ── Whisper without audio: return clear error ──
        if model_info.get("audio_type") == "whisper" and not payload.audio_base64:
            raise HTTPException(
                status_code = 400,
                detail = "Whisper models require audio input. Please upload an audio file.",
            )

        if not getattr(request.state, "skip_api_monitor", False):
            monitor_id = api_monitor.start(
                endpoint = request.url.path,
                via_api_key = _request_used_api_key(request),
                method = request.method,
                model = model_name,
                prompt = _monitor_prompt_from_messages(payload.messages),
                context_length = _monitor_context_length(),
                subject = current_subject,
            )

        # ── Audio INPUT path: decode WAV and route to audio input generation ──
        if payload.audio_base64 and model_info.get("has_audio_input"):
            try:
                audio_array = _decode_audio_base64(payload.audio_base64)
                system_prompt, chat_messages, _ = _extract_content_parts(payload.messages)
            except Exception as e:
                api_monitor.fail(monitor_id, _friendly_error(e))
                raise
            cancel_event = threading.Event()
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())

            # Apply recommended sampling + operator pins to the omitted fields before generating,
            # so audio-input (non-whisper) generation honors `unsloth run --temperature` and
            # per-model recommendations like chat does. Whisper (ASR) ignores these fields.
            _fill_recommended_sampling_openai(
                payload, getattr(backend, "active_model_name", None) or model_name
            )

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
                    # Compare sends audio_base64 and use_adapter in one body.
                    use_adapter = payload.use_adapter,
                    cancel_event = cancel_event,
                )

            if payload.stream:
                _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
                _tracker = _TrackedCancel.for_payload(cancel_event, payload, *_cancel_keys)
                _tracker.__enter__()

                async def audio_input_stream():
                    disconnect_watcher = asyncio.create_task(
                        _await_disconnect_then_cancel(request, cancel_event)
                    )
                    try:
                        yield _chat_role_chunk(completion_id, created, model_name)

                        gen = audio_input_generate()
                        _DONE = object()
                        cancelled = False
                        while True:
                            if cancel_event.is_set():
                                cancelled = True
                                break
                            if await request.is_disconnected():
                                cancel_event.set()
                                api_monitor.finish(monitor_id, "cancelled")
                                return
                            chunk_text = await asyncio.to_thread(next, gen, _DONE)
                            if chunk_text is _DONE:
                                break
                            if isinstance(chunk_text, GenStreamError):
                                _msg = _friendly_gen_stream_error(chunk_text)
                                api_monitor.fail(monitor_id, _msg)
                                yield _openai_stream_error_sse(
                                    {"error": {"message": _msg, "type": "server_error"}}
                                )
                                return
                            if chunk_text:
                                api_monitor.append_reply(monitor_id, chunk_text)
                                yield _chat_content_chunk(
                                    completion_id, created, model_name, chunk_text
                                )

                        api_monitor.finish(monitor_id, "cancelled" if cancelled else "completed")
                        yield _chat_final_chunk(completion_id, created, model_name, "stop")
                        yield "data: [DONE]\n\n"
                    except asyncio.CancelledError:
                        cancel_event.set()
                        api_monitor.finish(monitor_id, "cancelled")
                        raise
                    except Exception as e:
                        logger.error(f"Error during audio input streaming: {e}", exc_info = True)
                        _msg = _friendly_error(e)
                        api_monitor.fail(monitor_id, _msg)
                        yield _openai_stream_error_sse(
                            {"error": {"message": _msg, "type": "server_error"}}
                        )
                    finally:
                        await _stop_local_disconnect_cancel_watcher(disconnect_watcher)
                        _tracker.__exit__(None, None, None)

                return _SameTaskStreamingResponse(
                    audio_input_stream(),
                    unstarted_cleanup = _tracked_cancel_unstarted_cleanup(_tracker),
                    media_type = "text/event-stream",
                    headers = {
                        "Cache-Control": "no-cache",
                        "Connection": "close",
                        "X-Accel-Buffering": "no",
                    },
                )
            else:
                # `stream` defaults to False, so this is the ordinary shape of an audio-input chat and it
                # holds the worker for the whole request. Unregistered, a swap counted zero generations
                # and cancelled it instead of 409ing (/unload runs no idle drain).
                _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
                _tracker = _TrackedCancel.for_payload(cancel_event, payload, *_cancel_keys)
                _tracker.__enter__()
                try:
                    full_text = ""
                    for chunk_text in audio_input_generate():
                        if isinstance(chunk_text, GenStreamError):
                            _msg = _friendly_gen_stream_error(chunk_text)
                            api_monitor.fail(monitor_id, _msg)
                            raise HTTPException(status_code = 500, detail = _msg)
                        full_text += chunk_text
                except HTTPException:
                    raise
                except Exception as e:
                    api_monitor.fail(monitor_id, _friendly_error(e))
                    raise
                finally:
                    # Nested under the except arms too: api_monitor.fail() can throw, and a leaked entry 409s swaps.
                    _tracker.__exit__(None, None, None)
                api_monitor.set_reply(monitor_id, full_text)
                api_monitor.finish(monitor_id)
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
                return _model_json_response(response)

    if monitor_id is None and not getattr(request.state, "skip_api_monitor", False):
        monitor_id = api_monitor.start(
            endpoint = request.url.path,
            via_api_key = _request_used_api_key(request),
            method = request.method,
            model = model_name,
            prompt = _monitor_prompt_from_messages(payload.messages),
            context_length = _monitor_context_length(),
            subject = current_subject,
        )

    # Finalize the monitor entry on validation rejection before raising.
    def _reject(status_code: int, detail: Any) -> "HTTPException":
        if monitor_id is not None:
            fail_detail = detail if isinstance(detail, str) else json.dumps(detail, default = str)
            api_monitor.fail(monitor_id, fail_detail)
        return HTTPException(status_code = status_code, detail = detail)

    def _reject_unsupported_n(path_label: str) -> "HTTPException":
        return _reject(
            400,
            openai_error_body(
                f"n > 1 is not supported for {path_label}.",
                status = 400,
                code = "unsupported_parameter",
                param = "n",
            ),
        )

    # Apply per-model recommended sampling (and any operator UNSLOTH_SAMPLING_* pin) to the
    # fields the client omitted, so agents and API clients get the model's tuned defaults
    # unless they set the field explicitly. Placed after external-provider routing (which
    # returned above) so only local llama-server / transformers requests are touched, and it
    # covers both the passthrough and non-passthrough branches below since both read payload.*.
    _reco_model_id = (
        getattr(llama_backend, "model_identifier", None)
        if using_gguf
        else getattr(backend, "active_model_name", None)
    ) or model_name
    _fill_recommended_sampling_openai(payload, _reco_model_id)

    # ── Standard OpenAI function-calling pass-through (GGUF only) ────
    # When a client (opencode / Claude Code via OpenAI compat / Cursor /
    # Continue / ...) sends standard OpenAI `tools` without Unsloth's
    # `enable_tools` shorthand, forward the request to llama-server
    # verbatim so structured `tool_calls` flow back to the client. This
    # branch runs BEFORE `_extract_content_parts` because that helper is
    # unaware of `role="tool"` messages and assistant messages that only
    # carry `tool_calls` (content=None) — both of which are valid in
    # multi-turn client-side tool loops.
    effective_max_tokens = _effective_openai_max_tokens(payload)

    normalized_stop = _normalize_stop_sequences(payload.stop)

    _has_tool_messages = _has_openai_tool_history(payload.messages)
    _has_tool_catalog = bool(payload.tools and len(payload.tools) > 0)
    _has_active_tool_catalog = _has_tool_catalog and payload.tool_choice != "none"
    _has_client_tool_contract = _has_active_tool_catalog or _has_tool_messages
    # The Unsloth tool loop needs a tool-capable backend, so a request that asks
    # for it on a backend that can't run it (DiffusionGemma forces supports_tools
    # off) must not steal client tools from the passthrough (#6851).
    _studio_tool_loop_requested = (
        _explicit_studio_tool_loop_requested(payload) and llama_backend.supports_tools
    )
    _client_disabled_tool_calls = payload.tool_choice == "none" and not _studio_tool_loop_requested
    _supports_tool_passthrough = getattr(
        llama_backend, "supports_tool_passthrough", llama_backend.supports_tools
    )
    if (
        using_gguf
        and not _studio_tool_loop_requested
        and _has_client_tool_contract
        and not _supports_tool_passthrough
    ):
        raise _reject(
            400,
            openai_error_body(
                (
                    "Client-supplied tools or tool-call history require a GGUF chat template "
                    "with tool-call support; the current model/template does not advertise tools."
                ),
                status = 400,
                code = "unsupported_parameter",
                param = "tools" if payload.tools else "messages",
            ),
        )
    # Shared with the token counter, so a count can never describe a route the completion does
    # not take. Guided decoding routes here too: the non-passthrough path calls
    # generate_chat_completion, which has no response_format kwarg and would silently drop the
    # schema. No ``supports_tools`` needed -- grammars are independent of it.
    if using_gguf and _takes_tool_passthrough(payload, llama_backend):
        if _wants_multiple_choices(payload):
            raise _reject_unsupported_n("GGUF tool or response_format passthrough")
        if payload.audio_base64:
            # This path forwards the request verbatim, so the transcoded audio
            # never gets injected. (The agentic tool loop below does support
            # audio.)
            raise _reject(
                400,
                "Audio input is not supported together with guided decoding or client-supplied tools yet.",
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
            raise _reject(
                400,
                "Image provided but current GGUF model does not support vision.",
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
                monitor_id = monitor_id,
            )
        _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
        _tracker = _TrackedCancel.for_payload(cancel_event, payload, *_cancel_keys)
        _tracker.__enter__()
        try:
            return await _openai_passthrough_non_streaming(
                llama_backend,
                payload,
                model_name,
                monitor_id = monitor_id,
                request = request,
                cancel_event = cancel_event,
            )
        finally:
            _tracker.__exit__(None, None, None)

    # ── Parse messages (handles multimodal content parts) ─────
    # Reuse the pre-hook parse when auto-switch did it, else parse now.
    if _pre_parsed is not None:
        system_prompt, chat_messages, extracted_image_b64 = _pre_parsed
    else:
        system_prompt, chat_messages, extracted_image_b64 = _extract_content_parts(payload.messages)

    if not chat_messages:
        raise _reject(400, "At least one non-system message is required.")

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
                raise _reject(
                    400,
                    "Audio provided but current GGUF model does not support audio input.",
                )
            if len(payload.audio_base64) > _MAX_AUDIO_B64_CHARS:
                raise _reject(413, "Audio file is too large (max ~25 MB).")
            try:
                audio_b64, audio_format = await asyncio.to_thread(
                    _prepare_audio_for_llama, payload.audio_base64
                )
            except Exception as e:
                logger.warning("Audio decode failed: %s", e, exc_info = True)
                raise _reject(400, "Could not decode the provided audio file.")

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

        def _new_chat_reasoning_extractor():
            return _ResponsesReasoningExtractor(
                parse_think_markers = _responses_should_parse_think_markers(
                    payload,
                    llama_backend,
                )
            )

        def _gguf_chat_delta_line(delta: ChoiceDelta, finish_reason = None) -> str:
            if delta.reasoning_content is not None and delta.content is None:
                delta = delta.model_copy(update = {"content": ""})
            chunk = ChatCompletionChunk(
                id = completion_id,
                created = created,
                model = model_name,
                choices = [
                    ChunkChoice(
                        delta = delta,
                        finish_reason = finish_reason,
                    )
                ],
            )
            return f"data: {chunk.model_dump_json(exclude_none = True)}\n\n"

        # ── Tool-calling path (agentic loop) ──────────────────
        # `_effective_enable_tools` lets `unsloth run --enable-tools/--disable-tools`
        # hard-override the per-request value, else falls back to
        # `payload.enable_tools`. `mcp_enabled=true` also opens the tool loop so
        # MCP-only callers needn't flip a second flag, BUT must still honor a
        # CLI `--disable-tools` policy -- checking the raw policy here keeps
        # `mcp_enabled` from re-enabling tools the operator explicitly forbade.
        from state.tool_policy import get_tool_policy as _get_tool_policy_g

        _cli_policy = _get_tool_policy_g()
        _tools_on = False if _client_disabled_tool_calls else _effective_enable_tools(payload)
        _mcp_allowed = (
            not _client_disabled_tool_calls
            and bool(payload.mcp_enabled)
            and _cli_policy is not False
        )
        use_tools = (_tools_on or _mcp_allowed) and llama_backend.supports_tools

        if use_tools:
            tools_to_use = await _select_request_tools(
                payload, tools_on = _tools_on, mcp_allowed = _mcp_allowed
            )
            # Skip the tool loop when no tool survived, so the safetensors
            # loop's "empty = allow all" semantic can't reach built-in tools
            # the caller didn't opt into. Callers who omit enabled_tools still
            # get ALL_TOOLS here, so this only suppresses the loop when
            # discovery + opt-in left it genuinely empty.
            if not tools_to_use:
                use_tools = False

        if use_tools:
            # permission_mode ask/auto require the confirm gate for Unsloth's own
            # tool loop. The request validator self-enables confirm only for
            # request-level tool signals (enable_tools/enabled_tools/mcp_enabled);
            # when a CLI policy (--enable-tools) forces the loop on without those,
            # derive confirm here so the mode still gates the call (and a
            # non-stream ask/auto request is rejected below rather than running
            # unprompted). off/full never prompt, so they are excluded.
            _effective_confirm = _permission_mode_confirm(payload)
            # Bypass Permissions suppresses confirm, so the stream requirement
            # (the gate needs streaming to prompt) no longer applies. auto with an
            # always-safe-only selection never prompts, so it needs no stream even
            # though _effective_confirm stays true for the loop's per-call gate.
            if (
                _confirm_gate_needs_stream(payload)
                and not payload.bypass_permissions
                and not payload.stream
            ):
                raise _reject(
                    400,
                    openai_error_body(
                        "confirm_tool_calls requires stream=true for local tool execution.",
                        status = 400,
                        code = "invalid_request_error",
                        param = "confirm_tool_calls",
                    ),
                )
            if _wants_multiple_choices(payload):
                raise _reject_unsupported_n("GGUF tool chat completions")
            # ── Tool-use system prompt nudge ──────────────────────
            _nudge = _build_tool_action_nudge(
                tools = tools_to_use,
                model_name = model_name,
            )

            # Nudge the model to ground in attached documents instead of memory.
            _nudge = _apply_rag_nudge(_nudge, tools_to_use, rag_scope = payload.rag_scope)

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
            # Active tool names gating the bare-rehearsal strip, matching the loop gate.
            _gguf_display_tool_names = _display_tool_name_gate(tools_to_use)

            # ── Strip stale tool-call XML from conversation history ─
            for _msg in gguf_messages:
                if _msg.get("role") == "assistant" and isinstance(_msg.get("content"), str):
                    # Gate on enabled tool names, like the live strip, so a documented inactive
                    # ``foo[ARGS]{...}`` survives in the replayed prompt context.
                    _msg["content"] = _strip_tool_xml_for_display(
                        _msg["content"],
                        auto_heal_tool_calls = _gguf_auto_heal_tool_calls,
                        enabled_tool_names = _gguf_display_tool_names,
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
                    nudge_tool_calls = payload.nudge_tool_calls,
                    max_tool_iterations = payload.max_tool_calls_per_message
                    if payload.max_tool_calls_per_message is not None
                    else 25,
                    tool_call_timeout = payload.tool_call_timeout
                    if payload.tool_call_timeout is not None
                    else 300,
                    session_id = payload.session_id,
                    thread_id = payload.thread_id,
                    rag_scope = payload.rag_scope,
                    disable_parallel_tool_use = payload.parallel_tool_calls is False,
                    # Bypass Permissions takes precedence over the confirm gate:
                    # never prompt while bypassing.
                    confirm_tool_calls = _effective_confirm and not bool(payload.bypass_permissions),
                    bypass_permissions = bool(payload.bypass_permissions),
                    permission_mode = payload.permission_mode,
                )

            _tool_admission_mode = "chat_tool_stream" if payload.stream else "chat_tool_nonstream"
            try:
                reservation, admission_config = _openai_llama_admission_reserve(
                    request = request,
                    llama_backend = llama_backend,
                )
            except LlamaAdmissionQueueFull as exc:
                _llama_admission_log(
                    "queue-full",
                    snapshot = exc.snapshot,
                    request = request,
                    mode = _tool_admission_mode,
                    completion_id = completion_id,
                    level = "warning",
                )
                api_monitor.fail(monitor_id, str(exc))
                raise _openai_admission_http_exception(exc, status_code = 429)

            _tool_sentinel = object()
            # True only once the sync generator returned on its own; see _gguf_decode_finished.
            _tool_decode_finished = False

            _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
            _tracker = _TrackedCancel.for_payload(cancel_event, payload, *_cancel_keys)
            _tracker.__enter__()

            async def gguf_tool_stream():
                nonlocal _tool_decode_finished
                gen = None
                next_task = None
                stream_completed = False
                # A call parked on the approval prompt is not decoding, so it gives its slot back;
                # otherwise unanswered prompts hold every slot.
                _parked = False

                async def _park_admission(on: bool, *, wait: bool = True):
                    nonlocal _parked
                    if on == _parked:
                        return
                    # This run's own lease, not a fresh lookup: queues are keyed by base_url and a
                    # reload mints a new port, so re-resolving could release someone else's slot.
                    lease = reservation.lease_nowait()
                    if lease is None:
                        return
                    if on:
                        # Refused when the budget is spent: the slot stays here,
                        # so there is nothing to take back afterwards.
                        if not lease.park():
                            return
                    elif wait:
                        # Resuming: park() may have handed our slot to a waiter, so wait for room instead
                        # of putting two holders on one slot.
                        await lease.unpark_async(cancel_event = cancel_event)
                    else:
                        # Tearing down; the lease is released separately.
                        lease.unpark()
                    _parked = on

                disconnect_watcher = asyncio.create_task(
                    _await_disconnect_then_cancel(request, cancel_event)
                )
                try:
                    yield _chat_role_chunk(completion_id, created, model_name)

                    # Iterate the sync generator in a thread so the event loop
                    # stays free for disconnect detection.
                    gen = gguf_generate_with_tools()
                    prev_text = ""
                    reasoning_extractor = _new_chat_reasoning_extractor()
                    _stream_usage = None
                    _stream_timings = None
                    _stream_finish = None

                    def _flush_reasoning_extractor():
                        final_reasoning, final_visible = reasoning_extractor.finish()
                        chunks = []
                        if final_reasoning:
                            chunks.append(
                                _gguf_chat_delta_line(
                                    ChoiceDelta(reasoning_content = final_reasoning)
                                )
                            )
                        if final_visible:
                            api_monitor.append_reply(monitor_id, final_visible)
                            chunks.append(_gguf_chat_delta_line(ChoiceDelta(content = final_visible)))
                        return chunks

                    while True:
                        if cancel_event.is_set():
                            break
                        if await request.is_disconnected():
                            cancel_event.set()
                            api_monitor.finish(monitor_id, "cancelled")
                            return

                        next_task = asyncio.create_task(
                            asyncio.to_thread(next, gen, _tool_sentinel)
                        )
                        try:
                            # Stall-timeout wait: keepalive while the generator stays
                            # silent (e.g. prefill between tool iterations). asyncio.wait
                            # never cancels next_task, matching the finally-drain shield.
                            while True:
                                done_tasks, _ = await asyncio.wait(
                                    {next_task},
                                    timeout = _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S,
                                )
                                if done_tasks:
                                    break
                                yield _OPENAI_PASSTHROUGH_SSE_KEEPALIVE
                            event = next_task.result()
                        finally:
                            if next_task.done():
                                next_task = None
                        if event is _tool_sentinel:
                            _tool_decode_finished = True
                            break

                        # Anything after the gated tool_start means the user answered.
                        if not (
                            event["type"] == "tool_start" and event.get("awaiting_confirmation")
                        ):
                            await _park_admission(False)

                        if event["type"] == "heartbeat":
                            # Tool-wrapper heartbeat while a server-side tool blocks; keeps SSE alive.
                            yield _OPENAI_PASSTHROUGH_SSE_KEEPALIVE
                            continue

                        if event["type"] in ("tool_output", "tool_args"):
                            # Live stdout/stderr or tool-call arguments, forwarded
                            # verbatim for the UI. Final result still arrives in tool_end.
                            yield f"data: {json.dumps(event)}\n\n"
                            continue

                        if event["type"] == "status":
                            # Empty status marks an iteration boundary in the
                            # GGUF tool loop (e.g. after a re-prompt). Reset the
                            # cumulative cursor so the next assistant turn
                            # streams cleanly.
                            if not event["text"]:
                                for chunk in _flush_reasoning_extractor():
                                    yield chunk
                                prev_text = ""
                                reasoning_extractor = _new_chat_reasoning_extractor()
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
                                for chunk in _flush_reasoning_extractor():
                                    yield chunk
                                prev_text = ""
                                reasoning_extractor = _new_chat_reasoning_extractor()
                                # Yielded just before the loop blocks on the user.
                                await _park_admission(bool(event.get("awaiting_confirmation")))
                            yield f"data: {json.dumps(event)}\n\n"
                            continue

                        if event["type"] == "metadata":
                            _stream_usage = event.get("usage")
                            _stream_timings = event.get("timings")
                            _stream_finish = event.get("finish_reason")
                            continue

                        if event["type"] == "reasoning_summary":
                            # Forward server-side reasoning timing to the UI.
                            yield f"data: {json.dumps(event)}\n\n"
                            continue

                        # "content" type -- cumulative text. Sanitize the full
                        # cumulative then diff against the last sanitized
                        # snapshot so cross-chunk XML tags are handled correctly.
                        raw_cumulative = event.get("text", "")
                        clean_cumulative = _strip_tool_xml_for_display(
                            raw_cumulative,
                            auto_heal_tool_calls = _gguf_auto_heal_tool_calls,
                            enabled_tool_names = _gguf_display_tool_names,
                        )
                        new_text = clean_cumulative[len(prev_text) :]
                        prev_text = clean_cumulative
                        if not new_text:
                            continue
                        reasoning_delta, visible_delta = reasoning_extractor.feed(new_text)
                        if reasoning_delta:
                            api_monitor.mark_first_token(monitor_id)
                            yield _gguf_chat_delta_line(
                                ChoiceDelta(reasoning_content = reasoning_delta)
                            )
                        if visible_delta:
                            api_monitor.append_reply(monitor_id, visible_delta)
                            yield _gguf_chat_delta_line(ChoiceDelta(content = visible_delta))

                    for chunk in _flush_reasoning_extractor():
                        yield chunk

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
                    # Emit the terminal chunk carrying finish_reason before the
                    # optional usage chunk and [DONE], so OpenAI-compatible
                    # clients can detect stop/length/tool_calls.
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
                    _monitor_usage(
                        monitor_id,
                        _stream_usage,
                        _monitor_context_length(),
                        timings = _stream_timings,
                        stop_reason = _clamp_finish_reason(_stream_finish)
                        if _stream_finish
                        else None,
                    )
                    api_monitor.finish(
                        monitor_id, "cancelled" if cancel_event.is_set() else "completed"
                    )
                    stream_completed = True
                    yield "data: [DONE]\n\n"

                except asyncio.CancelledError:
                    cancel_event.set()
                    api_monitor.finish(monitor_id, "cancelled")
                    raise
                except Exception as e:
                    logger.error(f"Error during GGUF tool streaming: {e}", exc_info = True)
                    api_monitor.fail(monitor_id, _friendly_error(e))
                    # Recover if an MTP+tensor crash killed the server mid-stream.
                    get_llama_cpp_backend()._maybe_recover_from_mtp_crash(e)
                    error_chunk = _openai_stream_error_chunk(e)
                    yield _openai_stream_error_sse(error_chunk)
                finally:
                    # A disconnect mid-approval must not leave a slot parked.
                    await _park_admission(False, wait = False)
                    try:
                        if not stream_completed:
                            cancel_event.set()
                        task_to_drain = next_task
                        next_task = None
                        while task_to_drain is not None and not task_to_drain.done():
                            try:
                                await asyncio.shield(task_to_drain)
                            except asyncio.CancelledError:
                                cancel_event.set()
                                continue
                            except Exception:
                                break
                        if task_to_drain is not None and task_to_drain.done():
                            try:
                                task_to_drain.exception()
                            except (asyncio.CancelledError, Exception):
                                pass
                        if gen is not None and not stream_completed:
                            try:
                                await asyncio.to_thread(gen.close)
                            except (RuntimeError, ValueError):
                                pass
                            except Exception:
                                logger.debug(
                                    "Error closing GGUF tool stream generator during cleanup",
                                    exc_info = True,
                                )
                        await _stop_local_disconnect_cancel_watcher(disconnect_watcher)
                    finally:
                        _tracker.__exit__(None, None, None)

            if payload.stream:
                stream_lease = reservation.lease_nowait()
                admission_wait_started_at = None
                if stream_lease is None:
                    admission_wait_started_at = time.monotonic()
                    _llama_admission_log(
                        "queued",
                        reservation,
                        request = request,
                        mode = _tool_admission_mode,
                        completion_id = completion_id,
                        level = "debug",
                    )

                async def admitted_gguf_tool_stream():
                    lease = stream_lease
                    stream_started = False
                    stream_cancelled = False
                    try:
                        if lease is None:
                            async for wait_item in _openai_admission_wait_stream_chunks(
                                reservation,
                                admission_config,
                                request = request,
                                cancel_event = cancel_event,
                            ):
                                if isinstance(wait_item, str):
                                    yield wait_item
                                    continue
                                lease = wait_item
                                _llama_admission_log(
                                    "granted-after-wait",
                                    reservation,
                                    request = request,
                                    mode = _tool_admission_mode,
                                    wait_started_at = admission_wait_started_at,
                                    completion_id = completion_id,
                                    level = "debug",
                                )
                                break
                        if lease is None:
                            return
                        await _raise_if_openai_admission_cancelled(
                            reservation,
                            request = request,
                            cancel_event = cancel_event,
                        )
                        iterator = gguf_tool_stream()
                        stream_started = True
                        try:
                            async for chunk in iterator:
                                # Release before the yield; see gguf_stream_chunks.
                                if (
                                    lease is not None
                                    and _tool_decode_finished
                                    and chunk == _SSE_DONE_CHUNK
                                ):
                                    lease.release()
                                yield chunk
                        except asyncio.CancelledError:
                            stream_cancelled = True
                            raise
                        finally:
                            await _close_openai_admitted_stream_iterator(
                                iterator,
                                cancelled = stream_cancelled,
                            )
                    except LlamaAdmissionTimeout as exc:
                        _llama_admission_log(
                            "timeout",
                            reservation,
                            request = request,
                            mode = _tool_admission_mode,
                            wait_started_at = admission_wait_started_at,
                            completion_id = completion_id,
                            level = "warning",
                        )
                        api_monitor.fail(monitor_id, str(exc))
                        yield _openai_stream_error_sse(
                            _openai_admission_error_body(exc, status_code = 503)
                        )
                    except LlamaAdmissionCancelled:
                        _llama_admission_log(
                            "cancelled-before-upstream",
                            reservation,
                            request = request,
                            mode = _tool_admission_mode,
                            wait_started_at = admission_wait_started_at,
                            completion_id = completion_id,
                            level = "debug",
                        )
                        api_monitor.finish(monitor_id, "cancelled")
                        return
                    except asyncio.CancelledError:
                        api_monitor.finish(monitor_id, "cancelled")
                        raise
                    except HTTPException as exc:
                        status_code = getattr(exc, "status_code", 500) or 500
                        detail = exc.detail
                        error = (
                            detail
                            if isinstance(detail, dict) and "error" in detail
                            else openai_error_body(str(detail), status = status_code)
                        )
                        api_monitor.fail(monitor_id, str(detail))
                        yield _openai_stream_error_sse(error)
                    finally:
                        if lease is not None:
                            lease.release()
                        if not stream_started:
                            api_monitor.finish(monitor_id, "cancelled")
                            reservation.cancel()
                            _tracker.__exit__(None, None, None)

                async def _gguf_tool_admission_unstarted_cleanup() -> None:
                    api_monitor.finish(monitor_id, "cancelled")
                    if stream_lease is not None:
                        stream_lease.release()
                    reservation.cancel()
                    _tracker.__exit__(None, None, None)

                return _SameTaskStreamingResponse(
                    admitted_gguf_tool_stream(),
                    unstarted_cleanup = _gguf_tool_admission_unstarted_cleanup,
                    media_type = "text/event-stream",
                    headers = {
                        "Cache-Control": "no-cache",
                        "Connection": "close",
                        "X-Accel-Buffering": "no",
                    },
                )

            # Non-streaming JSON: drain the agentic generator into one
            # ChatCompletion, like the standard GGUF `else` branch. stream:false
            # with tools enabled used to return an SSE body, breaking
            # non-streaming clients; `unsloth studio run --model` forces tools on
            # process-wide, so plain requests reach this path (#6570).
            def _drain_gguf_tool_loop():
                full_text = ""
                usage = None
                finish = None
                timings = None
                gen = gguf_generate_with_tools()
                try:
                    for event in gen:
                        if cancel_event.is_set():
                            break
                        if event.get("type") == "metadata":
                            usage = event.get("usage")
                            finish = event.get("finish_reason")
                            timings = event.get("timings")
                        elif event.get("type") == "content":
                            # Content is cumulative within a turn and resets
                            # between turns, so the last event holds the final
                            # turn's text. As in the safetensors drain, a visible
                            # preamble emitted before a tool call (its own earlier
                            # turn) isn't carried -- only the final turn is.
                            full_text = _strip_tool_xml_for_display(
                                event.get("text", ""),
                                auto_heal_tool_calls = _gguf_auto_heal_tool_calls,
                                enabled_tool_names = _gguf_display_tool_names,
                            )
                    return full_text, usage, finish, timings
                finally:
                    # Close the generator on early break/cancel so the underlying
                    # llama-server stream socket is released, like the SSE path.
                    try:
                        gen.close()
                    except (RuntimeError, ValueError):
                        pass

            drain_task = None

            async def _drain_cancelled_gguf_tool_task():
                if drain_task is None:
                    return
                while not drain_task.done():
                    try:
                        await asyncio.shield(drain_task)
                    except asyncio.CancelledError:
                        cancel_event.set()
                        continue
                    except Exception:
                        break
                if drain_task.done():
                    try:
                        drain_task.exception()
                    except (asyncio.CancelledError, Exception):
                        pass

            admission_lease = None
            admission_wait_started_at = None
            try:
                if reservation.lease_nowait() is None:
                    admission_wait_started_at = time.monotonic()
                    _llama_admission_log(
                        "queued",
                        reservation,
                        request = request,
                        mode = _tool_admission_mode,
                        completion_id = completion_id,
                        level = "debug",
                    )
                admission_lease = await _wait_for_openai_admission_non_streaming(
                    reservation,
                    admission_config,
                    request = request,
                    cancel_event = cancel_event,
                )
                if admission_wait_started_at is not None:
                    _llama_admission_log(
                        "granted-after-wait",
                        reservation,
                        request = request,
                        mode = _tool_admission_mode,
                        wait_started_at = admission_wait_started_at,
                        completion_id = completion_id,
                        level = "debug",
                    )
                await _raise_if_openai_admission_cancelled(
                    reservation,
                    request = request,
                    cancel_event = cancel_event,
                )
                drain_task = asyncio.create_task(asyncio.to_thread(_drain_gguf_tool_loop))
                (
                    full_text,
                    completion_usage,
                    completion_finish,
                    completion_timings,
                ) = await asyncio.shield(drain_task)
                reasoning_text, visible_text = _extract_responses_reasoning(
                    full_text,
                    parse_think_markers = _responses_should_parse_think_markers(
                        payload, llama_backend
                    ),
                )
                message_kwargs = {"content": visible_text}
                if reasoning_text:
                    message_kwargs["reasoning_content"] = reasoning_text
                _usage = completion_usage or {}
                _prompt_tokens = _usage.get("prompt_tokens") or 0
                _completion_tokens = _usage.get("completion_tokens") or 0
                response = ChatCompletion(
                    id = completion_id,
                    created = created,
                    model = model_name,
                    choices = [
                        CompletionChoice(
                            message = CompletionMessage(**message_kwargs),
                            finish_reason = _clamp_finish_reason(completion_finish),
                        )
                    ],
                    usage = CompletionUsage(
                        prompt_tokens = _prompt_tokens,
                        completion_tokens = _completion_tokens,
                        total_tokens = _prompt_tokens + _completion_tokens,
                        prompt_tokens_details = _prompt_tokens_details(
                            _usage.get("prompt_tokens_details")
                        ),
                    ),
                )
                api_monitor.set_reply(monitor_id, visible_text)
                _monitor_usage(
                    monitor_id,
                    {
                        "prompt_tokens": _prompt_tokens,
                        "completion_tokens": _completion_tokens,
                        "total_tokens": _prompt_tokens + _completion_tokens,
                    },
                    _monitor_context_length(),
                    timings = completion_timings,
                    stop_reason = _clamp_finish_reason(completion_finish)
                    if completion_finish
                    else None,
                )
                api_monitor.finish(
                    monitor_id, "cancelled" if cancel_event.is_set() else "completed"
                )
                return _model_json_response(response)
            except asyncio.CancelledError:
                cancel_event.set()
                await _drain_cancelled_gguf_tool_task()
                api_monitor.finish(monitor_id, "cancelled")
                reservation.cancel()
                if admission_lease is not None:
                    admission_lease.release()
                _tracker.__exit__(None, None, None)
                raise
            except LlamaAdmissionTimeout as exc:
                _llama_admission_log(
                    "timeout",
                    reservation,
                    request = request,
                    mode = _tool_admission_mode,
                    wait_started_at = admission_wait_started_at,
                    completion_id = completion_id,
                    level = "warning",
                )
                api_monitor.fail(monitor_id, str(exc))
                if admission_lease is not None:
                    admission_lease.release()
                _tracker.__exit__(None, None, None)
                raise _openai_admission_http_exception(exc, status_code = 503)
            except LlamaAdmissionCancelled as exc:
                _llama_admission_log(
                    "cancelled-before-upstream",
                    reservation,
                    request = request,
                    mode = _tool_admission_mode,
                    wait_started_at = admission_wait_started_at,
                    completion_id = completion_id,
                    level = "debug",
                )
                api_monitor.finish(monitor_id, "cancelled")
                if admission_lease is not None:
                    admission_lease.release()
                _tracker.__exit__(None, None, None)
                raise HTTPException(
                    status_code = 499,
                    detail = _openai_admission_error_body(exc, status_code = 499),
                )
            except Exception as e:
                logger.error(f"Error during GGUF tool completion: {e}", exc_info = True)
                api_monitor.fail(monitor_id, _friendly_error(e))
                # Recover if an MTP+tensor crash killed the server.
                get_llama_cpp_backend()._maybe_recover_from_mtp_crash(e)
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
            finally:
                if admission_lease is not None:
                    admission_lease.release()
                _tracker.__exit__(None, None, None)

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
        # True only once the sync generator returned on its own: only then has _open_stream's
        # client exited. A cancel still emits [DONE] without it.
        _gguf_decode_finished = False

        if payload.stream:
            if _wants_multiple_choices(payload):
                raise _reject_unsupported_n("streaming GGUF chat completions")
            _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
            _tracker = _TrackedCancel.for_payload(cancel_event, payload, *_cancel_keys)
            _tracker.__enter__()
            try:
                reservation, admission_config = _openai_llama_admission_reserve(
                    request = request,
                    llama_backend = llama_backend,
                )
            except LlamaAdmissionQueueFull as exc:
                _tracker.__exit__(None, None, None)
                _llama_admission_log(
                    "queue-full",
                    snapshot = exc.snapshot,
                    request = request,
                    mode = "chat_standard_stream",
                    completion_id = completion_id,
                    level = "warning",
                )
                api_monitor.fail(monitor_id, str(exc))
                raise _openai_admission_http_exception(exc, status_code = 429)

            async def gguf_stream_chunks():
                nonlocal _gguf_decode_finished
                disconnect_watcher = asyncio.create_task(
                    _await_disconnect_then_cancel(request, cancel_event)
                )
                gen = None
                next_task = None
                stream_completed = False
                try:
                    yield _chat_role_chunk(completion_id, created, model_name)

                    # Iterate the sync generator in a thread so the event loop
                    # stays free for disconnect detection.
                    gen = gguf_generate()
                    prev_text = ""
                    reasoning_extractor = _new_chat_reasoning_extractor()
                    _stream_usage = None
                    _stream_timings = None
                    _stream_finish = None
                    while True:
                        if cancel_event.is_set():
                            break
                        if await request.is_disconnected():
                            cancel_event.set()
                            api_monitor.finish(monitor_id, "cancelled")
                            return
                        next_task = asyncio.create_task(
                            asyncio.to_thread(next, gen, _gguf_sentinel)
                        )
                        try:
                            # Stall-timeout wait: keepalive while the generator stays
                            # silent (e.g. no-tool prefill). asyncio.wait never cancels
                            # next_task, matching the finally-drain shield (see GGUF stream).
                            while True:
                                done_tasks, _ = await asyncio.wait(
                                    {next_task},
                                    timeout = _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S,
                                )
                                if done_tasks:
                                    break
                                yield _OPENAI_PASSTHROUGH_SSE_KEEPALIVE
                            cumulative = next_task.result()
                        finally:
                            if next_task.done():
                                next_task = None
                        if cumulative is _gguf_sentinel:
                            _gguf_decode_finished = True
                            break
                        # Capture server metadata for the final usage chunk
                        if isinstance(cumulative, dict):
                            if cumulative.get("type") == "metadata":
                                _stream_usage = cumulative.get("usage")
                                _stream_timings = cumulative.get("timings")
                                _stream_finish = cumulative.get("finish_reason")
                            elif cumulative.get("type") == "diffusion_frame":
                                # Diffusion frame (per-step canvas): pass through as a raw SSE line on the
                                # tool_status channel. No assistant text, so it never enters the cumulative diff.
                                yield f"data: {json.dumps(cumulative)}\n\n"
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
                        reasoning_delta, visible_delta = reasoning_extractor.feed(new_text)
                        if reasoning_delta:
                            api_monitor.mark_first_token(monitor_id)
                            yield _gguf_chat_delta_line(
                                ChoiceDelta(reasoning_content = reasoning_delta)
                            )
                        if visible_delta:
                            api_monitor.append_reply(monitor_id, visible_delta)
                            yield _gguf_chat_delta_line(ChoiceDelta(content = visible_delta))

                    final_reasoning, final_visible = reasoning_extractor.finish()
                    if final_reasoning:
                        yield _gguf_chat_delta_line(ChoiceDelta(reasoning_content = final_reasoning))
                    if final_visible:
                        api_monitor.append_reply(monitor_id, final_visible)
                        yield _gguf_chat_delta_line(ChoiceDelta(content = final_visible))

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
                    # Emit the terminal chunk carrying finish_reason before the
                    # optional usage chunk and [DONE], so OpenAI-compatible
                    # clients can detect stop/length/tool_calls.
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
                    _monitor_usage(
                        monitor_id,
                        _stream_usage,
                        _monitor_context_length(),
                        timings = _stream_timings,
                        stop_reason = _clamp_finish_reason(_stream_finish)
                        if _stream_finish
                        else None,
                    )
                    api_monitor.finish(
                        monitor_id, "cancelled" if cancel_event.is_set() else "completed"
                    )
                    stream_completed = True
                    yield "data: [DONE]\n\n"

                except asyncio.CancelledError:
                    cancel_event.set()
                    api_monitor.finish(monitor_id, "cancelled")
                    raise
                except Exception as e:
                    logger.error(f"Error during GGUF streaming: {e}", exc_info = True)
                    api_monitor.fail(monitor_id, _friendly_error(e))
                    error_chunk = _openai_stream_error_chunk(e)
                    yield _openai_stream_error_sse(error_chunk)
                finally:
                    try:
                        if not stream_completed:
                            cancel_event.set()
                        task_to_drain = next_task
                        next_task = None
                        while task_to_drain is not None and not task_to_drain.done():
                            try:
                                await asyncio.shield(task_to_drain)
                            except asyncio.CancelledError:
                                cancel_event.set()
                                continue
                            except Exception:
                                break
                        if task_to_drain is not None and task_to_drain.done():
                            try:
                                task_to_drain.exception()
                            except (asyncio.CancelledError, Exception):
                                pass
                        if gen is not None and not stream_completed:
                            try:
                                await asyncio.to_thread(gen.close)
                            except (RuntimeError, ValueError):
                                pass
                            except Exception:
                                logger.debug(
                                    "Error closing GGUF stream generator during cleanup",
                                    exc_info = True,
                                )
                        await _stop_local_disconnect_cancel_watcher(disconnect_watcher)
                    finally:
                        _tracker.__exit__(None, None, None)

            stream_lease = reservation.lease_nowait()
            admission_wait_started_at = None
            if stream_lease is None:
                admission_wait_started_at = time.monotonic()
                _llama_admission_log(
                    "queued",
                    reservation,
                    request = request,
                    mode = "chat_standard_stream",
                    completion_id = completion_id,
                    level = "debug",
                )

            async def admitted_gguf_stream_chunks():
                lease = stream_lease
                stream_started = False
                stream_cancelled = False
                try:
                    if lease is None:
                        async for wait_item in _openai_admission_wait_stream_chunks(
                            reservation,
                            admission_config,
                            request = request,
                            cancel_event = cancel_event,
                        ):
                            if isinstance(wait_item, str):
                                yield wait_item
                                continue
                            lease = wait_item
                            _llama_admission_log(
                                "granted-after-wait",
                                reservation,
                                request = request,
                                mode = "chat_standard_stream",
                                wait_started_at = admission_wait_started_at,
                                completion_id = completion_id,
                                level = "debug",
                            )
                            break
                    if lease is None:
                        return
                    await _raise_if_openai_admission_cancelled(
                        reservation,
                        request = request,
                        cancel_event = cancel_event,
                    )
                    iterator = gguf_stream_chunks()
                    stream_started = True
                    try:
                        async for chunk in iterator:
                            # The slot is idle once the sync generator returned and the stream ends
                            # with the plain sentinel. The finally only runs at ASGI teardown, so
                            # waiting for it starves the next request. Release before the yield: a
                            # stalled send() or a consumer that stops pulling parks us there, and
                            # Starlette never aclose()s a body iterator. Release is idempotent, so
                            # the finally stays the backstop. Exact equality, not endswith:
                            # _openai_stream_error_sse ends in the same sentinel before its
                            # cleanup runs, and that stream still owns the slot.
                            if (
                                lease is not None
                                and _gguf_decode_finished
                                and chunk == _SSE_DONE_CHUNK
                            ):
                                lease.release()
                            yield chunk
                    except asyncio.CancelledError:
                        stream_cancelled = True
                        raise
                    finally:
                        await _close_openai_admitted_stream_iterator(
                            iterator,
                            cancelled = stream_cancelled,
                        )
                except LlamaAdmissionTimeout as exc:
                    _llama_admission_log(
                        "timeout",
                        reservation,
                        request = request,
                        mode = "chat_standard_stream",
                        wait_started_at = admission_wait_started_at,
                        completion_id = completion_id,
                        level = "warning",
                    )
                    api_monitor.fail(monitor_id, str(exc))
                    yield _openai_stream_error_sse(
                        _openai_admission_error_body(exc, status_code = 503)
                    )
                except LlamaAdmissionCancelled:
                    _llama_admission_log(
                        "cancelled-before-upstream",
                        reservation,
                        request = request,
                        mode = "chat_standard_stream",
                        wait_started_at = admission_wait_started_at,
                        completion_id = completion_id,
                        level = "debug",
                    )
                    api_monitor.finish(monitor_id, "cancelled")
                    return
                except asyncio.CancelledError:
                    api_monitor.finish(monitor_id, "cancelled")
                    raise
                except HTTPException as exc:
                    status_code = getattr(exc, "status_code", 500) or 500
                    detail = exc.detail
                    error = (
                        detail
                        if isinstance(detail, dict) and "error" in detail
                        else openai_error_body(str(detail), status = status_code)
                    )
                    api_monitor.fail(monitor_id, str(detail))
                    yield _openai_stream_error_sse(error)
                finally:
                    if lease is not None:
                        lease.release()
                    if not stream_started:
                        api_monitor.finish(monitor_id, "cancelled")
                        reservation.cancel()
                        _tracker.__exit__(None, None, None)

            async def _gguf_admission_unstarted_cleanup() -> None:
                api_monitor.finish(monitor_id, "cancelled")
                if stream_lease is not None:
                    stream_lease.release()
                reservation.cancel()
                _tracker.__exit__(None, None, None)

            return _SameTaskStreamingResponse(
                admitted_gguf_stream_chunks(),
                unstarted_cleanup = _gguf_admission_unstarted_cleanup,
                media_type = "text/event-stream",
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "close",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            try:
                reservation, admission_config = _openai_llama_admission_reserve(
                    request = request,
                    llama_backend = llama_backend,
                )
            except LlamaAdmissionQueueFull as exc:
                _llama_admission_log(
                    "queue-full",
                    snapshot = exc.snapshot,
                    request = request,
                    mode = "chat_standard_nonstream",
                    completion_id = completion_id,
                    level = "warning",
                )
                api_monitor.fail(monitor_id, str(exc))
                raise _openai_admission_http_exception(exc, status_code = 429)

            _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
            _tracker = _TrackedCancel.for_payload(cancel_event, payload, *_cancel_keys)
            _tracker.__enter__()
            admission_lease = None
            admission_wait_started_at = None
            try:
                if reservation.lease_nowait() is None:
                    admission_wait_started_at = time.monotonic()
                    _llama_admission_log(
                        "queued",
                        reservation,
                        request = request,
                        mode = "chat_standard_nonstream",
                        completion_id = completion_id,
                        level = "debug",
                    )
                admission_lease = await _wait_for_openai_admission_non_streaming(
                    reservation,
                    admission_config,
                    request = request,
                    cancel_event = cancel_event,
                )
                if admission_wait_started_at is not None:
                    _llama_admission_log(
                        "granted-after-wait",
                        reservation,
                        request = request,
                        mode = "chat_standard_nonstream",
                        wait_started_at = admission_wait_started_at,
                        completion_id = completion_id,
                        level = "debug",
                    )
                await _raise_if_openai_admission_cancelled(
                    reservation,
                    request = request,
                    cancel_event = cancel_event,
                )
            except asyncio.CancelledError:
                api_monitor.finish(monitor_id, "cancelled")
                reservation.cancel()
                if admission_lease is not None:
                    admission_lease.release()
                _tracker.__exit__(None, None, None)
                raise
            except LlamaAdmissionTimeout as exc:
                _llama_admission_log(
                    "timeout",
                    reservation,
                    request = request,
                    mode = "chat_standard_nonstream",
                    wait_started_at = admission_wait_started_at,
                    completion_id = completion_id,
                    level = "warning",
                )
                api_monitor.fail(monitor_id, str(exc))
                if admission_lease is not None:
                    admission_lease.release()
                _tracker.__exit__(None, None, None)
                raise _openai_admission_http_exception(exc, status_code = 503)
            except LlamaAdmissionCancelled as exc:
                _llama_admission_log(
                    "cancelled-before-upstream",
                    reservation,
                    request = request,
                    mode = "chat_standard_nonstream",
                    wait_started_at = admission_wait_started_at,
                    completion_id = completion_id,
                    level = "debug",
                )
                api_monitor.finish(monitor_id, "cancelled")
                if admission_lease is not None:
                    admission_lease.release()
                _tracker.__exit__(None, None, None)
                raise HTTPException(
                    status_code = 499,
                    detail = _openai_admission_error_body(exc, status_code = 499),
                )

            try:
                # ``n`` requests several independent completions; the single
                # decode slot yields one at a time, so loop sequentially.
                drain_task = None

                async def _drain_cancelled_gguf_task():
                    if drain_task is None:
                        return
                    while not drain_task.done():
                        try:
                            await asyncio.shield(drain_task)
                        except asyncio.CancelledError:
                            cancel_event.set()
                            continue
                        except Exception:
                            break
                    if drain_task.done():
                        try:
                            drain_task.exception()
                        except (asyncio.CancelledError, Exception):
                            pass

                def _drain_gguf_choices():
                    _n = payload.n or 1
                    _choices = []
                    _monitor_replies = []
                    _prompt_tokens = 0
                    _sum_completion = 0
                    _prompt_details = None
                    _last_timings = None
                    _last_finish = None
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
                                    _last_timings = token.get("timings")
                                    _last_finish = completion_finish
                                continue
                            full_text = token

                        reasoning_text, visible_text = _extract_responses_reasoning(
                            full_text,
                            parse_think_markers = _responses_should_parse_think_markers(
                                payload,
                                llama_backend,
                            ),
                        )
                        message_kwargs = {"content": visible_text}
                        if reasoning_text:
                            message_kwargs["reasoning_content"] = reasoning_text
                        _choices.append(
                            CompletionChoice(
                                index = _idx,
                                message = CompletionMessage(**message_kwargs),
                                finish_reason = _clamp_finish_reason(completion_finish),
                            )
                        )
                        _monitor_replies.append(visible_text)
                        if completion_usage:
                            # The prompt is shared across all n choices, so count its
                            # tokens ONCE (OpenAI bills only generated tokens for each
                            # extra choice). Only completion_tokens accumulates.
                            _prompt_tokens = completion_usage.get("prompt_tokens") or _prompt_tokens
                            _sum_completion += completion_usage.get("completion_tokens") or 0
                            if _prompt_details is None:
                                _prompt_details = completion_usage.get("prompt_tokens_details")
                    return (
                        _n,
                        _choices,
                        _monitor_replies,
                        _prompt_tokens,
                        _sum_completion,
                        _prompt_details,
                        _last_timings,
                        _last_finish,
                    )

                drain_task = asyncio.create_task(asyncio.to_thread(_drain_gguf_choices))
                (
                    _n,
                    _choices,
                    _monitor_replies,
                    _prompt_tokens,
                    _sum_completion,
                    _prompt_details,
                    _last_timings,
                    _last_finish,
                ) = await asyncio.shield(drain_task)

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
                monitor_reply = _monitor_replies[-1] if _monitor_replies else ""
                if _n > 1:
                    monitor_reply = "\n\n".join(
                        f"Choice {_idx + 1}:\n{text}" for _idx, text in enumerate(_monitor_replies)
                    )
                api_monitor.set_reply(monitor_id, monitor_reply)
                _monitor_usage(
                    monitor_id,
                    {
                        "prompt_tokens": _prompt_tokens,
                        "completion_tokens": _sum_completion,
                        "total_tokens": _prompt_tokens + _sum_completion,
                    },
                    _monitor_context_length(),
                    timings = _last_timings,
                    stop_reason = _clamp_finish_reason(_last_finish) if _last_finish else None,
                )
                api_monitor.finish(monitor_id)
                return _model_json_response(response)

            except asyncio.CancelledError:
                cancel_event.set()
                await _drain_cancelled_gguf_task()
                api_monitor.finish(monitor_id, "cancelled")
                raise
            except Exception as e:
                logger.error(f"Error during GGUF completion: {e}", exc_info = True)
                api_monitor.fail(monitor_id, _friendly_error(e))
                # Recover if an MTP+tensor crash killed the server.
                get_llama_cpp_backend()._maybe_recover_from_mtp_crash(e)
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
            finally:
                if admission_lease is not None:
                    admission_lease.release()
                _tracker.__exit__(None, None, None)
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
    # Named templates may expose native reasoning only in their ``tool_use``
    # branch. Use a truthy placeholder for Unsloth-managed tools, whose concrete
    # schemas are selected below, and the request schemas for client passthrough.
    _sf_server_tool_intent = bool(
        _effective_enable_tools(payload) or _explicit_studio_tool_loop_requested(payload)
    )
    _sf_template_tools = payload.tools if payload.tool_choice != "none" else None
    if not _sf_template_tools and _sf_server_tool_intent:
        _sf_template_tools = ({},)

    def _sf_response_protocol(tools = None):
        features = _detect_safetensors_features(backend, _sf_tpl, tools = tools)
        parse_think = bool(
            features.get("supports_reasoning") or features.get("reasoning_always_on")
        )
        reasoning_prefilled = _sf_reasoning_prefill_mode(
            features,
            payload.enable_thinking,
            _sf_tpl,
            reasoning_effort = payload.reasoning_effort,
        )
        return features, parse_think, reasoning_prefilled

    # GGUF parity: split canonical <think> output into reasoning_content. The
    # selected template branch must match whether this request renders tools.
    _sf_features, _sf_parse_think, _sf_reasoning_prefilled = _sf_response_protocol(
        _sf_template_tools
    )

    def _new_sf_reasoning_extractor():
        return _ResponsesReasoningExtractor(
            parse_think_markers = _sf_parse_think,
            reasoning_prefilled = _sf_reasoning_prefilled,
        )

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
        _sf_tools_to_use = await _select_request_tools(
            payload, tools_on = _sf_tools_on, mcp_allowed = _sf_mcp_allowed
        )
        # Mirror the GGUF path: refuse to enter the tool loop when nothing
        # survived, so a model-emitted built-in call can't piggy-back on the
        # empty allow-list.
        if not _sf_tools_to_use:
            _sf_use_tools = False

    if _sf_use_tools:
        # permission_mode ask/auto require the confirm gate for Unsloth's own tool
        # loop; when a CLI policy (--enable-tools) forces the loop on without a
        # request-level tool signal, derive confirm here so the mode still gates
        # the call (matching the GGUF path). off/full never prompt.
        _sf_effective_confirm = _permission_mode_confirm(payload)
        # Bypass Permissions suppresses confirm, so the stream requirement
        # (the gate needs streaming to prompt) no longer applies. auto with an
        # always-safe-only selection never prompts, so it needs no stream even
        # though _sf_effective_confirm stays true for the loop's per-call gate.
        if (
            _confirm_gate_needs_stream(payload)
            and not payload.bypass_permissions
            and not payload.stream
        ):
            raise _reject(
                400,
                openai_error_body(
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
        _sf_nudge = _apply_rag_nudge(_sf_nudge, _sf_tools_to_use, rag_scope = payload.rag_scope)

        _sf_system_prompt = system_prompt
        if _sf_nudge:
            if _sf_system_prompt:
                _sf_system_prompt = _sf_system_prompt.rstrip() + "\n\n" + _sf_nudge
            else:
                _sf_system_prompt = _sf_nudge

        _sf_auto_heal_tool_calls = (
            payload.auto_heal_tool_calls if payload.auto_heal_tool_calls is not None else True
        )
        # Active tool names gating the bare-rehearsal strip, matching the loop gate.
        _sf_display_tool_names = _display_tool_name_gate(_sf_tools_to_use)

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
                            enabled_tool_names = _sf_display_tool_names,
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
                presence_penalty = payload.presence_penalty,
                cancel_event = cancel_event,
                enable_thinking = payload.enable_thinking,
                reasoning_effort = payload.reasoning_effort,
                preserve_thinking = payload.preserve_thinking,
                auto_heal_tool_calls = _sf_auto_heal_tool_calls,
                nudge_tool_calls = payload.nudge_tool_calls,
                max_tool_iterations = _sf_tool_budget,
                tool_call_timeout = payload.tool_call_timeout
                if payload.tool_call_timeout is not None
                else 300,
                session_id = payload.session_id,
                thread_id = payload.thread_id,
                rag_scope = payload.rag_scope,
                # Bypass Permissions takes precedence over the confirm gate:
                # never prompt while bypassing.
                confirm_tool_calls = _sf_effective_confirm and not bool(payload.bypass_permissions),
                bypass_permissions = bool(payload.bypass_permissions),
                permission_mode = payload.permission_mode,
                use_adapter = payload.use_adapter,
                stats_holder = _sf_stats_holder,
                reasoning_prefilled = _sf_reasoning_prefilled,
            )

        _sf_tool_sentinel = object()
        _sf_cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
        _sf_tracker = _TrackedCancel.for_payload(cancel_event, payload, *_sf_cancel_keys)
        _sf_tracker.__enter__()

        async def sf_tool_stream():
            gen = None
            _sf_next_task = None
            disconnect_watcher = asyncio.create_task(
                _await_disconnect_then_cancel(request, cancel_event)
            )
            try:
                yield _chat_role_chunk(completion_id, created, model_name)

                gen = sf_generate_with_tools()
                prev_text = ""
                reasoning_extractor = _new_sf_reasoning_extractor()

                def _sf_flush_reasoning():
                    # Drain the extractor at turn/stream end (mirrors GGUF); only visible text hits the monitor.
                    fr, fv = reasoning_extractor.finish()
                    out = []
                    if fr:
                        out.append(_chat_reasoning_chunk(completion_id, created, model_name, fr))
                    if fv:
                        api_monitor.append_reply(monitor_id, fv)
                        out.append(_chat_content_chunk(completion_id, created, model_name, fv))
                    return out

                while True:
                    if cancel_event.is_set():
                        backend.reset_generation_state(cancel_event)
                        break
                    if await request.is_disconnected():
                        cancel_event.set()
                        backend.reset_generation_state(cancel_event)
                        api_monitor.finish(monitor_id, "cancelled")
                        return

                    # Stall keepalive (see GGUF tool stream): silent backend segments
                    # must not leave the SSE stream idle past proxy timeouts.
                    _sf_next_task = asyncio.create_task(
                        asyncio.to_thread(next, gen, _sf_tool_sentinel)
                    )
                    while True:
                        _sf_done, _ = await asyncio.wait(
                            {_sf_next_task},
                            timeout = _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S,
                        )
                        if _sf_done:
                            break
                        yield _OPENAI_PASSTHROUGH_SSE_KEEPALIVE
                    event = _sf_next_task.result()
                    # Done; drop the reference so the finally-block drain no-ops.
                    _sf_next_task = None
                    if event is _sf_tool_sentinel:
                        break
                    if isinstance(event, GenStreamError):
                        backend.reset_generation_state(cancel_event)
                        _msg = _friendly_gen_stream_error(event)
                        api_monitor.fail(monitor_id, _msg)
                        yield _openai_stream_error_sse(
                            {"error": {"message": _msg, "type": "server_error"}}
                        )
                        return
                    if not isinstance(event, dict):
                        raise RuntimeError(
                            f"Invalid safetensors tool event: {type(event).__name__}"
                        )

                    if event["type"] == "heartbeat":
                        # Tool-execution wrapper heartbeat -> SSE keepalive.
                        yield _OPENAI_PASSTHROUGH_SSE_KEEPALIVE
                        continue

                    if event["type"] in ("tool_output", "tool_args"):
                        # Live stdout/stderr, or tool-call arguments as the model writes them.
                        yield f"data: {json.dumps(event)}\n\n"
                        continue

                    if event["type"] == "status":
                        if not event["text"]:
                            # Iteration boundary: flush reasoning, then a fresh prefilled extractor for the next turn.
                            for _c in _sf_flush_reasoning():
                                yield _c
                            prev_text = ""
                            reasoning_extractor = _new_sf_reasoning_extractor()
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
                            # Flush reasoning before tool_start so the thinking block closes ahead of the card.
                            for _c in _sf_flush_reasoning():
                                yield _c
                            prev_text = ""
                            reasoning_extractor = _new_sf_reasoning_extractor()
                        yield f"data: {json.dumps(event)}\n\n"
                        continue

                    # Diff cumulative cleaned text against last snapshot.
                    raw_cumulative = event.get("text", "")
                    clean_cumulative = _strip_tool_xml_for_display(
                        raw_cumulative,
                        auto_heal_tool_calls = _sf_auto_heal_tool_calls,
                        enabled_tool_names = _sf_display_tool_names,
                    )
                    new_text = clean_cumulative[len(prev_text) :]
                    prev_text = clean_cumulative
                    if not new_text:
                        continue
                    # Split reasoning vs visible; only visible reaches the monitor.
                    reasoning_delta, visible_delta = reasoning_extractor.feed(new_text)
                    if reasoning_delta:
                        api_monitor.mark_first_token(monitor_id)
                        yield _chat_reasoning_chunk(
                            completion_id, created, model_name, reasoning_delta
                        )
                    if visible_delta:
                        api_monitor.append_reply(monitor_id, visible_delta)
                        yield _chat_content_chunk(completion_id, created, model_name, visible_delta)

                for _c in _sf_flush_reasoning():
                    yield _c
                yield _chat_final_chunk(completion_id, created, model_name, "stop")
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
                    _monitor_usage(monitor_id, _stats.get("usage"), timings = _stats.get("timings"))
                api_monitor.finish(
                    monitor_id, "cancelled" if cancel_event.is_set() else "completed"
                )
                yield "data: [DONE]\n\n"

            except asyncio.CancelledError:
                cancel_event.set()
                backend.reset_generation_state(cancel_event)
                api_monitor.finish(monitor_id, "cancelled")
                raise
            except GenStreamErrorRaised as exc:
                backend.reset_generation_state(cancel_event)
                _msg = _friendly_gen_stream_error(exc)
                api_monitor.fail(monitor_id, _msg)
                yield _openai_stream_error_sse({"error": {"message": _msg, "type": "server_error"}})
            except Exception:
                backend.reset_generation_state(cancel_event)
                # Generic wire message; full trace stays in the log (CWE-209:
                # transformers/torch errors may leak paths).
                logger.exception("safetensors tool stream error")
                api_monitor.fail(monitor_id, "An internal error occurred.")
                error_chunk = {
                    "error": {
                        "message": "An internal error occurred.",
                        "type": "server_error",
                    },
                }
                yield _openai_stream_error_sse(error_chunk)
            finally:
                await _stop_local_disconnect_cancel_watcher(disconnect_watcher)
                # Drain a still-running next(gen) worker before closing: closing
                # mid-next(gen) raises ValueError('generator already executing') and
                # skips the generator's cleanup finally. Matches the GGUF tool stream.
                await _drain_pending_next_task(_sf_next_task, cancel_event)
                if gen is not None:
                    try:
                        # Offload the close so the generator's cleanup runs off the event
                        # loop (matches the GGUF SSE path); a disconnect can't stall the loop.
                        await asyncio.to_thread(gen.close)
                    except (RuntimeError, ValueError):
                        pass
                _sf_tracker.__exit__(None, None, None)

        if payload.stream:
            return _SameTaskStreamingResponse(
                sf_tool_stream(),
                unstarted_cleanup = _tracked_cancel_unstarted_cleanup(_sf_tracker),
                media_type = "text/event-stream",
                headers = {
                    "Cache-Control": "no-cache",
                    "Connection": "close",
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
                    if isinstance(event, GenStreamError):
                        raise HTTPException(
                            status_code = 500,
                            detail = _friendly_gen_stream_error(event),
                        )
                    if not isinstance(event, dict):
                        raise RuntimeError(
                            f"Invalid safetensors tool event: {type(event).__name__}"
                        )
                    if event.get("type") == "content":
                        full_text = _strip_tool_xml_for_display(
                            event.get("text", ""),
                            auto_heal_tool_calls = _sf_auto_heal_tool_calls,
                            enabled_tool_names = _sf_display_tool_names,
                        )
                return full_text

            content_text = await asyncio.to_thread(_drain_to_text)
            # Split prefilled <think> out of the visible answer (GGUF parity); the monitor gets visible text only.
            _reasoning_text, _visible_text = _extract_responses_reasoning(
                content_text,
                parse_think_markers = _sf_parse_think,
                reasoning_prefilled = _sf_reasoning_prefilled,
            )
            api_monitor.set_reply(monitor_id, _visible_text)
            _stats = _sf_stats_holder.get("stats")
            if _stats:
                _monitor_usage(monitor_id, _stats.get("usage"), timings = _stats.get("timings"))
            api_monitor.finish(monitor_id, "cancelled" if cancel_event.is_set() else "completed")
            _sf_msg_kwargs = {"content": _visible_text}
            if _reasoning_text:
                _sf_msg_kwargs["reasoning_content"] = _reasoning_text
            response = ChatCompletion(
                id = completion_id,
                created = created,
                model = model_name,
                choices = [
                    CompletionChoice(
                        message = CompletionMessage(**_sf_msg_kwargs),
                        finish_reason = "stop",
                    )
                ],
            )
            return _model_json_response(response)
        except asyncio.CancelledError:
            cancel_event.set()
            backend.reset_generation_state(cancel_event)
            api_monitor.finish(monitor_id, "cancelled")
            raise
        except GenStreamErrorRaised as exc:
            backend.reset_generation_state(cancel_event)
            _msg = _friendly_gen_stream_error(exc)
            api_monitor.fail(monitor_id, _msg)
            raise HTTPException(status_code = 500, detail = _msg)
        except HTTPException as exc:
            backend.reset_generation_state(cancel_event)
            api_monitor.fail(monitor_id, str(exc.detail))
            raise
        except Exception:
            backend.reset_generation_state(cancel_event)
            # CWE-209: generic detail; full trace in log.
            logger.exception("safetensors tool completion error")
            api_monitor.fail(monitor_id, "An internal error occurred.")
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
        presence_penalty = payload.presence_penalty,
    )
    # Forward reasoning kwargs; the worker/template wrapper peels off any the
    # template doesn't accept.
    if payload.enable_thinking is not None:
        gen_kwargs["enable_thinking"] = payload.enable_thinking
    if payload.reasoning_effort is not None:
        gen_kwargs["reasoning_effort"] = payload.reasoning_effort
    if payload.preserve_thinking is not None:
        gen_kwargs["preserve_thinking"] = payload.preserve_thinking

    # ── Client-tool passthrough (safetensors + MLX) ──────────────
    # Client tools (or tool-result history) without server-side tools: render
    # tools into the template, generate one turn, heal text-form calls (#6801).
    # supports_tools=False falls through to plain relay (GGUF gate parity).
    _sf_has_tool_msgs = any(m.role == "tool" or m.tool_calls for m in payload.messages)
    # Gate on _sf_use_tools (did the server-side path claim the request?), not
    # raw mcp_enabled: an empty MCP registry must not silently drop client tools.
    _sf_client_tools = (
        not _effective_enable_tools(payload)
        and not _sf_use_tools
        and image is None
        and not _sf_is_gptoss
        and _sf_features.get("supports_tools", False)
        and ((payload.tools and len(payload.tools) > 0) or _sf_has_tool_msgs)
    )
    # apply_chat_template sanitizes the catalog it renders, so a tool dropped for unsafe
    # markup never reached the prompt. Gating the healer on the caller's list instead would
    # promote a dropped tool with a clean NAME out of text-form output, handing the client a
    # call for a tool the model was never shown (#7066).
    from core.inference.chat_template_helpers import (
        chat_render_target as _sf_chat_render_target,
        markup_for_tokenizer as _sf_markup_for,
        neutralize_tool_descriptions as _sf_neutralize_tools,
        renderable_tool_catalog_for_targets as _sf_renderable_tools,
    )

    _sf_markup = _sf_markup_for(_sf_model_info.get("tokenizer"))

    # A text-only tool request on a vision model renders through a different object on each
    # backend: MLX keeps the PROCESSOR when it has a usable template (_generate_vlm), the
    # transformers path unwraps to the nested tokenizer (_generate_chat_response_inner).
    # Authorizing against one lets the other's render drop a tool the healer still holds, so
    # both are profiled and the catalog is the intersection. The MLX rule is shared with
    # _generate_vlm rather than restated, so the two cannot drift (#7066).
    _sf_processor = _sf_model_info.get("processor")
    _sf_tokenizer = _sf_model_info.get("tokenizer")
    _sf_mlx_target = _sf_chat_render_target(_sf_processor, _sf_tokenizer)
    _sf_hf_target = getattr(_sf_mlx_target, "tokenizer", _sf_mlx_target)
    _sf_chat_targets = (
        (_sf_mlx_target,) if _sf_hf_target is _sf_mlx_target else (_sf_mlx_target, _sf_hf_target)
    )
    _sf_healing_tools = (
        # Safe under EVERY template this turn could select: when the active one drops the
        # schema the render falls back to the native template, whose profile can drop a tool
        # the active profile kept (#7066). In a thread because the first request resolves
        # that native template through AutoTokenizer.from_pretrained, which would otherwise
        # block the event loop for every concurrent request.
        await asyncio.to_thread(
            _sf_renderable_tools,
            payload.tools,
            _sf_chat_targets,
            _sf_model_info,
            active_model_name = backend.active_model_name,
        )
        if _sf_client_tools
        else None
    )
    _sf_heal = (
        heal_gate(payload.auto_heal_tool_calls, _sf_healing_tools, payload.tool_choice)
        if _sf_client_tools
        else None
    )
    if _sf_client_tools:
        # Re-derive from payload.messages so tool_calls / role="tool" history
        # survives templating; fold system/developer into one leading system
        # message (templates reject "developer") and clear prompt to avoid a dup.
        gen_kwargs["messages"] = _set_or_prepend_system_message(
            _structured_tool_history_for_local_template(
                _flatten_content_parts_for_local_template(_openai_messages_for_passthrough(payload))
            ),
            system_prompt,
        )
        gen_kwargs["system_prompt"] = ""
        # tool_choice="none": keep history templating but advertise no tools
        # (heal_gate is off, markup would relay as prose). A forced function
        # narrows templating to that one schema. Both mirror the GGUF path,
        # where llama-server honors tool_choice itself.
        _sf_tc = payload.tool_choice
        _sf_forced = None
        if isinstance(_sf_tc, dict) and isinstance(_sf_tc.get("function"), dict):
            _sf_forced = _sf_tc["function"].get("name")
        if _sf_tc == "none":
            gen_kwargs["tools"] = None
        elif isinstance(_sf_forced, str):
            gen_kwargs["tools"] = [
                t
                for t in payload.tools or []
                if isinstance(t, dict)
                and isinstance(t.get("function"), dict)
                and t["function"].get("name") == _sf_forced
            ] or None
        else:
            gen_kwargs["tools"] = payload.tools

    # The potential tool context above is needed before server/client routing is
    # known. This standard path now has the exact schemas that will be rendered,
    # so resolve reasoning parsing again to keep empty registries, forced-tool
    # misses, and tool_choice="none" on the marker-free template branch.
    _, _sf_parse_think, _sf_reasoning_prefilled = _sf_response_protocol(gen_kwargs.get("tools"))

    # Request-scoped usage/timings receptacle (filled at gen_done).
    stats_holder: dict = {}

    if payload.use_adapter is not None:

        def generate(messages_override = None):
            kw = (
                gen_kwargs
                if messages_override is None
                else {**gen_kwargs, "messages": messages_override}
            )
            return backend.generate_with_adapter_control(
                use_adapter = payload.use_adapter,
                cancel_event = cancel_event,
                stats_holder = stats_holder,
                **kw,
            )
    else:

        def generate(messages_override = None):
            kw = (
                gen_kwargs
                if messages_override is None
                else {**gen_kwargs, "messages": messages_override}
            )
            return backend.generate_chat_response(
                cancel_event = cancel_event,
                stats_holder = stats_holder,
                **kw,
            )

    # ── Streaming response ────────────────────────────────────────
    if payload.stream:
        _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
        _tracker = _TrackedCancel.for_payload(cancel_event, payload, *_cancel_keys)
        _tracker.__enter__()

        async def stream_chunks():
            gen = None
            _next_task = None
            disconnect_watcher = asyncio.create_task(
                _await_disconnect_then_cancel(request, cancel_event)
            )
            try:
                yield _chat_role_chunk(completion_id, created, model_name)

                # Client-tool passthrough: heal text-form calls on the fly
                # (None => relay verbatim).
                healer = StreamToolCallHealer(_sf_heal, _sf_healing_tools) if _sf_heal else None
                heal_state = {"idx": 0}

                prev_text = ""
                # Split prefilled <think> into reasoning_content deltas (GGUF parity); single turn, serves MLX.
                reasoning_extractor = _new_sf_reasoning_extractor()
                # Run the sync generator in a worker thread so it can't block the event
                # loop. Critical for compare mode: a second request's blocking _gen_lock
                # acquisition would otherwise freeze the loop and stall both streams.
                _DONE = object()  # sentinel for generator exhaustion
                gen = generate()
                while True:
                    if cancel_event.is_set():
                        backend.reset_generation_state(cancel_event)
                        break
                    # Stall keepalive (see safetensors tool stream) each window while
                    # next(gen) runs in a worker. next(gen, _DONE) returns _DONE rather
                    # than raising StopIteration (which can't cross asyncio futures).
                    _next_task = asyncio.create_task(asyncio.to_thread(next, gen, _DONE))
                    while True:
                        _done_tasks, _ = await asyncio.wait(
                            {_next_task},
                            timeout = _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S,
                        )
                        if _done_tasks:
                            break
                        yield _OPENAI_PASSTHROUGH_SSE_KEEPALIVE
                    cumulative = _next_task.result()
                    # Done; drop the reference so the finally-block drain no-ops.
                    _next_task = None
                    if cumulative is _DONE:
                        break
                    if isinstance(cumulative, GenStreamError):
                        backend.reset_generation_state(cancel_event)
                        _msg = _friendly_gen_stream_error(cumulative)
                        api_monitor.fail(monitor_id, _msg)
                        yield _openai_stream_error_sse(
                            {"error": {"message": _msg, "type": "server_error"}}
                        )
                        return
                    if await request.is_disconnected():
                        cancel_event.set()
                        backend.reset_generation_state(cancel_event)
                        api_monitor.finish(monitor_id, "cancelled")
                        return
                    new_text = cumulative[len(prev_text) :]
                    prev_text = cumulative
                    if not new_text:
                        continue
                    # Split prefilled <think> reasoning first (GGUF/MLX parity),
                    # then route only the visible text through the client-tool
                    # healer so tool markup inside a reasoning block is not promoted.
                    reasoning_delta, visible_delta = reasoning_extractor.feed(new_text)
                    if reasoning_delta:
                        api_monitor.mark_first_token(monitor_id)
                        yield _chat_reasoning_chunk(
                            completion_id, created, model_name, reasoning_delta
                        )
                    if visible_delta:
                        if healer is None:
                            # Monitor mirrors the verbatim relay; with healing on,
                            # _sf_heal_events_to_sse records the healed events instead.
                            api_monitor.append_reply(monitor_id, visible_delta)
                            yield _chat_content_chunk(
                                completion_id, created, model_name, visible_delta
                            )
                        else:
                            for line in _sf_heal_events_to_sse(
                                healer.feed(visible_delta),
                                completion_id,
                                created,
                                model_name,
                                heal_state,
                                payload.parallel_tool_calls,
                                monitor_id,
                            ):
                                yield line

                final_reasoning, final_visible = reasoning_extractor.finish()
                if final_reasoning:
                    yield _chat_reasoning_chunk(completion_id, created, model_name, final_reasoning)
                if final_visible:
                    if healer is None:
                        api_monitor.append_reply(monitor_id, final_visible)
                        yield _chat_content_chunk(completion_id, created, model_name, final_visible)
                    else:
                        for line in _sf_heal_events_to_sse(
                            healer.feed(final_visible),
                            completion_id,
                            created,
                            model_name,
                            heal_state,
                            payload.parallel_tool_calls,
                            monitor_id,
                        ):
                            yield line

                # A cancelled stream must not promote buffered-but-incomplete
                # markup: finalize()'s allow_incomplete heal would execute a tool
                # the user just cancelled. Disconnect returns earlier; "Stop" only
                # sets cancel_event, so guard on it here too.
                _cancelled = cancel_event.is_set()
                if healer is not None and not _cancelled:
                    for line in _sf_heal_events_to_sse(
                        healer.finalize(),
                        completion_id,
                        created,
                        model_name,
                        heal_state,
                        payload.parallel_tool_calls,
                        monitor_id,
                    ):
                        yield line

                _finish = (
                    "tool_calls"
                    if (healer is not None and not _cancelled and healer.healed)
                    else "stop"
                )
                yield _chat_final_chunk(completion_id, created, model_name, _finish)
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
                    _monitor_usage(monitor_id, _stats.get("usage"), timings = _stats.get("timings"))
                api_monitor.finish(
                    monitor_id, "cancelled" if cancel_event.is_set() else "completed"
                )
                yield "data: [DONE]\n\n"

            except asyncio.CancelledError:
                cancel_event.set()
                backend.reset_generation_state(cancel_event)
                api_monitor.finish(monitor_id, "cancelled")
                raise
            except GenStreamErrorRaised as exc:
                # Adapter-controlled (compare-mode) backend failure. Honor the
                # public flag so operational errors surface their real message.
                backend.reset_generation_state(cancel_event)
                _msg = _friendly_gen_stream_error(exc)
                api_monitor.fail(monitor_id, _msg)
                yield _openai_stream_error_sse({"error": {"message": _msg, "type": "server_error"}})
            except Exception as e:
                backend.reset_generation_state(cancel_event)
                logger.error(f"Error during OpenAI streaming: {e}", exc_info = True)
                _msg = _friendly_error(e)
                api_monitor.fail(monitor_id, _msg)
                error_chunk = {
                    "error": {
                        "message": _msg,
                        "type": "server_error",
                    },
                }
                yield _openai_stream_error_sse(error_chunk)
            finally:
                await _stop_local_disconnect_cancel_watcher(disconnect_watcher)
                # Drain a still-running next(gen) worker before closing: closing
                # mid-next(gen) raises ValueError('generator already executing') and
                # skips the generator's cleanup finally. Matches the safetensors stream.
                await _drain_pending_next_task(_next_task, cancel_event)
                if gen is not None:
                    try:
                        # Offload the close so the generator's cleanup runs off the event
                        # loop (matches the GGUF SSE path); a disconnect can't stall the loop.
                        await asyncio.to_thread(gen.close)
                    except (RuntimeError, ValueError):
                        pass
                _tracker.__exit__(None, None, None)

        return _SameTaskStreamingResponse(
            stream_chunks(),
            unstarted_cleanup = _tracked_cancel_unstarted_cleanup(_tracker),
            media_type = "text/event-stream",
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "close",
                "X-Accel-Buffering": "no",
            },
        )

    # ── Non-streaming response ────────────────────────────────────
    else:
        # `stream` defaults to False, so this is the default shape of a standard (non-GGUF) chat and
        # generate() holds the worker throughout. Unregistered, a swap cancelled this run rather
        # than returning 409 (/unload runs no idle drain).
        _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
        _tracker = _TrackedCancel.for_payload(cancel_event, payload, *_cancel_keys)
        _tracker.__enter__()
        try:
            full_text = ""
            for token in generate():
                if isinstance(token, GenStreamError):
                    backend.reset_generation_state(cancel_event)
                    _msg = _friendly_gen_stream_error(token)
                    api_monitor.fail(monitor_id, _msg)
                    raise HTTPException(status_code = 500, detail = _msg)
                full_text = token

            # Split prefilled <think> reasoning (GGUF parity); also covers MLX via
            # the shared generate(). Client-tool healing then runs on the visible
            # text so tool markup inside a reasoning block is never promoted.
            _reasoning_text, _visible_text = _extract_responses_reasoning(
                full_text,
                parse_think_markers = _sf_parse_think,
                reasoning_prefilled = _sf_reasoning_prefilled,
            )
            # Client-tool passthrough: promote text-form calls; opt-in single
            # nudge retry on unparseable tool markup.
            _msg = {"role": "assistant", "content": _visible_text}
            if _reasoning_text:
                _msg["reasoning_content"] = _reasoning_text
            _finish = "stop"
            if _sf_heal:
                if heal_openai_message(_msg, _sf_heal, _sf_healing_tools):
                    _finish = "tool_calls"
                elif nudge_enabled(payload.nudge_tool_calls):
                    _data = {
                        "choices": [{"message": {"role": "assistant", "content": _visible_text}}]
                    }
                    if nudge_should_retry(_data, _sf_heal, _sf_healing_tools):
                        # A failed retry must not 500 the request; keep the first
                        # response (GGUF nudge parity). The retry's generate()
                        # overwrites stats_holder, so save the first attempt's stats
                        # and restore them if the retry is discarded.
                        _first_stats = stats_holder.get("stats")
                        try:
                            retry_text = ""
                            for token in generate(
                                [*gen_kwargs["messages"], *nudge_messages(_data, _sf_heal)]
                            ):
                                retry_text = token
                            # Re-split reasoning on the retry so its visible text is
                            # what heals into a call (and reaches the monitor).
                            _retry_reasoning, _retry_visible = _extract_responses_reasoning(
                                retry_text,
                                parse_think_markers = _sf_parse_think,
                                reasoning_prefilled = _sf_reasoning_prefilled,
                            )
                            retry_msg = {"role": "assistant", "content": _retry_visible}
                            if _retry_reasoning:
                                retry_msg["reasoning_content"] = _retry_reasoning
                            if heal_openai_message(retry_msg, _sf_heal, _sf_healing_tools):
                                _visible_text, _msg, _finish = (
                                    _retry_visible,
                                    retry_msg,
                                    "tool_calls",
                                )
                            else:
                                # Retry produced no healable call -> first response wins.
                                stats_holder["stats"] = _first_stats
                        except Exception as retry_exc:
                            logger.debug(
                                "Nudge retry failed; keeping first response: %s", retry_exc
                            )
                            stats_holder["stats"] = _first_stats
                # parallel_tool_calls=false: cap to one call (GGUF parity).
                if payload.parallel_tool_calls is False:
                    _tcs = _msg.get("tool_calls")
                    if isinstance(_tcs, list) and len(_tcs) > 1:
                        _msg["tool_calls"] = _tcs[:1]

            response = ChatCompletion(
                id = completion_id,
                created = created,
                model = model_name,
                choices = [
                    CompletionChoice(
                        message = CompletionMessage(
                            content = _msg["content"],
                            reasoning_content = _msg.get("reasoning_content"),
                            tool_calls = _msg.get("tool_calls"),
                        ),
                        finish_reason = _finish,
                    )
                ],
            )
            _monitor_reply = _msg.get("content") or ""
            if _finish == "tool_calls":
                _tcs = _msg.get("tool_calls") or []
                _calls_text = "; ".join(
                    f"{(tc.get('function') or {}).get('name', '')}"
                    f"({(tc.get('function') or {}).get('arguments', '')})"
                    for tc in _tcs
                )
                _monitor_reply = (_msg.get("content") or "") + (
                    f"[tool_calls] {_calls_text}" if _calls_text else ""
                )
            api_monitor.set_reply(monitor_id, _monitor_reply)
            _stats = stats_holder.get("stats")
            if _stats:
                _monitor_usage(monitor_id, _stats.get("usage"), timings = _stats.get("timings"))
            api_monitor.finish(monitor_id)
            return _model_json_response(response)

        except HTTPException:
            raise
        except GenStreamErrorRaised as exc:
            # Adapter-controlled (compare-mode) backend failure. Honor the public
            # flag so operational errors surface their real message.
            backend.reset_generation_state(cancel_event)
            _msg = _friendly_gen_stream_error(exc)
            api_monitor.fail(monitor_id, _msg)
            raise HTTPException(status_code = 500, detail = _msg)
        except Exception as e:
            backend.reset_generation_state(cancel_event)
            logger.error(f"Error during OpenAI completion: {e}", exc_info = True)
            api_monitor.fail(monitor_id, _friendly_error(e))
            raise HTTPException(status_code = 500, detail = safe_error_detail(e))
        finally:
            # Nested under the except arms too: reset_generation_state() can throw, and a leaked entry 409s swaps.
            _tracker.__exit__(None, None, None)


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
    await _authenticate_header_or_query(request, token)

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

# `owned_by` marker on every /v1/models entry (loaded and available alike).
_OWNED_BY = "unsloth-studio"


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
        # Advertise the repo id an auto-switch load recorded, not the concrete
        # on-disk load path, so /v1/models never leaks a host path or lists a
        # model twice (path plus repo id).
        entry = {
            # Advertised repo id after an auto-switch load, else a clean public id,
            # never the absolute .gguf path (which leaks the host filesystem layout).
            "id": _llama_public_model_id(llama_backend),
            "object": "model",
            "created": _created,
            "owned_by": _OWNED_BY,
        }
        _quant = getattr(llama_backend, "hf_variant", None)
        if _quant and _quant_reference_resolves(entry["id"], _quant):
            entry["quant"] = _quant
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
            "id": public_model_id(backend.active_model_name),
            "object": "model",
            "created": _created,
            "owned_by": _OWNED_BY,
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


# Brief cache for the local-model filesystem scan so repeated /v1/models calls
# don't rescan the HF cache and models dirs on every request.
_CATALOG_CACHE: dict = {"at": 0.0, "models": []}
# Ids the last catalog scan listed, rebuilt only when that scan is replaced.
_ADVERTISED_CACHE: dict = {"at": None, "paths": {}}


def _quant_reference_resolves(model_id: Optional[str], quant: str) -> bool:
    """Whether ``<model_id>:<quant>`` still resolves once this model is not resident.

    A standalone .gguf takes its quant from the filename, but the resolver stores
    such files with no quants, so advertising one hands out a pin that dies the
    moment another model loads.
    """
    from core.inference.local_model_resolver import (
        index_is_built,
        recently_downloaded,
        resolve_local_gguf,
        warm_index_soon,
    )

    if not model_id:
        return False
    # A cold index proves nothing, and publishing on no proof is what hands out the
    # dead pin; warm so the next response carries the quant.
    warm_index_soon()
    return resolve_local_gguf(f"{model_id}:{quant}", allow_scan = False) is not None


def _advertised_local_path(model: str) -> Optional[str]:
    """On-disk path of *model* if the last /v1/models scan listed it, else None.

    Cache-only, never scans. The catalog scans on its own schedule, so it can have
    advertised a local model the resolver index has not picked up yet, which is
    evidence the name means something other than the resident one.
    """
    if _ADVERTISED_CACHE["at"] != _CATALOG_CACHE["at"]:
        paths = {}
        for info in _CATALOG_CACHE["models"] or ():
            cid = getattr(info, "model_id", None) or public_model_id(getattr(info, "id", None))
            path = getattr(info, "path", None)
            if cid and path:
                paths.setdefault(cid.strip().lower(), path)
        _ADVERTISED_CACHE.update(at = _CATALOG_CACHE["at"], paths = paths)
    return _ADVERTISED_CACHE["paths"].get(model.strip().lower())


_CATALOG_TTL_S = 30.0
# Per-loop lock (like _auto_switch_lock): a module-level asyncio.Lock ties its
# waiters to the loop that first awaited it, so a second event loop awaiting it
# in a multi-loop ASGI process can hang. The cache double-check keeps correctness
# even when two loops each scan once.
_catalog_locks: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_catalog_locks_guard = threading.Lock()


def _catalog_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    with _catalog_locks_guard:
        lock = _catalog_locks.get(loop)
        if lock is None:
            lock = _catalog_locks[loop] = asyncio.Lock()
        return lock


async def _cached_local_catalog() -> list:
    """Locally available models (models dir + HF caches + LM Studio + scan
    folders), cached for a few seconds. Returns a list of LocalModelInfo.

    The scan walks several directories and stats many files, so it runs in a
    worker thread (asyncio.to_thread) -- calling it inline would block the event
    loop and stall every concurrent request and in-flight inference stream. A
    lock with a double-check collapses a burst of simultaneous /v1/models calls
    into a single scan instead of one per request."""
    # Validity is keyed on "at" (set only after a scan), not on list contents, so
    # an empty/errored scan is still cached instead of rescanning on every poll.
    now = time.monotonic()
    if _CATALOG_CACHE["at"] and (now - _CATALOG_CACHE["at"]) <= _CATALOG_TTL_S:
        return _CATALOG_CACHE["models"]
    async with _catalog_lock():
        now = time.monotonic()
        if _CATALOG_CACHE["at"] and (now - _CATALOG_CACHE["at"]) <= _CATALOG_TTL_S:
            return _CATALOG_CACHE["models"]
        try:
            from routes.models import collect_local_models
            _CATALOG_CACHE["models"] = await asyncio.to_thread(
                collect_local_models, Path("./models").resolve()
            )
        except Exception as exc:
            logger.debug("model catalog scan failed: %s", exc)
            _CATALOG_CACHE["models"] = []
        # Stamp after the scan, not the pre-scan "now": a scan slower than the TTL
        # would otherwise leave the cache already expired, so every waiter rescans.
        _CATALOG_CACHE["at"] = time.monotonic()
    return _CATALOG_CACHE["models"]


async def _openai_catalog_objects() -> list[dict]:
    """Every model the server knows about for ``GET /v1/models``: the loaded
    model(s) plus locally available (downloaded/cached) models discovered by
    scanning. Loaded entries keep their context fields and are marked
    ``loaded: true``. All ids are clean public ids (never absolute paths)."""
    _created = int(time.time())
    # Loaded models first (clean ids + context fields), marked loaded.
    by_id: dict[str, dict] = {}
    # Off-loop: _openai_model_objects() is sync and calls get_inference_backend(), whose cold
    # build waits on detection. Inline, an early GET /v1/models held the loop for the import.
    for entry in await asyncio.to_thread(_openai_model_objects):
        by_id[entry["id"]] = {**entry, "loaded": True}

    # Locally available (downloaded/cached) models that are not already loaded.
    # Advertise only GGUF models /v1 can actually serve (llama.cpp). GGUF-ness is
    # read from the on-disk files, not model_format: the HF-cache scanner leaves
    # model_format unset for GGUF snapshots, so a model_format filter would drop
    # every cached GGUF. The file checks run off the loop.
    from core.inference.local_model_resolver import local_gguf_quants

    catalog = await _cached_local_catalog()
    # One scan yields both "is this servable" and its on-disk quants, so no second pass.
    servable = await asyncio.to_thread(
        lambda: [(i, q) for i in catalog if (q := local_gguf_quants(i)) is not None]
    )
    for info, quants in servable:
        cid = getattr(info, "model_id", None) or public_model_id(getattr(info, "id", None))
        if not cid or cid in by_id:
            continue
        obj = {
            "id": cid,
            "object": "model",
            "created": _created,
            "owned_by": _OWNED_BY,
            # A manual load keys the resident entry by path basename while the catalog
            # uses the alias, so match on the path or the alias reads as not loaded.
            # llama-only: a Transformers model live from a directory that also holds
            # GGUF exports must not mark one of these GGUF entries loaded, or the
            # examples pin a quant nothing can serve with switching off.
            "loaded": _resolves_to_resident(getattr(info, "path", None), llama_only = True),
        }
        # The id stays bare for OpenAI compat; a client appends ":<quant>" to pin one.
        # For the resident model that must be the quant actually loaded, not the
        # preferred one on disk, or the listing advertises alias:Q4 while Q8 serves.
        resident_quant = getattr(get_llama_cpp_backend(), "hf_variant", None)
        if obj["loaded"] and resident_quant:
            obj["quant"] = resident_quant
        elif quants:
            obj["quant"] = quants[0]
        display = getattr(info, "display_name", None)
        if display:
            obj["display_name"] = display
        by_id[cid] = obj

    return list(by_id.values())


@router.get("/models")
async def openai_list_models(current_subject: str = Depends(get_current_subject)):
    """
    OpenAI-compatible model listing endpoint (``GET /v1/models``).

    Lists every model available on this server -- the loaded model(s) plus
    locally available (downloaded/cached) models -- not only what is resident in
    memory. Each entry carries a clean public id and a ``loaded`` flag.
    """
    return {"object": "list", "data": await _openai_catalog_objects()}


@router.get("/models/{model_id:path}")
async def openai_retrieve_model(model_id: str, current_subject: str = Depends(get_current_subject)):
    """
    OpenAI-compatible single-model retrieval endpoint (``GET /v1/models/{id}``).

    Returns the bare model object when ``model_id`` matches a known model
    (loaded or locally available), or 404 model_not_found otherwise. Defined
    after the LIST route so it does not shadow it; ``{model_id:path}`` keeps ids
    with slashes intact.
    """
    from core.inference.model_ids import model_id_matches

    # Loaded models resolve without a catalog scan (the common case); only build
    # the full catalog -- which may hit the filesystem -- for unloaded ids. Match
    # case-insensitively, like the catalog loop below and the resolver's index.
    # Off-loop like the catalog helper: the singleton's cold build waits on detection.
    _loaded = await asyncio.to_thread(_openai_model_objects)
    for entry in _loaded:
        eid = entry["id"]
        if isinstance(eid, str) and eid.lower() == model_id.lower():
            return {**entry, "loaded": True}

    objects = await _openai_catalog_objects()
    for model in objects:
        # Case-insensitive to match the resolver, which lowercases its index.
        mid = model.get("id")
        if isinstance(mid, str) and mid.lower() == model_id.lower():
            return model
    # Backward compatibility: a client may still send the legacy raw identifier
    # (e.g. an absolute .gguf path cached from an older /v1/models). Map it to the
    # loaded model's object so it keeps working, without ever echoing the path back.
    # Key each raw id to the SAME public id its /v1/models entry uses: an
    # auto-switch load advertises a repo id while its identifier is the snapshot
    # path, so public_model_id(path) would miss the advertised entry and 404 a
    # model that is in fact loaded.
    llama_backend = get_llama_cpp_backend()
    backend = await asyncio.to_thread(get_inference_backend)
    raw_to_public: list[tuple[str, Optional[str]]] = []
    if llama_backend.is_loaded and llama_backend.model_identifier:
        raw_to_public.append(
            (llama_backend.model_identifier, _llama_public_model_id(llama_backend))
        )
    if backend.active_model_name:
        raw_to_public.append(
            (backend.active_model_name, public_model_id(backend.active_model_name))
        )
    for raw, clean in raw_to_public:
        if model_id_matches(model_id, raw):
            for entry in _loaded:
                if entry["id"] == clean:
                    return {**entry, "loaded": True}
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


def _flatten_monitor_prompt(value) -> str:
    """Flatten an OpenAI prompt/input field (str or list) into the single
    string the api_monitor prompt preview expects."""
    if isinstance(value, list):
        return "\n".join(str(part) for part in value)
    return str(value)


def _completions_prompt_present(body: dict) -> bool:
    """Whether a completions body carries a usable ``prompt`` (non-empty)."""
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        return prompt != ""
    if isinstance(prompt, (list, tuple)):
        return len(prompt) > 0
    return prompt is not None


@router.post("/completions")
async def openai_completions(request: Request, current_subject: str = Depends(get_current_subject)):
    """
    OpenAI-compatible text completions endpoint (non-chat).

    Proxies to the running llama-server's ``/v1/completions``. Only available
    when a GGUF model is loaded.
    """
    llama_backend = get_llama_cpp_backend()

    # Reject a request with no prompt before any automatic load so an invalid
    # request never swaps or reloads the resident model (as chat/embeddings already
    # validate before switching). Gate on every automatic-load trigger.
    if _automatic_model_load_may_run():
        try:
            _pre = await request.json()
        except (json.JSONDecodeError, ValueError):
            _pre = None
        if isinstance(_pre, dict):
            _pre_prompt = _pre.get("prompt")
            if _pre_prompt is not None and not isinstance(_pre_prompt, (str, list, tuple)):
                # An object/number prompt is a deterministic client error (only a
                # string or array is valid); reject it before the switch so a bad
                # shape can't load a GGUF only to be rejected by llama-server after.
                raise HTTPException(status_code = 400, detail = "'prompt' must be a string or array.")
            if not _completions_prompt_present(_pre):
                raise HTTPException(status_code = 400, detail = "'prompt' is required for completions.")

    # Opt-in: load the requested local GGUF before the loaded-state check.
    body = await _auto_switch_from_request_body(request, current_subject)
    if not llama_backend.is_loaded:
        _status, _detail = await _no_model_loaded_error(
            "No GGUF model loaded. Load a GGUF model first.",
            _raw_body_model(body),
            request,
            status = 503,
        )
        raise HTTPException(status_code = _status, detail = _detail)
    if not isinstance(body, dict):
        # Re-read to re-raise a malformed-body error (post-503, pre-feature behavior);
        # a valid non-dict body such as a list is a clean 400 rather than a 500.
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code = 400, detail = "Request body must be a JSON object")

    _resolved_max_tokens = _effective_openai_max_tokens_from_values(body.get("max_tokens"))
    body["max_tokens"] = (
        _resolved_max_tokens
        if _resolved_max_tokens is not None
        else (llama_backend.context_length or _DEFAULT_MAX_TOKENS_FLOOR)
    )
    # Apply per-model recommended sampling and any operator UNSLOTH_SAMPLING_* pin to the raw
    # body so /v1/completions honors the same pins as /v1/chat/completions; it is otherwise a
    # verbatim proxy that would keep llama-server's defaults for every omitted sampling field.
    _fill_recommended_sampling_completions(body, getattr(llama_backend, "model_identifier", None))
    target_url = f"{llama_backend.base_url}/v1/completions"
    is_stream = body.get("stream", False)
    prompt_text = _flatten_monitor_prompt(body.get("prompt", ""))
    monitor_model = str(body.get("model") or _llama_public_model_id(llama_backend) or "default")
    monitor_id = api_monitor.start(
        endpoint = request.url.path,
        via_api_key = _request_used_api_key(request),
        method = request.method,
        model = monitor_model,
        prompt = prompt_text,
        context_length = llama_backend.context_length,
        subject = current_subject,
    )

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
            _direct_llama_request_started()
            client = httpx.AsyncClient(
                timeout = _llama_streaming_generation_timeout(),
                trust_env = False,
            )
            resp = None
            bytes_iter = None
            disconnect_event = threading.Event()
            disconnect_watcher = None
            # This proxy relays straight from llama-server, so the swap gate has to see it: without an
            # entry a non-forced /unload counts zero generations and tears the server down mid-response.
            # Sharing disconnect_event lets a forced swap stop the relay through the check it already
            # polls. Entered inside the body generator, so a response whose body never starts leaves
            # nothing behind (see _responses_stream). No thread_id: public API surface, not a chat.
            _tracker = _TrackedCancel(disconnect_event, model = monitor_model, kind = "completions")
            _tracker.__enter__()
            try:
                req = client.build_request(
                    "POST", target_url, json = body, headers = {"Connection": "close"}
                )
                first_token_deadline = time.monotonic() + _DEFAULT_FIRST_TOKEN_TIMEOUT_S
                # Same event the relay loop polls, so a forced swap ends the request during prefill
                # instead of only once headers arrive.
                resp = await _send_stream_with_preheader_cancel(
                    client, req, disconnect_event, request = request
                )
                if resp is None:
                    api_monitor.finish(monitor_id, "cancelled")
                    return
                if resp.status_code != 200:
                    err_bytes = await resp.aread()
                    err_text = err_bytes.decode("utf-8", errors = "replace")
                    api_monitor.fail(monitor_id, err_text[:500])
                    raise RuntimeError(f"llama-server returned {resp.status_code}: {err_text}")
                disconnect_watcher = asyncio.create_task(
                    _await_disconnect_then_close(request, resp, disconnect_event)
                )
                bytes_iter = resp.aiter_bytes()
                buffer = b""
                async for chunk in _aiter_llama_stream_items(
                    bytes_iter,
                    cancel_event = disconnect_event,
                    request = request,
                    first_token_deadline = first_token_deadline,
                    response = resp,
                ):
                    buffer += chunk
                    while b"\n\n" in buffer:
                        event, buffer = buffer.split(b"\n\n", 1)
                        _monitor_openai_sse_event(
                            monitor_id,
                            event,
                            llama_backend.context_length,
                        )
                        out = _cmpl_stream_event_out(event, _include_usage)
                        if out is not None:
                            yield out + b"\n\n"
                if not disconnect_event.is_set() and buffer:
                    _monitor_openai_sse_event(
                        monitor_id,
                        buffer,
                        llama_backend.context_length,
                    )
                    out = _cmpl_stream_event_out(buffer, _include_usage)
                    if out is not None:
                        # Re-add the SSE separator the split consumed, so a final
                        # event arriving without a trailing blank line is still
                        # terminated for the client's parser.
                        yield out + b"\n\n"
                if disconnect_event.is_set():
                    api_monitor.finish(monitor_id, "cancelled")
                    return
                api_monitor.finish(monitor_id)
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.CloseError) as e:
                if not disconnect_event.is_set():
                    logger.error("openai_completions stream error: %s", e)
                    api_monitor.fail(monitor_id, _friendly_error(e))
                    error_chunk = _openai_stream_error_chunk(e)
                    yield _openai_stream_error_sse_bytes(error_chunk)
                    return
                api_monitor.finish(monitor_id, "cancelled")
                return
            except asyncio.CancelledError:
                disconnect_event.set()
                api_monitor.finish(monitor_id, "cancelled")
                raise
            except Exception as e:
                if disconnect_event.is_set():
                    api_monitor.finish(monitor_id, "cancelled")
                    return
                logger.error("openai_completions stream error: %s", e)
                api_monitor.fail(monitor_id, _friendly_error(e))
                error_chunk = _openai_stream_error_chunk(e)
                yield _openai_stream_error_sse_bytes(error_chunk)
                return
            finally:
                # Nested so a close-time failure still unregisters; a phantom entry 409s swaps.
                try:
                    await _aclose_stream_resources(
                        watchers = (disconnect_watcher,),
                        iterator = bytes_iter,
                        resp = resp,
                        client = client,
                    )
                finally:
                    _direct_llama_request_finished()
                    _tracker.__exit__(None, None, None)

        return _sse_streaming_response(_stream())
    else:
        # ``stream`` defaults to false, so this common shape registers with the swap gate like the
        # streaming branch: unregistered, a non-forced /unload counts zero generations and kills
        # llama-server mid-request, and force_cancel_active has no event. Unpooled client so a
        # cancel-close hits this call only.
        _cancel_event = threading.Event()
        _client = _cancelable_nonstreaming_client()
        _tracker = _TrackedCancel(_cancel_event, model = monitor_model, kind = "completions")
        _tracker.__enter__()
        _direct_llama_request_started()
        _cancel_watcher = asyncio.create_task(
            _await_cancel_or_disconnect_then_close_client(
                cancel_event = _cancel_event,
                request = request,
                client = _client,
            )
        )
        try:
            try:
                resp = await _client.post(
                    target_url,
                    json = body,
                    timeout = _llama_non_streaming_generation_timeout(),
                )
            except httpx.RequestError:
                # The watcher closed the client out from under the request: report the cancel, not a transport failure.
                if _cancel_event.is_set():
                    raise asyncio.CancelledError()
                raise
            if _cancel_event.is_set():
                raise asyncio.CancelledError()
        except asyncio.CancelledError:
            api_monitor.finish(monitor_id, "cancelled")
            raise
        except Exception as e:
            api_monitor.fail(monitor_id, _friendly_error(e))
            raise
        finally:
            # Nested so a close-time failure still unregisters; a phantom entry 409s swaps.
            try:
                await _stop_local_disconnect_cancel_watcher(_cancel_watcher)
                try:
                    await _client.aclose()
                except Exception:
                    pass
            finally:
                _direct_llama_request_finished()
                _tracker.__exit__(None, None, None)

        if resp.status_code != 200:
            api_monitor.fail(monitor_id, resp.text[:500])
            raise _openai_passthrough_error(resp.status_code, resp.text)
        try:
            _monitor_openai_chunk(monitor_id, resp.json(), llama_backend.context_length)
        except Exception:
            pass
        api_monitor.finish(monitor_id)

        return Response(
            content = _rewrite_cmpl_id(resp.content),
            status_code = resp.status_code,
            media_type = "application/json",
        )


# =====================================================================
# OpenAI-Compatible Embeddings Proxy  (/embeddings → /v1/embeddings)
# =====================================================================


def _embeddings_input_present(body: dict) -> bool:
    """Whether an embeddings body carries a usable ``input`` (non-empty)."""
    inp = body.get("input")
    if isinstance(inp, str):
        return inp != ""
    if isinstance(inp, (list, tuple)):
        return len(inp) > 0
    return inp is not None


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
    # Reject a request with no input before any automatic load so an invalid
    # request never swaps or reloads the resident model (as chat/responses/messages
    # already validate before switching). Gate on every automatic-load trigger,
    # not just auto-switch, since a standalone idle TTL can also reload here.
    if _automatic_model_load_may_run():
        try:
            _pre = await request.json()
        except (json.JSONDecodeError, ValueError):
            _pre = None
        if isinstance(_pre, dict):
            _pre_input = _pre.get("input")
            if _pre_input is not None and not isinstance(_pre_input, (str, list, tuple)):
                # An object/number input is a deterministic client error (only a
                # string or array is valid); reject it before the switch so a bad
                # shape can't load a GGUF only to be rejected by llama-server after.
                raise HTTPException(status_code = 400, detail = "'input' must be a string or array.")
            if not _embeddings_input_present(_pre):
                raise HTTPException(status_code = 400, detail = "'input' is required for embeddings.")
    # Embeddings is a model-bearing inference path too, so honor auto-switch. Unlike
    # vision (cheaply pre-checked via a companion mmproj), GGUF pooling capability has
    # no reliable pre-load probe -- is_embedding_model keys on a sentence-transformers
    # modules.json a bare .gguf never has -- so embeddings auto-switch is best-effort:
    # a non-embedding target switches, then llama-server returns a no-pooling error.
    body = await _auto_switch_from_request_body(request, current_subject)
    if not llama_backend.is_loaded:
        _status, _detail = await _no_model_loaded_error(
            "No GGUF model loaded. Load a GGUF model first.",
            _raw_body_model(body),
            request,
            status = 503,
        )
        raise HTTPException(status_code = _status, detail = _detail)
    if not isinstance(body, dict):
        # Re-read to re-raise a malformed-body error (post-503, pre-feature behavior);
        # a valid non-dict body such as a list is a clean 400 rather than a 500.
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code = 400, detail = "Request body must be a JSON object")

    target_url = f"{llama_backend.base_url}/v1/embeddings"
    prompt_text = _flatten_monitor_prompt(body.get("input", ""))
    monitor_id = None
    if not getattr(request.state, "skip_api_monitor", False):
        monitor_id = api_monitor.start(
            endpoint = request.url.path,
            via_api_key = _request_used_api_key(request),
            method = request.method,
            model = str(body.get("model") or _llama_public_model_id(llama_backend) or "default"),
            prompt = prompt_text,
            context_length = llama_backend.context_length,
            subject = current_subject,
        )

    # Same gate registration as the completions proxy: unregistered, a non-forced /unload counts
    # zero generations and kills llama-server mid-embedding. Unpooled client so a cancel-close
    # hits this call only.
    _cancel_event = threading.Event()
    _client = _cancelable_nonstreaming_client()
    _tracker = _TrackedCancel(
        _cancel_event,
        model = str(body.get("model") or _llama_public_model_id(llama_backend) or "default"),
        kind = "embeddings",
    )
    _tracker.__enter__()
    _cancel_watcher = asyncio.create_task(
        _await_cancel_or_disconnect_then_close_client(
            cancel_event = _cancel_event,
            request = request,
            client = _client,
        )
    )
    try:
        try:
            resp = await _client.post(
                target_url,
                json = body,
                timeout = _DEFAULT_FIRST_TOKEN_TIMEOUT_S,
            )
        except httpx.RequestError:
            # The watcher closed the client out from under the request: report the cancel, not a transport failure.
            if _cancel_event.is_set():
                raise asyncio.CancelledError()
            raise
        if _cancel_event.is_set():
            raise asyncio.CancelledError()
    except asyncio.CancelledError:
        api_monitor.finish(monitor_id, "cancelled")
        raise
    except Exception as exc:
        api_monitor.fail(monitor_id, _friendly_error(exc))
        raise
    finally:
        # Nested so a close-time failure still unregisters; a phantom entry 409s swaps.
        try:
            await _stop_local_disconnect_cancel_watcher(_cancel_watcher)
            try:
                await _client.aclose()
            except Exception:
                pass
        finally:
            _tracker.__exit__(None, None, None)
    if resp.status_code != 200:
        api_monitor.fail(monitor_id, resp.text[:500])
    else:
        try:
            _monitor_openai_chunk(monitor_id, resp.json(), _monitor_context_length())
        except Exception:
            pass
        api_monitor.finish(monitor_id)
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


def _responses_tool_output_content(output: Union[str, list]) -> Union[str, list]:
    """Return Chat Completions-safe content for a Responses tool result."""
    if isinstance(output, str):
        return output if output.strip() else "(no output)"

    if not output:
        return "(no output)"

    text_parts: list[str] = []
    chat_parts: list = []
    has_multimodal = False
    for part in output:
        if not isinstance(part, dict):
            return json.dumps(output)
        part_type = part.get("type")
        if part_type in ("input_text", "output_text", "text"):
            text = part.get("text")
            if text is None:
                _raise_unsupported_openai_parameter(
                    "input",
                    "Responses function_call_output.output text parts require a text field.",
                )
            text = str(text)
            text_parts.append(text)
            chat_parts.append(TextContentPart(type = "text", text = text))
            continue
        if part_type == "input_image":
            image_url = part.get("image_url")
            if not isinstance(image_url, str) or not image_url:
                if part.get("file_id"):
                    _raise_unsupported_openai_parameter(
                        "input",
                        "Responses function_call_output.output input_image parts with file_id are not supported by the local adapter. Use image_url instead.",
                    )
                _raise_unsupported_openai_parameter(
                    "input",
                    "Responses function_call_output.output input_image parts require an image_url string.",
                )
            detail = part.get("detail", "auto")
            if detail is None:
                detail = "auto"
            if detail not in ("auto", "low", "high", "original"):
                _raise_unsupported_openai_parameter(
                    "input",
                    "Responses function_call_output.output input_image detail must be auto, low, high, or original.",
                )
            chat_parts.append(
                ImageContentPart(
                    type = "image_url",
                    image_url = ImageUrl(url = image_url, detail = detail),
                )
            )
            has_multimodal = True
            continue
        if part_type == "input_file":
            _raise_unsupported_openai_parameter(
                "input",
                "Responses function_call_output.output input_file parts are not supported by the local adapter.",
            )
        return json.dumps(output)

    if has_multimodal:
        return chat_parts

    text = "\n".join(text_parts)
    return text if text.strip() else "(no output)"


_RESPONSES_THINK_OPEN = "<think>"
_RESPONSES_THINK_CLOSE = "</think>"
_RESPONSES_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "max", "xhigh"}


def _coerce_responses_reasoning_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_coerce_responses_reasoning_text(part) for part in value)
    if isinstance(value, dict):
        for key in ("text", "reasoning_text", "content"):
            text = _coerce_responses_reasoning_text(value.get(key))
            if text:
                return text
        return ""
    return json.dumps(value)


def _responses_marker_holdback(text: str, markers: tuple[str, ...]) -> int:
    """Number of trailing chars to retain because they may start a marker."""
    for size in range(min(len(text), max(len(m) for m in markers) - 1), 0, -1):
        suffix = text[-size:]
        if any(marker.startswith(suffix) for marker in markers):
            return size
    return 0


class _ResponsesReasoningExtractor:
    """Split local <think> markup into Responses reasoning and visible text."""

    def __init__(
        self,
        *,
        parse_think_markers: bool = False,
        reasoning_prefilled: bool = False,
    ) -> None:
        self._buffer = ""
        # reasoning_prefilled: the template inserts an unclosed <think>, so output begins inside
        # the block; start in reasoning until the first close tag. Existing callers pass False.
        self._in_reasoning = reasoning_prefilled
        # Splitting requires marker parsing; a prefilled open implies it.
        self._parse_think_markers = parse_think_markers or reasoning_prefilled

    def feed(
        self,
        text: str = "",
        reasoning_content: Any = None,
    ) -> tuple[str, str]:
        reasoning_parts: list[str] = []
        visible_parts: list[str] = []
        structured_reasoning = _coerce_responses_reasoning_text(reasoning_content)
        if structured_reasoning:
            reasoning_parts.append(structured_reasoning)
        if text:
            self._buffer += text
        if not self._parse_think_markers:
            visible_parts.append(self._buffer)
            self._buffer = ""
            return "".join(reasoning_parts), "".join(visible_parts)

        while self._buffer:
            if self._in_reasoning:
                close_idx = self._buffer.find(_RESPONSES_THINK_CLOSE)
                if close_idx != -1:
                    reasoning_parts.append(
                        self._buffer[:close_idx].replace(_RESPONSES_THINK_OPEN, "")
                    )
                    self._buffer = self._buffer[close_idx + len(_RESPONSES_THINK_CLOSE) :]
                    self._in_reasoning = False
                    continue
                # Hold back a trailing partial of either marker: the close (clean split across chunks)
                # and a stray open (a re-emitted <think> is suppressed, not leaked).
                keep = _responses_marker_holdback(
                    self._buffer, (_RESPONSES_THINK_CLOSE, _RESPONSES_THINK_OPEN)
                )
                if keep == len(self._buffer):
                    break
                emit = self._buffer[:-keep] if keep else self._buffer
                reasoning_parts.append(emit.replace(_RESPONSES_THINK_OPEN, ""))
                self._buffer = self._buffer[-keep:] if keep else ""
                break

            open_idx = self._buffer.find(_RESPONSES_THINK_OPEN)
            close_idx = self._buffer.find(_RESPONSES_THINK_CLOSE)
            if close_idx != -1 and (open_idx == -1 or close_idx < open_idx):
                visible_parts.append(self._buffer[:close_idx])
                self._buffer = self._buffer[close_idx + len(_RESPONSES_THINK_CLOSE) :]
                continue
            if open_idx != -1:
                visible_parts.append(self._buffer[:open_idx])
                self._buffer = self._buffer[open_idx + len(_RESPONSES_THINK_OPEN) :]
                self._in_reasoning = True
                continue

            keep = _responses_marker_holdback(
                self._buffer,
                (_RESPONSES_THINK_OPEN, _RESPONSES_THINK_CLOSE),
            )
            if keep == len(self._buffer):
                break
            visible_parts.append(self._buffer[:-keep] if keep else self._buffer)
            self._buffer = self._buffer[-keep:] if keep else ""
            break

        return "".join(reasoning_parts), "".join(visible_parts)

    def finish(self) -> tuple[str, str]:
        if not self._buffer:
            return "", ""
        remaining = self._buffer
        self._buffer = ""
        if not self._parse_think_markers:
            return "", remaining
        if self._in_reasoning:
            self._in_reasoning = False
            return remaining.replace(_RESPONSES_THINK_OPEN, ""), ""
        return "", remaining.replace(_RESPONSES_THINK_CLOSE, "")


def _extract_responses_reasoning(
    text: str = "",
    reasoning_content: Any = None,
    *,
    parse_think_markers: bool = False,
    reasoning_prefilled: bool = False,
) -> tuple[str, str]:
    extractor = _ResponsesReasoningExtractor(
        parse_think_markers = parse_think_markers,
        reasoning_prefilled = reasoning_prefilled,
    )
    reasoning, visible = extractor.feed(text, reasoning_content)
    final_reasoning, final_visible = extractor.finish()
    return reasoning + final_reasoning, visible + final_visible


def _responses_should_parse_think_markers(
    chat_req: ChatCompletionRequest, llama_backend: Any = None
) -> bool:
    if llama_backend is not None and getattr(llama_backend, "is_loaded", False):
        if getattr(llama_backend, "reasoning_always_on", False):
            return True
        if getattr(llama_backend, "supports_reasoning", False):
            return True
        return False
    if chat_req.enable_thinking is True:
        return True
    return chat_req.enable_thinking is None and chat_req.reasoning_effort not in (None, "none")


def _responses_reasoning_output_item(reasoning_text: str, item_id: Optional[str] = None) -> dict:
    kwargs: dict[str, Any] = {
        "status": "completed",
        "summary": [],
        "content": [ResponsesOutputReasoningContent(text = reasoning_text)],
    }
    if item_id is not None:
        kwargs["id"] = item_id
    return ResponsesOutputReasoning(**kwargs).model_dump()


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

    def _with_system(msgs: list[ChatMessage]) -> list[ChatMessage]:
        if not system_parts:
            return msgs
        merged = "\n\n".join(p for p in system_parts if p)
        return [ChatMessage(role = "system", content = merged), *msgs]

    # Simple string input
    if isinstance(payload.input, str):
        if payload.input:
            messages.append(ChatMessage(role = "user", content = payload.input))
        return _with_system(messages)

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
            # Flatten pure text arrays for broad template compatibility, and
            # forward image URL outputs as real multimodal parts for vision models.
            output = _responses_tool_output_content(item.output)
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

    return _with_system(messages)


def _build_chat_request(
    payload: ResponsesRequest, messages: list[ChatMessage], stream: bool
) -> ChatCompletionRequest:
    """Build a ChatCompletionRequest from a ResponsesRequest.

    Tools and ``tool_choice`` are translated from the flat Responses shape to
    the nested Chat Completions shape here so the existing #5099
    ``/v1/chat/completions`` client-side pass-through picks them up unchanged.
    """
    chat_kwargs: dict = dict(
        messages = messages,
        stream = stream,
    )
    # Only forward an explicitly set model so an omitted Responses model stays
    # reload-only when openai_chat_completions re-checks on the non-streaming path.
    if "model" in payload.model_fields_set:
        chat_kwargs["model"] = payload.model
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

    # ``chat_template_kwargs`` (e.g. ``{"enable_thinking": true}``) arrives via
    # the Responses extra-body: ResponsesRequest has ``extra="allow"``, so the
    # OpenAI SDK's ``extra_body`` spread lands the dict in ``model_extra``. The
    # downstream Chat Completions paths consume the typed ``enable_thinking``
    # field -- the non-streaming path lifts it in ``openai_chat_completions``
    # only when it is still ``None``, and the streaming pass-through reads
    # ``payload.enable_thinking`` directly -- so lift it here, mirroring that
    # handler, to cover both Responses paths.
    explicit_enable_thinking = False
    _extra = getattr(payload, "model_extra", None)
    if isinstance(_extra, dict):
        _tpl_kw = _extra.get("chat_template_kwargs")
        if isinstance(_tpl_kw, dict) and "enable_thinking" in _tpl_kw:
            chat_kwargs["enable_thinking"] = bool(_tpl_kw["enable_thinking"])
            explicit_enable_thinking = True
        # auto_heal_tool_calls / nudge_tool_calls are not typed on
        # ResponsesRequest; lift them from the extra-body so passthrough
        # healing (and the opt-in nudge) honor them on both paths.
        if isinstance(_extra.get("auto_heal_tool_calls"), bool):
            chat_kwargs["auto_heal_tool_calls"] = _extra["auto_heal_tool_calls"]
        if isinstance(_extra.get("nudge_tool_calls"), bool):
            chat_kwargs["nudge_tool_calls"] = _extra["nudge_tool_calls"]

    if isinstance(payload.reasoning, dict):
        effort = payload.reasoning.get("effort")
        if isinstance(effort, str) and effort in _RESPONSES_REASONING_EFFORTS:
            if not explicit_enable_thinking:
                chat_kwargs["reasoning_effort"] = effort
                chat_kwargs["enable_thinking"] = effort != "none"
            elif chat_kwargs.get("enable_thinking") is False:
                chat_kwargs["reasoning_effort"] = "none"
            elif effort != "none":
                chat_kwargs["reasoning_effort"] = effort

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
    payload: ResponsesRequest,
    messages: list[ChatMessage],
    request: Request,
    current_subject: Optional[str] = None,
) -> JSONResponse:
    """Handle a non-streaming Responses API call."""
    chat_req = _build_chat_request(payload, messages, stream = False)
    request_state = getattr(request, "state", None)
    if request_state is None:
        request_state = type("_RequestState", (), {})()
        try:
            setattr(request, "state", request_state)
        except Exception:
            request_state = None
    previous_skip_monitor = (
        bool(getattr(request_state, "skip_api_monitor", False))
        if request_state is not None
        else False
    )
    monitor_id = None
    if not previous_skip_monitor:
        monitor_id = api_monitor.start(
            endpoint = getattr(getattr(request, "url", None), "path", "/v1/responses"),
            method = getattr(request, "method", "POST"),
            via_api_key = _request_used_api_key(request),
            model = payload.model,
            prompt = _monitor_prompt_from_messages(messages),
            context_length = _monitor_context_length(),
            subject = current_subject,
        )
    if request_state is not None:
        request_state.skip_api_monitor = True

    try:
        result = await openai_chat_completions(chat_req, request)

        # openai_chat_completions returns a JSONResponse for non-streaming.
        if isinstance(result, Response):
            body = json.loads(result.body.decode())
        else:
            body = result

        choices = body.get("choices", [])
        text = ""
        reasoning_text = ""
        tool_calls: list[dict] = []
        if choices:
            msg = choices[0].get("message", {}) or {}
            raw_content = msg.get("content", "") or ""
            raw_text = raw_content if isinstance(raw_content, str) else json.dumps(raw_content)
            llama_backend = get_llama_cpp_backend()
            reasoning_text, text = _extract_responses_reasoning(
                raw_text,
                msg.get("reasoning_content"),
                parse_think_markers = _responses_should_parse_think_markers(chat_req, llama_backend),
            )
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
        if reasoning_text:
            output_items.append(_responses_reasoning_output_item(reasoning_text))
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
        api_monitor.set_reply(monitor_id, text or _monitor_tool_calls_text(tool_calls))
        _monitor_usage(monitor_id, usage_data, _monitor_context_length())
        api_monitor.finish(monitor_id)
        return _model_json_response(response)
    except asyncio.CancelledError:
        api_monitor.finish(monitor_id, "cancelled")
        raise
    except Exception as exc:
        api_monitor.fail(monitor_id, _friendly_error(exc))
        raise
    finally:
        if request_state is not None:
            request_state.skip_api_monitor = previous_skip_monitor


async def _responses_stream(
    payload: ResponsesRequest,
    messages: list[ChatMessage],
    request: Request,
    monitor_id: Optional[str] = None,
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

    Output items are allocated as upstream deltas appear. Reasoning/text deltas
    open top-level ``reasoning`` / ``message`` items; each tool call from
    ``delta.tool_calls[]`` is promoted to its own top-level ``function_call``
    item (one per distinct ``tool_calls[].index``) and relayed as
    ``response.function_call_arguments.delta`` / ``.done`` events so clients
    (Codex, OpenAI Python SDK) can reconstruct the call incrementally and reply
    with a ``function_call_output`` item next turn.
    """
    resp_id = f"resp_{uuid.uuid4().hex[:12]}"
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
        _status, _detail = await _no_model_loaded_error(
            "Streaming /v1/responses requires a GGUF model loaded via "
            "llama-server. Use non-streaming /v1/responses, "
            "/v1/chat/completions, or load a GGUF model.",
            _switch_model_for_payload(payload),
            request,
            status = 400,
        )
        raise HTTPException(status_code = _status, detail = _detail)

    # Direct pass-through bypasses the openai_chat_completions image gate.
    if not llama_backend.is_vision and any(
        isinstance(m.content, list) and any(isinstance(p, ImageContentPart) for p in m.content)
        for m in messages
    ):
        raise HTTPException(
            status_code = 400,
            detail = "Image provided but current GGUF model does not support vision.",
        )

    # Streaming /v1/responses builds the passthrough body directly (bypassing
    # openai_chat_completions), so apply recommended sampling here too.
    _fill_recommended_sampling_openai(chat_req, getattr(llama_backend, "model_identifier", None))
    body = _build_openai_passthrough_body(
        chat_req, backend_ctx = llama_backend.context_length, llama_backend = llama_backend
    )
    body["stream_options"] = {"include_usage": True}
    target_url = f"{llama_backend.base_url}/v1/chat/completions"
    # The stream's own disconnect event, shared with the cancel/active-generation registries:
    # this path decodes on llama-server, so a non-forced /unload must see it and refuse instead
    # of tearing the server down mid-response. Entered inside the body generator below, so a
    # response whose body never starts leaves nothing behind.
    cancel_event = threading.Event()
    _tracker = _TrackedCancel.for_payload(cancel_event, payload, resp_id)
    try:
        reservation, admission_config = _openai_llama_admission_reserve(
            request = request,
            llama_backend = llama_backend,
        )
    except LlamaAdmissionQueueFull as exc:
        _llama_admission_log(
            "queue-full",
            snapshot = exc.snapshot,
            request = request,
            mode = "responses_stream",
            completion_id = resp_id,
            level = "warning",
        )
        api_monitor.fail(monitor_id, str(exc))
        raise _openai_admission_http_exception(exc, status_code = 429)

    def _responses_admission_failed_sse(exc: Exception, *, status_code: int) -> str:
        return (
            "event: response.failed\n"
            "data: "
            + json.dumps(
                {
                    "type": "response.failed",
                    "response": {
                        "id": resp_id,
                        "object": "response",
                        "created_at": created_at,
                        "status": "failed",
                        "model": _llama_public_model_id(llama_backend, payload.model)
                        or payload.model,
                        "output": [],
                        "usage": {
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                        "error": {
                            "code": status_code,
                            "message": str(exc),
                        },
                    },
                }
            )
            + "\n\n"
        )

    async def event_generator():
        # Clean public id for every response envelope. Prefer the loaded model's
        # id so the stream agrees with /v1/models, chat/completions and the
        # non-streaming twin; fall back to a sanitized payload.model (a legacy
        # raw .gguf path is stripped, never echoed back). Use the advertised-id
        # helper, not the raw identifier: after an auto-switch to a cached HF GGUF
        # the identifier is the snapshot path while the repo id lives in
        # _openai_advertised_id, so the raw form would stream a snapshot basename.
        _clean_model = _llama_public_model_id(llama_backend, payload.model) or payload.model
        full_text = ""
        full_reasoning = ""
        input_tokens = 0
        output_tokens = 0
        extractor = _ResponsesReasoningExtractor(
            parse_think_markers = _responses_should_parse_think_markers(chat_req, llama_backend)
        )
        reasoning_state: dict[str, Any] = {"output_index": None, "item_id": None, "opened": False}
        message_state: dict[str, Any] = {
            "output_index": None,
            "item_id": None,
            "opened": False,
            "text": "",
        }
        # Message items already closed mid-stream (a healed tool call splits
        # the assistant text into separate message items, as native Responses
        # streams do). Kept for the final response.completed snapshot.
        closed_message_states: list[dict] = []
        # Per-tool-call state keyed by Chat Completions `tool_calls[].index`,
        # stable across chunks for the same call. Values:
        #   {output_index, item_id, call_id, name, arguments, opened}
        tool_call_state: dict[int, dict] = {}
        next_output_index = 0
        # Text-form tool calls promoted back to structured calls (declared
        # client tools only); dormant once grammar-mode structured deltas appear.
        _allowed_tools = heal_gate(
            getattr(chat_req, "auto_heal_tool_calls", None),
            body.get("tools"),
            body.get("tool_choice"),
        )
        healer = StreamToolCallHealer(_allowed_tools, body.get("tools")) if _allowed_tools else None
        healed_tc_index = 0

        def _healed_tc(call: dict):
            # Chat-delta shape for a healed call. Indexes live in a disjoint
            # range so a healed call can never merge into a structured call's
            # state slot; parallel_tool_calls=false caps healed calls too (the
            # upstream cap ran before injection).
            nonlocal healed_tc_index
            if payload.parallel_tool_calls is False and healed_tc_index >= 1:
                return None
            tc = {
                "index": 1_000_000 + healed_tc_index,
                "id": call["id"],
                "type": "function",
                "function": call["function"],
            }
            healed_tc_index += 1
            return tc

        def _sse(event_name: str, payload: dict) -> str:
            return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n"

        def _tool_call_delta_events(tc: dict) -> list:
            # One Chat Completions tool_calls delta -> Responses SSE events,
            # allocating/merging per-call state (shared by the structured loop
            # and the healer's promoted calls).
            events = []
            idx = tc.get("index", 0)
            st = tool_call_state.get(idx)
            fn = tc.get("function") or {}
            if st is None:
                # First chunk for this tool call -- allocate an
                # output_index and emit output_item.added.
                st = {
                    "output_index": _claim_output_index(),
                    "item_id": f"fc_{uuid.uuid4().hex[:12]}",
                    "call_id": tc.get("id") or "",
                    "name": fn.get("name") or "",
                    "arguments": "",
                    "opened": False,
                }
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
                events.append(_sse("response.output_item.added", item_added))
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
                events.append(_sse("response.function_call_arguments.delta", args_delta_event))
            elif arg_delta:
                # Buffer args until we can open the item (some models
                # send id/name in the same chunk as the first arg delta;
                # if not, stash).
                st["arguments"] += arg_delta
            return events

        def _claim_output_index() -> int:
            nonlocal next_output_index
            output_index = next_output_index
            next_output_index += 1
            return output_index

        def _apply_usage(u) -> None:
            nonlocal input_tokens, output_tokens
            if not u:
                return
            input_tokens = u.get("prompt_tokens", input_tokens)
            output_tokens = u.get("completion_tokens", output_tokens)
            _monitor_usage(monitor_id, u, llama_backend.context_length)

        def _ensure_reasoning_open() -> list[str]:
            if reasoning_state["opened"]:
                return []
            reasoning_state["output_index"] = _claim_output_index()
            reasoning_state["item_id"] = f"rs_{uuid.uuid4().hex[:12]}"
            reasoning_state["opened"] = True
            output_index = reasoning_state["output_index"]
            item_id = reasoning_state["item_id"]
            return [
                _sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "type": "reasoning",
                            "id": item_id,
                            "status": "in_progress",
                            "summary": [],
                            "content": [],
                        },
                    },
                ),
                _sse(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {"type": "reasoning_text", "text": ""},
                    },
                ),
            ]

        def _ensure_message_open() -> list[str]:
            if message_state["opened"]:
                return []
            message_state["output_index"] = _claim_output_index()
            message_state["item_id"] = f"msg_{uuid.uuid4().hex[:12]}"
            message_state["opened"] = True
            output_index = message_state["output_index"]
            item_id = message_state["item_id"]
            return [
                _sse(
                    "response.output_item.added",
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": {
                            "type": "message",
                            "id": item_id,
                            "status": "in_progress",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                ),
                _sse(
                    "response.content_part.added",
                    {
                        "type": "response.content_part.added",
                        "item_id": item_id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []},
                    },
                ),
            ]

        def _close_message_item() -> list[str]:
            """Close the open message item so later text opens a fresh one.

            Emits the same done-event triplet the end-of-stream close loop
            would, records the item for the final snapshot, and resets the
            state in place. No-op when no message item is open.
            """
            if not message_state["opened"]:
                return []
            text = message_state["text"]
            events = [
                _sse(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "item_id": message_state["item_id"],
                        "output_index": message_state["output_index"],
                        "content_index": 0,
                        "text": text,
                    },
                ),
                _sse(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "item_id": message_state["item_id"],
                        "output_index": message_state["output_index"],
                        "content_index": 0,
                        "part": {"type": "output_text", "text": text, "annotations": []},
                    },
                ),
                _sse(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": message_state["output_index"],
                        "item": {
                            "type": "message",
                            "id": message_state["item_id"],
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": text, "annotations": []}],
                        },
                    },
                ),
            ]
            closed_message_states.append(dict(message_state))
            message_state.update(
                {"output_index": None, "item_id": None, "opened": False, "text": ""}
            )
            return events

        def _healed_event_sse(events) -> list[str]:
            """Serialize healer events preserving their order.

            Text around a healed call must keep its position relative to the
            function_call item (output indexes are claimed in emission order),
            so never split an event list into all-text-then-all-calls. A healed
            call also CLOSES any open message item, so trailing text opens a
            fresh message with a later output index, exactly like a native
            Responses stream that interleaves messages and calls.
            """
            nonlocal full_text
            out: list[str] = []
            for kind, value in events:
                if kind == "text":
                    if not value:
                        continue
                    out.extend(_ensure_message_open())
                    full_text += value
                    message_state["text"] += value
                    api_monitor.append_reply(monitor_id, value)
                    out.append(
                        _sse(
                            "response.output_text.delta",
                            {
                                "type": "response.output_text.delta",
                                "item_id": message_state["item_id"],
                                "output_index": message_state["output_index"],
                                "content_index": 0,
                                "delta": value,
                            },
                        )
                    )
                else:
                    tc = _healed_tc(value)
                    if tc is None:
                        continue
                    out.extend(_close_message_item())
                    out.extend(_tool_call_delta_events(tc))
            return out

        def _snapshot_output() -> list[dict]:
            """Snapshot of all completed output items for response.completed."""
            indexed_items: list[tuple[int, dict]] = []
            if reasoning_state["opened"]:
                indexed_items.append(
                    (
                        reasoning_state["output_index"],
                        {
                            "type": "reasoning",
                            "id": reasoning_state["item_id"],
                            "status": "completed",
                            "summary": [],
                            "content": [{"type": "reasoning_text", "text": full_reasoning}],
                        },
                    )
                )
            # Closed copies keep opened=True (snapshotted before reset); the
            # live state contributes only when a message is currently open.
            for msg_st in [*closed_message_states, message_state]:
                if not msg_st["opened"]:
                    continue
                indexed_items.append(
                    (
                        msg_st["output_index"],
                        {
                            "type": "message",
                            "id": msg_st["item_id"],
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": msg_st["text"],
                                    "annotations": [],
                                }
                            ],
                        },
                    )
                )
            for st in tool_call_state.values():
                indexed_items.append(
                    (
                        st["output_index"],
                        {
                            "type": "function_call",
                            "id": st["item_id"],
                            "status": "completed",
                            "call_id": st["call_id"],
                            "name": st["name"],
                            "arguments": st["arguments"],
                        },
                    )
                )
            return [item for _, item in sorted(indexed_items, key = lambda pair: pair[0])]

        def _failed_response_payload(exc: Exception, status_code: int) -> dict:
            return {
                "type": "response.failed",
                "response": {
                    "id": resp_id,
                    "object": "response",
                    "created_at": created_at,
                    "status": "failed",
                    "model": _clean_model,
                    "output": _snapshot_output(),
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": input_tokens + output_tokens,
                    },
                    "error": {
                        "code": status_code,
                        "message": _friendly_error(exc),
                    },
                },
            }

        # ── Preamble events ──
        yield _sse(
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": resp_id,
                    "object": "response",
                    "created_at": created_at,
                    "status": "in_progress",
                    "model": _clean_model,
                    "output": [],
                    "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                },
            },
        )

        # ── Direct httpx lifecycle to llama-server ──
        # Full same-task open + close, same pattern as
        # _openai_passthrough_stream and _anthropic_passthrough_stream: no
        # `async with`, explicit aclose of lines_iter BEFORE resp / client so
        # the innermost httpcore byte stream is finalised in this task (not via
        # the asyncgen GC in a sibling task).
        client = httpx.AsyncClient(
            timeout = _llama_streaming_generation_timeout(),
            trust_env = False,
        )
        resp = None
        lines_iter = None
        disconnect_watcher = None
        # Tracked per-run event: a client disconnect and a forced reload both land here.
        disconnect_event = cancel_event
        try:
            req = client.build_request(
                "POST", target_url, json = body, headers = {"Connection": "close"}
            )
            first_token_deadline = time.monotonic() + _DEFAULT_FIRST_TOKEN_TIMEOUT_S
            try:
                # Same event the loop below polls: prefill can run for the whole first-token window,
                # and only the send watcher can end it early.
                resp = await _send_stream_with_preheader_cancel(
                    client, req, disconnect_event, request = request
                )
                if resp is None:
                    api_monitor.finish(monitor_id, "cancelled")
                    return
            except httpx.RequestError as e:
                logger.error("responses stream: upstream unreachable: %s", e)
                api_monitor.fail(monitor_id, _friendly_error(e))
                yield _sse(
                    "response.failed",
                    {
                        "type": "response.failed",
                        "response": {
                            "id": resp_id,
                            "object": "response",
                            "created_at": created_at,
                            "status": "failed",
                            "model": _clean_model,
                            "output": [],
                            "error": {"code": 502, "message": _friendly_error(e)},
                        },
                    },
                )
                return

            if resp.status_code != 200:
                err_bytes = await resp.aread()
                err_text = err_bytes.decode("utf-8", errors = "replace")
                logger.error(
                    "responses stream upstream error: status=%s body=%s",
                    resp.status_code,
                    err_text[:500],
                )
                api_monitor.fail(monitor_id, err_text[:500])
                yield _sse(
                    "response.failed",
                    {
                        "type": "response.failed",
                        "response": {
                            "id": resp_id,
                            "object": "response",
                            "created_at": created_at,
                            "status": "failed",
                            "model": _clean_model,
                            "output": [],
                            "error": {
                                "code": resp.status_code,
                                "message": _friendly_upstream_error(err_text[:500]),
                            },
                        },
                    },
                )
                return

            lines_iter = resp.aiter_lines()
            disconnect_watcher = asyncio.create_task(
                _await_disconnect_then_close(request, resp, disconnect_event)
            )
            async for raw_line in _aiter_llama_stream_items(
                lines_iter,
                cancel_event = disconnect_event,
                request = request,
                first_token_deadline = first_token_deadline,
                response = resp,
            ):
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
                    _apply_usage(chunk_data.get("usage"))
                    continue

                delta = choices[0].get("delta", {}) or {}
                reasoning_delta, visible_delta = extractor.feed(
                    delta.get("content") or "",
                    delta.get("reasoning_content"),
                )
                if reasoning_delta:
                    for event in _ensure_reasoning_open():
                        yield event
                    full_reasoning += reasoning_delta
                    yield _sse(
                        "response.reasoning_text.delta",
                        {
                            "type": "response.reasoning_text.delta",
                            "item_id": reasoning_state["item_id"],
                            "output_index": reasoning_state["output_index"],
                            "content_index": 0,
                            "delta": reasoning_delta,
                        },
                    )
                # Heal text-form tool calls in the visible stream (never in
                # reasoning text): promoted calls join the structured tc loop
                # below through the same state machinery, and healer events are
                # emitted IN ORDER so text after a healed call never jumps ahead
                # of the function_call item. Once a structured delta arrives,
                # grammar mode worked and the healer goes dormant.
                if healer is not None and not healer.dormant:
                    healed_events = []
                    if delta.get("tool_calls"):
                        # Held text preceded the structured call; the call's own
                        # deltas follow in the structured loop below.
                        healed_events = healer.structured_tool_call_seen()
                        if visible_delta:
                            healed_events.append(("text", visible_delta))
                    elif visible_delta:
                        healed_events = healer.feed(visible_delta)
                    visible_delta = ""
                    for event in _healed_event_sse(healed_events):
                        yield event
                if visible_delta:
                    for event in _ensure_message_open():
                        yield event
                    full_text += visible_delta
                    message_state["text"] += visible_delta
                    api_monitor.append_reply(monitor_id, visible_delta)
                    yield _sse(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": message_state["item_id"],
                            "output_index": message_state["output_index"],
                            "content_index": 0,
                            "delta": visible_delta,
                        },
                    )

                for tc in delta.get("tool_calls") or []:
                    if (
                        payload.parallel_tool_calls is False
                        and healed_tc_index >= 1
                        and tc.get("index", 0) not in tool_call_state
                    ):
                        # A healed call already consumed the single allowed slot;
                        # _drop_parallel_tool_call_deltas only sees native indexes,
                        # so a native index-0 call would still open a second
                        # function_call item. Skip it (and its later argument
                        # deltas, which never allocate a state either).
                        continue
                    for event in _tool_call_delta_events(tc):
                        yield event

                _apply_usage(chunk_data.get("usage"))
        except asyncio.CancelledError:
            disconnect_event.set()
            api_monitor.finish(monitor_id, "cancelled")
            raise
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.CloseError) as e:
            if not disconnect_event.is_set():
                logger.error("responses stream error: %s", e)
                api_monitor.fail(monitor_id, _friendly_error(e))
                status_code = 400 if _classify_llama_generation_error(e) is not None else 500
                yield _sse(
                    "response.failed",
                    _failed_response_payload(e, status_code),
                )
                return
        except Exception as e:
            if disconnect_event.is_set():
                api_monitor.finish(monitor_id, "cancelled")
                return
            logger.error("responses stream error: %s", e)
            api_monitor.fail(monitor_id, _friendly_error(e))
            status_code = 400 if _classify_llama_generation_error(e) is not None else 500
            yield _sse(
                "response.failed",
                _failed_response_payload(e, status_code),
            )
            return
        finally:
            await _aclose_stream_resources(
                watchers = (disconnect_watcher,),
                iterator = lines_iter,
                resp = resp,
                client = client,
            )

        if disconnect_event.is_set():
            api_monitor.finish(monitor_id, "cancelled")
            return

        final_reasoning, final_visible = extractor.finish()
        if final_reasoning:
            for event in _ensure_reasoning_open():
                yield event
            full_reasoning += final_reasoning
            yield _sse(
                "response.reasoning_text.delta",
                {
                    "type": "response.reasoning_text.delta",
                    "item_id": reasoning_state["item_id"],
                    "output_index": reasoning_state["output_index"],
                    "content_index": 0,
                    "delta": final_reasoning,
                },
            )
        # Last-chance heal of any held residue (e.g. a tool block the model
        # never closed) before the trailing visible text is flushed; events
        # keep healer order so trailing text stays behind a healed call.
        if healer is not None:
            events = (healer.feed(final_visible) if final_visible else []) + healer.finalize()
            final_visible = ""
            for event in _healed_event_sse(events):
                yield event
        if final_visible:
            for event in _ensure_message_open():
                yield event
            full_text += final_visible
            message_state["text"] += final_visible
            api_monitor.append_reply(monitor_id, final_visible)
            yield _sse(
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": message_state["item_id"],
                    "output_index": message_state["output_index"],
                    "content_index": 0,
                    "delta": final_visible,
                },
            )

        close_items: list[tuple[int, str, dict[str, Any]]] = []
        if reasoning_state["opened"]:
            close_items.append((reasoning_state["output_index"], "reasoning", reasoning_state))
        if message_state["opened"]:
            close_items.append((message_state["output_index"], "message", message_state))
        close_items.extend((st["output_index"], "tool", st) for st in tool_call_state.values())

        for _, kind, st in sorted(close_items, key = lambda item: item[0]):
            if kind == "reasoning":
                yield _sse(
                    "response.reasoning_text.done",
                    {
                        "type": "response.reasoning_text.done",
                        "item_id": st["item_id"],
                        "output_index": st["output_index"],
                        "content_index": 0,
                        "text": full_reasoning,
                    },
                )
                yield _sse(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "item_id": st["item_id"],
                        "output_index": st["output_index"],
                        "content_index": 0,
                        "part": {"type": "reasoning_text", "text": full_reasoning},
                    },
                )
                yield _sse(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": st["output_index"],
                        "item": {
                            "type": "reasoning",
                            "id": st["item_id"],
                            "status": "completed",
                            "summary": [],
                            "content": [{"type": "reasoning_text", "text": full_reasoning}],
                        },
                    },
                )
                continue

            if kind == "message":
                # Per-item text: message items closed mid-stream (healed-call
                # rotation) already emitted their done events, so this state
                # carries only its own text, not the whole stream's.
                _msg_text = st["text"]
                yield _sse(
                    "response.output_text.done",
                    {
                        "type": "response.output_text.done",
                        "item_id": st["item_id"],
                        "output_index": st["output_index"],
                        "content_index": 0,
                        "text": _msg_text,
                    },
                )
                yield _sse(
                    "response.content_part.done",
                    {
                        "type": "response.content_part.done",
                        "item_id": st["item_id"],
                        "output_index": st["output_index"],
                        "content_index": 0,
                        "part": {"type": "output_text", "text": _msg_text, "annotations": []},
                    },
                )
                yield _sse(
                    "response.output_item.done",
                    {
                        "type": "response.output_item.done",
                        "output_index": st["output_index"],
                        "item": {
                            "type": "message",
                            "id": st["item_id"],
                            "status": "completed",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": _msg_text, "annotations": []}
                            ],
                        },
                    },
                )
                continue

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
                yield _sse("response.output_item.added", item_added)
                if st["arguments"]:
                    yield _sse(
                        "response.function_call_arguments.delta",
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": st["item_id"],
                            "output_index": st["output_index"],
                            "delta": st["arguments"],
                        },
                    )
                st["opened"] = True

            args_done = {
                "type": "response.function_call_arguments.done",
                "item_id": st["item_id"],
                "output_index": st["output_index"],
                "name": st["name"],
                "arguments": st["arguments"],
            }
            yield _sse("response.function_call_arguments.done", args_done)

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
            api_monitor.append_reply(monitor_id, _monitor_call_text(st["name"], st["arguments"]))
            yield _sse("response.output_item.done", item_done)

        # response.completed
        total_tokens = input_tokens + output_tokens
        completed_response = {
            "type": "response.completed",
            "response": {
                "id": resp_id,
                "object": "response",
                "created_at": created_at,
                "status": "completed",
                "model": _clean_model,
                "output": _snapshot_output(),
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                },
            },
        }
        api_monitor.finish(monitor_id)
        yield _sse("response.completed", completed_response)

    async def admitted_event_generator():
        # Register for the body's whole lifetime, admission wait included: the run holds a decode
        # slot from here on, so /load and /unload must count it. __exit__ runs from the finally below.
        _tracker.__enter__()
        lease = reservation.lease_nowait()
        admission_wait_started_at = None
        stream_started = False
        stream_cancelled = False
        iterator = None
        try:
            if lease is None:
                admission_wait_started_at = time.monotonic()
                _llama_admission_log(
                    "queued",
                    reservation,
                    request = request,
                    mode = "responses_stream",
                    completion_id = resp_id,
                    level = "debug",
                )
                # The tracked event, not just the client socket: registered above, so a forced swap's
                # cancel_all() reaches this run while it is still queued. Otherwise it takes a lease it was
                # told to give up and the post-cancel drain waits out the round trip it just cancelled.
                async for wait_item in _openai_admission_wait_stream_chunks(
                    reservation,
                    admission_config,
                    request = request,
                    cancel_event = cancel_event,
                ):
                    if isinstance(wait_item, str):
                        yield wait_item
                        continue
                    lease = wait_item
                    _llama_admission_log(
                        "granted-after-wait",
                        reservation,
                        request = request,
                        mode = "responses_stream",
                        wait_started_at = admission_wait_started_at,
                        completion_id = resp_id,
                        level = "debug",
                    )
                    break
            if lease is None:
                return
            await _raise_if_openai_admission_cancelled(
                reservation,
                request = request,
                cancel_event = cancel_event,
            )
            iterator = event_generator()
            stream_started = True
            try:
                async for chunk in iterator:
                    yield chunk
            except asyncio.CancelledError:
                stream_cancelled = True
                api_monitor.finish(monitor_id, "cancelled")
                raise
            finally:
                await _close_openai_admitted_stream_iterator(
                    iterator,
                    cancelled = stream_cancelled,
                )
        except LlamaAdmissionTimeout as exc:
            _llama_admission_log(
                "timeout",
                reservation,
                request = request,
                mode = "responses_stream",
                wait_started_at = admission_wait_started_at,
                completion_id = resp_id,
                level = "warning",
            )
            api_monitor.fail(monitor_id, str(exc))
            yield _responses_admission_failed_sse(exc, status_code = 503)
        except LlamaAdmissionCancelled:
            _llama_admission_log(
                "cancelled-before-upstream",
                reservation,
                request = request,
                mode = "responses_stream",
                wait_started_at = admission_wait_started_at,
                completion_id = resp_id,
                level = "debug",
            )
            api_monitor.finish(monitor_id, "cancelled")
            return
        except asyncio.CancelledError:
            api_monitor.finish(monitor_id, "cancelled")
            raise
        finally:
            if lease is not None:
                lease.release()
            if not stream_started:
                api_monitor.finish(monitor_id, "cancelled")
                reservation.cancel()
            _tracker.__exit__(None, None, None)

    async def _responses_admission_unstarted_cleanup() -> None:
        api_monitor.finish(monitor_id, "cancelled")
        reservation.cancel()

    return _SameTaskStreamingResponse(
        admitted_event_generator(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "close",
            "X-Accel-Buffering": "no",
        },
        unstarted_cleanup = _responses_admission_unstarted_cleanup,
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
    # System/developer-only input normalises to a non-empty list, so reject it
    # before the switch (mirror chat) or an invalid request evicts the resident
    # model only for the chat handler to 400 it as having no non-system message.
    if not any(m.role not in ("system", "developer") for m in messages):
        raise HTTPException(status_code = 400, detail = "At least one non-system message is required.")
    # Reject a malformed function tool before any model load, mirroring the
    # /v1/chat/completions check, so an invalid request never switches the model.
    # Built-in tools (web_search, mcp, ...) carry no name and are dropped later.
    for _tool in payload.tools or []:
        if not isinstance(_tool, dict) or _tool.get("type") != "function":
            continue
        _name = _tool.get("name")
        if not isinstance(_name, str) or not _name.strip():
            raise HTTPException(
                status_code = 400,
                detail = openai_error_body(
                    "Invalid 'tools': each function tool must have a 'name'.",
                    status = 400,
                    code = "invalid_value",
                    param = "tools",
                ),
            )
    # Reject a forcing-function tool_choice with no name before the switch (mirror
    # chat), so a malformed request can't evict the model. Responses forces with
    # {"type": "function", "name": "X"}; the streaming path would otherwise forward
    # the bad choice and the non-streaming path only 400s after the swap.
    _tc = payload.tool_choice
    if isinstance(_tc, dict) and _tc.get("type") == "function":
        _tc_name = _tc.get("name")
        if not isinstance(_tc_name, str) or not _tc_name.strip():
            raise HTTPException(
                status_code = 400,
                detail = openai_error_body(
                    "Invalid 'tool_choice': the forced function must have a 'name'.",
                    status = 400,
                    code = "invalid_value",
                    param = "tool_choice",
                ),
            )
    # After input validation so a 400 never triggers a load. Switches the
    # streaming path; non-streaming re-checks via the idempotent chat handler.
    # require_vision rejects a swap to a text-only target before it runs, so an
    # image request can't evict the resident vision model only to 400 afterwards
    # (the non-streaming chat re-check short-circuits on _already_serving).
    await _maybe_auto_switch_model(
        _switch_model_for_payload(payload),
        request,
        current_subject,
        require_vision = _messages_have_image(messages),
    )

    if payload.stream:
        monitor_id = None
        if not getattr(request.state, "skip_api_monitor", False):
            monitor_id = api_monitor.start(
                endpoint = request.url.path,
                via_api_key = _request_used_api_key(request),
                method = request.method,
                model = payload.model,
                prompt = _monitor_prompt_from_messages(messages),
                context_length = _monitor_context_length(),
                subject = current_subject,
            )
        try:
            return await _responses_stream(payload, messages, request, monitor_id)
        except HTTPException as exc:
            detail = exc.detail
            if not isinstance(detail, str):
                detail = json.dumps(detail, default = str)
            api_monitor.fail(monitor_id, detail)
            raise
        except Exception as exc:
            api_monitor.fail(monitor_id, _friendly_error(exc))
            raise
    return await _responses_non_streaming(payload, messages, request, current_subject)


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
# Server tools that never need a confirmation prompt (read-only / non code-
# executing; mirrors the unconditional-safe names in is_potentially_unsafe_tool_call).
# Any other selected tool (terminal, python, render_html) can require the gate
# this channel has no way to present, so an omitted permission_mode ("ask") only
# asks then. render_html is excluded because a networked canvas prompts in auto,
# and this channel invokes the loop without confirm; auto/ask reject, off/full run.
_ANTHROPIC_UNPROMPTED_SAFE_TOOLS = frozenset({"web_search", "search_knowledge_base"})


def _anthropic_requested_studio_tools(tools: Optional[list]) -> set[str]:
    requested: set[str] = set()
    for tool in tools or []:
        td = tool if isinstance(tool, dict) else tool.model_dump()
        if td.get("input_schema") is not None or anthropic_schema_client_tool_kind(td) is not None:
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
    """Select Unsloth tools requested through Anthropic tools and extensions."""
    if not requested_studio_tools and enabled_tools is None:
        return all_tools

    selected_names = set(requested_studio_tools)
    if enabled_tools is not None:
        selected_names.update(enabled_tools)

    return [tool for tool in all_tools if tool["function"]["name"] in selected_names]


def _image_bytes_to_png_b64(raw: bytes) -> str:
    """Decode raw image bytes and re-encode to a base64-ascii PNG string.

    llama-server's stb_image only handles a few formats (JPEG/PNG/BMP/...); re-
    encoding to PNG keeps JPEG/WebP/... inputs loadable. Raises on undecodable
    input; callers wrap the call in ``try`` -> HTTPException(400)."""
    from PIL import Image

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format = "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
                png_b64 = _image_bytes_to_png_b64(raw)
            except Exception:
                raise HTTPException(
                    status_code = 400,
                    detail = "Failed to process image.",
                )
            part["image_url"] = {"url": f"data:image/png;base64,{png_b64}"}

    return has_image


def _validate_anthropic_client_tools(tools) -> None:
    # Reject malformed client tools before any model load, so an invalid request
    # never evicts the loaded model. AnthropicTool relaxed name/input_schema to
    # Optional for server tools, so the converter silently drops incomplete
    # entries; surface them as 400 here. Recognized Anthropic-schema client
    # tools use type/name without input_schema; other type declarations are
    # server tools (unrecognized server tools remain no-ops).
    for tool in tools or []:
        td = tool if isinstance(tool, dict) else tool.model_dump()
        name, type_, schema = td.get("name"), td.get("type"), td.get("input_schema")
        schema_client_kind = anthropic_schema_client_tool_kind(td)
        if schema is None and not isinstance(type_, str):
            raise HTTPException(
                status_code = 400,
                detail = f"Tool {name!r} is missing required field 'input_schema'.",
            )
        if (schema is not None or schema_client_kind is not None) and (
            not isinstance(name, str) or not name
        ):
            raise HTTPException(
                status_code = 400,
                detail = "Client tool is missing required field 'name'.",
            )


def _append_to_system_message(messages: list[dict], addition: str) -> list[dict]:
    """Append text to the leading system/developer message, or prepend one."""
    if not addition:
        return messages
    copied = [dict(msg) for msg in messages]
    for msg in copied:
        if msg.get("role") not in ("system", "developer"):
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            msg["content"] = content.rstrip() + "\n\n" + addition
            return copied
    return [{"role": "system", "content": addition}, *copied]


@router.post("/chat/count_tokens")
async def chat_count_tokens(
    payload: ChatCountTokensRequest, current_subject: str = Depends(get_current_subject)
):
    """Count prompt tokens for OpenAI-form chat messages using the loaded tokenizer.

    Unlike the /v1 count endpoints this never auto-switches: ``model`` is informational. The
    caller is a background recount with no abort signal, so switching could drag the backend back
    to the model loaded when the count started, a reload the client's guards cannot undo."""
    # Admitted only while nothing generates, and stood down at the next checkpoint if that changes:
    # admission is not atomic with the work, and true mutual exclusion would put a lock in front of
    # generation startup, which is the cost this avoids. Refusing here also covers the second tab or
    # the script against /api that the client-side gate cannot. Deliberately coarse: _TrackedCancel
    # registers external-provider runs too, so those decline a count they could have served;
    # narrowing it means trusting a kind/model field to decide whether to work next to a decode, and
    # being wrong there costs inference time while over-refusing only costs a redraw.
    if active_generations.count() > 0:
        raise HTTPException(
            status_code = 503,
            detail = "Cannot count tokens while a generation is in progress.",
        )

    # /apply-template swaps each image for a short media marker, so refuse rather than undercount.
    if _request_has_image(payload):
        raise HTTPException(
            status_code = 503,
            detail = "Cannot count tokens for messages containing images.",
        )
    # Same for audio: the completion injects the recording, this cannot.
    if getattr(payload, "audio_base64", None):
        raise HTTPException(
            status_code = 503,
            detail = "Cannot count tokens for messages containing audio.",
        )

    llama_backend = get_llama_cpp_backend()
    if not llama_backend.is_loaded:
        raise HTTPException(
            status_code = 503,
            detail = _no_model_loaded_detail("No GGUF model loaded. Load a GGUF model first."),
        )

    # Same sanitization the GGUF chat path runs before generation. Route FIRST: the passthrough
    # does not merge adjacent user turns, so coalescing here would price a prompt it never sends
    # (two user turns split by an empty assistant sentinel, after a stopped response).
    _takes_passthrough = _takes_tool_passthrough(payload, llama_backend)
    openai_messages = _strip_provider_synthetic_tool_history(
        _drop_empty_assistant_sentinels([m.model_dump(exclude_none = True) for m in payload.messages])
    )
    if not _takes_passthrough:
        openai_messages = _coalesce_consecutive_user_turns(openai_messages)
    _system_prompt, _, _ = _extract_content_parts(payload.messages)
    openai_messages = _set_or_prepend_system_message(openai_messages, _system_prompt)

    # A PENDING turn (unanswered user message or tool result) is the one shape the tool loop
    # answers from exactly these messages, splicing in whatever build_rag_autoinject retrieves --
    # thousands of tokens this never sees, so the bar would claim room the generation lacks. Any
    # other shape ends on an assistant turn, where retrieval has no user message to run against.
    if (
        payload.rag_scope
        and openai_messages
        and openai_messages[-1].get("role") in ("user", "tool")
    ):
        raise HTTPException(
            status_code = 503,
            detail = "Cannot count tokens for a pending turn that would retrieve documents.",
        )

    # The passthrough is the only route that puts the caller's own catalog on the wire; every other
    # renders from the selection below, so a catalog surviving here would price schemas never sent.
    openai_tools = _passthrough_client_tools(payload) if _takes_passthrough else None
    # The CLI hard-override still applies: _effective_enable_tools resolves it into _tools_on.
    _client_disabled_tool_calls = getattr(payload, "tool_choice", None) == "none" and not (
        _explicit_studio_tool_loop_requested(payload) and llama_backend.supports_tools
    )
    # Schemas and the nudge are a large share of the prompt: price the completion's own selection.
    _tools_on = False if _client_disabled_tool_calls else _effective_enable_tools(payload)
    # Same rule the completion path resolves, so the two agree on whether MCP schemas are in the
    # prompt at all. MCP alone turns tools on there, hence the widened branch below.
    from state.tool_policy import get_tool_policy as _get_tool_policy_ct

    _mcp_on = (
        not _client_disabled_tool_calls
        and bool(getattr(payload, "mcp_enabled", False))
        and _get_tool_policy_ct() is not False
    )
    # Never discovery on a count: get_enabled_mcp_tools spawns stdio servers, writes cache and
    # cool-off state, and blocks a probe timeout on a server that is down. _mcp_allowed stays
    # False because it is the flag that reaches the network; schemas come from the cache that
    # path fills instead, and an incomplete view is declined rather than undercounted.
    _mcp_allowed = False
    _mcp_tools: list[dict] = []
    if _mcp_on and not _takes_passthrough and llama_backend.supports_tools:
        from core.inference.tools import cached_mcp_tools
        _mcp_tools, _mcp_complete = cached_mcp_tools()
        if not _mcp_complete:
            raise HTTPException(
                status_code = 503,
                detail = "Cannot count tokens until enabled MCP tools have been discovered.",
            )
    if not _takes_passthrough and (_tools_on or _mcp_on) and llama_backend.supports_tools:
        tools_to_use = await _select_request_tools(
            payload, tools_on = _tools_on, mcp_allowed = _mcp_allowed
        )
        # Appended in the position _select_request_tools would have used, so the order matches.
        tools_to_use = tools_to_use + _mcp_tools
        if tools_to_use:
            openai_tools = tools_to_use
            openai_messages = _append_to_system_message(
                openai_messages,
                _apply_rag_nudge(
                    _build_tool_action_nudge(
                        tools = tools_to_use,
                        model_name = _llama_public_model_id(llama_backend, payload.model),
                    ),
                    tools_to_use,
                    rag_scope = payload.rag_scope,
                ),
            )

            # The GGUF tool path strips leaked markup from replayed history before rendering,
            # so without the same strip the count prices text it removes.
            _count_auto_heal = (
                payload.auto_heal_tool_calls if payload.auto_heal_tool_calls is not None else True
            )
            _count_history_gate = _display_tool_name_gate(tools_to_use)
            openai_messages = [dict(msg) for msg in openai_messages]
            for _msg in openai_messages:
                if _msg.get("role") == "assistant" and isinstance(_msg.get("content"), str):
                    _msg["content"] = _strip_tool_xml_for_display(
                        _msg["content"],
                        auto_heal_tool_calls = _count_auto_heal,
                        enabled_tool_names = _count_history_gate,
                    ).strip()

    # llama-server falls back to the load-time --chat-template-kwargs per key a request omits,
    # so omitting these prices the template in whatever mode the model was LOADED in.
    _template_kwargs = llama_backend._request_reasoning_kwargs(
        payload.enable_thinking,
        payload.reasoning_effort,
        payload.preserve_thinking,
    )

    # Whose tokenizer this is, in the shape /api/inference/status publishes: another tab's load
    # moves it while the caller's own checkpoint guard sees no change, so report and let it drop.
    _tokenizer_model = _llama_status_checkpoint_id(llama_backend)

    # Re-checked immediately before the only work that reaches llama-server, because everything
    # between here and the entry check awaits, so a run can have started in the gap.
    if active_generations.count() > 0:
        raise HTTPException(
            status_code = 503,
            detail = "Cannot count tokens while a generation is in progress.",
        )

    from core.inference.llama_cpp import CountAborted

    try:
        count = await asyncio.to_thread(
            llama_backend.count_chat_tokens,
            openai_messages,
            None,
            openai_tools,
            strict = True,
            chat_template_kwargs = _template_kwargs,
            # Polled between /apply-template and /tokenize: admission and the work are separate
            # steps, so a run starting in between is caught here and the second round trip is not.
            should_abort = lambda: active_generations.count() > 0,
        )
    except CountAborted:
        raise HTTPException(
            status_code = 503,
            detail = "Cannot count tokens while a generation is in progress.",
        )
    except Exception:
        raise HTTPException(
            status_code = 503,
            detail = "Unable to count tokens with the loaded model tokenizer.",
        )
    # A load landing mid-count leaves the total attributable to neither model.
    if _llama_status_checkpoint_id(llama_backend) != _tokenizer_model:
        raise HTTPException(
            status_code = 503,
            detail = "The loaded model changed while counting tokens.",
        )
    return JSONResponse(content = {"input_tokens": int(count), "model": _tokenizer_model})


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
    # Reject malformed tools before the switch, like /messages, so an invalid
    # count request can't evict the loaded model.
    _validate_anthropic_client_tools(payload.tools)
    # Count with the requested model's tokenizer, like the sibling /messages.
    # Carry the vision guard too: an image count naming a text-only GGUF must not
    # evict a loaded vision model for a swap that can't serve the request.
    await _maybe_auto_switch_model(
        _switch_model_for_payload(payload),
        request,
        current_subject,
        require_vision = _anthropic_request_has_image(payload),
    )

    llama_backend = get_llama_cpp_backend()
    if not llama_backend.is_loaded:
        _status, _detail = await _no_model_loaded_error(
            "No GGUF model loaded. Load a GGUF model first.",
            _switch_model_for_payload(payload),
            request,
            status = 503,
        )
        raise HTTPException(status_code = _status, detail = _detail)

    # Same Anthropic → OpenAI translation as anthropic_messages: system is
    # folded into the messages list, so pass system=None to the counter.
    openai_messages = anthropic_messages_to_openai(
        [m.model_dump() for m in payload.messages],
        payload.system,
    )
    # Apply the same sanitization /messages does before generation, so the count
    # matches the prompt the real request would build (otherwise empty-assistant
    # sentinels / synthetic tool history inflate the count or hit the fallback).
    # Coalesce adjacent user turns left behind by dropping an empty / null assistant
    # turn, so a strict GGUF chat template does not 400 on non-alternating roles
    # (mirrors the GGUF chat path); a no-op for already-alternating histories.
    openai_messages = _coalesce_consecutive_user_turns(
        _strip_provider_synthetic_tool_history(_drop_empty_assistant_sentinels(openai_messages))
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

    # Default-off parity: with no automatic load possible and nothing loaded, 503
    # before any request-shape check, exactly as the pre-feature endpoint did. When
    # an automatic load can run (auto-switch or a standalone idle TTL), fall through
    # so validation runs before the reload hook gets a chance to restore the model.
    # Plain detail, not _no_model_loaded_error: that helper leaves this case unchanged.
    if not llama_backend.is_loaded and not _automatic_model_load_may_run():
        raise HTTPException(
            status_code = 503,
            detail = _no_model_loaded_detail("No GGUF model loaded. Load a GGUF model first."),
        )

    # max_tokens is a required field on the Anthropic Messages API; real Anthropic
    # returns a 400 invalid_request_error when it is omitted. Validate before
    # auto-switch so a rejected request never triggers a model load.
    if payload.max_tokens is None:
        raise HTTPException(
            status_code = 400,
            detail = anthropic_error_body(
                "max_tokens: field required",
                status = 400,
                err_type = "invalid_request_error",
            ),
        )

    # Reject malformed client tools before any model load (see helper), so an
    # invalid request never evicts the loaded model.
    _validate_anthropic_client_tools(payload.tools)

    # Mixing Anthropic server tools with custom client tools is unsupported (the
    # server-tool loop can't relay client functions back to the caller). Reject
    # before the switch too -- it depends only on the payload -- so an invalid
    # request never evicts the loaded model. Reused below for tool routing.
    requested_studio_tools = _anthropic_requested_studio_tools(payload.tools)
    _has_client_tool = any(
        (t if isinstance(t, dict) else t.model_dump()).get("input_schema") is not None
        or anthropic_schema_client_tool_kind(t) is not None
        for t in payload.tools or []
    )
    _explicit_server_tools = bool(requested_studio_tools) or (
        payload.enable_tools is True and _effective_enable_tools(payload) is not False
    )
    if _explicit_server_tools and _has_client_tool:
        raise HTTPException(
            status_code = 400,
            detail = (
                "Mixing Anthropic server tools (e.g. web_search_20250305) "
                "with custom client tools in a single request is not "
                "supported. Send them in separate requests."
            ),
        )

    # Reject an unsupported confirm-gated permission mode for Unsloth's own
    # ("server") Anthropic tools before the switch, mirroring the malformed- and
    # mixed-tool checks above. ask always wants a per-call pause this passthrough
    # cannot offer, so it 400s whenever server tools are selected. auto only needs
    # the gate for an unsafe call, so (like the omitted default) it runs for a
    # safe-only selection (web_search/RAG) and 400s when a gate-needing tool is
    # selected (local terminal/python, or render_html whose networked canvas
    # prompts and cannot be gated on this channel). Rejecting must happen before the
    # switch so an invalid request never evicts the resident model; it is
    # determined from the requested tools alone (backend tool support is only known
    # post-switch); an image request can never take the server-tool path, so it is
    # excluded as in the server_tools gate below. off/full and an explicit
    # confirm_tool_calls=False opt-out always pass.
    # A process-wide ``--enable-tools`` policy is only a default for ordinary
    # chat. It must not steal an explicit Anthropic client-tool catalog (Claude
    # Code's Write/Edit/Bash tools) and turn it into Unsloth's local tool loop.
    # An explicit per-request server-tool ask was rejected as mixed mode above.
    _enable_pre = False if _has_client_tool else _effective_enable_tools(payload)
    _server_tools_requested_pre = (
        _enable_pre or (_enable_pre is None and bool(requested_studio_tools))
    ) and not _anthropic_request_has_image(payload)
    if _server_tools_requested_pre:
        from core.inference.tools import ALL_TOOLS as _ALL_TOOLS_PRE

        _selected_pre = _select_anthropic_server_tools(
            _ALL_TOOLS_PRE, requested_studio_tools, payload.enabled_tools
        )
        _perm_mode_pre = getattr(payload, "permission_mode", None)
        _confirm_opt_out_pre = getattr(payload, "confirm_tool_calls", None) is False
        _gated_tool_selected_pre = any(
            tool["function"]["name"] not in _ANTHROPIC_UNPROMPTED_SAFE_TOOLS
            for tool in _selected_pre
        )
        # An explicit confirm_tool_calls=False opts out of the gate entirely (it
        # wins over the mode, mirroring _permission_mode_confirm and the GGUF path),
        # so it never rejects -- not even under ask.
        if not _confirm_opt_out_pre and (
            _perm_mode_pre == "ask"
            or (_perm_mode_pre in ("auto", None) and _gated_tool_selected_pre)
        ):
            raise HTTPException(
                status_code = 400,
                detail = anthropic_error_body(
                    "permission_mode 'ask' has no confirmation channel for Anthropic "
                    "Messages server tools, and 'auto' (or the omitted default) cannot "
                    "gate a local 'terminal'/'python' tool here; set 'off' or 'full'.",
                    status = 400,
                    err_type = "invalid_request_error",
                ),
            )

    # require_vision rejects a swap to a text-only target before it runs, so an
    # image request can't evict the resident vision model only to hit the vision
    # guard (_normalize_anthropic_openai_images) below after the load.
    await _maybe_auto_switch_model(
        _switch_model_for_payload(payload),
        request,
        current_subject,
        require_vision = _anthropic_request_has_image(payload),
    )
    if not llama_backend.is_loaded:
        _status, _detail = await _no_model_loaded_error(
            "No GGUF model loaded. Load a GGUF model first.",
            _switch_model_for_payload(payload),
            request,
            status = 503,
        )
        raise HTTPException(status_code = _status, detail = _detail)

    # Advertised repo id after an auto-switch load, else a clean public id, never
    # the local .gguf path (and a legacy raw path in payload.model is sanitized).
    model_name = _llama_public_model_id(llama_backend, payload.model)
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
    # Coalesce adjacent user turns left behind by dropping an empty / null assistant
    # turn, so a strict GGUF chat template does not 400 on non-alternating roles
    # (mirrors the GGUF chat path); a no-op for already-alternating histories.
    openai_messages = _coalesce_consecutive_user_turns(
        _strip_provider_synthetic_tool_history(_drop_empty_assistant_sentinels(openai_messages))
    )

    # Enforce vision guard + re-encode embedded images to PNG so the Anthropic
    # endpoint matches /v1/chat/completions.
    _has_image = _normalize_anthropic_openai_images(openai_messages, llama_backend.is_vision)

    # Fill omitted sampling fields with the per-model recommendation (or an operator
    # UNSLOTH_SAMPLING_* pin); an explicit client value wins unless the operator pinned it.
    # Anthropic sampling fields are Optional, so None already marks "client omitted".
    from utils.inference.inference_config import resolve_effective_sampling

    _anthropic_sampling = resolve_effective_sampling(
        getattr(llama_backend, "model_identifier", None) or model_name,
        {
            "temperature": payload.temperature,
            "top_p": payload.top_p,
            "top_k": payload.top_k,
            "min_p": payload.min_p,
            "repetition_penalty": payload.repetition_penalty,
            "presence_penalty": payload.presence_penalty,
        },
    )
    temperature = _anthropic_sampling["temperature"]
    top_p = _anthropic_sampling["top_p"]
    top_k = _anthropic_sampling["top_k"]
    min_p = _anthropic_sampling["min_p"]
    repetition_penalty = _anthropic_sampling["repetition_penalty"]
    presence_penalty = _anthropic_sampling["presence_penalty"]
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
    # the `not image_b64` gate in /v1/chat/completions. requested_studio_tools and
    # the mixed-mode rejection were computed before the switch above.
    openai_client_tools = [
        tool
        for tool in anthropic_tools_to_openai(payload.tools or [])
        if tool.get("function", {}).get("name") not in requested_studio_tools
    ]

    # An Anthropic server-tool declaration implies server-tool mode, but only
    # when tools aren't explicitly disabled (CLI --disable-tools or per-request
    # enable_tools=false). Explicit False always wins.
    _enable = False if _has_client_tool else _effective_enable_tools(payload)
    server_tools = (
        (_enable or (_enable is None and bool(requested_studio_tools)))
        and llama_backend.supports_tools
        and not _has_image
    )
    client_tools = (
        not server_tools
        and len(openai_client_tools) > 0
        and getattr(llama_backend, "supports_tool_passthrough", llama_backend.supports_tools)
    )

    # Anthropic tool_choice.disable_parallel_tool_use caps the response to a
    # single tool_use block. Computed here so BOTH the client-tool passthrough
    # and the server-tool path honor it.
    _disable_parallel = bool(
        isinstance(payload.tool_choice, dict)
        and payload.tool_choice.get("disable_parallel_tool_use")
    )

    monitor_id = None
    monitor_context_length = _monitor_context_length()
    request_state = getattr(request, "state", None)
    if not getattr(request_state, "skip_api_monitor", False):
        request_url = getattr(request, "url", None)
        monitor_id = api_monitor.start(
            endpoint = getattr(request_url, "path", "/v1/messages"),
            method = getattr(request, "method", "POST"),
            via_api_key = _request_used_api_key(request),
            model = model_name,
            prompt = _monitor_prompt_from_messages(openai_messages),
            context_length = monitor_context_length,
            subject = current_subject,
        )

    async def _monitored_anthropic(coro):
        try:
            response = await coro
        except asyncio.CancelledError:
            cancel_event.set()
            api_monitor.finish(monitor_id, "cancelled")
            raise
        except Exception as exc:
            api_monitor.fail(monitor_id, _friendly_error(exc))
            raise
        return _monitor_anthropic_response(
            response,
            monitor_id,
            monitor_context_length,
            cancel_event,
        )

    async def _tracked_anthropic_non_streaming(coro):
        """Register a non-streaming /v1/messages run with the swap gate.

        `stream` defaults to false, so this is the route's common shape, and all
        three helpers hold llama-server for the whole await. /unload runs no idle
        drain, so unregistered a swap tore the server down mid-request; only the
        streaming siblings registered. No cancel keys, unlike the streaming
        tool/plain siblings: the gate reaches a run through the registry, and
        keys would add a cancel surface to a public API.
        """
        _tracker = _TrackedCancel(cancel_event, model = model_name, kind = "messages")
        _tracker.__enter__()
        try:
            return await _monitored_anthropic(coro)
        finally:
            # _monitored_anthropic's bookkeeping can throw; a leaked entry 409s later swaps.
            _tracker.__exit__(None, None, None)

    # ── Admission control ─────────────────────────────────────
    # Bound concurrent llama-server generations to the backend's serving slots via a
    # FIFO queue keyed by base_url (shared with /v1/chat/completions, same slots).
    # Excess requests queue; a streaming waiter gets SSE keep-alives, the queue 429s
    # once full. Mirrors the OpenAI passthrough admission wiring. Streaming takes the
    # slot when the response is built and drops it when the body finishes or is
    # abandoned; the non-stream path holds it across the single awaited generation.
    _anthropic_admission_mode = "anthropic_stream" if payload.stream else "anthropic_nonstream"

    async def _admitted_anthropic_stream(
        orig_body,
        reservation,
        admission_config,
        stream_lease,
        prior_cleanup = None,
    ):
        lease = stream_lease
        stream_cancelled = False
        body_started = False
        wait_started_at = None
        try:
            if lease is None:
                wait_started_at = time.monotonic()
                _llama_admission_log(
                    "queued",
                    reservation,
                    request = request,
                    mode = _anthropic_admission_mode,
                )
                async for wait_item in _openai_admission_wait_stream_chunks(
                    reservation,
                    admission_config,
                    request = request,
                    cancel_event = cancel_event,
                ):
                    if isinstance(wait_item, str):
                        yield wait_item
                        continue
                    lease = wait_item
                    break
                _llama_admission_log(
                    "granted-after-wait",
                    reservation,
                    request = request,
                    mode = _anthropic_admission_mode,
                    wait_started_at = wait_started_at,
                )
            if lease is None:
                return
            body_started = True
            async for chunk in orig_body:
                yield chunk
        except asyncio.CancelledError:
            # Must reach the monitored generator as CancelledError, not aclose's
            # GeneratorExit, or its handler never finalizes the monitor entry.
            stream_cancelled = True
            raise
        except LlamaAdmissionTimeout as exc:
            api_monitor.fail(monitor_id, str(exc))
            _llama_admission_log(
                "timeout",
                reservation,
                request = request,
                mode = _anthropic_admission_mode,
                wait_started_at = wait_started_at,
                level = "warning",
            )
            yield build_anthropic_sse_event(
                "error",
                anthropic_error_body(str(exc), status = 503),
            )
        except LlamaAdmissionCancelled:
            _llama_admission_log(
                "cancelled-before-upstream",
                reservation,
                request = request,
                mode = _anthropic_admission_mode,
                wait_started_at = wait_started_at,
            )
            return
        finally:
            # Closing can raise (a raw body re-raises CancelledError after
            # teardown), and a slot lost that way never comes back: with no queue
            # timeout the pool just shrinks and later callers wait forever. Keep
            # the release in its own finally, as the /responses wiring does.
            try:
                if body_started:
                    await _close_openai_admitted_stream_iterator(
                        orig_body,
                        cancelled = stream_cancelled,
                    )
                else:
                    # Gave up while queued: the monitored body never ran, so nothing
                    # downstream finalizes the entry or exits the response's tracker.
                    api_monitor.finish(monitor_id, "cancelled")
                    await _release_unstarted_anthropic_stream(orig_body, prior_cleanup)
            finally:
                if lease is not None:
                    lease.release()
                else:
                    reservation.cancel()

    async def _admitted_anthropic(coro):
        try:
            reservation, admission_config = _openai_llama_admission_reserve(
                request = request, llama_backend = llama_backend
            )
        except LlamaAdmissionQueueFull as exc:
            coro.close()
            api_monitor.fail(monitor_id, str(exc))
            _llama_admission_log(
                "queue-full",
                snapshot = getattr(exc, "snapshot", None),
                request = request,
                mode = _anthropic_admission_mode,
                level = "warning",
            )
            raise _anthropic_admission_http_exception(exc, status_code = 429)
        except BaseException:
            # Reserving never awaited the generation, so close it rather than
            # leave an un-awaited coroutine behind.
            coro.close()
            raise

        if payload.stream:
            stream_lease = reservation.lease_nowait()
            # Set up the stream (token count + tracker enter) and surface a pre-response
            # cancel now, exactly as the un-admitted path did; the upstream generation is
            # deferred to body iteration, so the slot is only held while tokens flow.
            try:
                # Token counting calls llama-server, so a dead backend raises here
                # with the slot already taken. cancel() covers both cases: it
                # releases the lease if one was granted, else drops the waiter.
                monitored = await _monitored_anthropic(coro)
            except BaseException:
                reservation.cancel()
                raise
            orig_body = getattr(monitored, "body_iterator", None)
            if orig_body is None:
                reservation.cancel()
                return monitored

            # Replacing body_iterator would strand the response's own pre-start
            # hook (the passthrough uses one to exit its cancel tracker), so chain
            # to it instead of clobbering it.
            prior_cleanup = getattr(monitored, "_unstarted_cleanup", None)

            async def _unstarted_cleanup() -> None:
                # The body never ran, so nothing else closes out the monitor entry.
                api_monitor.finish(monitor_id, "cancelled")
                try:
                    await _release_unstarted_anthropic_stream(orig_body, prior_cleanup)
                finally:
                    # A BaseException here is swallowed upstream, so releasing
                    # outside the finally would shrink the pool silently.
                    reservation.cancel()

            monitored.body_iterator = _admitted_anthropic_stream(
                orig_body, reservation, admission_config, stream_lease, prior_cleanup
            )
            monitored._unstarted_cleanup = _unstarted_cleanup
            return monitored

        lease = None
        try:
            lease = await _wait_for_openai_admission_non_streaming(
                reservation,
                admission_config,
                request = request,
                cancel_event = cancel_event,
            )
            # Registered only once admitted: a queued request is not holding
            # llama-server, so it has no business blocking a swap.
            monitored = await _tracked_anthropic_non_streaming(coro)
            return monitored
        except LlamaAdmissionTimeout as exc:
            coro.close()
            api_monitor.fail(monitor_id, str(exc))
            raise _anthropic_admission_http_exception(exc, status_code = 503)
        except LlamaAdmissionCancelled as exc:
            coro.close()
            api_monitor.finish(monitor_id, "cancelled")
            raise _anthropic_admission_http_exception(exc, status_code = 499)
        except BaseException:
            # Cancelled while queued (shutdown, outer task cancel): the generation
            # coroutine was never awaited, so close it rather than leak it.
            if lease is None:
                coro.close()
                api_monitor.finish(monitor_id, "cancelled")
            raise
        finally:
            if lease is not None:
                lease.release()
            else:
                reservation.cancel()

    # ── Client-side pass-through path ─────────────────────────
    if client_tools:
        openai_tools = openai_client_tools

        if payload.stream:
            return await _admitted_anthropic(
                _anthropic_passthrough_stream(
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
                    auto_heal_tool_calls = payload.auto_heal_tool_calls,
                )
            )
        return await _admitted_anthropic(
            _anthropic_passthrough_non_streaming(
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
                auto_heal_tool_calls = payload.auto_heal_tool_calls,
                nudge_tool_calls = payload.nudge_tool_calls,
                request = request,
                cancel_event = cancel_event,
            )
        )

    if server_tools:
        # Bypass Permissions suppresses confirm, so both flags together is fine.
        if bool(getattr(payload, "confirm_tool_calls", False)) and not bool(
            getattr(payload, "bypass_permissions", False)
        ):
            api_monitor.fail(
                monitor_id,
                "confirm_tool_calls is not supported for Anthropic Messages server tools.",
            )
            raise HTTPException(
                status_code = 400,
                detail = anthropic_error_body(
                    "confirm_tool_calls is not supported for Anthropic Messages server tools.",
                    status = 400,
                    err_type = "invalid_request_error",
                ),
            )
        from core.inference.tools import ALL_TOOLS

        # ask/auto (and an omitted mode selecting a gate-needing terminal/python
        # tool) were already rejected before the auto-switch above, so an invalid
        # confirm-gated request never evicts the resident model; the selection
        # here just picks the tools for the actual server-tool loop.
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

        # Strip stale tool-call XML via the protected display helper (think rehearsal and [TOOL_CALLS]
        # prose survive), gated on enabled tool names so documented inactive examples are kept.
        _anthropic_history_gate = _display_tool_name_gate(openai_tools)
        for _msg in openai_messages:
            if _msg.get("role") == "assistant" and isinstance(_msg.get("content"), str):
                _msg["content"] = _strip_tool_xml_for_display(
                    _msg["content"],
                    auto_heal_tool_calls = True,
                    enabled_tool_names = _anthropic_history_gate,
                ).strip()

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
                nudge_tool_calls = payload.nudge_tool_calls,
                tool_call_timeout = 300,
                session_id = payload.session_id,
                thread_id = payload.thread_id,
                # Anthropic passthrough has no rag_scope field (RAG is local-only).
                rag_scope = getattr(payload, "rag_scope", None),
                disable_parallel_tool_use = _disable_parallel,
                bypass_permissions = bool(payload.bypass_permissions),
                permission_mode = getattr(payload, "permission_mode", None),
                promote_reasoning_only = False,
            )

        if payload.stream:
            return await _admitted_anthropic(
                _anthropic_tool_stream(
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
            )
        return await _admitted_anthropic(
            _anthropic_tool_non_streaming(
                _run_tool_gen,
                message_id,
                model_name,
                disable_parallel_tool_use = _disable_parallel,
                openai_tools = openai_tools,
            )
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
            promote_reasoning_only = False,
        )

    if payload.stream:
        return await _admitted_anthropic(
            _anthropic_plain_stream(
                request,
                cancel_event,
                _run_plain_gen,
                message_id,
                model_name,
                llama_backend = llama_backend,
                openai_messages = openai_messages,
            )
        )
    return await _admitted_anthropic(
        _anthropic_plain_non_streaming(
            _run_plain_gen,
            message_id,
            model_name,
        )
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

    # Gate the display strip on the declared tools: an inactive NAME[ARGS]{...} in a final
    # answer is prose and must survive in the delivered text.
    _display_names = _display_tool_name_gate(openai_tools)

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
        # The server-tool loop decodes on llama-server for its whole body, so without an entry a
        # non-forced /unload saw zero generations and tore the server down mid-response. Entered
        # inside the body generator so a response whose body never starts leaves nothing behind.
        # No thread_id: public API surface.
        _tracker = _TrackedCancel(cancel_event, model = model_name, kind = "messages")
        _tracker.__enter__()
        try:
            emitter = AnthropicStreamEmitter()
            for line in emitter.start(message_id, model_name, input_tokens = input_tokens):
                yield line

            captured_finish_reason = None
            # Response ends on a pending tool_use block rather than final text; a server tool
            # that keeps generating flips this back to False.
            ends_on_tool_use = False
            tool_blocks_emitted = 0
            drop_until_tool_end = False
            # Last drop-branch keepalive, seeded to stream start so a chatty tool busy past the
            # stall window still gets one though its events are dropped.
            _last_drop_keepalive = time.monotonic()

            gen = run_gen()
            _next_task = None
            # Watcher to cancel on disconnect: the in-loop poll fires only between events,
            # so a mid-prefill disconnect would hold the decode slot.
            disconnect_watcher = asyncio.create_task(
                _await_disconnect_then_cancel(request, cancel_event)
            )
            try:
                while True:
                    if cancel_event.is_set() or await request.is_disconnected():
                        cancel_event.set()
                        return
                    # Stall keepalive (see GGUF tool stream): silent backend segments must not
                    # leave the SSE stream idle past proxy timeouts.
                    _next_task = asyncio.create_task(asyncio.to_thread(next, gen, _sentinel))
                    while True:
                        _done_tasks, _ = await asyncio.wait(
                            {_next_task},
                            timeout = _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S,
                        )
                        if _done_tasks:
                            break
                        yield _OPENAI_PASSTHROUGH_SSE_KEEPALIVE
                    event = _next_task.result()
                    # Done; drop the reference so the finally-block drain no-ops.
                    _next_task = None
                    if event is _sentinel:
                        break
                    etype = event.get("type")
                    if etype == "heartbeat":
                        # Tool-wrapper heartbeat -> SSE keepalive, checked BEFORE the drop skip:
                        # a dropped tool still runs and suppresses the stall keepalive.
                        yield _OPENAI_PASSTHROUGH_SSE_KEEPALIVE
                        continue
                    if etype in ("tool_output", "tool_args"):
                        # No Anthropic Messages equivalent (the full call/result follow in tool_use /
                        # tool_result), so drop them. They suppress the stall keepalive, so emit a
                        # rate-limited one instead of going silent past the ~100s proxy cap.
                        _now = time.monotonic()
                        if _now - _last_drop_keepalive >= _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S:
                            _last_drop_keepalive = _now
                            yield _OPENAI_PASSTHROUGH_SSE_KEEPALIVE
                        continue
                    if drop_until_tool_end:
                        # disable_parallel_tool_use: skip every event until (and
                        # including) this dropped tool call's tool_end.
                        if etype == "tool_end":
                            drop_until_tool_end = False
                        continue
                    if etype == "metadata":
                        _fr = event.get("finish_reason")
                        if _fr is not None:
                            captured_finish_reason = _fr
                    # Strip leaked tool-call XML first, so a purely-tool-XML content event doesn't
                    # count as text. The protected helper keeps <think> rehearsal and balanced
                    # [TOOL_CALLS] trailing prose, which a raw sub corrupts.
                    if etype == "content":
                        event = dict(event)
                        event["text"] = _strip_tool_xml_for_display(
                            event["text"],
                            auto_heal_tool_calls = True,
                            enabled_tool_names = _display_names,
                        )
                    # disable_parallel_tool_use: keep only the first tool_use block, dropping
                    # later tool_start/tool_end pairs (by state, not id: ids may be empty).
                    if etype == "tool_start":
                        if disable_parallel_tool_use and tool_blocks_emitted >= 1:
                            drop_until_tool_end = True
                            continue
                        ends_on_tool_use = True
                    elif etype == "tool_end":
                        tool_blocks_emitted += 1
                        # Unsloth ran the tool server-side, so the response no longer ends on a pending
                        # client action; otherwise stop_reason "tool_use" tells the client to run it again.
                        ends_on_tool_use = False
                    elif etype == "content" and event.get("text"):
                        ends_on_tool_use = False
                    for line in emitter.feed(event):
                        yield line
            except Exception as e:
                logger.error("anthropic_messages stream error: %s", e)
                # force = True so an unclassified mid-stream failure emits an SSE error instead
                # of a message_stop that masks a truncated turn as a clean finish.
                _error_event = _anthropic_stream_error_event(e, force = True)
                if _error_event is not None:
                    yield _error_event
                    return
            finally:
                await _stop_local_disconnect_cancel_watcher(disconnect_watcher)
                # Drain a still-running next(gen) worker first, so a mid-prefill disconnect releases
                # its resources; closing first races into 'already executing'.
                await _drain_pending_next_task(_next_task, cancel_event)
                if gen is not None:
                    try:
                        await asyncio.to_thread(gen.close)
                    except (RuntimeError, ValueError):
                        pass

            stop_reason = openai_finish_to_anthropic_stop(
                captured_finish_reason, had_tool_calls = ends_on_tool_use
            )
            for line in emitter.finish(stop_reason = stop_reason, stop_sequence = None):
                yield line
        finally:
            _tracker.__exit__(None, None, None)

    return _sse_streaming_response(_stream())


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
        # Registered like the tool stream above: this default /v1/messages path decodes on
        # llama-server, so without an entry a non-forced /unload tore it down mid-response.
        _tracker = _TrackedCancel(cancel_event, model = model_name, kind = "messages")
        _tracker.__enter__()
        try:
            emitter = AnthropicStreamEmitter()
            for line in emitter.start(message_id, model_name, input_tokens = input_tokens):
                yield line

            captured_finish_reason = None

            gen = run_gen()
            _next_task = None
            # Watcher to cancel on disconnect: the in-loop poll fires only between chunks,
            # so a mid-prefill disconnect would hold the decode slot.
            disconnect_watcher = asyncio.create_task(
                _await_disconnect_then_cancel(request, cancel_event)
            )
            try:
                while True:
                    if cancel_event.is_set() or await request.is_disconnected():
                        cancel_event.set()
                        return
                    # Stall keepalive each window while next(gen) runs in a worker.
                    _next_task = asyncio.create_task(asyncio.to_thread(next, gen, _sentinel))
                    while True:
                        _done_tasks, _ = await asyncio.wait(
                            {_next_task},
                            timeout = _LOCAL_TOOL_STREAM_STALL_KEEPALIVE_S,
                        )
                        if _done_tasks:
                            break
                        yield _OPENAI_PASSTHROUGH_SSE_KEEPALIVE
                    cumulative = _next_task.result()
                    # Done; drop the reference so the finally-block drain no-ops.
                    _next_task = None
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
                # force = True so an unclassified mid-stream failure emits an SSE error instead
                # of a message_stop that masks a truncated turn as a clean finish.
                _error_event = _anthropic_stream_error_event(e, force = True)
                if _error_event is not None:
                    yield _error_event
                    return
            finally:
                await _stop_local_disconnect_cancel_watcher(disconnect_watcher)
                # Drain a still-running next(gen) worker first, so a mid-prefill disconnect releases
                # its resources; closing first races into 'already executing'.
                await _drain_pending_next_task(_next_task, cancel_event)
                if gen is not None:
                    try:
                        await asyncio.to_thread(gen.close)
                    except (RuntimeError, ValueError):
                        pass

            stop_reason = openai_finish_to_anthropic_stop(
                captured_finish_reason, had_tool_calls = False
            )
            for line in emitter.finish(stop_reason = stop_reason, stop_sequence = None):
                yield line
        finally:
            _tracker.__exit__(None, None, None)

    return _sse_streaming_response(_stream())


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


def _anthropic_message_json_response(
    message_id, model_name, content_blocks, stop_reason, usage
) -> Response:
    """Assemble the terminal Anthropic non-streaming JSON response shared by the
    tool / plain / passthrough paths."""
    return _model_json_response(
        AnthropicMessagesResponse(
            id = message_id,
            model = model_name,
            content = content_blocks,
            stop_reason = stop_reason,
            usage = AnthropicUsage(
                input_tokens = usage.get("prompt_tokens", 0),
                output_tokens = usage.get("completion_tokens", 0),
            ),
        )
    )


async def _anthropic_tool_non_streaming(
    run_gen,
    message_id,
    model_name,
    disable_parallel_tool_use = False,
    openai_tools = None,
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
    # Gate the display strip on the declared tools: an inactive NAME[ARGS]{...} in a final
    # answer is prose and must survive in the delivered text.
    _display_names = _display_tool_name_gate(openai_tools)
    # Pending client tool_use; cleared by tool_end (server execution) or
    # trailing text. See the stop_reason mapping below.
    ends_on_tool_use = False

    events = _collect_anthropic_events(run_gen)

    for event in events:
        etype = event.get("type", "")
        if etype == "content":
            # Strip leaked tool XML (protected helper keeps think rehearsal and trailing prose).
            clean = _strip_tool_xml_for_display(
                event["text"], auto_heal_tool_calls = True, enabled_tool_names = _display_names
            )
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

    return _anthropic_message_json_response(
        message_id, model_name, content_blocks, stop_reason, usage
    )


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

    return _anthropic_message_json_response(
        message_id, model_name, content_blocks, stop_reason, usage
    )


# =====================================================================
# Client-side tool pass-through (Anthropic-native tools field)
# =====================================================================


_JSON_SCHEMA_MAP_KEYWORDS = frozenset(
    {
        "$defs",
        "definitions",
        "dependentSchemas",
        "patternProperties",
        "properties",
    }
)
_JSON_SCHEMA_SINGLE_KEYWORDS = frozenset(
    {
        "additionalProperties",
        "contains",
        "contentSchema",
        "else",
        "if",
        "items",
        "not",
        "propertyNames",
        "then",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_JSON_SCHEMA_LIST_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_LLAMA_GRAMMAR_MAX_REPETITION = 2000
_JSON_SCHEMA_REPETITION_KEYWORDS = frozenset({"maxItems", "maxLength", "minItems", "minLength"})


def _llama_compatible_tool_schema(schema):
    """Return a llama.cpp-compatible copy of one JSON Schema node.

    JSON Schema ``pattern`` expressions match anywhere in a string, so an
    unanchored pattern is valid and cannot be made compatible by merely adding
    ``^`` and ``$`` without changing its meaning. llama.cpp's grammar converter
    currently rejects those patterns outright. Its grammar parser likewise
    rejects repetition bounds above 2000. Omit only those unsupported
    constraints from the local-backend copy; the agent retains and validates
    its original schema, while every compatible constraint still reaches
    llama.cpp.
    """
    if not isinstance(schema, dict):
        return schema

    compatible = dict(schema)
    pattern = compatible.get("pattern")
    if isinstance(pattern, str) and not (pattern.startswith("^") and pattern.endswith("$")):
        compatible.pop("pattern")
    # llama-grammar.cpp refuses repetition bounds above its sane-default
    # threshold. Dropping the local-backend constraint preserves every value
    # the client schema accepts; capping it would incorrectly reject otherwise
    # valid tool arguments.
    for keyword in _JSON_SCHEMA_REPETITION_KEYWORDS:
        bound = compatible.get(keyword)
        if (
            isinstance(bound, int)
            and not isinstance(bound, bool)
            and bound > _LLAMA_GRAMMAR_MAX_REPETITION
        ):
            compatible.pop(keyword)

    for keyword in _JSON_SCHEMA_MAP_KEYWORDS:
        children = compatible.get(keyword)
        if isinstance(children, dict):
            compatible[keyword] = {
                key: _llama_compatible_tool_schema(value) for key, value in children.items()
            }

    for keyword in _JSON_SCHEMA_SINGLE_KEYWORDS:
        child = compatible.get(keyword)
        if isinstance(child, dict):
            compatible[keyword] = _llama_compatible_tool_schema(child)

    for keyword in _JSON_SCHEMA_LIST_KEYWORDS:
        children = compatible.get(keyword)
        if isinstance(children, list):
            compatible[keyword] = [_llama_compatible_tool_schema(value) for value in children]

    return compatible


def _llama_compatible_tools(openai_tools):
    if not isinstance(openai_tools, list):
        return openai_tools

    compatible_tools = []
    for tool in openai_tools:
        if not isinstance(tool, dict):
            compatible_tools.append(tool)
            continue
        function = tool.get("function")
        parameters = function.get("parameters") if isinstance(function, dict) else None
        if not isinstance(parameters, dict):
            compatible_tools.append(tool)
            continue
        compatible_tools.append(
            {
                **tool,
                "function": {
                    **function,
                    "parameters": _llama_compatible_tool_schema(parameters),
                },
            }
        )
    return compatible_tools


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
    markup = None,
):
    from core.inference.chat_template_helpers import (
        neutralize_control_markup_in_messages,
        neutralize_tool_descriptions,
        reconciled_tool_choice,
    )

    # The one place to break markup: llama-server applies the template itself, and both
    # /v1/messages bodies come from here, never the OpenAI builder below (#7066).
    # *markup* is the loaded model's profile, so passthrough leaves another family's
    # marker alone exactly as generate_chat_completion does (#7066).
    _pt_markup = markup
    body = {
        "messages": neutralize_control_markup_in_messages(openai_messages, None, _pt_markup),
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": stream,
    }
    # Tested after the rewrite: an all-injected catalog drops to empty, and
    # "tools": [] would still advertise tool use.
    safe_tools = neutralize_tool_descriptions(openai_tools, None, _pt_markup)
    if safe_tools:
        body["tools"] = _llama_compatible_tools(safe_tools)
        # A mixed catalog keeps safe_tools non-empty while dropping the one tool the client
        # forced; forwarding that choice would name an unadvertised function and hand
        # llama-server back the raw markup. Fall back to "auto" to stay consistent (#7066).
        tool_choice = reconciled_tool_choice(tool_choice, openai_tools, safe_tools)
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
    if seed is not None:
        body["seed"] = seed
    if stream and stream_options is not None:
        body["stream_options"] = stream_options
    body["max_tokens"] = (
        max_tokens if max_tokens is not None else (backend_ctx or _DEFAULT_MAX_TOKENS_FLOOR)
    )
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


def _nudge_retry_messages(
    body,
    data,
    allowed_tools,
    markup = None,
):
    """The nudge retry's message list, re-neutralized like the enable-tools loop.

    The appended suffix is not sanitized text: the assistant turn replays the model's own
    failed output, and the user turn interpolates ``allowed_tools``, which ``heal_gate``
    derives from the RAW catalog on the /v1/messages path -- so a name dropped from
    ``tools`` for carrying markup would come straight back as prose the template renders
    as structure (#7066). Wrapping the whole concatenation rather than just the suffix is
    free: the rewrite is idempotent and returns unchanged messages as-is, so the already
    neutralized prefix stays byte-identical and llama-server still reuses the slot's KV
    cache, the entire point of appending instead of rebuilding."""
    from core.inference.chat_template_helpers import neutralize_control_markup_in_messages

    # Same profile the body was built with: sweeping the retry with the curated patterns
    # would rewrite a prefix the first attempt preserved, so the prefix would no longer be
    # byte-identical and the slot's KV cache would miss (#7066).
    return neutralize_control_markup_in_messages(
        [*body.get("messages", []), *nudge_messages(data, allowed_tools)], None, markup
    )


async def _anthropic_passthrough_retry_url(llama_backend, exc):
    """Fresh upstream URL after respawning a dead llama-server, else None.

    A crashed server relaunches on a NEW ephemeral port, so a passthrough still
    holding the old base_url keeps failing until the next load. Mirrors the
    respawn-and-retry in generate_chat_completion. None when an MTP+tensor crash
    already scheduled its own recovery, or when nothing needed respawning.
    """
    recover = getattr(llama_backend, "_maybe_recover_from_mtp_crash", None)
    if recover is not None and recover(exc):
        return None
    # Only the first caller gets True above; the rest must not respawn the same
    # MTP config underneath the fallback that is already reloading without it.
    if getattr(llama_backend, "_mtp_runtime_fallback_in_progress", False):
        return None
    respawn = getattr(llama_backend, "_respawn_if_dead", None)
    if respawn is None or not await asyncio.to_thread(respawn):
        return None
    logger.warning("llama-server was unreachable; respawned it and retrying the passthrough")
    return f"{llama_backend.base_url}/v1/chat/completions"


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
    auto_heal_tool_calls = None,
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
        markup = getattr(llama_backend, "markup_profile", None),
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
    # No thread_id: public API surface, but still registered so a reload cannot yank
    # llama-server out from under it. Built here, entered below inside _stream().
    _tracker = _TrackedCancel(
        cancel_event,
        cancel_id,
        session_id,
        message_id,
        model = model_name,
        kind = "messages",
    )

    async def _stream():
        # Entered inside the body, not eagerly: aclose() runs no body on a generator
        # that never started, so a client that drops first would leave the run
        # registered until restart, 409-ing every swap. Ahead of the first yield, so
        # the opening lines are covered as well.
        _tracker.__enter__()
        emitter = AnthropicPassthroughEmitter()
        # Promote text-form tool calls (declared client tools only) into tool_use blocks;
        # verbatim when healing is off or no tools. tool_choice is already OpenAI-shaped.
        # Sanitized catalog, not the caller's: a tool dropped for unsafe markup never reached
        # the prompt, so promoting it would hand the client an unadvertised tool_use (#7066).
        from core.inference.chat_template_helpers import neutralize_tool_descriptions

        _healing_tools = neutralize_tool_descriptions(
            openai_tools, None, getattr(llama_backend, "markup_profile", None)
        )
        # The reconciled choice the body carries, not the caller's: a dropped forced tool
        # was already sent as "auto", and gating on the stale name would intersect the safe
        # names with a removed one and disable healing outright. "none" survives
        # reconciliation, so it still forbids promotion (#7066).
        _allowed_tools = heal_gate(auto_heal_tool_calls, _healing_tools, body.get("tool_choice"))
        if _allowed_tools:
            emitter.enable_healing(
                _allowed_tools,
                _healing_tools,
                disable_parallel_tool_use = disable_parallel_tool_use,
            )
        # These yields sit outside the teardown try below, so a disconnect while
        # the opening lines are being sent would strand the tracker. __exit__ is
        # idempotent, so the normal path still exits once, down there.
        try:
            for line in emitter.start(message_id, model_name, input_tokens = input_tokens):
                yield line
        except BaseException:
            _tracker.__exit__(None, None, None)
            raise

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
            timeout = _llama_streaming_generation_timeout(),
            limits = httpx.Limits(max_keepalive_connections = 0),
            trust_env = False,
        )
        resp = None
        lines_iter = None
        cancel_watcher = None
        disconnect_watcher = None
        try:
            url = target_url
            try:
                req = client.build_request("POST", url, json = body, headers = {"Connection": "close"})
                first_token_deadline = time.monotonic() + _DEFAULT_FIRST_TOKEN_TIMEOUT_S
                resp = await _send_stream_with_preheader_cancel(
                    client, req, cancel_event, request = request
                )
            except httpx.ConnectError as exc:
                # Nothing has streamed yet, so a respawned server can be retried once
                # on its new port without duplicating output.
                url = await _anthropic_passthrough_retry_url(llama_backend, exc)
                if url is None:
                    raise
                req = client.build_request("POST", url, json = body, headers = {"Connection": "close"})
                first_token_deadline = time.monotonic() + _DEFAULT_FIRST_TOKEN_TIMEOUT_S
                resp = await _send_stream_with_preheader_cancel(
                    client, req, cancel_event, request = request
                )
            if resp is None:
                return

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
                        _friendly_upstream_error(_err_text),
                        status = resp.status_code,
                    ),
                )
                return

            # Watchers unblock aiter_lines() during prefill, before in-loop
            # cancel/disconnect checks can run.
            cancel_watcher = asyncio.create_task(_await_cancel_then_close(cancel_event, resp))
            disconnect_watcher = asyncio.create_task(
                _await_disconnect_then_close(request, resp, cancel_event)
            )
            lines_iter = resp.aiter_lines()
            async for raw_line in _aiter_llama_stream_items(
                lines_iter,
                cancel_event = cancel_event,
                request = request,
                first_token_deadline = first_token_deadline,
                response = resp,
            ):
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
        except Exception as e:
            if not cancel_event.is_set():
                logger.error("anthropic_messages passthrough stream error: %s", e)
                get_llama_cpp_backend()._maybe_recover_from_mtp_crash(e)
                event = _anthropic_stream_error_event(
                    e,
                    force = True,
                )
                if event is not None:
                    yield event
                return
        finally:
            # Same shape as the OpenAI passthrough: the tracker exits after the closes,
            # and the bounded teardown awaits cannot hold it indefinitely.
            try:
                await _aclose_stream_resources(
                    watchers = (cancel_watcher, disconnect_watcher),
                    iterator = lines_iter,
                    resp = resp,
                    client = client,
                )
            finally:
                _release_admission(tracker = _tracker)

        for line in emitter.finish():
            yield line

    # The tracker is entered eagerly above, but _stream()'s finally is what exits
    # it. Closing an async generator that never started is a no-op, so hand the
    # response a cleanup hook or a pre-start give-up leaks the registry entry.
    return _sse_streaming_response(
        _stream(),
        unstarted_cleanup = _tracked_cancel_unstarted_cleanup(_tracker),
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
    auto_heal_tool_calls = None,
    nudge_tool_calls = None,
    request: Optional[Request] = None,
    cancel_event = None,
):
    """Non-streaming client-side pass-through.

    Both POSTs run on a per-request client so a Stop or a forced swap can close
    it and interrupt them. The pooled ``nonstreaming_client()`` cannot be closed
    without disturbing unrelated calls, which left this path registered with the
    swap gate but deaf to the event it registered.
    """
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
        markup = getattr(llama_backend, "markup_profile", None),
    )

    _client = _cancelable_nonstreaming_client()
    _cancel_watcher = asyncio.create_task(
        _await_cancel_or_disconnect_then_close_client(
            cancel_event = cancel_event,
            request = request,
            client = _client,
        )
    )

    async def _post(payload_body):
        nonlocal target_url
        try:
            return await _client.post(
                target_url,
                json = payload_body,
                timeout = _llama_non_streaming_generation_timeout(),
            )
        except httpx.RequestError as exc:
            # The watcher closes the client to break a blocked POST, so a transport error
            # with the event set is the cancel, not a failure.
            if cancel_event is not None and cancel_event.is_set():
                raise asyncio.CancelledError()
            # Nothing was returned yet, so retry once against the respawned server's
            # new port; the nudge retry below then reuses the same fresh URL.
            retry_url = (
                await _anthropic_passthrough_retry_url(llama_backend, exc)
                if isinstance(exc, httpx.ConnectError)
                else None
            )
            if retry_url is None:
                raise
            target_url = retry_url
            return await _client.post(
                target_url,
                json = payload_body,
                timeout = _llama_non_streaming_generation_timeout(),
            )

    try:
        resp = await _post(body)

        if resp.status_code != 200:
            raise HTTPException(
                status_code = resp.status_code,
                detail = _friendly_upstream_error(resp.text[:500]),
            )

        data = resp.json()
        # tool_choice is already OpenAI-shaped. Sanitized as in the streaming path: with
        # nudging on, the retry would otherwise name a tool dropped from the prompt (#7066).
        from core.inference.chat_template_helpers import neutralize_tool_descriptions

        _healing_tools = neutralize_tool_descriptions(
            openai_tools, None, getattr(llama_backend, "markup_profile", None)
        )
        # The reconciled choice the body carries, not the caller's: a dropped forced tool
        # was already sent as "auto", and gating on the stale name would intersect the safe
        # names with a removed one and disable healing outright. "none" survives
        # reconciliation, so it still forbids promotion (#7066).
        _allowed_tools = heal_gate(auto_heal_tool_calls, _healing_tools, body.get("tool_choice"))

        # Opt-in single-retry nudge (mirrors the OpenAI passthrough): the tool call came out
        # unusable; re-ask with the prompt prefix intact so the KV cache is reused.
        if (
            _allowed_tools
            and nudge_enabled(nudge_tool_calls)
            and nudge_should_retry(data, _allowed_tools, _healing_tools)
        ):
            retry_body = {
                **body,
                "messages": _nudge_retry_messages(
                    body, data, _allowed_tools, getattr(llama_backend, "markup_profile", None)
                ),
            }
            try:
                retry_resp = await _post(retry_body)
                if retry_resp.status_code == 200:
                    retry_data = retry_resp.json()
                    if response_has_promotable_calls(retry_data, _allowed_tools, openai_tools):
                        data = retry_data
            except (httpx.RequestError, ValueError) as exc:
                logger.warning("tool-call nudge retry failed; keeping original: %s", exc)

        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")

        healing_active = bool(_allowed_tools)
        healed_events = (
            heal_openai_message_events(message, _allowed_tools, openai_tools)
            if healing_active
            else None
        )

        content_blocks = []
        tool_calls = []
        if healed_events:
            emitted_tool_uses = 0
            for kind, value in healed_events:
                if kind == "text":
                    text = str(value).strip()
                    if text:
                        content_blocks.append(AnthropicResponseTextBlock(text = text))
                    continue
                if disable_parallel_tool_use and emitted_tool_uses >= 1:
                    continue
                fn = value.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(value)
                emitted_tool_uses += 1
                content_blocks.append(
                    AnthropicResponseToolUseBlock(
                        id = anthropic_tool_use_id(value.get("id")),
                        name = fn.get("name", ""),
                        input = args,
                    )
                )
        else:
            text = message.get("content") or ""
            if text:
                # Keep unpromoted bytes when healing is active; legacy stripping is only for opted-out
                # or no-client-tool requests. The protected helper preserves <think> rehearsal and
                # balanced [TOOL_CALLS] prose, gated on the declared tools so an inactive
                # NAME[ARGS]{...} example is kept.
                if not healing_active:
                    text = _strip_tool_xml_for_display(
                        text,
                        auto_heal_tool_calls = True,
                        enabled_tool_names = _display_tool_name_gate(openai_tools),
                    )
                text = text.strip()
                if text:
                    content_blocks.append(AnthropicResponseTextBlock(text = text))

            tool_calls = message.get("tool_calls") or []
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

        stop_reason = openai_finish_to_anthropic_stop(
            finish_reason, had_tool_calls = bool(tool_calls)
        )

        usage = data.get("usage") or {}
        return _anthropic_message_json_response(
            message_id, model_name, content_blocks, stop_reason, usage
        )
    finally:
        await _stop_local_disconnect_cancel_watcher(_cancel_watcher)
        try:
            await _client.aclose()
        except Exception:
            pass


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


def _merge_user_content(a: Any, b: Any) -> Any:
    """Join two user ``content`` values: strings with a blank line, else as concatenated parts."""
    if isinstance(a, str) and isinstance(b, str):
        if not a:
            return b
        if not b:
            return a
        return a + "\n\n" + b

    def _parts(c: Any) -> list:
        if c is None:
            return []
        if isinstance(c, str):
            return [{"type": "text", "text": c}] if c else []
        if isinstance(c, list):
            return list(c)
        return [{"type": "text", "text": str(c)}]

    return _parts(a) + _parts(b)


def _coalesce_consecutive_user_turns(messages: list[dict]) -> list[dict]:
    """Merge adjacent user turns so the GGUF history stays alternating.

    Dropping an empty assistant turn (0-token reply or Stop-button sentinel) can
    leave two user turns in a row, which makes strict templates (Gemma 3, ...)
    raise "Conversation roles must alternate" -> llama-server 400. Only user turns
    merge (assistant/tool turns may carry tool_calls/tool_call_id); multimodal
    parts are preserved; no-op for already-alternating histories.
    """
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "user" and out and out[-1].get("role") == "user":
            prev = dict(out[-1])
            prev["content"] = _merge_user_content(prev.get("content"), m.get("content"))
            out[-1] = prev
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


def _splice_image_into_last_user(messages: list[dict], image_part: dict) -> None:
    """Splice an image content part into the last user message, in place.

    String content becomes a text part plus the image; an existing content-part
    list gets the image appended; any other shape is replaced by the lone image.
    With no user message present, a new user turn carrying the image is appended."""
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


def _openai_messages_for_passthrough(payload) -> list[dict]:
    """Build OpenAI-format message dicts for the /v1/chat/completions
    passthrough path.

    ``payload.messages`` are dumped through Pydantic (dropping unset optional
    fields), so they're already standard OpenAI format -- including
    ``role="tool"`` tool-result messages and assistant messages carrying
    structured ``tool_calls``. Content-parts images already in the list are
    left untouched.

    When a client uses Unsloth's legacy ``image_base64`` top-level field, the
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
        raw = base64.b64decode(payload.image_base64)
        png_b64 = _image_bytes_to_png_b64(raw)
    except Exception:
        raise HTTPException(
            status_code = 400,
            detail = "Failed to process image.",
        )

    data_url = f"data:image/png;base64,{png_b64}"
    image_part = {"type": "image_url", "image_url": {"url": data_url}}

    _splice_image_into_last_user(messages, image_part)

    return messages


def _flatten_content_parts_for_local_template(messages: list[dict]) -> list[dict]:
    """Flatten OpenAI content-part lists to plain strings.

    Local text templates take string content and raise on part lists (e.g. a
    remote ``image_url`` that leaves ``image is None``): keep the text parts,
    drop the rest, like the plain non-GGUF path. GGUF keeps the parts."""
    out = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            ]
            msg = {**msg, "content": "\n".join(text_parts) if text_parts else ""}
        out.append(msg)
    return out


def _structured_tool_history_for_local_template(messages: list[dict]) -> list[dict]:
    """Deserialize assistant ``tool_calls[].function.arguments`` JSON strings to
    mappings for local templating.

    Clients send prior-turn arguments as JSON strings, but local templates take
    mappings (some raise on strings). Only the internal messages copy is
    rewritten; the HTTP response stays OpenAI-shaped and unparseable strings
    are left untouched."""
    out = []
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            new_calls = []
            for tc in tool_calls:
                fn = tc.get("function") if isinstance(tc, dict) else None
                args = fn.get("arguments") if isinstance(fn, dict) else None
                if isinstance(args, str):
                    try:
                        parsed = json.loads(args)
                    except ValueError:
                        parsed = None
                    if isinstance(parsed, dict):
                        tc = {**tc, "function": {**fn, "arguments": parsed}}
                new_calls.append(tc)
            msg = {**msg, "tool_calls": new_calls}
        out.append(msg)
    return out


def _openai_messages_for_gguf_chat(payload, is_vision: bool) -> tuple[list[dict], bool]:
    """Build llama-server messages for the standard GGUF chat path.

    llama-server accepts OpenAI multimodal content parts directly. Preserve all
    per-turn ``image_url`` parts so multi-image chat history keeps each image
    attached to its original turn.
    """
    # Coalesce only on the GGUF chat path (strict Jinja template); the tool path
    # reuses this via _set_or_prepend_system_message. Passthrough forwards verbatim.
    messages = _coalesce_consecutive_user_turns(
        _strip_provider_synthetic_tool_history(
            _drop_empty_assistant_sentinels(
                [m.model_dump(exclude_none = True) for m in payload.messages]
            )
        )
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
        _splice_image_into_last_user(messages, image_part)
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

    Only known OpenAI / llama-server fields are forwarded, so Unsloth-specific
    extensions (``enable_tools``, ``enabled_tools``, ``session_id``, ...) never
    leak to the backend.
    """
    messages = _openai_messages_for_passthrough(payload)
    system_prompt, _, _ = _extract_content_parts(payload.messages)
    messages = _set_or_prepend_system_message(messages, system_prompt)
    # Markup is broken in _build_passthrough_payload, shared with both /v1/messages (#7066).
    tool_choice = payload.tool_choice if payload.tool_choice is not None else "auto"
    tools = _passthrough_client_tools(payload)
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
        tools,
        payload.temperature,
        payload.top_p,
        payload.top_k,
        # Honor max_completion_tokens on the tools/response_format passthrough too.
        _effective_openai_max_tokens(payload),
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
        markup = getattr(llama_backend, "markup_profile", None),
    )


async def _openai_passthrough_stream(
    request,
    cancel_event,
    llama_backend,
    payload,
    model_name,
    completion_id,
    monitor_id: Optional[str] = None,
):
    _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
    _tracker = _TrackedCancel.for_payload(cancel_event, payload, *_cancel_keys)
    _tracker.__enter__()
    try:
        reservation, admission_config = _openai_llama_admission_reserve(
            request = request,
            llama_backend = llama_backend,
        )
    except LlamaAdmissionQueueFull as exc:
        _tracker.__exit__(None, None, None)
        _llama_admission_log(
            "queue-full",
            snapshot = exc.snapshot,
            request = request,
            mode = "chat_passthrough_stream",
            completion_id = completion_id,
            level = "warning",
        )
        api_monitor.fail(monitor_id, str(exc))
        raise _openai_admission_http_exception(exc, status_code = 429)

    lease = reservation.lease_nowait()
    if lease is not None:
        try:
            await _raise_if_openai_admission_cancelled(
                reservation,
                request = request,
                cancel_event = cancel_event,
            )
        except asyncio.CancelledError:
            api_monitor.finish(monitor_id, "cancelled")
            lease.release()
            _tracker.__exit__(None, None, None)
            raise
        except LlamaAdmissionCancelled as exc:
            lease.release()
            _tracker.__exit__(None, None, None)
            api_monitor.finish(monitor_id, "cancelled")
            raise HTTPException(
                status_code = 499,
                detail = _openai_admission_error_body(exc, status_code = 499),
            )
        return await _openai_passthrough_stream_admitted(
            request,
            cancel_event,
            llama_backend,
            payload,
            model_name,
            completion_id,
            monitor_id = monitor_id,
            admission_lease = lease,
            tracker = _tracker,
        )

    admission_wait_started_at = time.monotonic()
    _llama_admission_log(
        "queued",
        reservation,
        request = request,
        mode = "chat_passthrough_stream",
        completion_id = completion_id,
        level = "debug",
    )

    async def _queued_stream():
        admitted_started = False
        admitted_body_owns_cleanup = False
        admitted_response = None
        admitted_body_cancelled = False
        try:
            async for wait_item in _openai_admission_wait_stream_chunks(
                reservation,
                admission_config,
                request = request,
                cancel_event = cancel_event,
            ):
                if isinstance(wait_item, str):
                    yield wait_item
                    continue
                _llama_admission_log(
                    "granted-after-wait",
                    reservation,
                    request = request,
                    mode = "chat_passthrough_stream",
                    wait_started_at = admission_wait_started_at,
                    completion_id = completion_id,
                    level = "debug",
                )
                await _raise_if_openai_admission_cancelled(
                    reservation,
                    request = request,
                    cancel_event = cancel_event,
                )
                admitted_response = await _openai_passthrough_stream_admitted(
                    request,
                    cancel_event,
                    llama_backend,
                    payload,
                    model_name,
                    completion_id,
                    monitor_id = monitor_id,
                    admission_lease = wait_item,
                    tracker = _tracker,
                )
                admitted_started = True
                iterator = admitted_response.body_iterator
                admitted_body_owns_cleanup = True
                try:
                    async for chunk in iterator:
                        yield chunk
                except asyncio.CancelledError:
                    admitted_body_cancelled = True
                    raise
                finally:
                    await _close_openai_admitted_stream_iterator(
                        iterator,
                        cancelled = admitted_body_cancelled,
                    )
                    if not admitted_body_owns_cleanup:
                        cleanup = getattr(admitted_response, "_unstarted_cleanup", None)
                        if cleanup is not None:
                            await cleanup()
                return
        except LlamaAdmissionTimeout as exc:
            _llama_admission_log(
                "timeout",
                reservation,
                request = request,
                mode = "chat_passthrough_stream",
                wait_started_at = admission_wait_started_at,
                completion_id = completion_id,
                level = "warning",
            )
            api_monitor.fail(monitor_id, str(exc))
            yield _openai_stream_error_sse(_openai_admission_error_body(exc, status_code = 503))
        except LlamaAdmissionCancelled:
            _llama_admission_log(
                "cancelled-before-upstream",
                reservation,
                request = request,
                mode = "chat_passthrough_stream",
                wait_started_at = admission_wait_started_at,
                completion_id = completion_id,
                level = "debug",
            )
            api_monitor.finish(monitor_id, "cancelled")
            return
        except asyncio.CancelledError:
            api_monitor.finish(monitor_id, "cancelled")
            raise
        except HTTPException as exc:
            status_code = getattr(exc, "status_code", 500) or 500
            detail = exc.detail
            error = (
                detail
                if isinstance(detail, dict) and "error" in detail
                else openai_error_body(str(detail), status = status_code)
            )
            api_monitor.fail(monitor_id, str(detail))
            yield _openai_stream_error_sse(error)
        finally:
            if not admitted_started:
                api_monitor.finish(monitor_id, "cancelled")
                reservation.cancel()
                _tracker.__exit__(None, None, None)

    async def _queued_unstarted_cleanup() -> None:
        api_monitor.finish(monitor_id, "cancelled")
        reservation.cancel()
        _tracker.__exit__(None, None, None)

    return _SameTaskStreamingResponse(
        _queued_stream(),
        media_type = "text/event-stream",
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "close",
            "X-Accel-Buffering": "no",
        },
        unstarted_cleanup = _queued_unstarted_cleanup,
    )


async def _openai_passthrough_stream_admitted(
    request,
    cancel_event,
    llama_backend,
    payload,
    model_name,
    completion_id,
    monitor_id: Optional[str] = None,
    *,
    admission_lease: LlamaAdmissionLease,
    tracker,
):
    """Streaming client-side pass-through after Unsloth granted an upstream slot.

    Forwards the client's OpenAI function-calling request to llama-server and
    relays the SSE stream back with minimal normalization (reasoning-only
    deltas gain ``content: ""``; errors and missing terminal markers get a
    closing ``[DONE]``), preserving llama-server's native response ``id``,
    ``finish_reason`` (including ``"tool_calls"``), ``delta.tool_calls``, and
    any client-requested trailing ``usage`` chunk so the client sees a
    standard OpenAI response.

    Reasoning/tool-call splitting is delegated to llama-server (``--jinja
    --reasoning-format auto``), so ``delta.content`` carries no raw markup and is
    deliberately not re-parsed locally, unlike the ``/completion`` paths.
    """
    _tracker = tracker
    target_url = f"{llama_backend.base_url}/v1/chat/completions"
    upstream_headers = _openai_passthrough_upstream_headers(llama_backend = llama_backend)

    client = None
    resp = None
    send_task: Optional[asyncio.Task[Optional[httpx.Response]]] = None

    async def _aclose_send_task(task: Optional[asyncio.Task[Optional[httpx.Response]]]) -> None:
        if task is None:
            return
        if not task.done():
            task.cancel()
        # Bounded: the send polls Request.is_disconnected() before dispatch, which can
        # swallow cancel(). Abandoning it is safe because the caller closes the per-request
        # client right after, tearing down whatever response it later produces. #7617
        done, _pending = await asyncio.wait({task}, timeout = _TEARDOWN_TASK_STOP_TIMEOUT_S)
        if not done:
            task.add_done_callback(_discard_task_outcome)
            return
        try:
            task_resp = task.result()
            if task_resp is not None:
                try:
                    await task_resp.aclose()
                except Exception:
                    pass
        except (asyncio.CancelledError, Exception):
            pass

    # Keep tracker cleanup paired if pre-header dispatch is cancelled.
    try:
        body = _build_openai_passthrough_body(
            payload, backend_ctx = llama_backend.context_length, llama_backend = llama_backend
        )
        # Text-form tool calls from small models get promoted to structured calls on
        # the way back (declared client tools only); requests without tools or with
        # auto_heal_tool_calls=false keep the unhealed relay. tool_choice constrains
        # the allowlist ("none" disables, a forced function narrows to it).
        _allowed_tools = heal_gate(
            payload.auto_heal_tool_calls, body.get("tools"), body.get("tool_choice")
        )

        # Keep the pre-header window short so accepted SSE clients receive
        # immediate headers in the common timeout-reduced stall.
        client = httpx.AsyncClient(
            timeout = _llama_streaming_generation_timeout(),
            limits = httpx.Limits(max_keepalive_connections = 0),
            trust_env = False,
        )
        _truncate_budget = (
            _OVERFLOW_TRUNCATE_MAX_RETRIES if _overflow_truncation_requested(payload) else 0
        )

        while True:
            try:
                req = client.build_request("POST", target_url, json = body, headers = upstream_headers)
                first_token_deadline = time.monotonic() + _DEFAULT_FIRST_TOKEN_TIMEOUT_S
                send_task = asyncio.create_task(
                    _send_stream_with_preheader_cancel(
                        client,
                        req,
                        cancel_event,
                        request = request,
                        mark_cancel_on_cancel = False,
                    )
                )
                done, _ = await asyncio.wait(
                    {send_task},
                    timeout = _OPENAI_PASSTHROUGH_PREHEADER_STATUS_WINDOW_S,
                    return_when = asyncio.FIRST_COMPLETED,
                )
                if send_task not in done:
                    break

                # Dispatch returned quickly enough to preserve pre-header status.
                resp = await send_task
                send_task = None
            except httpx.RequestError as e:
                # llama-server subprocess crashed / starting / unreachable.
                logger.error("openai passthrough stream: upstream unreachable: %s", e)
                api_monitor.fail(monitor_id, _friendly_error(e))
                # Nested so a cancel inside _aclose_send_task's wait cannot skip the closes.
                # The outer handler releases too, but _release_admission is idempotent.
                try:
                    await _aclose_send_task(send_task)
                finally:
                    try:
                        await _aclose_stream_resources(resp = resp, client = client)
                    finally:
                        _release_admission(admission_lease, _tracker)
                raise HTTPException(
                    status_code = 502,
                    detail = _friendly_error(e),
                )
            if resp is None and send_task is not None and not send_task.done():
                break
            if resp is None:
                if cancel_event is not None:
                    cancel_event.set()
                api_monitor.finish(monitor_id, "cancelled")
                try:
                    await _aclose_send_task(send_task)
                finally:
                    try:
                        await _aclose_stream_resources(client = client)
                    finally:
                        _release_admission(admission_lease, _tracker)
                return _SameTaskStreamingResponse(
                    iter(()),
                    media_type = "text/event-stream",
                    headers = {
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
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
            resp = None
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
            api_monitor.fail(monitor_id, err_text[:500])
            raise _openai_passthrough_error(upstream_status, err_text)

        # Keep tracker cleanup paired if pre-header dispatch is cancelled after we
        # have already committed headers.
        async def _stream():
            # Same httpx lifecycle pattern as _anthropic_passthrough_stream:
            # save resp.aiter_lines() so the finally block can aclose() it on
            # our task. See that function for full rationale.
            lines_iter = None
            # Watchers unblock aiter_lines() during prefill, before in-loop
            # cancel/disconnect checks can run.
            cancel_watcher = None
            disconnect_watcher = None

            nonlocal resp, send_task, first_token_deadline, _truncate_budget
            nonlocal client
            monitor_done = False
            saw_finish_reason = False
            saw_done = False
            saw_stream_error = False
            saw_stream_item = False
            saw_tool_call_delta = False
            terminal_seen = False
            last_chunk_id = completion_id
            last_chunk_model = model_name
            last_chunk_created = int(time.time())
            healer = (
                StreamToolCallHealer(_allowed_tools, body.get("tools")) if _allowed_tools else None
            )
            healed_call_index = 0

            def _synthetic_finish_line() -> str:
                healed = healer is not None and healer.healed
                finish_reason = "tool_calls" if (saw_tool_call_delta or healed) else "stop"
                chunk = ChatCompletionChunk(
                    id = last_chunk_id,
                    created = last_chunk_created,
                    model = last_chunk_model,
                    choices = [
                        ChunkChoice(
                            delta = ChoiceDelta(),
                            finish_reason = finish_reason,
                        )
                    ],
                )
                return f"data: {chunk.model_dump_json(exclude_none = True)}"

            def _healer_sse_lines(events) -> list:
                # Serialize healer events as chunks matching the upstream stream's
                # id/model/created so clients see one coherent completion.
                nonlocal healed_call_index
                lines = []
                for kind, value in events:
                    if kind == "text":
                        if not value:
                            continue
                        delta = {"content": value}
                    else:
                        # parallel_tool_calls=false caps healed calls too (the SSE
                        # line cap only sees structured upstream deltas).
                        if payload.parallel_tool_calls is False and healed_call_index >= 1:
                            continue
                        delta = {
                            "tool_calls": [
                                {
                                    "index": healed_call_index,
                                    "id": value["id"],
                                    "type": "function",
                                    "function": value["function"],
                                }
                            ]
                        }
                        healed_call_index += 1
                    chunk = {
                        "id": last_chunk_id,
                        "object": "chat.completion.chunk",
                        "created": last_chunk_created,
                        "model": last_chunk_model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                    lines.append("data: " + json.dumps(chunk, ensure_ascii = False))
                return lines

            stall_timeout_s = _openai_compat_stream_stall_timeout()

            def _terminal_read_timeout_s() -> Optional[float]:
                if terminal_seen:
                    return _OPENAI_PASSTHROUGH_TERMINAL_GRACE_S
                return stall_timeout_s

            def _heal_transform(chunk_data: dict, raw_line: str) -> list:
                """SSE lines to emit in place of one upstream line (healing on)."""
                choices = chunk_data.get("choices")
                if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)):
                    return [raw_line]
                choice = choices[0]
                delta = choice.get("delta")
                delta = delta if isinstance(delta, dict) else {}
                if delta.get("tool_calls"):
                    # Structured call streamed: grammar mode worked. Flush any held
                    # text (it preceded the call) and relay verbatim from here on.
                    lines = _healer_sse_lines(healer.structured_tool_call_seen())
                    if healed_call_index:
                        if payload.parallel_tool_calls is False:
                            # A healed call already consumed the single allowed
                            # slot; the upstream SSE cap keeps native index 0, so
                            # drop the native call here or the client gets two.
                            del delta["tool_calls"]
                            if delta or choice.get("finish_reason") or chunk_data.get("usage"):
                                lines.append("data: " + json.dumps(chunk_data, ensure_ascii = False))
                            return lines
                        # A healed call already went out on index 0..n-1; OpenAI
                        # clients merge tool-call deltas by index, so shift the
                        # native calls into the next indexes or they would merge
                        # into the healed call.
                        for tc in delta["tool_calls"]:
                            if isinstance(tc, dict) and isinstance(tc.get("index"), int):
                                tc["index"] += healed_call_index
                        return lines + ["data: " + json.dumps(chunk_data, ensure_ascii = False)]
                    return lines + [raw_line]
                content = delta.get("content")
                finish = choice.get("finish_reason")
                if not isinstance(content, str) or not content:
                    if not finish:
                        return [raw_line]
                    # Finish chunk: last-chance heal of the residue, and rewrite a
                    # "stop" into "tool_calls" when text-form calls were promoted.
                    lines = _healer_sse_lines(healer.finalize())
                    if healer.healed and finish == "stop":
                        choice["finish_reason"] = "tool_calls"
                        return lines + ["data: " + json.dumps(chunk_data, ensure_ascii = False)]
                    return lines + [raw_line]
                events = healer.feed(content)
                if finish:
                    events += healer.finalize()
                if not finish and events == [("text", content)]:
                    # Nothing held or promoted: the healer passed the chunk
                    # through whole, so keep the verbatim upstream bytes.
                    return [raw_line]
                del delta["content"]
                prefix_lines = []
                if delta:
                    prefix_chunk = {k: v for k, v in chunk_data.items() if k != "usage"}
                    prefix_choice = dict(choice)
                    prefix_choice["delta"] = dict(delta)
                    prefix_choice["finish_reason"] = None
                    prefix_chunk["choices"] = [prefix_choice]
                    prefix_lines.append("data: " + json.dumps(prefix_chunk, ensure_ascii = False))
                    delta.clear()
                lines = prefix_lines + _healer_sse_lines(events)
                if delta or finish or chunk_data.get("usage"):
                    if healer.healed and finish == "stop":
                        choice["finish_reason"] = "tool_calls"
                    lines.append("data: " + json.dumps(chunk_data, ensure_ascii = False))
                return lines

            try:
                while True:
                    if send_task is not None:
                        last_keepalive_at = time.monotonic()
                        while not send_task.done():
                            # Wake often enough that _preheader_cancelled keeps
                            # cancel/disconnect latency sub-second during prefill;
                            # keepalives still pace off last_keepalive_at.
                            wait_timeout = min(
                                _STREAM_DISCONNECT_POLL_TIMEOUT_S,
                                _OPENAI_PASSTHROUGH_PENDING_RESPONSE_KEEPALIVE_S,
                            )
                            done, _ = await asyncio.wait(
                                {send_task},
                                timeout = wait_timeout,
                                return_when = asyncio.FIRST_COMPLETED,
                            )
                            if send_task in done:
                                break
                            if await _preheader_cancelled(cancel_event, request):
                                api_monitor.finish(monitor_id, "cancelled")
                                return
                            # The downstream SSE response is already committed;
                            # keep strict clients and proxies from treating a long
                            # llama-server prefill/header wait as a dead stream.
                            now = time.monotonic()
                            if (
                                now - last_keepalive_at
                                >= _OPENAI_PASSTHROUGH_PENDING_RESPONSE_KEEPALIVE_S
                            ):
                                last_keepalive_at = now
                                yield _OPENAI_PASSTHROUGH_SSE_KEEPALIVE
                        if resp is None:
                            try:
                                resp = send_task.result()
                            except httpx.RequestError as e:
                                logger.error(
                                    "openai passthrough stream: upstream unreachable: %s", e
                                )
                                api_monitor.fail(monitor_id, _friendly_error(e))
                                yield _openai_stream_error_sse(_openai_stream_error_chunk(e))
                                return
                            send_task = None

                    if resp is None:
                        api_monitor.finish(monitor_id, "cancelled")
                        return
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
                    resp = None
                    if (
                        _truncate_budget > 0
                        and _classify_llama_generation_error(Exception(err_text))
                        and _apply_overflow_truncation(body, err_text)
                    ):
                        _truncate_budget -= 1
                        req = client.build_request(
                            "POST", target_url, json = body, headers = upstream_headers
                        )
                        first_token_deadline = time.monotonic() + _DEFAULT_FIRST_TOKEN_TIMEOUT_S
                        send_task = asyncio.create_task(
                            _send_stream_with_preheader_cancel(
                                client,
                                req,
                                cancel_event,
                                request = request,
                                mark_cancel_on_cancel = False,
                            )
                        )
                        continue

                    upstream_error = _openai_passthrough_error(upstream_status, err_text)
                    error_payload = (
                        upstream_error.detail
                        if isinstance(upstream_error.detail, dict)
                        else openai_error_body(
                            str(upstream_error.detail),
                            status = upstream_status,
                        )
                    )
                    api_monitor.fail(monitor_id, err_text[:500])
                    yield _openai_stream_error_sse(error_payload)
                    return

                cancel_watcher = asyncio.create_task(_await_cancel_then_close(cancel_event, resp))
                disconnect_watcher = asyncio.create_task(
                    _await_disconnect_then_close(request, resp, cancel_event)
                )
                lines_iter = resp.aiter_lines()
                async for raw_line in _aiter_llama_stream_items(
                    lines_iter,
                    cancel_event = cancel_event,
                    request = request,
                    first_token_deadline = first_token_deadline,
                    response = resp,
                    post_first_item_read_timeout_s = _terminal_read_timeout_s,
                ):
                    if not raw_line:
                        continue
                    if not raw_line.startswith("data:"):
                        continue
                    saw_stream_item = True
                    data_text = raw_line[5:].strip()
                    if data_text == "[DONE]":
                        saw_done = True
                        # Upstream ended without a finish chunk: heal the residue
                        # first so the synthetic finish sees healer.healed.
                        if healer is not None and not saw_stream_error:
                            for held_line in _healer_sse_lines(healer.finalize()):
                                _monitor_openai_sse_line(
                                    monitor_id, held_line, llama_backend.context_length
                                )
                                yield held_line + "\n\n"
                        if (
                            not saw_finish_reason
                            and not saw_stream_error
                            and not cancel_event.is_set()
                        ):
                            finish_line = _synthetic_finish_line()
                            _monitor_openai_sse_line(
                                monitor_id,
                                finish_line,
                                llama_backend.context_length,
                            )
                            yield finish_line + "\n\n"
                            saw_finish_reason = True
                        _monitor_openai_sse_line(
                            monitor_id,
                            raw_line,
                            llama_backend.context_length,
                        )
                        yield raw_line + "\n\n"
                        monitor_done = True
                        break
                    raw_line = _normalize_openai_passthrough_sse_line(
                        raw_line,
                        cap_parallel_tool_calls = payload.parallel_tool_calls is False,
                    )
                    data_text = raw_line[5:].strip()
                    try:
                        chunk_data = json.loads(data_text)
                    except json.JSONDecodeError:
                        chunk_data = None
                    if isinstance(chunk_data, dict):
                        if isinstance(chunk_data.get("id"), str):
                            last_chunk_id = chunk_data["id"]
                        if isinstance(chunk_data.get("model"), str):
                            last_chunk_model = chunk_data["model"]
                        if isinstance(chunk_data.get("created"), int):
                            last_chunk_created = chunk_data["created"]
                        choices = chunk_data.get("choices")
                        if isinstance(choices, list) and choices:
                            choice = choices[0]
                            if isinstance(choice, dict):
                                if choice.get("finish_reason"):
                                    saw_finish_reason = True
                                delta = choice.get("delta")
                                if isinstance(delta, dict) and delta.get("tool_calls"):
                                    saw_tool_call_delta = True
                        # Detect an error chunk independently of API monitoring
                        # (skip_api_monitor returns early), else the synthetic
                        # finish would fire after a failed stream.
                        if _monitor_openai_error_message(chunk_data):
                            saw_stream_error = True
                    # With healing active, a content-bearing line may be replaced by
                    # held/promoted chunks; otherwise the single (already
                    # normalized) line relays unchanged (monitored exactly as
                    # emitted either way).
                    if (
                        healer is not None
                        and not healer.dormant
                        and isinstance(chunk_data, dict)
                        and not saw_stream_error
                    ):
                        out_lines = _heal_transform(chunk_data, raw_line)
                    else:
                        out_lines = [raw_line]
                    # If a trailing usage-only chunk (include_usage) arrives before
                    # any finish chunk, emit the synthetic finish first so the order
                    # stays finish -> usage -> [DONE], matching the other streams.
                    if (
                        isinstance(chunk_data, dict)
                        and chunk_data.get("usage")
                        and not (
                            isinstance(chunk_data.get("choices"), list) and chunk_data["choices"]
                        )
                        and not saw_finish_reason
                        and not saw_stream_error
                        and not cancel_event.is_set()
                    ):
                        if healer is not None:
                            # Residue must precede the finish it may upgrade.
                            held = _healer_sse_lines(healer.finalize())
                            for held_line in held:
                                _monitor_openai_sse_line(
                                    monitor_id, held_line, llama_backend.context_length
                                )
                                yield held_line + "\n\n"
                        finish_line = _synthetic_finish_line()
                        _monitor_openai_sse_line(
                            monitor_id, finish_line, llama_backend.context_length
                        )
                        yield finish_line + "\n\n"
                        saw_finish_reason = True
                    for out_line in out_lines:
                        monitor_event = _monitor_openai_sse_line(
                            monitor_id,
                            out_line,
                            llama_backend.context_length,
                        )
                        if monitor_event == "error":
                            saw_stream_error = True
                        # Relay to preserve llama-server's native id,
                        # finish_reason, delta.tool_calls, and usage chunks.
                        yield out_line + "\n\n"
                        if monitor_event == "done":
                            monitor_done = True
                            break
                        terminal_state = (
                            _openai_passthrough_terminal_state_from_data(chunk_data)
                            if out_line is raw_line
                            else _openai_passthrough_sse_line_terminal_state(out_line)
                        )
                        if terminal_state == "usage" or (
                            terminal_state == "finish" and not _wants_stream_usage(payload)
                        ):
                            done_line = _SSE_DONE_LINE
                            _monitor_openai_sse_line(
                                monitor_id,
                                done_line,
                                llama_backend.context_length,
                            )
                            yield done_line + "\n\n"
                            saw_done = True
                            monitor_done = True
                            break
                        if terminal_state == "finish":
                            terminal_seen = True
                    if monitor_done:
                        break
                if not saw_done and not saw_stream_error and not cancel_event.is_set():
                    # Synthesize a finish chunk only if one was not already
                    # emitted (e.g. before a trailing usage-only chunk), but
                    # always close with [DONE] whenever the upstream omitted it,
                    # so the stream ends on the [DONE] sentinel either way.
                    if healer is not None:
                        for held_line in _healer_sse_lines(healer.finalize()):
                            _monitor_openai_sse_line(
                                monitor_id, held_line, llama_backend.context_length
                            )
                            yield held_line + "\n\n"
                    if not saw_finish_reason:
                        finish_line = _synthetic_finish_line()
                        _monitor_openai_sse_line(
                            monitor_id,
                            finish_line,
                            llama_backend.context_length,
                        )
                        yield finish_line + "\n\n"
                    done_line = _SSE_DONE_LINE
                    _monitor_openai_sse_line(
                        monitor_id,
                        done_line,
                        llama_backend.context_length,
                    )
                    yield done_line + "\n\n"
                    monitor_done = True
                if not monitor_done:
                    api_monitor.finish(
                        monitor_id,
                        "cancelled" if cancel_event.is_set() else "completed",
                    )
            except asyncio.CancelledError:
                api_monitor.finish(monitor_id, "cancelled")
                raise
            except httpx.ReadTimeout as e:
                if terminal_seen and not saw_stream_error and not cancel_event.is_set():
                    done_line = _SSE_DONE_LINE
                    _monitor_openai_sse_line(
                        monitor_id,
                        done_line,
                        llama_backend.context_length,
                    )
                    yield done_line + "\n\n"
                    api_monitor.finish(monitor_id)
                    return
                if cancel_event.is_set():
                    api_monitor.finish(monitor_id, "cancelled")
                    return
                logger.error(
                    "openai passthrough stream %s: %s",
                    "stalled mid-response" if saw_stream_item else "timeout",
                    e,
                )
                api_monitor.fail(monitor_id, _friendly_error(e))
                get_llama_cpp_backend()._maybe_recover_from_mtp_crash(e)
                err = _openai_stream_error_chunk(e)
                yield _openai_stream_error_sse(err)
            except (httpx.RemoteProtocolError, httpx.ReadError, httpx.CloseError) as e:
                # Watcher closed resp on cancel. Emit nothing extra; the client
                # initiated the cancel or already disconnected.
                if not cancel_event.is_set():
                    api_monitor.fail(monitor_id, "Stream interrupted")
                    get_llama_cpp_backend()._maybe_recover_from_mtp_crash(e)
                    raise
                api_monitor.finish(monitor_id, "cancelled")
            except HTTPException as exc:
                status_code = getattr(exc, "status_code", 500) or 500
                detail = exc.detail
                error_payload = (
                    detail
                    if isinstance(detail, dict) and "error" in detail
                    else openai_error_body(str(detail), status = status_code)
                )
                api_monitor.fail(monitor_id, str(detail))
                yield _openai_stream_error_sse(error_payload)
            except Exception as e:
                if cancel_event.is_set():
                    api_monitor.finish(monitor_id, "cancelled")
                    return
                # 200 headers already flushed; errors must go in the SSE body.
                logger.error("openai passthrough stream error: %s", e)
                api_monitor.fail(monitor_id, _friendly_error(e))
                get_llama_cpp_backend()._maybe_recover_from_mtp_crash(e)
                err = _openai_stream_error_chunk(e)
                yield _openai_stream_error_sse(err)
            finally:
                # Close the upstream stream first: on disconnect llama-server keeps decoding
                # until resp is closed, so releasing the slot earlier admits a second request
                # past --parallel. Safe to hold the slot across these closes because every
                # task await in them is bounded, and the aclose() calls do not block on
                # HTTP/1.1 to a local llama-server. #7617
                try:
                    await _aclose_send_task(send_task)
                finally:
                    try:
                        await _aclose_stream_resources(
                            watchers = (cancel_watcher, disconnect_watcher),
                            iterator = lines_iter,
                            resp = resp,
                            client = client,
                        )
                    finally:
                        _release_admission(admission_lease, _tracker)

        async def _unstarted_cleanup() -> None:
            # Client disconnected before the body stream started, so _stream()'s
            # finally never ran. Release the eagerly-opened upstream resp/client
            # and the cancel-registry entry here; the watchers and line iterator
            # are created inside _stream(), so there is nothing else to close.
            try:
                await _aclose_send_task(send_task)
            finally:
                try:
                    await _aclose_stream_resources(resp = resp, client = client)
                finally:
                    _release_admission(admission_lease, _tracker)

        return _SameTaskStreamingResponse(
            _stream(),
            media_type = "text/event-stream",
            headers = {
                "Cache-Control": "no-cache",
                "Connection": "close",
                "X-Accel-Buffering": "no",
            },
            unstarted_cleanup = _unstarted_cleanup,
        )
    except BaseException as exc:
        if isinstance(exc, asyncio.CancelledError):
            if cancel_event is not None:
                cancel_event.set()
            api_monitor.finish(monitor_id, "cancelled")
        else:
            detail = exc.detail if isinstance(exc, HTTPException) else _friendly_error(exc)
            api_monitor.fail(monitor_id, str(detail))
        try:
            await _aclose_send_task(send_task)
        finally:
            try:
                await _aclose_stream_resources(resp = resp, client = client)
            finally:
                _release_admission(admission_lease, _tracker)
        raise


async def _openai_passthrough_non_streaming(
    llama_backend,
    payload,
    model_name,
    monitor_id: Optional[str] = None,
    *,
    request: Optional[Request] = None,
    cancel_event = None,
):
    """Non-streaming pass-through guarded by local llama-server admission."""
    try:
        reservation, admission_config = _openai_llama_admission_reserve(
            request = request,
            llama_backend = llama_backend,
        )
    except LlamaAdmissionQueueFull as exc:
        _llama_admission_log(
            "queue-full",
            snapshot = exc.snapshot,
            request = request,
            mode = "chat_passthrough_nonstream",
            level = "warning",
        )
        api_monitor.fail(monitor_id, str(exc))
        raise _openai_admission_http_exception(exc, status_code = 429)

    lease = None
    admission_wait_started_at = None
    try:
        if reservation.lease_nowait() is None:
            admission_wait_started_at = time.monotonic()
            _llama_admission_log(
                "queued",
                reservation,
                request = request,
                mode = "chat_passthrough_nonstream",
                level = "debug",
            )
        lease = await _wait_for_openai_admission_non_streaming(
            reservation,
            admission_config,
            request = request,
            cancel_event = cancel_event,
        )
        if admission_wait_started_at is not None:
            _llama_admission_log(
                "granted-after-wait",
                reservation,
                request = request,
                mode = "chat_passthrough_nonstream",
                wait_started_at = admission_wait_started_at,
                level = "debug",
            )
        await _raise_if_openai_admission_cancelled(
            reservation,
            request = request,
            cancel_event = cancel_event,
        )
        return await _openai_passthrough_non_streaming_upstream(
            llama_backend,
            payload,
            model_name,
            monitor_id = monitor_id,
            request = request,
            cancel_event = cancel_event,
        )
    except LlamaAdmissionTimeout as exc:
        _llama_admission_log(
            "timeout",
            reservation,
            request = request,
            mode = "chat_passthrough_nonstream",
            wait_started_at = admission_wait_started_at,
            level = "warning",
        )
        api_monitor.fail(monitor_id, str(exc))
        raise _openai_admission_http_exception(exc, status_code = 503)
    except LlamaAdmissionCancelled as exc:
        _llama_admission_log(
            "cancelled-before-upstream",
            reservation,
            request = request,
            mode = "chat_passthrough_nonstream",
            wait_started_at = admission_wait_started_at,
            level = "debug",
        )
        api_monitor.finish(monitor_id, "cancelled")
        raise HTTPException(
            status_code = 499,
            detail = _openai_admission_error_body(exc, status_code = 499),
        )
    except asyncio.CancelledError:
        api_monitor.finish(monitor_id, "cancelled")
        reservation.cancel()
        raise
    finally:
        if lease is not None:
            lease.release()


async def _openai_passthrough_non_streaming_upstream(
    llama_backend,
    payload,
    model_name,
    monitor_id: Optional[str] = None,
    *,
    request: Optional[Request] = None,
    cancel_event = None,
):
    """Non-streaming client-side pass-through for /v1/chat/completions.

    Returns llama-server's JSON response verbatim so the client sees the native
    response ``id``, ``finish_reason`` (including ``"tool_calls"``), structured
    ``tool_calls``, and accurate ``usage`` token counts.
    """
    target_url = f"{llama_backend.base_url}/v1/chat/completions"
    upstream_headers = _openai_passthrough_upstream_headers(llama_backend = llama_backend)
    body = _build_openai_passthrough_body(
        payload, backend_ctx = llama_backend.context_length, llama_backend = llama_backend
    )
    body["stream"] = False
    body.pop("stream_options", None)

    _truncate_budget = (
        _OVERFLOW_TRUNCATE_MAX_RETRIES if _overflow_truncation_requested(payload) else 0
    )

    async def _post(body_to_send):
        if cancel_event is None and request is None:
            return await nonstreaming_client().post(
                target_url,
                json = body_to_send,
                headers = upstream_headers,
                timeout = _llama_non_streaming_generation_timeout(),
            )

        if cancel_event is None:
            cancel = threading.Event()
        else:
            cancel = cancel_event
        client = _cancelable_nonstreaming_client()
        watcher = asyncio.create_task(
            _await_cancel_or_disconnect_then_close_client(
                cancel_event = cancel,
                request = request,
                client = client,
            )
        )
        try:
            try:
                response = await client.post(
                    target_url,
                    json = body_to_send,
                    headers = upstream_headers,
                    timeout = _llama_non_streaming_generation_timeout(),
                )
            except httpx.RequestError:
                if cancel.is_set():
                    raise asyncio.CancelledError()
                raise
            if cancel.is_set():
                raise asyncio.CancelledError()
            return response
        finally:
            # Bounded: the watcher polls Request.is_disconnected(), which can swallow
            # cancel(). The client it owns is closed below either way. #7617
            try:
                await _stop_local_disconnect_cancel_watcher(watcher)
            except (asyncio.CancelledError, Exception):
                pass
            try:
                await client.aclose()
            except Exception:
                pass

    while True:
        try:
            resp = await _post(body)
        except asyncio.CancelledError:
            api_monitor.finish(monitor_id, "cancelled")
            raise
        except httpx.RequestError as e:
            # llama-server subprocess crashed / starting / unreachable. Surface the
            # same friendly message the sync chat path emits so operators don't see
            # a bare 500 with no diagnostic.
            logger.error("openai passthrough non-streaming: upstream unreachable: %s", e)
            api_monitor.fail(monitor_id, _friendly_error(e))
            get_llama_cpp_backend()._maybe_recover_from_mtp_crash(e)
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
        api_monitor.fail(monitor_id, resp.text[:500])
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
    _allowed_tools = heal_gate(
        payload.auto_heal_tool_calls, body.get("tools"), body.get("tool_choice")
    )

    try:
        data = resp.json()
    except Exception as exc:
        # Non-JSON / unparseable upstream body: relay verbatim as before.
        logger.warning(
            "openai passthrough non-streaming: response not JSON, relaying raw: %s",
            exc,
        )
        api_monitor.finish(monitor_id)
        return Response(content = resp.content, media_type = "application/json")

    # Opt-in single-retry nudge: the model clearly tried to call a tool (signal
    # present) but nothing parseable/declared came out, so re-ask once with the
    # original prompt prefix intact (llama-server reuses the slot's KV cache)
    # plus a two-message nudge suffix. The retry replaces the original response
    # only when it actually yields a usable call.
    if (
        _allowed_tools
        and nudge_enabled(payload.nudge_tool_calls)
        and nudge_should_retry(data, _allowed_tools, body.get("tools"))
    ):
        retry_body = {
            **body,
            "messages": _nudge_retry_messages(
                body, data, _allowed_tools, getattr(llama_backend, "markup_profile", None)
            ),
        }
        try:
            retry_resp = await _post(retry_body)
            if retry_resp.status_code == 200:
                retry_data = retry_resp.json()
                if response_has_promotable_calls(retry_data, _allowed_tools, body.get("tools")):
                    resp, data = retry_resp, retry_data
        except asyncio.CancelledError:
            api_monitor.finish(monitor_id, "cancelled")
            raise
        except (httpx.RequestError, ValueError) as exc:
            logger.warning("tool-call nudge retry failed; keeping original: %s", exc)

    changed = False
    for choice in data.get("choices", []):
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message")
        if not isinstance(msg, dict):
            continue

        # Small models emit tool calls as text instead of structured tool_calls;
        # promote them (declared client tools only) so the agent sees a real call.
        # Truncation wins over the upgrade (same rule as the streaming and
        # Anthropic paths): a call cut off at max_tokens keeps
        # finish_reason="length" so the client knows the arguments may be
        # incomplete, while the healed call itself stays attached.
        if _allowed_tools and heal_openai_message(msg, _allowed_tools, body.get("tools")):
            if choice.get("finish_reason") == "stop":
                choice["finish_reason"] = "tool_calls"
            changed = True

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

    _monitor_openai_chunk(monitor_id, data, llama_backend.context_length)
    api_monitor.finish(monitor_id)
    if not changed:
        # Nothing mutated: relay the upstream bytes verbatim, skipping a
        # redundant parse + re-serialize round-trip.
        return Response(content = resp.content, media_type = "application/json")
    return JSONResponse(content = data)


# ──────────────────────────────────────────────────────────────────────────
# Diffusion (local text-to-image). Studio-only routes (studio_router is not mounted under /v1); the backend is in-process and
# synchronous, so blocking calls are offloaded with asyncio.to_thread. Single error boundary: the backend raises, we map to HTTP.
# ──────────────────────────────────────────────────────────────────────────


def _diffusion_training_active() -> bool:
    """Whether a diffusion (SDXL) LoRA job is running. Best-effort so a load is never
    blocked just because the training service could not be imported/read."""
    try:
        from core.training.diffusion_training_service import get_diffusion_training_service
        return get_diffusion_training_service().is_active()
    except Exception:  # noqa: BLE001
        return False


@contextmanager
def _diffusion_training_admission():
    """Hold the diffusion trainer's GPU-admission interlock for this load's registration.

    The guards below only cover the instant they run. A load then selects its engine, acquires
    the arbiter and registers with the backend, and a ``/train/diffusion/start`` reserving inside
    that window frees residents this load has not registered yet, so the trainer comes up beside
    a brand-new pipeline. Registering the admission under the same lock ``reserve()`` takes makes
    the two mutually exclusive: this raises (409) once a start is reserved, and a start raises
    while an admission is open.

    Fails open on an import error, like the guards it complements. Covers the DIFFUSION trainer
    only; the LLM trainer admits loads that fit beside it, which is a different contract."""
    try:
        from core.training.diffusion_training_service import get_diffusion_training_service
        service = get_diffusion_training_service()
    except Exception:  # noqa: BLE001 -- unknowable state never blocks a load
        yield
        return
    with service.gpu_load_admission():
        yield


def _guard_diffusion_load_against_training() -> None:
    """Refuse loading an image model while a training run is active. Unlike chat,
    a diffusion pipeline's VRAM can't be cheaply estimated before the load, so the
    load is refused outright rather than fit-checked. No-op when training is
    inactive or its state can't be read. Raises HTTP 409."""
    from core.training import get_training_backend

    try:
        llm_active = get_training_backend().is_training_active()
    except Exception as e:
        # The two probes are independent: an unreadable LLM backend must not disable the diffusion interlock below.
        logger.warning("Could not check training state for image-load guard: %s", e)
        llm_active = False
    # An SDXL LoRA trainer runs in its own subprocess on the same GPU, so an image load must be refused while one is active.
    if not llm_active and not _diffusion_training_active():
        return
    raise HTTPException(
        status_code = 409,
        detail = (
            "Can't load an image model while training is running: the diffusion "
            "pipeline would compete with the training run for GPU memory. Training "
            "was left untouched. Try again after training finishes."
        ),
    )


@studio_router.post("/images/download-plan", response_model = DiffusionDownloadPlanResponse)
async def diffusion_download_plan(
    request: DiffusionLoadRequest, current_subject: str = Depends(get_current_subject)
):
    """The repos + files this pick needs, so the frontend can stage them through the Hub
    download manager (one mechanism, one panel) instead of the loader downloading inline.

    Validates the same way /images/load does, so an unloadable pick fails here rather than
    after a multi-GB download."""
    from core.inference.diffusion import (
        get_diffusion_backend,
        resolve_local_single_file,
        resolve_model_kind,
    )
    from core.inference.diffusion_engine_router import predict_engine
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP
    from utils.native_path_leases import redact_native_paths

    backend = get_diffusion_backend()
    try:
        kind = resolve_model_kind(request.gguf_filename, request.model_kind)
        # Same bare-single-file-directory reinterpretation as the load route, so the plan describes the load that will actually run.
        if kind == "pipeline" and not request.gguf_filename:
            sole = await asyncio.to_thread(resolve_local_single_file, request.model_path)
            if sole is not None:
                request.gguf_filename = sole
                kind = resolve_model_kind(sole)
        fam = await asyncio.to_thread(
            backend.validate_load_request,
            request.model_path,
            gguf_filename = request.gguf_filename,
            family_override = request.family_override,
            model_kind = kind,
            base_repo = request.base_repo,
        )
        # Plan for the engine /images/load will pick, not diffusers unconditionally: a GGUF on a GPU-less host routes to native
        # sd.cpp, which reads different files. predict_engine applies the policy without activating anything.
        planner = backend
        if fam is not None and predict_engine(fam, model_kind = kind) == ENGINE_SD_CPP:
            from core.inference.sd_cpp_backend import get_sd_cpp_backend
            planner = get_sd_cpp_backend()
        plan = await asyncio.to_thread(
            planner.download_plan,
            request.model_path,
            gguf_filename = request.gguf_filename,
            base_repo = request.base_repo,
            family_override = request.family_override,
            model_kind = kind,
            hf_token = request.hf_token,
            transformer_quant = request.transformer_quant,
            # An fp8 encoder request loads a hosted pre-cast checkpoint, so the plan must stage that file instead of the dense encoder shards.
            text_encoder_quant = request.text_encoder_quant,
            speed_mode = request.speed_mode,
            # The dense-quant prefetch decision also reads the memory policy, prequant path and adapter selection, so the plan must see the same values the load will.
            memory_mode = request.memory_mode,
            cpu_offload = request.cpu_offload,
            transformer_prequant_path = request.transformer_prequant_path,
            loras = request.loras,
        )
        return DiffusionDownloadPlanResponse(**plan)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code = 400, detail = redact_native_paths(str(exc)))


@studio_router.post("/images/load", response_model = DiffusionStatusResponse)
async def load_diffusion_model(
    request: DiffusionLoadRequest, current_subject: str = Depends(get_current_subject)
):
    from core.inference.diffusion import (
        get_diffusion_backend,
        resolve_local_single_file,
        resolve_model_kind,
    )
    from core.inference.diffusion_device import resolve_diffusion_device_target
    from core.inference.diffusion_engine_router import (
        active_engine_name,
        annotate_status,
        begin_load_on,
        engine_for,
        predict_engine,
        select_and_activate_engine,
    )
    from core.inference.gpu_arbiter import acquire_for, release, DIFFUSION
    from utils.native_path_leases import redact_native_paths

    backend = get_diffusion_backend()
    try:
        # Resolve the load kind once (gguf / single_file / pipeline) so validation, engine selection and the load agree. A bad kind raises here, so a 400.
        kind = resolve_model_kind(request.gguf_filename, request.model_kind)
        # A local On-Device pick can be a bare single-file .safetensors directory; if it holds exactly one checkpoint, reinterpret it as a single_file load so all three paths agree.
        if kind == "pipeline" and not request.gguf_filename:
            sole = await asyncio.to_thread(resolve_local_single_file, request.model_path)
            if sole is not None:
                request.gguf_filename = sole
                kind = resolve_model_kind(sole)
        # Validate cheaply BEFORE touching the GPU: an unloadable pick must not evict a working chat model and then 400. The family also drives engine selection.
        fam = await asyncio.to_thread(
            backend.validate_load_request,
            request.model_path,
            gguf_filename = request.gguf_filename,
            family_override = request.family_override,
            model_kind = kind,
            base_repo = request.base_repo,
        )
        # Refuse while training is running: a multi-GB pipeline would compete with the training subprocess for VRAM.
        _guard_diffusion_load_against_training()
        # Take the GPU from chat only on a non-CPU device: gate on the device, not the engine name.
        # Pure resolve, so it can run before selection, which the refusal below has to precede.
        device = await asyncio.to_thread(lambda: resolve_diffusion_device_target().device)
        needs_gpu = device != "cpu"

        def _preflight(target):
            # Gated/unreadable-companion refusal, asked of ONE engine (they check different repos).
            return target.preflight_base_access(
                request.model_path,
                fam,
                gguf_filename = request.gguf_filename,
                model_kind = kind,
                base_repo = request.base_repo,
                hf_token = request.hf_token,
            )

        # Last refusal before anything is torn down: a gated/unreadable companion repo. The download
        # plan checks the same, but the images page falls back to THIS route when that call fails,
        # and the loader's own copy runs after acquire_for already evicted chat. Must precede
        # selection too: activating the other engine unloads the current one, so a pick refused
        # afterwards destroys the model this preserves. Fails open on offline/transient, and runs
        # only where something is at stake -- a GPU handoff, or an engine switch.
        try:
            pending_name = predict_engine(fam, model_kind = kind) if fam is not None else None
        except Exception:  # noqa: BLE001 -- a probe failure must not refuse a loadable pick
            pending_name = None
        preflighted = None
        if pending_name is not None and (needs_gpu or pending_name != active_engine_name()):
            preflighted = engine_for(pending_name)
            await asyncio.to_thread(_preflight, preflighted)

        # Pick the engine for this host (diffusers on GPU, native sd.cpp otherwise), installing sd-cli if needed, BEFORE evicting chat.
        engine = await asyncio.to_thread(
            select_and_activate_engine, fam, hf_token = request.hf_token, model_kind = kind
        )
        # predict_engine is selection's read-only twin: it never installs, so a host whose sd-cli
        # install then fails lands on the OTHER engine. Re-ask the engine actually activated when
        # the prediction missed, so neither the GPU handoff nor the load runs on an unread
        # companion; a correct prediction already made this call, so it is never paid twice. Runs
        # on the CPU path too when a preflight was owed there, since the switch is what is at stake.
        if (needs_gpu or preflighted is not None) and engine is not preflighted:
            await asyncio.to_thread(_preflight, engine)

        def _start_engine_load():
            # Kicks the slow load onto a background thread and returns at once (the client polls images/load-progress).
            return engine.begin_load(
                request.model_path,
                gguf_filename = request.gguf_filename,
                base_repo = request.base_repo,
                family_override = request.family_override,
                hf_token = request.hf_token,
                cpu_offload = request.cpu_offload,
                memory_mode = request.memory_mode,
                speed_mode = request.speed_mode,
                text_encoder_quant = request.text_encoder_quant,
                transformer_quant = request.transformer_quant,
                transformer_quant_fast_accum = request.transformer_quant_fast_accum,
                transformer_prequant_path = request.transformer_prequant_path,
                attention_backend = request.attention_backend,
                transformer_cache = request.transformer_cache,
                transformer_cache_threshold = request.transformer_cache_threshold,
                model_kind = kind,
                loras = [(s.id, s.weight) for s in request.loras] if request.loras else None,
            )

        def _begin_load():
            # Under the router transition lock: begin_load on a deactivated engine leaves a resident model nothing can reach.
            return begin_load_on(engine, _start_engine_load)

        if needs_gpu:
            # Register the in-flight load UNDER the arbiter lock: otherwise a competing acquire in that gap evicts DIFFUSION before
            # the load is marked, finds nothing to cancel, and both allocate at once. The training admission wraps the same span.
            def _acquire_and_begin():
                with _diffusion_training_admission():
                    return acquire_for(DIFFUSION, _begin_load)

            status_dict = await asyncio.to_thread(_acquire_and_begin)
        else:
            # A CPU-only native load never touches the GPU, but switching FROM a previous GPU load leaves DIFFUSION marked as owner, so release (owner-guarded).
            await asyncio.to_thread(release, DIFFUSION)
            status_dict = await asyncio.to_thread(_begin_load)
        return DiffusionStatusResponse(**annotate_status(status_dict))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code = 400, detail = redact_native_paths(str(exc)))
    except RuntimeError as exc:
        # A load is already in progress.
        raise HTTPException(status_code = 409, detail = str(exc))


# Count of finished generations still writing their PNG/gallery records; generate-progress reports active while above 0. Mutated only on the event loop, so no lock.
_diffusion_persist_active = 0


_GENERATE_FAILURE_FALLBACK = "Image generation failed."
# Failure classes worth naming in the UI, as FIXED text: the engine's own message can embed local paths and argv, so only the class is reported.
_GENERATE_FAILURE_CLASSES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("out of memory", "outofmemory", "oom"),
        "The device ran out of memory. Try a smaller size, fewer steps, or a smaller batch.",
    ),
    (
        ("sd-server connection lost", "sd-cli exited", "process exited", "ggml_abort", "signal"),
        "The native image renderer stopped unexpectedly. Switch the engine to diffusers, or see "
        "the server log for its output.",
    ),
)


def _generate_failure_detail(message: str) -> str:
    """A user-facing reason for a failed generation, built only from fixed text.

    The bare literal left a real failure undiagnosable from the UI: on a Metal host the native
    renderer aborts inside its own text encoder, and the page showed "Image generation failed."
    with nothing to act on. Naming the CLASS of failure keeps the message useful without echoing
    the engine's text, which can carry local paths and argv."""
    text = str(message or "").lower()
    for needles, detail in _GENERATE_FAILURE_CLASSES:
        if any(n in text for n in needles):
            return f"{_GENERATE_FAILURE_FALLBACK} {detail}"
    return _GENERATE_FAILURE_FALLBACK


@studio_router.post("/images/generate", response_model = DiffusionGenerateResponse)
async def generate_diffusion_image(
    request: DiffusionGenerateRequest, current_subject: str = Depends(get_current_subject)
):
    from core.inference import image_gallery
    from core.inference.diffusion_engine_router import get_active_diffusion_engine
    from core.inference.diffusion_families import (
        DIFFUSION_CANCELLED_MSG,
        DIFFUSION_NOT_LOADED_MSG,
    )

    backend = get_active_diffusion_engine()
    try:
        result = await asyncio.to_thread(
            backend.generate,
            prompt = request.prompt,
            negative_prompt = request.negative_prompt,
            width = request.width,
            height = request.height,
            steps = request.steps,
            guidance = request.guidance,
            seed = request.seed,
            batch_size = request.batch_size,
            prompts = request.prompts,
            seeds = request.seeds,
            init_image = request.init_image,
            mask_image = request.mask_image,
            strength = request.strength,
            upscale = request.upscale,
            reference_images = request.reference_images,
            loras = [(l.id, l.weight) for l in request.loras] if request.loras else None,
            controlnet = (
                (
                    request.controlnet.id,
                    request.controlnet.image,
                    request.controlnet.control_type,
                    request.controlnet.strength,
                    request.controlnet.guidance_start,
                    request.controlnet.guidance_end,
                )
                if request.controlnet
                else None
            ),
        )
    except ValueError as exc:
        # Bad client input (undecodable image/mask, or an unsupported workflow): a 400 with the reason, not a generic 500.
        raise HTTPException(status_code = 400, detail = str(exc))
    except RuntimeError as exc:
        # Only "no model loaded" / user-cancelled are client-state (409); both engines raise these two EXACT messages. The
        # native engine also raises RuntimeError for failures whose text embeds the sd-cli tail, so match the sentinels exactly.
        msg = str(exc)
        if msg in (DIFFUSION_NOT_LOADED_MSG, DIFFUSION_CANCELLED_MSG):
            raise HTTPException(status_code = 409, detail = msg)
        logger.error("diffusion.generate_failed: %s", exc, exc_info = True)
        raise HTTPException(status_code = 500, detail = _generate_failure_detail(msg))
    except Exception as exc:
        logger.error("diffusion.generate_failed: %s", exc, exc_info = True)
        raise HTTPException(status_code = 500, detail = "Image generation failed.")

    # Persist each image with its full recipe. BOTH engines batch with a distinct seed per image, returned in ``seeds``, so each is individually reproducible.
    created_at = time.time()
    per_image_seeds = result.get("seeds")
    # A prompts/seeds LIST drives the image count and each image's own seed, so persist those as single-image recipes keyed on that seed.
    list_driven = bool(request.prompts or request.seeds)

    def _persist() -> list[dict]:
        records = []
        for index, image in enumerate(result["images"]):
            seed = (
                per_image_seeds[index]
                if per_image_seeds and index < len(per_image_seeds)
                else result["seed"]
            )
            records.append(
                image_gallery.save(
                    image,
                    {
                        # A prompts-list batch records each image's OWN prompt so its recipe replays exactly.
                        "prompt": (
                            request.prompts[index]
                            if request.prompts and index < len(request.prompts)
                            else request.prompt
                        ),
                        "negative_prompt": request.negative_prompt,
                        # Persist the ACTUAL output size, not the request sliders: the conditioned workflows derive it from the upload.
                        "width": getattr(image, "width", None) or request.width,
                        "height": getattr(image, "height", None) or request.height,
                        "steps": request.steps,
                        "guidance": request.guidance,
                        "seed": seed,
                        # Base seed the batch launched with. The native engine derives per-image seeds as base + index, so restore replays from this base. A list-driven image carries its OWN seed.
                        "batch_seed": seed if list_driven else result["seed"],
                        # Position within the batch (shared timestamp), so the export filename stays unique.
                        "batch_index": index,
                        # The batch shares one seed, so reproducing a batch_index>0 image needs the original batch_size.
                        "batch_size": 1 if list_driven else request.batch_size,
                        "model": result.get("repo_id"),
                        # The BUILD the image came off, not just the repo id: a GGUF repo holds many quants, a dense load may be torchao-
                        # quantised, and a bake is not the same build as the adapter-less one. Absent on records written before this existed.
                        "model_kind": result.get("model_kind"),
                        "gguf_filename": result.get("gguf_filename"),
                        "transformer_quant": result.get("transformer_quant"),
                        "baked_loras": list(result.get("baked_loras") or []),
                        # The adapters APPLIED to this generation. A baked-but-disabled adapter is recorded above as part of the build instead.
                        "loras": [f"{l.id}:{l.weight:g}" for l in request.loras or []],
                        "controlnet": (
                            f"{request.controlnet.id}:{request.controlnet.control_type}:"
                            f"{request.controlnet.strength:g}"
                            # strength 0 is disabled and skipped before loading/conditioning, so do not claim a ControlNet was applied in the recipe/metadata.
                            if request.controlnet and request.controlnet.strength > 0
                            else None
                        ),
                        # The conditioned workflows keep their scalar settings here. The source, mask, reference and control IMAGES are
                        # deliberately not persisted (user uploads with their own lifetime), so the client asks for them again on restore.
                        "workflow": result.get("workflow"),
                        "strength": request.strength,
                        "upscale": request.upscale,
                        "controlnet_guidance": (
                            f"{request.controlnet.guidance_start:g}:{request.controlnet.guidance_end:g}"
                            if request.controlnet and request.controlnet.strength > 0
                            else None
                        ),
                        "reference_image_count": len(request.reference_images or []) or None,
                        "created_at": created_at,
                    },
                )
            )
        return records

    # Hold generate-progress "active" across the persist so a reload mount probe cannot refresh the gallery before these records exist.
    global _diffusion_persist_active
    _diffusion_persist_active += 1
    try:
        records = await asyncio.to_thread(_persist)
    except Exception as exc:
        logger.error("diffusion.persist_failed: %s", exc)
        raise HTTPException(status_code = 500, detail = "Failed to save the generated image.")
    finally:
        _diffusion_persist_active -= 1

    return DiffusionGenerateResponse(images = [GalleryImage(**r) for r in records])


@studio_router.get("/images/gallery", response_model = GalleryListResponse)
async def list_gallery_images(
    limit: int = 50,
    offset: int = 0,
    current_subject: str = Depends(get_current_subject),
):
    from pydantic import ValidationError

    from core.inference import image_gallery

    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    # Validate inside the pager so offset / limit / has_more all count over the accepted domain: a record that fails
    # GalleryImage(**r) only after slicing let a leading bad row return an empty page with has_more=True, stalling scroll.
    def _valid_gallery_image(record: dict) -> bool:
        try:
            GalleryImage(**record)
        except ValidationError:
            return False
        return True

    # Fetch one extra to learn whether more remain, without a second scan.
    records = await asyncio.to_thread(
        image_gallery.list_images, limit + 1, offset, valid = _valid_gallery_image
    )
    has_more = len(records) > limit
    images = [GalleryImage(**r) for r in records[:limit]]
    return GalleryListResponse(images = images, has_more = has_more)


@studio_router.get("/images/gallery/{image_id}/file")
async def get_gallery_image_file(
    image_id: str, current_subject: str = Depends(get_current_subject)
):
    from core.inference import image_gallery

    # Ownership-gate the serve like delete/clear: resolve only a Studio-owned PNG, so a guessed stem cannot stream out a foreign file.
    path = await asyncio.to_thread(image_gallery.owned_image_path, image_id)
    if path is None:
        raise HTTPException(status_code = 404, detail = "Image not found.")
    data = await asyncio.to_thread(path.read_bytes)
    # Immutable content (id is unique per image), so let the browser cache it.
    return Response(
        content = data,
        media_type = "image/png",
        headers = {"Cache-Control": "private, max-age=31536000, immutable"},
    )


@studio_router.delete("/images/gallery/{image_id}")
async def delete_gallery_image(image_id: str, current_subject: str = Depends(get_current_subject)):
    from core.inference import image_gallery

    deleted = await asyncio.to_thread(image_gallery.delete, image_id)
    if not deleted:
        raise HTTPException(status_code = 404, detail = "Image not found.")
    return {"deleted": True}


@studio_router.delete("/images/gallery")
async def clear_gallery_images(current_subject: str = Depends(get_current_subject)):
    from core.inference import image_gallery
    removed = await asyncio.to_thread(image_gallery.clear)
    return {"removed": removed}


@studio_router.post("/images/unload", response_model = DiffusionStatusResponse)
async def unload_diffusion_model(current_subject: str = Depends(get_current_subject)):
    from core.inference.diffusion_engine_router import annotate_status, get_active_diffusion_engine
    from core.inference.gpu_arbiter import release_if, DIFFUSION

    status_dict = await asyncio.to_thread(get_active_diffusion_engine().unload)
    # Drop DIFFUSION ownership only if nothing is resident AND no load is in flight, or a later chat load skips eviction and
    # OOMs the new pipeline. An in-flight load reads is_loaded False, so gate on loading_repo_ids() and use release_if.
    engine = get_active_diffusion_engine()
    await asyncio.to_thread(
        release_if,
        DIFFUSION,
        lambda: not engine.loading_repo_ids() and not engine.is_loaded,
    )
    return DiffusionStatusResponse(**annotate_status(status_dict))


@studio_router.get("/images/status", response_model = DiffusionStatusResponse)
async def diffusion_status(current_subject: str = Depends(get_current_subject)):
    from core.inference.diffusion_engine_router import active_status
    return DiffusionStatusResponse(**active_status())


@studio_router.get("/images/info", response_model = DiffusionInferenceInfoResponse)
async def diffusion_inference_info(current_subject: str = Depends(get_current_subject)):
    """Static per-family footprint summary for the Advanced Dtype tradeoff.

    Hardware-independent (served from the pure auto-policy tables, no GPU probing), so it
    is cheap and safe to fetch before anything is loaded."""
    from core.inference.diffusion_inference_info import family_inference_infos
    return DiffusionInferenceInfoResponse(families = family_inference_infos())


@studio_router.get("/images/load-progress", response_model = DiffusionLoadProgressResponse)
async def diffusion_load_progress(current_subject: str = Depends(get_current_subject)):
    from core.inference.diffusion_engine_router import get_active_diffusion_engine
    return DiffusionLoadProgressResponse(**get_active_diffusion_engine().load_progress())


@studio_router.get("/images/generate-progress", response_model = DiffusionGenerateProgressResponse)
async def diffusion_generate_progress(current_subject: str = Depends(get_current_subject)):
    from core.inference.diffusion_engine_router import get_active_diffusion_engine

    progress = get_active_diffusion_engine().generate_progress()
    # A finished generation still persisting its gallery record counts as active, so a reload probe keeps polling.
    if _diffusion_persist_active > 0 and not progress["active"]:
        progress = {**progress, "active": True}
    return DiffusionGenerateProgressResponse(**progress)


# ──────────────────────────────────────────────────────────────────────────
# OpenAI-compatible images API (POST /v1/images/generations). The inference router is mounted at both /api/inference and /v1, so this
# also answers /v1/images/generations for OpenAI clients. The Studio Image tab uses the richer /images/generate above; this is the spec shape.
# ──────────────────────────────────────────────────────────────────────────


# Diffusion dims must land in [256, 2048] on a multiple of 16; the named OpenAI sizes all satisfy this. Mirrors DiffusionGenerateRequest.
_IMAGE_SIZE_RE = _re.compile(r"^(\d{1,5})\s*x\s*(\d{1,5})$")
_IMAGE_DIM_MIN, _IMAGE_DIM_MAX = 256, 2048
# Sanitized 503 detail shared by the pre-check and the unload-race branch, so both "no image model" responses stay identical.
_NO_IMAGE_MODEL_MSG = "No image model loaded. Load an image model first."


def _parse_openai_image_size(size: str) -> tuple[int, int]:
    """OpenAI ``size`` -> (width, height). ``auto``/empty -> 1024x1024 (~1MP, what
    these models target). Raises ValueError with a client-facing message."""
    text = (size or "").strip().lower()
    if text in ("", "auto"):
        return 1024, 1024
    match = _IMAGE_SIZE_RE.match(text)
    if not match:
        raise ValueError("size must be 'auto' or '<width>x<height>', e.g. '1024x1024'.")
    width, height = int(match.group(1)), int(match.group(2))
    for label, value in (("width", width), ("height", height)):
        if not _IMAGE_DIM_MIN <= value <= _IMAGE_DIM_MAX:
            raise ValueError(f"size {label} must be between {_IMAGE_DIM_MIN} and {_IMAGE_DIM_MAX}.")
        if value % 16 != 0:
            raise ValueError(f"size {label} must be a multiple of 16.")
    return width, height


# response_format=url links must be fetchable by whoever received them: a client downloads data[].url with a plain GET
# and no Authorization header, so mint a short-lived HMAC link (1h, like OpenAI) and leave the gallery route bearer-only.
_IMAGE_LINK_TTL = 3600
_IMAGE_LINK_SECRET = _secrets.token_bytes(32)


def _sign_image_id(image_id: str) -> str:
    exp = int(time.time()) + _IMAGE_LINK_TTL
    payload = f"{image_id}.{exp}"
    sig = _hmac.new(_IMAGE_LINK_SECRET, payload.encode(), _hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _verify_image_link_token(token: str) -> Optional[str]:
    """The image id a valid, unexpired token names, else None. Gallery ids are
    ``[A-Za-z0-9_-]`` so the two dots always split id / expiry / signature."""
    try:
        image_id, exp_s, sig = token.rsplit(".", 2)
    except ValueError:
        return None
    expected = _hmac.new(
        _IMAGE_LINK_SECRET, f"{image_id}.{exp_s}".encode(), _hashlib.sha256
    ).hexdigest()
    if not _hmac.compare_digest(sig, expected):
        return None
    try:
        if int(exp_s) < int(time.time()):
            return None
    except ValueError:
        return None
    return image_id


def _absolute_image_url(request: Request, image_id: str) -> str:
    """The absolute, directly fetchable link for one gallery image, on the request's own
    scheme+host. Signed rather than bearer-gated (see above), so a standard image client can
    download it; b64_json still avoids the round trip entirely."""
    relative = (
        f"/api/inference/images/gallery/{image_id}/file-signed?token={_sign_image_id(image_id)}"
    )
    return str(request.base_url).rstrip("/") + relative


@studio_router.get("/images/gallery/{image_id}/file-signed")
async def get_gallery_image_file_signed(image_id: str, token: str = Query(...)):
    """Serve one gallery PNG gated by the HMAC token instead of the bearer, for the
    response_format=url links a plain image client downloads. Same ownership gate as the
    authenticated route, and the token names the single image it may serve."""
    from core.inference import image_gallery

    if _verify_image_link_token(token) != image_id:
        raise HTTPException(status_code = 401, detail = "Invalid or expired image link.")
    path = await asyncio.to_thread(image_gallery.owned_image_path, image_id)
    if path is None:
        raise HTTPException(status_code = 404, detail = "Image not found.")
    data = await asyncio.to_thread(path.read_bytes)
    return Response(
        content = data,
        media_type = "image/png",
        headers = {"Cache-Control": "private, max-age=31536000, immutable"},
    )


@router.post(
    "/images/generations",
    response_model = ImageGenerationResponse,
    response_model_exclude_none = True,
)
async def openai_image_generations(
    body: ImageGenerationRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """OpenAI-compatible text-to-image (POST /v1/images/generations).

    Generates ``n`` images from ``prompt`` on the loaded diffusion model and
    returns them as URLs (default) or base64 PNGs per ``response_format``. Steps
    and guidance have no OpenAI knob, so they default per loaded model."""
    from core.inference import image_gallery
    from core.inference.diffusion_engine_router import get_active_diffusion_engine
    from core.inference.diffusion_families import default_generation_params

    if body.stream:
        raise HTTPException(
            status_code = 400,
            detail = openai_error_body(
                "Streaming image generation is not supported.", status = 400, param = "stream"
            ),
        )
    try:
        width, height = _parse_openai_image_size(body.size)
    except ValueError as exc:
        raise HTTPException(
            status_code = 400, detail = openai_error_body(str(exc), status = 400, param = "size")
        )

    # Use the active engine (diffusers OR native sd.cpp), the same accessor /images/generate uses.
    backend = get_active_diffusion_engine()
    status = backend.status()
    if not status.get("loaded"):
        # Mirror /v1/completions and /v1/embeddings, which 503 when their backend is not loaded.
        raise HTTPException(status_code = 503, detail = _NO_IMAGE_MODEL_MSG)

    # An edit-only model needs an input image this API cannot supply; refuse with a 400 rather than a backend 500.
    workflows = status.get("workflows") or []
    if workflows and "txt2img" not in workflows:
        raise HTTPException(
            status_code = 400,
            detail = openai_error_body(
                "The loaded image model is edit-only (it requires an input image); "
                "load a text-to-image model to use this endpoint.",
                status = 400,
                param = "model",
            ),
        )

    # Fall back to the resolved base repo so a local-path load still gets the right per-model steps/guidance.
    steps, guidance = default_generation_params(status.get("repo_id"), status.get("base_repo"))
    try:
        result = await asyncio.to_thread(
            backend.generate,
            prompt = body.prompt,
            width = width,
            height = height,
            steps = steps,
            guidance = guidance,
            batch_size = body.n,
        )
    except Exception as exc:  # noqa: BLE001 (single boundary, sanitized envelope)
        # A RuntimeError with the model now unloaded means it was evicted mid-call (a race): 503. Every other failure is a real 500 whose raw message must not reach the client.
        if isinstance(exc, RuntimeError) and not backend.is_loaded:
            raise HTTPException(status_code = 503, detail = _NO_IMAGE_MODEL_MSG)
        logger.error("openai_images.generate_failed: %s", exc)
        raise HTTPException(status_code = 500, detail = "Image generation failed.")

    created = int(time.time())
    want_b64 = body.response_format == "b64_json"
    # Persist each image with its full recipe, like /images/generate, so url links resolve and images show in the gallery.
    recipe = {
        "prompt": body.prompt,
        "negative_prompt": None,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance": guidance,
        # The batch shares one base seed, so restoring a batch_index>0 sibling needs the original batch_size.
        "batch_size": body.n,
        "model": result.get("repo_id"),
        "created_at": float(created),
    }
    # The diffusers batch shares one seed; the native batch uses a distinct seed per image, so record each image's own seed.
    per_image_seeds = result.get("seeds")

    def _persist() -> list[ImageGenerationData]:
        items: list[ImageGenerationData] = []
        for index, image in enumerate(result["images"]):
            seed = (
                per_image_seeds[index]
                if per_image_seeds and index < len(per_image_seeds)
                else result["seed"]
            )
            # batch_seed is the base the native engine derives per-image seeds from, so restore does not double-advance.
            record = image_gallery.save(
                image,
                {**recipe, "batch_index": index, "seed": seed, "batch_seed": result["seed"]},
            )
            if want_b64:
                encoded = image_gallery.image_b64(record["id"])
                if encoded is None:  # vanished between write and read — fail the call
                    raise RuntimeError("generated image could not be read back for encoding")
                items.append(ImageGenerationData(b64_json = encoded))
            else:
                items.append(ImageGenerationData(url = _absolute_image_url(request, record["id"])))
        return items

    try:
        data = await asyncio.to_thread(_persist)
    except Exception as exc:  # noqa: BLE001
        logger.error("openai_images.persist_failed: %s", exc)
        raise HTTPException(status_code = 500, detail = "Failed to save the generated image.")

    return ImageGenerationResponse(created = created, data = data)
