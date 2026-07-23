# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Inference API routes for model loading and text generation.
"""

from functools import wraps
import os
import sys
import time
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse, JSONResponse, Response

from starlette.background import BackgroundTask
from starlette.requests import ClientDisconnect
from typing import Any, Callable, List, Literal, Optional, Union
import json
import httpx
from loggers import get_logger
import asyncio
import threading
import weakref
from contextlib import ExitStack


import re as _re

# Model size extraction (shared with core/inference/llama_cpp.py)
from utils.models import extract_model_size_b as _extract_model_size_b

from utils.api_errors import openai_error_body, anthropic_error_body
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


def _sse_streaming_response(content) -> StreamingResponse:
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
        LlamaCppBackend,
        _DEFAULT_FIRST_TOKEN_TIMEOUT_S,
        _DEFAULT_MAX_TOKENS_FLOOR,
        _DEFAULT_STREAM_STALL_TIMEOUT_S,
        _canonicalize_spec_mode,
        _extra_args_set_spec_type,
        _hf_offline_if_dns_dead,
        detect_reasoning_flags,
    )
    from core.inference.llama_server_args import (
        _effective_tensor_parallel,
        _tensor_parallel_matches_loaded,
        extra_args_disable_mmproj,
        parse_split_mode_override,
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
        _DEFAULT_FIRST_TOKEN_TIMEOUT_S,
        _DEFAULT_MAX_TOKENS_FLOOR,
        _DEFAULT_STREAM_STALL_TIMEOUT_S,
        _canonicalize_spec_mode,
        _extra_args_set_spec_type,
        _hf_offline_if_dns_dead,
        detect_reasoning_flags,
    )
    from core.inference.llama_server_args import (
        _effective_tensor_parallel,
        _tensor_parallel_matches_loaded,
        extra_args_disable_mmproj,
        parse_split_mode_override,
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


def _openai_admission_log(
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
        "openai admission %s: mode=%s path=%s completion_id=%s capacity=%s active=%s queued=%s wait_ms=%s",
        event,
        mode,
        _openai_admission_request_path(request),
        completion_id,
        getattr(snapshot, "capacity", None),
        getattr(snapshot, "active", None),
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


def _tracked_cancel_unstarted_cleanup(tracker):
    """unstarted_cleanup that exits ``tracker`` on a pre-start disconnect, when
    the generator's finally (which normally exits it) never runs."""

    async def _cleanup() -> None:
        tracker.__exit__(None, None, None)

    return _cleanup


async def _aclose_stream_resources(
    *,
    watchers = (),
    iterator = None,
    resp = None,
    client = None,
) -> None:
    """Tear down an httpx streaming generator's resources in the required order:
    cancel + await each watcher task, then aclose() the byte/line iterator, the
    response, and the client. Each step swallows its own exceptions so teardown
    always completes; a close-time CancelledError is re-raised only after every
    step has run. See _anthropic_passthrough_stream for the ordering rationale."""
    for watcher in watchers:
        if watcher is not None:
            watcher.cancel()
            try:
                await watcher
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
        send_task.cancel()
        try:
            await send_task
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
        cancel_task.cancel()
        try:
            await cancel_task
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
    LoadRequest,
    UnloadRequest,
    TranscribeRequest,
    SttLoadRequest,
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


def _permission_mode_confirm(payload) -> bool:
    """Effective confirm-gate intent for Unsloth's own local tool loop.

    Honors the documented default that an unset permission_mode behaves as
    "ask". An explicit confirm_tool_calls (True or False) wins; explicit
    ask/auto always engage the gate (a non-streaming one is then rejected, since
    it cannot prompt); off/full never prompt. An unset mode defaults to ask, but
    that is only realizable on a streaming request, so a non-streaming unset
    request keeps the legacy run-without-gate behavior instead of 400ing. Used
    at the pre-switch guard and the per-backend tool paths so a forced tool loop
    (CLI --enable-tools) with the default mode still gates streaming requests.
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


async def _stop_local_disconnect_cancel_watcher(watcher) -> None:
    watcher.cancel()
    try:
        await watcher
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
):
    if not usage:
        return
    api_monitor.set_usage(
        monitor_id,
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens"),
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens"),
        total_tokens = usage.get("total_tokens"),
        context_length = context_length,
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
    _monitor_usage(monitor_id, data.get("usage"), context_length)
    # Defensive: ignore malformed shapes so the helper never raises into the
    # streaming generator and aborts the user's response.
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    reply_parts: list[tuple[int, str]] = []
    for idx, choice in enumerate(choices):
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}
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
        api_monitor.append_reply(monitor_id, reply_parts[0][1])
        return
    api_monitor.append_reply(
        monitor_id,
        "\n\n".join(f"Choice {idx + 1}:\n{text}" for idx, text in reply_parts),
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


def _monitor_openai_sse_line(
    monitor_id: Optional[str],
    raw_line: str,
    context_length = None,
) -> Optional[str]:
    if not monitor_id:
        return None
    # SSE spec allows `data:value` and `data: value`; accept both.
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


def _monitor_context_length() -> Optional[int]:
    llama_backend = get_llama_cpp_backend()
    if getattr(llama_backend, "is_loaded", False):
        context_length = _positive_int_or_none(getattr(llama_backend, "context_length", None))
        if context_length is not None:
            return context_length
    backend = get_inference_backend()
    if not backend.active_model_name:
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


def _monitor_active_model() -> Optional[str]:
    llama_backend = get_llama_cpp_backend()
    if getattr(llama_backend, "is_loaded", False):
        return getattr(llama_backend, "model_identifier", None)
    backend = get_inference_backend()
    return backend.active_model_name


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


def _request_matches_loaded_settings(
    request: LoadRequest,
    llama_backend: LlamaCppBackend,
    effective_chat_template_override: Optional[str] = None,
) -> bool:
    """True iff every runtime setting on the request matches the loaded server.
    Caller has already checked model+variant+is_loaded. See #5401.

    ``effective_chat_template_override`` is the resolved template that will be
    launched (user override, else a bundled family template such as the
    gemma-4 override), so the dedup compares against what the backend actually
    holds rather than the raw request field. Defaults to the request field for
    callers that do not resolve a bundled override."""
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
            strip_tensor_split = _should_strip_tensor_split(request),
            strip_offload = request.gpu_memory_mode == "manual",
        )
    )
    if not _tensor_parallel_matches_loaded(
        effective_extra, request.tensor_parallel, llama_backend.tensor_parallel
    ):
        return False
    # The diffusion runner is mode-agnostic (it always reports "auto" and ignores
    # the layer/MoE/split knobs), so a standing manual preference in the request
    # must not force a needless reload -- only the GPU pick matters.
    if not llama_backend.is_diffusion:
        if request.gpu_memory_mode != llama_backend.gpu_memory_mode:
            return False
        # Manual: a layer-count change always reloads; MoE/split only matter with
        # an explicit offload (gpu_layers >= 0), so a leftover value under Auto
        # must not force one. Mirrors LlamaCppBackend._already_in_target_state.
        if request.gpu_memory_mode == "manual" and (
            request.gpu_layers != llama_backend.gpu_layers
            or (
                request.gpu_layers >= 0
                and (
                    request.n_cpu_moe != llama_backend.n_cpu_moe
                    or (request.tensor_split or None) != (llama_backend.tensor_split or None)
                )
            )
        ):
            return False
    # A changed GPU pick must reload. The diffusion runner collapses a multi-GPU
    # request to its single lowest device (it drives one device only), so the
    # backend records just that device; compare the request the same way, or a
    # multi-GPU pick that resolves to the same device needlessly reloads.
    if llama_backend.is_diffusion:
        _req_gpu_ids = [sorted(request.gpu_ids)[0]] if request.gpu_ids else None
    else:
        _req_gpu_ids = sorted(request.gpu_ids) if request.gpu_ids else None
    if _req_gpu_ids != llama_backend.gpu_ids:
        return False
    # Preserved tensor->layer fallback (both report tensor=off, so the check above
    # matches): if the user now explicitly drops tensor intent, reload so placement
    # re-selects instead of keeping the all-GPU mask (#6659). The effective check
    # includes the env, so an env-only tensor (LLAMA_ARG_SPLIT_MODE=tensor) that
    # can't actually be dropped falls through to the env-downgrade match, not a loop.
    if llama_backend.layer_preserves_tensor_intent and _is_explicit_tensor_drop(request):
        return False
    # Spec decoding works on vision models too (MTP is mmproj-compatible,
    # llama.cpp #22673; the old ``not is_vision`` gate is gone), so compare
    # the real requested mode -- coercing vision to ``off`` here used to
    # swallow every spec-mode change on a vision model as already_loaded.
    req_mode = _canonicalize_spec_mode(request.speculative_type) or "auto"
    backend_mode = llama_backend.requested_spec_mode or "auto"
    if req_mode != backend_mode:
        return False
    # Prior HF load fell back with drafter_not_found: a same-settings reload must
    # retry the download, not dedupe to the stale fallback. HF only (hf_repo set);
    # local/native loads have no download to retry (handled by the path compare).
    if (
        llama_backend.hf_repo
        and llama_backend.spec_fallback_reason == "drafter_not_found"
        and req_mode in ("auto", "mtp", "mtp+ngram")
        and not _extra_args_set_spec_type(effective_extra)
    ):
        return False
    # spec_draft_n_max only matters with an MTP variant; None means "platform
    # default" and matches whatever the backend chose.
    if backend_mode in ("mtp", "mtp+ngram") and request.spec_draft_n_max is not None:
        if int(request.spec_draft_n_max) != (llama_backend.spec_draft_n_max or 0):
            return False
    _effective_cto = (
        effective_chat_template_override
        if effective_chat_template_override is not None
        else request.chat_template_override
    )
    if (_effective_cto or None) != (llama_backend.chat_template_override or None):
        return False
    # llama_extra_args=None means "inherit"; only an explicit differing list
    # forces a reload. On the inherit path, refuse to match if stored extras
    # contain any shadow flag, so the reload path strips them rather than
    # leaving a stale override in effect. (backend_extra computed above.)
    if request.llama_extra_args is None:
        # Mirror the reload's conditional strips, so a preserved non-tensor mode
        # (row/none/layer) isn't seen as stale and doesn't trigger a needless
        # reload of a healthy server, while an inherited offload/ratio flag that
        # the reload *would* strip is correctly seen as stale.
        if (
            backend_extra
            and strip_shadowing_flags(
                backend_extra,
                strip_split_mode = _should_strip_split_mode(request, backend_extra),
                strip_tensor_split = _should_strip_tensor_split(request),
                strip_offload = request.gpu_memory_mode == "manual",
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


async def _wait_for_model_switch_idle(*, current_request_counted: bool) -> None:
    """Wait until a model replacement cannot interrupt active inference.

    The caller holds ``inference_lifecycle_gate``, which prevents new inference
    from starting while existing requests drain. Auto-switch requests that have
    resolved their targets are scheduler waiters, not active generations, so
    exclude them to avoid a queue deadlock.
    """
    from core.inference.llama_keepwarm import other_inference_request_count
    while True:
        queued_switches = _switch_waiter_count()
        if current_request_counted and queued_switches > 0:
            queued_switches -= 1
        active_others = other_inference_request_count(
            current_request_counted = current_request_counted,
            include_pending = False,
        )
        if active_others <= queued_switches:
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


_DISABLE_OPENAI_AUTO_SWITCH_SCOPE_KEY = "_unsloth_disable_openai_auto_switch"
# Sentinel a raw-body endpoint passes when the request omits ``model``: it must
# only restore an idle-freed model, never run the resolver (so a downloaded GGUF
# literally named "default" can't be swapped to). The NUL keeps it off any index.
_RELOAD_ONLY_MODEL = "\x00reload-only"


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
    compat) and no remote download is triggered. ``require_vision`` rejects a swap
    to a text-only target before it runs, so an image request can't evict the
    resident vision model only to 400 afterwards.
    """
    from utils.openai_auto_switch_settings import (
        get_openai_auto_switch_enabled,
        get_auto_unload_idle_seconds,
        get_model_override,
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
                or getattr(get_inference_backend(), "active_model_name", None)
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
        bare = ":" not in requested_model

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
                        # Apply this model's saved launch flags so the swap honors the config.
                        override = get_model_override(override_id)
                        load_kwargs = {"model_path": target_id, "gguf_variant": variant}
                        if override.get("llama_extra_args") is not None:
                            load_kwargs["llama_extra_args"] = override["llama_extra_args"]
                        if override.get("max_seq_length") is not None:
                            load_kwargs["max_seq_length"] = override["max_seq_length"]
                        # Reuse the load impl so its dedup, tensor fallback, and threading
                        # apply. Call the impl directly: we already hold the lifecycle gate
                        # the /load route would otherwise take, so the route would deadlock.
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
        with open(adapter_cfg_path) as f:
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
) -> float:
    """KV-cache VRAM (GB) at the larger of max_seq_length and any `--ctx-size`/`-c`
    override, over n_parallel slots, with the default f16 cache so the estimate is
    never below what the server allocates. 0 if metadata is unreadable."""
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
        kv = probe._estimate_kv_cache_bytes(ctx, n_parallel = max(1, n_parallel or 1))
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
                main, max_seq_length, llama_extra_args, n_parallel
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


def _classify_diffusion_gguf(config: ModelConfig) -> Optional[bool]:
    """Classify a GGUF as diffusion, normal, or unknown before it is loaded.

    ``None`` is important here: a remote GGUF whose header is not cached can
    still be routed to the single-GPU diffusion runner after download. Treating
    that case as normal would let Manual mode skip the training guard even
    though the runner ignores Manual's llama-server placement controls.
    """
    identity = " ".join(
        str(getattr(config, attr, "") or "") for attr in ("identifier", "gguf_hf_repo", "gguf_file")
    ).lower()
    if "diffusion" in identity:
        return True

    try:
        main = getattr(config, "gguf_file", None)
        if not (main and Path(main).is_file()):
            repo = getattr(config, "gguf_hf_repo", None)
            variant = getattr(config, "gguf_variant", None)
            if repo and variant:
                from hub.utils.gguf import resolve_local_gguf_path
                main = resolve_local_gguf_path(repo, variant)
        if not main or not Path(main).is_file():
            return None

        probe = LlamaCppBackend()
        probe._read_gguf_metadata(str(main))
        if probe.is_diffusion:
            return True
        # A successfully decoded architecture proves that this is a normal
        # llama-server GGUF. No architecture means the lightweight probe could
        # not establish the routing decision, so preserve the unknown state.
        if getattr(probe, "_architecture", None):
            return False
        return None
    except Exception as e:
        logger.debug("Could not identify diffusion GGUF for training guard: %s", e)
        return None


def _guard_chat_load_against_training(
    config: ModelConfig,
    *,
    model_identifier: str,
    hf_token: Optional[str],
    load_in_4bit: bool,
    max_seq_length: int,
    requested_gpu_ids: Optional[List[int]],
    llama_extra_args: Optional[list[str]] = None,
    n_parallel: int = 1,
    gpu_memory_mode: Literal["auto", "manual"] = "auto",
) -> None:
    """Protect active training from automatically placed chat-model loads.

    No-op when training is inactive or unknown. `load_in_4bit` must be the
    effective quantization (see _effective_load_in_4bit). Manual chat-GGUF
    placement is an explicit override: Auto layers delegate fitting to
    llama.cpp's ``--fit`` and pinned layers are owned by the user, so neither is
    estimated here. Diffusion is still guarded because its mode-agnostic runner
    ignores those controls and uses one GPU. An unclassified GGUF is guarded as
    potentially diffusion until its local header proves otherwise. Other loads
    raise HTTP 409 when they would not fit beside training.
    """
    from core.training import get_training_backend
    from routes.training_vram import can_load_chat_during_training

    try:
        if not get_training_backend().is_training_active():
            return
    except Exception as e:
        logger.warning("Could not check training state for chat-load guard: %s", e)
        return

    is_gguf = bool(getattr(config, "is_gguf", False))
    diffusion_kind = _classify_diffusion_gguf(config) if is_gguf else False
    if is_gguf and gpu_memory_mode == "manual" and diffusion_kind is False:
        return

    diffusion_gpu = None
    if is_gguf and diffusion_kind is not False:
        # Use the same token selection as the runner: an explicit pick wins,
        # followed by DG_GPU, the first parent-visible token, then GPU 0.
        diffusion_gpu = LlamaCppBackend._diffusion_gpu_arg(
            requested_gpu_ids,
            cpu_only = LlamaCppBackend._effective_gpu_count() == 0,
        )

    required_override_gb = (
        _estimate_gguf_required_gb(
            config,
            hf_token = hf_token,
            max_seq_length = max_seq_length,
            llama_extra_args = llama_extra_args,
            n_parallel = n_parallel,
        )
        if is_gguf
        else None
    )

    ok, info = can_load_chat_during_training(
        model_name = model_identifier,
        hf_token = hf_token,
        load_in_4bit = load_in_4bit,
        max_seq_length = max_seq_length,
        requested_gpu_ids = requested_gpu_ids,
        is_gguf = is_gguf,
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
    if not llama_backend.extra_args:
        return extra_llama_args
    # Inherit the previous load's extras (the chat-settings Apply path doesn't
    # round-trip them; an explicit [] still clears). Gated on (model_identifier,
    # hf_variant) to refuse cross-model pickup, and shadowing flags are
    # stripped so an inherited override can't win the last-wins CLI
    # parse against a freshly-supplied first-class field.
    source = llama_backend.extra_args_source
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
            llama_backend.extra_args,
            strip_context = "max_seq_length" in fields_set,
            strip_cache = "cache_type_kv" in fields_set,
            strip_spec = ("speculative_type" in fields_set or "spec_draft_n_max" in fields_set),
            strip_template = (
                "chat_template_override" in fields_set
                or effective_chat_template_override is not None
            ),
            strip_split_mode = _should_strip_split_mode(request, llama_backend.extra_args),
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
        return await _load_model_impl(request, fastapi_request, current_subject)


async def _load_model_impl(
    request: LoadRequest,
    fastapi_request: Request,
    current_subject: str,
    *,
    current_request_counted: bool = False,
):
    from core.inference.llama_cpp import LlamaServerNotFoundError

    # A new load starts here; arm the progress throttle so this load's first
    # sampled step logs even if it reports 100% immediately (cached/small load).
    _reset_load_progress_step()

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

        # Manual mode owns the offload flags: strip them from EXPLICIT extras
        # too (the inherited path already does), or a last-wins --gpu-layers /
        # --fit in extras re-enables GPU offload on a load status reports as
        # CPU-only. Manual + per-GPU ratio owns --tensor-split the same way.
        if request.gpu_memory_mode == "manual" and extra_llama_args:
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
                and _request_matches_loaded_settings(
                    request,
                    llama_backend,
                    effective_chat_template_override,
                )
                # Skip if a prior audio probe failed -- let load_model retry.
                and getattr(llama_backend, "_audio_probed", True)
            ):
                logger.info(
                    "Model already loaded (GGUF): "
                    f"{model_log_label} variant={request.gguf_variant or llama_backend.hf_variant}, skipping reload"
                )
                inference_config = load_inference_config(llama_backend.model_identifier)

                _gguf_audio = getattr(llama_backend, "_audio_type", None)
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
                    is_diffusion = llama_backend.is_diffusion,
                    is_audio = _gguf_is_audio,
                    audio_type = _gguf_audio,
                    has_audio_input = getattr(llama_backend, "_has_audio_input", False),
                    inference = inference_config,
                    # GGUF loads via llama.cpp: auto_map never executes, so inert (matches validate_model).
                    requires_trust_remote_code = False,
                    context_length = llama_backend.context_length,
                    max_context_length = llama_backend.max_context_length,
                    native_context_length = llama_backend.native_context_length,
                    supports_reasoning = llama_backend.supports_reasoning,
                    reasoning_style = llama_backend.reasoning_style,
                    reasoning_effort_levels = llama_backend.reasoning_effort_levels,
                    reasoning_always_on = llama_backend.reasoning_always_on,
                    supports_preserve_thinking = llama_backend.supports_preserve_thinking,
                    supports_tools = llama_backend.supports_tools,
                    chat_template = llama_backend.chat_template,
                    speculative_type = llama_backend.requested_spec_mode,
                    spec_draft_n_max = llama_backend.spec_draft_n_max,
                    tensor_parallel = llama_backend.tensor_parallel,
                    gpu_memory_mode = llama_backend.gpu_memory_mode,
                    gpu_layers = llama_backend.gpu_layers,
                    n_cpu_moe = llama_backend.n_cpu_moe,
                    tensor_split = llama_backend.tensor_split,
                    n_layers = llama_backend.n_layers,
                    n_moe_layers = llama_backend.n_moe_layers,
                    gpu_ids = llama_backend.gpu_ids,
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

        # GGUF supports gpu_ids: validate the pick up front (before the training
        # guard) so a bad pick is a clean 400, not masked by a VRAM 409. Rejects
        # negative / out-of-range / duplicate ids and UUID/MIG parents. XPU hosts
        # are rejected outright: the picker's indices are torch-xpu ordinals neither
        # applicator speaks (CUDA/HIP masks don't apply, the Vulkan --device pin
        # uses ggml's own Vulkan ordinals), so a pick could land on the wrong device.
        if config.is_gguf and effective_gpu_ids is not None:
            from utils.hardware import DeviceType, get_device
            from utils.hardware.hardware import resolve_requested_gpu_ids

            if get_device() == DeviceType.XPU:
                raise HTTPException(
                    status_code = 400,
                    detail = (
                        "GPU selection (gpu_ids) is not supported on Intel XPU. "
                        "Omit gpu_ids to use all devices."
                    ),
                )
            # Same reasoning for a Vulkan-only build: --device pins ggml's own
            # Vulkan ordinals, so a physical pick can land on the wrong card on
            # masked or non-contiguous hosts.
            if LlamaCppBackend._is_vulkan_backend():
                raise HTTPException(
                    status_code = 400,
                    detail = (
                        "GPU selection (gpu_ids) is not supported with a Vulkan "
                        "llama.cpp build: physical GPU ids have no defined "
                        "mapping to Vulkan device ordinals. Omit gpu_ids to use "
                        "all devices."
                    ),
                )
            try:
                resolve_requested_gpu_ids(effective_gpu_ids)
            except ValueError as exc:
                raise HTTPException(status_code = 400, detail = str(exc)) from exc
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
            if await asyncio.to_thread(latest_tier_active_for, config.identifier, request.hf_token):
                effective_load_in_4bit = False
                logger.info(
                    f"Latest-transformers sidecar active for '{model_log_label}' - "
                    "sizing and loading in 16-bit (4-bit is disabled for brand-new "
                    "architectures)"
                )

        # Inherit the previous same-model load's pass-through extras when this
        # request omits the field (a settings-Apply reload doesn't round-trip
        # them); shadow-stripped so an inherited flag can't override a
        # first-class field the caller did set (#5401).
        extra_llama_args = _resolve_inherited_extra_args(
            request,
            config,
            model_identifier,
            extra_llama_args,
            effective_chat_template_override,
        )

        # Apply the training coexistence policy before the unload step below
        # frees the resident model. Off-loop: the default-mode guard does sync work.
        await asyncio.to_thread(
            _guard_chat_load_against_training,
            config,
            model_identifier = model_identifier,
            hf_token = request.hf_token,
            load_in_4bit = effective_load_in_4bit,
            max_seq_length = request.max_seq_length,
            requested_gpu_ids = effective_gpu_ids,
            llama_extra_args = extra_llama_args,
            n_parallel = getattr(fastapi_request.app.state, "llama_parallel_slots", 1),
            gpu_memory_mode = request.gpu_memory_mode,
        )

        # ── GGUF path: load via llama-server ──────────────────────
        if config.is_gguf:
            llama_backend = get_llama_cpp_backend()
            unsloth_backend = get_inference_backend()

            if config.gguf_hf_repo:
                from core.inference.llama_cpp import gguf_load_in_flight
                gguf_load_stack.enter_context(gguf_load_in_flight(config.gguf_hf_repo))

            # Block cache writes that would race the download manager. This runs
            # after pass-through argument inheritance so a carried --no-mmproj
            # changes the companion requirement exactly as it does for the load.
            if config.gguf_hf_repo:
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

            # Keep the resident model alive until every active generation finishes;
            # the caller's lifecycle gate blocks new starts.
            await _wait_for_model_switch_idle(current_request_counted = current_request_counted)
            # A sidecar install can reserve the gate while inference drains, after the
            # route-level checks above, so recheck before replacing either backend.
            _raise_if_sidecar_swap_in_progress()

            # Unload any active Unsloth model only after every hub conflict check.
            if unsloth_backend.active_model_name:
                logger.info(
                    f"Unloading Unsloth model '{unsloth_backend.active_model_name}' before loading GGUF"
                )
                await asyncio.to_thread(
                    unsloth_backend.unload_model, unsloth_backend.active_model_name
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
                chat_template_override = effective_chat_template_override,
                cache_type_kv = request.cache_type_kv,
                speculative_type = request.speculative_type,
                spec_draft_n_max = request.spec_draft_n_max,
                gpu_memory_mode = request.gpu_memory_mode,
                gpu_layers = request.gpu_layers,
                n_cpu_moe = request.n_cpu_moe,
                tensor_split = request.tensor_split,
                gpu_ids = effective_gpu_ids,
                n_parallel = _n_parallel,
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

            # Tensor intent for this load: the request itself, or a preserved
            # multi-GPU layer fallback carried across a reload of the SAME model that
            # doesn't drop it (e.g. a ctx-only change), so a fitting model doesn't
            # silently collapse to one GPU. Only an explicit non-tensor --split-mode
            # override counts as the drop -- the tensor field echo / unrelated extras keep
            # the preserved placement; the same-model guard stops a switch-without-unload
            # inheriting the prior model's intent.
            _explicit_tensor_drop = _is_explicit_tensor_drop(request)
            # Compare the resolved config.identifier (what load_model stores), not the
            # raw request id: from_identifier normalizes shorthands (adds unsloth/, fixes
            # case), so a reload with the shorthand would otherwise miss the match and
            # drop the carry-forward. #6659
            _same_model_loaded = (
                llama_backend.is_loaded
                and (llama_backend.model_identifier or "").lower()
                == (config.identifier or "").lower()
            )
            # model_identifier is variant-agnostic for HF repos and dir-level for a
            # local multi-variant directory, so also require the loaded quant to match
            # (path else variant, mirroring _already_in_target_state) -- otherwise a
            # different variant inherits the prior one's preserved intent. #6659
            if _same_model_loaded:
                if config.gguf_file and llama_backend.gguf_path:
                    try:
                        _same_model_loaded = (
                            Path(llama_backend.gguf_path).resolve()
                            == Path(config.gguf_file).resolve()
                        )
                    except OSError:
                        _same_model_loaded = False
                else:
                    _same_model_loaded = (llama_backend.hf_variant or "").lower() == (
                        config.gguf_variant or ""
                    ).lower()
            _tensor_intent_overall = _effective_tensor_parallel(
                extra_llama_args, request.tensor_parallel
            ) or _carry_preserved_tensor_intent(
                preserved = llama_backend.layer_preserves_tensor_intent,
                same_model = _same_model_loaded,
                explicit_drop = _explicit_tensor_drop,
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
                    # True on the layer fallback retry (tensor wanted overall but not on
                    # this attempt): keep multi-GPU. Mirrors the fallback's key.
                    preserve_multi_gpu_on_layer = bool(
                        _tensor_intent_overall
                        and not _effective_tensor_parallel(attempt_extra_args, tensor_parallel)
                    ),
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
                is_diffusion = llama_backend.is_diffusion,
                is_audio = _gguf_is_audio,
                audio_type = _gguf_audio,
                has_audio_input = llama_backend._has_audio_input,
                inference = inference_config,
                # GGUF loads via llama.cpp: auto_map never executes, so inert (matches validate_model).
                requires_trust_remote_code = False,
                context_length = llama_backend.context_length,
                max_context_length = llama_backend.max_context_length,
                native_context_length = llama_backend.native_context_length,
                supports_reasoning = llama_backend.supports_reasoning,
                reasoning_style = llama_backend.reasoning_style,
                reasoning_effort_levels = llama_backend.reasoning_effort_levels,
                reasoning_always_on = llama_backend.reasoning_always_on,
                supports_preserve_thinking = llama_backend.supports_preserve_thinking,
                supports_tools = llama_backend.supports_tools,
                cache_type_kv = llama_backend.cache_type_kv,
                chat_template = llama_backend.chat_template,
                speculative_type = llama_backend.requested_spec_mode,
                spec_draft_n_max = llama_backend.spec_draft_n_max,
                tensor_parallel = llama_backend.tensor_parallel,
                gpu_memory_mode = llama_backend.gpu_memory_mode,
                gpu_layers = llama_backend.gpu_layers,
                n_cpu_moe = llama_backend.n_cpu_moe,
                tensor_split = llama_backend.tensor_split,
                n_layers = llama_backend.n_layers,
                n_moe_layers = llama_backend.n_moe_layers,
                gpu_ids = llama_backend.gpu_ids,
            )

        # ── Standard path: load via Unsloth/transformers ──────────
        backend = get_inference_backend()

        # Unload any active GGUF model first
        llama_backend = get_llama_cpp_backend()
        await _wait_for_model_switch_idle(current_request_counted = current_request_counted)
        _raise_if_sidecar_swap_in_progress()
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
            gpu_ids = effective_gpu_ids,
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

        logger.info(
            f"Loaded model: {model_log_label if native_grant_backed else config.identifier}"
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
            is_audio = config.is_audio,
            audio_type = config.audio_type,
            has_audio_input = config.has_audio_input,
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
    from core.inference.llama_cpp import LlamaServerNotFoundError

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

        # Apply the same training coexistence policy as /load before the frontend
        # unloads the current model.
        effective_gpu_ids = request.gpu_ids if request.gpu_ids else None
        # Mirror /load: GGUF supports gpu_ids, so validate the pick (a bad one is
        # a clean 400) before the guard sizes the model against training VRAM.
        # XPU-host picks are rejected like /load (no defined mapping from the
        # picker's torch-xpu ordinals to the launcher's device spaces).
        if config.is_gguf and effective_gpu_ids is not None:
            from utils.hardware import DeviceType, get_device
            from utils.hardware.hardware import resolve_requested_gpu_ids

            if get_device() == DeviceType.XPU:
                raise HTTPException(
                    status_code = 400,
                    detail = (
                        "GPU selection (gpu_ids) is not supported on Intel XPU. "
                        "Omit gpu_ids to use all devices."
                    ),
                )
            if LlamaCppBackend._is_vulkan_backend():
                raise HTTPException(
                    status_code = 400,
                    detail = (
                        "GPU selection (gpu_ids) is not supported with a Vulkan "
                        "llama.cpp build: physical GPU ids have no defined "
                        "mapping to Vulkan device ordinals. Omit gpu_ids to use "
                        "all devices."
                    ),
                )
            try:
                resolve_requested_gpu_ids(effective_gpu_ids)
            except ValueError as exc:
                raise HTTPException(status_code = 400, detail = str(exc)) from exc
        effective_load_in_4bit = _effective_load_in_4bit(config, request.load_in_4bit)

        # Both checks cover the [adapter, base] set (matching the scan route and workers):
        # either repo can ship auto_map code or a poisoned pickle.
        security_targets = [config.identifier]
        try:
            from utils.models.model_config import get_base_model_from_lora_identifier

            # Resolve a LOCAL or REMOTE adapter's base so its code/weights are reviewed too.
            _base = get_base_model_from_lora_identifier(model_identifier, request.hf_token)
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
                    check_upgrade_for_model, _target, request.hf_token
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
            requires_trust_remote_code = any(
                _requires_trust_remote_code_for_model(_t, request.hf_token)
                for _t in security_targets
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
                latest_tier_active_for, config.identifier, request.hf_token
            ):
                effective_load_in_4bit = False
        # A metadata-only probe reads the GGUF header and allocates no VRAM, so the
        # training guard must not refuse it. Real loads omit include_context_length /
        # include_chat_template, and /load applies the guard again.
        if not (request.include_context_length or request.include_chat_template):
            # Match /load's inherited llama.cpp extras and parallel slot count so
            # validation cannot pass a smaller estimate than the subsequent load.
            effective_extra_args = _resolve_inherited_extra_args(
                request, config, model_identifier, None
            )
            # Off-loop: guard does sync nvidia-smi / HF work.
            await asyncio.to_thread(
                _guard_chat_load_against_training,
                config,
                model_identifier = model_identifier,
                hf_token = request.hf_token,
                load_in_4bit = effective_load_in_4bit,
                max_seq_length = request.max_seq_length,
                requested_gpu_ids = effective_gpu_ids,
                llama_extra_args = effective_extra_args,
                n_parallel = (
                    getattr(fastapi_request.app.state, "llama_parallel_slots", 1)
                    if fastapi_request is not None
                    else 1
                ),
                gpu_memory_mode = request.gpu_memory_mode,
            )

        # A selected GGUF loads via llama.cpp: auto_map Python and root pickle weights in a
        # mixed repo are inert for this load, so gating on them is a false positive. Only
        # run the security preflight for non-GGUF loads (requires_trust_remote_code was
        # already resolved above for the sizing flip).
        requires_security_review = False
        if not is_gguf:
            requires_security_review = any(
                _requires_security_review_for_model(_t, request.hf_token) for _t in security_targets
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

        if other_inference_request_count(current_request_counted = False, include_pending = False) > 0:
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
        backend = get_inference_backend()
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
                # Recheck under the gate: new streams bump their in-flight count while
                # holding it, so once held nothing slips past (the pre-gate check is only
                # a fast path and can be outlasted by a wait on a long /load).
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
        # model promptly. /load holds the lifecycle gate for the whole (multi-minute) load,
        # so gating first would make the cancel wait it out. cancel_load only tears the
        # loading subprocess down (no unload command), so it is safe off-gate.
        backend = get_inference_backend()
        loading = getattr(backend, "get_loading_model", lambda: None)()
        if (
            loading is not None
            and hasattr(backend, "cancel_load")
            and (request.model_path == loading or request.model_path.lower() == loading.lower())
        ):
            if await asyncio.to_thread(backend.cancel_load, request.model_path):
                note_model_unloaded()
                logger.info(f"Cancelled in-flight load: {request.model_path}")
                return UnloadResponse(status = "unloaded", model = request.model_path)

        # Same "stop loading" fast path for a still-loading GGUF (llama-server spawned,
        # health check not yet passed). A gated unload would wait out the multi-minute
        # load; unload_model() sets the cancel_event load_model polls off its own lock and
        # kills the child, sending no worker command, so it is safe off-gate like
        # cancel_load. The gated GGUF branch below handles the already-loaded case. Gate on
        # the loading model (identifier or native label): the single llama-server loads one
        # GGUF at a time, so an unload for a different model must not cancel this load.
        llama_backend = get_llama_cpp_backend()
        if (
            llama_backend.is_active
            and not llama_backend.is_loaded
            and (
                llama_backend.model_identifier == request.model_path
                or is_registered_native_path_label(
                    llama_backend.model_identifier, request.model_path
                )
            )
        ):
            await asyncio.to_thread(llama_backend.unload_model)
            note_model_unloaded()
            logger.info(f"Cancelled in-flight GGUF load: {request.model_path}")
            return UnloadResponse(status = "unloaded", model = request.model_path)

        # Serialize with /load under the same lifecycle gate: the Unsloth unload now runs
        # off the event loop (asyncio.to_thread), so without this a concurrent /load could
        # swap in a fresh subprocess mid-unload and the unload command would land on the
        # new worker. The gate makes load and unload exclusive.
        async with inference_lifecycle_gate():
            # Check if the GGUF backend has this model loaded or is loading it.
            llama_backend = get_llama_cpp_backend()
            if llama_backend.is_active and (
                llama_backend.model_identifier == request.model_path
                or is_registered_native_path_label(
                    llama_backend.model_identifier, request.model_path
                )
                or not llama_backend.is_loaded
            ):
                # A manual unload is a deliberate user action: tear down now even if a
                # request is mid-stream (only the automatic idle loop defers to it).
                llama_backend.unload_model()
                note_model_unloaded()
                logger.info(f"Unloaded GGUF model: {request.model_path}")
                return UnloadResponse(status = "unloaded", model = request.model_path)

            # Unload from Unsloth backend off the event loop: unload takes _gen_lock, which
            # a slow SSE stream paused between tokens still holds, so a sync call would block
            # the loop that drives the stream's next token and the lock release.
            backend = get_inference_backend()
            await asyncio.to_thread(backend.unload_model, request.model_path)
            note_model_unloaded()
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


@studio_router.get("/monitor")
async def get_api_monitor(current_subject: str = Depends(get_current_subject)):
    """Return recent OpenAI-compatible API activity for Unsloth."""
    active_model = _monitor_active_model()
    active_requests = api_monitor.active_count(subject = current_subject)
    if active_requests:
        operating_status = "generating"
    elif active_model:
        operating_status = "ready"
    else:
        operating_status = "idle"
    return {
        "status": operating_status,
        "active_model": active_model,
        "context_length": _monitor_context_length(),
        "active_requests": active_requests,
        "entries": api_monitor.snapshot(include_details = False, subject = current_subject),
    }


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
                    backend.reset_generation_state()
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
            backend.reset_generation_state()
            raise
        except Exception as e:
            cancel_event.set()
            backend.reset_generation_state()
            logger.error(f"Error during generation: {e}", exc_info = True)
            yield f"data: {json.dumps({'error': _friendly_error(e)})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            await _stop_local_disconnect_cancel_watcher(disconnect_watcher)
            if not completed and not cancel_event.is_set():
                cancel_event.set()
                backend.reset_generation_state()
            if gen is not None:
                try:
                    await asyncio.to_thread(gen.close)
                except (RuntimeError, ValueError):
                    pass

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
                model_identifier = None if _native_grant_backed else _model_id,
                is_vision = llama_backend.is_vision,
                is_gguf = True,
                is_diffusion = llama_backend.is_diffusion,
                gguf_variant = llama_backend.hf_variant,
                is_audio = getattr(llama_backend, "_is_audio", False),
                audio_type = _audio_type,
                has_audio_input = getattr(llama_backend, "_has_audio_input", False),
                loading = [],
                loaded = [_display_model_id] if _display_model_id else [],
                inference = _inference_cfg,
                # GGUF status: auto_map never executes, so inert (matches validate_model).
                requires_trust_remote_code = False,
                supports_reasoning = llama_backend.supports_reasoning,
                reasoning_style = llama_backend.reasoning_style,
                reasoning_effort_levels = llama_backend.reasoning_effort_levels,
                reasoning_always_on = llama_backend.reasoning_always_on,
                supports_preserve_thinking = llama_backend.supports_preserve_thinking,
                supports_tools = llama_backend.supports_tools,
                chat_template = llama_backend.chat_template,
                context_length = llama_backend.context_length,
                max_context_length = llama_backend.max_context_length,
                native_context_length = llama_backend.native_context_length,
                cache_type_kv = llama_backend.cache_type_kv,
                chat_template_override = _reported_chat_template_override,
                speculative_type = llama_backend.requested_spec_mode,
                spec_draft_n_max = llama_backend.spec_draft_n_max,
                tensor_parallel = llama_backend.tensor_parallel,
                gpu_memory_mode = llama_backend.gpu_memory_mode,
                gpu_layers = llama_backend.gpu_layers,
                n_cpu_moe = llama_backend.n_cpu_moe,
                tensor_split = llama_backend.tensor_split,
                requested_context_length = llama_backend.requested_n_ctx,
                n_layers = llama_backend.n_layers,
                n_moe_layers = llama_backend.n_moe_layers,
                gpu_ids = llama_backend.gpu_ids,
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

    # Pick backend — both return (wav_bytes, sample_rate)
    llama_backend = get_llama_cpp_backend()
    if llama_backend.is_loaded and getattr(llama_backend, "_is_audio", False):
        # Advertised repo id after an auto-switch load, else a clean public id,
        # never the absolute .gguf path.
        model_name = _llama_public_model_id(llama_backend)
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
        model_name = public_model_id(backend.active_model_name)
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
        wav_bytes, sample_rate = await asyncio.to_thread(gen)
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
# Speech-to-text (STT) sidecar  (/audio/transcribe, /audio/stt/*)
# =====================================================================


def _resolve_stt_engine(engine: Optional[str]) -> str:
    """Normalize the requested STT engine name; default is Transformers."""
    normalized = (engine or "transformers").strip().lower()
    if normalized in ("", "transformers", "whisper"):
        return "transformers"
    if normalized in ("gguf", "ggml", "whisper_cpp", "whisper.cpp"):
        return "gguf"
    raise HTTPException(
        status_code = 422,
        detail = f"Unknown STT engine '{engine}'. Use 'transformers' or 'gguf'.",
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
    return resolved


def _stt_sidecar_for(engine: str):
    if engine == "gguf":
        from core.inference.stt_ggml_sidecar import get_ggml_stt_sidecar
        return get_ggml_stt_sidecar()
    from core.inference.stt_sidecar import get_stt_sidecar
    return get_stt_sidecar()


@studio_router.get("/audio/stt/status")
async def stt_status(
    model: Optional[str] = None, current_subject: str = Depends(get_current_subject)
):
    """Report STT availability and which model, if any, is resident.

    ``model`` extends the Transformers ``downloaded_models`` check to a
    custom Hugging Face repository beyond the curated defaults.
    """
    from core.inference import stt_ggml_sidecar, stt_sidecar
    from core.inference.stt_sidecar import (
        DEFAULT_STT_MODEL,
        STT_MODELS,
        get_stt_sidecar,
        is_available,
    )

    sidecar = get_stt_sidecar()
    ggml = stt_ggml_sidecar.get_ggml_stt_sidecar()
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
    module = stt_ggml_sidecar if engine == "gguf" else stt_sidecar
    try:
        # Transformers accepts custom `owner/model` repos, so confirm the repo is
        # a Whisper checkpoint (metadata-only) before snapshot_download pulls a
        # possibly-large non-STT repo into the shared cache. Curated ids
        # short-circuit; GGUF only accepts curated ids, so it needs no check.
        if engine != "gguf":
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


@studio_router.post("/audio/stt/load")
async def stt_load(payload: SttLoadRequest, current_subject: str = Depends(get_current_subject)):
    """Load the selected STT model after the user starts local dictation."""
    from core.inference.stt_sidecar import (
        SttLoadCancelledError,
        SttModelCompatibilityError,
        SttModelIdError,
        SttModelNotDownloadedError,
        SttUnavailableError,
        get_stt_sidecar,
    )

    sidecar = _stt_sidecar_for(_resolve_serving_stt_engine(payload.engine))
    try:
        await asyncio.to_thread(sidecar.load, payload.model)
    except SttModelNotDownloadedError as e:
        raise HTTPException(status_code = 409, detail = str(e))
    except SttUnavailableError as e:
        raise HTTPException(status_code = 501, detail = str(e))
    except SttLoadCancelledError as e:
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
        engines = ["transformers", "gguf"]
    else:
        # Use the serving resolver: a "gguf" pick without whisper-server is
        # actually served by the Transformers fallback, so unload must target
        # that same engine or the resident model is never freed.
        engines = [_resolve_serving_stt_engine(engine)]
    # Attempt every engine even if one raises, so failing to unload one never
    # skips freeing the other (both can be resident after a switch).
    failed: list[str] = []
    for name in engines:
        try:
            await asyncio.to_thread(_stt_sidecar_for(name).unload)
        except Exception as exc:  # noqa: BLE001 - report after attempting all engines
            logger.warning("Failed to unload STT engine '%s': %s", name, exc)
            failed.append(name)
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


_MEMORY_CAPTURE_MONITOR_LABEL = "[memory capture]"


def _monitor_prompt_for_request(request: Request, messages: list) -> str:
    if getattr(request.state, "redact_memory_capture_monitor", False):
        return _MEMORY_CAPTURE_MONITOR_LABEL
    original = getattr(request.state, "memory_original_messages", None)
    return _monitor_prompt_from_messages(original if isinstance(original, list) else messages)


def _redact_memory_monitor_reply(request: Request) -> bool:
    return bool(getattr(request.state, "redact_memory_capture_monitor", False))


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
            method = request.method,
            model = model,
            prompt = _monitor_prompt_for_request(request, payload.messages),
            context_length = None,
            subject = current_subject,
            redact_reply = _redact_memory_monitor_reply(request),
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
                if monitor_event == "done":
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


async def _openai_chat_completions_impl(
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
    # Memory is opt-in; invalid capture sources are no-ops.
    from core.inference import memory as chat_memory

    scope = payload.memory_scope
    recall_enabled, auto_save_enabled = (
        chat_memory.get_memory_settings() if scope is not None else (False, False)
    )

    def _new_chat_cancel_event():
        cancel_event = threading.Event()
        request.state.memory_cancel_event = cancel_event
        return cancel_event

    if payload.request_purpose == "memory_capture":
        if scope is None or not auto_save_enabled:
            return JSONResponse({"id": "memory-capture-skipped", "choices": []})
        try:
            _, _, evidence = chat_memory.verify_source(scope.thread_id, scope.source_message_id)
        except chat_memory.MemoryValidationError:
            return JSONResponse({"id": "memory-capture-skipped", "choices": []})

        request.state.redact_memory_capture_monitor = bool(
            getattr(request.state, "internal_memory_capture", False)
        )
        # Rebuild this bounded private request from persisted evidence.
        payload.messages = [
            ChatMessage(role = "system", content = chat_memory.CAPTURE_SYSTEM_PROMPT),
            ChatMessage(role = "user", content = evidence),
        ]
        payload.stream = True
        payload.temperature = 0
        payload.top_p = 1
        payload.top_k = 20
        payload.min_p = 0
        payload.repetition_penalty = 1
        payload.presence_penalty = 0
        payload.max_tokens = 128
        payload.max_completion_tokens = None
        payload.n = 1
        payload.stop = None
        payload.logprobs = False
        payload.top_logprobs = None
        payload.stream_options = None
        payload.tools = None
        payload.tool_choice = "none"
        payload.enable_tools = False
        payload.enabled_tools = []
        payload.mcp_enabled = False
        payload.enable_thinking = False
        payload.reasoning_effort = "none"
        payload.thinking = None
        payload.image_base64 = None
        payload.audio_base64 = None
        payload.rag_scope = None
        payload.enable_prompt_caching = False
        payload.fast_mode = False

        # Local capture only uses the resident model; switch races are no-ops.
        if not (payload.provider_id or payload.provider_type):
            requested_model = payload.model if isinstance(payload.model, str) else None
            llama_backend = get_llama_cpp_backend()
            inference_backend = get_inference_backend()
            resident_ids = [
                getattr(llama_backend, "model_identifier", None)
                if llama_backend.is_loaded
                else None,
                getattr(llama_backend, "_openai_advertised_id", None)
                if llama_backend.is_loaded
                else None,
                getattr(inference_backend, "active_model_name", None),
            ]
            if not requested_model or not any(
                model_id_matches(requested_model, resident)
                or (
                    isinstance(resident, str)
                    and public_model_id(resident)
                    and requested_model.casefold() == public_model_id(resident).casefold()
                )
                for resident in resident_ids
            ):
                return JSONResponse({"id": "memory-capture-skipped", "choices": []})
            disable_openai_auto_switch_for_request(request.scope)

    if scope is not None:
        if payload.request_purpose == "chat" and (
            scope.allow_explicit_commands or (scope.auto_capture and auto_save_enabled)
        ):

            def _commit_deterministic_memory() -> None:
                if scope.allow_explicit_commands:
                    chat_memory.explicit_command(scope.thread_id, scope.source_message_id)
                if scope.auto_capture and chat_memory.get_memory_settings()[1]:
                    chat_memory.direct_statement(scope.thread_id, scope.source_message_id)

            request.state.memory_commit = _commit_deterministic_memory
        try:
            if scope.recall and recall_enabled:
                context = chat_memory.recall_context(
                    scope.thread_id,
                    scope.source_message_id,
                    include_ids = payload.request_purpose == "memory_capture",
                )
                if context:
                    request.state.memory_original_messages = list(payload.messages)
                    payload.messages = list(payload.messages)
                    # Keep leading instructions and cache prefixes before memory.
                    insert_at = 0
                    while insert_at < len(payload.messages) and payload.messages[
                        insert_at
                    ].role in {"system", "developer"}:
                        insert_at += 1
                    payload.messages.insert(insert_at, ChatMessage(role = "system", content = context))
        except Exception:
            # Recall fails open without logging source content.
            logger.warning("chat_memory.prepass_failed")

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
        cancel_event = getattr(request.state, "memory_cancel_event", None)
        if cancel_event is None:
            cancel_event = _new_chat_cancel_event()
        tracker = _TrackedCancel(cancel_event, payload.cancel_id, payload.session_id)
        tracker.__enter__()
        if not getattr(request.state, "skip_api_monitor", False):
            tts_monitor_id = api_monitor.start(
                endpoint = request.url.path,
                method = request.method,
                model = model_label,
                prompt = _monitor_prompt_for_request(request, payload.messages),
                context_length = context_length,
                subject = current_subject,
                redact_reply = _redact_memory_monitor_reply(request),
            )
        try:
            response = await generate_audio(payload, request)
        except asyncio.CancelledError:
            cancel_event.set()
            api_monitor.finish(tts_monitor_id, "cancelled")
            raise
        except Exception as e:
            api_monitor.fail(tts_monitor_id, _friendly_error(e))
            raise
        finally:
            tracker.__exit__(None, None, None)
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
        api_monitor.finish(tts_monitor_id, "cancelled" if cancel_event.is_set() else "completed")
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
        backend = get_inference_backend()
        if not backend.active_model_name:
            raise HTTPException(
                status_code = 400,
                detail = _no_model_loaded_detail("No model loaded. Call POST /inference/load first."),
            )
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
                method = request.method,
                model = model_name,
                prompt = _monitor_prompt_for_request(request, payload.messages),
                context_length = _monitor_context_length(),
                subject = current_subject,
                redact_reply = _redact_memory_monitor_reply(request),
            )

        # ── Audio INPUT path: decode WAV and route to audio input generation ──
        if payload.audio_base64 and model_info.get("has_audio_input"):
            try:
                audio_array = _decode_audio_base64(payload.audio_base64)
                system_prompt, chat_messages, _ = _extract_content_parts(payload.messages)
            except Exception as e:
                api_monitor.fail(monitor_id, _friendly_error(e))
                raise
            cancel_event = _new_chat_cancel_event()
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
            method = request.method,
            model = model_name,
            prompt = _monitor_prompt_for_request(request, payload.messages),
            context_length = _monitor_context_length(),
            subject = current_subject,
            redact_reply = _redact_memory_monitor_reply(request),
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
    # Route guided-decoding requests through the verbatim passthrough so
    # ``response_format`` (JSON schema) reaches llama-server and the model's
    # GBNF-constrained output comes back unmodified. The non-passthrough GGUF
    # path below calls ``generate_chat_completion`` which has no response_format
    # kwarg, so the schema gets silently dropped and data_designer falls back to
    # free-form sampling. Guided decoding does not require ``supports_tools`` --
    # the grammar machinery is independent of tool-call parsing.
    _has_response_format = _extract_response_format(payload) is not None
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
    _tools_passthrough = _supports_tool_passthrough and _has_client_tool_contract
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
    if (
        using_gguf
        and not _studio_tool_loop_requested
        and (_tools_passthrough or _has_response_format)
    ):
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

        cancel_event = _new_chat_cancel_event()
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
        _tracker = _TrackedCancel(cancel_event, *_cancel_keys)
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

        cancel_event = _new_chat_cancel_event()

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
                _openai_admission_log(
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

            _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
            _tracker = _TrackedCancel(cancel_event, *_cancel_keys)
            _tracker.__enter__()

            async def gguf_tool_stream():
                gen = None
                next_task = None
                stream_completed = False
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
                            break

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
                    _monitor_usage(monitor_id, _stream_usage, _monitor_context_length())
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
                    _openai_admission_log(
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
                                _openai_admission_log(
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
                        _openai_admission_log(
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
                        _openai_admission_log(
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
                gen = gguf_generate_with_tools()
                try:
                    for event in gen:
                        if cancel_event.is_set():
                            break
                        if event.get("type") == "metadata":
                            usage = event.get("usage")
                            finish = event.get("finish_reason")
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
                    return full_text, usage, finish
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
                    _openai_admission_log(
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
                    _openai_admission_log(
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
                full_text, completion_usage, completion_finish = await asyncio.shield(drain_task)
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
                _openai_admission_log(
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
                _openai_admission_log(
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

        if payload.stream:
            if _wants_multiple_choices(payload):
                raise _reject_unsupported_n("streaming GGUF chat completions")
            _cancel_keys = (payload.cancel_id, payload.session_id, completion_id)
            _tracker = _TrackedCancel(cancel_event, *_cancel_keys)
            _tracker.__enter__()
            try:
                reservation, admission_config = _openai_llama_admission_reserve(
                    request = request,
                    llama_backend = llama_backend,
                )
            except LlamaAdmissionQueueFull as exc:
                _tracker.__exit__(None, None, None)
                _openai_admission_log(
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
                    _monitor_usage(monitor_id, _stream_usage, _monitor_context_length())
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
                _openai_admission_log(
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
                            _openai_admission_log(
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
                    _openai_admission_log(
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
                    _openai_admission_log(
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
                _openai_admission_log(
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
            _tracker = _TrackedCancel(cancel_event, *_cancel_keys)
            _tracker.__enter__()
            admission_lease = None
            admission_wait_started_at = None
            try:
                if reservation.lease_nowait() is None:
                    admission_wait_started_at = time.monotonic()
                    _openai_admission_log(
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
                    _openai_admission_log(
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
                _openai_admission_log(
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
                _openai_admission_log(
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
                    )

                drain_task = asyncio.create_task(asyncio.to_thread(_drain_gguf_choices))
                (
                    _n,
                    _choices,
                    _monitor_replies,
                    _prompt_tokens,
                    _sum_completion,
                    _prompt_details,
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

    cancel_event = _new_chat_cancel_event()
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
        _sf_tracker = _TrackedCancel(cancel_event, *_sf_cancel_keys)
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
                        backend.reset_generation_state()
                        break
                    if await request.is_disconnected():
                        cancel_event.set()
                        backend.reset_generation_state()
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
                        backend.reset_generation_state()
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
                    _monitor_usage(monitor_id, _stats.get("usage"))
                api_monitor.finish(
                    monitor_id, "cancelled" if cancel_event.is_set() else "completed"
                )
                yield "data: [DONE]\n\n"

            except asyncio.CancelledError:
                cancel_event.set()
                backend.reset_generation_state()
                api_monitor.finish(monitor_id, "cancelled")
                raise
            except GenStreamErrorRaised as exc:
                backend.reset_generation_state()
                _msg = _friendly_gen_stream_error(exc)
                api_monitor.fail(monitor_id, _msg)
                yield _openai_stream_error_sse({"error": {"message": _msg, "type": "server_error"}})
            except Exception:
                backend.reset_generation_state()
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
                _monitor_usage(monitor_id, _stats.get("usage"))
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
            backend.reset_generation_state()
            api_monitor.finish(monitor_id, "cancelled")
            raise
        except GenStreamErrorRaised as exc:
            backend.reset_generation_state()
            _msg = _friendly_gen_stream_error(exc)
            api_monitor.fail(monitor_id, _msg)
            raise HTTPException(status_code = 500, detail = _msg)
        except HTTPException as exc:
            backend.reset_generation_state()
            api_monitor.fail(monitor_id, str(exc.detail))
            raise
        except Exception:
            backend.reset_generation_state()
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
    _sf_heal = (
        heal_gate(payload.auto_heal_tool_calls, payload.tools, payload.tool_choice)
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
        _tracker = _TrackedCancel(cancel_event, *_cancel_keys)
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
                healer = StreamToolCallHealer(_sf_heal, payload.tools) if _sf_heal else None
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
                        backend.reset_generation_state()
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
                        backend.reset_generation_state()
                        _msg = _friendly_gen_stream_error(cumulative)
                        api_monitor.fail(monitor_id, _msg)
                        yield _openai_stream_error_sse(
                            {"error": {"message": _msg, "type": "server_error"}}
                        )
                        return
                    if await request.is_disconnected():
                        cancel_event.set()
                        backend.reset_generation_state()
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
                    _monitor_usage(monitor_id, _stats.get("usage"))
                api_monitor.finish(
                    monitor_id, "cancelled" if cancel_event.is_set() else "completed"
                )
                yield "data: [DONE]\n\n"

            except asyncio.CancelledError:
                cancel_event.set()
                backend.reset_generation_state()
                api_monitor.finish(monitor_id, "cancelled")
                raise
            except GenStreamErrorRaised as exc:
                # Adapter-controlled (compare-mode) backend failure. Honor the
                # public flag so operational errors surface their real message.
                backend.reset_generation_state()
                _msg = _friendly_gen_stream_error(exc)
                api_monitor.fail(monitor_id, _msg)
                yield _openai_stream_error_sse({"error": {"message": _msg, "type": "server_error"}})
            except Exception as e:
                backend.reset_generation_state()
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
        try:
            full_text = ""
            for token in generate():
                if isinstance(token, GenStreamError):
                    backend.reset_generation_state()
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
                if heal_openai_message(_msg, _sf_heal, payload.tools):
                    _finish = "tool_calls"
                elif nudge_enabled(payload.nudge_tool_calls):
                    _data = {
                        "choices": [{"message": {"role": "assistant", "content": _visible_text}}]
                    }
                    if nudge_should_retry(_data, _sf_heal, payload.tools):
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
                            if heal_openai_message(retry_msg, _sf_heal, payload.tools):
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
                _monitor_usage(monitor_id, _stats.get("usage"))
            api_monitor.finish(monitor_id)
            return _model_json_response(response)

        except HTTPException:
            raise
        except GenStreamErrorRaised as exc:
            # Adapter-controlled (compare-mode) backend failure. Honor the public
            # flag so operational errors surface their real message.
            backend.reset_generation_state()
            _msg = _friendly_gen_stream_error(exc)
            api_monitor.fail(monitor_id, _msg)
            raise HTTPException(status_code = 500, detail = _msg)
        except Exception as e:
            backend.reset_generation_state()
            logger.error(f"Error during OpenAI completion: {e}", exc_info = True)
            api_monitor.fail(monitor_id, _friendly_error(e))
            raise HTTPException(status_code = 500, detail = safe_error_detail(e))


def _memory_stream_state(chunk) -> tuple[bool, bool]:
    """Return (done, failed) for OpenAI-compatible SSE carried by a chunk."""
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", errors = "replace")
    done = failed = False
    for line in str(chunk).splitlines():
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            done = True
            continue
        try:
            event = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            continue
        failed = failed or (isinstance(event, dict) and "error" in event)
    return done, failed


def _commit_memory_safely(commit) -> None:
    try:
        commit()
    except Exception:
        logger.warning("chat_memory.commit_failed")


def _defer_memory_commit(
    response,
    commit,
    *,
    cancel_event = None,
):
    if commit is None:
        return response

    def _cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    if isinstance(response, StreamingResponse):
        body_iterator = response.body_iterator

        async def _cancel_body_iterator() -> None:
            if cancel_event is not None:
                cancel_event.set()
            athrow = getattr(body_iterator, "athrow", None)
            if athrow is not None:
                try:
                    await athrow(asyncio.CancelledError())
                    return
                except (asyncio.CancelledError, StopAsyncIteration):
                    return
                except RuntimeError:
                    pass
            aclose = getattr(body_iterator, "aclose", None)
            if aclose is not None:
                await aclose()

        async def _commit_after_stream():
            done = failed = False
            try:
                async for chunk in body_iterator:
                    chunk_done, chunk_failed = _memory_stream_state(chunk)
                    done = done or chunk_done
                    failed = failed or chunk_failed
                    yield chunk
            except (asyncio.CancelledError, GeneratorExit):
                failed = True
                await _cancel_body_iterator()
                raise
            finally:
                if done and not failed and not _cancelled():
                    _commit_memory_safely(commit)

        response.body_iterator = _commit_after_stream()
        return response
    if 200 <= response.status_code < 300:
        original_background = response.background

        async def _commit_after_send() -> None:
            if original_background is not None:
                await original_background()
            if not _cancelled():
                _commit_memory_safely(commit)

        response.background = BackgroundTask(_commit_after_send)
    return response


async def _dispatch_openai_chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    current_subject: str,
    *,
    internal_memory_capture: bool,
):
    # Clear memory state when tests reuse a request object.
    request.state.memory_commit = None
    request.state.memory_original_messages = None

    request.state.memory_cancel_event = None
    request.state.internal_memory_capture = internal_memory_capture
    request.state.redact_memory_capture_monitor = False
    if internal_memory_capture:
        payload.request_purpose = "memory_capture"
    elif payload.request_purpose == "memory_capture":
        raise HTTPException(
            status_code = 400,
            detail = "memory_capture requests must use /v1/chat/completions/memory-capture.",
        )
    response = await _openai_chat_completions_impl(payload, request, current_subject)
    return _defer_memory_commit(
        response,
        request.state.memory_commit,
        cancel_event = request.state.memory_cancel_event,
    )


@router.post("/chat/completions")
@wraps(_openai_chat_completions_impl, assigned = ())
async def openai_chat_completions(
    payload: ChatCompletionRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    if not hasattr(request, "state"):
        return await _openai_chat_completions_impl(payload, request, current_subject)
    return await _dispatch_openai_chat_completions(
        payload,
        request,
        current_subject,
        internal_memory_capture = False,
    )


@router.post("/chat/completions/memory-capture", include_in_schema = False)
async def openai_memory_capture_completions(
    payload: ChatCompletionRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    """Run Studio's private memory extraction protocol with monitor redaction."""
    return await _dispatch_openai_chat_completions(
        payload,
        request,
        current_subject,
        internal_memory_capture = True,
    )


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
    for entry in _openai_model_objects():
        by_id[entry["id"]] = {**entry, "loaded": True}

    # Locally available (downloaded/cached) models that are not already loaded.
    # Advertise only GGUF models /v1 can actually serve (llama.cpp). GGUF-ness is
    # read from the on-disk files, not model_format: the HF-cache scanner leaves
    # model_format unset for GGUF snapshots, so a model_format filter would drop
    # every cached GGUF. The file checks run off the loop.
    from core.inference.local_model_resolver import info_has_local_gguf

    catalog = await _cached_local_catalog()
    servable = await asyncio.to_thread(lambda: [i for i in catalog if info_has_local_gguf(i)])
    for info in servable:
        cid = getattr(info, "model_id", None) or public_model_id(getattr(info, "id", None))
        if not cid or cid in by_id:
            continue
        obj = {
            "id": cid,
            "object": "model",
            "created": _created,
            "owned_by": _OWNED_BY,
            "loaded": False,
        }
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
    _loaded = _openai_model_objects()
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
    backend = get_inference_backend()
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
        raise HTTPException(
            status_code = 503,
            detail = _no_model_loaded_detail("No GGUF model loaded. Load a GGUF model first."),
        )
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
    target_url = f"{llama_backend.base_url}/v1/completions"
    is_stream = body.get("stream", False)
    prompt_text = _flatten_monitor_prompt(body.get("prompt", ""))
    monitor_id = api_monitor.start(
        endpoint = request.url.path,
        method = request.method,
        model = str(body.get("model") or _llama_public_model_id(llama_backend) or "default"),
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
            client = httpx.AsyncClient(
                timeout = _llama_streaming_generation_timeout(),
                trust_env = False,
            )
            resp = None
            bytes_iter = None
            disconnect_event = threading.Event()
            disconnect_watcher = None
            try:
                req = client.build_request(
                    "POST", target_url, json = body, headers = {"Connection": "close"}
                )
                first_token_deadline = time.monotonic() + _DEFAULT_FIRST_TOKEN_TIMEOUT_S
                resp = await _send_stream_with_preheader_cancel(client, req, request = request)
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
                await _aclose_stream_resources(
                    watchers = (disconnect_watcher,),
                    iterator = bytes_iter,
                    resp = resp,
                    client = client,
                )

        return _sse_streaming_response(_stream())
    else:
        try:
            resp = await nonstreaming_client().post(
                target_url,
                json = body,
                timeout = _llama_non_streaming_generation_timeout(),
            )
        except asyncio.CancelledError:
            api_monitor.finish(monitor_id, "cancelled")
            raise
        except Exception as e:
            api_monitor.fail(monitor_id, _friendly_error(e))
            raise

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
        raise HTTPException(
            status_code = 503,
            detail = _no_model_loaded_detail("No GGUF model loaded. Load a GGUF model first."),
        )
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
            method = request.method,
            model = str(body.get("model") or _llama_public_model_id(llama_backend) or "default"),
            prompt = prompt_text,
            context_length = llama_backend.context_length,
            subject = current_subject,
        )

    try:
        resp = await nonstreaming_client().post(
            target_url,
            json = body,
            timeout = _DEFAULT_FIRST_TOKEN_TIMEOUT_S,
        )
    except asyncio.CancelledError:
        api_monitor.finish(monitor_id, "cancelled")
        raise
    except Exception as exc:
        api_monitor.fail(monitor_id, _friendly_error(exc))
        raise
    if resp.status_code != 200:
        api_monitor.fail(monitor_id, resp.text[:500])
    else:
        try:
            _monitor_usage(monitor_id, resp.json().get("usage"), _monitor_context_length())
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
        raise HTTPException(
            status_code = 400,
            detail = _no_model_loaded_detail(
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
    try:
        reservation, admission_config = _openai_llama_admission_reserve(
            request = request,
            llama_backend = llama_backend,
        )
    except LlamaAdmissionQueueFull as exc:
        _openai_admission_log(
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
        disconnect_event = threading.Event()
        try:
            req = client.build_request(
                "POST", target_url, json = body, headers = {"Connection": "close"}
            )
            first_token_deadline = time.monotonic() + _DEFAULT_FIRST_TOKEN_TIMEOUT_S
            try:
                resp = await _send_stream_with_preheader_cancel(client, req, request = request)
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
        lease = reservation.lease_nowait()
        admission_wait_started_at = None
        stream_started = False
        stream_cancelled = False
        iterator = None
        try:
            if lease is None:
                admission_wait_started_at = time.monotonic()
                _openai_admission_log(
                    "queued",
                    reservation,
                    request = request,
                    mode = "responses_stream",
                    completion_id = resp_id,
                    level = "debug",
                )
                async for wait_item in _openai_admission_wait_stream_chunks(
                    reservation,
                    admission_config,
                    request = request,
                    cancel_event = None,
                ):
                    if isinstance(wait_item, str):
                        yield wait_item
                        continue
                    lease = wait_item
                    _openai_admission_log(
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
                cancel_event = None,
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
            _openai_admission_log(
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
            _openai_admission_log(
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
    # entries; surface them as 400 here. A `type` field marks a server-tool
    # declaration (unrecognized server tools are no-ops); anything else without
    # input_schema or name is malformed.
    for tool in tools or []:
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
        raise HTTPException(
            status_code = 503,
            detail = _no_model_loaded_detail("No GGUF model loaded. Load a GGUF model first."),
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
        for t in payload.tools or []
    )
    if requested_studio_tools and _has_client_tool:
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
    _enable_pre = _effective_enable_tools(payload)
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
        raise HTTPException(
            status_code = 503,
            detail = _no_model_loaded_detail("No GGUF model loaded. Load a GGUF model first."),
        )

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
    _enable = _effective_enable_tools(payload)
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

    # ── Client-side pass-through path ─────────────────────────
    if client_tools:
        openai_tools = openai_client_tools

        if payload.stream:
            return await _monitored_anthropic(
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
        return await _monitored_anthropic(
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
            )

        if payload.stream:
            return await _monitored_anthropic(
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
        return await _monitored_anthropic(
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
        )

    if payload.stream:
        return await _monitored_anthropic(
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
    return await _monitored_anthropic(
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
        # Last drop-branch keepalive, seeded to stream start so a chatty tool busy
        # past the stall window still gets a keepalive though its events are dropped.
        _last_drop_keepalive = time.monotonic()

        gen = run_gen()
        _next_task = None
        # Watcher to cancel on disconnect: the in-loop poll fires only between
        # events, so a mid-prefill disconnect would otherwise hold the decode slot.
        disconnect_watcher = asyncio.create_task(
            _await_disconnect_then_cancel(request, cancel_event)
        )
        try:
            while True:
                if cancel_event.is_set() or await request.is_disconnected():
                    cancel_event.set()
                    return
                # Stall keepalive (see GGUF tool stream): silent backend segments
                # must not leave the SSE stream idle past proxy timeouts.
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
                    # Tool-wrapper heartbeat -> SSE keepalive, checked BEFORE the drop
                    # skip: a dropped tool still runs server-side and its events keep the
                    # stall keepalive from firing, so dropping heartbeats would go silent.
                    yield _OPENAI_PASSTHROUGH_SSE_KEEPALIVE
                    continue
                if etype in ("tool_output", "tool_args"):
                    # Live stdout / arg streaming have no Anthropic Messages equivalent
                    # (the full call/result follow in tool_use / tool_result), so drop them.
                    # They keep the stall keepalive from firing, so a chatty tool would go
                    # silent past the ~100s proxy cap; emit a rate-limited keepalive instead.
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
                # Strip leaked tool-call XML from content events first, so a
                # content event that was purely tool XML doesn't count as text.
                # Protected helper preserves <think> rehearsal and balanced
                # [TOOL_CALLS] trailing prose (raw _TOOL_XML_RE.sub corrupts both).
                if etype == "content":
                    event = dict(event)
                    event["text"] = _strip_tool_xml_for_display(
                        event["text"],
                        auto_heal_tool_calls = True,
                        enabled_tool_names = _display_names,
                    )
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
                    # A tool_end means Unsloth executed the tool server-side, so
                    # the response no longer ends on a pending client action.
                    # Without this, a server tool that produces no trailing text
                    # would be mislabeled stop_reason "tool_use", telling the
                    # client to run a tool Unsloth already ran.
                    ends_on_tool_use = False
                elif etype == "content" and event.get("text"):
                    ends_on_tool_use = False
                for line in emitter.feed(event):
                    yield line
        except Exception as e:
            logger.error("anthropic_messages stream error: %s", e)
            # force = True so an unclassified mid-stream failure (llama-server crash,
            # decode OOM, dropped socket) still emits an SSE error and returns, instead
            # of a normal message_stop that masks a truncated turn as a clean finish.
            _error_event = _anthropic_stream_error_event(e, force = True)
            if _error_event is not None:
                yield _error_event
                return
        finally:
            await _stop_local_disconnect_cancel_watcher(disconnect_watcher)
            # Drain a still-running next(gen) worker before closing, so a mid-prefill
            # disconnect releases the thread/generator/tool resources. Closing first
            # would race into ValueError('generator already executing').
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
        emitter = AnthropicStreamEmitter()
        for line in emitter.start(message_id, model_name, input_tokens = input_tokens):
            yield line

        captured_finish_reason = None

        gen = run_gen()
        _next_task = None
        # Watcher to cancel on disconnect: the in-loop poll fires only between
        # chunks, so a mid-prefill disconnect would otherwise hold the decode slot.
        disconnect_watcher = asyncio.create_task(
            _await_disconnect_then_cancel(request, cancel_event)
        )
        try:
            while True:
                if cancel_event.is_set() or await request.is_disconnected():
                    cancel_event.set()
                    return
                # Stall keepalive (see Anthropic tool stream) each window while
                # next(gen) runs in a worker.
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
            # force = True so an unclassified mid-stream failure (llama-server crash,
            # decode OOM, dropped socket) still emits an SSE error and returns, instead
            # of a normal message_stop that masks a truncated turn as a clean finish.
            _error_event = _anthropic_stream_error_event(e, force = True)
            if _error_event is not None:
                yield _error_event
                return
        finally:
            await _stop_local_disconnect_cancel_watcher(disconnect_watcher)
            # Drain a still-running next(gen) worker before closing, so a mid-prefill
            # disconnect releases the thread/generator/model resources. Closing first
            # would race into ValueError('generator already executing').
            await _drain_pending_next_task(_next_task, cancel_event)
            if gen is not None:
                try:
                    await asyncio.to_thread(gen.close)
                except (RuntimeError, ValueError):
                    pass

        stop_reason = openai_finish_to_anthropic_stop(captured_finish_reason, had_tool_calls = False)
        for line in emitter.finish(stop_reason = stop_reason, stop_sequence = None):
            yield line

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
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "stream": stream,
    }
    if openai_tools:
        body["tools"] = openai_tools
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
        # Promote text-form tool calls (declared client tools only) into
        # tool_use blocks; verbatim behavior when healing is off or no tools.
        # tool_choice arrives here already converted to the OpenAI shape.
        _allowed_tools = heal_gate(auto_heal_tool_calls, openai_tools, tool_choice)
        if _allowed_tools:
            emitter.enable_healing(
                _allowed_tools,
                openai_tools,
                disable_parallel_tool_use = disable_parallel_tool_use,
            )
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
            timeout = _llama_streaming_generation_timeout(),
            limits = httpx.Limits(max_keepalive_connections = 0),
            trust_env = False,
        )
        resp = None
        lines_iter = None
        cancel_watcher = None
        disconnect_watcher = None
        try:
            req = client.build_request(
                "POST", target_url, json = body, headers = {"Connection": "close"}
            )
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
            # Same shape as the OpenAI passthrough: a close-time CancelledError
            # re-raised by _aclose_stream_resources must not skip the tracker exit.
            try:
                await _aclose_stream_resources(
                    watchers = (cancel_watcher, disconnect_watcher),
                    iterator = lines_iter,
                    resp = resp,
                    client = client,
                )
            finally:
                _tracker.__exit__(None, None, None)

        for line in emitter.finish():
            yield line

    return _sse_streaming_response(_stream())


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

    resp = await nonstreaming_client().post(
        target_url,
        json = body,
        timeout = _llama_non_streaming_generation_timeout(),
    )

    if resp.status_code != 200:
        raise HTTPException(
            status_code = resp.status_code,
            detail = _friendly_upstream_error(resp.text[:500]),
        )

    data = resp.json()
    # tool_choice arrives here already converted to the OpenAI shape.
    _allowed_tools = heal_gate(auto_heal_tool_calls, openai_tools, tool_choice)

    # Opt-in single-retry nudge (mirrors the OpenAI passthrough): the model
    # tried to call a tool but nothing usable came out; re-ask once with the
    # prompt prefix intact so llama-server's KV cache is reused.
    if (
        _allowed_tools
        and nudge_enabled(nudge_tool_calls)
        and nudge_should_retry(data, _allowed_tools, openai_tools)
    ):
        retry_body = {
            **body,
            "messages": [*body.get("messages", []), *nudge_messages(data, _allowed_tools)],
        }
        try:
            retry_resp = await nonstreaming_client().post(
                target_url,
                json = retry_body,
                timeout = _llama_non_streaming_generation_timeout(),
            )
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
            # Keep unpromoted bytes when healing is active; legacy stripping is
            # only for opted-out or no-client-tool requests. Protected helper (not
            # raw _TOOL_XML_RE.sub): preserves <think> rehearsal and balanced
            # [TOOL_CALLS] trailing prose, gated on the declared tools so an
            # inactive NAME[ARGS]{...} example in the final text is kept.
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

    stop_reason = openai_finish_to_anthropic_stop(finish_reason, had_tool_calls = bool(tool_calls))

    usage = data.get("usage") or {}
    return _anthropic_message_json_response(
        message_id, model_name, content_blocks, stop_reason, usage
    )


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
    tool_choice = payload.tool_choice if payload.tool_choice is not None else "auto"
    tools = payload.tools
    if payload.tool_choice == "none" and not _has_openai_tool_history(payload.messages):
        tools = None
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
    _tracker = _TrackedCancel(cancel_event, *_cancel_keys)
    _tracker.__enter__()
    try:
        reservation, admission_config = _openai_llama_admission_reserve(
            request = request,
            llama_backend = llama_backend,
        )
    except LlamaAdmissionQueueFull as exc:
        _tracker.__exit__(None, None, None)
        _openai_admission_log(
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
    _openai_admission_log(
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
                _openai_admission_log(
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
            _openai_admission_log(
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
            _openai_admission_log(
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
        try:
            task_resp = await task
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
                await _aclose_send_task(send_task)
                await _aclose_stream_resources(resp = resp, client = client)
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
                    await _aclose_stream_resources(client = client)
                finally:
                    try:
                        admission_lease.release()
                    finally:
                        _tracker.__exit__(None, None, None)
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
                # _aclose_stream_resources re-raises a close-time CancelledError
                # only after finishing teardown, and the tracker exits either way.
                try:
                    await _aclose_send_task(send_task)
                    await _aclose_stream_resources(
                        watchers = (cancel_watcher, disconnect_watcher),
                        iterator = lines_iter,
                        resp = resp,
                        client = client,
                    )
                finally:
                    try:
                        admission_lease.release()
                    finally:
                        _tracker.__exit__(None, None, None)

        async def _unstarted_cleanup() -> None:
            # Client disconnected before the body stream started, so _stream()'s
            # finally never ran. Release the eagerly-opened upstream resp/client
            # and the cancel-registry entry here; the watchers and line iterator
            # are created inside _stream(), so there is nothing else to close.
            try:
                await _aclose_send_task(send_task)
                await _aclose_stream_resources(resp = resp, client = client)
            finally:
                try:
                    admission_lease.release()
                finally:
                    _tracker.__exit__(None, None, None)

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
            await _aclose_stream_resources(resp = resp, client = client)
        finally:
            try:
                admission_lease.release()
            finally:
                _tracker.__exit__(None, None, None)
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
        _openai_admission_log(
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
            _openai_admission_log(
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
            _openai_admission_log(
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
        _openai_admission_log(
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
        _openai_admission_log(
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
            watcher.cancel()
            try:
                await watcher
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
            "messages": [*body.get("messages", []), *nudge_messages(data, _allowed_tools)],
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
