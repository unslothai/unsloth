# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turn a forwarded route's failure into an MCP ToolError the agent can act on. Unsloth Studio routes fail in several shapes (``{"detail"}`` on /api, the OpenAI envelope on /v1, gpu_busy three ways, a 200 that carries ``_deferred_error`` or ``status: "error"``, an NDJSON error line), and every message is scrubbed of host paths before it leaves.

A failure the agent can branch on starts with a tag, ``[<code> k=v] <message>``, such as ``[gpu_busy retry_after=12]`` or ``[not_loaded kind=image]``. A plain refusal has no tag."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

import httpx
from fastmcp.exceptions import ToolError

from hub.utils.host_paths import redact_paths_in_text
from studio_mcp.forward import parse_json

# Route errors sometimes tell a browser client which route to call next ("Call POST /inference/load
# first."). The path scrub turns that route into <path>, and an MCP agent cannot call routes anyway.
_ROUTE_CLAUSE = re.compile(
    r"\s*\b(?:Call|Use|See) (?:GET|POST|PUT|PATCH|DELETE) /[\w/{}.:-]*(?: first)?\.(?=\s|$)"
)
# The load and unload routes offer a force_cancel_active flag the MCP tools do not have.
_FORCE_CANCEL_CLAUSE = re.compile(r",? or retry with force_cancel_active(?=[.\s]|$)")
_FORCE_CANCEL_SENTENCE = re.compile(r"\s*[^.]*\bforce_cancel_active\b[^.]*\.?")


def _scrub(text: Any) -> str:
    if isinstance(text, str):
        text = _ROUTE_CLAUSE.sub("", text)
        if "force_cancel_active" in text:
            text = _FORCE_CANCEL_SENTENCE.sub("", _FORCE_CANCEL_CLAUSE.sub("", text))
    return redact_paths_in_text(text)


_MAX_MESSAGE_CHARS = 2000
_DEFERRED_ERROR_KEY = "_deferred_error"
REFUSAL_HEADER = "x-unsloth-refusal"
MEMORY_REFUSAL = "memory-estimate"
MULTI_MODEL_REFUSAL = "multi-model-always"
# Copy the Keep multiple models loaded refusal uses when no header or code came with it.
MULTI_MODEL_MARKERS = ("is set to Always", "is set to keep loaded")

_TAG_CODE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_TAG_VALUE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

# The kind a refusal names, as unload_model takes it.
UNLOAD_KINDS = {"chat": "llm", "audio": "audio", "image": "image", "video": "video"}
MULTI_MODEL_HINT = "Call unload_model(kind='{kind}') to free the GPU; retrying won't help."
MULTI_MODEL_HINT_ANY = (
    "Call unload_model(kind=…) for the model it names to free the GPU; retrying won't help."
)
ACTIVE_GENERATIONS_HINT = (
    "Wait for the running chats to finish, or cancel a chat you started with"
    " cancel(kind='chat', id=<its cancel_id>)."
)
MEMORY_HINT = "Pass allow_oversized=true to try anyway."
# Guidance for a route's own error code, used in place of the tool's status hint. Empty for now.
CODE_HINTS: dict[str, str] = {}

_UNLOAD_KIND = re.compile(r"would unload the (\w+) model")
_NOT_LOADED = re.compile(r"^No (?:(diffusion|video) )?model (?:is )?loaded")
# The kind a bare "No model loaded" is about, by the route that said it. First match wins.
_NOT_LOADED_PATHS = (
    ("/v1/audio/transcriptions", "stt"),
    ("/api/inference/audio/stt/", "stt"),
    ("/v1/audio/", "audio"),
    ("/api/inference/audio/", "audio"),
    ("/v1/images/", "image"),
    ("/api/inference/images/", "image"),
    ("/v1/videos", "video"),
    ("/api/inference/video/", "video"),
)

# llama-server's startup output, as core/inference/llama_cpp.py _with_startup_diagnostics frames it.
_LLAMA_OUTPUT = "\n\nllama-server output:\n"
_LLAMA_LOG = "\n\nFull log:"
_LLAMA_ERRORS = "\n\nllama-server errors:\n"
_ERROR_LINE = re.compile(
    r"error|fail|exception|abort|unable|cannot|out of memory|not enough|invalid|missing",
    re.IGNORECASE,
)


def _brief(text: Any) -> Any:
    """Keep llama-server's error lines, not its whole startup log, so the cause fits the cap."""
    if not isinstance(text, str):
        return text
    text = text.split(_LLAMA_LOG, 1)[0]
    head, found, output = text.partition(_LLAMA_OUTPUT)
    if not found:
        return text
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    lines = ([line for line in lines if _ERROR_LINE.search(line)] or lines[-5:])[-8:]
    room = _MAX_MESSAGE_CHARS - 40 - len(head) - len(_LLAMA_ERRORS)
    while len(lines) > 1 and len("\n".join(lines)) > room:
        lines.pop(0)
    return f"{head}{_LLAMA_ERRORS}{chr(10).join(lines)}" if lines else head


def _tag(code: Any, **fields: Any) -> Optional[str]:
    """``code k=v``, or None for a code that is not a plain identifier. A bad value drops its pair."""
    if not isinstance(code, str) or not _TAG_CODE.match(code):
        return None
    pairs = [
        f"{key}={value}"
        for key, value in fields.items()
        if value is not None and not isinstance(value, bool) and _TAG_VALUE.match(str(value))
    ]
    return " ".join([code, *pairs])


def tool_error(
    message: Any,
    hint: Optional[str] = None,
    *,
    tag: Optional[str] = None,
) -> ToolError:
    """A scrubbed ToolError. ``hint`` is the tool's own guidance and goes on after the scrub, which would otherwise read a route like /v1/models as a path and drop it. ``tag`` (``code k=v``) leads the message in brackets."""
    text = _scrub(_brief(message)).strip() or "Unsloth Studio returned an error"
    if len(text) > _MAX_MESSAGE_CHARS:
        text = text[:_MAX_MESSAGE_CHARS] + "…"
    if hint:
        text = f"{text} {hint}"
    if tag and _TAG_CODE.match(tag.split(" ", 1)[0]):
        text = f"[{tag}] {text}"
    return ToolError(text)


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii = False, default = str)


def _validation_summary(errors: list) -> str:
    parts = []
    for error in errors[:10]:
        if not isinstance(error, dict):
            parts.append(_text(error))
            continue
        loc = [str(part) for part in error.get("loc") or () if part != "body"]
        msg = _text(error.get("msg") or "invalid")
        parts.append(f"{'.'.join(loc)}: {msg}" if loc else msg)
    return "Invalid arguments: " + "; ".join(parts)


def _gpu_busy(body: Any) -> Optional[tuple[str, Any]]:
    """(message, retry_after) for any of the three gpu_busy shapes."""
    if not isinstance(body, dict):
        return None
    candidates = [body, body.get("detail"), body.get("error")]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("error") == "gpu_busy" or candidate.get("code") == "gpu_busy":
            return _text(candidate.get("message")), candidate.get("retry_after")
    return None


def _detail_message(detail: Any) -> str:
    if isinstance(detail, list):
        return _validation_summary(detail)
    if isinstance(detail, dict):
        for key in ("message", "detail", "error"):
            if isinstance(detail.get(key), str) and detail[key]:
                return detail[key]
    return _text(detail)


def _refusal(headers: Mapping[str, str]) -> str:
    return str(headers.get(REFUSAL_HEADER) or "").strip().lower()


def _not_loaded_kind(message: str, path: str) -> Optional[str]:
    match = _NOT_LOADED.match(message.strip())
    if match is None:
        return None
    if match.group(1):
        return "image" if match.group(1) == "diffusion" else "video"
    return next((kind for prefix, kind in _NOT_LOADED_PATHS if path.startswith(prefix)), "chat")


def _describe(
    status: int,
    body: Any,
    headers: Mapping[str, str],
    path: str = "",
) -> tuple[Optional[str], str, Optional[str]]:
    """(tag, message, own hint). An own hint replaces the tool's status hint."""
    # Scrubbed before the hints go on: the redactor drops everything after a path it finds.
    retry_after = headers.get("retry-after")
    busy = _gpu_busy(body)
    if busy is not None:
        message, body_retry = busy
        seconds = retry_after or body_retry
        hint = f" Retry after {seconds} s." if seconds is not None else ""
        tag = _tag("gpu_busy", retry_after = seconds)
        return tag, f"GPU busy: {_scrub(message).rstrip('.')}.{hint}", ""

    detail = body.get("detail") if isinstance(body, dict) else None
    envelope = body.get("error") if isinstance(body, dict) else None
    code = None
    if isinstance(body, dict) and "detail" in body:
        if isinstance(detail, list):
            return None, _scrub(_validation_summary(detail)), None
        message = _detail_message(detail)
        if isinstance(detail, dict):
            code = detail.get("code")
    elif isinstance(envelope, dict):
        message = _text(envelope.get("message"))
        # /v1 validation is a 400 whose envelope names the offending field.
        code = envelope.get("code")
        if status == 400 and envelope.get("param"):
            return _tag(code), f"Invalid arguments: {_scrub(message)}", None
    elif isinstance(body, dict) and isinstance(body.get("message"), str):
        message = body["message"]
    else:
        message = _text(body)
    message = _brief(message)
    scrubbed = _scrub(message)
    suffix = f"{scrubbed} (HTTP {status})"

    if (
        _refusal(headers) == MULTI_MODEL_REFUSAL
        or code == "multi_model_always"
        or (status == 409 and any(marker in message for marker in MULTI_MODEL_MARKERS))
    ):
        # Retrying cannot help: the setting keeps the other model loaded until it is unloaded.
        found = detail.get("kind") if isinstance(detail, dict) else None
        if not isinstance(found, str):
            match = _UNLOAD_KIND.search(message)
            found = match.group(1) if match else None
        kind = UNLOAD_KINDS.get(found or "")
        if kind is None:
            return _tag("multi_model_always"), suffix, MULTI_MODEL_HINT_ANY
        return _tag("multi_model_always", kind = found), suffix, MULTI_MODEL_HINT.format(kind = kind)

    if isinstance(detail, dict) and "active_generations" in (detail.get("error"), code):
        # The route's thread_ids are browser conversation ids; an agent can only cancel its own chats.
        tag = _tag("active_generations", running = detail.get("running"))
        return tag, suffix, ACTIVE_GENERATIONS_HINT

    if status == 400 and _refusal(headers) == MEMORY_REFUSAL:
        return _tag("memory_estimate"), f"{scrubbed} {MEMORY_HINT}", None

    tag = _tag(code)
    if tag is None:
        kind = _not_loaded_kind(message, path)
        tag = _tag("not_loaded", kind = kind) if kind else None
    if status == 503 and retry_after:
        suffix += f" Retry after {retry_after} s."
    return tag, suffix, CODE_HINTS.get(code) if isinstance(code, str) else None


def _body(resp: httpx.Response) -> Any:
    try:
        return parse_json(resp.content)
    except (ValueError, UnicodeDecodeError):
        return resp.text.strip()


def _request_path(resp: httpx.Response) -> str:
    # A Response built by hand (tests) has no request, and .request raises then.
    try:
        return resp.request.url.path
    except RuntimeError:
        return ""


def raise_for_payload(payload: Any, *, path: str = "") -> Any:
    """Raise for an error carried inside a successful response; return the payload otherwise. ``path`` is the route that answered, for the tag's kind."""
    if isinstance(payload, dict):
        deferred = payload.get(_DEFERRED_ERROR_KEY)
        if deferred is not None:
            # /load and /unload commit a 200 to keep a tunnel open, then report a late failure here.
            deferred = deferred if isinstance(deferred, dict) else {"detail": deferred}
            status = deferred.get("status_code")
            status = status if isinstance(status, int) else 500
            # Headers are lost on a padded answer unless the route copied them in.
            sent = deferred.get("headers")
            sent = sent if isinstance(sent, dict) else {}
            headers = {str(key).lower(): str(value) for key, value in sent.items()}
            tag, message, hint = _describe(
                status, {"detail": deferred.get("detail")}, headers, path
            )
            raise tool_error(message, hint, tag = tag)
        if payload.get("type") == "error":
            nested = payload.get("error")
            message = payload.get("message") or (
                nested.get("message") if isinstance(nested, dict) else nested
            )
            raise tool_error(_text(message))
    return payload


def raise_for_status_field(payload: Any) -> Any:
    """``/api/train/start`` refuses with a 200 and ``status: "error"``."""
    if isinstance(payload, dict) and payload.get("status") == "error":
        raise tool_error(_text(payload.get("message") or payload.get("error")))
    return payload


def raise_for_route(
    resp: httpx.Response,
    *,
    payload: Any = None,
    hints: Optional[Mapping[int, str]] = None,
) -> Any:
    """Raise a ToolError when the forwarded call failed; otherwise return its parsed JSON (or ``payload`` when the caller already parsed it, e.g. the last NDJSON line). ``hints`` maps a status to what the agent should do about it; an answer with its own guidance (gpu_busy, Keep multiple models loaded, running chats) keeps that instead."""
    path = _request_path(resp)
    if resp.status_code >= 400:
        tag, message, own = _describe(resp.status_code, _body(resp), resp.headers, path)
        hint = own if own is not None else (hints or {}).get(resp.status_code)
        raise tool_error(message, hint or None, tag = tag)
    if payload is None:
        try:
            payload = parse_json(resp.content) if resp.content.strip() else None
        except (ValueError, UnicodeDecodeError):
            return None
    return raise_for_payload(payload, path = path)
