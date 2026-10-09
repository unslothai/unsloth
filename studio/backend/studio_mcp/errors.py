# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turn a forwarded route's failure into an MCP ToolError the agent can act on. Studio routes fail in several shapes (``{"detail"}`` on /api, the OpenAI envelope on /v1, gpu_busy three ways, a 200 that carries ``_deferred_error`` or ``status: "error"``, an NDJSON error line), and every message is scrubbed of host paths before it leaves."""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

import httpx
from fastmcp.exceptions import ToolError

from hub.utils.host_paths import redact_paths_in_text

_MAX_MESSAGE_CHARS = 2000
_DEFERRED_ERROR_KEY = "_deferred_error"
REFUSAL_HEADER = "x-unsloth-refusal"
MEMORY_REFUSAL = "memory-estimate"


def tool_error(message: Any, hint: Optional[str] = None) -> ToolError:
    """A scrubbed ToolError. ``hint`` is the tool's own guidance and goes on after the scrub, which would otherwise read a route like /v1/models as a path and drop it."""
    text = redact_paths_in_text(message).strip() or "Studio returned an error"
    if len(text) > _MAX_MESSAGE_CHARS:
        text = text[:_MAX_MESSAGE_CHARS] + "…"
    return ToolError(f"{text} {hint}" if hint else text)


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


def _describe(status: int, body: Any, headers: Mapping[str, str]) -> str:
    # Scrubbed before the hints go on: the redactor drops everything after a path it finds.
    retry_after = headers.get("retry-after")
    busy = _gpu_busy(body)
    if busy is not None:
        message, body_retry = busy
        seconds = retry_after or body_retry
        hint = f" Retry after {seconds} s." if seconds is not None else ""
        return f"GPU busy: {redact_paths_in_text(message).rstrip('.')}.{hint}"

    if isinstance(body, dict) and "detail" in body:
        if isinstance(body["detail"], list):
            return redact_paths_in_text(_validation_summary(body["detail"]))
        message = _detail_message(body["detail"])
    elif isinstance(body, dict) and isinstance(body.get("error"), dict):
        envelope = body["error"]
        message = redact_paths_in_text(_text(envelope.get("message")))
        # /v1 validation is a 400 whose envelope names the offending field.
        if status == 400 and envelope.get("param"):
            return f"Invalid arguments: {message}"
        if envelope.get("code"):
            message = f"{message} ({envelope['code']})"
    elif isinstance(body, dict) and isinstance(body.get("message"), str):
        message = body["message"]
    else:
        message = _text(body)

    message = redact_paths_in_text(message)
    if status == 400 and headers.get(REFUSAL_HEADER) == MEMORY_REFUSAL:
        return f"{message} Pass allow_oversized=true to try anyway."
    message = f"{message} (HTTP {status})"
    if status == 503 and retry_after:
        message += f" Retry after {retry_after} s."
    return message


def _body(resp: httpx.Response) -> Any:
    try:
        return json.loads(resp.content.strip())
    except (ValueError, UnicodeDecodeError):
        return resp.text.strip()


def raise_for_payload(payload: Any) -> Any:
    """Raise for an error carried inside a successful response; return the payload otherwise."""
    if isinstance(payload, dict):
        deferred = payload.get(_DEFERRED_ERROR_KEY)
        if deferred is not None:
            # /load and /unload commit a 200 to keep a tunnel open, then report a late failure here.
            deferred = deferred if isinstance(deferred, dict) else {"detail": deferred}
            status = deferred.get("status_code")
            status = status if isinstance(status, int) else 500
            raise tool_error(_describe(status, {"detail": deferred.get("detail")}, {}))
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
    """Raise a ToolError when the forwarded call failed; otherwise return its parsed JSON (or ``payload`` when the caller already parsed it, e.g. the last NDJSON line). ``hints`` maps a status to what the agent should do about it; a GPU-busy answer keeps its retry hint instead."""
    if resp.status_code >= 400:
        body = _body(resp)
        hint = None if _gpu_busy(body) is not None else (hints or {}).get(resp.status_code)
        raise tool_error(_describe(resp.status_code, body, resp.headers), hint)
    if payload is None:
        try:
            payload = json.loads(resp.content.strip()) if resp.content.strip() else None
        except (ValueError, UnicodeDecodeError):
            return None
    return raise_for_payload(payload)
