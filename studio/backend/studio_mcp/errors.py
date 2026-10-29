# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Route failures as MCP ToolError, host paths scrubbed; leading [code k=v] tag marks actionable codes"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

import httpx
from fastmcp.exceptions import ToolError

from hub.utils.host_paths import redact_paths_in_text
from studio_mcp import problems
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

_TAG_CODE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_TAG_VALUE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

MEMORY_HINT = "Pass allow_oversized=true to try anyway."

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
    code = tag.split(" ", 1)[0] if tag else ""
    # A message the route already led with its own [code] tag keeps just that one.
    if code and _TAG_CODE.match(code) and not text.startswith(f"[{code}"):
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


def _describe(
    status: int,
    body: Any,
    headers: Mapping[str, str],
    tools: Optional[frozenset[str]] = None,
) -> tuple[Optional[str], str, Optional[str]]:
    """(tag, message, own hint). An own hint replaces the tool's status hint. ``tools`` are the
    caller's, so a hint names only tools it has; None reads them from the current caller."""
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
            if code is None and detail.get("error") == "download_busy":
                seconds = detail.get("retry_after")
                hint = f" Retry after {seconds} s." if seconds is not None else ""
                tag = _tag("download_busy", retry_after = seconds)
                return tag, f"Another speech-to-text download is running.{hint}", ""
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
    problem = problems.from_headers(headers)
    if problem is None and (
        _refusal(headers) == MULTI_MODEL_REFUSAL or code == "multi_model_always"
    ):
        found = detail.get("kind") if isinstance(detail, dict) else None
        problem = problems.Problem(
            "multi_model_always", {"kind": found} if isinstance(found, str) else {}
        )
    if (
        problem is None
        and isinstance(detail, dict)
        and "active_generations" in (detail.get("error"), code)
    ):
        # The route's thread_ids are browser conversation ids; an agent cancels only its own chats.
        problem = problems.Problem("active_generations", {"running": detail.get("running")})
    if problem is None and status == 400 and _refusal(headers) == MEMORY_REFUSAL:
        return _tag("memory_estimate"), f"{scrubbed} {MEMORY_HINT}", None
    if problem is None:
        if status == 503 and retry_after:
            suffix += f" Retry after {retry_after} s."
        return _tag(code), suffix, None
    # Retrying a multi_model_always refusal cannot help, so it never gets the 503 retry line.
    if status == 503 and retry_after and problem.code != "multi_model_always":
        suffix += f" Retry after {retry_after} s."
    fields = (
        {"running": problem.fields.get("running")}
        if problem.code == "active_generations"
        else problem.tag_fields()
    )
    hint = problems.hint(problem, problems.caller_tools() if tools is None else tools)
    return _tag(problem.code, **fields), suffix, hint


def _body(resp: httpx.Response) -> Any:
    try:
        return parse_json(resp.content)
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
            # Headers are lost on a padded answer unless the route copied them in.
            sent = deferred.get("headers")
            sent = sent if isinstance(sent, dict) else {}
            headers = {str(key).lower(): str(value) for key, value in sent.items()}
            tag, message, hint = _describe(status, {"detail": deferred.get("detail")}, headers)
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
    if resp.status_code >= 400:
        tag, message, own = _describe(resp.status_code, _body(resp), resp.headers)
        hint = own if own is not None else (hints or {}).get(resp.status_code)
        raise tool_error(message, hint or None, tag = tag)
    if payload is None:
        try:
            payload = parse_json(resp.content) if resp.content.strip() else None
        except (ValueError, UnicodeDecodeError):
            return None
    return raise_for_payload(payload)


# Audio failures carry a code (detail.code on /api, error.code on /v1); each gets what to do next.
def _audio_failure(body: Any) -> Optional[dict]:
    if not isinstance(body, dict):
        return None
    for holder in (body.get("detail"), body.get("error")):
        if isinstance(holder, dict) and isinstance(holder.get("code"), str):
            return holder if holder["code"] in problems.AUDIO_CODES else None
    return None


def audio_code(body: Any) -> Optional[str]:
    """The audio failure code a route's error body names, if it is one MCP has a hint for."""
    failure = _audio_failure(body)
    return failure["code"] if failure is not None else None


def audio_hint(body: Any, tools: Optional[frozenset[str]] = None) -> Optional[str]:
    failure = _audio_failure(body)
    if failure is None:
        return None
    names = []
    for row in failure.get("smaller_variants") or []:
        name = row.get("variant") if isinstance(row, dict) else row
        if isinstance(name, str) and name:
            names.append(name)
    tools = problems.caller_tools() if tools is None else tools
    return problems.audio_hint(failure["code"], names, tools)


def raise_for_audio(resp: httpx.Response, *, hints: Optional[Mapping[int, str]] = None) -> Any:
    """``raise_for_route`` for an audio route: a coded failure starts with its ``[code]`` tag and ends with what to do about it."""
    body = _body(resp) if resp.status_code >= 400 else None
    code = audio_code(body)
    if code is None or _gpu_busy(body) is not None:
        return raise_for_route(resp, hints = hints)
    try:
        return raise_for_route(resp, hints = {**(hints or {}), resp.status_code: audio_hint(body)})
    except ToolError as exc:
        # The /v1 envelope repeats the code after the message; the tag already names it.
        text = str(exc).replace(f" ({code})", "", 1)
        raise ToolError(text if text.startswith("[") else f"[{code}] {text}") from None
