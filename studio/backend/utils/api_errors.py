# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Error-envelope helpers for the OpenAI/Anthropic-compatible ``/v1/*`` API surface.

FastAPI's defaults emit ``{"detail": ...}`` bodies (status 422 for validation,
``exc.status_code`` for ``HTTPException``). Real OpenAI/Anthropic clients expect
provider-specific error envelopes instead, so this module re-wraps Unsloth's own
client-error responses on the ``/v1/*`` surface:

- OpenAI surface (``/v1/chat/completions``, ``/v1/completions``, ``/v1/models``,
  ``/v1/responses``, ``/v1/embeddings``, ...)::

      {"error": {"message": str, "type": str, "param": None|str, "code": None|str}}

- Anthropic surface (any path starting with ``/v1/messages``)::

      {"type": "error", "error": {"type": str, "message": str}}

CRITICAL: the exception handlers installed by :func:`install_api_error_handlers`
are global, but they ONLY transform responses for paths that start with ``/v1/``.
For every other path (``/api/...``, frontend routes) they reproduce FastAPI's
default behavior byte-for-byte, because the Unsloth frontend depends on the
``{"detail": ...}`` shape for ``/api/*``.

Public contract (other modules depend on these):

- ``OPENAI_TYPE_BY_STATUS`` / ``ANTHROPIC_TYPE_BY_STATUS``: status -> type maps.
- ``openai_error_body(message, *, status=400, err_type=None, code=None, param=None)``
- ``anthropic_error_body(message, *, status=400, err_type=None)``
- ``is_anthropic_path(path)``
- ``error_body_for_path(path, message, *, status, err_type=None, code=None, param=None)``
- ``install_api_error_handlers(app)``
"""

import math
import re
from itertools import islice

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from fastapi.utils import is_body_allowed_for_status_code
from starlette.exceptions import HTTPException as StarletteHTTPException


# Status-code -> error ``type`` string for the OpenAI error envelope.
OPENAI_TYPE_BY_STATUS = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    409: "conflict_error",
    413: "invalid_request_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "api_error",
}

# Status-code -> error ``type`` string for the Anthropic error envelope.
ANTHROPIC_TYPE_BY_STATUS = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    409: "conflict_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "api_error",
    529: "overloaded_error",
}


def openai_error_body(
    message,
    *,
    status = 400,
    err_type = None,
    code = None,
    param = None,
) -> dict:
    """Build an OpenAI-style error envelope.

    Returns ``{"error": {"message", "type", "param", "code"}}``. The ``param``
    and ``code`` keys are always present (value may be ``None``). ``err_type``
    defaults to :data:`OPENAI_TYPE_BY_STATUS` for ``status`` (``"api_error"``
    fallback).
    """
    return {
        "error": {
            "message": str(message),
            "type": err_type or OPENAI_TYPE_BY_STATUS.get(status, "api_error"),
            "param": param,
            "code": code,
        }
    }


def anthropic_error_body(
    message,
    *,
    status = 400,
    err_type = None,
) -> dict:
    """Build an Anthropic-style error envelope.

    Returns ``{"type": "error", "request_id": None, "error": {"type", "message"}}``.
    ``request_id`` is a required (nullable) field on the spec's ErrorResponse;
    Unsloth has no request-id system, so it is null. ``err_type`` defaults to
    :data:`ANTHROPIC_TYPE_BY_STATUS` for ``status`` (``"api_error"`` fallback).
    """
    return {
        "type": "error",
        "request_id": None,
        "error": {
            "type": err_type or ANTHROPIC_TYPE_BY_STATUS.get(status, "api_error"),
            "message": str(message),
        },
    }


def is_anthropic_path(path: str) -> bool:
    """True iff ``path`` belongs to the Anthropic surface (``/v1/messages*``)."""
    return path.startswith("/v1/messages")


def wants_api_error_envelope(path: str) -> bool:
    """True for the OpenAI/Anthropic-compatible surfaces: the ``/v1/*`` mount and
    the preview ``/p/<run>[/<ckpt>]/v1/*`` mount."""
    return path.startswith("/v1/") or (path.startswith("/p/") and "/v1/" in path)


def error_body_for_path(
    path,
    message,
    *,
    status,
    err_type = None,
    code = None,
    param = None,
) -> dict:
    """Dispatch to the correct envelope builder based on ``path``.

    Anthropic surface paths use :func:`anthropic_error_body` (``code``/``param``
    are not part of that envelope and are ignored); all other ``/v1/*`` paths use
    :func:`openai_error_body`.
    """
    if is_anthropic_path(path):
        return anthropic_error_body(message, status = status, err_type = err_type)
    return openai_error_body(message, status = status, err_type = err_type, code = code, param = param)


def _summarize_validation_errors(errors) -> tuple:
    """Derive a readable one-line message and (optional) body param from ``exc.errors()``.

    Returns ``(summary, param)``. ``summary`` is a human-readable string like
    ``"messages: Field required"``. ``param`` is the offending body field name when
    one can be extracted (used as the OpenAI envelope ``param``), else ``None``.

    Malformed-JSON bodies surface here as ``type == "json_invalid"`` and get a
    dedicated message.
    """
    if not errors:
        return "Invalid request", None

    first = errors[0]
    if first.get("type") == "json_invalid":
        return "Invalid JSON in request body", None

    loc = first.get("loc", ()) or ()
    msg = first.get("msg", "Invalid request")

    # Extract the body field name (the loc element after a leading "body").
    param = None
    loc_parts = [p for p in loc if p not in ("body",)]
    if loc and loc[0] == "body" and loc_parts:
        # First non-"body" element that is a field name (string).
        for part in loc_parts:
            if isinstance(part, str):
                param = part
                break

    label = ".".join(str(p) for p in loc_parts) if loc_parts else ".".join(str(p) for p in loc)
    summary = f"{label}: {msg}" if label else str(msg)
    return summary, param


# A validation error carries the offending value under "input". For a JSON-body route
# handed a non-JSON body (a multipart upload posted to /api/inference/audio/transcribe
# is the case that surfaced this) that value is the whole raw payload, and
# jsonable_encoder renders bytes with ``o.decode()``, which raises UnicodeDecodeError on
# any binary. The handler then failed, turning a 422 into a 500 whose traceback embedded
# the escaped payload: one 531 KB upload produced a single 2.2 MB log line.
#
# Clients only need loc/msg/type, so the input is summarized rather than echoed. That
# also stops a large but perfectly decodable body from being mirrored back and logged.
_MAX_ECHOED_INPUT_CHARS = 200
# A body that is a huge container of small values is just as unbounded as one huge
# string: a JSON route handed an array of 200k integers would otherwise have every
# element copied into the 422 body. Keep enough to identify the payload, drop the rest.
_MAX_ECHOED_ITEMS = 20
_MAX_ECHOED_DEPTH = 4


def _truncate_text(value: str) -> str:
    if len(value) > _MAX_ECHOED_INPUT_CHARS:
        value = value[:_MAX_ECHOED_INPUT_CHARS] + f"... (truncated, {len(value)} chars)"
    # A JSON body may legally contain a lone surrogate ("\ud800"), which survives
    # parsing but cannot be UTF-8 encoded; Starlette's JSONResponse encodes with
    # ensure_ascii = False, so echoing one turns the 422 back into a 500.
    if _LONE_SURROGATE_RE.search(value):
        value = _LONE_SURROGATE_RE.sub(lambda m: f"\\u{ord(m.group()):04x}", value)
    return value


# Digits, not characters: str() on a very large int raises above sys.get_int_max_str_digits(),
# and json.dumps would emit every digit otherwise.
_MAX_ECHOED_INT_DIGITS = 100
_LONE_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")


def _summarize_int(value: int) -> object:
    if -(10**_MAX_ECHOED_INT_DIGITS) < value < 10**_MAX_ECHOED_INT_DIGITS:
        return value
    # bit_length, not str(): str() is what raises above the digit limit.
    return f"<integer with about {value.bit_length() * 3 // 10} digits>"


def _summarize_error_input(value, depth: int = 0):
    """Return a JSON-safe, size-bounded stand-in for an error's ``input`` value."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} bytes of binary data>"
    if isinstance(value, str):
        return _truncate_text(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return _summarize_int(value)
    if isinstance(value, float) and not math.isfinite(value):
        # NaN and Infinity survive jsonable_encoder but Starlette's JSONResponse
        # dumps with allow_nan = False, so echoing one turns the 422 into a 500.
        return repr(value)
    if isinstance(value, dict):
        if depth >= _MAX_ECHOED_DEPTH:
            return f"<dict with {len(value)} keys>"
        # islice, not a slice of items(): a 10 MB object should not be materialized
        # into a list just to keep the first 20 entries. A key can be arbitrarily
        # long too, so it gets the same budget as a value.
        out = {
            _truncate_text(k) if isinstance(k, str) else k: _summarize_error_input(v, depth + 1)
            for k, v in islice(value.items(), _MAX_ECHOED_ITEMS)
        }
        if len(value) > _MAX_ECHOED_ITEMS:
            out["..."] = f"({len(value) - _MAX_ECHOED_ITEMS} more keys)"
        return out
    if isinstance(value, (list, tuple)):
        if depth >= _MAX_ECHOED_DEPTH:
            return f"<sequence of {len(value)} items>"
        out = [_summarize_error_input(v, depth + 1) for v in islice(value, _MAX_ECHOED_ITEMS)]
        if len(value) > _MAX_ECHOED_ITEMS:
            out.append(f"... ({len(value) - _MAX_ECHOED_ITEMS} more items)")
        return out
    return value


# One error dictionary per rejected array element is normal for a route that validates
# each item, so the count itself is unbounded even when every entry is tiny.
_MAX_ECHOED_ERRORS = 20


def safe_validation_errors(errors) -> list:
    """FastAPI's ``exc.errors()`` with every ``input`` made JSON-encodable."""
    safe = []
    total = len(errors) if hasattr(errors, "__len__") else None
    for err in islice(errors, _MAX_ECHOED_ERRORS):
        if not isinstance(err, dict):
            safe.append(err)
            continue
        cleaned = dict(err)
        # A typed mapping puts the offending key straight into loc (CreateResearchRun
        # has budgets: dict[str, int]), so loc is user-controlled and unbounded too.
        loc = cleaned.get("loc")
        if isinstance(loc, (list, tuple)):
            cleaned["loc"] = [
                _truncate_text(part) if isinstance(part, str) else part
                for part in islice(loc, _MAX_ECHOED_ITEMS)
            ]
        if "input" in cleaned:
            cleaned["input"] = _summarize_error_input(cleaned["input"])
        # A validator that quotes the submitted value reaches "msg" too: models/
        # training.py's _parse_lr raises f"... (got {v!r})", so a megabyte-long
        # learning_rate would come back in full even with "input" summarized.
        if isinstance(cleaned.get("msg"), str):
            cleaned["msg"] = _truncate_text(cleaned["msg"])
        # ctx can carry the triggering exception object, which is not JSON either,
        # and whose str() quotes the same value.
        ctx = cleaned.get("ctx")
        if isinstance(ctx, dict):
            cleaned["ctx"] = {
                k: (v if isinstance(v, (int, float, bool, type(None))) else _truncate_text(str(v)))
                for k, v in ctx.items()
            }
        safe.append(cleaned)
    if total is not None and total > _MAX_ECHOED_ERRORS:
        safe.append(
            {
                "type": "too_many_errors",
                "loc": [],
                "msg": f"... ({total - _MAX_ECHOED_ERRORS} more validation errors omitted)",
            }
        )
    return safe


def install_api_error_handlers(app) -> None:
    """Register validation + HTTPException handlers that emit ``/v1/*`` envelopes.

    Both handlers are global but only transform responses for OpenAI/Anthropic-
    compatible surfaces (see :func:`wants_api_error_envelope`: the ``/v1/*`` mount
    and the preview ``/p/.../v1/*`` mount). Every other path reproduces FastAPI's
    default ``{"detail": ...}`` behavior exactly so the Unsloth frontend keeps working.
    """

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(request, exc):
        path = request.url.path
        if wants_api_error_envelope(path):
            # Same sanitizing as the 422 branch: /v1 builds its message from msg,
            # and a validator that quotes the submitted value (models/inference.py
            # embeds an unsupported block's type with btype!r) makes msg unbounded.
            summary, param = _summarize_validation_errors(safe_validation_errors(exc.errors()))
            return JSONResponse(
                status_code = 400,
                content = error_body_for_path(path, summary, status = 400, param = param),
            )
        # Default FastAPI behavior for every other path, minus the raw input echo
        # (see safe_validation_errors: encoding it raised and turned 422 into 500).
        return JSONResponse(
            status_code = 422,
            content = {"detail": jsonable_encoder(safe_validation_errors(exc.errors()))},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(request, exc):
        path = request.url.path
        headers = getattr(exc, "headers", None)
        # Statuses like 204/304/1xx must not carry a body — mirror FastAPI's
        # default http_exception_handler, which returns a bodiless Response.
        if not is_body_allowed_for_status_code(exc.status_code):
            return Response(status_code = exc.status_code, headers = headers)
        if wants_api_error_envelope(path):
            detail = exc.detail
            # Already a fully-formed envelope: pass through untouched.
            if isinstance(detail, dict) and ("error" in detail or detail.get("type") == "error"):
                return JSONResponse(
                    status_code = exc.status_code,
                    content = detail,
                    headers = headers,
                )
            # A dict carrying our individual fields.
            if isinstance(detail, dict):
                message = detail.get("message", detail)
                err_type = detail.get("type")
                code = detail.get("code")
                param = detail.get("param")
            else:
                # Plain message string (the common HTTPException case).
                message = detail
                err_type = None
                code = None
                param = None
            return JSONResponse(
                status_code = exc.status_code,
                content = error_body_for_path(
                    path,
                    message,
                    status = exc.status_code,
                    err_type = err_type,
                    code = code,
                    param = param,
                ),
                headers = headers,
            )
        # Default FastAPI behavior for every other path.
        return JSONResponse(
            status_code = exc.status_code,
            content = {"detail": exc.detail},
            headers = headers,
        )
