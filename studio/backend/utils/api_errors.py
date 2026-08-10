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


def _summarize_error_input(value):
    """Return a JSON-safe stand-in for a validation error's ``input`` value."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(value))} bytes of binary data>"
    if isinstance(value, str) and len(value) > _MAX_ECHOED_INPUT_CHARS:
        return value[:_MAX_ECHOED_INPUT_CHARS] + f"... (truncated, {len(value)} chars)"
    if isinstance(value, dict):
        return {k: _summarize_error_input(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_summarize_error_input(v) for v in value]
    return value


def safe_validation_errors(errors) -> list:
    """FastAPI's ``exc.errors()`` with every ``input`` made JSON-encodable."""
    safe = []
    for err in errors:
        if not isinstance(err, dict):
            safe.append(err)
            continue
        cleaned = dict(err)
        if "input" in cleaned:
            cleaned["input"] = _summarize_error_input(cleaned["input"])
        # ctx can carry the triggering exception object, which is not JSON either.
        ctx = cleaned.get("ctx")
        if isinstance(ctx, dict):
            cleaned["ctx"] = {
                k: (v if isinstance(v, (str, int, float, bool, type(None))) else str(v))
                for k, v in ctx.items()
            }
        safe.append(cleaned)
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
            summary, param = _summarize_validation_errors(exc.errors())
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
