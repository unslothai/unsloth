# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Audio errors that are safe to show a client verbatim.

``safe_error_detail`` flattens everything to "An internal error occurred", right for a
failure but wrong for a capability answer, where the reason is the whole reply. These
carry no path or user input, so the route passes them straight through.
"""

from __future__ import annotations

import re


AUDIO_UNSUPPORTED_CODE = "audio_unsupported_backend"
AUDIO_RUNTIME_ERROR_CODE = "audio_runtime_error"

_MAX_RUNTIME_DETAIL_CHARS = 300
# A URL is matched to keep it (its tokens are redact_log_text's job); else absolute/UNC/file:// paths.
_ABSOLUTE_PATH_RE = re.compile(
    r"(?P<url>\b(?:https?|wss?|ftp)://[^\s\"'`,;]+)"
    r"|(?<![\w/])(?:file://)?(?:/|[A-Za-z]:[\\/]|\\\\)[^\s\"'`,;]+"
)


class AudioGenerationCancelledError(RuntimeError):
    """The generation was stopped rather than failing.

    A bare RuntimeError hit the route's generic handler, so an idle auto-unload
    interrupting a Speak reported HTTP 500 instead of a retryable cancellation.
    """


class AudioBackendUnsupportedError(RuntimeError):
    """The model loaded fine; this backend cannot do this audio task.

    No retry, shorter input or freed memory helps.
    """

    def __init__(
        self,
        detail: str,
        *,
        hint: str | None = None,
    ):
        self.detail = detail
        self.hint = hint
        super().__init__(detail if not hint else f"{detail} {hint}")

    @property
    def message(self) -> str:
        return self.args[0] if self.args else self.detail


class AudioRuntimeError(RuntimeError):
    """The audio runtime refused or failed a request and said why ("CosyVoice3 requires reference
    audio"). ``status`` is the runtime's HTTP status, when it answered with one.

    The text is the runtime's own, so it can name a path or echo a token: send it to a client only
    through ``audio_runtime_http_error``.
    """

    def __init__(
        self,
        detail: str,
        *,
        status: int | None = None,
    ):
        self.detail = detail
        self.status = status
        super().__init__(detail)


def _redacted_line(text: str) -> str:
    from utils.log_redaction import redact_log_text

    cleaned = redact_log_text(str(text or ""))
    cleaned = _ABSOLUTE_PATH_RE.sub(_path_tail, cleaned)
    return " ".join(cleaned.split())


def sanitize_runtime_detail(text: str) -> str:
    """One bounded line of runtime error text with credentials and filesystem paths removed."""
    cleaned = _redacted_line(text)
    if len(cleaned) > _MAX_RUNTIME_DETAIL_CHARS:
        cleaned = cleaned[: _MAX_RUNTIME_DETAIL_CHARS - 3].rstrip() + "..."
    return cleaned


def sanitize_runtime_tail(text: str, limit: int = 280) -> str:
    """The last ``limit`` characters of a runtime log, redacted before the cut so a path or token
    straddling it is never shown half-redacted."""
    cleaned = _redacted_line(text)
    if len(cleaned) > limit:
        cleaned = "..." + cleaned[-(limit - 3) :].split(" ", 1)[-1]
    return cleaned


def _path_tail(match: "re.Match[str]") -> str:
    if match.group("url"):
        return match.group("url")
    tail = match.group(0).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return tail or "..."


def audio_runtime_http_error(
    error: AudioRuntimeError, fallback: str = "An internal error occurred"
) -> tuple[int, str]:
    """A runtime 4xx is the request's fault, so it maps to 400 rather than an auth or routing code."""
    status = error.status
    if status == 503:
        code = 503
    elif status is not None and 400 <= status < 500:
        code = 400
    else:
        code = 500
    return code, sanitize_runtime_detail(error.detail) or fallback
