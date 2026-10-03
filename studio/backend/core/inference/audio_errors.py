# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Audio errors that are safe to show a client verbatim.

``safe_error_detail`` flattens everything to "An internal error occurred", right for a
failure but wrong for a capability answer, where the reason is the whole reply. These
carry no path or user input, so the route passes them straight through.
"""

from __future__ import annotations

import re


# Tagged on the worker's audio_error payload so the parent recognises the case without matching on prose.
AUDIO_UNSUPPORTED_CODE = "audio_unsupported_backend"
# Tagged the same way for a request the audio runtime answered with an error of its own.
AUDIO_RUNTIME_ERROR_CODE = "audio_runtime_error"

# Long enough for a runtime sentence ("CosyVoice3 requires reference audio"), short enough for a toast.
_MAX_RUNTIME_DETAIL_CHARS = 300
# An absolute POSIX, drive-letter or UNC path. Not after a word character, ':' or '/', so a URL's
# "//host" and a "family:name" pair stay as written.
_ABSOLUTE_PATH_RE = re.compile(r"(?<![\w:/])(?:/|[A-Za-z]:[\\/]|\\\\)[^\s\"'`,;]+")


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


def sanitize_runtime_detail(text: str) -> str:
    """One bounded line of runtime error text with credentials and filesystem paths removed."""
    from utils.log_redaction import redact_log_text

    cleaned = redact_log_text(str(text or ""))
    cleaned = _ABSOLUTE_PATH_RE.sub(_path_tail, cleaned)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > _MAX_RUNTIME_DETAIL_CHARS:
        cleaned = cleaned[: _MAX_RUNTIME_DETAIL_CHARS - 3].rstrip() + "..."
    return cleaned


def _path_tail(match: "re.Match[str]") -> str:
    tail = match.group(0).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return tail or "..."


def audio_runtime_http_error(
    error: AudioRuntimeError, fallback: str = "An internal error occurred"
) -> tuple[int, str]:
    """The HTTP status and client-safe ``detail`` for a runtime error.

    A runtime 4xx (a refused input or option) is the request's fault and becomes 400, never a
    401/403/404 the client would read as its own auth or routing; a busy runtime stays 503;
    everything else is 500, as before, but with the runtime's reason instead of the fallback.
    """
    status = error.status
    if status == 503:
        code = 503
    elif status is not None and 400 <= status < 500:
        code = 400
    else:
        code = 500
    return code, sanitize_runtime_detail(error.detail) or fallback
