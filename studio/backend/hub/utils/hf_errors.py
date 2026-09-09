# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Map Hugging Face Hub client-side errors to HTTP status codes."""

from __future__ import annotations

from typing import Optional


def hf_error_status(exc: Exception) -> Optional[int]:
    # Client-side HF errors should surface as the status they mean, not a generic 500.
    name = type(exc).__name__
    if name in (
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "EntryNotFoundError",
    ):
        return 404
    # "I could not ask" is not "it is not there": these two say the Hub was unreachable or
    # offline was forced, which is transient and retryable. Answering 404 told callers the
    # repository was missing, and openai_auto_download caches that verdict for ten minutes.
    if name in ("LocalEntryNotFoundError", "OfflineModeIsEnabled"):
        return 503
    if name == "GatedRepoError":
        return 403
    if name == "HFValidationError":
        return 400
    # HfHubHTTPError subclasses carry the upstream response status.
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(code, int) and 400 <= code < 500:
        return code
    return None
