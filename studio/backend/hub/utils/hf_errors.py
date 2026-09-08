# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Map Hugging Face Hub client-side errors to HTTP status codes."""

from __future__ import annotations

from typing import Optional


def hf_error_status(exc: Exception) -> Optional[int]:
    # Client-side HF errors should surface as 4xx, not a generic 500.
    name = type(exc).__name__
    if name in (
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "EntryNotFoundError",
        # "Could not reach the Hub, and the cache could not answer either." That is a
        # missing resource from the caller's side, not a server fault. Both became
        # reachable once cache reads started requiring authorization: a caller refused
        # the cache falls through to a Hub that is offline or unreachable, and the pair
        # below is what comes back. They were landing on the generic 500.
        "LocalEntryNotFoundError",
        "OfflineModeIsEnabled",
    ):
        return 404
    if name == "GatedRepoError":
        return 403
    if name == "HFValidationError":
        return 400
    # HfHubHTTPError subclasses carry the upstream response status.
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(code, int) and 400 <= code < 500:
        return code
    return None
