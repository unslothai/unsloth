# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Map Hugging Face Hub client-side errors to HTTP status codes and user-facing text."""

from __future__ import annotations

import re
from typing import Optional

_NOT_ON_MODELSCOPE = re.compile(
    r"[\w.\-]+/[\w.\-]+ is not on ModelScope\. Switch the model source to Hugging Face in Settings to use it\."
)


def not_on_modelscope(repo: str) -> str:
    return f"{repo} is not on ModelScope. Switch the model source to Hugging Face in Settings to use it."


def modelscope_missing(error: object, repo: Optional[str] = None) -> Optional[str]:
    """datasets drops the missing-repo sentence, so a ModelScope DatasetNotFoundError stands in for it."""
    links: dict[int, object] = {}
    pending = [error]
    while pending:
        link = pending.pop()
        if link is not None and id(link) not in links:
            links[id(link)] = link
            pending += [getattr(link, "__cause__", None), getattr(link, "__context__", None)]
    for link in links.values():
        found = _NOT_ON_MODELSCOPE.search(str(link))
        if found:
            return found.group(0)
    if repo and any(type(link).__name__ == "DatasetNotFoundError" for link in links.values()):
        from utils.hub_settings import MODELSCOPE, active_source
        if active_source() == MODELSCOPE:
            return not_on_modelscope(repo)
    return None


def hf_error_status(exc: Exception) -> Optional[int]:
    name = type(exc).__name__
    if name in (
        "RepositoryNotFoundError",
        "RevisionNotFoundError",
        "EntryNotFoundError",
    ):
        return 404
    # Unreachable/offline is transient, not 404: openai_auto_download caches 404 for ten minutes.
    if name in ("LocalEntryNotFoundError", "OfflineModeIsEnabled"):
        return 503
    if name == "GatedRepoError":
        return 403
    if name == "HFValidationError":
        return 400
    code = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(code, int) and 400 <= code < 500:
        return code
    return None
