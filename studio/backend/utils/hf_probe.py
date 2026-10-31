# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe optional Hub files via metadata, since hf_hub_download can leave a dangling ref after a 404."""

from __future__ import annotations

from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)


def hf_file_definitely_absent(
    repo_id: str,
    filename: str,
    *,
    repo_type: str = "model",
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """True only when the Hub confirms absence; any other failure returns False so existing handling
    runs."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

        from hub.utils.hf_tokens import call_with_anonymous_retry
    except Exception:  # noqa: BLE001 - an unimportable hub is the caller's problem, not ours
        return False

    try:
        url = hf_hub_url(
            repo_id = repo_id,
            filename = filename,
            repo_type = repo_type,
            revision = revision,
        )
        # A refused credential is not an answer about the file: ask once more without it.
        call_with_anonymous_retry(lambda sent: get_hf_file_metadata(url, token = sent), token)
    except LocalEntryNotFoundError:
        return False
    except EntryNotFoundError:
        return True
    except Exception as exc:  # noqa: BLE001 - no answer is not an answer of "absent"
        logger.debug("Could not probe %s for %s: %s", repo_id, filename, exc)
        return False
    return False
