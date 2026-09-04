# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe optional Hub files without caching missing entries.

``hf_hub_download`` can leave a ref pointing to a snapshot that is not on disk after a 404, causing
``scan_cache_dir`` to omit the repo. Metadata requests answer the same question without changing
the cache.
"""

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
    """Return True only when the Hub confirms that *filename* is absent.

    Offline, authentication, rate-limit, and resolution failures return False so callers preserve
    their existing download and error handling.
    """
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError
    except Exception:  # noqa: BLE001 - an unimportable hub is the caller's problem, not ours
        return False

    try:
        url = hf_hub_url(
            repo_id = repo_id,
            filename = filename,
            repo_type = repo_type,
            revision = revision,
        )
        get_hf_file_metadata(url, token = token)
    except LocalEntryNotFoundError:
        return False
    except EntryNotFoundError:
        return True
    except Exception as exc:  # noqa: BLE001 - no answer is not an answer of "absent"
        logger.debug("Could not probe %s for %s: %s", repo_id, filename, exc)
        return False
    return False
