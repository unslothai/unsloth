# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Earlier blobs a GGUF repo declares equivalent to its current files.

A repo can replace a file with new bytes under the same name (for example a different GGUF layout
of the same weights) and list the old blob in ``unsloth_alternates.json`` at its root:

    {"alternates": {"Model-UD-Q4_K_XL.gguf": ["<sha256 of the earlier blob>", ...]}}

A user who already has a listed blob is current, so the Hub inventory does not offer an update.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from loggers import get_logger

logger = get_logger(__name__)

ALTERNATES_FILENAME = "unsloth_alternates.json"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TTL_SECONDS = 600.0
_cache: dict[tuple[str, str], tuple[float, dict[str, frozenset[str]]]] = {}
_lock = threading.Lock()


def parse_alternates(data: object) -> dict[str, frozenset[str]]:
    """``{repo path: earlier sha256s}`` from the manifest; anything malformed is dropped."""
    table = data.get("alternates") if isinstance(data, dict) else None
    if not isinstance(table, dict):
        return {}
    out: dict[str, frozenset[str]] = {}
    for path, hashes in table.items():
        if not isinstance(path, str) or not isinstance(hashes, list):
            continue
        valid = frozenset(
            h.lower() for h in hashes if isinstance(h, str) and _SHA256_RE.match(h.lower())
        )
        if valid:
            out[path.replace("\\", "/").lstrip("/")] = valid
    return out


def repo_alternate_blobs(
    repo_id: str, hf_token, siblings: Optional[Iterable]
) -> dict[str, frozenset[str]]:
    """The repo's declared alternates, or {} when its listing has no manifest or it cannot be read.
    Fetched only when ``siblings`` (the listing already in hand) names the manifest."""
    sibling = next(
        (s for s in siblings or () if getattr(s, "rfilename", None) == ALTERNATES_FILENAME), None
    )
    if sibling is None:
        return {}
    key = (repo_id, str(getattr(sibling, "blob_id", None) or ""))
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit is not None and now - hit[0] < _TTL_SECONDS:
            return hit[1]
    try:
        from huggingface_hub import hf_hub_download

        from hub.utils.hf_tokens import call_hub_with_anonymous_retry
        from utils.hf_cache_settings import get_hf_cache_paths

        local = call_hub_with_anonymous_retry(
            hf_hub_download,
            hf_token,
            repo_id = repo_id,
            filename = ALTERNATES_FILENAME,
            cache_dir = str(get_hf_cache_paths().hub_cache),
            etag_timeout = 10,
        )
        result = parse_alternates(json.loads(Path(local).read_text(encoding = "utf-8")))
    except Exception as exc:  # noqa: BLE001 -- no manifest means the old behaviour
        logger.debug("%s unreadable for %s: %s", ALTERNATES_FILENAME, repo_id, exc)
        result = {}
    with _lock:
        _cache[key] = (now, result)
    return result
