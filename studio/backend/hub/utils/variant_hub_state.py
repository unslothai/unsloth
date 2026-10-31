# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The last Hub listing per repo, kept beside the cache.

A collapsed On Device row answers from disk (no Hub round-trip), so the only place this is read is
an expander. When the Hub cannot answer, the listing it published last time is still the best
answer available — it is what the repo said when it was last reachable.

Records hold Hub-derived fields only. ``downloaded``, ``partial``, sizes of the files on disk and
cache paths always come from the cache walk, so a stale record can never claim bytes that were
deleted or hide bytes that were downloaded. A record is stamped with its observation time and the
credential and endpoint it was read with, and is discarded when any of those no longer match
(another account sees another variant set) or when it is older than ``MAX_AGE_SECONDS``.

Callers mark the merged answer ``cached`` rather than ``live`` so the UI does not offer an update
or a companion download the unreachable Hub cannot carry out.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

from hub.utils.state_dir import repo_cache_basename, state_root

logger = get_logger(__name__)

_SUBDIR = "variant-hub"
_SCHEMA_VERSION = 1
# Old enough that the repo may have moved on; the record stops being worth showing as "last known".
MAX_AGE_SECONDS = 24 * 3600
_MAX_FILE_BYTES = 4 * 1024 * 1024
# The fields a record may carry. Anything the disk answers is refused here, not just omitted by the
# writer, so a hand-edited or future record cannot smuggle local state into a Hub answer.
_HUB_ONLY_VARIANT_FIELDS = (
    "quant",
    "filename",
    "update_available",
    "pending_drafter_filename",
    "pending_drafter_size_bytes",
    "download_size_bytes",
)


def _identity(hf_token: Optional[str]) -> str:
    """Anonymous is its own identity, so a gated listing is never served to a signed-out caller."""
    token = hf_token.strip() if isinstance(hf_token, str) else ""
    if not token:
        return "anon"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _endpoint() -> str:
    try:
        from utils.utils import hf_endpoint_url

        return hf_endpoint_url().rstrip("/")
    except Exception:  # noqa: BLE001 - a mirror switch only misses the cache
        return "https://huggingface.co"


def _record_path(repo_id: str) -> Optional[Path]:
    root = state_root(create = False)
    if root is None:
        return None
    try:
        name = repo_cache_basename("model", repo_id)
    except ValueError:
        return None
    if len(name) > 200:
        # Repo ids that long are not real; hashing keeps a path within filesystem limits.
        name = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return root / _SUBDIR / f"{name}.json"


def record_listing(repo_id: str, response: Any, *, hf_token: Optional[str] = None) -> None:
    """Store the Hub-derived fields of a live ``GgufVariantsResponse``. Best effort, never raises."""
    variants = []
    for variant in getattr(response, "variants", None) or ():
        entry = {}
        for field in _HUB_ONLY_VARIANT_FIELDS:
            value = getattr(variant, field, None)
            if value is not None:
                entry[field] = value
        if entry.get("quant"):
            variants.append(entry)
    payload = {
        "version": _SCHEMA_VERSION,
        "repo_id": repo_id,
        "observed_at": time.time(),
        "identity": _identity(hf_token),
        "endpoint": _endpoint(),
        "has_vision": bool(getattr(response, "has_vision", False)),
        "default_variant": getattr(response, "default_variant", None),
        "variants": variants,
    }
    path = _record_path(repo_id)
    if path is None:
        return
    tmp = None
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid():x}")
        tmp.write_text(
            json.dumps(payload, ensure_ascii = False, separators = (",", ":")), encoding = "utf-8"
        )
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("Could not store the Hub listing for %s: %s", repo_id, exc)
        try:
            if tmp is not None:
                tmp.unlink(missing_ok = True)
        except OSError:
            pass


def stored_listing(
    repo_id: str,
    *,
    hf_token: Optional[str] = None,
    max_age_seconds: float = MAX_AGE_SECONDS,
    now: Optional[float] = None,
) -> Optional[dict]:
    """The record for ``repo_id`` if it still applies to this caller, else None.

    A record from another credential or another Hub endpoint is not this caller's answer, and an
    expired one is not worth showing.
    """
    path = _record_path(repo_id)
    if path is None:
        return None
    try:
        if path.stat().st_size > _MAX_FILE_BYTES:
            return None
        payload = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != _SCHEMA_VERSION:
        return None
    if payload.get("repo_id") != repo_id:
        return None
    if payload.get("identity") != _identity(hf_token) or payload.get("endpoint") != _endpoint():
        return None
    observed_at = payload.get("observed_at")
    if not isinstance(observed_at, (int, float)):
        return None
    age = (now if now is not None else time.time()) - float(observed_at)
    if age < 0 or age > max_age_seconds:
        return None
    if not isinstance(payload.get("variants"), list):
        return None
    return payload


def merged_response(response: Any, stored: dict) -> Any:
    """``response`` (answered from disk) with the stored Hub-only fields applied per quant.

    Disk facts stay as they are: only the update verdict, the pending companion and the size of the
    download an update would fetch are taken from the record.
    """
    by_quant = {}
    for entry in stored.get("variants") or ():
        if not isinstance(entry, dict):
            continue
        quant = entry.get("quant")
        if isinstance(quant, str) and quant:
            by_quant[quant.strip().lower()] = entry
    if not by_quant:
        return response
    variants = []
    for variant in getattr(response, "variants", None) or ():
        entry = by_quant.get(str(getattr(variant, "quant", "") or "").strip().lower())
        if entry is None:
            variants.append(variant)
            continue
        update: dict = {}
        if "update_available" in entry:
            update["update_available"] = bool(entry["update_available"])
        if "pending_drafter_filename" in entry:
            update["pending_drafter_filename"] = entry["pending_drafter_filename"]
        if "pending_drafter_size_bytes" in entry:
            update["pending_drafter_size_bytes"] = int(entry["pending_drafter_size_bytes"] or 0)
        if "download_size_bytes" in entry:
            update["download_size_bytes"] = int(entry["download_size_bytes"] or 0)
        variants.append(variant.model_copy(update = update) if update else variant)
    stored_default = stored.get("default_variant")
    return response.model_copy(
        update = {
            "variants": variants,
            "has_vision": bool(stored.get("has_vision", response.has_vision)),
            "default_variant": (
                stored_default if isinstance(stored_default, str) and stored_default else response.default_variant
            ),
            "hub_state": "cached",
            "observed_at": float(stored["observed_at"]),
        }
    )
