# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persistent, per-user trust_remote_code approval cache.

The consent gate (``consent.py``) pins an approval to a content fingerprint (sha256 of
every repo ``.py``). Without persistence the dialog reappears on every fresh load. This
module remembers a user's explicit approval on disk and lets the gate skip the dialog on a
later load of the SAME unchanged repo, while staying correct:

* keyed per subject -- one user's approval never auto-runs code for another;
* validated by BOTH the commit SHA (cheap, avoids re-download when unchanged) AND the
  content fingerprint (authoritative; a new/edited ``.py`` changes it). Either change
  forces re-consent;
* CRITICAL is never stored or honored (guarded on write and read);
* fail-safe: any store/SHA error degrades to "ask again", never to "auto-approve".
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from loggers import get_logger
from utils.paths import storage_roots
from utils.security.remote_code_scan import CRITICAL

logger = get_logger(__name__)

_SCHEMA_VERSION = 1
_lock = threading.RLock()

# Memoized SHA lookups, keyed (target, token-hash) so authed/unauthed reads stay separate.
_sha_cache: dict[tuple[str, str], Optional[str]] = {}


@dataclass
class StoredApproval:
    commit_sha: Optional[str]
    fingerprint: str
    max_severity: Optional[str]
    approved_at: str


def cache_disabled() -> bool:
    return os.environ.get("UNSLOTH_TRC_APPROVAL_CACHE_DISABLE", "").lower() in ("1", "true", "yes")


def _store_path():
    return storage_roots.studio_root() / "security" / "remote_code_approvals.json"


def _token_key(hf_token: Optional[str]) -> str:
    """Non-reversible discriminator; empty when no token, never the raw token."""
    return hashlib.sha256(hf_token.encode("utf-8")).hexdigest()[:12] if hf_token else ""


def _env_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes") or os.environ.get(
        "TRANSFORMERS_OFFLINE", ""
    ).lower() in ("1", "true", "yes")


def approval_target_key(targets) -> str:
    """Stable key for the combined load unit (a LoRA pins adapter + base together), using
    the same casing normalization as the fingerprint so identity never disagrees."""
    from utils.security.consent import _fingerprint_target_key

    keys = sorted(_fingerprint_target_key(t) for t in dict.fromkeys(targets) if t)
    return "\x1f".join(keys)


def _load() -> dict:
    """Parsed store, or an empty skeleton on any error (fail-safe = re-prompt)."""
    try:
        with open(_store_path()) as f:
            data = json.load(f)
        if isinstance(data, dict) and data.get("version") == _SCHEMA_VERSION:
            return data
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("Could not read remote-code approvals (%s); ignoring", exc)
    return {"version": _SCHEMA_VERSION, "subjects": {}}


def _save(data: dict) -> None:
    """Atomic write (tmp + os.replace), best-effort 0600."""
    path = _store_path()
    storage_roots.ensure_dir(path.parent)
    tmp = path.parent / f".{path.name}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent = 2)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
    except Exception as exc:
        logger.warning("Could not write remote-code approvals (%s)", exc)
        try:
            tmp.unlink(missing_ok = True)
        except OSError:
            pass


def lookup(subject: str, target_key: str) -> Optional[StoredApproval]:
    """The stored approval for (subject, target_key), or None. A CRITICAL entry (e.g. a
    hand-edited store) is refused so it can never auto-approve."""
    if not subject or cache_disabled():
        return None
    with _lock:
        entry = _load().get("subjects", {}).get(subject, {}).get(target_key)
    if not isinstance(entry, dict) or not entry.get("fingerprint"):
        return None
    if entry.get("max_severity") == CRITICAL:
        return None
    return StoredApproval(
        commit_sha = entry.get("commit_sha"),
        fingerprint = entry["fingerprint"],
        max_severity = entry.get("max_severity"),
        approved_at = entry.get("approved_at", ""),
    )


def record(
    subject: str,
    target_key: str,
    *,
    commit_sha: Optional[str],
    fingerprint: str,
    max_severity: Optional[str],
) -> None:
    """Persist a user's explicit approval. CRITICAL is never stored."""
    if not subject or not fingerprint or cache_disabled() or max_severity == CRITICAL:
        return
    with _lock:
        data = _load()
        data.setdefault("subjects", {}).setdefault(subject, {})[target_key] = {
            "commit_sha": commit_sha,
            "fingerprint": fingerprint,
            "max_severity": max_severity,
            "approved_at": datetime.now(timezone.utc).isoformat(),
        }
        _save(data)


def forget(subject: str, target_key: str) -> None:
    """Drop an approval (e.g. the user declined / discarded the download)."""
    if not subject:
        return
    with _lock:
        data = _load()
        if data.get("subjects", {}).get(subject, {}).pop(target_key, None) is not None:
            _save(data)


def clear() -> None:
    """Test helper: drop the on-disk store and the SHA memo."""
    with _lock:
        try:
            _store_path().unlink(missing_ok = True)
        except OSError:
            pass
        _sha_cache.clear()


def resolve_commit_sha(target: str, hf_token: Optional[str] = None) -> Optional[str]:
    """Immutable HF commit SHA for *target*, or None for a local path / offline / error.
    None forces the gate onto the authoritative fingerprint check (never fail-open)."""
    from utils.paths import is_local_path

    cache_key = (target, _token_key(hf_token))
    if cache_key in _sha_cache:
        return _sha_cache[cache_key]
    sha: Optional[str] = None
    try:
        if not is_local_path(target) and not _env_offline():
            from huggingface_hub import HfApi
            sha = HfApi().model_info(target, token = hf_token).sha
    except Exception as exc:
        logger.debug("Could not resolve commit sha for '%s': %s", target, exc)
        sha = None
    _sha_cache[cache_key] = sha
    return sha


def resolve_combined_sha(targets, hf_token: Optional[str] = None) -> Optional[str]:
    """Combined SHA for the load unit; None if ANY target's SHA is unresolvable (all or
    nothing), so a partial result can never satisfy the SHA fast path."""
    from utils.security.consent import _fingerprint_target_key

    parts = []
    for target in dict.fromkeys(targets):
        if not target:
            continue
        sha = resolve_commit_sha(target, hf_token)
        if sha is None:
            return None
        parts.append(f"{_fingerprint_target_key(target)}={sha}")
    return "\x1f".join(sorted(parts)) if parts else None
