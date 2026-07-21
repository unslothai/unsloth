# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persistent, per-user trust_remote_code approval cache.

Remembers a user's explicit approval so the consent gate can skip only the DIALOG on a
later load of the SAME unchanged code. The gate ALWAYS re-scans (the cache never skips the
scan), so CRITICAL is hard-blocked every time and a hand-edited store cannot auto-approve
malicious code. Keyed per subject; honored only when the content fingerprint matches AND
the scanner-rules version matches AND (when resolvable) the commit SHA matches. CRITICAL is
never stored or honored, and any store/SHA error degrades to "ask again", never auto-approve.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from loggers import get_logger
from utils.paths import storage_roots
from utils.security.remote_code_scan import CRITICAL, SCAN_RULES_VERSION

logger = get_logger(__name__)

_SCHEMA_VERSION = 1
_lock = threading.RLock()

# Re-exported so the gate can compare a stored approval's ruleset to the live one.
SCANNER_VERSION = SCAN_RULES_VERSION


@dataclass
class StoredApproval:
    commit_sha: Optional[str]
    fingerprint: str
    max_severity: Optional[str]
    approved_at: str
    scanner_version: int = 0


def cache_disabled() -> bool:
    return os.environ.get("UNSLOTH_TRC_APPROVAL_CACHE_DISABLE", "").lower() in (
        "1",
        "true",
        "yes",
    )


def _store_path():
    return storage_roots.studio_root() / "security" / "remote_code_approvals.json"


def _env_offline() -> bool:
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in (
        "1",
        "true",
        "yes",
    ) or os.environ.get("TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes")


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
        # Validate the shape, not just the version: a hand-edited ``subjects`` that is not a
        # dict (e.g. ``[]``) would otherwise crash lookup/record instead of failing safe.
        if (
            isinstance(data, dict)
            and data.get("version") == _SCHEMA_VERSION
            and isinstance(data.get("subjects"), dict)
        ):
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


@contextlib.contextmanager
def _file_lock():
    """Best-effort cross-process exclusive lock over the store. Inference/export/training
    record approvals from separate subprocesses, so the in-process RLock is not enough: two
    processes could each read the same JSON and clobber the other's entry on ``os.replace``.
    Holding this around the read-modify-write serializes them. Degrades to a no-op if OS
    locking is unavailable (the consequence is only an occasional extra prompt)."""
    path = _store_path()
    try:
        storage_roots.ensure_dir(path.parent)
        fd = os.open(
            str(path.parent / f"{path.name}.lock"), os.O_CREAT | os.O_RDWR, 0o600
        )
    except Exception:
        yield
        return
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
        except Exception:
            pass  # locking unavailable; the thread lock still applies
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                with contextlib.suppress(Exception):
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def lookup(subject: str, target_key: str) -> Optional[StoredApproval]:
    """The stored approval for (subject, target_key), or None. A CRITICAL entry (e.g. a
    hand-edited store) is refused so it can never seed an approval."""
    if not subject or cache_disabled():
        return None
    with _lock:
        subj = _load().get("subjects", {}).get(subject, {})
    entry = subj.get(target_key) if isinstance(subj, dict) else None
    if not isinstance(entry, dict) or not entry.get("fingerprint"):
        return None
    if entry.get("max_severity") == CRITICAL:
        return None
    return StoredApproval(
        commit_sha = entry.get("commit_sha"),
        fingerprint = entry["fingerprint"],
        max_severity = entry.get("max_severity"),
        approved_at = entry.get("approved_at", ""),
        scanner_version = entry.get("scanner_version", 0),
    )


def record(
    subject: str,
    target_key: str,
    *,
    commit_sha: Optional[str],
    fingerprint: str,
    max_severity: Optional[str],
    scanner_version: int = SCANNER_VERSION,
) -> None:
    """Persist a user's explicit approval. CRITICAL is never stored."""
    if not subject or not fingerprint or cache_disabled() or max_severity == CRITICAL:
        return
    with _lock, _file_lock():
        data = _load()
        subjects = data.setdefault("subjects", {})
        subj = subjects.get(subject)
        if not isinstance(subj, dict):  # tolerate a hand-edited non-dict entry
            subj = subjects[subject] = {}
        subj[target_key] = {
            "commit_sha": commit_sha,
            "fingerprint": fingerprint,
            "max_severity": max_severity,
            "scanner_version": scanner_version,
            "approved_at": datetime.now(timezone.utc).isoformat(),
        }
        _save(data)


def forget(subject: str, target_key: str) -> None:
    """Drop an approval (e.g. the user declined / discarded the download)."""
    if not subject:
        return
    with _lock, _file_lock():
        data = _load()
        subj = data.get("subjects", {}).get(subject)
        if isinstance(subj, dict) and subj.pop(target_key, None) is not None:
            _save(data)


def clear() -> None:
    """Test helper: drop the on-disk store."""
    with _lock:
        try:
            _store_path().unlink(missing_ok = True)
        except OSError:
            pass


def resolve_commit_sha(target: str, hf_token: Optional[str] = None) -> Optional[str]:
    """Current HF commit SHA for *target*, or None (local path / offline / error). Resolved
    fresh every call: the default branch is mutable, so a cached SHA could mask a moved repo
    and reuse stale consent. None falls back to the authoritative fingerprint (never fail-open).
    """
    from utils.paths import is_local_path
    try:
        if is_local_path(target) or _env_offline():
            return None
        from huggingface_hub import HfApi
        return HfApi().model_info(target, token = hf_token).sha
    except Exception as exc:
        logger.debug("Could not resolve commit sha for '%s': %s", target, exc)
        return None


def resolve_combined_sha(targets, hf_token: Optional[str] = None) -> Optional[str]:
    """Combined SHA over the primary targets; None if ANY is unresolvable. A cheap secondary
    gate only -- the fingerprint (which also covers external auto_map repos) stays
    authoritative, so a None here just falls back to the fingerprint, never weakens it."""
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
