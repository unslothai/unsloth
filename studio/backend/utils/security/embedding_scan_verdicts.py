# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persistent, per-user cache of Hugging Face "clean" security verdicts for embedding models.

The offline embedding-security gate is fail-CLOSED: with no network to reach HF's scan, a cached
pickle weight cannot be verified, so it is blocked. This cache lets a pickle model the user already
loaded ONLINE (and that HF scanned clean) load again OFFLINE, without weakening the gate for an
unknown or never-scanned pickle. A record is honored offline only when the active cached commit and
EVERY load-root pickle's sha256 exactly match what was recorded; anything else keeps blocking.

Binding: repo id + full commit SHA + an exact map of snapshot-relative pickle name -> sha256. The
commit pins HF's immutable content; the per-file sha256 detects a locally swapped pickle at that
commit; the exact map (not a set) detects an added / renamed / newly load-relevant pickle.

Threat model. This protects against STALE revisions and ACCIDENTAL / non-concurrent cache tampering.
A same-user attacker with arbitrary filesystem write -- to the HF cache OR to this store under
``studio_root()/security/`` -- is OUTSIDE the enforceable boundary: they can already replace the
weights, imported Python, or loader config, and could forge a record beside a matching poisoned
pickle. A local HMAC/signing key in the same account does not change that. The store is written
0600 in the Studio-private dir and every read failure degrades to "no record" (i.e. block); a
record is never trusted online (the Hub is always re-queried) and is deleted on an authoritative
unsafe verdict. A 30-day TTL bounds offline reuse of a verdict HF may have since revised. The
sha256 is computed locally just before load, so a verify->load TOCTOU window remains (narrow; the
same-user-write threat already dominates), and a Hub scanner false negative is recorded faithfully
-- safetensors remains the stronger, format-level defense and the UI/logs say "previously cleared
by the Hub scan", not "proven safe".
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from typing import Optional

from loggers import get_logger
from utils.paths import storage_roots

logger = get_logger(__name__)

_SCHEMA_VERSION = 1
# A recorded verdict is honored for at most this long, bounding offline reuse of a verdict HF may
# have revised since (an offline process cannot learn about a rescan). An online clean load rewrites
# the record and refreshes the window.
_TTL_SECONDS = 30 * 24 * 60 * 60
_HASH_CHUNK = 1024 * 1024  # 1 MiB streaming reads so a multi-GB weight never loads into memory

_lock = threading.RLock()


def cache_disabled() -> bool:
    """True when the verdict cache is turned off by env, forcing the pure fail-closed offline gate."""
    return os.environ.get("UNSLOTH_EMBED_VERDICT_CACHE_DISABLE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _store_path():
    return storage_roots.studio_root() / "security" / "embedding_scan_verdicts.json"


def _key(repo_id: str) -> str:
    """Case-folded store key. Hub repo ids are case-insensitive, so ``BAAI/bge-m3`` and
    ``baai/bge-m3`` must resolve to one record."""
    return (repo_id or "").strip().lower()


def sha256_file(path) -> Optional[str]:
    """Streaming sha256 hex digest of *path*, or None if it cannot be read (an unreadable file must
    never verify as a match -> the caller keeps blocking)."""
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _load() -> dict:
    """Parsed store, or an empty skeleton on any error (fail-safe = no records = block offline)."""
    try:
        with open(_store_path()) as f:
            data = json.load(f)
        if (
            isinstance(data, dict)
            and data.get("version") == _SCHEMA_VERSION
            and isinstance(data.get("records"), dict)
        ):
            return data
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning("Could not read embedding scan verdicts (%s); ignoring", exc)
    return {"version": _SCHEMA_VERSION, "records": {}}


def _save(data: dict) -> None:
    """Atomic write (tmp + os.replace), best-effort 0600 in the Studio-private security dir."""
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
        logger.warning("Could not write embedding scan verdicts (%s)", exc)
        try:
            tmp.unlink(missing_ok = True)
        except OSError:
            pass


@contextlib.contextmanager
def _file_lock():
    """Best-effort cross-process exclusive lock around a read-modify-write. Training / export /
    inference record from separate subprocesses, so the in-process RLock alone would let two
    processes each read the store and clobber the other's record on ``os.replace``. Degrades to a
    no-op when OS locking is unavailable (the only consequence is a lost optimization record, which
    just causes a later offline block, never a bypass)."""
    path = _store_path()
    try:
        storage_roots.ensure_dir(path.parent)
        fd = os.open(str(path.parent / f"{path.name}.lock"), os.O_CREAT | os.O_RDWR, 0o600)
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


def _clean_pickles(pickles) -> Optional[dict]:
    """Validate a ``{rel-path: sha256}`` map: a non-empty dict of str->64-hex-char values, or None.
    Rejects a hand-edited entry whose shape is wrong so it can never seed a spurious match."""
    if not isinstance(pickles, dict) or not pickles:
        return None
    out = {}
    for rel, digest in pickles.items():
        if not isinstance(rel, str) or not isinstance(digest, str):
            return None
        d = digest.strip().lower()
        if len(d) != 64 or any(c not in "0123456789abcdef" for c in d):
            return None
        out[rel] = d
    return out


def record_clean(repo_id: str, commit: Optional[str], pickles: dict) -> None:
    """Persist a clean verdict for *repo_id* at *commit*: the exact ``{snapshot-relative pickle
    name: sha256}`` map the loader will deserialize. No-op when the cache is disabled, the commit is
    unknown, or the map is empty/malformed (nothing to attest)."""
    if cache_disabled() or not commit:
        return
    cleaned = _clean_pickles(pickles)
    if cleaned is None:
        return
    key = _key(repo_id)
    if not key:
        return
    with _lock, _file_lock():
        data = _load()
        records = data.setdefault("records", {})
        if not isinstance(records, dict):  # tolerate a hand-edited non-dict
            records = data["records"] = {}
        records[key] = {
            "commit": commit,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "pickles": cleaned,
        }
        _save(data)


def lookup(repo_id: str, commit: Optional[str]) -> Optional[dict]:
    """The recorded ``{rel: sha256}`` map for *repo_id*, but ONLY when the stored commit equals
    *commit* and the record is within the TTL. None otherwise, on a disabled cache, or on any error
    -- the offline caller then keeps blocking."""
    if cache_disabled() or not commit:
        return None
    key = _key(repo_id)
    if not key:
        return None
    with _lock:
        entry = _load().get("records", {}).get(key)
    if not isinstance(entry, dict) or entry.get("commit") != commit:
        return None
    recorded_at = entry.get("recorded_at")
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(recorded_at)).total_seconds()
    except (TypeError, ValueError):
        return None  # unparseable timestamp -> treat as no record
    if age < 0 or age > _TTL_SECONDS:
        return None
    return _clean_pickles(entry.get("pickles"))


def forget(repo_id: str) -> None:
    """Drop a repo's recorded verdict (called when HF returns an authoritative unsafe status, so a
    now-flagged commit cannot keep loading offline on a stale clean record)."""
    key = _key(repo_id)
    if not key:
        return
    with _lock, _file_lock():
        data = _load()
        records = data.get("records", {})
        if isinstance(records, dict) and records.pop(key, None) is not None:
            _save(data)


def clear() -> None:
    """Test helper: drop the on-disk store."""
    with _lock:
        try:
            _store_path().unlink(missing_ok = True)
        except OSError:
            pass
