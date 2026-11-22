# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One Hub call in place of one per file when a repo's cached snapshot is already current.

Files under a snapshot are fixed by its commit, and the same loader code on the same library versions asks for the
same files. So once a revalidating load of commit C finished (``note_validated``), a later load whose Hub ``main``
still names C reads from the cache (``current_snapshot``). Anything else revalidates per file as before.
``UNSLOTH_DIFFUSION_HUB_SNAPSHOT=0`` turns it off. Torch-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Callable, Optional

SNAPSHOT_ENV = "UNSLOTH_DIFFUSION_HUB_SNAPSHOT"
_FILE_NAME = "diffusion_hub_snapshots.json"
_FORMAT = 1
_MAX_ENTRIES = 64
_SOURCES = (
    "diffusion_flux3.py",
    "diffusion_small_host.py",
    "diffusion_manual_cast.py",
    "diffusion_hub_snapshot.py",
)
_LOCK = threading.Lock()
_DIGEST: list = []


def enabled() -> bool:
    raw = (os.environ.get(SNAPSHOT_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def _versions() -> dict:
    from importlib import metadata

    out = {}
    for name in ("transformers", "diffusers", "huggingface_hub", "tokenizers"):
        try:
            out[name] = metadata.version(name)
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def _code_digest() -> str:
    if not _DIGEST:
        h = hashlib.sha256(f"format={_FORMAT}".encode())
        here = Path(__file__).resolve().parent
        for name in _SOURCES:
            h.update(name.encode())
            try:
                h.update((here / name).read_bytes())
            except OSError:
                h.update(b"<missing>")
        _DIGEST.append(h.hexdigest()[:20])
    return _DIGEST[0]


def _repo_root(repo_id: str, cache_dir: Optional[str]) -> Path:
    from huggingface_hub import constants
    from huggingface_hub.file_download import repo_folder_name
    return Path(cache_dir or constants.HF_HUB_CACHE) / repo_folder_name(
        repo_id = repo_id, repo_type = "model"
    )


def _local_commit(repo_id: str, cache_dir: Optional[str], revision: str) -> Optional[str]:
    root = _repo_root(repo_id, cache_dir)
    try:
        commit = (root / "refs" / revision).read_text(encoding = "utf-8").strip()
    except OSError:
        return None
    return commit if commit and (root / "snapshots" / commit).is_dir() else None


def _listing(repo_id: str, cache_dir: Optional[str], commit: str) -> list:
    """Cached files (and recorded absences) for ``commit``, sorted. A dangling link is not held: a cache-only read
    silently drops it (e.g. a processor's chat template on transformers 4.57.6 / 5.17)."""
    root = _repo_root(repo_id, cache_dir)
    out = []
    for base, tag in (
        (root / "snapshots" / commit, ""),
        (root / ".no_exist" / commit, ".no_exist/"),
    ):
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                out.append(tag + path.relative_to(base).as_posix())
    return sorted(out)


def _key(repo_id: str, cache_dir: Optional[str], components: Any) -> str:
    blob = json.dumps(
        {
            "repo": repo_id,
            "cache": os.path.realpath(str(_repo_root(repo_id, cache_dir))),
            "components": components,
            "versions": _versions(),
            "code": _code_digest(),
        },
        sort_keys = True,
        default = str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def _store_path() -> Optional[Path]:
    try:
        from utils.paths.storage_roots import cache_root
        return Path(cache_root()) / _FILE_NAME
    except Exception:  # noqa: BLE001
        return None


def _read() -> dict:
    path = _store_path()
    try:
        table = (
            json.loads(path.read_text(encoding = "utf-8"))
            if path is not None and path.is_file()
            else {}
        )
    except Exception:  # noqa: BLE001
        return {}
    return table if isinstance(table, dict) and table.get("format") == _FORMAT else {}


def _write(table: dict) -> None:
    path = _store_path()
    if path is None:
        return
    try:
        entries = dict(list((table.get("entries") or {}).items())[-_MAX_ENTRIES:])
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"format": _FORMAT, "entries": entries}), encoding = "utf-8")
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - a lost record costs one revalidating load
        pass


def _hub_id(repo_id: Any) -> bool:
    return (
        isinstance(repo_id, str)
        and repo_id.count("/") == 1
        and not os.path.isdir(os.path.expanduser(repo_id))
    )


def _remote_sha(repo_id: str, token: Optional[str], revision: str, api: Any) -> Optional[str]:
    from . import hub_reachability as reach

    if reach.offline_mode():
        return None
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    info = reach.model_info(api, repo_id, revision = revision, token = token or None)
    return str(getattr(info, "sha", "") or "") or None


def current_snapshot(
    repo_id: Any,
    components: Any,
    *,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
    logger: Any = None,
    api: Any = None,
) -> Optional[str]:
    """The validated cached commit if still current on the Hub and fully cached, else None. Never raises."""
    try:
        if not enabled() or not _hub_id(repo_id):
            return None
        commit = _local_commit(repo_id, cache_dir, revision)
        if commit is None:
            return None
        with _LOCK:
            entry = (_read().get("entries") or {}).get(_key(repo_id, cache_dir, components))
        if not isinstance(entry, dict) or entry.get("commit") != commit:
            return None
        if entry.get("files") != _listing(repo_id, cache_dir, commit):
            return None
        if _remote_sha(repo_id, token, revision, api) != commit:
            return None
        if logger is not None:
            logger.info(
                "diffusion.hub_snapshot: %s main is still the validated cached commit %s; its components read "
                "from the cache (one Hub call, not one per file)",
                repo_id,
                commit[:10],
            )
        return commit
    except Exception:  # noqa: BLE001 - unanswerable: the per-file revalidation runs as before
        return None


def note_validated(
    repo_id: Any,
    components: Any,
    *,
    token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
    api: Any = None,
) -> bool:
    """Record a finished revalidating load, only if the Hub confirms the commit (offline records nothing)."""
    try:
        if not enabled() or not _hub_id(repo_id):
            return False
        commit = _local_commit(repo_id, cache_dir, revision)
        if commit is None or _remote_sha(repo_id, token, revision, api) != commit:
            return False
        key = _key(repo_id, cache_dir, components)
        files = _listing(repo_id, cache_dir, commit)
        with _LOCK:
            table = _read()
            entries = dict(table.get("entries") or {})
            entries.pop(key, None)
            entries[key] = {"commit": commit, "files": files}
            _write({"entries": entries})
        return True
    except Exception:  # noqa: BLE001
        return False


def forget(
    repo_id: Any,
    components: Any,
    *,
    cache_dir: Optional[str] = None,
) -> None:
    """Drop the record: the next load revalidates every file and records again. Never raises."""
    try:
        key = _key(repo_id, cache_dir, components)
        with _LOCK:
            table = _read()
            entries = dict(table.get("entries") or {})
            if entries.pop(key, None) is not None:
                _write({"entries": entries})
    except Exception:  # noqa: BLE001
        pass


def cache_first(
    load: Callable[[bool], Any],
    *,
    cached: bool,
    name: str,
    logger: Any = None,
    on_fallback: Optional[Callable[[], None]] = None,
) -> Any:
    """``load(local_files_only)``: cache-only when ``cached``, else (or when that read fails) the revalidating read."""
    if cached:
        try:
            return load(True)
        except (OSError, ValueError) as exc:
            if on_fallback is not None:
                on_fallback()
            if logger is not None:
                logger.info(
                    "diffusion.hub_snapshot: %s not complete in the cache (%s: %s); reading it from the Hub",
                    name,
                    type(exc).__name__,
                    str(exc)[:200],
                )
    return load(False)
