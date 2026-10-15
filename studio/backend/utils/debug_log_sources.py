# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Find the log files the Settings > Logs viewer is allowed to read. The client never names a path: it gets opaque ids from `list_sources` and hands one back, and `resolve_source_id` re-runs this same walk and matches the digest, so the only paths that can reach open() are ones this module produced."""

from __future__ import annotations

import fnmatch
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# (subdir under studio home, filename glob). backend-* is the Tauri capture of stdout,
# the only record when the backend dies before disk logging starts.
FAMILIES: dict[str, tuple[str, str]] = {
    "server": ("logs/server", "server-*.log"),
    "llama-server": ("logs/llama-server", "llama-*.log"),
    "diffusion-server": ("logs/diffusion-server", "diffusion-*.log"),
    "desktop-backend": ("logs", "backend-*.log"),
    "desktop-install": ("logs", "install-*.log"),
    "desktop-update": ("logs", "update-*.log"),
    "desktop-repair": ("logs", "repair-*.log"),
    "desktop-shell": ("", "tauri.log*"),
}

# Several per family: the llama runner writes one file per load attempt.
MAX_SOURCES_PER_FAMILY = 10

_DIGEST_CHARS = 16


@dataclass(frozen = True)
class LogSource:
    id: str
    family: str
    label: str
    realpath: str
    size_bytes: int
    modified_at: float
    is_current: bool


def candidate_roots() -> list[Path]:
    """Every studio home a log file might be under, most specific first. studio_root() infers a root from the installer venv while the runners resolve their own base (llama_cpp.py:_swa_cache_path) without that inference, so on a venv install with no env var set the two disagree and scanning only one loses the runtime logs a failed model load is chased through."""
    roots: list[Path] = []

    def _add(path: Optional[Path]) -> None:
        if path is None:
            return
        try:
            resolved = Path(os.path.realpath(path))
        except (OSError, ValueError):
            return
        if not any(_identity(resolved) == _identity(known) for known in roots):
            roots.append(resolved)

    try:
        from utils.paths import studio_root
        _add(studio_root())
    except Exception:
        pass

    # Mirror _swa_cache_path exactly, or a different installation's logs get pulled in.
    env_home = (
        os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME") or ""
    ).strip()
    if env_home:
        # Both spellings: the runners may write to a literal "~" directory when unexpanded.
        for spelling in (Path(env_home).expanduser(), Path(env_home)):
            try:
                _add(spelling)
            except (OSError, ValueError):
                pass
    else:
        try:
            _add(Path.home() / ".unsloth" / "studio")
        except (OSError, RuntimeError):
            pass

    return roots


def _identity(path) -> str:
    """Windows realpath keeps the extended-length prefix only sometimes, so one drive can look like two."""
    text = os.path.normcase(str(path))
    for prefix in ("\\\\?\\unc\\", "\\\\?\\UNC\\", "\\\\?\\"):
        if text.startswith(prefix):
            text = ("\\\\" if prefix.lower().endswith("unc\\") else "") + text[len(prefix) :]
            break
    return text


def _is_inside(real, real_dir) -> bool:
    inner, outer = _identity(real), _identity(real_dir)
    return inner == outer or inner.startswith(outer.rstrip(os.sep) + os.sep)


def _digest(realpath: str) -> str:
    return hashlib.sha256(realpath.encode("utf-8", "surrogateescape")).hexdigest()[:_DIGEST_CHARS]


def _family_files(family: str) -> list[Path]:
    """Real, contained, regular files for one family, newest first."""
    subdir, pattern = FAMILIES[family]
    found: dict[str, tuple[Path, float]] = {}
    for root in candidate_roots():
        directory = root / subdir
        try:
            if not directory.is_dir():
                continue
            real_dir = Path(os.path.realpath(directory))
        except OSError:
            continue
        try:
            entries = list(directory.glob(pattern))
        except OSError:
            continue
        # Filenames embed creation time, so a name presort avoids stat on thousands of files.
        entries.sort(key = lambda entry: entry.name, reverse = True)
        entries = entries[: MAX_SOURCES_PER_FAMILY * 3]
        for entry in entries:
            try:
                real = Path(os.path.realpath(entry))
                # The target must stay inside, so a symlink cannot expose files like ~/.ssh/id_rsa.
                if not _is_inside(real, real_dir):
                    continue
                if not real.is_file():
                    continue
                if not fnmatch.fnmatch(real.name, pattern):
                    continue
                stat = real.stat()
            except (OSError, ValueError):
                continue
            # Keyed on the folded spelling so a case-insensitive volume does not list a file twice.
            found.setdefault(_identity(real), (real, stat.st_mtime))
    ordered = sorted(found.values(), key = lambda item: item[1], reverse = True)
    return [path for path, _ in ordered[:MAX_SOURCES_PER_FAMILY]]


def _is_current(family: str, path: Path, newest: Optional[Path]) -> bool:
    if family == "server":
        # Single-process uvicorn: our pid is in the active log name; suffix match avoids pid prefixes.
        return path.name.endswith(f"-pid{os.getpid()}.log")
    return newest is not None and path == newest


def list_sources() -> list[LogSource]:
    sources: list[LogSource] = []
    for family in FAMILIES:
        files = _family_files(family)
        newest = files[0] if files else None
        for path in files:
            try:
                stat = path.stat()
            except OSError:
                continue
            real = str(path)
            sources.append(
                LogSource(
                    id = f"{family}:{_digest(real)}",
                    family = family,
                    label = path.name,
                    realpath = real,
                    size_bytes = stat.st_size,
                    modified_at = stat.st_mtime,
                    is_current = _is_current(family, path, newest),
                )
            )
    return sources


def resolve_source_id(source_id: str) -> Optional[Path]:
    """Opaque id back to a path, by rebuilding the allowlist and matching. Deliberately not a decode: nothing the caller sends is ever turned into a path, so there is no string that can traverse anywhere."""
    if not isinstance(source_id, str):
        return None
    family, sep, digest = source_id.partition(":")
    if not sep or family not in FAMILIES or len(digest) != _DIGEST_CHARS:
        return None
    for path in _family_files(family):
        if _digest(str(path)) == digest:
            return path
    return None


def source_id_for_path(
    raw: Optional[str], sources: Optional[list[LogSource]] = None
) -> Optional[str]:
    """The opaque id of the source a WRITER's own spelling of a path names, if any."""
    if not isinstance(raw, str) or not raw.strip():
        return None
    wanted = set()
    for expand in (False, True):
        try:
            spelling = Path(raw.strip())
            wanted.add(_identity(os.path.realpath(spelling.expanduser() if expand else spelling)))
        except (OSError, ValueError, RuntimeError):
            pass
    if not wanted:
        return None
    for source in list_sources() if sources is None else sources:
        try:
            if _identity(os.path.realpath(source.realpath)) in wanted:
                return source.id
        except (OSError, ValueError):
            continue
    return None


def default_source_id() -> Optional[str]:
    """The active server session if we have one, else the newest log we found."""
    sources = list_sources()
    if not sources:
        return None
    for source in sources:
        if source.family == "server" and source.is_current:
            return source.id
    # No live session: pick the newest file across all families, not the newest server log.
    return max(sources, key = lambda s: s.modified_at).id


def file_logging_disabled() -> bool:
    return os.environ.get("UNSLOTH_STUDIO_NO_FILE_LOG") == "1"


def source_is_frozen(source_id: Optional[str]) -> bool:
    """Whether nothing will ever be appended to this source again. UNSLOTH_STUDIO_NO_FILE_LOG only skips _setup_server_disk_logging in run.py; the runners and the Tauri shell keep writing their own files, so treating the setting as global labelled a live llama-server log an earlier session that would not update."""
    if not file_logging_disabled():
        return False
    family = (source_id or "").partition(":")[0]
    return family == "server"
