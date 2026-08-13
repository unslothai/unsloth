# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Find the log files the Settings > Debugging viewer is allowed to read.

The client never names a path. It gets opaque ids from `list_sources` and hands
one back; `resolve_source_id` re-runs this same walk and matches the digest, so
the only paths that can ever reach open() are ones this module produced.
"""

from __future__ import annotations

import fnmatch
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# (subdirectory under a studio home, filename glob). The three writers:
# run.py:_setup_server_disk_logging, and the llama / diffusion runners in
# core/inference/llama_cpp.py.
FAMILIES: dict[str, tuple[str, str]] = {
    "server": ("logs/server", "server-*.log"),
    "llama-server": ("logs/llama-server", "llama-*.log"),
    "diffusion-server": ("logs/diffusion-server", "diffusion-*.log"),
}

# Per family, so a busy host cannot make the picker unusable. The llama runner
# writes one file per load ATTEMPT, and after a retry the interesting one is
# often the second to last, so this has to be several, not one.
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
    """Every studio home a log file might be under, most specific first.

    studio_root() infers a root from the installer venv; the llama and diffusion
    runners resolve their own base (llama_cpp.py:_swa_cache_path) and do NOT do
    that inference. On a venv install with no env var set the two disagree, and
    scanning only one of them loses exactly the runtime logs a user chasing a
    failed model load came here for.
    """
    roots: list[Path] = []

    def _add(path: Optional[Path]) -> None:
        if path is None:
            return
        try:
            resolved = Path(os.path.realpath(path))
        except (OSError, ValueError):
            return
        if resolved not in roots:
            roots.append(resolved)

    try:
        from utils.paths import studio_root
        _add(studio_root())
    except Exception:
        pass

    # Mirror _swa_cache_path exactly: env override if set, otherwise the legacy
    # home, never both. Adding the legacy home while an override is active would
    # pull a DIFFERENT installation's logs into this one's viewer.
    env_home = (
        os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME") or ""
    ).strip()
    if env_home:
        try:
            _add(Path(env_home).expanduser())
        except (OSError, ValueError):
            pass
    else:
        try:
            _add(Path.home() / ".unsloth" / "studio")
        except (OSError, RuntimeError):
            pass

    return roots


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
        for entry in entries:
            try:
                real = Path(os.path.realpath(entry))
                # A symlink dropped into the log directory must not become a
                # reader for ~/.ssh/id_rsa, so the TARGET has to stay inside.
                if not real.is_relative_to(real_dir):
                    continue
                if not real.is_file():
                    continue
                if not fnmatch.fnmatch(real.name, pattern):
                    continue
                stat = real.stat()
            except (OSError, ValueError):
                continue
            found[str(real)] = (real, stat.st_mtime)
    ordered = sorted(found.values(), key = lambda item: item[1], reverse = True)
    return [path for path, _ in ordered[:MAX_SOURCES_PER_FAMILY]]


def _is_current(family: str, path: Path, newest: Optional[Path]) -> bool:
    if family == "server":
        # uvicorn runs single process here, so our own pid is the one in the
        # active session's filename: an exact match, not a newest-file guess.
        return f"pid{os.getpid()}" in path.name
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
    """Opaque id back to a path, by rebuilding the allowlist and matching.

    Deliberately not a decode: nothing the caller sends is ever turned into a
    path, so there is no string that can traverse anywhere.
    """
    if not isinstance(source_id, str):
        return None
    family, sep, digest = source_id.partition(":")
    if not sep or family not in FAMILIES or len(digest) != _DIGEST_CHARS:
        return None
    for path in _family_files(family):
        if _digest(str(path)) == digest:
            return path
    return None


def default_source_id() -> Optional[str]:
    """The active server session if we have one, else the newest log we found."""
    sources = list_sources()
    if not sources:
        return None
    for source in sources:
        if source.family == "server" and source.is_current:
            return source.id
    for source in sources:
        if source.family == "server":
            return source.id
    return max(sources, key = lambda s: s.modified_at).id


def file_logging_disabled() -> bool:
    return os.environ.get("UNSLOTH_STUDIO_NO_FILE_LOG") == "1"
