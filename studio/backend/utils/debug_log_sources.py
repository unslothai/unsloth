# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Find the log files the Settings > Logs viewer is allowed to read.

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

# (subdirectory under a studio home, filename glob).
#
# Python writers: run.py:_setup_server_disk_logging and the llama / diffusion
# runners in core/inference/llama_cpp.py. The desktop families come from the
# Tauri shell (src-tauri/src/diagnostics/phase_log.rs) and land in the logs
# directory ITSELF, with tauri.log at the home root (rotates to tauri.log.1).
# backend-* is the shell's capture of backend stdout, and the only record that
# exists when the backend dies BEFORE _setup_server_disk_logging runs.
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

# Per family, so a busy host cannot make the picker unusable. Several, not one:
# the llama runner writes a file per load ATTEMPT, so after a retry the useful
# one is often not the newest.
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
    device_id: int | None = None
    inode: int | None = None


def candidate_roots() -> list[Path]:
    """Every studio home a log file might be under, most specific first.

    studio_root() infers a root from the installer venv; the runners resolve
    their own base (llama_cpp.py:_swa_cache_path) without that inference. On a
    venv install with no env var set the two disagree, so scanning only one
    loses the runtime logs a failed model load is chased through.
    """
    roots: list[Path] = []

    def _add(path: Optional[Path]) -> None:
        if path is None:
            return
        try:
            resolved = Path(os.path.realpath(path))
        except (OSError, ValueError):
            return
        # Folded, so a case-only difference is not scanned twice on a
        # case-insensitive volume.
        if not any(_identity(resolved) == _identity(known) for known in roots):
            roots.append(resolved)

    try:
        from utils.paths import studio_root
        _add(studio_root())
    except Exception:
        pass

    # Mirror _swa_cache_path exactly: env override if set, else the legacy home,
    # never both. Both would pull a DIFFERENT installation's logs into this one.
    env_home = (
        os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME") or ""
    ).strip()
    if env_home:
        # Both spellings, because writer and reader disagree about the tilde:
        # _swa_cache_path builds Path(home) raw, so an unexpanded value (systemd
        # EnvironmentFile, dotenv) makes the runners write to a directory NAMED
        # "~" while expanduser looks in the real home. Safe unlike the
        # env-versus-legacy case above: one value, two spellings, so neither can
        # be another installation's home.
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
    """One comparable spelling of a path, for containment and for dedup.

    Two Windows quirks. realpath is called separately for the directory and for
    each entry, and ntpath.realpath decides PER CALL whether to keep the \\\\?\\
    extended-length prefix, so the directory can come back as C:\\... and the
    file as \\\\?\\C:\\..., which pathlib reads as two different DRIVES:
    containment fails and the whole family is silently dropped. And normcase
    folds case (identity on POSIX), so a case-insensitive volume cannot yield
    one file twice under two spellings.
    """
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


def _family_files(family: str, max_sources: Optional[int] = MAX_SOURCES_PER_FAMILY) -> list[Path]:
    """Real, contained, regular files for one family, newest first."""
    subdir, pattern = FAMILIES[family]
    found: dict[str, tuple[Path, float]] = {}
    for root in candidate_roots():
        directory = root / subdir
        try:
            if not directory.is_dir():
                continue
            real_root = Path(os.path.realpath(root))
            real_dir = Path(os.path.realpath(directory))
            # A family directory can itself be a symlink. Trusting its resolved
            # target as the containment root would turn the export into a glob
            # over an unrelated directory outside every Studio home.
            if not _is_inside(real_dir, real_root):
                continue
        except (OSError, ValueError):
            continue
        try:
            entries = list(directory.glob(pattern))
        except OSError:
            continue
        # Nothing prunes logs/llama-server and one file is written per load
        # ATTEMPT, so a real install reaches five figures (this host: 11,794)
        # and realpath + stat on every one cost ~356ms, at a 1 Hz poll. Every
        # family's filename embeds its creation time (server-YYYYmmdd-HHMMSS,
        # llama-<epoch>, diffusion-<epoch>, desktop ms epoch), so name order
        # tracks time order and this presort leaves a handful to stat. Survivors
        # are still ordered by real mtime below, and the slice is wide enough to
        # keep a file whose mtime moved after it was written.
        entries.sort(key = lambda entry: entry.name, reverse = True)
        if max_sources is not None:
            entries = entries[: max_sources * 3]
        for entry in entries:
            try:
                real = Path(os.path.realpath(entry))
                # The TARGET must stay inside, so a symlink dropped into the log
                # directory cannot become a reader for ~/.ssh/id_rsa.
                if not _is_inside(real, real_dir):
                    continue
                if not real.is_file():
                    continue
                if not fnmatch.fnmatch(real.name, pattern):
                    continue
                stat = real.stat()
            except (OSError, ValueError):
                continue
            # Keyed on the folded spelling so a case-insensitive volume cannot
            # list one file twice; the first spelling seen is kept, so the id
            # digest stays over the real path.
            found.setdefault(_identity(real), (real, stat.st_mtime))
    ordered = sorted(found.values(), key = lambda item: item[1], reverse = True)
    if max_sources is None:
        return [path for path, _ in ordered]
    return [path for path, _ in ordered[:max_sources]]


def _is_current(family: str, path: Path, newest: Optional[Path]) -> bool:
    if family == "server":
        # uvicorn is single process here, so our own pid is in the active
        # session's filename: an exact match, not a newest-file guess. Anchored
        # on the suffix because a substring test for "pid1234" would also match
        # a retained ...-pid12345.log.
        return path.name.endswith(f"-pid{os.getpid()}.log")
    return newest is not None and path == newest


def list_sources(max_sources_per_family: Optional[int] = MAX_SOURCES_PER_FAMILY) -> list[LogSource]:
    sources: list[LogSource] = []
    for family in FAMILIES:
        files = _family_files(family, max_sources_per_family)
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
                    device_id = stat.st_dev,
                    inode = stat.st_ino,
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
    # No live session: the newest file across every family, NOT any retained
    # server log. Preferring a stale server log opened the tab on a previous run
    # while the llama log holding the failure sat one entry down, which is the
    # state after UNSLOTH_STUDIO_NO_FILE_LOG=1 or a failed log setup.
    return max(sources, key = lambda s: s.modified_at).id


def file_logging_disabled() -> bool:
    return os.environ.get("UNSLOTH_STUDIO_NO_FILE_LOG") == "1"


def source_is_frozen(source_id: Optional[str]) -> bool:
    """Whether nothing will ever be appended to this source again.

    UNSLOTH_STUDIO_NO_FILE_LOG only skips _setup_server_disk_logging in run.py.
    The runners and the Tauri shell keep writing their own files, so treating
    the setting as global labelled a live llama-server log an earlier session
    that would not update, while it was still being appended to.
    """
    if not file_logging_disabled():
        return False
    family = (source_id or "").partition(":")[0]
    return family == "server"
