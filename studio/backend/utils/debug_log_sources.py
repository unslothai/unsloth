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

# (subdirectory under a studio home, filename glob).
#
# The Python writers are run.py:_setup_server_disk_logging and the llama /
# diffusion runners in core/inference/llama_cpp.py.
#
# The desktop families below are written by the Tauri shell
# (src-tauri/src/diagnostics/phase_log.rs) into the logs directory ITSELF, not a
# subdirectory. They matter more than their position here suggests: backend-*
# is the shell's capture of the backend's own stdout, and it is the only record
# that exists when the backend dies BEFORE _setup_server_disk_logging runs. In
# that case logs/server is empty, so without these the viewer would tell a user
# whose app failed to start that nothing has been logged, which is the exact
# dead end this tab was built to remove. tauri.log sits at the home root and
# rotates to tauri.log.1.
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
        # Folded, so an UNSLOTH_STUDIO_HOME that differs from the inferred root
        # only in case is not scanned twice on a case-insensitive volume.
        if not any(_identity(resolved) == _identity(known) for known in roots):
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


def _identity(path) -> str:
    """One comparable spelling of a path, for containment and for dedup.

    Two jobs, both platform quirks:

    os.path.realpath is called separately for the directory and for each entry,
    and on Windows ntpath.realpath decides PER CALL whether to keep the
    \\\\?\\ extended-length prefix (it strips it only if the short form still
    resolves). With a deep studio home on a host without long-path support the
    directory can come back as C:\\... while the file comes back as
    \\\\?\\C:\\..., which pathlib reads as two different DRIVES: containment then
    fails and the whole family is silently dropped.

    normcase folds case on Windows and is the identity on POSIX, so a
    case-insensitive volume stops yielding the same file twice under two
    spellings while Linux keeps its case-sensitive comparison unchanged.
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
        # Nothing prunes logs/llama-server, and one file is written per load
        # ATTEMPT, so a real install reaches five figures: this host has 11,794.
        # Calling realpath + stat on every one cost ~356ms, and this endpoint is
        # polled once a second. Every family's filename embeds its creation time
        # (server-YYYYmmdd-HHMMSS, llama-<epoch>, diffusion-<epoch>, and the
        # desktop ones a millisecond epoch), so name order tracks time order and
        # a cheap presort leaves only a handful to interrogate. The survivors are
        # still ordered by real mtime below, and the slice is wide enough that a
        # file whose mtime moved after it was written is still considered.
        entries.sort(key = lambda entry: entry.name, reverse = True)
        entries = entries[: MAX_SOURCES_PER_FAMILY * 3]
        for entry in entries:
            try:
                real = Path(os.path.realpath(entry))
                # A symlink dropped into the log directory must not become a
                # reader for ~/.ssh/id_rsa, so the TARGET has to stay inside.
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
            # list the same file twice, but the first spelling seen is what is
            # kept, so the id digest stays over the real path.
            found.setdefault(_identity(real), (real, stat.st_mtime))
    ordered = sorted(found.values(), key = lambda item: item[1], reverse = True)
    return [path for path, _ in ordered[:MAX_SOURCES_PER_FAMILY]]


def _is_current(family: str, path: Path, newest: Optional[Path]) -> bool:
    if family == "server":
        # uvicorn runs single process here, so our own pid is the one in the
        # active session's filename: an exact match, not a newest-file guess.
        # Anchored on the suffix, because run.py writes server-{stamp}-pid{pid}
        # and a substring test for "pid1234" also matches a retained
        # ...-pid12345.log, which would mark two sources current and could open
        # the wrong one by default.
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
