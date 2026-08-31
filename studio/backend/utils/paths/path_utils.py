# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Path utilities for model and dataset handling
"""

import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar
import structlog
from loggers import get_logger

logger = get_logger(__name__)

# Opening a cloud placeholder for data recalls it. These attributes are available through
# ``stat_result.st_file_attributes`` on Windows without reading file contents.
_WINDOWS_CONTENT_RECALL_ATTRIBUTES = (
    0x00001000  # FILE_ATTRIBUTE_OFFLINE
    | 0x00040000  # FILE_ATTRIBUTE_RECALL_ON_OPEN
    | 0x00400000  # FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS
)


def file_contents_available_locally(path, stat_result = None) -> bool:
    """Whether opening *path* can read data without recalling a cloud placeholder.

    Non-Windows files have no ``st_file_attributes`` and are treated as local. An
    inaccessible path is not safe to open during inventory discovery.
    """
    try:
        info = stat_result if stat_result is not None else os.stat(path)
    except OSError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    return not bool(attributes & _WINDOWS_CONTENT_RECALL_ATTRIBUTES)


# ── macOS Finder metadata companions ───────────────────────────
# A volume without native xattrs (exFAT, FAT, most SMB and NFS) makes macOS keep a file's xattrs
# in a "._" companion carrying the same extension, so it answers every name-shaped question the
# way the real file does and sorts ahead of it. Nothing may be refused for the prefix alone: a
# user's own "._model.gguf" is a real model, and only the magic bytes settle it.

_MAGIC = b"\x00\x05\x16\x07"

PathLike = TypeVar("PathLike", str, Path)


def is_appledouble_name(path: str) -> bool:
    """A name test only: it decides which files are worth opening, never what one is."""
    return str(path).replace("\\", "/").rsplit("/", 1)[-1].startswith("._")


def has_appledouble_magic(path: Path) -> bool:
    """The four bytes ``file(1)`` reads to report "AppleDouble encoded Macintosh file"."""
    try:
        # Directory scans reach here with whatever the volume holds, and opening a FIFO blocks
        # until someone writes to it. Only a regular file can carry the magic anyway.
        if not path.is_file():
            return False
        with open(path, "rb") as handle:
            return handle.read(len(_MAGIC)) == _MAGIC
    except OSError:
        return False


def is_appledouble_metadata(path: Path) -> bool:
    """True only for a ``._`` file whose bytes ARE AppleDouble."""
    path = Path(path)
    return is_appledouble_name(path.name) and has_appledouble_magic(path)


def drop_appledouble_metadata(paths: Iterable[PathLike]) -> list[PathLike]:
    """*paths* without the entries that are Finder metadata, preserving order and type."""
    return [p for p in paths if not is_appledouble_metadata(Path(p))]


def any_not_appledouble_metadata(paths: Iterable[PathLike]) -> bool:
    """Whether *paths* holds anything that is not Finder metadata, stopping at the first.

    Callers hand this a live ``glob``, which materializing would walk in full.
    """
    return any(not is_appledouble_metadata(Path(p)) for p in paths)


def _shadowed_name(path: str) -> str:
    head, _, name = str(path).replace("\\", "/").rpartition("/")
    return f"{head}/{name[2:]}" if head else name[2:]


def drop_shadowed_appledouble_names(
    # Optional[...] rather than `| None`: this module has no `from __future__ import
    # annotations`, so its annotations are evaluated at import, and PEP 604 unions are a
    # TypeError on the declared 3.9 floor. tests/test_python39_compatibility.py gates it.
    files: list[str],
    *,
    subject_key: Optional[Callable[[str], object]] = None,
) -> list[str]:
    """*files* without the ``._`` entries whose subject is present in the same listing.

    For remote listings, which carry no bytes to read, so a sole candidate survives whatever it
    is called. *subject_key* widens what counts as the subject, for files that come in sets.
    """
    key = subject_key or (lambda name: name)
    present = {key(f.replace("\\", "/")) for f in files}
    return [f for f in files if not (is_appledouble_name(f) and key(_shadowed_name(f)) in present)]


# Per-process cache to avoid repeated cache-dir scans for the same identifier.
_CACHE_CASE_RESOLUTION_MEMO: dict[str, str] = {}

# Instrumentation counters for operational visibility.
_CACHE_CASE_RESOLUTION_STATS: dict[str, int] = {
    "calls": 0,
    "memo_hits": 0,
    "exact_hits": 0,
    "variant_hits": 0,
    "tie_breaks": 0,
    "fallbacks": 0,
    "errors": 0,
}


def _is_wsl() -> bool:
    """Detect if we are running inside WSL (Windows Subsystem for Linux)."""
    if sys.platform == "win32":
        return False
    try:
        with open("/proc/version", "r", encoding = "utf-8") as f:
            return "microsoft" in f.read().lower()
    except Exception:
        return False


_IS_WSL: bool = _is_wsl()


def normalize_path(path: str) -> str:
    """Normalize filesystem paths for cross-platform use.

    WSL maps drive-letter paths to ``/mnt/<drive>/...``; native Windows keeps
    the drive and normalizes separators; elsewhere slashes are forward-only.
    """
    if not path:
        return path

    # Handle Windows drive letters (C:\\ or c:\\)
    if len(path) >= 3 and path[1] == ":" and path[2] in ("\\", "/"):
        # Map to /mnt/<drive>/ only under WSL; native Windows keeps the drive letter.
        if _IS_WSL:
            drive = path[0].lower()
            rest = path[3:].replace("\\", "/")
            return f"/mnt/{drive}/{rest}"
        return path.replace("\\", "/")

    # Already Unix-style or relative
    return path.replace("\\", "/")


def wsl_automount_root() -> str:
    """DrvFs root WSL maps Windows drives under, with a trailing slash.

    Set via ``/etc/wsl.conf`` ``[automount] root``, so hard-coding ``/mnt/``
    mistranslates drive paths on a host that moved it (``root = /`` puts C: at ``/c/``).
    """
    default = "/mnt/"
    if not _IS_WSL:
        return default
    try:
        import configparser

        parser = configparser.ConfigParser(inline_comment_prefixes = ("#", ";"))
        parser.read("/etc/wsl.conf", encoding = "utf-8")
        root = parser.get("automount", "root", fallback = "").strip().strip("\"'")
    except Exception:
        return default
    if not root:
        return default
    return root if root.endswith("/") else f"{root}/"


_WSL_AUTOMOUNT_ROOT: str = wsl_automount_root()


def _looks_windows_shaped(path: str) -> bool:
    """True for a drive-letter path (``C:\\x``, ``c:/x``) or a UNC path (``\\\\host\\share``)."""
    if path.startswith("\\\\"):
        return True
    return len(path) >= 3 and path[1] == ":" and path[2] in ("\\", "/")


def host_normalize_path(path: str) -> str:
    """Normalize a path this process is about to open, honouring ``[automount] root``.

    Not :func:`normalize_path`: that hard-codes ``/mnt/`` to predict where the model
    *loader* will look, while a path read from another tool's config is stat-ed here.

    Separators are rewritten only when the path is Windows-shaped, or on Windows itself
    where a backslash cannot be anything else. Everywhere else, WSL included, a path that
    names no drive is a POSIX path, and a backslash in it is a legal filename character:
    rewriting it would silently lose a directory that has one in its name.
    """
    if not path:
        return path

    if _looks_windows_shaped(path):
        if _IS_WSL and path[1:2] == ":":
            drive = path[0].lower()
            rest = path[3:].replace("\\", "/")
            return f"{_WSL_AUTOMOUNT_ROOT}{drive}/{rest}"
        return path.replace("\\", "/")

    if os.name == "nt":
        return path.replace("\\", "/")

    return path


def is_local_path(path: str) -> bool:
    """
    Check if path is a local filesystem path vs HuggingFace model identifier.

    Examples:
        True: /home/user/model, C:\\models, ./model, ~/model
        False: unsloth/llama-3.1-8b, microsoft/phi-2
    """
    if not path:
        return False

    # Exists on disk → local (covers relative paths like "outputs/foo").
    try:
        if Path(normalize_path(path)).expanduser().exists():
            return True
    except Exception:
        pass

    # Obvious HF patterns
    if path.count("/") == 1 and not path.startswith(("/", ".", "~")):
        return False  # Looks like org/model format

    # Filesystem indicators
    return (
        path.startswith(("/", ".", "~"))  # Unix absolute/relative
        or ":" in path  # Windows drive or URL
        or "\\" in path  # Windows separator
        or os.path.isabs(path)  # System-absolute
    )


def get_cache_path(model_name: str) -> Optional[Path]:
    """Get HuggingFace cache path for a model if it exists."""
    cache_dir = _hf_hub_cache_dir()
    resolved_name = resolve_cached_repo_id_case(model_name)
    model_cache_name = resolved_name.replace("/", "--")
    model_cache_path = cache_dir / f"models--{model_cache_name}"

    return model_cache_path if model_cache_path.exists() else None


def is_model_cached(model_name: str) -> bool:
    """Check if model is downloaded in HuggingFace cache."""
    cache_path = get_cache_path(model_name)
    if not cache_path:
        return False

    # Check for model files
    for suffix in [".safetensors", ".bin", ".json"]:
        if any_not_appledouble_metadata(cache_path.rglob(f"*{suffix}")):
            return True

    return False


def _hf_hub_cache_dir() -> Path:
    """Return HF cache root honoring HF_HUB_CACHE when available."""
    from utils.hf_cache_settings import get_hf_cache_paths
    return get_hf_cache_paths().hub_cache


def resolve_cached_repo_id_case(model_name: str, use_memo: bool = True) -> str:
    """Resolve repo_id to the exact casing already present in local HF cache.

    Policy: prefer the requested/canonical repo_id, but reuse a case-variant's
    exact cached spelling if one already exists in local HF cache. Avoids
    duplicate downloads while preserving user intent where possible.
    """
    _CACHE_CASE_RESOLUTION_STATS["calls"] += 1

    if not model_name or "/" not in model_name:
        _CACHE_CASE_RESOLUTION_STATS["fallbacks"] += 1
        return model_name

    cache_dir = _hf_hub_cache_dir()
    if not cache_dir.exists():
        _CACHE_CASE_RESOLUTION_STATS["fallbacks"] += 1
        return model_name

    expected_dir = f"models--{model_name.replace('/', '--')}"

    # Exact-case path first so a new exact match beats a memoized variant.
    exact_path = cache_dir / expected_dir
    if exact_path.is_dir():
        if use_memo:
            _CACHE_CASE_RESOLUTION_MEMO[model_name] = model_name
        _CACHE_CASE_RESOLUTION_STATS["exact_hits"] += 1
        return model_name

    # Revalidate memoized entries on disk to avoid stale results.
    if use_memo:
        cached = _CACHE_CASE_RESOLUTION_MEMO.get(model_name)
        if cached is not None:
            cached_path = cache_dir / f"models--{cached.replace('/', '--')}"
            if cached_path.is_dir():
                _CACHE_CASE_RESOLUTION_STATS["memo_hits"] += 1
                return cached
            # Stale entry -- drop it and re-scan below
            _CACHE_CASE_RESOLUTION_MEMO.pop(model_name, None)

    expected_lower = expected_dir.lower()
    try:
        candidates: list[str] = []
        for entry in cache_dir.iterdir():
            if not entry.is_dir():
                continue
            if entry.name.lower() != expected_lower:
                continue
            if not entry.name.startswith("models--"):
                continue
            repo_part = entry.name[len("models--") :]
            if not repo_part:
                continue
            candidates.append(repo_part.replace("--", "/"))

        if candidates:
            # Deterministic tie-break if multiple case variants coexist
            resolved = sorted(candidates)[0]
            if len(candidates) > 1:
                _CACHE_CASE_RESOLUTION_STATS["tie_breaks"] += 1
            _CACHE_CASE_RESOLUTION_STATS["variant_hits"] += 1
            if use_memo:
                _CACHE_CASE_RESOLUTION_MEMO[model_name] = resolved
            return resolved
    except Exception as exc:
        _CACHE_CASE_RESOLUTION_STATS["errors"] += 1
        logger.debug(f"Could not resolve cached repo_id case for '{model_name}': {exc}")

    _CACHE_CASE_RESOLUTION_STATS["fallbacks"] += 1
    return model_name


def get_cache_case_resolution_stats() -> dict[str, int]:
    """Return a copy of case-resolution instrumentation counters."""
    return dict(_CACHE_CASE_RESOLUTION_STATS)


def reset_cache_case_resolution_state() -> None:
    """Clear resolver memo and counters (primarily for tests)."""
    _CACHE_CASE_RESOLUTION_MEMO.clear()
    for key in _CACHE_CASE_RESOLUTION_STATS:
        _CACHE_CASE_RESOLUTION_STATS[key] = 0


def _wsl_reveal_in_explorer(path: Path, is_file: bool) -> bool:
    import subprocess
    if not _IS_WSL:
        return False
    try:
        windows_path = subprocess.run(
            ["wslpath", "-w", str(path)],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            check = True,
            timeout = 10,
        ).stdout.strip()
        if not windows_path:
            return False
        argument = f"/select,{windows_path}" if is_file else windows_path
        subprocess.Popen(["explorer.exe", argument])
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def reveal_in_file_manager(path: Path, expect_dir: bool = False) -> None:
    """Open the OS file manager with *path* selected (best effort per platform).

    Raises ``FileNotFoundError`` when the target is gone: the Linux branch falls
    back to the parent, which for a sandbox is the root holding every other
    chat's.

    ``expect_dir`` refuses anything that is not a real directory, symlinks
    included, since both would take the file branch and name that same parent.
    One ``lstat`` answers type and link-ness together, leaving no window between
    the checks (``is_dir()`` follows links; ``follow_symlinks = False`` is 3.13+
    only, and this runs on 3.10). Off by default: the cached-model reveal points
    at a file, and a symlinked one, as an HF cache snapshot is a link farm.
    """
    import stat as stat_module
    import subprocess

    if expect_dir:
        try:
            entry = os.lstat(path)  # No-follow, and the only stat here.
        except OSError as exc:
            raise FileNotFoundError(str(path)) from exc
        if not stat_module.S_ISDIR(entry.st_mode):
            raise FileNotFoundError(str(path))
        is_dir, is_file = True, False
    else:
        if not path.exists():
            raise FileNotFoundError(str(path))
        # Decided ONCE and then only read; each branch used to re-stat.
        is_dir = path.is_dir()
        is_file = not is_dir and path.is_file()
        if not is_dir and not is_file:
            raise FileNotFoundError(str(path))

    target = str(path)
    if sys.platform == "darwin":
        cmd = ["open", "-R", target] if is_file else ["open", target]
        subprocess.Popen(cmd)
    elif os.name == "nt":
        if is_file:
            subprocess.Popen(["explorer", f"/select,{target}"])
        else:
            os.startfile(target)  # noqa: S606 - local user's own file manager
    elif not _wsl_reveal_in_explorer(path, is_file):
        # No cross-desktop "select file" standard on Linux; open the directory.
        subprocess.Popen(["xdg-open", str(path.parent) if is_file else target])
