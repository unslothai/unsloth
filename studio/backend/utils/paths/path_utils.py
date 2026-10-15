# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Path utilities for model and dataset handling."""

import os
import sys
from pathlib import Path
from typing import Callable, Iterable, Optional, TypeVar
import structlog
from loggers import get_logger

logger = get_logger(__name__)

# Opening a cloud placeholder recalls it; these attributes are readable without opening.
_WINDOWS_CONTENT_RECALL_ATTRIBUTES = 0x00001000 | 0x00040000 | 0x00400000


def file_contents_available_locally(path, stat_result = None) -> bool:
    """Whether opening *path* can read data without recalling a cloud placeholder. Non-Windows files have no ``st_file_attributes`` and are treated as local; an inaccessible path is not safe to open during inventory discovery."""
    try:
        info = stat_result if stat_result is not None else os.stat(path)
    except OSError:
        return False
    attributes = int(getattr(info, "st_file_attributes", 0) or 0)
    return not bool(attributes & _WINDOWS_CONTENT_RECALL_ATTRIBUTES)


# macOS stores xattrs in ._ companions on exFAT/FAT/SMB/NFS; only the magic identifies one,
# since a user's own ._model.gguf is a real model.
_MAGIC = b"\x00\x05\x16\x07"

PathLike = TypeVar("PathLike", str, Path)


def is_appledouble_name(path: str) -> bool:
    """A name test only: it decides which files are worth opening, never what one is."""
    return str(path).replace("\\", "/").rsplit("/", 1)[-1].startswith("._")


def has_appledouble_magic(path: Path) -> bool:
    """The four bytes ``file(1)`` reads to report "AppleDouble encoded Macintosh file"."""
    try:
        # Opening a FIFO blocks, so only regular files are checked.
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
    """Whether *paths* holds anything that is not Finder metadata, stopping at the first. Callers hand this a live ``glob``, which materializing would walk in full."""
    return any(not is_appledouble_metadata(Path(p)) for p in paths)


def _shadowed_name(path: str) -> str:
    head, _, name = str(path).replace("\\", "/").rpartition("/")
    return f"{head}/{name[2:]}" if head else name[2:]


def drop_shadowed_appledouble_names(
    # Optional[...] not PEP 604: no future annotations here and the floor is 3.9 (test gated).
    files: list[str],
    *,
    subject_key: Optional[Callable[[str], object]] = None,
) -> list[str]:
    """*files* without the ``._`` entries whose subject is present in the same listing. For remote listings, which carry no bytes to read, so a sole candidate survives whatever it is called. *subject_key* widens what counts as the subject, for files that come in sets."""
    key = subject_key or (lambda name: name)
    present = {key(f.replace("\\", "/")) for f in files}
    return [f for f in files if not (is_appledouble_name(f) and key(_shadowed_name(f)) in present)]


_CACHE_CASE_RESOLUTION_MEMO: dict[str, str] = {}

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
    """Normalize filesystem paths for cross-platform use: WSL maps drive-letter paths to ``/mnt/<drive>/...``, native Windows keeps the drive and normalizes separators, elsewhere slashes are forward-only."""
    if not path:
        return path

    if len(path) >= 3 and path[1] == ":" and path[2] in ("\\", "/"):
        if _IS_WSL:
            drive = path[0].lower()
            rest = path[3:].replace("\\", "/")
            return f"/mnt/{drive}/{rest}"
        return path.replace("\\", "/")

    return path.replace("\\", "/")


def wsl_automount_root() -> str:
    """DrvFs root WSL maps Windows drives under, with a trailing slash. Set via ``/etc/wsl.conf`` ``[automount] root``, so hard-coding ``/mnt/`` mistranslates drive paths on a host that moved it (``root = /`` puts C: at ``/c/``)."""
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
    """Separators change only for Windows-shaped paths: a POSIX backslash is a legal filename character."""
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
    """Whether path is a local filesystem path rather than a HuggingFace model identifier: local for /home/user/model, ./model or ~/model, not local for unsloth/llama-3.1-8b or microsoft/phi-2."""
    if not path:
        return False

    try:
        if Path(normalize_path(path)).expanduser().exists():
            return True
    except Exception:
        pass

    if path.count("/") == 1 and not path.startswith(("/", ".", "~")):
        return False

    return path.startswith(("/", ".", "~")) or ":" in path or "\\" in path or os.path.isabs(path)


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

    for suffix in [".safetensors", ".bin", ".json"]:
        if any_not_appledouble_metadata(cache_path.rglob(f"*{suffix}")):
            return True

    return False


def _hf_hub_cache_dir() -> Path:
    """Return HF cache root honoring HF_HUB_CACHE when available."""
    from utils.hf_cache_settings import get_hf_cache_paths
    return get_hf_cache_paths().hub_cache


def resolve_cached_repo_id_case(model_name: str, use_memo: bool = True) -> str:
    """Resolve repo_id to the exact casing already present in the local HF cache. Prefer the requested/canonical repo_id, but reuse a case-variant's exact cached spelling when one already exists, avoiding duplicate downloads while preserving user intent where possible."""
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

    if use_memo:
        cached = _CACHE_CASE_RESOLUTION_MEMO.get(model_name)
        if cached is not None:
            cached_path = cache_dir / f"models--{cached.replace('/', '--')}"
            if cached_path.is_dir():
                _CACHE_CASE_RESOLUTION_STATS["memo_hits"] += 1
                return cached
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


def _comparable_path(path, pathmod) -> str:
    """*path* spelled the way two paths are compared: on Windows without the extended-length
    prefix, which realpath keeps on one side only for a long path, and case folded."""
    text = os.fspath(path)
    if pathmod.sep == "\\":
        if text[:8].upper() == "\\\\?\\UNC\\":
            text = "\\\\" + text[8:]
        elif text[:4] == "\\\\?\\":
            text = text[4:]
    return pathmod.normcase(pathmod.normpath(text))


def is_path_within(
    path,
    root,
    *,
    allow_root: bool = False,
    pathmod = os.path,
) -> bool:
    """Uses commonpath, not a prefix test, so a sibling folder sharing a name prefix is not inside."""
    candidate, base = _comparable_path(path, pathmod), _comparable_path(root, pathmod)
    try:
        common = pathmod.commonpath([candidate, base])
    except ValueError:
        return False
    return common == base and (allow_root or candidate != base)


def same_path(
    left,
    right,
    *,
    pathmod = os.path,
) -> bool:
    """Whether two resolved paths are one, compared as ``is_path_within`` compares them."""
    return _comparable_path(left, pathmod) == _comparable_path(right, pathmod)


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
        subprocess.Popen(
            ["explorer.exe", "/select,", windows_path]
            if is_file
            else ["explorer.exe", windows_path]
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def reveal_in_file_manager(path: Path, expect_dir: bool = False) -> None:
    """Raises FileNotFoundError when gone, rather than letting Linux reveal the parent folder."""
    import stat as stat_module
    import subprocess

    if expect_dir:
        try:
            entry = os.lstat(path)
        except OSError as exc:
            raise FileNotFoundError(str(path)) from exc
        if not stat_module.S_ISDIR(entry.st_mode):
            raise FileNotFoundError(str(path))
        is_dir, is_file = True, False
    else:
        if not path.exists():
            raise FileNotFoundError(str(path))
        is_dir = path.is_dir()
        is_file = not is_dir and path.is_file()
        if not is_dir and not is_file:
            raise FileNotFoundError(str(path))

    target = str(path)
    if sys.platform == "darwin":
        cmd = ["open", "-R", target] if is_file else ["open", target]
        subprocess.Popen(cmd)
    elif os.name == "nt":
        if is_file and '"' not in target:
            subprocess.Popen(f'explorer /select,"{target}"')
        elif is_file:
            os.startfile(str(path.parent))  # noqa: S606 - local user's own file manager
        else:
            os.startfile(target)  # noqa: S606 - local user's own file manager
    elif not _wsl_reveal_in_explorer(path, is_file):
        # No cross-desktop select-file standard on Linux; open the directory.
        subprocess.Popen(["xdg-open", str(path.parent) if is_file else target])


# Viewer-opened types only: scripts or installers would execute, and these files are model-written.
DEFAULT_APP_OPEN_EXTENSIONS = frozenset(
    {
        ".pdf",
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".tsv",
        ".json",
        ".jsonl",
        ".xml",
        ".yaml",
        ".yml",
        ".log",
        ".rtf",
        ".docx",
        ".xlsx",
        ".pptx",
        ".odt",
        ".ods",
        ".odp",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".bmp",
        ".avif",
        ".tif",
        ".tiff",
        ".mp3",
        ".wav",
        ".flac",
        ".ogg",
        ".m4a",
        ".mp4",
        ".mov",
        ".webm",
        ".mkv",
        ".parquet",
        ".ipynb",
    }
)


_OPEN_STAGING_MAX_AGE_S = 24 * 60 * 60


def _prune_open_staging(root: Path) -> None:
    import shutil
    import time

    cutoff = time.time() - _OPEN_STAGING_MAX_AGE_S
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for entry in entries:
        try:
            if entry.lstat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors = True)
        except OSError:
            continue


def _opened_path(handle: int) -> Optional[str]:
    """Where an open file really is, or None where the OS can't say."""
    if sys.platform == "darwin":
        import fcntl
        return os.fsdecode(fcntl.fcntl(handle, fcntl.F_GETPATH, bytes(1024)).rstrip(b"\0"))
    proc = f"/proc/self/fd/{handle}"
    return os.readlink(proc) if os.path.islink(proc) else None


def _stage_for_open(path: Path, root: Optional[Path] = None) -> Path:
    """Opens without following links, then hard-links that inode into a private dir a swap cannot reach."""
    import shutil
    import stat as stat_module
    import tempfile

    from utils.paths.storage_roots import cache_root

    try:
        handle = os.open(
            path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        )
    except OSError as exc:
        raise FileNotFoundError(str(path)) from exc
    try:
        info = os.fstat(handle)
        if not stat_module.S_ISREG(info.st_mode):
            raise FileNotFoundError(str(path))
        if root is not None:
            real = _opened_path(handle)
            if real is None:
                # By name, so it must still be the file that was opened (Windows junctions).
                real = os.path.realpath(path)
                same = os.stat(real)
                if (same.st_dev, same.st_ino) != (info.st_dev, info.st_ino):
                    raise FileNotFoundError(str(path))
            if not Path(real).is_relative_to(os.path.realpath(root)):
                raise FileNotFoundError(str(path))
        staging = cache_root() / "open-staging"
        staging.mkdir(parents = True, exist_ok = True)
        _prune_open_staging(staging)
        target = Path(tempfile.mkdtemp(dir = staging)) / path.name
        try:
            os.link(path, target, follow_symlinks = False)
            linked = os.lstat(target)
            # Swapped since the open: copy what was opened.
            if (linked.st_dev, linked.st_ino) != (info.st_dev, info.st_ino):
                target.unlink()
                raise OSError("file changed")
        except (OSError, NotImplementedError):
            with os.fdopen(os.dup(handle), "rb") as source, open(target, "xb") as copy:
                shutil.copyfileobj(source, copy)
        return target
    finally:
        os.close(handle)


def open_in_default_app(path: Path, root: Optional[Path] = None) -> None:
    """Refuses extensions outside DEFAULT_APP_OPEN_EXTENSIONS and symlinks, and opens a staged name."""
    import stat as stat_module
    import subprocess

    if path.suffix.lower() not in DEFAULT_APP_OPEN_EXTENSIONS:
        raise PermissionError(str(path))
    try:
        entry = os.lstat(path)
    except OSError as exc:
        raise FileNotFoundError(str(path)) from exc
    if not stat_module.S_ISREG(entry.st_mode):
        raise FileNotFoundError(str(path))
    target = str(_stage_for_open(path, root))
    if sys.platform == "darwin":
        subprocess.Popen(["open", target])
    elif os.name == "nt":
        os.startfile(target)  # noqa: S606 - local user's own default app
    elif _IS_WSL:
        windows_path = subprocess.run(
            ["wslpath", "-w", target],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            check = True,
            timeout = 10,
        ).stdout.strip()
        subprocess.Popen(["explorer.exe", windows_path])
    else:
        subprocess.Popen(["xdg-open", target])


# macOS _PC_CASE_SENSITIVE, which Python has no name for.
_PC_CASE_SENSITIVE = 11


def macos_volume_ignores_case(path: str) -> bool:
    """Whether the macOS volume holding ``path`` (or its nearest folder that exists) ignores case,
    as APFS and HFS+ do unless formatted case-sensitive. True off macOS, where only a test acting
    as macOS asks."""
    if sys.platform != "darwin":
        return True
    probe = path
    while True:
        try:
            return os.pathconf(probe, _PC_CASE_SENSITIVE) == 0
        except FileNotFoundError:
            parent = os.path.dirname(probe)
            if not parent or parent == probe:
                return True
            probe = parent
        except (OSError, ValueError):
            return True
