# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Path validators and storage roots for the Hub layer."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from loggers import get_logger

logger = get_logger(__name__)


def _infer_studio_home_from_venv() -> Optional[Path]:
    try:
        prefix = Path(sys.prefix).resolve()
    except (OSError, ValueError):
        return None
    if prefix.name != "unsloth_studio":
        return None
    candidate = prefix.parent
    shim_name = "unsloth.exe" if os.name == "nt" else "unsloth"
    try:
        if (candidate / "share" / "studio.conf").is_file() or (
            candidate / "bin" / shim_name
        ).is_file():
            return candidate
    except OSError:
        return None
    return None


def studio_root() -> Path:
    override = (os.environ.get("UNSLOTH_STUDIO_HOME") or "").strip()
    if not override:
        override = (os.environ.get("STUDIO_HOME") or "").strip()
    if override:
        try:
            return Path(override).expanduser().resolve()
        except (OSError, ValueError):
            return Path(override).expanduser()
    inferred = _infer_studio_home_from_venv()
    if inferred is not None:
        return inferred
    return Path.home() / ".unsloth" / "studio"


def cache_root() -> Path:
    return studio_root() / "cache"


def assets_root() -> Path:
    return studio_root() / "assets"


def datasets_root() -> Path:
    return assets_root() / "datasets"


def dataset_uploads_root() -> Path:
    return datasets_root() / "uploads"


def recipe_datasets_root() -> Path:
    return datasets_root() / "recipes"


def outputs_root() -> Path:
    return studio_root() / "outputs"


def exports_root() -> Path:
    return studio_root() / "exports"


def tmp_root() -> Path:
    return Path(tempfile.gettempdir()) / "unsloth-studio"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents = True, exist_ok = True)
    return path


def legacy_hf_cache_dir() -> Path:
    return cache_root() / "huggingface" / "hub"


def hf_default_cache_dir() -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub"


def _is_wsl() -> bool:
    if sys.platform == "win32":
        return False
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except Exception:
        return False


_IS_WSL = _is_wsl()


def _wsl_automount_root() -> str:
    """DrvFs root under which WSL maps Windows drives, with a trailing slash.

    Defaults to ``/mnt/`` but is user-configurable via ``/etc/wsl.conf``
    (``[automount] root``), so hard-coding ``/mnt/`` mistranslates Windows paths
    on a host with a custom root (e.g. ``root = /`` → ``C:`` at ``/c/``)."""
    default = "/mnt/"
    if not _IS_WSL:
        return default
    try:
        import configparser

        parser = configparser.ConfigParser(inline_comment_prefixes = ("#", ";"))
        parser.read("/etc/wsl.conf")
        root = parser.get("automount", "root", fallback = "").strip().strip("\"'")
    except Exception:
        return default
    if not root:
        return default
    return root if root.endswith("/") else f"{root}/"


_WSL_AUTOMOUNT_ROOT = _wsl_automount_root()


def normalize_path(path: str) -> str:
    if not path:
        return path
    if len(path) >= 3 and path[1] == ":" and path[2] in ("\\", "/"):
        if _IS_WSL:
            drive = path[0].lower()
            rest = path[3:].replace("\\", "/")
            return f"{_WSL_AUTOMOUNT_ROOT}{drive}/{rest}"
        return path.replace("\\", "/")
    return path.replace("\\", "/")


def _host_path(path: str | Path) -> Path:
    return Path(normalize_path(str(path))).expanduser()


def is_local_path(path: str) -> bool:
    if not path:
        return False
    normalized = normalize_path(path)
    has_local_syntax = (
        path.startswith(("/", ".", "~"))
        or ":" in path
        or "\\" in path
        or os.path.isabs(path)
        or os.path.isabs(normalized)
    )
    if path.count("/") == 1 and not has_local_syntax:
        return False
    try:
        if has_local_syntax and Path(normalized).expanduser().exists():
            return True
    except Exception:
        pass
    return has_local_syntax


_VALID_REPO_ID_SEGMENT = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?$")
_MAX_REPO_ID_LENGTH = 96


def is_valid_repo_id(repo_id: str) -> bool:
    """Validate Hugging Face ``repo_name`` or ``namespace/repo_name`` IDs."""
    if not repo_id or repo_id != repo_id.strip():
        return False
    if repo_id.endswith(".git"):
        return False
    if "--" in repo_id or ".." in repo_id:
        return False
    segments = repo_id.split("/")
    if len(segments) not in (1, 2):
        return False
    # Match huggingface_hub.validate_repo_id: the 96-char limit applies per
    # segment (repo name / namespace), not to the whole "namespace/repo_name"
    # string, so long-but-valid repo names are not falsely rejected.
    return all(
        segment not in ("", ".", "..")
        and len(segment) <= _MAX_REPO_ID_LENGTH
        and _VALID_REPO_ID_SEGMENT.fullmatch(segment) is not None
        for segment in segments
    )


_GGUF_VARIANT_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_MAX_GGUF_VARIANT_LENGTH = 512


def is_valid_gguf_variant(variant: str) -> bool:
    """Validate Hub GGUF variant keys.

    Known quant labels are short tokens (``Q4_K_M``), but unknown GGUF layouts
    use a snapshot-relative key derived from the filename and may contain
    slashes or spaces.
    """
    if not variant or variant != variant.strip():
        return False
    if len(variant) > _MAX_GGUF_VARIANT_LENGTH:
        return False
    if _GGUF_VARIANT_CONTROL_CHARS.search(variant) or not variant.isprintable():
        return False
    normalized = variant.replace("\\", "/")
    return all(segment not in ("", ".", "..") for segment in normalized.split("/"))


def ollama_model_dirs() -> list[Path]:
    """Return Ollama model directories that exist on disk."""
    dirs: list[Path] = []
    seen: set[str] = set()

    def _add(p: Path | str) -> None:
        try:
            expanded = _host_path(p)
            resolved = expanded.resolve()
            is_dir = expanded.is_dir()
        except (OSError, RuntimeError, ValueError):
            return
        key = str(resolved)
        if key in seen or not is_dir:
            return
        seen.add(key)
        dirs.append(expanded)

    ollama_env = os.environ.get("OLLAMA_MODELS")
    if ollama_env:
        _add(ollama_env)
    _add(Path.home() / ".ollama" / "models")
    _add(Path("/usr/share/ollama/.ollama/models"))
    _add(Path("/var/lib/ollama/.ollama/models"))
    return dirs


# Per-process memo for resolve_cached_repo_id_case. Bounded LRU so a long-lived
# process touching many repo ids can't grow it without limit; evicted cold
# entries simply recompute on next use.
_CACHE_CASE_RESOLUTION_MEMO_MAX = 512
_CACHE_CASE_RESOLUTION_MEMO: "OrderedDict[tuple[str, str], str]" = OrderedDict()
_CACHE_CASE_RESOLUTION_LOCK = threading.Lock()


def _memo_get(memo_key: tuple[str, str]) -> Optional[str]:
    with _CACHE_CASE_RESOLUTION_LOCK:
        value = _CACHE_CASE_RESOLUTION_MEMO.get(memo_key)
        if value is not None:
            _CACHE_CASE_RESOLUTION_MEMO.move_to_end(memo_key)
        return value


def _memo_set(memo_key: tuple[str, str], value: str) -> None:
    with _CACHE_CASE_RESOLUTION_LOCK:
        _CACHE_CASE_RESOLUTION_MEMO[memo_key] = value
        _CACHE_CASE_RESOLUTION_MEMO.move_to_end(memo_key)
        while len(_CACHE_CASE_RESOLUTION_MEMO) > _CACHE_CASE_RESOLUTION_MEMO_MAX:
            _CACHE_CASE_RESOLUTION_MEMO.popitem(last = False)


def _memo_drop(memo_key: tuple[str, str]) -> None:
    with _CACHE_CASE_RESOLUTION_LOCK:
        _CACHE_CASE_RESOLUTION_MEMO.pop(memo_key, None)


def _hf_hub_cache_dir() -> Path:
    from utils.hf_cache_settings import get_hf_cache_paths

    return get_hf_cache_paths().hub_cache


def _hf_hub_cache_dirs() -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        key = str(resolved)
        if key in seen or not resolved.is_dir():
            return
        seen.add(key)
        roots.append(resolved)

    from utils.hf_cache_settings import known_hf_hub_caches

    for configured in known_hf_hub_caches():
        _add(configured)
    try:
        _add(legacy_hf_cache_dir())
        _add(hf_default_cache_dir())
    except Exception as exc:
        logger.debug("Could not enumerate secondary HF cache roots: %s", exc)
    return roots


def lmstudio_model_dirs() -> list[Path]:
    dirs: list[Path] = []
    seen: set[str] = set()

    def _add(path: Path | str) -> None:
        try:
            expanded = _host_path(path)
            resolved = expanded.resolve()
        except (OSError, RuntimeError, ValueError):
            return
        key = str(resolved)
        if key in seen or not expanded.is_dir():
            return
        seen.add(key)
        dirs.append(expanded)

    settings_path = Path.home() / ".lmstudio" / "settings.json"
    if settings_path.is_file():
        try:
            settings = json.loads(settings_path.read_text(encoding = "utf-8"))
            downloads = settings.get("downloadsFolder", "")
            if downloads:
                _add(downloads)
        except Exception:
            pass
    _add(Path.home() / ".lmstudio" / "models")
    _add(Path.home() / ".cache" / "lm-studio" / "models")
    return dirs


def well_known_model_dirs() -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(lmstudio_model_dirs())
    candidates.extend(ollama_model_dirs())
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    candidates.append(Path.home() / "models")
    candidates.append(Path.home() / "Models")

    out: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        key = str(resolved)
        if key in seen or not resolved.is_dir():
            continue
        seen.add(key)
        out.append(resolved)
    return out


def _assert_contained(resolved: Path, root: Path) -> None:
    try:
        resolved_real = Path(os.path.realpath(resolved))
        root_real = Path(os.path.realpath(root))
    except OSError as exc:
        raise ValueError(f"path resolution failed: {exc}") from exc
    try:
        resolved_real.relative_to(root_real)
    except ValueError as exc:
        raise ValueError(f"path escapes root: {resolved!s}") from exc


def path_is_same_or_child(path: Path, root: Path) -> bool:
    """True when *path* is *root* or lives beneath it.

    Compares real (symlink-resolved, case-normalized) paths so the check holds
    through symlinks and on case-insensitive filesystems, where a plain
    ``Path.is_relative_to`` would miss a casing-only match. Returns False on any
    resolution error rather than raising.
    """
    try:
        path_real = os.path.normcase(os.path.realpath(str(path)))
        root_real = os.path.normcase(os.path.realpath(str(root)))
        return os.path.commonpath([path_real, root_real]) == root_real
    except (OSError, ValueError):
        return False


def resolve_dataset_path(path_value: str) -> Path:
    raw = str(path_value or "").strip()
    if "\x00" in raw:
        raise ValueError("dataset path may not contain null bytes")
    # Normalize first so Windows/UNC and backslash paths resolve like the rest
    # of the Hub path layer (e.g. C:\data -> /mnt/c/data on WSL) and a
    # backslashed '..' is caught by the traversal guard below.
    normalized = normalize_path(raw)
    path = Path(normalized).expanduser()
    if ".." in path.parts:
        raise ValueError(f"dataset path may not contain '..' segments: {raw!r}")
    if path.is_absolute():
        for root in (datasets_root(), dataset_uploads_root(), recipe_datasets_root()):
            try:
                _assert_contained(path, root)
                return path
            except ValueError:
                continue
        raise ValueError(f"dataset path must be relative or under a dataset root: {raw!r}")

    parts = [part for part in Path(normalized).parts if part not in ("", ".")]
    if parts[:2] == ["assets", "datasets"]:
        parts = parts[2:]
    if parts and parts[0] == "uploads":
        cleaned = Path(*parts[1:]) if len(parts) > 1 else Path()
        return dataset_uploads_root() / cleaned
    if parts and parts[0] == "recipes":
        cleaned = Path(*parts[1:]) if len(parts) > 1 else Path()
        return recipe_datasets_root() / cleaned

    cleaned = Path(*parts) if parts else Path()
    candidates = [
        dataset_uploads_root() / cleaned,
        recipe_datasets_root() / cleaned,
        datasets_root() / cleaned,
        dataset_uploads_root() / cleaned.name,
        recipe_datasets_root() / cleaned.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_cached_repo_id_case(
    model_name: str,
    use_memo: bool = True,
    repo_type: str = "model",
) -> str:
    """Resolve repo_id to the exact casing already present in local HF cache.

    Prefers the requested casing, but if a case-variant already exists in
    local HF cache, reuses that exact cached spelling so we don't trigger
    a duplicate download.
    """
    if not model_name or "/" not in model_name:
        return model_name

    cache_dirs = _hf_hub_cache_dirs()
    if not cache_dirs:
        return model_name

    prefix = f"{repo_type}s--"
    expected_dir = f"{prefix}{model_name.replace('/', '--')}"
    memo_key = (repo_type, model_name)

    for cache_dir in cache_dirs:
        exact_path = cache_dir / expected_dir
        if exact_path.is_dir():
            if use_memo:
                _memo_set(memo_key, model_name)
            return model_name

    if use_memo:
        cached = _memo_get(memo_key)
        if cached is not None:
            if any(
                (cache_dir / f"{prefix}{cached.replace('/', '--')}").is_dir()
                for cache_dir in cache_dirs
            ):
                return cached
            _memo_drop(memo_key)

    expected_lower = expected_dir.lower()
    try:
        candidates: set[str] = set()
        for cache_dir in cache_dirs:
            for entry in cache_dir.iterdir():
                if not entry.is_dir():
                    continue
                if entry.name.lower() != expected_lower:
                    continue
                # The lowercased full-name match already proves the prefix
                # matches; a case-sensitive startswith would reject a mixed-case
                # imported dir such as Models--Org--Repo.
                repo_part = entry.name[len(prefix) :]
                if not repo_part:
                    continue
                candidates.add(repo_part.replace("--", "/"))

        if candidates:
            resolved = sorted(candidates)[0]
            if use_memo:
                _memo_set(memo_key, resolved)
            return resolved
    except Exception as exc:
        logger.debug(f"resolve_cached_repo_id_case failed for {model_name!r}: {exc}")

    return model_name


__all__ = [
    "assets_root",
    "cache_root",
    "dataset_uploads_root",
    "datasets_root",
    "ensure_dir",
    "exports_root",
    "hf_default_cache_dir",
    "is_local_path",
    "is_valid_gguf_variant",
    "is_valid_repo_id",
    "legacy_hf_cache_dir",
    "lmstudio_model_dirs",
    "normalize_path",
    "ollama_model_dirs",
    "outputs_root",
    "path_is_same_or_child",
    "recipe_datasets_root",
    "resolve_cached_repo_id_case",
    "resolve_dataset_path",
    "studio_root",
    "tmp_root",
    "well_known_model_dirs",
]
