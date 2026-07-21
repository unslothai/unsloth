# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Live, persisted Hugging Face cache routing for Unsloth Studio.

Hugging Face reads cache environment variables at import time.  Studio therefore
owns an explicit cache snapshot for each operation instead of trying to refresh
``huggingface_hub.constants`` in the long-running API process.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Mapping, Optional


CACHE_HOME_SETTING_KEY = "hugging_face_cache_home"
CACHE_HISTORY_SETTING_KEY = "hugging_face_cache_history"
MAX_CACHE_HISTORY = 16

CacheSource = Literal["default", "studio", "environment"]

_CACHE_ENV_KEYS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_XET_CACHE",
)
# Imported by storage_roots._setup_cache_env before Studio seeds defaults.
_EXPLICIT_CACHE_ENV = {
    key: value.strip()
    for key in _CACHE_ENV_KEYS
    if (value := os.environ.get(key)) is not None and value.strip()
}
_settings_lock = threading.RLock()
_spawn_env_lock = threading.RLock()


@dataclass(frozen = True)
class HuggingFaceCachePaths:
    cache_home: Path
    hub_cache: Path
    xet_cache: Path
    source: CacheSource
    environment_variable: Optional[str] = None

    @property
    def editable(self) -> bool:
        return self.source != "environment"

    @property
    def is_custom(self) -> bool:
        return self.source == "studio"

    def child_env(self, base: Optional[Mapping[str, str]] = None) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        # Do not rewrite HF_HOME. It also owns HF's token path, and credentials
        # must not be moved onto a removable cache volume.
        env["HF_HUB_CACHE"] = str(self.hub_cache)
        env["HF_XET_CACHE"] = str(self.xet_cache)
        env.pop("HUGGINGFACE_HUB_CACHE", None)
        return env


def _default_cache_home() -> Path:
    xdg = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    return (Path(xdg).expanduser() if xdg else Path.home() / ".cache") / "huggingface"


def _canonical(path: Path | str) -> Path:
    return Path(path).expanduser().resolve(strict = False)


def _environment_paths() -> Optional[HuggingFaceCachePaths]:
    if not _EXPLICIT_CACHE_ENV:
        return None
    explicit_home = _EXPLICIT_CACHE_ENV.get("HF_HOME")
    explicit_hub = _EXPLICIT_CACHE_ENV.get("HF_HUB_CACHE") or _EXPLICIT_CACHE_ENV.get(
        "HUGGINGFACE_HUB_CACHE"
    )
    explicit_xet = _EXPLICIT_CACHE_ENV.get("HF_XET_CACHE")
    default_home = _default_cache_home()
    hf_home = _canonical(explicit_home) if explicit_home else default_home
    hub = _canonical(explicit_hub) if explicit_hub else hf_home / "hub"
    xet = _canonical(explicit_xet) if explicit_xet else hf_home / "xet"
    controlling = next(
        key
        for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "HF_XET_CACHE")
        if key in _EXPLICIT_CACHE_ENV
    )
    # Settings describes model downloads, so an explicit hub path is the
    # displayed/opened location even when HF_HOME points somewhere else for
    # credentials or XET data.
    display_home = (
        (hub.parent if explicit_hub and hub.name.lower() == "hub" else hub)
        if explicit_hub
        else hf_home
    )
    return HuggingFaceCachePaths(display_home, hub, xet, "environment", controlling)


def _stored_cache_home() -> Optional[Path]:
    try:
        from storage.studio_db import get_app_setting
        value = get_app_setting(CACHE_HOME_SETTING_KEY, None)
    except Exception:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _canonical(value.strip())
    except (OSError, RuntimeError, ValueError):
        return None


def get_hf_cache_paths() -> HuggingFaceCachePaths:
    env_paths = _environment_paths()
    if env_paths is not None:
        return env_paths
    stored = _stored_cache_home()
    if stored is not None:
        return HuggingFaceCachePaths(stored, stored / "hub", stored / "xet", "studio")
    home = _default_cache_home()
    return HuggingFaceCachePaths(home, home / "hub", home / "xet", "default")


def active_hf_hub_cache() -> str:
    """Return the current hub cache as a string for library call kwargs."""

    return str(get_hf_cache_paths().hub_cache)


@contextmanager
def child_environment_for_spawn(environment: Mapping[str, str]) -> Iterator[None]:
    """Apply captured env before spawn imports the child entrypoint.

    Applying variables only inside the multiprocessing target can be too late
    for libraries that snapshot environment variables at import. The lock keeps
    this short parent-process override atomic through ``Process.start()``.
    """

    with _spawn_env_lock:
        missing = object()
        saved_environment: dict[str, str | object] = {}
        for key, value in environment.items():
            saved_environment[key] = os.environ.get(key, missing)
            os.environ[key] = value
        try:
            yield
        finally:
            for key, previous in saved_environment.items():
                if previous is missing:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = str(previous)


def initialize_hf_cache_environment() -> HuggingFaceCachePaths:
    """Seed import-time HF variables once during backend startup."""

    paths = get_hf_cache_paths()
    # Preserve an explicit HF_HOME, otherwise keep credentials at the platform
    # default while routing cache bytes through the selected home.
    os.environ.setdefault("HF_HOME", str(_default_cache_home()))
    os.environ["HF_HUB_CACHE"] = str(paths.hub_cache)
    os.environ["HF_XET_CACHE"] = str(paths.xet_cache)
    if "HUGGINGFACE_HUB_CACHE" not in _EXPLICIT_CACHE_ENV:
        os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
    for directory in (paths.hub_cache, paths.xet_cache):
        try:
            directory.mkdir(parents = True, exist_ok = True)
        except OSError:
            pass
    return paths


def _validate_cache_home(raw_path: str) -> Path:
    value = raw_path.strip()
    if not value:
        raise ValueError("Choose a cache folder.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise ValueError("The Hugging Face cache folder must be an absolute path.")
    try:
        resolved = candidate.resolve(strict = False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("The Hugging Face cache folder is invalid.") from exc

    if resolved.parent == resolved:
        raise ValueError("Choose a folder inside the filesystem or drive root.")
    try:
        from hub.storage.scan_folders import is_denied_system_path
        if is_denied_system_path(str(resolved)):
            raise ValueError("System folders cannot be used for model downloads.")
    except ImportError:
        pass

    parent = resolved.parent
    if not parent.exists() or not parent.is_dir():
        raise ValueError("The parent folder does not exist.")
    try:
        resolved.mkdir(exist_ok = True)
        if not resolved.is_dir():
            raise ValueError("The selected cache location is not a folder.")
        for child in (resolved / "hub", resolved / "xet"):
            child.mkdir(exist_ok = True)
            with tempfile.NamedTemporaryFile(prefix = ".unsloth-write-test-", dir = child):
                pass
    except PermissionError as exc:
        raise ValueError("Studio does not have permission to write to this folder.") from exc
    except OSError as exc:
        raise ValueError(f"Studio cannot use this cache folder: {exc}") from exc
    return resolved


def _stored_history() -> list[Path]:
    try:
        from storage.studio_db import get_app_setting
        raw = get_app_setting(CACHE_HISTORY_SETTING_KEY, [])
    except Exception:
        raw = []
    if not isinstance(raw, list):
        return []
    out: list[Path] = []
    seen: set[str] = set()
    for value in raw:
        if not isinstance(value, str) or not value.strip():
            continue
        try:
            path = _canonical(value)
        except (OSError, RuntimeError, ValueError):
            continue
        key = os.path.normcase(str(path))
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out[:MAX_CACHE_HISTORY]


def set_hf_cache_home(cache_home: Optional[str]) -> HuggingFaceCachePaths:
    if _environment_paths() is not None:
        raise RuntimeError("The Hugging Face cache location is managed by an environment variable.")
    with _settings_lock:
        previous = _stored_cache_home()
        next_home = _validate_cache_home(cache_home) if cache_home is not None else None
        history = _stored_history()
        if previous is not None and previous != next_home:
            history.insert(0, previous)
        deduped: list[str] = []
        seen: set[str] = set()
        for path in history:
            key = os.path.normcase(str(path))
            if key in seen or path == next_home:
                continue
            seen.add(key)
            deduped.append(str(path))
            if len(deduped) >= MAX_CACHE_HISTORY:
                break
        from storage.studio_db import upsert_app_settings

        upsert_app_settings(
            {
                CACHE_HOME_SETTING_KEY: str(next_home) if next_home is not None else None,
                CACHE_HISTORY_SETTING_KEY: deduped,
            }
        )
    # Inventory scans are cached independently from settings. Invalidate after
    # persistence so the next request sees both the new active root and history.
    from hub.utils.inventory_scan import invalidate_hf_cache_scans

    invalidate_hf_cache_scans()
    return get_hf_cache_paths()


def known_hf_cache_homes() -> list[Path]:
    paths = get_hf_cache_paths()
    stored = _stored_cache_home()
    candidates: list[Path] = []
    if paths.source != "environment":
        candidates.append(paths.cache_home)
    elif explicit_home := _EXPLICIT_CACHE_ENV.get("HF_HOME"):
        candidates.append(_canonical(explicit_home))
    if stored is not None:
        candidates.append(stored)
    candidates.extend([*_stored_history(), _default_cache_home()])
    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            canonical = _canonical(candidate)
        except (OSError, RuntimeError, ValueError):
            continue
        key = os.path.normcase(str(canonical))
        if key in seen:
            continue
        seen.add(key)
        out.append(canonical)
    return out


def known_hf_hub_caches() -> list[Path]:
    active = get_hf_cache_paths()
    out = [active.hub_cache]
    seen = {os.path.normcase(str(_canonical(active.hub_cache)))}
    for home in known_hf_cache_homes():
        hub = _canonical(home / "hub")
        key = os.path.normcase(str(hub))
        if key not in seen:
            seen.add(key)
            out.append(hub)
    return out


def cache_status(paths: Optional[HuggingFaceCachePaths] = None) -> dict:
    paths = paths or get_hf_cache_paths()
    available = paths.cache_home.is_dir()
    writable = available and os.access(paths.cache_home, os.W_OK | os.X_OK)
    free_bytes: Optional[int] = None
    if available:
        try:
            free_bytes = int(shutil.disk_usage(paths.cache_home).free)
        except OSError:
            pass
    return {
        "cache_home": str(paths.cache_home),
        "hub_cache": str(paths.hub_cache),
        "xet_cache": str(paths.xet_cache),
        "source": paths.source,
        "editable": paths.editable,
        "is_custom": paths.is_custom,
        "available": available,
        "writable": writable,
        "free_bytes": free_bytes,
        "environment_variable": paths.environment_variable,
    }
