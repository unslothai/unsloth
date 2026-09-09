# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persist and validate the llama.cpp directory selected in Unsloth settings."""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

CUSTOM_LLAMA_CPP_PATH_SETTING_KEY = "custom_llama_cpp_path"
MAX_CUSTOM_LLAMA_CPP_PATH_LENGTH = 32767
MANAGED_LLAMA_CPP_PATH_MARKER = "UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH"

_settings_lock = threading.RLock()
_path_revision = 0


def mark_managed_llama_cpp_path(directory: Path | str) -> bool:
    """Mark Unsloth's inherited install path without hiding a real env override."""
    configured = os.environ.get("UNSLOTH_LLAMA_CPP_PATH", "").strip()
    if not configured:
        os.environ.pop(MANAGED_LLAMA_CPP_PATH_MARKER, None)
        return False
    try:
        managed = Path(directory).expanduser().resolve(strict = False)
        inherited = Path(configured).expanduser().resolve(strict = False)
        is_managed = inherited == managed
    except (OSError, RuntimeError, ValueError):
        is_managed = False
    if is_managed:
        os.environ[MANAGED_LLAMA_CPP_PATH_MARKER] = "1"
    else:
        os.environ.pop(MANAGED_LLAMA_CPP_PATH_MARKER, None)
    return is_managed


@contextmanager
def llama_cpp_path_selection_guard() -> Iterator[None]:
    """Serialize a runtime path snapshot with a settings write.

    Model loads and UI saves share this lock so reload status sees one snapshot.
    """
    with _settings_lock:
        yield


def llama_server_binary_name(platform: Optional[str] = None) -> str:
    return "llama-server.exe" if (platform or sys.platform) == "win32" else "llama-server"


def llama_server_candidates(
    directory: Path | str, *, platform: Optional[str] = None
) -> tuple[Path, ...]:
    """Supported llama.cpp build layouts, in the runtime's search order."""
    root = Path(directory)
    binary_name = llama_server_binary_name(platform)
    candidates = [
        root / binary_name,
        root / "build" / "bin" / binary_name,
    ]
    if (platform or sys.platform) == "win32":
        candidates.append(root / "build" / "bin" / "Release" / binary_name)
    return tuple(candidates)


def _usable_binary(path: Path, *, platform: Optional[str] = None) -> bool:
    try:
        if not path.is_file():
            return False
    except OSError:
        return False
    return (platform or sys.platform) == "win32" or os.access(path, os.X_OK)


def resolve_llama_server_binary(
    directory: Path | str, *, platform: Optional[str] = None
) -> Optional[Path]:
    """Return the first executable llama-server in a supported layout."""
    return next(
        (
            candidate
            for candidate in llama_server_candidates(directory, platform = platform)
            if _usable_binary(candidate, platform = platform)
        ),
        None,
    )


def get_stored_custom_llama_cpp_path() -> Optional[Path]:
    """The Unsloth-selected directory, or ``None`` when automatic discovery is active."""
    try:
        from storage.studio_db import get_app_setting
        value = get_app_setting(CUSTOM_LLAMA_CPP_PATH_SETTING_KEY, None)
    except Exception:
        # A settings DB problem must not take the bundled runtime down with it.
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > MAX_CUSTOM_LLAMA_CPP_PATH_LENGTH:
        return None
    return Path(value).expanduser()


def _environment_override() -> tuple[Optional[str], Optional[str], bool]:
    """``(path, variable, direct_binary)`` for the existing environment pins."""
    direct = os.environ.get("LLAMA_SERVER_PATH", "").strip()
    if direct:
        return direct, "LLAMA_SERVER_PATH", True
    directory = os.environ.get("UNSLOTH_LLAMA_CPP_PATH", "").strip()
    if directory and os.environ.get(MANAGED_LLAMA_CPP_PATH_MARKER) != "1":
        return directory, "UNSLOTH_LLAMA_CPP_PATH", False
    return None, None, False


def custom_llama_cpp_path_source() -> str:
    """The active custom-path authority: environment, studio, or default."""
    env_path, _variable, _direct = _environment_override()
    if env_path is not None:
        return "environment"
    if get_stored_custom_llama_cpp_path() is not None:
        return "studio"
    return "default"


def _canonical_directory(value: str) -> Path:
    raw = value.strip()
    if not raw:
        raise ValueError("Choose a llama.cpp folder or use the bundled runtime.")
    if len(raw) > MAX_CUSTOM_LLAMA_CPP_PATH_LENGTH:
        raise ValueError("The llama.cpp folder path is too long.")
    try:
        directory = Path(raw).expanduser().resolve(strict = True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ValueError("The llama.cpp folder does not exist or cannot be accessed.") from exc
    if not directory.is_dir():
        raise ValueError("The custom llama.cpp path must be a folder.")
    if resolve_llama_server_binary(directory) is None:
        binary_name = llama_server_binary_name()
        raise ValueError(
            f"No executable {binary_name} was found in that folder or its build/bin directory."
        )
    return directory


def set_custom_llama_cpp_path(value: Optional[str]) -> Optional[Path]:
    """Store a validated directory. ``None`` restores automatic discovery."""
    global _path_revision
    env_path, variable, _direct = _environment_override()
    if env_path is not None:
        raise RuntimeError(f"The llama.cpp path is managed by the {variable} environment variable.")
    directory = _canonical_directory(value) if value is not None else None
    with _settings_lock:
        from storage.studio_db import upsert_app_settings
        upsert_app_settings(
            {CUSTOM_LLAMA_CPP_PATH_SETTING_KEY: (str(directory) if directory is not None else None)}
        )
        _path_revision += 1
    return directory


def custom_llama_cpp_path_revision() -> int:
    """In-process revision used to retire sidecars launched before a path save."""
    with _settings_lock:
        return _path_revision


def custom_llama_cpp_path_status() -> dict:
    """UI payload describing the effective custom-path selection."""
    env_path, variable, direct_binary = _environment_override()
    source = "default"
    path: Optional[Path] = None
    binary: Optional[Path] = None

    if env_path is not None:
        source = "environment"
        path = Path(env_path).expanduser()
        if direct_binary:
            binary = path if _usable_binary(path) else None
        else:
            binary = resolve_llama_server_binary(path)
    else:
        path = get_stored_custom_llama_cpp_path()
        if path is not None:
            source = "studio"
            binary = resolve_llama_server_binary(path)

    return {
        "path": str(path) if path is not None else None,
        "source": source,
        "editable": source != "environment",
        "available": source == "default" or binary is not None,
        "resolved_binary": str(binary) if binary is not None else None,
        "environment_variable": variable,
    }
