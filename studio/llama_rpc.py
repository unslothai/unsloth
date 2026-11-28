# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where the managed llama.cpp bundle and its ``ggml-rpc-server`` live. Standard library only."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

_RPC_SERVER_NAMES = ("ggml-rpc-server", "rpc-server", "ggml-rpc-server.exe", "rpc-server.exe")
_WINDOWS_EXECUTABLE_SUFFIXES = (".exe", ".com", ".bat", ".cmd")
_BUNDLE_SUBDIRS = (("build", "bin"), ("build", "bin", "Release"), ("bin",), ())
_DEFAULT_STUDIO_ROOT = Path.home() / ".unsloth" / "studio"


_MASTER_ROOT_NOTE = ".unsloth-master-root"


def _resolved(value: str) -> Path:
    path = Path(value).expanduser()
    try:
        return path.resolve()
    except (OSError, RuntimeError, ValueError):
        return path


def _studio_root() -> Path:
    for var in ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME"):
        value = (os.environ.get(var) or "").strip()
        if value:
            return Path(value).expanduser()
    return _DEFAULT_STUDIO_ROOT


def _master_root(studio: Path) -> Optional[Path]:
    """UNSLOTH_HOME, else the root setup recorded in the Studio tree; llama.cpp sits beside studio/ there."""
    override = (os.environ.get("UNSLOTH_HOME") or "").strip()
    if override:
        return _resolved(override)
    try:
        recorded = (studio / "share" / _MASTER_ROOT_NOTE).read_text(encoding = "utf-8").strip()
    except (OSError, ValueError):
        return None
    if not recorded:
        return None
    master = _resolved(recorded)
    try:
        here = studio.resolve()
        # Honoured as storage_roots honours it: only while it exists, holds studio, and is not the legacy tree.
        if (
            master.is_dir()
            and (here == master or master in here.parents)
            and here != _DEFAULT_STUDIO_ROOT.resolve()
        ):
            return master
    except (OSError, RuntimeError, ValueError):
        pass
    return None


def _on_windows(windows: Optional[bool] = None) -> bool:
    return (os.name == "nt") if windows is None else bool(windows)


def _is_executable_file(path, windows: Optional[bool] = None) -> bool:
    """Whether this host would run `path`. POSIX means the exec bit; Windows means the suffix."""
    try:
        if not os.path.isfile(path):
            return False
        if _on_windows(windows):
            return os.path.splitext(str(path))[1].lower() in _WINDOWS_EXECUTABLE_SUFFIXES
        return os.access(path, os.X_OK)
    except OSError:
        return False


def rpc_server_names(windows: Optional[bool] = None) -> Tuple[str, ...]:
    """`_RPC_SERVER_NAMES` with the platform's own spelling first."""
    suffixed = tuple(
        n for n in _RPC_SERVER_NAMES if n.lower().endswith(_WINDOWS_EXECUTABLE_SUFFIXES)
    )
    plain = tuple(n for n in _RPC_SERVER_NAMES if n not in suffixed)
    return (suffixed + plain) if _on_windows(windows) else (plain + suffixed)


def llama_bundle_dir() -> Path:
    """The managed llama.cpp bundle, resolved as the installer resolves it."""
    override = (os.environ.get("UNSLOTH_LLAMA_CPP_PATH") or "").strip()
    if override:
        return Path(override).expanduser()
    root = _studio_root()
    master = _master_root(root)
    if master is not None:
        return master / "llama.cpp"
    try:
        is_default = root.resolve() == _DEFAULT_STUDIO_ROOT.resolve()
    except (OSError, ValueError, RuntimeError):
        is_default = root == _DEFAULT_STUDIO_ROOT
    if not is_default:
        return root / "llama.cpp"
    return Path.home() / ".unsloth" / "llama.cpp"


def _find_in_bundle(
    root: Path,
    names: Tuple[str, ...],
    executable: bool = False,
) -> Optional[Path]:
    for parts in _BUNDLE_SUBDIRS:
        base = root.joinpath(*parts) if parts else root
        for name in names:
            candidate = base / name
            try:
                if not candidate.is_file():
                    continue
                if executable and not _is_executable_file(candidate):
                    continue
            except OSError:
                continue
            return candidate
    return None


def rpc_server_location() -> Tuple[Optional[str], Path]:
    """(ggml-rpc-server, the bundle root it came from); the root falls back to the managed one."""
    for root in (llama_bundle_dir(), Path.home() / "src" / "llamacpp-rpc"):
        found = _find_in_bundle(root, rpc_server_names(), executable = True)
        if found is not None:
            return str(found), root
    found = shutil.which("ggml-rpc-server") or shutil.which("rpc-server")
    if found is None:
        return None, llama_bundle_dir()
    parent = Path(found).parent
    return found, parent.parent if parent.name == "bin" else parent


def rpc_server_binary() -> Optional[str]:
    """Path to ggml-rpc-server, or None."""
    return rpc_server_location()[0]
