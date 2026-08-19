# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolve a usable Node.js executable at runtime.

The installer provisions an isolated Node under ``<UNSLOTH_HOME>/node`` but only
puts it on PATH for the *setup* process, never the user's shell. So backend code
that shells out to ``node`` at runtime (the OXC validator) cannot rely on PATH.
``resolve_node_executable`` prefers a version-adequate system Node, else the
managed isolated Node (same floor the installer applies: ^20.19 || >=22.12 || >=23).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from utils.subprocess_compat import windows_hidden_subprocess_kwargs

_NODE_VERSION_PROBE_TIMEOUT_SECONDS = 10


# Keep in sync with the setup scripts' Node floor: Get-NodeDecision (setup.ps1) /
# decide_node_source (setup.sh). Vite 8 needs Node ^20.19 || >=22.12 || >=23.
def _version_meets_floor(version: str) -> bool:
    """True iff a ``node -v`` string clears the installer's version bar."""
    match = re.match(r"v?(\d+)\.(\d+)", version.strip())
    if not match:
        return False
    major, minor = int(match.group(1)), int(match.group(2))
    return (major == 20 and minor >= 19) or (major == 22 and minor >= 12) or major >= 23


def managed_node_dir() -> Path:
    """Isolated Node install dir. Mirrors ``_find_llama_server_binary``: shares a
    parent with llama.cpp -- ``<STUDIO_HOME>`` in custom mode, else legacy ``~/.unsloth``."""
    legacy_node = Path.home() / ".unsloth" / "node"
    try:
        # Lazy import (mirrors _find_llama_server_binary) so this module stays
        # importable even if utils.paths cannot be loaded.
        from utils.paths.storage_roots import studio_root

        resolved = studio_root()
        legacy_studio = Path.home() / ".unsloth" / "studio"
        try:
            is_legacy = resolved.resolve() == legacy_studio.resolve()
        except (OSError, ValueError):
            is_legacy = resolved == legacy_studio
        return legacy_node if is_legacy else (resolved / "node")
    except (ImportError, OSError, ValueError):
        # Degraded env (utils.paths unavailable): still honor an explicit
        # STUDIO_HOME override before the legacy default, mirroring studio_root().
        override = (
            os.environ.get("UNSLOTH_STUDIO_HOME") or os.environ.get("STUDIO_HOME") or ""
        ).strip()
        if override:
            try:
                return Path(override).expanduser().resolve() / "node"
            except (OSError, ValueError):
                return Path(override).expanduser() / "node"
        return legacy_node


def managed_node_binary() -> Path:
    """Node executable in the isolated install: ``<dir>/node.exe`` on Windows, ``<dir>/bin/node`` else."""
    node_dir = managed_node_dir()
    if os.name == "nt":
        return node_dir / "node.exe"
    return node_dir / "bin" / "node"


def managed_node_bin_dir() -> Path | None:
    """Directory holding the isolated node/npm/npx executables, or None if not installed."""
    node_dir = managed_node_dir()
    bin_dir = node_dir if os.name == "nt" else node_dir / "bin"
    try:
        return bin_dir if bin_dir.is_dir() else None
    except OSError:
        return None


# Success-only memoization, like _resolved_node: the installer may finish after the
# first probe, so a negative verdict must not stick until restart.
_managed_node_ok: bool = False
_usable_node_cache: dict[str, bool] = {}


def _reset_managed_node_check() -> None:
    """Clear the memoized Node verdicts (used by tests)."""
    global _managed_node_ok
    _managed_node_ok = False
    _usable_node_cache.clear()


def _path_has_usable_node(path: str) -> bool:
    """Whether ``path`` provides both a floor-clearing ``node`` and an ``npx``.
    decide_node_source() installs the managed runtime unless both hold, so a host
    with node but no npm still needs it."""
    try:
        node = shutil.which("node", path = path)
        npx = shutil.which("npx", path = path)
    except OSError:
        return False
    if not node or not npx:
        return False
    if _usable_node_cache.get(node):
        return True
    ok = _node_version_ok(node)
    if ok:
        _usable_node_cache[node] = True
    return ok


def managed_node_usable() -> bool:
    """Whether the managed Node clears the version floor, mirroring the managed branch
    of resolve_node_executable(). Setup leaves an install in place when it picks the
    system runtime, so a stale dir must not win the lookup."""
    global _managed_node_ok
    if _managed_node_ok:
        return True
    binary = managed_node_binary()
    try:
        present = binary.is_file()
    except OSError:
        return False
    _managed_node_ok = present and _node_version_ok(str(binary))
    return _managed_node_ok


def path_with_managed_node(base_path: str | None = None) -> str:
    """``base_path`` (default: this process's PATH) with the managed Node bin dir
    prepended, unchanged when it is unusable or already there. The installer puts the
    isolated Node on PATH for setup only, so subprocesses must be handed it."""
    current = os.environ.get("PATH", "") if base_path is None else base_path
    bin_dir = managed_node_bin_dir()
    if bin_dir is None:
        return current
    # Never shadow a runtime the PATH already reaches (resolve_node_executable order).
    if _path_has_usable_node(current):
        return current
    if not managed_node_usable():
        return current
    bin_str = str(bin_dir)
    # An empty component means the working directory on POSIX; dropping it loses it.
    entries = current.split(os.pathsep) if current else []
    normalized = os.path.normcase(os.path.normpath(bin_str))
    if any(entry and os.path.normcase(os.path.normpath(entry)) == normalized for entry in entries):
        return current
    return os.pathsep.join([bin_str, *entries])


def _node_version_ok(executable: str) -> bool:
    """Run ``<executable> -v`` and check it clears the floor; False on any error."""
    try:
        result = subprocess.run(
            [executable, "-v"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = _NODE_VERSION_PROBE_TIMEOUT_SECONDS,
            **windows_hidden_subprocess_kwargs(),
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    return _version_meets_floor(result.stdout)


# Memoize ONLY a confirmed version-adequate executable: the installer runs in a
# separate process and may finish after the first probe here, so a negative /
# last-resort result must not be cached (it would stick until a backend restart).
_resolved_node: str | None = None


def _reset_resolved_node() -> None:
    """Clear the memoized executable (used by tests)."""
    global _resolved_node
    _resolved_node = None


def resolve_node_executable() -> str | None:
    """Resolve a usable node executable, or None.

    Order: version-adequate system ``node`` on PATH; else the managed isolated
    Node if adequate; else bare ``node`` (may be None). Only an adequate result
    is memoized, so a Node installed after the first probe is picked up live.
    """
    global _resolved_node
    if _resolved_node is not None:
        return _resolved_node

    system_node = shutil.which("node")
    if system_node and _node_version_ok(system_node):
        _resolved_node = system_node
        return _resolved_node

    managed = managed_node_binary()
    try:
        managed_present = managed.is_file()
    except OSError:
        managed_present = False
    if managed_present and _node_version_ok(str(managed)):
        _resolved_node = str(managed)
        return _resolved_node

    # Last-resort system node (may be None), NOT cached so a later install is picked up.
    return system_node
