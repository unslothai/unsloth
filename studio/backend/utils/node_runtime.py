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
# Module-level so the Windows-only npx branch below stays reachable from tests.
_IS_WINDOWS = os.name == "nt"


# Keep in sync with the setup scripts' floors, decide_node_source (setup.sh) / Get-NodeDecision
# (setup.ps1): Vite 8 needs Node ^20.19 || >=22.12 || >=23, and both require npm >= 11 before
# accepting a system runtime.
_NPM_MAJOR_FLOOR = 11


def _npm_meets_floor(version: str) -> bool:
    """True iff an ``npm -v`` string clears the installer's npm floor."""
    match = re.match(r"v?(\d+)", version.strip())
    return bool(match) and int(match.group(1)) >= _NPM_MAJOR_FLOOR


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


# Success-only memoization, like _resolved_node: the installer may finish
_managed_node_ok: bool = False
_usable_node_cache: dict[tuple[str, str | None], bool] = {}


def _reset_managed_node_check() -> None:
    """Clear the memoized Node verdicts (used by tests)."""
    global _managed_node_ok
    _managed_node_ok = False
    _usable_node_cache.clear()


def _path_has_usable_node(
    path: str,
    require_npm: bool = True,
    require_npx: bool = True,
) -> bool:
    """Whether ``path`` provides what a stdio command actually uses. The installers gate
    on node plus npm and never look at npx, so each launcher is checked against what it
    needs: node alone, node plus npm, or node plus the npx that launches it."""
    try:
        node = shutil.which("node", path = path)
        npm = shutil.which("npm", path = path) if require_npm else None
        npx = shutil.which("npx", path = path) if require_npx else None
    except OSError:
        return False
    if not node:
        return False
    if require_npm and not npm:
        return False
    if require_npx and not npx:
        return False
    launcher = npx if require_npx else npm
    if _IS_WINDOWS and launcher:
        # npm's generated npm.cmd and npx.cmd both run the node.exe beside them when
        # there is one, so that is the runtime to validate, not whatever ``node``
        # resolves to first.
        sibling = os.path.join(os.path.dirname(launcher), "node.exe")
        try:
            if os.path.isfile(sibling):
                node = sibling
        except OSError:
            return False
    if not _probe_ok(node, _node_version_ok, path):
        return False
    # The installers' npm floor still applies to an npx-only PATH: npx-cli.js hands off to
    # the npm library beside it, so ``npx -v`` prints that npm's version and stands in for
    # the missing npm launcher. Falling back to it keeps the floor instead of skipping it.
    floor_launcher = npm if require_npm else (npx if require_npx else None)
    return _probe_ok(floor_launcher, _npm_version_ok, path) if floor_launcher else True


def _probe_ok(
    executable: str,
    check,
    path: str | None = None,
) -> bool:
    """Version check for one executable, memoized on success (see _usable_node_cache).
    The PATH is part of the key: npm and npx are ``#!/usr/bin/env node`` scripts, so the
    same shim resolves a different runtime under a different PATH and can clear the floor
    on one and fail it on another. Keying on the executable alone would let the first
    PATH that passed answer for every later one."""
    cache_key = (executable, path)
    if _usable_node_cache.get(cache_key):
        return True
    ok = check(executable, path)
    if ok:
        _usable_node_cache[cache_key] = True
    return ok


def _managed_probe_path() -> str | None:
    """The managed bin dir itself: its npm shim needs the node sitting beside it."""
    bin_dir = managed_node_bin_dir()
    return str(bin_dir) if bin_dir is not None else None


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
    _managed_node_ok = present and _node_version_ok(str(binary), _managed_probe_path())
    return _managed_node_ok


def path_with_managed_node(
    base_path: str | None = None,
    require_npm: bool = True,
    require_npx: bool = True,
) -> str:
    """``base_path`` (default: this process's PATH) with the managed Node bin dir
    moved to the front, unchanged when it is unusable or the PATH already resolves a
    runtime. The installer puts it on PATH for setup only, so subprocesses need it."""
    current = os.environ.get("PATH", "") if base_path is None else base_path
    bin_dir = managed_node_bin_dir()
    if bin_dir is None:
        return current
    # Never shadow a runtime the PATH already reaches (resolve_node_executable order).
    if _path_has_usable_node(current, require_npm = require_npm, require_npx = require_npx):
        return current
    if not managed_node_usable():
        return current
    bin_str = str(bin_dir)
    # An empty component means the working directory on POSIX; dropping it loses it.
    entries = current.split(os.pathsep) if current else []
    normalized = os.path.normcase(os.path.normpath(bin_str))
    # Drop any existing occurrence rather than keep it: this runs only when PATH resolves no usable
    # runtime, so a managed dir sitting behind a stale one must move up.
    kept = [
        entry
        for entry in entries
        if not (entry and os.path.normcase(os.path.normpath(entry)) == normalized)
    ]
    return os.pathsep.join([bin_str, *kept])


def _npm_version_ok(executable: str, path: str | None = None) -> bool:
    """Run ``<executable> -v`` and check npm clears the installer floor."""
    return _probe_version(executable, _npm_meets_floor, path)


def _node_version_ok(executable: str, path: str | None = None) -> bool:
    """Run ``<executable> -v`` and check it clears the floor; False on any error."""
    return _probe_version(executable, _version_meets_floor, path)


def _probe_version(
    executable: str,
    meets_floor,
    path: str | None = None,
) -> bool:
    """Run ``<executable> -v`` and apply ``meets_floor``; False on any error. ``path`` is
    the PATH the server would run with: npm and npx are ``#!/usr/bin/env node`` scripts,
    so probing them under the backend's own PATH can fail to find the candidate's node."""
    try:
        result = subprocess.run(
            [executable, "-v"],
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = _NODE_VERSION_PROBE_TIMEOUT_SECONDS,
            env = {**os.environ, "PATH": path} if path is not None else None,
            **windows_hidden_subprocess_kwargs(),
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    if result.returncode != 0:
        return False
    return meets_floor(result.stdout)


# Memoize ONLY a confirmed version-adequate executable: the installer runs in a separate process
# and may finish after the first probe, so a negative result must not stick until a restart.
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
