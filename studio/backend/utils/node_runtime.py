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


def _node_version_ok(executable: str) -> bool:
    """Run ``<executable> -v`` and check it clears the floor; False on any error."""
    try:
        result = subprocess.run(
            [executable, "-v"],
            capture_output = True,
            text = True,
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
