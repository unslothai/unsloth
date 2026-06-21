# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolve a usable Node.js executable at runtime.

The installer (studio/setup.{sh,ps1} + install_node_prebuilt.py) provisions an
isolated Node under ``<UNSLOTH_HOME>/node`` and only prepends it to PATH for the
*setup* process -- the user's shell PATH is deliberately never modified. So
backend code that shells out to ``node`` at runtime (e.g. the OXC data-recipe
validator) cannot rely on PATH alone: a user who got the isolated Node (because
their system Node was missing or too old) would have no ``node`` on PATH.

``resolve_node_executable`` bridges that gap: prefer a version-adequate system
Node, else fall back to the managed isolated Node, mirroring the version bar the
installer applies (Node ^20.19 || >=22.12 || >=23).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from utils.subprocess_compat import windows_hidden_subprocess_kwargs

# Same floor the setup scripts enforce in their Node decision (Vite 8 needs
# Node ^20.19.0 || >=22.12.0 || >=23). Keep in sync with Get-NodeDecision
# (setup.ps1) / decide_node_source (setup.sh).
_NODE_VERSION_PROBE_TIMEOUT_SECONDS = 10


def _version_meets_floor(version: str) -> bool:
    """True iff a ``node -v`` string clears the installer's version bar."""
    match = re.match(r"v?(\d+)\.(\d+)", version.strip())
    if not match:
        return False
    major, minor = int(match.group(1)), int(match.group(2))
    return (major == 20 and minor >= 19) or (major == 22 and minor >= 12) or major >= 23


def managed_node_dir() -> Path:
    """The isolated Node install dir, matching the installer layout.

    Mirrors ``_find_llama_server_binary`` (core/inference/llama_cpp.py): Node and
    llama.cpp share the same parent -- ``<STUDIO_HOME>`` in env/custom mode, else
    the legacy ``~/.unsloth`` (the sibling of ``~/.unsloth/studio``).
    """
    legacy_node = Path.home() / ".unsloth" / "node"
    try:
        # Lazy import (mirrors _find_llama_server_binary) so this module stays
        # importable -- and falls back to the legacy root -- even if utils.paths
        # cannot be loaded.
        from utils.paths.storage_roots import studio_root

        resolved = studio_root()
        legacy_studio = Path.home() / ".unsloth" / "studio"
        try:
            is_legacy = resolved.resolve() == legacy_studio.resolve()
        except (OSError, ValueError):
            is_legacy = resolved == legacy_studio
        return legacy_node if is_legacy else (resolved / "node")
    except (ImportError, OSError, ValueError):
        # Degraded import environment (utils.paths unavailable): still honor an
        # explicit STUDIO_HOME override, mirroring studio_root()'s priority,
        # before falling back to the legacy default.
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
    """The node executable inside the isolated install (Windows vs Unix layout).

    Matches ``node_binary_path`` in install_node_prebuilt.py: ``<dir>/node.exe``
    on Windows, ``<dir>/bin/node`` elsewhere.
    """
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


# Memoize ONLY a confirmed version-adequate executable. The installer runs in a
# separate process and may finish AFTER the backend first probes here, so a
# negative / last-resort result must not be cached -- otherwise the validator
# would keep reporting "Node not found" until a backend restart even after Node
# appears on disk.
_resolved_node: str | None = None


def _reset_resolved_node() -> None:
    """Clear the memoized executable (used by tests)."""
    global _resolved_node
    _resolved_node = None


def resolve_node_executable() -> str | None:
    """Resolve a usable node executable, or None if none is available.

    Order: a version-adequate system ``node`` on PATH; else the managed isolated
    Node if present and adequate; else fall back to whatever bare ``node`` is on
    PATH (may be None) to preserve the pre-isolation behaviour. Only a
    version-adequate result is memoized, so a Node installed after the first
    probe is picked up without a backend restart.
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

    # Nothing adequate yet -- return a last-resort system node (may be None)
    # WITHOUT caching, so a later install/upgrade is picked up on the next call.
    return system_node
