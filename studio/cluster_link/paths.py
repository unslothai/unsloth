# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where cluster state lives, and owner-only writes for keys and the peer registry."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

ENV_STATE_DIR = "UNSLOTH_CLUSTER_STATE_DIR"
ENV_RUN_DIR = "UNSLOTH_CLUSTER_RUN_DIR"


def _studio_root() -> Path:
    try:
        from utils.paths.storage_roots import studio_root
        return studio_root()
    except Exception:
        pass
    for var in ("UNSLOTH_STUDIO_HOME", "STUDIO_HOME"):
        value = (os.environ.get(var) or "").strip()
        if value:
            return Path(value).expanduser()
    return Path.home() / ".unsloth" / "studio"


def state_dir() -> Path:
    """``<studio root>/cluster``; ``UNSLOTH_CLUSTER_STATE_DIR`` lets tests run several nodes on one host."""
    override = (os.environ.get(ENV_STATE_DIR) or "").strip()
    return Path(override).expanduser() if override else _studio_root() / "cluster"


def run_dir() -> Path:
    """Per-user, machine-wide runtime state (which agent owns which rpc port): independent of any
    state or cache directory a worker was started with, so two workers always see each other."""
    override = (os.environ.get(ENV_RUN_DIR) or "").strip()
    if override:
        return private_dir(Path(override).expanduser())
    try:
        user = str(os.getuid())
    except AttributeError:  # Windows
        import getpass
        user = getpass.getuser()
    return private_dir(Path(tempfile.gettempdir()) / f"unsloth-cluster-{user}")


def private_dir(path: Path) -> Path:
    path.mkdir(parents = True, exist_ok = True)
    if os.name != "nt":
        os.chmod(path, 0o700)
    return path


def write_private(path: Path, data: bytes) -> None:
    """Atomic replace, created 0600: a reader never sees a half-written key or registry."""
    private_dir(path.parent)
    fd, tmp = tempfile.mkstemp(dir = str(path.parent), prefix = f".{path.name}.")
    try:
        if os.name != "nt":
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
