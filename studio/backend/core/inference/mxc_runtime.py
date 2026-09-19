# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Resolve the pinned Windows MXC supervisor without importing it off Windows."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys

MXC_REVISION = "ca7ea12ac6bd9f5420d6adecb37e32a8158da476"
MXC_SCHEMA_VERSION = "0.8.0-alpha"
RUNNER_PROTOCOL_VERSION = 1


class MxcRuntimeUnavailable(RuntimeError):
    pass


def runner_path() -> Path:
    if sys.platform != "win32":
        raise MxcRuntimeUnavailable("the MXC runtime is Windows-only")
    native_root = Path(__file__).resolve().parents[3] / "native" / "mxc-runner"
    packaged = native_root / "bin" / "unsloth-mxc-runner.exe"
    if packaged.is_file():
        return packaged
    # Source checkouts use Cargo's output. Installers must ship the manifest-owned
    # bin path above; no PATH lookup or model-controlled helper path is accepted.
    development = native_root / "target" / "release" / "unsloth-mxc-runner.exe"
    if development.is_file() and (Path(__file__).resolve().parents[4] / ".git").exists():
        return development
    raise MxcRuntimeUnavailable(
        "the pinned Unsloth MXC supervisor is not installed; rerun Windows Studio setup"
    )


def installation_identity() -> str:
    path = runner_path()
    digest = hashlib.sha256()
    digest.update(MXC_REVISION.encode())
    digest.update(MXC_SCHEMA_VERSION.encode())
    digest.update(path.read_bytes())
    return digest.hexdigest()
