# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Configuration boundary for persistent training output storage."""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath


def resolve_configured_outputs_root(*, default: Path) -> Path:
    """Return the configured training-output root, preserving the legacy default."""
    raw = (os.environ.get("UNSLOTH_OUTPUTS_DIR") or "").strip()
    if not raw:
        return default
    if "\x00" in raw:
        raise ValueError("UNSLOTH_OUTPUTS_DIR may not contain null bytes")
    if ".." in raw.replace("\\", "/").split("/"):
        raise ValueError("UNSLOTH_OUTPUTS_DIR may not contain '..' segments")
    path = Path(raw).expanduser()
    # Reject drive-relative Windows paths on every host as well as ordinary
    # relative paths.  The setting is a storage contract, not a cwd-relative hint.
    if not path.is_absolute() or (PureWindowsPath(raw).drive and not PureWindowsPath(raw).is_absolute()):
        raise ValueError("UNSLOTH_OUTPUTS_DIR must be an absolute path")
    try:
        return path.resolve(strict = False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"UNSLOTH_OUTPUTS_DIR could not be resolved: {exc}") from exc
