# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persist and auto-detect the root used for training checkpoints."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


CHECKPOINT_HOME_SETTING_KEY = "checkpoint_saving_home"
CheckpointSource = Literal["default", "studio", "environment", "colab", "kaggle"]


@dataclass(frozen = True)
class CheckpointLocation:
    path: Path
    source: CheckpointSource
    editable: bool = True
    environment_variable: Optional[str] = None

    @property
    def is_custom(self) -> bool:
        return self.source == "studio"


def _canonical(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict = False)


def notebook_browse_roots() -> list[Path]:
    """Trusted notebook workspace roots that folder pickers may browse.

    These are platform-owned data workspaces, not arbitrary filesystem roots.
    Only return roots for a detected platform and only when they actually exist.
    """

    candidates: list[Path] = []
    is_colab = bool(os.environ.get("COLAB_BACKEND_URL") or os.environ.get("COLAB_JUPYTER_IP"))
    if is_colab or Path("/content").is_dir():
        candidates.append(Path("/content"))
        candidates.append(Path("/content/drive/MyDrive"))
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        candidates.append(Path("/kaggle/working"))
    return [path for path in candidates if path.is_dir()]


def _detected_default() -> CheckpointLocation:
    override = (os.environ.get("UNSLOTH_OUTPUTS_DIR") or "").strip()
    if override:
        return CheckpointLocation(
            _canonical(override), "environment", False, "UNSLOTH_OUTPUTS_DIR"
        )
    # COLAB_* is more reliable than merely finding /content on a host. Prefer
    # mounted Drive so checkpoints survive a runtime reset, otherwise use the
    # runtime disk rather than returning a path whose parent does not exist.
    is_colab = bool(os.environ.get("COLAB_BACKEND_URL") or os.environ.get("COLAB_JUPYTER_IP"))
    if is_colab or Path("/content").is_dir():
        drive = Path("/content/drive/MyDrive")
        base = drive if drive.is_dir() else Path("/content")
        return CheckpointLocation(base / "unsloth_outputs", "colab")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/working").is_dir():
        return CheckpointLocation(Path("/kaggle/working/unsloth_outputs"), "kaggle")

    from utils.paths.storage_roots import studio_root

    return CheckpointLocation(studio_root() / "outputs", "default")


def _stored_path() -> Optional[Path]:
    try:
        from storage.studio_db import get_app_setting

        value = get_app_setting(CHECKPOINT_HOME_SETTING_KEY, None)
    except Exception:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return _canonical(value)
    except (OSError, RuntimeError, ValueError):
        return None


def get_checkpoint_location() -> CheckpointLocation:
    detected = _detected_default()
    if detected.source == "environment":
        return detected
    stored = _stored_path()
    return CheckpointLocation(stored, "studio") if stored is not None else detected


def _validate_path(raw: str) -> Path:
    value = raw.strip()
    if not value:
        raise ValueError("Choose a checkpoint folder.")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise ValueError("The checkpoint folder must be an absolute path.")
    resolved = candidate.resolve(strict = False)
    if resolved.parent == resolved:
        raise ValueError("Choose a folder inside the filesystem or drive root.")
    if not resolved.parent.is_dir():
        raise ValueError("The parent folder does not exist.")
    try:
        resolved.mkdir(exist_ok = True)
        with tempfile.NamedTemporaryFile(prefix = ".unsloth-write-test-", dir = resolved):
            pass
    except PermissionError as exc:
        raise ValueError("Studio does not have permission to write to this folder.") from exc
    except OSError as exc:
        raise ValueError(f"Studio cannot use this checkpoint folder: {exc}") from exc
    return resolved


def set_checkpoint_location(path: Optional[str]) -> CheckpointLocation:
    if _detected_default().source == "environment":
        raise RuntimeError("The checkpoint location is managed by UNSLOTH_OUTPUTS_DIR.")
    selected = _validate_path(path) if path is not None else None
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({CHECKPOINT_HOME_SETTING_KEY: str(selected) if selected else None})
    return get_checkpoint_location()
