# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installation-wide settings for the Decision API (/v1/systemone). Environment variables win."""

from __future__ import annotations

import os
import sys
from typing import Any

ENABLED_KEY = "systemone_enabled"
MODEL_KEY = "systemone_model"
DEVICE_KEY = "systemone_device"
DEFAULT_MODEL = "laya-multilingual"
DEVICES = ("cpu", "gpu")

ENV_DISABLE = "UNSLOTH_SYSTEMONE_DISABLE"
ENV_MODEL = "UNSLOTH_SYSTEMONE_MODEL"
ENV_DEVICE = "UNSLOTH_SYSTEMONE_DEVICE"


def _owner_setting(key: str) -> Any:
    # Installation-wide: a managed account's API key must see the owner's switch, not its own database.
    from storage.studio_db import get_app_setting
    from utils.account_context import OWNER, bind_account, reset_account

    token = bind_account(OWNER)
    try:
        return get_app_setting(key, None)
    except Exception:
        return None
    finally:
        reset_account(token)


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def enabled_locked() -> bool:
    return _env(ENV_DISABLE) == "1"


def model_locked() -> bool:
    return bool(_env(ENV_MODEL))


def device_locked() -> bool:
    return bool(_env(ENV_DEVICE))


def runtime_supported() -> bool:
    # extras-no-deps.txt installs laya only on Python 3.10+, its own floor.
    return sys.version_info >= (3, 10)


def get_enabled() -> bool:
    if enabled_locked():
        return False
    return _owner_setting(ENABLED_KEY) is True


def get_model() -> str:
    if model_locked():
        return _env(ENV_MODEL)
    from core.systemone.catalog import CHECKPOINTS

    stored = _owner_setting(MODEL_KEY)
    return stored if stored in CHECKPOINTS else DEFAULT_MODEL


def get_device() -> str:
    if device_locked():
        return "gpu" if _env(ENV_DEVICE).lower() in ("gpu", "auto") else "cpu"
    stored = _owner_setting(DEVICE_KEY)
    return stored if stored in DEVICES else "cpu"


def validate(
    *,
    enabled: bool | None = None,
    model: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    from core.systemone.catalog import CHECKPOINTS

    values: dict[str, Any] = {}
    if enabled is not None:
        if enabled_locked():
            raise ValueError(f"The Decision API is turned off by {ENV_DISABLE}.")
        if enabled and not runtime_supported():
            raise ValueError("The Decision API needs Python 3.10 or newer.")
        values[ENABLED_KEY] = bool(enabled)
    if model is not None:
        if model_locked():
            raise ValueError(f"The Decision API model is set by {ENV_MODEL}.")
        if model not in CHECKPOINTS:
            raise ValueError(f"Unknown Decision API model: {model}")
        values[MODEL_KEY] = model
    if device is not None:
        if device_locked():
            raise ValueError(f"The Decision API device is set by {ENV_DEVICE}.")
        if device not in DEVICES:
            raise ValueError("Device must be cpu or gpu.")
        values[DEVICE_KEY] = device
    return values


def save(values: dict[str, Any]) -> None:
    from storage.studio_db import upsert_app_settings
    if values:
        upsert_app_settings(values)


def gpu_available() -> bool:
    try:
        from utils.hardware.hardware import DeviceType, get_device as detected_device
        return detected_device() in (DeviceType.CUDA, DeviceType.XPU, DeviceType.MLX)
    except Exception:
        return False
