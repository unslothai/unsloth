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


def runtime_unavailable_reason() -> str | None:
    # The vendored laya needs Python 3.10+ and torch (absent on --no-torch).
    if sys.version_info < (3, 10):
        return "The Decision API needs Python 3.10 or newer."
    try:
        from importlib.util import find_spec
        if find_spec("torch") is None:
            return "The Decision API needs PyTorch, which this Studio install does not include."
    except Exception:
        pass
    return None


def get_enabled() -> bool:
    if enabled_locked():
        return False
    return _owner_setting(ENABLED_KEY) is True


def get_model() -> str:
    if model_locked():
        return _env(ENV_MODEL)
    from core.systemone.catalog import CHECKPOINTS, fine_tune, parse_connection

    stored = _owner_setting(MODEL_KEY)
    if stored in CHECKPOINTS or parse_connection(stored):
        return stored
    if isinstance(stored, str) and fine_tune(stored) is not None:
        return stored
    return DEFAULT_MODEL


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
    from core.systemone.catalog import (
        CHECKPOINTS,
        CLEF_NEEDS_GPU_SETTING,
        decision_connections,
        fine_tune,
        parse_connection,
        resolve,
    )

    values: dict[str, Any] = {}
    if enabled is not None:
        if enabled_locked():
            raise ValueError(f"The Decision API is turned off by {ENV_DISABLE}.")
        values[ENABLED_KEY] = bool(enabled)
    if model is not None:
        if model_locked():
            raise ValueError(f"The Decision API model is set by {ENV_MODEL}.")
        connection = parse_connection(model)
        tuned = None if model in CHECKPOINTS else fine_tune(model)
        if tuned is not None:
            # Stored under the prefix for the folder's layout, whichever one the caller used.
            model = tuned.name
        if (
            model not in CHECKPOINTS
            and tuned is None
            and not (
                connection
                and any(
                    row["id"] == connection.provider_id and connection.model in models
                    for row, models in decision_connections()
                )
            )
        ):
            raise ValueError(f"Unknown Decision API model: {model}")
        values[MODEL_KEY] = model
    if device is not None:
        if device_locked():
            raise ValueError(f"The Decision API device is set by {ENV_DEVICE}.")
        if device not in DEVICES:
            raise ValueError("Device must be cpu or gpu.")
        values[DEVICE_KEY] = device
    serving = enabled if enabled is not None else model is not None and get_enabled()
    local = parse_connection(get_model() if model is None else model) is None
    if serving and local and (reason := runtime_unavailable_reason()):
        raise ValueError(reason)
    # Clef has no CPU path, so a Clef model and a CPU device are never stored together while serving.
    active = enabled if enabled is not None else get_enabled()
    if active and local:
        chosen = resolve(get_model() if model is None else model)
        wanted = get_device() if device is None else device
        if getattr(chosen, "layout", None) == "clef" and wanted != "gpu":
            raise ValueError(CLEF_NEEDS_GPU_SETTING)
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
