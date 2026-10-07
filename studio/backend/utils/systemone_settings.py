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
BACKEND_KEY = "systemone_backend"
BACKENDS = ("auto", "llama.cpp", "pytorch")
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


def clef_pytorch_unavailable_reason() -> str | None:
    if reason := runtime_unavailable_reason():
        return reason
    from importlib.metadata import PackageNotFoundError, version
    from packaging.version import Version

    try:
        supported = Version(version("transformers")) >= Version("5.5.0")
    except PackageNotFoundError:
        supported = False
    if not supported:
        return "Clef PyTorch needs Transformers 5.5.0 or newer. Update Unsloth before enabling it."
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


def get_backend() -> str:
    stored = _owner_setting(BACKEND_KEY)
    return stored if stored in BACKENDS else "auto"


def validate(
    *,
    enabled: bool | None = None,
    model: str | None = None,
    device: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    from core.systemone.catalog import (
        CHECKPOINTS,
        decision_connections,
        fine_tune,
        parse_connection,
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
    if backend is not None:
        if backend not in BACKENDS:
            raise ValueError("Backend must be auto, llama.cpp or pytorch.")
        values[BACKEND_KEY] = backend
    serving = (
        enabled
        if enabled is not None
        else (model is not None or backend is not None) and get_enabled()
    )
    local = parse_connection(get_model() if model is None else model) is None
    if serving and local:
        from core.systemone.catalog import ClefCheckpoint, default_checkpoint
        from core.systemone import runtime

        checkpoint = default_checkpoint() if model is None else CHECKPOINTS.get(model)
        uses_torch = True
        if isinstance(checkpoint, ClefCheckpoint):
            try:
                selected, _ = runtime.select_checkpoint(checkpoint, preference = backend)
            except runtime.Unavailable as exc:
                raise ValueError(exc.message) from None
            uses_torch = not runtime._clef().is_native(selected)
        if uses_torch:
            reason = (
                clef_pytorch_unavailable_reason()
                if isinstance(checkpoint, ClefCheckpoint)
                else runtime_unavailable_reason()
            )
            if reason:
                raise ValueError(reason)
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
