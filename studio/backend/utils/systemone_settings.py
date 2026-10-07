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
NATIVE_CTX_KEY = "systemone_native_ctx"
DEFAULT_MODEL = "laya-multilingual"
DEVICES = ("cpu", "gpu")
BACKENDS = ("auto", "llama.cpp", "pytorch")
DEFAULT_NATIVE_CTX = 16384
# -c / -b / -ub of the llama.cpp decision server; compute memory grows with it (~0.24 MiB GPU per token).
NATIVE_CTX_RANGE = (512, 65536)

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


def llama_cpp_only(name: Any) -> bool:
    """A GGUF-only catalog entry: llama.cpp serves it without PyTorch."""
    from core.systemone.catalog import CHECKPOINTS
    return getattr(CHECKPOINTS.get(name), "layout", None) == "gguf"


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


def clef_device() -> str:
    """Where a llama.cpp Clef runs: the chosen device, else the GPU when there is one (CPU needs ~22 GB RAM)."""
    if device_locked() or _owner_setting(DEVICE_KEY) in DEVICES:
        return get_device()
    return "gpu" if gpu_available() else "cpu"


def get_backend() -> str:
    stored = _owner_setting(BACKEND_KEY)
    return stored if stored in BACKENDS else "auto"


def _valid_ctx(value: Any) -> bool:
    low, high = NATIVE_CTX_RANGE
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def get_native_ctx() -> int:
    stored = _owner_setting(NATIVE_CTX_KEY)
    return stored if _valid_ctx(stored) else DEFAULT_NATIVE_CTX


def validate(
    *,
    enabled: bool | None = None,
    model: str | None = None,
    device: str | None = None,
    backend: str | None = None,
    native_ctx: int | None = None,
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
            raise ValueError("Runtime must be auto, llama.cpp or pytorch.")
        values[BACKEND_KEY] = backend
    if native_ctx is not None:
        if not _valid_ctx(native_ctx):
            low, high = NATIVE_CTX_RANGE
            raise ValueError(f"The llama.cpp context must be {low} to {high} tokens.")
        values[NATIVE_CTX_KEY] = native_ctx
    serving = enabled if enabled is not None else model is not None and get_enabled()
    name = get_model() if model is None else model
    local = parse_connection(name) is None and not llama_cpp_only(name)
    if serving and local and (reason := runtime_unavailable_reason()):
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
